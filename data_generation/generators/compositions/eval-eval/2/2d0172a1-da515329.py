"""Unwinding nested rooms: containment levels become pinwheel turn times.

Only standard-library modules and NumPy are used. No parent code, fixtures,
files, random state, or demonstrations are consulted by the input-only solver.
"""
from collections import Counter
import numpy as np


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= s <= 30 for s in a.shape), "invalid grid shape")
    _need(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _need(bool(np.all((a >= 0) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=False)


def _components(mask):
    seen = np.zeros(mask.shape, bool)
    result = []
    h, w = mask.shape
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        stack = [(int(y), int(x))]
        seen[y, x] = True
        cells = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dy, dx in DIRECTIONS:
                yy, xx = r+dy, c+dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(np.array(cells, dtype=int))
    return result


def _holes(mask):
    """Actual enclosure of one component, treating every other cell as free."""
    # No hole can extend outside a component's bounding box. Cropping avoids
    # flood-filling an entire 30x30 canvas for every one-cell bead or spoke.
    ys, xs = np.nonzero(mask)
    result = np.zeros(mask.shape, bool)
    if not len(ys):
        return result
    y0, y1, x0, x1 = int(ys.min()), int(ys.max())+1, int(xs.min()), int(xs.max())+1
    wall = np.pad(mask[y0:y1, x0:x1], 1)
    visited = np.zeros(wall.shape, bool)
    visited[0, 0] = True
    stack = [(0, 0)]
    h, w = wall.shape
    while stack:
        y, x = stack.pop()
        for dy, dx in DIRECTIONS:
            yy, xx = y+dy, x+dx
            if 0 <= yy < h and 0 <= xx < w and not wall[yy, xx] and not visited[yy, xx]:
                visited[yy, xx] = True
                stack.append((yy, xx))
    result[y0:y1, x0:x1] = (~wall & ~visited)[1:-1, 1:-1]
    return result


def _mask(shape, cells):
    a = np.zeros(shape, bool)
    a[cells[:, 0], cells[:, 1]] = True
    return a


def _near(a, b):
    """Eight-neighbor contact of two disjoint masks."""
    p = np.pad(a, 1)
    h, w = a.shape
    return any(np.any(p[1+dy:1+dy+h, 1+dx:1+dx+w] & b)
               for dy in (-1, 0, 1) for dx in (-1, 0, 1))


def _parse(input_grid):
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    _need(int(counts.max())*2 > a.size, "background must be the unique majority")
    candidates = []
    for color in colors:
        if color == bg:
            continue
        loops, strokes = [], []
        for cells in _components(a == color):
            component = _mask(a.shape, cells)
            hole = _holes(component)
            if hole.any():
                _need(len(_components(hole)) == 1, "multiple holes in a component")
                loops.append((component, hole))
            else:
                strokes.append(component)
        if loops:
            candidates.append((int(color), loops, strokes))
    _need(len(candidates) == 1, "one color must supply all closed contours")
    ink, loops, strokes = candidates[0]
    n = len(loops)
    _need(1 <= n <= 4 and len(strokes) == 4, "expected nested contours and four spokes")
    cross = np.logical_or.reduce(strokes)
    ys, xs = np.nonzero(cross)
    cy, cx = (int(ys.min()+ys.max())//2, int(xs.min()+xs.max())//2)
    length = int(cross.sum())//4
    _need(length in (1, 2), "unsupported input spoke length")
    expected = np.zeros(a.shape, bool)
    for dy, dx in DIRECTIONS:
        for d in range(1, length+1):
            y, x = cy+d*dy, cx+d*dx
            _need(0 <= y < a.shape[0] and 0 <= x < a.shape[1], "cross clipped")
            expected[y, x] = True
    _need(np.array_equal(expected, cross) and a[cy, cx] == bg, "invalid hollow cross")
    loops.sort(key=lambda item: int(item[1].sum()))
    previous = cross
    for i, (component, hole) in enumerate(loops):
        _need(bool(hole[previous].all()), "contours must form a strict nested chain")
        _need(not _near(component, previous), "ink components touch")
        previous = component | hole
        # All contours enclose the complete origin, not only its center.
        _need(bool(hole[cross].all()), "cross outside contour")
    tokens = []
    addresses = set()
    for color in colors:
        if int(color) in (bg, ink):
            continue
        for cells in _components(a == color):
            component = _mask(a.shape, cells)
            _need(not _holes(component).any(), "a marker must be filled")
            lo, hi = cells.min(axis=0), cells.max(axis=0)
            _need(len(cells) >= .35*int(np.prod(hi-lo+1)), "marker is not compact")
            sides = []
            if hi[0] < cy and lo[1] <= cx <= hi[1]:
                sides.append(0)
            if lo[1] > cx and lo[0] <= cy <= hi[0]:
                sides.append(1)
            if lo[0] > cy and lo[1] <= cx <= hi[1]:
                sides.append(2)
            if hi[1] < cx and lo[0] <= cy <= hi[0]:
                sides.append(3)
            _need(len(sides) == 1, "marker needs one compass half-axis")
            d = 0
            for _, hole in loops:
                inside = hole[cells[:, 0], cells[:, 1]]
                _need(bool(inside.all()) or not inside.any(), "marker straddles a contour")
                d += int(inside.all())
            address = (sides[0], n-d)
            _need(address not in addresses, "two markers at one address")
            addresses.add(address)
            tokens.append((address[0], address[1], int(color), len(cells)))
    _need(1 <= len({t[2] for t in tokens}) <= 3, "expected one to three marker colors")
    return dict(grid=a, bg=bg, ink=ink, n=n, center=(cy, cx), length=length,
                loops=loops, tokens=tokens)


def _paths(n, handedness=1):
    """Walk the four finite paths; return the mask and per-arm endpoints."""
    side, mid = 4*n+5, 2*n+2
    result = np.zeros((side, side), bool)
    endpoints = []
    for dy, dx in DIRECTIONS:
        y, x = mid, mid
        for _ in range(2):
            y, x = y+dy, x+dx
            result[y, x] = True
        tips = [(y, x)]
        dy, dx = handedness*dx, -handedness*dy
        for stage in range(1, n+1):
            for _ in range(4*stage):
                y, x = y+dy, x+dx
                _need(not result[y, x], "winding paths intersect")
                result[y, x] = True
            tips.append((y, x))
            dy, dx = handedness*dx, -handedness*dy
        endpoints.append(tips)
    return result, endpoints


def _draw_answer(n, bg, ink, tokens, *, handedness=1, flat=False, rays=False):
    mask, tips = _paths(n, handedness)
    if rays:
        # Deliberately wrong ablation: keep each address on its original ray.
        mask[:] = False
        mid = 2*n+2
        tips = []
        for dy, dx in DIRECTIONS:
            points = [(mid+(2+2*j)*dy, mid+(2+2*j)*dx) for j in range(n+1)]
            for d in range(1, 2*n+3):
                mask[mid+d*dy, mid+d*dx] = True
            tips.append(points)
    out = np.where(mask, ink, bg).astype(np.int8)
    for side, level, color, *_ in tokens:
        out[tips[side][0 if flat else level]] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every output cell from the supplied input alone."""
    info = _parse(input_grid)
    return _draw_answer(info['n'], info['bg'], info['ink'], info['tokens'])


def _outline(shape, center, distances, cuts):
    """A closed axis-aligned staircase polygon with four clipped corners."""
    cy, cx = center
    t, r, b, l = distances
    tl, tr, br, bl = cuts
    points = [(-t, -l+tl), (-t, r-tr)]
    for k in range(tr):
        points.extend([(-t+k+1, r-tr+k), (-t+k+1, r-tr+k+1)])
    points.append((b-br, r))
    for k in range(br):
        points.extend([(b-br+k, r-k-1), (b-br+k+1, r-k-1)])
    points.append((b, -l+bl))
    for k in range(bl):
        points.extend([(b-k-1, -l+bl-k), (b-k-1, -l+bl-k-1)])
    points.append((-t+tl, -l))
    for k in range(tl):
        points.extend([(-t+tl-k, -l+k+1), (-t+tl-k-1, -l+k+1)])
    points.append(points[0])
    out = np.zeros(shape, bool)
    for (y, x), (yy, xx) in zip(points, points[1:]):
        _need(y == yy or x == xx, "non-orthogonal contour edge")
        out[cy+min(y, yy):cy+max(y, yy)+1, cx+x] = True
        out[cy+yy, cx+min(x, xx):cx+max(x, xx)+1] = True
    return out


def _addresses(rng, n, witness, sparse=False):
    if witness:
        arm = int(rng.integers(4))
        other = (arm+int(rng.integers(1, 4))) % 4
        if sparse:
            # Missing levels on a ray defeat ranking the blobs along that
            # ray without inspecting the contours. All levels are still
            # demonstrated collectively across the four arms.
            chosen = {(arm, 0), (arm, n), (other, 1), (other, n)}
            if n == 3:
                third = int(rng.choice([s for s in range(4) if s not in (arm, other)]))
                chosen.add((third, 2))
        else:
            chosen = {(arm, j) for j in range(n+1)}
            chosen.add((other, n))
        # Equal bead counts in the two witnesses rule out counting blobs
        # as a substitute for counting enclosing contours.
        while len(chosen) < 5:
            choices = [(s, j) for s in range(4) for j in range(n+1)
                       if (s, j) not in chosen and
                       (not sparse or sum(a == s for a, _ in chosen) < 2)]
            chosen.add(choices[int(rng.integers(len(choices)))])
    else:
        count = int(rng.integers(3, min(10, 4*(n+1))+1))
        raw = rng.choice(4*(n+1), count, replace=False)
        chosen = {(int(k)//(n+1), int(k) % (n+1)) for k in raw}
        # Guarantee useful evidence even in the sparsest sampled layout.
        if len({p[1] for p in chosen}) == 1:
            s, j = next(iter(chosen))
            chosen.remove((s, j))
            chosen.add((s, (j+1) % (n+1)))
        if len({p[0] for p in chosen}) == 1:
            s, j = sorted(chosen)[0]
            chosen.remove((s, j))
            chosen.add(((s+int(rng.integers(1, 4))) % 4, j))
    return sorted(chosen)


def _make_pair(rng, n, palette, addresses, length, spacious=False):
    bg, ink, *marker_colors = palette
    # Bounded redraws handle the 30-cell bound and compact marker clearances.
    for attempt in range(100):
        distances = [rng.integers(length+2, length+4, size=4).astype(int)]
        for _ in range(1, n):
            upper = 4 if spacious and n <= 2 else 3
            distances.append(distances[-1] + rng.integers(2, upper+1, size=4))
        outer = distances[-1]
        # Exterior addresses need at least one in-canvas location.
        required = np.array([int((s, n) in addresses) for s in range(4)])
        margin = required + rng.integers(0, 4 if spacious else 3, size=4)
        h = int(outer[0]+outer[2]+1+margin[0]+margin[2])
        w = int(outer[1]+outer[3]+1+margin[1]+margin[3])
        if h > 30 or w > 30:
            continue
        # Independently extend either dimension, with arbitrary placement.
        add_h = int(rng.integers(0, 31-h))
        add_w = int(rng.integers(0, 31-w))
        top_add = int(rng.integers(add_h+1))
        left_add = int(rng.integers(add_w+1))
        margin += (top_add, add_w-left_add, add_h-top_add, left_add)
        h += add_h
        w += add_w
        center = (int(outer[0]+margin[0]), int(outer[3]+margin[3]))
        a = np.full((h, w), bg, np.int8)
        cy, cx = center
        cross = np.zeros(a.shape, bool)
        for dy, dx in DIRECTIONS:
            for d in range(1, length+1):
                cross[cy+d*dy, cx+d*dx] = True
        a[cross] = ink
        previous = cross
        loops = []
        for i, dims in enumerate(distances):
            # Rectangles remain possible; most loops lose actual corners.
            cuts = rng.integers(0, 4, size=4) if rng.random() < .85 else np.zeros(4, int)
            loop = _outline(a.shape, center, dims, cuts)
            hole = _holes(loop)
            if (len(_components(loop)) != 1 or len(_components(hole)) != 1
                    or not hole[previous].all() or _near(loop, previous)):
                loop = _outline(a.shape, center, dims, [0]*4)
                hole = _holes(loop)
            loops.append((loop, hole))
            a[loop] = ink
            previous = loop | hole
        occupied_markers = np.zeros(a.shape, bool)
        token_records = []
        possible = True
        for side, level in addresses:
            dy, dx = DIRECTIONS[side]
            lo = length+1 if level == 0 else int(distances[level-1][side])+1
            hi = int(distances[level][side])-1 if level < n else int(outer[side]+margin[side])
            if lo > hi:
                possible = False
                break
            color = int(rng.choice(marker_colors))
            # Try larger and irregular blobs first; every address always has
            # a single-pixel fallback on the contour-free cardinal axis.
            placed = False
            for trial in range(18):
                radial = int(rng.integers(1, min(3, hi-lo+1)+1)) if trial < 17 else 1
                tangent = int(rng.integers(1, 4)) if trial < 17 else 1
                start = int(rng.integers(lo, hi-radial+2))
                cells = []
                for d in range(start, start+radial):
                    for q in range(-(tangent//2), tangent-tangent//2):
                        # Tangential coordinate is perpendicular to the ray.
                        cells.append((cy+d*dy+q*dx, cx+d*dx-q*dy))
                cells = np.array(cells, int)
                if not (np.all(cells >= 0) and np.all(cells[:, 0] < h) and np.all(cells[:, 1] < w)):
                    continue
                if radial >= 2 and tangent >= 2 and rng.random() < .6:
                    # Remove one corner, preserving compactness and the axis.
                    cells = np.delete(cells, int(rng.choice([0, tangent-1, len(cells)-tangent, len(cells)-1])), axis=0)
                marker = _mask(a.shape, cells)
                if np.any(a[marker] != bg) or _near(marker, occupied_markers):
                    continue
                inside = [bool(hole[marker].all()) for _, hole in loops]
                partial = any(hole[marker].any() and not hole[marker].all() for _, hole in loops)
                if partial or n-sum(inside) != level:
                    continue
                a[marker] = color
                occupied_markers |= marker
                token_records.append((side, level, color, len(cells)))
                placed = True
                break
            if not placed:
                possible = False
                break
        if not possible or (a == bg).sum()*2 <= a.size:
            continue
        try:
            info = _parse(a)
        except ValueError:
            continue
        if sorted(info['tokens']) != sorted(token_records):
            continue
        out = _draw_answer(n, bg, ink, token_records)
        return dict(input=a, output=out)
    raise RuntimeError("bounded geometric redraw exhausted")


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError("illegal example count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    first_n = int(rng.choice([1, 2]))
    cap = first_n+1
    plan = [first_n, cap]
    while len(plan) < ntrain:
        plan.append(int(rng.integers(1, cap+1)))
    if regime == 0:
        tests = [cap+1]*ntest
    elif regime == 1:
        tests = [int(rng.integers(1, cap+1)) for _ in range(ntest)]
    else:
        tests = [int(rng.integers(1, cap)) for _ in range(ntest)]
    stable = rng.random() < .3
    color_count = int(rng.integers(1, 4))
    base = [int(v) for v in rng.choice(10, color_count+2, replace=False)]
    puzzle = dict(train=[], test=[])
    seen = set()
    for index, n in enumerate(plan+tests):
        train = index < ntrain
        witness = train and index < 2
        palette = base if stable else [int(v) for v in rng.choice(10, color_count+2, replace=False)]
        addresses = _addresses(rng, n, witness, sparse=witness and index == 1)
        length = index+1 if witness else int(rng.integers(1, 3))
        for _ in range(40):
            pair = _make_pair(rng, n, palette, addresses, length,
                              spacious=(not train and regime == 2))
            if _key(pair['input']) not in seen:
                break
        else:
            raise RuntimeError("unique input redraw exhausted")
        seen.add(_key(pair['input']))
        puzzle['train' if train else 'test'].append(pair)
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    return puzzle


def _analytic_answer(info):
    """Interval membership and closed-form corners, separate from path walking."""
    n, mid = info['n'], 2*info['n']+2
    yy, xx = np.indices((4*n+5, 4*n+5))
    y, x = yy-mid, xx-mid
    result = np.zeros(y.shape, bool)
    points = []
    for arm in range(4):
        # North-arm corner after j turns. Every pair of turns adds four to
        # the absolute coordinate; the sign reverses every two turns.
        corners = [(-2, 0)]
        for j in range(1, n+1):
            if j % 2:
                q = (j-1)//2
                corners.append(((-1)**(q+1)*(4*q+2), (-1)**q*(4*q+4)))
            else:
                q = j//2
                corners.append(((-1)**(q+1)*(4*q+2), (-1)**(q+1)*4*q))
        rotated = []
        for ry, rx in corners:
            for _ in range(arm):
                ry, rx = rx, -ry
            rotated.append((ry, rx))
        first = rotated[0]
        # Include initial two-cell spoke, excluding the origin.
        if first[0] == 0:
            result |= (y == 0) & (x*first[1] > 0) & (abs(x) <= 2)
        else:
            result |= (x == 0) & (y*first[0] > 0) & (abs(y) <= 2)
        for (ay, ax), (by, bx) in zip(rotated, rotated[1:]):
            if ay == by:
                result |= (y == ay) & (x >= min(ax, bx)) & (x <= max(ax, bx))
            else:
                result |= (x == ax) & (y >= min(ay, by)) & (y <= max(ay, by))
        points.append(rotated)
    out = np.where(result, info['ink'], info['bg']).astype(np.int8)
    for arm, level, color, _ in info['tokens']:
        ry, rx = points[arm][level]
        out[mid+ry, mid+rx] = color
    return out


def _witness(info):
    addresses = {(s, j) for s, j, _, _ in info['tokens']}
    n = info['n']
    return set(range(n+1)) <= {j for _, j in addresses} and any(
               (s, 0) in addresses and (s, n) in addresses
               and any((other, n) in addresses for other in range(4) if other != s)
               for s in range(4))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert _key(a) not in seen, 'duplicate input'
            seen.add(_key(a))
            info = _parse(a)
            n, bg, ink = info['n'], info['bg'], info['ink']
            assert b.shape == (4*n+5, 4*n+5)
            assert np.array_equal(b, _analytic_answer(info)), 'wrong winding or bead address'
            assert b[2*n+2, 2*n+2] == bg
            assert int(np.sum(b != bg)) == 8+8*n*(n+1), 'incorrect path occupancy'
            expected_counts = Counter(t[2] for t in info['tokens'])
            assert all(int(np.sum(b == c)) == k for c, k in expected_counts.items())
            assert set(np.unique(b)) == {bg, ink} | set(expected_counts)
            assert len(info['tokens']) >= 3
            assert len({t[0] for t in info['tokens']}) >= 2
            assert len({t[1] for t in info['tokens']}) >= 2
            if split == 'train':
                training.append(info)
    witnesses = [i for i in training if _witness(i)]
    assert any(a['n']+1 == b['n'] for a in witnesses for b in witnesses), 'missing successive-depth witnesses'
    assert {1, 2} <= {i['length'] for i in witnesses}, 'missing cross scale evidence'
    assert any(a['n']+1 == b['n'] and len(a['tokens']) == len(b['tokens']) == 5
               and max(Counter(t[0] for t in b['tokens']).values()) < b['n']+1
               for a in witnesses for b in witnesses), 'missing sparse equal-count witness'
    # Explicit concept ablations evaluated on the visible teaching data.
    assert any(not np.array_equal(_analytic_answer(i), _draw_answer(
        i['n'], i['bg'], i['ink'], i['tokens'], flat=True)) for i in training)
    assert any(not np.array_equal(_analytic_answer(i), _draw_answer(
        i['n'], i['bg'], i['ink'], i['tokens'], rays=True)) for i in training)
    assert any(not np.array_equal(_analytic_answer(i), _draw_answer(
        i['n'], i['bg'], i['ink'], i['tokens'], handedness=-1)) for i in training)
    assert max(_parse(p['input'])['n'] for p in puzzle['test']) <= max(i['n'] for i in training)+1


def _puzzle_key(puzzle):
    return tuple((s, tuple((_key(np.asarray(p['input'])), _key(np.asarray(p['output'])))
                           for p in puzzle[s])) for s in ('train', 'test'))


def self_test():
    """Bounded geometric, metamorphic, mutation, and 42-seed coverage checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, addresses, palettes, blob_areas, input_lengths, normalized = set(), set(), set(), set(), set(), set()
    ns = {'train': set(), 'test': set()}
    nonrect = exterior = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _parse(pair['input'])
                expected = pair['output']
                assert np.array_equal(solve(pair['input'].copy()), expected)
                shapes.add(pair['input'].shape)
                ns[split].add(info['n'])
                input_lengths.add(info['length'])
                palettes.add((info['bg'], info['ink']))
                addresses.add((info['n'], tuple(sorted((s, j) for s, j, _, _ in info['tokens']))))
                blob_areas.update(t[3] for t in info['tokens'])
                exterior += any(t[1] == info['n'] for t in info['tokens'])
                role_grid = np.where(expected == info['bg'], 0, np.where(expected == info['ink'], 1, 2))
                normalized.add(_key(role_grid))
                for wall, _ in info['loops']:
                    yy, xx = np.nonzero(wall)
                    box = wall[yy.min():yy.max()+1, xx.min():xx.max()+1]
                    nonrect += not (box[0].all() and box[-1].all() and box[:, 0].all() and box[:, -1].all())
                # Translation and an arbitrary palette permutation preserve addresses.
                permutation = np.random.default_rng(info['n']*391+info['length']).permutation(10)
                assert np.array_equal(solve(permutation[pair['input']]), permutation[expected])
        pair = puzzle['test'][0]
        a = pair['input']
        if max(a.shape) <= 28:
            info = _parse(a)
            shifted = np.pad(a, ((1, 1), (1, 1)), constant_values=info['bg'])
            assert np.array_equal(solve(shifted), pair['output'])
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len(shapes) >= 60 and len(addresses) >= 100 and len(normalized) >= 100
    assert len(palettes) >= 40 and len(blob_areas) >= 5
    assert ns['train'] == {1, 2, 3} and ns['test'] == {1, 2, 3, 4}
    assert input_lengths == {1, 2} and nonrect >= 100 and exterior >= 80
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    # Fixed abstract addresses, independently distorted metric realizations,
    # including larger input spokes, must retain exactly one canonical answer.
    rng = np.random.default_rng(19283)
    for n in range(1, 5):
        pattern = [(0, j) for j in range(n+1)] + [(2, n)]
        answers, drawings = set(), set()
        for k in range(6):
            pair = _make_pair(rng, n, [0, 1, 2], pattern, 1+k % 2, spacious=n <= 2)
            answers.add(_key(solve(pair['input'])))
            drawings.add(_key(pair['input']))
        assert len(answers) == 1 and len(drawings) == 6
    # Explicit small fixture: north inner/exterior beads after two contours.
    info = dict(n=2, bg=0, ink=1, tokens=[(0, 0, 2, 1), (0, 1, 3, 1), (0, 2, 4, 1)])
    b = _analytic_answer(info)
    assert b[4, 6] == 2 and b[4, 10] == 3 and b[12, 10] == 4
    reverse = dict(info, tokens=[(s, 2-j, c, ar) for s, j, c, ar in info['tokens']])
    assert not np.array_equal(b, _analytic_answer(reverse))
    for n in range(1, 5):
        info = dict(n=n, bg=0, ink=1, tokens=[])
        assert np.array_equal(_analytic_answer(info), _draw_answer(n, 0, 1, []))
        mask, _ = _paths(n)
        assert np.array_equal(np.rot90(mask), mask)
        assert not np.any(mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:])
    for a in range(2, 11):
        for bcount in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(471+a*13+bcount, a, bcount))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    for mutation in ('pixel', 'bead', 'crop', 'hand'):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
        pair = bad['test'][0]
        i = _parse(pair['input'])
        if mutation == 'pixel':
            pair['output'][2*i['n']+2, 2*i['n']+2] = i['ink']
        elif mutation == 'bead':
            color = i['tokens'][0][2]
            y, x = np.argwhere(pair['output'] == color)[0]
            pair['output'][y, x] = i['ink']
        elif mutation == 'crop':
            pair['output'] = pair['output'][1:-1, 1:-1]
        else:
            pair['output'] = _draw_answer(i['n'], i['bg'], i['ink'], i['tokens'], handedness=-1)
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError('corrupt answer accepted')
    return dict(seeds=42, pairs=sum(len(p[s]) for p in puzzles for s in p),
                input_shapes=len(shapes), address_patterns=len(addresses),
                bead_layouts=len(normalized), ink_background_palettes=len(palettes),
                blob_areas=sorted(blob_areas), nonrectangular_contours=nonrect,
                exterior_bead_pairs=exterior,
                train_contour_counts=sorted(ns['train']), test_contour_counts=sorted(ns['test']))
