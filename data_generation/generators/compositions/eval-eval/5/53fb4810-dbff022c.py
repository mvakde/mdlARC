"""Phase-matched contour stitching. Only NumPy and the standard library.

The input parser, projection and flood fill have no access to generation state.
No tape is painted beyond its original seed, except at compatible one-cell slits.
"""
from __future__ import annotations

import itertools
import numpy as np

STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'invalid grid dimensions'
    assert np.issubdtype(a.dtype, np.integer), 'integer grids required'
    assert np.all((0 <= a) & (a <= 9)), 'invalid palette'
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    todo = set(zip(*np.nonzero(mask)))
    dirs = STEPS + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else STEPS
    result = []
    while todo:
        first = min(todo)
        todo.remove(first)
        cells, stack = {first}, [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in dirs:
                p = r+dr, c+dc
                if p in todo:
                    todo.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append(cells)
    return result


def _box(cells):
    rr, cc = zip(*cells)
    return min(rr), min(cc), max(rr)-min(rr)+1, max(cc)-min(cc)+1


def _halo(mask):
    out = mask.copy()
    h, w = mask.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r0, r1 = max(0, dr), min(h, h+dr)
            c0, c1 = max(0, dc), min(w, w+dc)
            out[r0:r1, c0:c1] |= mask[r0-dr:r1-dr, c0-dc:c1-dc]
    return out


def _hub_mask(h, w, kind):
    m = np.ones((h, w), dtype=bool)
    if kind == 'clipped':
        m[0, 0] = m[0, -1] = m[-1, 0] = m[-1, -1] = False
    return m


def _hub_kind(mask):
    h, w = mask.shape
    if not (3 <= h <= 9 and 3 <= w <= 9):
        return None
    for kind in ('rectangle', 'clipped'):
        if np.array_equal(mask, _hub_mask(h, w, kind)):
            return kind
    return None


def _slice(box, direction, width, distance):
    t, l, h, w = box
    dr, dc = direction
    if dr:
        r = t-distance if dr == -1 else t+h-1+distance
        return [(r, l+(w-width)//2+j) for j in range(width)]
    c = l-distance if dc == -1 else l+w-1+distance
    return [(t+(h-width)//2+j, c) for j in range(width)]


def _maximum(shape, box, direction):
    t, l, h, w = box
    return {(-1, 0): t, (1, 0): shape[0]-t-h,
            (0, -1): l, (0, 1): shape[1]-l-w}[direction]


def _period(word):
    # ABA is one length-three word. ABAB is equivalent to two whole AB copies.
    return next(p for p in range(1, len(word)+1)
                if len(word) % p == 0 and all(s == word[i % p] for i, s in enumerate(word)))


def _parse(grid):
    colors, counts = np.unique(grid, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1, 'background tie'
    bg = int(colors[np.argmax(counts)])
    key_candidates = []
    for cells in _components(grid != bg, diagonal=True):
        t, l, h, w = _box(cells)
        if len(cells) != h*w or len({int(grid[p]) for p in cells}) < 2:
            continue
        lanes = []
        if h == 2 and w >= 3:
            if t == 0:
                lanes.append(('top', grid[t, l:l+w], grid[t+1, l:l+w]))
            if t+h == grid.shape[0]:
                lanes.append(('bottom', grid[t+1, l:l+w], grid[t, l:l+w]))
        if w == 2 and h >= 3:
            if l == 0:
                lanes.append(('left', grid[t:t+h, l], grid[t:t+h, l+1]))
            if l+w == grid.shape[1]:
                lanes.append(('right', grid[t:t+h, l+1], grid[t:t+h, l]))
        for side, outer, inner in lanes:
            src = tuple(map(int, outer))
            dst = tuple(map(int, inner))
            if len(set(src)) == len(src):
                key_candidates.append(dict(cells=cells, side=side, mapping=dict(zip(src, dst))))
    assert len(key_candidates) == 1, 'one unambiguous isolated edge key required'
    key = key_candidates[0]
    ignored = np.zeros(grid.shape, dtype=bool)
    for p in key['cells']:
        ignored[p] = True
    candidates = []
    for color in colors:
        if color == bg:
            continue
        groups = _components((grid == color) & ~ignored)
        if not groups:
            continue
        if all(_hub_kind(grid[t:t+h, l:l+w] == color) is not None
               for t, l, h, w in map(_box, groups)):
            candidates.append((int(color), groups))
    interpretations = []
    for hub_color, groups in candidates:
        # A distinct geometric hub role in this family; it never occurs in the key.
        if hub_color in set(key['mapping']) | set(key['mapping'].values()):
            continue
        threads = []
        try:
            for cells in groups:
                box = _box(cells)
                t, l, h, w = box
                faces = []
                for direction in STEPS:
                    maximum = _maximum(grid.shape, box, direction)
                    if maximum == 0:
                        continue
                    transverse = w if direction[0] else h
                    face = _slice(box, direction, transverse, 1)
                    indices = [j for j, p in enumerate(face)
                               if int(grid[p]) not in (bg, hub_color) and p not in key['cells']]
                    sw = len(indices)
                    start = (transverse-sw)//2
                    if not 1 <= sw <= transverse-2 or (transverse-sw) % 2:
                        continue
                    if indices != list(range(start, start+sw)):
                        continue
                    dr, dc = direction
                    if all((r-dr, c-dc) in cells for r, c in _slice(box, direction, sw, 1)):
                        faces.append((direction, sw, maximum))
                assert len(faces) == 1, 'hub must have one centered tape'
                direction, sw, maximum = faces[0]
                word, seed_cells = [], set()
                for d in range(1, maximum+1):
                    points = _slice(box, direction, sw, d)
                    values = tuple(int(grid[p]) for p in points)
                    if all(v not in (bg, hub_color) for v in values) and not set(points) & key['cells']:
                        word.append(values)
                        seed_cells.update(points)
                    else:
                        assert all(v == bg for v in values), 'seed ends with a full blank slice'
                        break
                assert word and len(word) < maximum, 'finite seed must have a blank delimiter'
                threads.append(dict(box=box, direction=direction, width=sw, maximum=maximum,
                                    word=word, seed=seed_cells, hub=cells,
                                    kind=_hub_kind(grid[t:t+h, l:l+w] == hub_color), period=_period(word)))
            protected = key['cells'] | set().union(*(s['hub'] | s['seed'] for s in threads))
            paths = set()
            for s in threads:
                ray = {p for d in range(len(s['word'])+1, s['maximum']+1)
                       for p in _slice(s['box'], s['direction'], s['width'], d)}
                assert not ray & protected, 'projection hits protected material'
                assert not ray & paths, 'overlapping projections are outside the family'
                paths.update(ray)
            interpretations.append((hub_color, threads))
        except AssertionError:
            continue
    assert len(interpretations) == 1, 'unambiguous hub role required'
    hub_color, threads = interpretations[0]
    return dict(background=bg, hub_color=hub_color, key=key, threads=threads)


def _stitches(grid, analysis, short_prefix=False, reverse_lanes=False):
    bg = analysis['background']
    proposals = {}
    for thread in analysis['threads']:
        word = thread['word']
        if reverse_lanes:
            word = [tuple(reversed(s)) for s in word]
        if short_prefix:
            p = next(p for p in range(1, len(word)+1)
                     if all(s == word[i % p] for i, s in enumerate(word)))
            word = word[:p]
        dr, dc = thread['direction']
        perpendicular = (0, 1) if dr else (1, 0)
        pr, pc = perpendicular
        for d in range(len(thread['word'])+1, thread['maximum']+1):
            values = word[(d-1) % len(word)]
            for (r, c), v in zip(_slice(thread['box'], thread['direction'], thread['width'], d), values):
                if grid[r, c] != bg:
                    continue
                a, b = (r-pr, c-pc), (r+pr, c+pc)
                if all(0 <= q[0] < grid.shape[0] and 0 <= q[1] < grid.shape[1] for q in (a, b)):
                    if int(grid[a]) == int(grid[b]) == v:
                        proposals[(r, c)] = v
    return proposals


def _regions(grid, bg, diagonal=False):
    h, w = grid.shape
    for cells in _components(grid == bg, diagonal=diagonal):
        border = any(r in (0, h-1) or c in (0, w-1) for r, c in cells)
        boundary = {q for r, c in cells for dr, dc in STEPS
                    for q in [(r+dr, c+dc)]
                    if 0 <= q[0] < h and 0 <= q[1] < w and grid[q] != bg}
        yield dict(cells=cells, border=border, boundary=boundary)


def _fill(stitched, analysis, diagonal=False):
    out = stitched.copy()
    fills = {}
    for region in _regions(stitched, analysis['background'], diagonal=diagonal):
        if region['border']:
            continue
        assert not region['boundary'] & analysis['key']['cells'], 'key cannot bound a cavity'
        boundary_colors = {int(stitched[p]) for p in region['boundary']}
        assert len(boundary_colors) == 1, 'mixed cavity ownership is outside the family'
        color = next(iter(boundary_colors))
        if color in analysis['key']['mapping']:
            v = analysis['key']['mapping'][color]
            for p in region['cells']:
                out[p] = v
                fills[p] = v
    return out, fills


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    analysis = _parse(grid)
    stitched = grid.copy()
    for p, v in _stitches(grid, analysis).items():
        stitched[p] = v
    return _fill(stitched, analysis)[0]


def _analyze(inp, out=None):
    inp = _grid(inp)
    a = _parse(inp)
    stitches = _stitches(inp, a)
    stitched = inp.copy()
    for p, v in stitches.items():
        stitched[p] = v
    expected, fills = _fill(stitched, a)
    if out is not None:
        assert np.array_equal(expected, _grid(out)), 'incorrect output'
    # Explicit counterfactuals, all based on the input pixels.
    without_a, _ = _fill(inp, a)
    prefix_stitched = inp.copy()
    for p, v in _stitches(inp, a, short_prefix=True).items():
        prefix_stitched[p] = v
    prefix_output, _ = _fill(prefix_stitched, a)
    reversed_stitched = inp.copy()
    for p, v in _stitches(inp, a, reverse_lanes=True).items():
        reversed_stitched[p] = v
    reversed_output, _ = _fill(reversed_stitched, a)
    diagonal_output, _ = _fill(stitched, a, diagonal=True)
    no_phase = inp.copy()
    mismatches = set()
    for s in a['threads']:
        pr, pc = (0, 1) if s['direction'][0] else (1, 0)
        for d in range(len(s['word'])+1, s['maximum']+1):
            vals = s['word'][(d-1) % len(s['word'])]
            for (r, c), v in zip(_slice(s['box'], s['direction'], s['width'], d), vals):
                p, x, y = (r, c), (r-pr, c-pc), (r+pr, c+pc)
                if inp[p] == a['background'] and all(0 <= q[0] < inp.shape[0] and 0 <= q[1] < inp.shape[1] for q in (x, y)):
                    if inp[x] == inp[y] != a['background']:
                        no_phase[p] = inp[x]
                        if int(inp[x]) != v:
                            mismatches.add(p)
    indiscriminate, _ = _fill(no_phase, a)
    sealed_stitches, leaky_stitches, inert_stitches = set(), set(), set()
    hole_regions = [reg for reg in _regions(stitched, a['background']) if not reg['border']]
    for p, v in stitches.items():
        neighboring_holes = [reg for reg in hole_regions if p in reg['boundary']]
        if v not in a['key']['mapping']:
            if neighboring_holes:
                inert_stitches.add(p)
        elif neighboring_holes:
            sealed_stitches.add(p)
        else:
            leaky_stitches.add(p)
    a.update(stitches=stitches, fills=fills, sealed=sealed_stitches, leaky=leaky_stitches,
             inert=inert_stitches, mismatches=mismatches,
             a_essential=not np.array_equal(without_a, expected),
             b_essential=not np.array_equal(stitched, expected),
             phase_essential=not np.array_equal(indiscriminate, expected),
             whole_word_essential=not np.array_equal(prefix_output, expected),
             four_connected_essential=not np.array_equal(diagonal_output, expected),
             lane_order_essential=not np.array_equal(reversed_output, expected),
             holes=hole_regions, expected=expected)
    return a


def _tile(rng, width, word):
    tile = np.empty((len(word), width), dtype=np.int8)
    tile[:, 0] = word
    for j in range(1, width):
        if len(word) == 3 and word[0] == word[-1]:
            # Keep the whole vector word ABA, including asymmetric lanes.
            tile[:, j] = [5-x for x in word] if j == 1 or rng.random() < .5 else word
        else:
            tile[:, j] = np.roll(word, int(rng.integers(1 if j == 1 else 0, len(word))))
    return tile


def _outline(rng, h, w, kind, gap_row, side, leaky):
    m = np.ones((h, w), dtype=bool)
    m[1:-1, 1:-1] = False
    if kind == 'profile' and w >= 4 and h >= 6:
        # A dent makes a nonrectangular cavity without creating one-cell corridors.
        m[1, w-2] = True
    if kind == 'multi' and w >= 5:
        m[:, w//2] = True
    if kind == 'clipped':
        m[0, 0] = m[0, -1] = m[-1, 0] = m[-1, -1] = False
    gap_col = 0 if side == 'near' else w-1
    m[gap_row, gap_col] = False
    if leaky:
        # This opening is perpendicular to the tape and outside its lanes.
        choices = [c for c in range(1, w-1) if not m[1, c]]
        m[0, int(rng.choice(choices))] = False
    return m


def _plan(rng, shape, box, tile, roles):
    """Backtracking over positions, widths, gap faces and lanes; no rejection loop."""
    t, l, h, w = box
    first = l+w+len(tile)+1  # an entirely blank slice separates seed and contours
    options = []
    for role in roles:
        choices = []
        for left in range(first, shape[1]-2):
            for cw in range(3, min(7, shape[1]-left+1)):
                for side in ('near', 'far'):
                    x = left if side == 'near' else left+cw-1
                    d = x-(l+w-1)
                    for lane in range(tile.shape[1]):
                        v = int(tile[(d-1) % len(tile), lane])
                        target = 2 if role in ('success', 'mismatch') else 4 if role == 'inert' else v
                        ok = (v == 2 if role == 'success' else v != 2 if role == 'mismatch'
                              else v == 4 if role == 'inert' else v in (2, 3))
                        if ok:
                            choices.append((left, cw, side, lane, target))
        rng.shuffle(choices)
        options.append(choices)
    def search(i, after):
        if i == len(roles):
            return []
        for cand in options[i]:
            left, cw, side, lane, target = cand
            if left < after:
                continue
            # Space for remaining minimally sized outlines and empty collars.
            if left+cw+(len(roles)-i-1)*4 > shape[1]:
                continue
            rest = search(i+1, left+cw+1)
            if rest is not None:
                return [cand]+rest
        return None
    return search(0, first)


def _primary(rng, length, width, third):
    H = int(rng.integers(18, 31))
    W = int(rng.integers(27, 31))
    # Internal construction identities are recolored bijectively before return.
    tile_word = [2, 3, 2] if third == 'leaky' and length == 3 else [2, 3, 4]
    if length > 3:
        tile_word = [2, 3, 4] + list(map(int, rng.choice([2, 3, 4], length-3)))
    tile = _tile(rng, width, tile_word)
    transverse = width+2*int(rng.integers(1, 3))
    longitudinal = int(rng.integers(3, 5))
    left = int(rng.integers(0, 2))
    y = int(rng.integers(7, H-6))
    box = (y-(transverse-width)//2, left, transverse, longitudinal)
    roles = ['success', 'mismatch', third]
    planned = _plan(rng, (H, W), box, tile, roles)
    if planned is None:
        # Wider canvas/shorter hub are sufficient for every supported tile.
        H, W, left, longitudinal, y = 30, 30, 0, 3, 14
        box = (y-(transverse-width)//2, left, transverse, longitudinal)
        planned = _plan(rng, (H, W), box, tile, roles)
    assert planned is not None, 'finite placement space unexpectedly exhausted'
    grid = np.zeros((H, W), dtype=np.int8)
    t, l, h, w = box
    kind = str(rng.choice(['rectangle', 'clipped']))
    grid[t:t+h, l:l+w][_hub_mask(h, w, kind)] = 1
    for d, values in enumerate(tile, 1):
        for p, v in zip(_slice(box, (0, 1), width, d), values):
            grid[p] = v
    for index, (role, (left, cw, side, lane, target)) in enumerate(zip(roles, planned)):
        ch = int(rng.integers(5, min(10, H-3)))
        gap_row = int(rng.integers(2, ch-2))
        top = y+lane-gap_row
        top = max(3, min(H-ch-1, top))
        gap_row = y+lane-top
        assert 1 <= gap_row < ch-1
        contour_kind = str(rng.choice(['rectangle', 'clipped', 'profile', 'multi']))
        if index == 0 and third == 'leaky':
            contour_kind = 'clipped'  # a four- versus eight-connectivity witness
        # A leak in a multi-hole object could coexist with a closed hole;
        # the mandatory leak witness is deliberately a single cavity.
        if role == 'leaky' and contour_kind == 'multi':
            contour_kind = 'profile'
        mask = _outline(rng, ch, cw, contour_kind, gap_row, side, role == 'leaky')
        grid[top:top+ch, left:left+cw][mask] = target
    # Directed chains are present in both splits, with A's successful closure
    # showing 2 -> 3 rather than following 3 -> 6 recursively.
    n = int(rng.integers(3, 6))
    sources = [2, 3, 5, 7, 8][:n]
    fills = [3, 6] + list(map(int, rng.choice([2, 3, 4, 6, 9], n-2)))
    order = rng.permutation(n)
    start = int(rng.integers(1, W-n))
    grid[0, start:start+n] = np.array(sources)[order]
    grid[1, start:start+n] = np.array(fills)[order]
    path = np.zeros(grid.shape, dtype=bool)
    for d in range(1, _maximum(grid.shape, box, (0, 1))+1):
        for p in _slice(box, (0, 1), width, d):
            path[p] = True
    return grid, path


def _add_thread(rng, grid, paths):
    """Try independent directions and free placements in the remaining space."""
    for attempt in range(90):
        k = int(rng.integers(4))
        work = np.rot90(grid, -k).copy()
        occupied = np.rot90(_halo((grid != 0) | paths), -k)
        H, W = work.shape
        sw = int(rng.integers(1, 3))
        sh, hw = sw+2, 3
        period = int(rng.integers(2, 4))
        if W < hw+period+6 or H < sh+4:
            continue
        t = int(rng.integers(2, H-sh))
        l = int(rng.integers(0, W-hw-period-4))
        box = (t, l, sh, hw)
        ray = np.zeros(work.shape, dtype=bool)
        ray[t+1:t+1+sw, l+hw:] = True
        body = np.zeros(work.shape, dtype=bool)
        body[t:t+sh, l:l+hw] = True
        if np.any(occupied & (ray | _halo(body))):
            continue
        tile = _tile(rng, sw, [2, 3, 4][:period])
        minx = l+hw+period+1
        possible = []
        for x in range(minx, W-2):
            v = int(tile[(x-(l+hw)) % period, 0])
            if v not in (2, 3, 4):
                continue
            top = t+1-2
            mask = _outline(rng, 5, 3, 'clipped', 2, 'near', False)
            if top < 2 or top+5 >= H:
                continue
            cb = np.zeros(work.shape, dtype=bool)
            cb[top:top+5, x:x+3] = True
            if not np.any(occupied & _halo(cb)):
                possible.append((x, top, mask, v))
        if not possible:
            continue
        x, top, mask, v = possible[int(rng.integers(len(possible)))]
        work[t:t+sh, l:l+hw][_hub_mask(sh, hw, 'clipped')] = 1
        for d, values in enumerate(tile, 1):
            for p, c in zip(_slice(box, (0, 1), sw, d), values):
                work[p] = c
        work[top:top+5, x:x+3][mask] = v
        return np.rot90(work, k).copy(), paths | np.rot90(ray, k)
    return grid, paths


def _safe_scene(length, width, third):
    """Constructive escape from bounded random packing, with all core witnesses."""
    inp = np.zeros((30, 30), dtype=np.int8)
    word = [2, 3, 2] if length == 3 and third == 'leaky' else [2, 3, 4, 2, 3][:length]
    tile = np.empty((length, width), dtype=np.int8)
    for j in range(width):
        tile[:, j] = ([5-x for x in word] if j % 2 else word) if word == [2, 3, 2] else np.roll(word, j)
    box = (13, 0, width+2, 3)
    inp[13:15+width, :3] = 1
    for d, values in enumerate(tile, 1):
        for p, v in zip(_slice(box, (0, 1), width, d), values):
            inp[p] = v
    after = 4+length
    for role, phase, color in [('success', 0, 2), ('mismatch', 1, 2),
                                (third, 2 if third == 'inert' else 0, 4 if third == 'inert' else 2)]:
        x = next(x for x in range(after, 28) if (x-3) % length == phase)
        mask = np.ones((7, 3), dtype=bool)
        mask[1:-1, 1] = False
        mask[3, 0] = False
        if role == 'success':
            mask[0, 0] = mask[0, -1] = mask[-1, 0] = mask[-1, -1] = False
        if role == 'leaky':
            mask[0, 1] = False
        inp[11:18, x:x+3][mask] = color
        after = x+4
    inp[0, 11:14] = [2, 3, 5]
    inp[1, 11:14] = [3, 6, 6]
    return inp


def _make_pair(rng, length, width, third, turn, extras):
    for attempt in range(16):
        inp, paths = _primary(rng, length, width, third)
        for _ in range(extras):
            inp, paths = _add_thread(rng, inp, paths)
        palette = rng.permutation(10).astype(np.int8)
        inp = palette[np.rot90(inp, turn)].copy()
        try:
            out = solve(inp)
            a = _analyze(inp, out)
            assert a['sealed'] and a['phase_essential'] and a['a_essential'] and a['b_essential']
            if third == 'leaky':
                assert a['leaky']
                if length == 3:
                    assert a['whole_word_essential'] and a['four_connected_essential']
            if third == 'inert':
                assert a['inert']
                if width >= 2:
                    assert a['lane_order_essential']
            return dict(input=inp, output=out)
        except AssertionError:
            continue
    # This branch requires no random packing and respects the same visible rule.
    inp = rng.permutation(10).astype(np.int8)[np.rot90(_safe_scene(length, width, third), turn)].copy()
    out = solve(inp)
    a = _analyze(inp, out)
    assert a['sealed'] and a['phase_essential'] and a['a_essential'] and a['b_essential']
    assert a['leaky'] if third == 'leaky' else a['inert']
    return dict(input=inp, output=out)


def _count(name, value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not lo <= int(value) <= hi:
        raise ValueError(f'{name} must be between {lo} and {hi}')
    return int(value)


def _fingerprint(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _unique_pair(pair, used):
    if _fingerprint(pair['input']) not in used:
        return pair
    colors = [int(c) for c in np.unique(pair['input'])]
    # At most 13 exclusions, and at least six visible colors: an injective
    # recoloring has far more candidates than a legal puzzle can exclude.
    for values in itertools.permutations(range(10), len(colors)):
        palette = np.arange(10, dtype=np.int8)
        palette[colors] = values
        candidate = {k: palette[v] for k, v in pair.items()}
        if _fingerprint(candidate['input']) not in used:
            return candidate
    raise AssertionError('injective recoloring space exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a = _count('num_train', num_train, 2, 10)
    b = _count('num_test', num_test, 1, 4)
    if a <= b:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed) % (1 << 64))
    turns = list(map(int, rng.permutation(4)))
    regime = str(rng.choice(['direction', 'longer']))
    puzzle, used = {'train': [], 'test': []}, set()
    for split, count in [('train', a), ('test', b)]:
        for i in range(count):
            if split == 'train':
                third = 'leaky' if i == 0 else 'inert' if i == 1 else str(rng.choice(['leaky', 'inert']))
                length = 3
                width = int(rng.integers(2, 4)) if i == 1 else int(rng.integers(1, 4))
                turn = turns[i % 2]
            else:
                third = str(rng.choice(['leaky', 'inert']))
                length = int(rng.integers(4, 6)) if regime == 'longer' else 3
                width = int(rng.integers(1, 4))
                turn = turns[2+i % 2] if regime == 'direction' else turns[i % 2]
            pair = _make_pair(rng, length, width, third, turn, int(rng.integers(3)))
            pair = _unique_pair(pair, used)
            used.add(_fingerprint(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    a, b = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    used, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            fp = _fingerprint(inp)
            assert fp not in used, 'duplicate input'
            used.add(fp)
            analysis = _analyze(inp, out)
            assert np.array_equal(inp[inp != analysis['background']], out[inp != analysis['background']]), 'original ink changed'
            protected = analysis['key']['cells'] | set().union(*(s['hub'] | s['seed'] for s in analysis['threads']))
            assert all(inp[p] == out[p] for p in protected), 'key, hub or seed changed'
            changed = set(zip(*np.nonzero(inp != out)))
            assert changed == set(analysis['stitches']) | set(analysis['fills']), 'writes outside permitted regions'
            assert analysis['a_essential'] and analysis['b_essential'], 'one parent is dispensable'
            records[split].append(analysis)
    train = records['train']
    assert any(x['sealed'] for x in train), 'teach successful topological closure'
    assert any(x['leaky'] for x in train), 'teach that a stitch need not enclose'
    assert any(x['inert'] for x in train), 'teach key selectivity'
    assert any(x['phase_essential'] for x in train), 'teach phase matching'
    assert any(x['whole_word_essential'] for x in train), 'teach the complete word by its answer'
    assert any(x['four_connected_essential'] for x in train), 'teach four-neighbor enclosure'
    assert any(x['lane_order_essential'] for x in train), 'teach lane order with a changed answer'
    assert any(s['width'] >= 2 for x in train for s in x['threads']), 'teach vector slices'
    assert any(s['word'][0] == s['word'][-1] and s['period'] == 3
               for x in train for s in x['threads']), 'teach whole ABA word rather than AB prefix'
    # The active closure with a source-to-source destination proves one lookup.
    assert any(x['key']['mapping'].get(v) in x['key']['mapping'] and
               x['key']['mapping'][x['key']['mapping'][v]] != x['key']['mapping'][v]
               for x in train for p, v in x['stitches'].items() if p in x['sealed']), 'teach nonrecursive lookup'


def self_test():
    assert _period([(2,), (3,), (2,)]) == 3
    assert _period([(2,), (3,), (2,), (3,)]) == 2
    for length in (3, 4, 5):
        for width in (1, 2, 3):
            for third in ('leaky', 'inert'):
                inp = _safe_scene(length, width, third)
                a = _analyze(inp, solve(inp))
                assert a['a_essential'] and a['b_essential'] and a['phase_essential']
                assert a['leaky'] if third == 'leaky' else a['inert']
                if length == 3 and third == 'leaky':
                    assert a['whole_word_essential'] and a['four_connected_essential']
                if width >= 2 and third == 'inert':
                    assert a['lane_order_essential']
    original = {'input': inp, 'output': solve(inp)}
    used = set()
    for _ in range(14):
        unique = _unique_pair(original, used)
        used.add(_fingerprint(unique['input']))
        assert np.array_equal(solve(unique['input']), unique['output'])
    # Hand-specified constant word, self-map, shared destination and a closed
    # chamber that needs no stitch. Labels here are derived directly by cells.
    manual = np.zeros((13, 24), dtype=np.int8)
    manual[0, 10:13] = [2, 3, 5]
    manual[1, 10:13] = [2, 6, 6]
    manual[6:9, :3] = 1
    manual[7, 3] = 2
    for left, color in ((8, 2), (16, 3)):
        manual[5:10, left:left+5] = color
        manual[6:9, left+1:left+4] = 0
    manual[7, 8] = 0
    expected = manual.copy()
    expected[7, 8] = 2
    expected[6:9, 9:12] = 2
    expected[6:9, 17:20] = 6
    for k in range(4):
        assert np.array_equal(solve(np.rot90(manual, k)), np.rot90(expected, k))
    # Complete repeated copies are equivalent; added original seed cells remain.
    longer_seed, longer_expected = manual.copy(), expected.copy()
    longer_seed[7, 3:7] = longer_expected[7, 3:7] = 2
    assert np.array_equal(solve(longer_seed), longer_expected)
    # A background destination / hub color in the key is outside conventions.
    # Altering a destination among legal payload colors remains input-defined.
    changed_key, changed_expected = manual.copy(), expected.copy()
    changed_key[1, 11] = changed_expected[1, 11] = 9
    changed_expected[6:9, 17:20] = 9
    assert np.array_equal(solve(changed_key), changed_expected)
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    fingerprints, shapes, structures = set(), set(), set()
    features = {'widths': set(), 'periods': set(), 'directions': set(), 'backgrounds': set(),
                'thread_counts': set(), 'key_lengths': set(), 'hole_sizes': set()}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        fingerprints.add(tuple(_fingerprint(q['input']) for s in ('train', 'test') for q in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                a = _analyze(inp, out)
                shapes.add(inp.shape)
                roles = {}
                normalized = tuple(roles.setdefault(int(c), len(roles)) for c in inp.flat)
                structures.add((inp.shape, normalized))
                features['backgrounds'].add(a['background'])
                features['thread_counts'].add(len(a['threads']))
                features['key_lengths'].add(len(a['key']['mapping']))
                features['hole_sizes'].update(len(r['cells']) for r in a['holes'])
                for s in a['threads']:
                    features['widths'].add(s['width'])
                    features['periods'].add(s['period'])
                    features['directions'].add(s['direction'])
                # Geometric equivariance, using only transformed inputs.
                for k in (1,):
                    assert np.array_equal(solve(np.rot90(inp, k)), np.rot90(out, k))
                assert np.array_equal(solve(np.fliplr(inp)), np.fliplr(out))
    assert len(fingerprints) == 42 and len(structures) >= 180
    assert len(shapes) >= 30
    assert features['widths'] == {1, 2, 3}
    assert features['directions'] == set(STEPS)
    assert features['backgrounds'] == set(range(10))
    assert features['periods'] >= {2, 3, 4, 5}
    assert len(features['thread_counts']) >= 2 and len(features['hole_sizes']) >= 10
    assert features['key_lengths'] == {3, 4, 5}
    assert tuple(_fingerprint(q['input']) for s in ('train', 'test') for q in generate_puzzle(0)[s]) == tuple(_fingerprint(q['input']) for s in ('train', 'test') for q in puzzles[0][s])
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            p = generate_puzzle(123, train, test)
            assert len(p['train']) == train and len(p['test']) == test
    for train, test in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in ('train', 'test')}
    p = damaged['test'][0]
    at = min(_parse(p['input'])['key']['cells'])
    p['output'][at] = (int(p['output'][at])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('key mutation accepted')
    return {'seeds': 42, 'pairs': 210, 'distinct_input_structures': len(structures),
            'input_shapes': len(shapes), 'features': {k: sorted(v) for k, v in features.items()}}
