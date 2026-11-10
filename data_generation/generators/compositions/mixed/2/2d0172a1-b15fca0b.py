"""Canonical closed-frame street maps, followed by unique unit-step routing.

Only NumPy and the standard library are used. No files, seeds, examples, or
generation metadata are consulted by solve(). See task_spec.md for the domain.
"""
from collections import deque
from functools import lru_cache
from itertools import product
import hashlib

import numpy as np


ROUTE = 4  # A fixed family convention: absent from every input.
HOST_SEEDS = (0, 924491657, 459763887, 117452902, 419467130, 760762025,
              243263401, 163304147, 404704322, 732849900, 553373197,
              393460441, 282547102, 559597042, 795682366, 377859669,
              286605386, 354112614, 317062742, 353113654, 467124515,
              159477789, 2588819, 581150027, 917583324, 310039879,
              829960272, 613104256, 855147250, 659793019, 303997420,
              882092128, 611253853, 750286048, 973778855, 475468239,
              21229136, 185798974, 98769763, 497308601, 728311412, 943782045)
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid bounds")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)),
             "integer colors 0..9 required")
    return a.astype(np.int8, copy=False)


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    answer = []
    for r, c in zip(*np.nonzero(mask)):
        if seen[r, c]:
            continue
        stack = [(int(r), int(c))]
        seen[r, c] = True
        cells = []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in _DIRS:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        coords = np.array(cells, dtype=int)
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        local = np.zeros(tuple(hi - lo + 1), dtype=bool)
        local[coords[:, 0] - lo[0], coords[:, 1] - lo[1]] = True
        answer.append(dict(cells=coords, lo=lo, hi=hi, mask=local))
    return answer


def _holes(mask):
    padded = np.pad(mask, 1)
    exterior = np.zeros(padded.shape, dtype=bool)
    exterior[0, 0] = True
    stack = [(0, 0)]
    h, w = padded.shape
    while stack:
        y, x = stack.pop()
        for dy, dx in _DIRS:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and not padded[yy, xx] and not exterior[yy, xx]:
                exterior[yy, xx] = True
                stack.append((yy, xx))
    return (~padded & ~exterior)[1:-1, 1:-1]


def _inside(obj, cells):
    relative = cells - obj['lo']
    if np.any(relative < 0) or np.any(relative >= obj['hole'].shape):
        return False
    return bool(obj['hole'][relative[:, 0], relative[:, 1]].all())


def _clusters(objects, axis):
    """Unambiguous rows/columns from overlapping bounding-box projections."""
    groups, end = [], -1
    for obj in sorted(objects, key=lambda o: int(o['lo'][axis])):
        start = int(obj['lo'][axis])
        if not groups or start > end:
            groups.append([])
        groups[-1].append(obj)
        end = max(end, int(obj['hi'][axis]))
    for group in groups:
        _require(max(int(o['lo'][axis]) for o in group) <=
                 min(int(o['hi'][axis]) for o in group), "ambiguous projection cluster")
    return groups


def _decode_objects(a):
    border = np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))
    _require(np.unique(border).size == 1, "uniform background border required")
    bg = int(border[0])
    colors = set(int(v) for v in np.unique(a))
    _require(len(colors) == 3 and ROUTE not in colors, "three input roles; 4 reserved")
    roles = {}
    for color in sorted(colors - {bg}):
        objs = _components(a == color)
        for obj in objs:
            obj['hole'] = _holes(obj['mask'])
        roles[color] = objs
    contour_colors = [c for c, objs in roles.items() if any(o['hole'].any() for o in objs)]
    _require(len(contour_colors) == 1, "one contour role required")
    wall = contour_colors[0]
    terminal = next(iter(colors - {bg, wall}))
    marks = roles[terminal]
    _require(len(marks) == 2 and not any(o['hole'].any() for o in marks),
             "exactly two marked filled blobs required")
    frames = [o for o in roles[wall] if o['hole'].any()]
    leaves = [o for o in roles[wall] if not o['hole'].any()]
    _require(frames and leaves, "frames with enclosed unmarked blobs required")
    for obj in leaves + marks:
        _require(obj['mask'].sum() >= .45 * obj['mask'].size, "compact filled blob required")
    for frame in frames:
        _require(len(_components(frame['hole'])) == 1, "one hole per contour")
        _require(not any(_inside(frame, f['cells']) for f in frames if f is not frame),
                 "this family uses unnested sibling contours")
        children = [b for b in leaves if _inside(frame, b['cells'])]
        _require(children, "empty frame")
        rows, cols = _clusters(children, 0), _clusters(children, 1)
        _require(len(children) == len(rows) * len(cols), "complete blob lattice required")
        for row in rows:
            for col in cols:
                _require(sum(any(o is p for p in col) for o in row) == 1,
                         "one blob per lattice site required")
        frame['nr'], frame['nc'] = len(rows), len(cols)
    _require(all(sum(_inside(f, b['cells']) for f in frames) == 1 for b in leaves),
             "every unmarked blob belongs to one frame")
    _require(not any(_inside(f, b['cells']) for f in frames for b in marks),
             "terminals must be in the common exterior face")
    return (bg, wall, terminal), frames, marks


def _pack(bands, top_gap, bottom_gap):
    """Construct canonical geometry from visible/composed rows of (nr,nc)."""
    widths = [sum(2 * nc + 3 for nr, nc in row) + len(row) - 1 for row in bands]
    _require(len(set(widths)) == 1, "all bands must normalize to equal width")
    heights = []
    for row in bands:
        hs = {2 * nr + 3 for nr, nc in row}
        _require(len(hs) == 1, "frames in a band must normalize to equal height")
        heights.append(hs.pop())
    height = sum(heights) + len(bands) - 1
    a = np.zeros((height + 6, widths[0] + 2), dtype=np.int8)
    boxes, gutter_columns = [], []
    y = 3
    for row, h in zip(bands, heights):
        x, gutters = 1, []
        for index, (nr, nc) in enumerate(row):
            w = 2 * nc + 3
            a[y, x:x+w] = a[y+h-1, x:x+w] = 1
            a[y:y+h, x] = a[y:y+h, x+w-1] = 1
            a[y+2:y+h-2:2, x+2:x+w-2:2] = 1
            boxes.append((y, x, y+h-1, x+w-1))
            x += w
            if index < len(row) - 1:
                gutters.append(x)
                x += 1
        gutter_columns.append(gutters)
        y += h + 1
    _require(0 <= top_gap < len(gutter_columns[0]) and
             0 <= bottom_gap < len(gutter_columns[-1]), "terminal gutter index")
    a[1, gutter_columns[0][top_gap]] = 2
    a[height+4, gutter_columns[-1][bottom_gap]] = 2
    return a, boxes, gutter_columns


def _canonicalize(input_grid):
    a = _grid(input_grid)
    palette, frames, marks = _decode_objects(a)
    # The terminal placements visibly select the band axis. Transposition is
    # undone before returning; no preferred physical orientation is imposed.
    oriented = None
    for axis in (0, 1):
        ordered_marks = sorted(marks, key=lambda o: int(o['lo'][axis]))
        if (ordered_marks[0]['hi'][axis] < min(f['lo'][axis] for f in frames) and
                ordered_marks[1]['lo'][axis] > max(f['hi'][axis] for f in frames)):
            _require(oriented is None, "ambiguous band axis")
            oriented = axis
    _require(oriented is not None, "marks must bracket all frame bands")
    axis = oriented
    cross = 1 - axis
    rows = _clusters(frames, axis)
    _require(2 <= len(rows) <= 4, "two to four frame bands required")
    bands, extents = [], []
    for row in rows:
        row.sort(key=lambda o: int(o['lo'][cross]))
        _require(2 <= len(row) <= 4, "two to four frames per band")
        _require(len({(int(o['lo'][axis]), int(o['hi'][axis])) for o in row}) == 1,
                 "band frames need common bounding-box ends")
        _require(all(b['lo'][cross] - f['hi'][cross] >= 2 for f, b in zip(row, row[1:])),
                 "separated frame columns required")
        extents.append((int(row[0]['lo'][cross]), int(row[-1]['hi'][cross])))
        bands.append([(f['nr'], f['nc']) if axis == 0 else (f['nc'], f['nr']) for f in row])
    _require(len(set(extents)) == 1, "aligned outer band extents required")
    ordered_marks = sorted(marks, key=lambda o: int(o['lo'][axis]))
    gap_indices = []
    for mark, row in zip(ordered_marks, (rows[0], rows[-1])):
        matches = [i for i, (left, right) in enumerate(zip(row, row[1:]))
                   if left['hi'][cross] < mark['lo'][cross] <= mark['hi'][cross] < right['lo'][cross]]
        _require(len(matches) == 1, "marked blob must align with one visible gutter")
        gap_indices.append(matches[0])
    canonical, boxes, gutters = _pack(bands, *gap_indices)
    _require(max(canonical.shape) <= 30, "canonical bounds")
    info = dict(palette=palette, bands=bands, boxes=boxes, gutters=gutters,
                gaps=gap_indices, transpose=axis == 1,
                raw_frame_count=len(frames), raw_blob_count=sum(f['nr']*f['nc'] for f in frames))
    return (canonical.T.copy() if axis == 1 else canonical), info


def _search(blocked, start, goal):
    """BFS, counting shortest paths with saturation at two."""
    h, w = blocked.shape
    distance = np.full((h, w), -1, dtype=np.int16)
    ways = np.zeros((h, w), dtype=np.int8)
    parents = {}
    queue = deque([start])
    distance[start], ways[start] = 0, 1
    while queue:
        y, x = queue.popleft()
        d = int(distance[y, x]) + 1
        for dy, dx in _DIRS:
            yy, xx = y + dy, x + dx
            if not (0 <= yy < h and 0 <= xx < w) or blocked[yy, xx]:
                continue
            q = (yy, xx)
            if distance[q] < 0:
                distance[q], ways[q] = d, ways[y, x]
                parents[q] = (y, x)
                queue.append(q)
            elif distance[q] == d:
                ways[q] = min(2, int(ways[q]) + int(ways[y, x]))
    if distance[goal] < 0:
        return [], 0, distance
    path = [goal]
    while path[-1] != start:
        path.append(parents[path[-1]])
    return path[::-1], int(ways[goal]), distance


def _route(canonical):
    terminals = [tuple(int(v) for v in p) for p in np.argwhere(canonical == 2)]
    _require(len(terminals) == 2, "two canonical terminals")
    path, count, distance = _search(canonical == 1, *terminals)
    _require(count == 1 and len(path) > 2, "unique nontrivial shortest path required")
    return path, distance


def _colorize(canonical, palette):
    lookup = np.arange(10, dtype=np.int8)
    lookup[:3] = palette
    lookup[ROUTE] = ROUTE
    return lookup[canonical]


def solve(input_grid: np.ndarray) -> np.ndarray:
    canonical, info = _canonicalize(input_grid)
    path, _ = _route(canonical)
    result = canonical.copy()
    for cell in path[1:-1]:
        _require(result[cell] == 0, "only background may be routed")
        result[cell] = ROUTE
    return _colorize(result, info['palette'])


@lru_cache(maxsize=64)
def _partitions(width, count):
    # 2*sum(nc)+4*count-1 == width, nc in 1..6.
    total = (width + 1 - 4 * count) // 2
    return tuple(p for p in product(range(1, 7), repeat=count) if sum(p) == total)


def _contrast_factors(canonical, bands, gutters, path):
    """Stretch only the winning detour's side, enough to reverse raw lengths."""
    x0, x1 = path[0][1], path[-1][1]
    if x0 != x1 or len(bands) != 3:
        return None
    mid_y = 4 + (2*bands[0][0][0]+3) + (2*bands[1][0][0]+3)//2
    chosen = [x for y, x in path if y == mid_y and x in gutters[1]]
    if len(chosen) != 1:
        return None
    near = chosen[0]
    far = [x for x in gutters[1] if (x-x0)*(near-x0) < 0 and abs(x-x0) > abs(near-x0)]
    if not far:
        return None
    extra = min(abs(x-x0) for x in far) - abs(near-x0) + 1
    columns = list(range(min(x0, near)+1, max(x0, near)))
    if extra > min(7, 30-canonical.shape[1], 3*len(columns)):
        return None
    factors = np.ones(canonical.shape[1], dtype=int)
    for col in columns:
        add = min(3, extra)
        factors[col] += add
        extra -= add
    return factors


def _raw_choice_mismatch(input_grid, canonical, info):
    """Does every shortest RAW pixel route avoid the canonical winning gutter?

    This diagnoses a shortcut, not the solution rule. Measure the whole width
    of the raw corridor, including insets, rather than a single center pixel.
    """
    if len(info['bands']) != 3:
        return False
    a = np.empty(input_grid.shape, dtype=np.int8)
    for index, color in enumerate(info['palette']):
        a[input_grid == color] = index
    c = canonical
    if info['transpose']:
        a, c = a.T, c.T
    path, _ = _route(c)
    mid_y = 4 + (2*info['bands'][0][0][0]+3) + (2*info['bands'][1][0][0]+3)//2
    chosen = [i for i, x in enumerate(info['gutters'][1]) if (mid_y, x) in path]
    if len(chosen) != 1:
        return False
    _, frames, marks = _decode_objects(a)
    row = _clusters(frames, 0)[1]
    row.sort(key=lambda o: int(o['lo'][1]))
    y = (int(row[0]['lo'][0])+int(row[0]['hi'][0]))//2
    left, right = row[chosen[0]:chosen[0]+2]
    low = int(left['cells'][left['cells'][:, 0] == y, 1].max())+1
    high = int(right['cells'][right['cells'][:, 0] == y, 1].min())
    marks.sort(key=lambda o: int(o['lo'][0]))
    endpoints = []
    for mark in marks:
        coords = mark['cells']
        center = coords.mean(axis=0)
        endpoints.append(tuple(int(v) for v in coords[np.argmin(((coords-center)**2).sum(axis=1))]))
    raw_path, _, ds = _search(a == 1, *endpoints)
    if not raw_path:
        return False
    _, _, dg = _search(a == 1, endpoints[1], endpoints[0])
    valid = (ds[y, low:high] >= 0) & (dg[y, low:high] >= 0)
    costs = (ds[y, low:high]+dg[y, low:high])[valid]
    return bool(costs.size and int(costs.min()) > len(raw_path)-1)


def _sample_map(rng, mode):
    for _ in range(256):
        if mode == 'easy':
            nband, width = 2, int(rng.choice((15, 17, 19, 21, 23)))
            counts = [int(rng.choice((2, 3))) if width >= 17 else 2 for _ in range(2)]
        elif mode == 'choice':
            nband, width = 3, int(rng.choice((19, 21, 23)))
            counts = [int(rng.choice((2, 3))), 3, int(rng.choice((2, 3)))]
        elif mode == 'four':
            nband, width = 4, int(rng.choice((19, 21, 23)))
            counts = [int(rng.choice((2, 3))) for _ in range(4)]
        else:  # Held-out recombination: four competing frame blocks in a band.
            nband, width = 3, int(rng.choice((23, 25)))
            counts = [int(rng.choice((2, 3))), 4, int(rng.choice((2, 3)))]
        choices = [_partitions(width, m) for m in counts]
        if any(not p for p in choices):
            continue
        parts = [ps[int(rng.integers(len(ps)))] for ps in choices]
        if nband == 3 and counts[0] == counts[-1] and rng.random() < .7:
            parts[-1] = parts[0]
        if nband == 2:
            nr = [int(rng.integers(1, 4)) for _ in range(nband)]
        elif nband == 3:
            nr = [1, 1, 1]
            if rng.random() < .3:
                nr[int(rng.integers(3))] = 2
        else:
            nr = [1] * 4
        bands = [[(r, c) for c in p] for r, p in zip(nr, parts)]
        top, bottom = int(rng.integers(counts[0]-1)), int(rng.integers(counts[-1]-1))
        canonical, boxes, gutters = _pack(bands, top, bottom)
        if max(canonical.shape) > 30:
            continue
        try:
            path, _ = _route(canonical)
        except ValueError:
            continue
        manhattan = sum(abs(a-b) for a, b in zip(path[0], path[-1]))
        if _turns(path) < 2:
            continue
        if mode != 'easy' and len(path)-1 <= manhattan:
            continue
        if mode == 'choice' and _contrast_factors(canonical, bands, gutters, path) is None:
            continue
        return canonical, boxes
    raise RuntimeError('bounded map search exhausted for ' + mode)


def _outline(h, w, rng):
    solid = np.ones((h, w), dtype=bool)
    # Corner steps remove metric rectangularity without opening the loop.
    for y, x in ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1)):
        if rng.random() < .65:
            solid[y, x] = False
    # An occasional shallow side bay lies wholly within the empty moat.
    if min(h, w) >= 7 and rng.random() < .45:
        if rng.integers(2):
            y = 0 if rng.integers(2) else h-1
            solid[y, 2:w-2] = False
        else:
            x = 0 if rng.integers(2) else w-1
            solid[2:h-2, x] = False
    padded = np.pad(solid, 1)
    eroded = np.ones_like(solid)
    for dy in range(3):
        for dx in range(3):
            eroded &= padded[dy:dy+h, dx:dx+w]
    return solid & ~eroded


def _stretch_factors(length, rng):
    factors = np.ones(length, dtype=int)
    budget = int(rng.integers(0, min(30-length, 7)+1))
    for _ in range(budget):
        # Input canvas border is never enlarged more than needed for translation.
        factors[int(rng.choice(np.flatnonzero(factors < 4)))] += 1
    return factors


def _metric_drawing(canonical, boxes, rng, x_factors=None):
    fy, fx = _stretch_factors(canonical.shape[0], rng), _stretch_factors(canonical.shape[1], rng)
    if x_factors is not None:
        fx = x_factors
    ys, xs = np.r_[0, np.cumsum(fy)], np.r_[0, np.cumsum(fx)]
    a = np.repeat(np.repeat(canonical, fy, axis=0), fx, axis=1)
    for y, x, yy, xx in boxes:
        lo_y, lo_x, hi_y, hi_x = ys[y], xs[x], ys[yy+1], xs[xx+1]
        view = a[lo_y:hi_y, lo_x:hi_x]
        # Clear only the stretched rectangular perimeter; keep its filled leaves.
        view[:fy[y]] = view[-fy[yy]:] = 0
        view[:, :fx[x]] = view[:, -fx[xx]:] = 0
        outline = _outline(hi_y-lo_y, hi_x-lo_x, rng)
        view[outline] = 1
    # Filled leaf geometry varies independently of the contour distortion.
    for color in (1, 2):
        for blob in _components(a == color):
            if _holes(blob['mask']).any():
                continue
            h, w = blob['mask'].shape
            if h >= 2 and w >= 2 and rng.random() < .65:
                y = int(blob['lo'][0]) + (h-1 if rng.integers(2) else 0)
                x = int(blob['lo'][1]) + (w-1 if rng.integers(2) else 0)
                a[y, x] = 0
    return a


def _key(a):
    return tuple(a.shape), a.astype(np.int8, copy=False).tobytes()


def _make_pair(rng, palette, mode):
    canonical, boxes = _sample_map(rng, mode)
    path, _ = _route(canonical)
    answer = canonical.copy()
    for cell in path[1:-1]:
        answer[cell] = ROUTE
    info = _canonicalize(canonical)[1]
    bias = _contrast_factors(canonical, info['bands'], info['gutters'], path) if mode == 'choice' else None
    for attempt in range(48):
        metric = _metric_drawing(canonical, boxes, rng, bias)
        if np.array_equal(metric, canonical):
            continue
        try:
            recovered, _ = _canonicalize(_colorize(metric, palette))
            if not np.array_equal(recovered, canonical):
                continue
            if mode == 'choice' and not _raw_choice_mismatch(metric, canonical, info):
                continue
        except ValueError:
            continue
        k, reflect = int(rng.integers(4)), bool(rng.integers(2))
        inp, out = np.rot90(metric, k), np.rot90(answer, k)
        if reflect:
            inp, out = inp[:, ::-1], out[:, ::-1]
        return {'input': _colorize(inp, palette).copy(), 'output': _colorize(out, palette).copy()}
    raise RuntimeError('bounded metric drawing search exhausted')


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(name + ' outside legal integer range')
    _require(num_train > num_test, 'num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = tuple(int(c) for c in rng.choice([c for c in range(10) if c != ROUTE], 3, replace=False))
    modes = ['easy', 'choice']
    while len(modes) < ntrain:
        modes.append(str(rng.choice(('easy', 'choice', 'four'))))
    rng.shuffle(modes)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, plan in (('train', modes), ('test', ['test'] * ntest)):
        for mode in plan:
            for _ in range(12):
                pair = _make_pair(rng, palette, mode)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct-input search exhausted')
    return puzzle


def _turns(path):
    directions = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:])]
    return sum(a != b for a, b in zip(directions, directions[1:]))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'pair lists required')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, palettes, witnesses, contrasts = set(), set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            canonical, info = _canonicalize(inp)
            palettes.add(info['palette'])
            _require(not np.array_equal(inp, _colorize(canonical, info['palette'])),
                     'normalization must be visible in generated demonstrations')
            path, ds = _route(canonical)
            _, _, dg = _search(canonical == 1, path[-1], path[0])
            length = len(path)-1
            # A second certificate: exactly D+1 cells lie on ANY shortest path.
            union = (ds >= 0) & (dg >= 0) & (ds + dg == length)
            _require(int(union.sum()) == len(path), 'shortest-path union is not unique')
            expected = _colorize(canonical, info['palette'])
            for cell in path[1:-1]:
                _require(canonical[cell] == 0, 'route overwrites object')
                expected[cell] = ROUTE
            _require(np.array_equal(out, expected), 'incorrect normalized route output')
            _require(int((out == info['palette'][2]).sum()) == 2, 'terminal collapse')
            _require(int((out == ROUTE).sum()) == len(path)-2, 'intermediate-only drawing')
            _require(int((canonical == 0).sum()) > len(path), 'off-route background required')
            _require(_turns(path) >= 2, 'nontrivial obstacle interaction required')
            detour = length - sum(abs(a-b) for a, b in zip(path[0], path[-1]))
            record = (len(info['bands']), max(map(len, info['bands'])), detour)
            if split == 'train':
                witnesses.append(record)
                contrasts.append(_raw_choice_mismatch(inp, canonical, info))
            else:
                _require(record[0] == 3 and record[1] == 4 and detour > 0,
                         'test must recombine familiar routing with four blocks')
    _require(len(palettes) == 1, 'shared puzzle palette')
    _require(any(r == 2 for r, m, d in witnesses), 'two-band demonstration required')
    _require(any(r == 3 and m == 3 and d > 0 for r, m, d in witnesses),
             'three-band unequal-detour demonstration required')
    _require(any(contrasts), 'a training example must distinguish raw-pixel routing from normalization first')


def _signature(puzzle):
    return tuple(_key(pair[k]) for s in ('train', 'test') for pair in puzzle[s] for k in ('input', 'output'))


def _ablation_tests():
    """Visible counterexamples, independent of the stochastic sampler."""
    cases = []
    for counts in (((1, 4), (4, 1)), ((2, 3), (3, 2))):
        # Identical four 7x11 contour boxes, canvas, and marked-blob positions.
        raw = np.zeros((25, 29), dtype=np.int8)
        raw[1, 14] = raw[23, 14] = 2
        for y, row in zip((4, 14), counts):
            for x, count in zip((2, 16), row):
                raw[y, x:x+11] = raw[y+6, x:x+11] = 1
                raw[y:y+7, x] = raw[y:y+7, x+10] = 1
                positions = {1: (5,), 2: (3, 7), 3: (2, 5, 8), 4: (2, 4, 6, 8)}[count]
                for offset in positions:
                    raw[y+3, x+offset] = 1
        canonical, info = _canonicalize(raw)
        expected, _, _ = _pack([[(1, c) for c in row] for row in counts], 0, 0)
        _require(np.array_equal(canonical, expected), 'blob-count normalization witness')
        output = solve(raw)
        _require(np.array_equal(solve(np.pad(raw, ((1, 1), (0, 1)))), output),
                 'translation/canvas-size invariance')
        cases.append((raw, canonical, output))
    _require(cases[0][0].shape == cases[1][0].shape and
             np.array_equal(cases[0][0] == 2, cases[1][0] == 2), 'fixed metric controls')
    _require(not np.array_equal(cases[0][2] == ROUTE, cases[1][2] == ROUTE),
             'omitting A must change highlighted cells')
    # Three bands: both middle gutters cross the same number of bands and need
    # four turns. Their routes have 24 versus 28 unit steps, so B selects one.
    canonical, _, _ = _pack([[(1, 3), (1, 4)], [(1, 1), (1, 1), (1, 3)],
                              [(1, 3), (1, 4)]], 0, 0)
    path, ds = _route(canonical)
    _, _, dg = _search(canonical == 1, path[-1], path[0])
    near, far = (11, 12), (11, 6)
    _require(int(ds[near]+dg[near]) == 24 and int(ds[far]+dg[far]) == 28,
             'unit-length alternative-route witness')
    _require(near in path and far not in path and _turns(path) == 4,
             'station/turn-count shortcut witness')
    # Metric distortion reverses the preferred middle gutter. Normalizing first
    # restores the original route; routing the pixels first prefers the left.
    factors = np.ones(canonical.shape[1], dtype=int)
    factors[11] = 4
    stretched = np.repeat(canonical, factors, axis=1)
    restored, _ = _canonicalize(stretched)
    _require(np.array_equal(restored, canonical) and
             np.array_equal(solve(stretched), solve(canonical)), 'metric invariance')
    endpoints = [tuple(int(v) for v in p) for p in np.argwhere(stretched == 2)]
    raw_path, _, raw_ds = _search(stretched == 1, *endpoints)
    _, _, raw_dg = _search(stretched == 1, endpoints[1], endpoints[0])
    _require(int(raw_ds[11, 6]+raw_dg[11, 6]) == len(raw_path)-1 and
             int(raw_ds[11, 15]+raw_dg[11, 15]) > len(raw_path)-1,
             'routing before A selects the wrong gutter')
    return cases


def self_test():
    """42 seeds, streamed; no puzzle corpus or trace is written or retained."""
    shapes, layouts, palettes, counts, terminals, detours, blob_counts = (set() for _ in range(7))
    nonrect = total = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                _require(np.array_equal(solve(inp), out), 'solver regression')
                canonical, info = _canonicalize(inp)
                path, _ = _route(canonical)
                shapes.add(inp.shape)
                layouts.add(hashlib.sha256(canonical.tobytes()+bytes(canonical.shape)).digest())
                palettes.add(info['palette'])
                counts.add(info['raw_frame_count'])
                terminals.add(tuple(tuple(p) for p in np.argwhere(canonical == 2)))
                detours.add(len(path)-1-sum(abs(a-b) for a, b in zip(path[0], path[-1])))
                blob_counts.add(info['raw_blob_count'])
                _, frames, _ = _decode_objects(inp)
                for frame in frames:
                    m = frame['mask']
                    total += 1
                    nonrect += not (m[0].all() and m[-1].all() and m[:, 0].all() and m[:, -1].all())
        # Rotation, reflection, and background translation retain the rule.
        pair = puzzle['test'][0]
        _require(np.array_equal(solve(pair['input'].T), pair['output'].T), 'transpose equivariance')
        _require(np.array_equal(solve(pair['input'][::-1]), pair['output'][::-1]), 'reflection equivariance')
        # Palette permutations must fix the explicitly reserved route color.
        permutation = np.arange(10)
        others = np.array([c for c in range(10) if c != ROUTE])
        permutation[others] = np.roll(others, 3)
        _require(np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']]),
                 'geometric role inference')
    _require(len(shapes) >= 30 and len(layouts) >= 60 and len(palettes) >= 30, 'structural diversity')
    _require(len(counts) >= 4 and len(terminals) >= 20 and len(detours) >= 3 and len(blob_counts) >= 8,
             'family-specific diversity')
    _require(nonrect > .6 * total, 'distorted contour coverage')
    _require(_signature(generate_puzzle(19)) == _signature(generate_puzzle(19)), 'determinism')
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(718+a*10+b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    _ablation_tests()
    p = generate_puzzle(0, 2, 1)
    damaged = {s: [dict(input=x['input'], output=x['output'].copy()) for x in p[s]] for s in p}
    out = damaged['test'][0]['output']
    out[tuple(np.argwhere(out == ROUTE)[0])] = _canonicalize(p['test'][0]['input'])[1]['palette'][0]
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError('deleted route cell accepted')
