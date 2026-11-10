"""Contour-phase filing: a cyclic geometric field becomes a compact hierarchy.

Only standard-library modules and NumPy are used. No parent code is imported.
The input-only decoder and the constructive sampler have separate depth routines.
"""
from collections import deque
import json

import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected an integer ARC grid of size 1..30")
    return a.astype(np.int8, copy=True)


def _components(mask):
    seen = np.zeros(mask.shape, bool)
    result = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        cells, queue = [], [(r, c)]
        seen[r, c] = True
        for y, x in queue:
            cells.append((y, x))
            for dy, dx in D4:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    queue.append((yy, xx))
        out = np.zeros(mask.shape, bool)
        rr, cc = zip(*cells)
        out[rr, cc] = True
        result.append(out)
    return result


def _hole(mask):
    """Enclosed space of this component alone, ignoring every other mark."""
    padded = np.pad(mask, 1)
    outside = np.zeros(padded.shape, bool)
    outside[0, 0] = True
    queue = [(0, 0)]
    h, w = padded.shape
    for y, x in queue:
        for dy, dx in D4:
            yy, xx = y + dy, x + dx
            if (0 <= yy < h and 0 <= xx < w and not padded[yy, xx]
                    and not outside[yy, xx]):
                outside[yy, xx] = True
                queue.append((yy, xx))
    return (~padded & ~outside)[1:-1, 1:-1]


def _depth(mask):
    """Distance from the 8-neighbor boundary, boundary cells numbered zero."""
    h, w = mask.shape
    dist = np.full(mask.shape, -1, np.int16)
    q = deque()
    for y, x in np.argwhere(mask):
        y, x = int(y), int(x)
        if any(not (0 <= y+dy < h and 0 <= x+dx < w and mask[y+dy, x+dx])
               for dy, dx in D8):
            dist[y, x] = 0
            q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in D8:
            yy, xx = y+dy, x+dx
            if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and dist[yy, xx] < 0:
                dist[yy, xx] = dist[y, x] + 1
                q.append((yy, xx))
    return dist


def _erosion(mask, diagonal=True):
    """Independent vectorized stripping routine used in construction/checks."""
    current = mask.copy()
    result = np.full(mask.shape, -1, np.int16)
    h, w = mask.shape
    level = 0
    while current.any():
        pad = np.pad(current, 1)
        interior = current.copy()
        for dy, dx in D8 if diagonal else D4:
            interior &= pad[1+dy:1+dy+h, 1+dx:1+dx+w]
        result[current & ~interior] = level
        current = interior
        level += 1
    return result


def _dilate(mask):
    h, w = mask.shape
    pad = np.pad(mask, 1)
    out = mask.copy()
    for dy, dx in D8:
        out |= pad[1+dy:1+dy+h, 1+dx:1+dx+w]
    return out


def _paths(hole, depth, length):
    h, w = hole.shape
    def inside(y, x):
        return 0 <= y < h and 0 <= x < w and hole[y, x]
    result = []
    for y, x in np.argwhere(depth == 0):
        y, x = int(y), int(x)
        for dy, dx in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            if inside(y-dy, x) or inside(y, x-dx):
                continue
            if not (inside(y+dy, x) and inside(y, x+dx)):
                continue
            cells = [(y+k*dy, x+k*dx) for k in range(length)]
            if all(inside(yy, xx) and depth[yy, xx] == k
                   for k, (yy, xx) in enumerate(cells)):
                result.append((cells, (dy, dx)))
    return result


def _objects(mask):
    """Recover actual contour/blob topology from a one-color drawing."""
    records = []
    labels = np.zeros(mask.shape, int)
    for i, component in enumerate(_components(mask)):
        coords = np.argwhere(component)
        hole = _hole(component)
        if hole.any() and len(_components(hole)) != 1:
            raise ValueError("a contour must enclose exactly one region")
        lo, hi = coords.min(0), coords.max(0)
        if not hole.any() and component.sum() < .35*np.prod(hi-lo+1):
            raise ValueError("terminal blobs must be compact")
        records.append(dict(mask=component, hole=hole, lo=lo, hi=hi,
                            kind='C' if hole.any() else 'B', parent=None,
                            children=[], period=(), phase=0, depth_value=None))
        labels[component] = i+1
    for dy, dx in ((1, 0), (0, 1), (1, 1), (1, -1)):
        a = labels[max(0, dy):labels.shape[0]+min(0, dy),
                   max(0, dx):labels.shape[1]+min(0, dx)]
        b = labels[max(0, -dy):labels.shape[0]-max(0, dy),
                   max(0, -dx):labels.shape[1]-max(0, dx)]
        if np.any((a > 0) & (b > 0) & (a != b)):
            raise ValueError("distinct objects may not touch, including diagonally")
    roots = []
    for i, obj in enumerate(records):
        containing = []
        for j, other in enumerate(records):
            if i == j or other['kind'] != 'C':
                continue
            inside = other['hole'][obj['mask']]
            if inside.any() and not inside.all():
                raise ValueError("objects may not cross contours")
            if inside.all():
                containing.append(j)
        if containing:
            parent = min(containing, key=lambda j: int(records[j]['hole'].sum()))
            obj['parent'] = parent
            records[parent]['children'].append(i)
        else:
            roots.append(i)
    if not roots or any(records[i]['kind'] != 'C' for i in roots):
        raise ValueError("every exterior object must be a contour")
    for obj in records:
        if (obj['kind'] == 'C') != bool(obj['children']):
            raise ValueError("contours must contain objects; blobs cannot contain objects")
    for ids in [roots]+[o['children'] for o in records]:
        if len({tuple(records[i]['lo']) for i in ids}) != len(ids):
            raise ValueError("siblings must have distinct bounding-box top-left coordinates")
    return records, roots


def _decode(input_grid):
    grid = _grid(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    ranked = sorted(range(10), key=lambda c: -int(counts[c]))
    bg, line = ranked[:2]
    if not counts[bg] > counts[line] > counts[ranked[2]] > 0:
        raise ValueError("roles require background > drawing > every individual key color")
    records, roots = _objects(grid == line)
    cues = {i: [] for i, obj in enumerate(records) if obj['kind'] == 'C'}
    for y, x in np.argwhere((grid != bg) & (grid != line)):
        candidates = [i for i in cues if records[i]['hole'][y, x]]
        if not candidates:
            raise ValueError("all palette cues must be inside a contour")
        owner = min(candidates, key=lambda i: int(records[i]['hole'].sum()))
        cues[owner].append((int(y), int(x)))
    for i, positions in cues.items():
        obj = records[i]
        depth = _depth(obj['hole'])
        if not positions:
            raise ValueError("each contour requires one local key")
        length = max(int(depth[y, x]) for y, x in positions)+1
        if not 1 <= length <= 4 or len(positions) != length:
            raise ValueError("keys have 1..4 visible entries at consecutive depths")
        readings = []
        for cells, direction in _paths(obj['hole'], depth, length):
            if set(cells) == set(positions):
                period = tuple(int(grid[y, x]) for y, x in cells)
                readings.append((period, direction))
        if not readings or len({p for p, _ in readings}) != 1:
            raise ValueError("key is not an unambiguous inward corner diagonal")
        period = readings[0][0]
        if len(set(period)) != length or bg in period or line in period:
            raise ValueError("local key entries must be distinct non-role colors")
        obj.update(period=period, depth=depth, directions=tuple(d for _, d in readings))
        for child in obj['children']:
            d = int(depth[records[child]['mask']].min())
            if d < 1:
                raise ValueError("every child needs a background moat from its parent")
            records[child]['depth_value'] = d
            records[child]['phase'] = d % length
    return dict(grid=grid, background=int(bg), drawing=int(line), objects=records, roots=roots)


def _pack(parts, bg, vertical=False):
    """Top/left alignment, with one background cell between consecutive tiles."""
    axis = 0 if vertical else 1
    shape = [max(p.shape[0] for p in parts), max(p.shape[1] for p in parts)]
    shape[axis] = sum(p.shape[axis] for p in parts)+len(parts)-1
    out = np.full(shape, bg, np.int8)
    start = 0
    for part in parts:
        h, w = part.shape
        y, x = (start, 0) if vertical else (0, start)
        out[y:y+h, x:x+w] = part
        start += part.shape[axis]+1
    return out


def _render(info, *, use_phases=True, collapse=True):
    objects, bg = info['objects'], info['background']
    def order(i):
        return int(objects[i]['lo'][1]), int(objects[i]['lo'][0])
    def tile(i, border):
        obj = objects[i]
        if obj['kind'] == 'B':
            shape = (1, 1) if collapse else tuple((obj['hi']-obj['lo']+1).tolist())
            return np.full(shape, border, np.int8)
        shelves = []
        for phase in range(len(obj['period'])) if use_phases else (0,):
            ids = [j for j in obj['children'] if not use_phases or objects[j]['phase'] == phase]
            if not ids:
                continue
            parts = [tile(j, obj['period'][objects[j]['phase']] if use_phases else info['drawing'])
                     for j in sorted(ids, key=order)]
            shelves.append(_pack(parts, bg))
        content = _pack(shelves, bg, vertical=True)
        h, w = content.shape
        out = np.full((h+4, w+4), bg, np.int8)
        out[0, :] = out[-1, :] = border
        out[:, 0] = out[:, -1] = border
        out[2:2+h, 2:2+w] = content
        return out
    return _pack([tile(i, info['drawing']) for i in sorted(info['roots'], key=order)], bg)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive topology, keys, geometric phases, and canonical shelves from input."""
    return _render(_decode(input_grid))


def _outline(rng, h, w, rich=False):
    """A rectangle with independently displaced side intervals: actual closed steps."""
    mask = np.zeros((h, w), bool)
    mask[0, :] = mask[-1, :] = True
    mask[:, 0] = mask[:, -1] = True
    sides = list(rng.permutation(4))[:int(rng.integers(1 if rich else 0, 5))]
    for side in sides:
        work = mask.T if side >= 2 else mask
        flipped = side % 2 == 1
        if flipped:
            work = work[::-1]
        width = work.shape[1]
        if width < 12:
            continue
        a = int(rng.integers(4, width-6))
        b = int(rng.integers(a+3, width-2))
        shift = int(rng.integers(1, 4))
        work[0, a+1:b] = False
        work[:shift+1, a] = work[:shift+1, b] = True
        work[shift, a:b+1] = True
    if len(_components(mask)) != 1 or len(_components(_hole(mask))) != 1:
        raise ValueError("invalid outline draw")
    return mask


def _blob(rng, small=False):
    h, w = (int(rng.integers(1, 3 if small else 4)) for _ in range(2))
    out = np.ones((h, w), bool)
    if h >= 2 and w >= 2 and rng.random() < .65:
        out[int(rng.choice([0, h-1])), int(rng.choice([0, w-1]))] = False
    return out


def _embed_mask(shape, part, y, x):
    out = np.zeros(shape, bool)
    h, w = part.shape
    out[y:y+h, x:x+w] = part
    return out


def _container(rng, h, w, period_length, children=(), blob_count=3,
               witness=False, small=False):
    """Place child loops and terminal blobs into an independently deformed hole."""
    wall = _outline(rng, h, w, rich=witness)
    hole = _hole(wall)
    depth = _erosion(hole)
    keys = _paths(hole, depth, period_length)
    if not keys:
        raise ValueError("outline lacks a corner key")
    cells, _ = keys[int(rng.integers(len(keys)))]
    reserved = np.zeros(wall.shape, bool)
    yy, xx = zip(*cells)
    reserved[yy, xx] = True
    occupied = wall.copy()
    # Leave cues visually separate from child objects, including diagonals.
    blocked = _dilate(reserved)
    child_interiors = np.zeros(wall.shape, bool)
    for part in children:
        ph, pw = part.shape
        locations = [(y, x) for y in range(2, h-ph-1) for x in range(2, w-pw-1)]
        rng.shuffle(locations)
        for y, x in locations:
            placed = _embed_mask(wall.shape, part, y, x)
            # The entire child tile, including its interior, stays inside the parent.
            box = _embed_mask(wall.shape, np.ones(part.shape, bool), y, x)
            if np.all(hole[box]) and not np.any(placed & (_dilate(occupied) | blocked | child_interiors)):
                occupied |= placed
                child_interiors |= _hole(placed)
                break
        else:
            raise ValueError("cannot fit child contour")
    # The first flat witness targets three phases and a repeated phase at a
    # genuinely different depth. Other placements are unconstrained by phase.
    targets = [1, 2, 3, 4] if witness else [None]*blob_count
    if witness:
        targets += [None]*(blob_count-4)
    for target in targets:
        placed_ok = False
        for _ in range(18):
            part = _blob(rng, small)
            ph, pw = part.shape
            locations = [(y, x) for y in range(2, h-ph-1) for x in range(2, w-pw-1)]
            rng.shuffle(locations)
            free = hole & ~(_dilate(occupied) | blocked | child_interiors)
            for y, x in locations:
                placed = _embed_mask(wall.shape, part, y, x)
                if np.all(free[placed]) and (target is None or int(depth[placed].min()) == target):
                    occupied |= placed
                    placed_ok = True
                    break
            if placed_ok:
                break
        if not placed_ok:
            raise ValueError("cannot fit blob at requested depth")
    return occupied


def _small_loop(rng, length=2, rich=False):
    h, w = (int(rng.integers(9 if length <= 3 else 12, 15)) for _ in range(2))
    return _container(rng, h, w, length, blob_count=int(rng.integers(1, 4)), small=True)


def _drawing(rng, profile):
    if profile == 'witness':
        h, w = (int(rng.integers(20, 29)) for _ in range(2))
        return _container(rng, h, w, 3, blob_count=int(rng.integers(4, 8)), witness=True), 3
    if profile in ('nested', 'branch', 'deep', 'long_nested'):
        if profile == 'deep':
            inner = _container(rng, 9, 9, 2, blob_count=int(rng.integers(1, 3)), small=True)
            middle = _container(rng, int(rng.integers(16, 20)), int(rng.integers(16, 20)),
                                2, [inner], blob_count=int(rng.integers(0, 2)), small=True)
            children = [middle]
        else:
            children = [_small_loop(rng) for _ in range(2 if profile == 'branch' else 1)]
        h, w = (int(rng.integers(24 if profile in ('branch', 'deep') else 21, 31))
                for _ in range(2))
        length = 4 if profile == 'long_nested' else 2
        core = _container(rng, h, w, length, children, blob_count=int(rng.integers(1, 4)), small=True)
        return core, length
    if profile == 'roots':
        parts = [_small_loop(rng) for _ in range(int(rng.integers(2, 4)))]
        axis = int(rng.integers(2))
        cross = 1-axis
        gaps = [int(rng.integers(1, 4)) for _ in parts[:-1]]
        shape = [0, 0]
        shape[axis] = sum(p.shape[axis] for p in parts)+sum(gaps)
        shape[cross] = max(p.shape[cross] for p in parts)+int(rng.integers(0, 3))
        if max(shape) > 30:
            raise ValueError("multiple roots exceed canvas")
        out = np.zeros(shape, bool)
        start = 0
        for i, p in enumerate(parts):
            offset = int(rng.integers(shape[cross]-p.shape[cross]+1))
            y, x = (start, offset) if axis == 0 else (offset, start)
            out[y:y+p.shape[0], x:x+p.shape[1]] |= p
            start += p.shape[axis]+(gaps[i] if i < len(gaps) else 0)
        return out, 2
    length = 4 if profile == 'long' else int(rng.integers(1, 4))
    h, w = (int(rng.integers(13 if length < 4 else 18, 29)) for _ in range(2))
    return _container(rng, h, w, length, blob_count=int(rng.integers(1, 8))), length


def _add_keys(rng, mask, palette, root_length, max_length):
    bg, line = palette[:2]
    records, roots = _objects(mask)
    grid = np.where(mask, line, bg).astype(np.int8)
    for i, obj in enumerate(records):
        if obj['kind'] != 'C':
            continue
        depth = _erosion(obj['hole'])
        direct = obj['hole'].copy()
        child_mask = np.zeros(mask.shape, bool)
        for j in obj['children']:
            child_mask |= records[j]['mask']
            direct &= ~records[j]['hole']
        free = direct & ~_dilate(child_mask) & ~mask
        lengths = [root_length] if i in roots else list(range(1, max_length+1))
        rng.shuffle(lengths)
        available = []
        for length in lengths:
            options = [(cells, direction) for cells, direction in _paths(obj['hole'], depth, length)
                       if all(free[y, x] for y, x in cells)]
            if options:
                available = options
                break
        if not available:
            raise ValueError("local contour has no free key")
        cells, direction = available[int(rng.integers(len(available)))]
        colors = tuple(int(c) for c in rng.choice(palette[2:], size=length, replace=False))
        for (y, x), color in zip(cells, colors):
            grid[y, x] = color
        obj.update(period=colors, depth=depth, directions=(direction,))
        for j in obj['children']:
            d = int(depth[records[j]['mask']].min())
            records[j]['depth_value'] = d
            records[j]['phase'] = d % length
    info = dict(grid=grid, background=bg, drawing=line, objects=records, roots=roots)
    return grid, _render(info)


def _feature(info):
    objs = info['objects']
    depths = []
    repeated = multi_phase = nonrect = distorted_phase = l1_phase = reset_difference = False
    for obj in objs:
        if obj['kind'] != 'C':
            continue
        nesting = 1
        parent = obj['parent']
        while parent is not None:
            nesting += 1
            parent = objs[parent]['parent']
        depths.append(nesting)
        phases = [objs[j]['phase'] for j in obj['children']]
        multi_phase |= len(set(phases)) >= 2
        repeated |= any(objs[a]['phase'] == objs[b]['phase']
                        and objs[a]['depth_value'] != objs[b]['depth_value']
                        for a in obj['children'] for b in obj['children'] if a != b)
        y, x = obj['lo']; yy, xx = obj['hi']
        local = obj['mask'][y:yy+1, x:xx+1]
        nonrect |= not (local[0].all() and local[-1].all() and local[:, 0].all() and local[:, -1].all())
        l1 = _erosion(obj['hole'], diagonal=False)
        if obj['parent'] is not None:
            parent_period = objs[obj['parent']]['period']
            reset_difference |= any(obj['period'][objs[j]['phase']] !=
                                    parent_period[objs[j]['depth_value'] % len(parent_period)]
                                    for j in obj['children'])
        for j in obj['children']:
            r, c = np.nonzero(objs[j]['mask'])
            box_d = int(np.minimum.reduce((r-y-1, yy-r-1, c-x-1, xx-c-1)).min())
            p = len(obj['period'])
            distorted_phase |= box_d % p != objs[j]['phase']
            l1_phase |= int(l1[objs[j]['mask']].min()) % p != objs[j]['phase']
    return dict(depth=max(depths), contours=len(depths),
                blobs=sum(o['kind'] == 'B' for o in objs),
                periods=tuple(len(o['period']) for o in objs if o['kind'] == 'C'),
                repeat=bool(repeated), multi_phase=bool(multi_phase),
                nonrect=bool(nonrect), bbox_difference=bool(distorted_phase),
                l1_difference=bool(l1_phase), reset_difference=bool(reset_difference))


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _make_pair(rng, palette, profile):
    for _ in range(160):
        try:
            mask, root_length = _drawing(rng, profile)
            h, w = mask.shape
            # Independent extra canvas margins may be zero, including edge contact.
            height = int(rng.integers(h, min(30, h+5)+1))
            width = int(rng.integers(w, min(30, w+5)+1))
            if rng.random() < .1:
                height = width = 30
            y, x = int(rng.integers(height-h+1)), int(rng.integers(width-w+1))
            mask = _embed_mask((height, width), mask, y, x)
            inp, out = _add_keys(rng, mask, palette, root_length,
                                 4 if profile in ('long', 'long_nested', 'deep') else 3)
            if max(out.shape) > 30:
                continue
            info = _decode(inp)
            assert np.array_equal(_render(info), out)
            f = _feature(info)
            if profile == 'witness' and not (f['repeat'] and f['multi_phase']
                                           and f['bbox_difference'] and f['l1_difference']):
                continue
            if profile == 'witness':
                areas = {int(o['mask'].sum()) for o in info['objects'] if o['kind'] == 'B'}
                if len(areas) < 2:
                    continue
            if profile == 'nested' and not (f['depth'] == 2 and f['multi_phase'] and f['reset_difference']):
                continue
            return dict(input=inp, output=out)
        except (ValueError, AssertionError):
            continue
    raise RuntimeError("bounded construction exhausted for "+profile)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(name+" is outside its legal integer range")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    stable = bool(rng.integers(2))
    base = [int(c) for c in rng.permutation(10)]
    puzzle = dict(train=[], test=[])
    seen = set()
    profiles = ['witness', 'nested'] + [str(rng.choice(['flat', 'nested', 'roots', 'branch']))
                                      for _ in range(num_train-2)]
    # No length-4 key or third nesting level in training. Tests recombine the
    # same local operations with at least one of those structural extensions.
    tests = [str(rng.choice(['long', 'long_nested', 'deep'])) for _ in range(num_test)]
    for split, schedule in (('train', profiles), ('test', tests)):
        for profile in schedule:
            palette = base if stable else [int(c) for c in rng.permutation(10)]
            for _ in range(12):
                pair = _make_pair(rng, palette, profile)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("bounded uniqueness search exhausted")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _topology(objects, roots):
    def tree(i):
        obj = objects[i]
        return ('B',) if obj['kind'] == 'B' else ('C', tuple(sorted(tree(j) for j in obj['children'])))
    return tuple(sorted(tree(i) for i in roots))


def validate_generated_puzzle(puzzle):
    """Check returned pixels, not private sampler metadata."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, train_features, test_features = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _decode(inp)
            assert np.array_equal(solve(inp), out), "incorrect shelves, colors, or crop"
            # Check the colored output as geometry irrespective of color names.
            objs, roots = _objects(out != info['background'])
            assert _topology(info['objects'], info['roots']) == _topology(objs, roots)
            for obj in objs:
                if obj['kind'] == 'B':
                    assert obj['mask'].sum() == 1, "every terminal becomes exactly one cell"
                else:
                    y, x = obj['lo']; yy, xx = obj['hi']
                    assert obj['mask'].sum() == 2*(yy-y+xx-x)
                    assert len(np.unique(out[obj['mask']])) == 1
                    children = [objs[j] for j in obj['children']]
                    assert np.array_equal(np.min([c['lo'] for c in children], 0), obj['lo']+2)
                    assert np.array_equal(np.max([c['hi'] for c in children], 0), obj['hi']-2)
            for obj in info['objects']:
                if obj['kind'] == 'C':
                    assert np.array_equal(obj['depth'], _erosion(obj['hole']))
            feature = _feature(info)
            (train_features if split == 'train' else test_features).append(feature)
            assert _key(inp) != _key(out)
    assert any(f['repeat'] and f['bbox_difference'] and f['l1_difference'] and f['multi_phase']
               for f in train_features), "training must expose true cyclic contour phases"
    assert any(f['depth'] >= 2 and f['multi_phase'] and f['reset_difference'] for f in train_features)
    assert {2, 3} <= {p for f in train_features for p in f['periods']}
    assert all(f['depth'] <= 2 and max(f['periods']) <= 3 for f in train_features)
    assert all(f['depth'] == 3 or 4 in f['periods'] for f in test_features)


def self_test():
    """Bounded 42-seed diversity, metamorphic, shortcut, and corruption checks."""
    arr = lambda rows: np.array([[int(c) for c in row] for row in rows.split()], np.int8)
    # Hand-specified answers, independent of the packing routine's construction.
    flat = np.zeros((19, 19), np.int8)
    flat[0, :] = flat[-1, :] = flat[:, 0] = flat[:, -1] = 4
    flat[1, 1], flat[2, 2], flat[3, 3] = 8, 2, 7
    flat[4:6, 10:12] = 4
    flat[3, 14] = flat[6, 6] = 4
    assert np.array_equal(solve(flat), arr('''4444444
4000004
4080004
4000004
4070704
4000004
4444444'''))
    nested = np.zeros((21, 21), np.int8)
    nested[0, :] = nested[-1, :] = nested[:, 0] = nested[:, -1] = 1
    nested[1, 1], nested[2, 2], nested[3, 3] = 2, 3, 4
    nested[7, 8:17] = nested[15, 8:17] = 1
    nested[7:16, 8] = nested[7:16, 16] = 1
    nested[8, 9], nested[9, 10] = 5, 6
    nested[11, 12] = nested[3, 12] = 1
    assert np.array_equal(solve(nested), arr('''111111111
100000001
102222201
102000201
102060201
102000201
102222201
100000001
104000001
100000001
111111111'''))
    puzzles, shapes, topologies, periods, blob_areas, directions = [], set(), set(), set(), set(), set()
    contour_counts, nesting, normalized_outputs, role_pairs = set(), set(), set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        puzzles.append(p)
        validate_generated_puzzle(p)
        for split in ('train', 'test'):
            for pair in p[split]:
                info = _decode(pair['input'])
                feature = _feature(info)
                shapes.add(pair['input'].shape)
                contour_counts.add(feature['contours'])
                nesting.add(feature['depth'])
                periods.update(feature['periods'])
                topologies.add(_topology(info['objects'], info['roots']))
                role_pairs.add((info['background'], info['drawing']))
                role_map = {}
                normalized_outputs.add(tuple(tuple(role_map.setdefault(int(c), len(role_map)) for c in row)
                                             for row in pair['output']))
                for obj in info['objects']:
                    if obj['kind'] == 'B':
                        blob_areas.add(int(obj['mask'].sum()))
                    else:
                        directions.update(obj['directions'])
        # Recoloring and translating the whole scene retain its answer.
        pair = p['test'][0]
        permutation = np.random.default_rng(1000+seed).permutation(10)
        assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
        bg = _decode(pair['input'])['background']
        if max(pair['input'].shape) <= 28:
            moved = np.pad(pair['input'], ((1, 1), (1, 1)), constant_values=bg)
            assert np.array_equal(solve(moved), pair['output'])
    signature = lambda p: tuple(_key(pair[k]) for s in ('train', 'test') for pair in p[s] for k in ('input', 'output'))
    assert len({signature(p) for p in puzzles}) == 42
    assert len(shapes) >= 45 and (30, 30) in shapes
    assert len(topologies) >= 15 and len(normalized_outputs) >= 70
    assert periods == {1, 2, 3, 4} and nesting == {1, 2, 3}
    assert len(contour_counts) >= 3 and len(blob_areas) >= 6 and len(role_pairs) >= 30
    assert directions == {(1, 1), (1, -1), (-1, 1), (-1, -1)}
    assert signature(generate_puzzle(0)) == signature(puzzles[0])
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(1200+a*10+b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Omitting A changes shelf grouping as well as color. Omitting B's blob
    # abstraction changes size; flattening containment changes the tree.
    for p in puzzles:
        assert any(_render(_decode(pair['input']), use_phases=False).shape != pair['output'].shape
                   for pair in p['train'])
        assert any(_key(_render(_decode(pair['input']), collapse=False)) != _key(pair['output'])
                   for pair in p['train'])
    # Same contour, objects, and topology; changing only key length changes
    # filing geometry, rather than merely changing the colors of a fixed answer.
    source = next(pair for pair in puzzles[0]['train']
                  if _feature(_decode(pair['input']))['contours'] == 1
                  and _feature(_decode(pair['input']))['repeat']
                  and _feature(_decode(pair['input']))['periods'] == (3,))
    changed = source['input'].copy()
    info = _decode(changed)
    root = info['objects'][info['roots'][0]]
    ys, xs = np.nonzero((changed != info['background']) & (changed != info['drawing']))
    deepest = max(zip(ys, xs), key=lambda cell: int(root['depth'][cell]))
    changed[deepest] = info['background']
    assert np.array_equal(changed == info['drawing'], source['input'] == info['drawing'])
    assert solve(changed).shape != source['output'].shape
    # Same bounding box/center, different shallowest pixel at a concave corner.
    samples = []
    for remove_corner in (False, True):
        a = np.zeros((21, 21), np.int8)
        a[0, :] = a[-1, :] = a[:, 0] = a[:, -1] = 1
        a[0, 13:16] = 0
        a[:5, 12] = a[:5, 16] = a[4, 12:17] = 1
        a[1, 1], a[2, 2], a[3, 3] = 2, 3, 4
        a[6:9, 8:11] = 1
        if remove_corner:
            a[6, 10] = 0
        decoded = _decode(a)
        blob = next(o for o in decoded['objects'] if o['kind'] == 'B')
        samples.append((tuple(blob['lo']), tuple(blob['hi']), blob['phase'], _key(solve(a))))
        expected = np.ones((5, 5), np.int8)
        expected[1:4, 1:4] = 0
        expected[2, 2] = 4 if remove_corner else 3
        assert np.array_equal(solve(a), expected)
    assert samples[0][:2] == samples[1][:2]
    assert samples[0][2] != samples[1][2] and samples[0][3] != samples[1][3]
    for mode in ('point', 'crop', 'key'):
        p = {s: [{k: np.array(v, copy=True) for k, v in pair.items()} for pair in puzzles[0][s]]
             for s in ('train', 'test')}
        pair = p['test'][0]
        info = _decode(pair['input'])
        bg = info['background']
        if mode == 'point':
            objs, _ = _objects(pair['output'] != bg)
            obj = next(o for o in objs if o['kind'] == 'B')
            pair['output'][obj['mask']] = bg
        elif mode == 'crop':
            pair['output'] = np.pad(pair['output'], 1, constant_values=bg)
        else:
            y, x = np.argwhere((pair['input'] != bg) & (pair['input'] != info['drawing']))[0]
            pair['input'][y, x] = bg
        try:
            validate_generated_puzzle(p)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("corrupted "+mode+" accepted")


if __name__ == '__main__':
    self_test()
    print(json.dumps({'self_test': 'passed', 'seeds': 42}, separators=(',', ':')))
