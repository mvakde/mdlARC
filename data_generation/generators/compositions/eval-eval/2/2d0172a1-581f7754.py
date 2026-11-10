"""Nested docking rooms: topology supplies the carriers and scope of each guide.

No parent module, fixture, seed, or construction record is used by solve().
"""
from collections import Counter
import numpy as np

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), 'grid dimensions must be 1..30')
    _require(np.issubdtype(a.dtype, np.integer), 'grid cells must be integers')
    _require(bool(np.all((a >= 0) & (a <= 9))), 'grid colors must be 0..9')
    return a.astype(np.int8, copy=False)


def _components(mask):
    seen = np.zeros(mask.shape, bool)
    result = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        cells, stack = [], [(r, c)]
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in DIRS:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(np.array(cells, dtype=int))
    return result


def _holes(mask):
    padded = np.pad(mask, 1)
    empty = ~padded
    reached = np.zeros(padded.shape, bool)
    reached[0, 0] = True
    stack = [(0, 0)]
    h, w = padded.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS:
            y, x = r + dr, c + dc
            if 0 <= y < h and 0 <= x < w and empty[y, x] and not reached[y, x]:
                reached[y, x] = True
                stack.append((y, x))
    return (empty & ~reached)[1:-1, 1:-1]


def _corner_position(shape, corner):
    return (shape[0] - 1 if corner[0] else 0,
            shape[1] - 1 if corner[1] else 0)


def _decode(value, background=None):
    grid = _grid(value)
    if background is None:
        colors, counts = np.unique(grid, return_counts=True)
        _require(int(counts.max()) * 2 > grid.size, 'input needs a strict majority background')
        background = int(colors[np.argmax(counts)])
    objects = []
    for cells in _components(grid != background):
        lo, hi = cells.min(0), cells.max(0)
        mask = np.zeros(grid.shape, bool)
        mask[cells[:, 0], cells[:, 1]] = True
        hole = _holes(mask)
        room = bool(hole.any())
        if room:
            _require(len(_components(hole)) == 1, 'a room must have one enclosed region')
        else:
            _require(len(cells) >= .4 * int(np.prod(hi - lo + 1)), 'leaf must be a compact filled blob')
        objects.append(dict(cells=cells, lo=lo, hi=hi, mask=mask, hole=hole,
                            room=room, children=[], parent=None))
    _require(objects, 'empty drawing')
    for i, obj in enumerate(objects):
        containers = []
        for j, parent in enumerate(objects):
            if i == j or not parent['room']:
                continue
            flags = parent['hole'][obj['cells'][:, 0], obj['cells'][:, 1]]
            _require(flags.all() or not flags.any(), 'components cross a contour')
            if flags.all():
                containers.append(j)
        if containers:
            obj['parent'] = min(containers, key=lambda j: int(objects[j]['hole'].sum()))
            objects[obj['parent']]['children'].append(i)
    roots = [i for i, obj in enumerate(objects) if obj['parent'] is None]
    _require(len(roots) == 1 and objects[roots[0]]['room'], 'one enclosing root room is required')
    root = roots[0]
    root_counts = Counter(int(grid[tuple(p)]) for p in objects[root]['cells'])
    wall = root_counts.most_common(1)[0][0]
    # Paired colors belong to the wall of this room only. Deeper occurrences
    # cannot steal a parent's handle, even when their literal colors coincide.
    for i, obj in enumerate(objects):
        if not obj['room']:
            _require(not obj['children'], 'a filled leaf cannot contain objects')
            continue
        _require(obj['children'], 'rooms cannot be empty')
        counts = Counter(int(grid[tuple(p)]) for p in obj['cells'])
        _require(counts[wall] >= 8, 'all rooms share the visible wall role')
        special = {c: n for c, n in counts.items() if c != wall}
        _require(all(n in (1, 2) for n in special.values()), 'invalid wall marker multiplicity')
        singles = [c for c, n in special.items() if n == 1]
        _require(len(singles) == (0 if i == root else 1), 'invalid incoming room badge')
        if singles:
            obj['key'] = singles[0]
            reg = next(tuple(map(int, p)) for p in obj['cells'] if grid[tuple(p)] == singles[0])
            corner = tuple(int(reg[k] == obj['hi'][k]) for k in (0, 1))
            _require(reg == tuple(obj['lo'] + np.array(_corner_position(obj['hi'] - obj['lo'] + 1, corner))),
                     'incoming badge must be a bounding-box corner')
            obj['reg'], obj['corner'] = reg, corner
        guides = []
        for color, n in special.items():
            if n != 2:
                continue
            points = [tuple(map(int, p)) for p in obj['cells'] if grid[tuple(p)] == color]
            a, b = sorted(points)
            horizontal = a[0] == b[0]
            vertical = a[1] == b[1]
            _require(horizontal != vertical, 'ports must define a row or column')
            axis = 0 if horizontal else 1
            guide = a[axis]
            _require(obj['lo'][axis] < guide < obj['hi'][axis], 'ports cannot be at corners')
            scan = [tuple(map(int, p)) for p in obj['cells'] if int(p[axis]) == guide]
            _require({a, b} == {min(scan, key=lambda p: p[1-axis]), max(scan, key=lambda p: p[1-axis])},
                     'ports must occupy opposite extremes of the same wall scan')
            guides.append((guide, color, axis))
        _require(1 <= len(guides) <= 3, 'a room needs one to three port pairs')
        _require(len({g[2] for g in guides}) == 1, 'guides in one room must be parallel')
        _require(len({g[0] for g in guides}) == len(guides), 'coincident local guides')
        obj['axis'] = guides[0][2]
        obj['guides'] = sorted(guides)
    for obj in objects:
        if not obj['room']:
            parent = objects[obj['parent']]
            keys = {g[1] for g in parent['guides']}
            counts = Counter(int(grid[tuple(p)]) for p in obj['cells'])
            candidates = [c for c in keys if counts[c] == 1]
            _require(len(candidates) == 1, 'a blob needs one unique once-occurring local registration color')
            obj['key'] = candidates[0]
            obj['reg'] = next(tuple(map(int, p)) for p in obj['cells'] if grid[tuple(p)] == obj['key'])
        if obj['parent'] is not None:
            keys = {g[1] for g in objects[obj['parent']]['guides']}
            _require(obj['key'] in keys, 'child registration lacks a port pair in its immediate room')
    for obj in objects:
        if obj['room']:
            _require({objects[j]['key'] for j in obj['children']} == {g[1] for g in obj['guides']},
                     'every guide must have a child')
    return dict(grid=grid, background=background, wall=wall, objects=objects, root=root)


def _dock_horizontal(entries, ordered_keys, background, wall, centered=False):
    """Align handles, pack each lane, then pack whole lane bands."""
    lanes, cursor = [], 0
    for key in ordered_keys:
        items = sorted((e for e in entries if e['key'] == key),
                       key=lambda e: (e['source'][1], e['source'][0]))
        handles = [((e['tile'].shape[0] - 1) // 2 if centered else e['handle'][0]) for e in items]
        above = max(handles)
        below = max(e['tile'].shape[0] - 1 - y for e, y in zip(items, handles))
        lane_h = above + 1 + below
        x, placements = 0, []
        for e, handle_y in zip(items, handles):
            placements.append((cursor + above - handle_y, x, e['tile']))
            x += e['tile'].shape[1] + 1
        lanes.append((key, cursor + above, x - 1, placements))
        cursor += lane_h + 1
    h, w = cursor - 1 + 4, max(l[2] for l in lanes) + 4
    result = np.full((h, w), background, np.int8)
    result[0, :] = result[-1, :] = wall
    result[:, 0] = result[:, -1] = wall
    for key, reference, _, placements in lanes:
        result[reference + 2, 0] = result[reference + 2, -1] = key
        for y, x, tile in placements:
            hh, ww = tile.shape
            result[y+2:y+2+hh, x+2:x+2+ww] = tile
    return result


def _predict(info, keep_blobs=False, centered=False, ignore_ports=False, flatten=False):
    objs, bg, wall = info['objects'], info['background'], info['wall']
    def visit(i):
        obj = objs[i]
        if not obj['room'] or flatten and i != info['root']:
            if keep_blobs and not obj['room']:
                r, c = obj['lo']; rr, cc = obj['hi']
                tile = np.full((rr-r+1, cc-c+1), bg, np.int8)
                p = obj['cells'] - obj['lo']
                tile[p[:, 0], p[:, 1]] = info['grid'][obj['cells'][:, 0], obj['cells'][:, 1]]
                handle = tuple(np.array(obj['reg']) - obj['lo'])
            else:
                tile, handle = np.array([[obj['key']]], np.int8), (0, 0)
            return tile, handle
        entries = []
        axis = obj['axis']
        for j in obj['children']:
            tile, handle = visit(j)
            source = objs[j]['reg']
            if axis == 1:
                tile, handle, source = tile.T, handle[::-1], source[::-1]
            entries.append(dict(tile=tile, handle=handle, source=source, key=objs[j]['key']))
        keys = [g[1] for g in obj['guides']]
        if ignore_ports:
            for e in entries:
                e['key'] = keys[0]
            keys = keys[:1]
        tile = _dock_horizontal(entries, keys, bg, wall, centered=centered)
        if axis == 1:
            tile = tile.T.copy()
        if i == info['root']:
            return tile, None
        handle = _corner_position(tile.shape, obj['corner'])
        tile[handle] = obj['key']
        return tile, handle
    return visit(info['root'])[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _predict(_decode(input_grid))


# Construction below is model-driven. It does not call the input decoder or
# solve to label an example. Its canonical assembly uses reference-relative
# colored coordinates, whereas prediction places array tiles.
def _leaf():
    return dict(kind='blob')


def _room(axis, lanes, children, assignments=None, corner=None):
    return dict(kind='room', axis=axis, lanes=lanes, children=children,
                assignments=assignments, corner=corner)


def _witness(index, rng):
    corners = ((0, 0), (1, 1)) if index == 0 else ((0, 1), (1, 0))
    children = []
    for k in range(2):
        # One and two local lanes; same and perpendicular axes both occur.
        children.append(_room(k, 1+k, [_leaf(), _leaf()], corner=corners[k]))
    children += [_leaf(), _leaf()]
    node = _room(index, 2, children, assignments=[0, 1, 0, 1])
    node['shared_port'] = True
    return node


def _random_tree(rng, max_depth=2, max_lanes=2, depth=1, budget=3):
    n = int(rng.choice([1, 2, 3, 4, 5], p=[.12, .32, .32, .18, .06]))
    lanes = int(rng.integers(1, min(n, max_lanes) + 1))
    children = [_leaf() for _ in range(n)]
    if depth < max_depth and budget > 1:
        available = list(map(int, rng.permutation(n)))
        for k in available[:min(2, budget-1)]:
            if rng.random() < .52:
                children[k] = _random_tree(rng, max_depth, max_lanes, depth+1, budget-1)
                budget -= 1
    return _room(int(rng.integers(2)), lanes, children)


def _shift_tree(rng, mode):
    if mode == 0:
        # Guaranteed extra scope; stochastic siblings and axes on the chain.
        inner = _room(int(rng.integers(2)), int(rng.integers(1, 3)), [_leaf(), _leaf()])
        middle_children = [inner] + [_leaf() for _ in range(int(rng.integers(0, 3)))]
        middle = _room(int(rng.integers(2)), int(rng.integers(1, min(2, len(middle_children))+1)), middle_children)
        children = [middle] + [_leaf() for _ in range(int(rng.integers(1, 4)))]
        return _room(int(rng.integers(2)), int(rng.integers(1, min(2, len(children))+1)), children)
    n = int(rng.integers(3, 6))
    children = [_leaf() for _ in range(n)]
    if rng.random() < .8:
        children[int(rng.integers(n))] = _random_tree(rng, 2, 2, 2, 1)
    return _room(int(rng.integers(2)), 3, children)


def _blob(rng, background, key, local_keys, palette, compact=False):
    cap = 2 if compact else 4
    h, w = int(rng.integers(1, cap+1)), int(rng.integers(1, cap+1))
    if h*w == 1 or rng.random() < .12:
        return np.array([[key]], np.int8), (0, 0)
    mask = np.ones((h, w), bool)
    # Removing corners creates small asymmetric filled clusters without holes.
    corners = list({(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)})
    rng.shuffle(corners)
    for p in corners[:int(rng.integers(min(3, len(corners))+1))]:
        trial = mask.copy(); trial[p] = False
        if trial.sum() >= max(2, .6*h*w) and len(_components(trial)) == 1 and not _holes(trial).any():
            mask = trial
    coords = np.argwhere(mask)
    lo, hi = coords.min(0), coords.max(0)
    mask = mask[lo[0]:hi[0]+1, lo[1]:hi[1]+1]
    coords = np.argwhere(mask)
    # Repeated colors of another local guide may form the body. Only the unique
    # once-occurring local identity is the handle, as in parent B's wider family.
    body_options = [c for c in palette if c not in (background, key) and (c not in local_keys or len(coords) >= 3)]
    body = int(rng.choice(body_options))
    tile = np.full(mask.shape, background, np.int8); tile[mask] = body
    reg = tuple(map(int, coords[int(rng.integers(len(coords)))]))
    tile[reg] = key
    return tile, reg


def _pack(rng, tiles, background, compact):
    # A slicing layout is only a renderer; guide assignment is independent of it.
    # Source geometry need not look like the eventual docking lanes.
    order = list(map(int, rng.permutation(len(tiles))))
    transpose = bool(rng.integers(2))
    parts = [t.T if transpose else t for t in tiles]
    row_count = int(rng.integers(1, min(3, len(parts))+1))
    cuts = sorted(map(int, rng.choice(np.arange(1, len(parts)), row_count-1, replace=False))) if row_count > 1 else []
    groups = [order[a:b] for a, b in zip([0]+cuts, cuts+[len(order)])]
    placements, y, max_w = {}, 0, 0
    for ids in groups:
        shifts = [int(rng.integers(0, 2 if compact else 4)) for _ in ids]
        row_h = max(parts[j].shape[0] + shift for j, shift in zip(ids, shifts))
        x = int(rng.integers(0, 2 if compact else 4))
        for j, shift in zip(ids, shifts):
            placements[j] = (y + shift, x)
            x += parts[j].shape[1] + int(rng.integers(1, 2 if compact else 4))
        max_w = max(max_w, x-1)
        y += row_h + int(rng.integers(1, 2 if compact else 4))
    h = max(placements[j][0] + parts[j].shape[0] for j in placements)
    w = max(placements[j][1] + parts[j].shape[1] for j in placements)
    canvas = np.full((h, w), background, np.int8)
    for j, (yy, xx) in placements.items():
        hh, ww = parts[j].shape
        canvas[yy:yy+hh, xx:xx+ww] = parts[j]
    if transpose:
        canvas = canvas.T.copy()
        placements = {j: p[::-1] for j, p in placements.items()}
    return canvas, placements


def _outline(rng, h, w, padding, compact):
    top, bottom, left, right = padding
    mask = np.zeros((h, w), bool)
    profiles = []
    for length, low, high, cap in ((w, left, w-right, top-2), (w, left, w-right, bottom-2),
                                   (h, top, h-bottom, left-2), (h, top, h-bottom, right-2)):
        profile = np.zeros(length, int)
        if cap and rng.random() < .88:
            if rng.random() < .5:
                profile[low:high] = rng.integers(0, cap+1, size=high-low)
            else:
                a = int(rng.integers(low, high)); b = int(rng.integers(a+1, high+1))
                profile[a:b] = int(rng.integers(1, cap+1))
        profiles.append(profile)
    t, b, l, r = profiles
    for x in range(w):
        mask[t[x], x] = True; mask[h-1-b[x], x] = True
        if x:
            mask[min(t[x-1], t[x]):max(t[x-1], t[x])+1, x] = True
            ya, yb = h-1-b[x-1], h-1-b[x]
            mask[min(ya, yb):max(ya, yb)+1, x] = True
    for y in range(h):
        mask[y, l[y]] = True; mask[y, w-1-r[y]] = True
        if y:
            mask[y, min(l[y-1], l[y]):max(l[y-1], l[y])+1] = True
            xa, xb = w-1-r[y-1], w-1-r[y]
            mask[y, min(xa, xb):max(xa, xb)+1] = True
    return mask


def _model_output(records, guides, axis, bg, wall, incoming, corner):
    # Place colored coordinates relative to the reference line. This is not the
    # solver's tile compositor, and labels are determined before decoding.
    pixels, refs, band_start, extent = {}, [], 2, 0
    for g, key in sorted(guides):
        row = sorted([rec for rec in records if rec['key'] == key],
                     key=lambda rec: (rec['source'][1-axis], rec['source'][axis]))
        neg = max(rec['handle'][axis] for rec in row)
        pos = max(rec['out'].shape[axis]-1-rec['handle'][axis] for rec in row)
        reference, along = band_start + neg, 2
        refs.append((reference, key))
        for rec in row:
            tile = rec['out']; handle = rec['handle']
            for p in np.argwhere(tile != bg):
                changed = reference + int(p[axis]) - handle[axis]
                fixed = along + int(p[1-axis])
                target = (changed, fixed) if axis == 0 else (fixed, changed)
                pixels[target] = int(tile[tuple(p)])
            along += tile.shape[1-axis] + 1
        extent = max(extent, along-1)
        band_start += neg + 1 + pos + 1
    changed_len, fixed_len = band_start + 1, extent + 2
    shape = (changed_len, fixed_len) if axis == 0 else (fixed_len, changed_len)
    out = np.full(shape, bg, np.int8)
    out[0, :] = out[-1, :] = wall; out[:, 0] = out[:, -1] = wall
    for p, color in pixels.items():
        out[p] = color
    for ref, key in refs:
        if axis == 0:
            out[ref, 0] = out[ref, -1] = key
        else:
            out[0, ref] = out[-1, ref] = key
    handle = None
    if incoming is not None:
        handle = _corner_position(shape, corner); out[handle] = incoming
    return out, handle


def _draw(node, rng, palette, incoming=None, local_keys=(), compact=False, forced_port=None):
    bg, wall = palette[:2]
    if node['kind'] == 'blob':
        inp, source = _blob(rng, bg, incoming, local_keys, palette, compact)
        return inp, np.array([[incoming]], np.int8), source, (0, 0)
    axis, lane_count = node['axis'], node['lanes']
    choices = [c for c in palette[2:] if c != incoming]
    if forced_port is None:
        keys = list(map(int, rng.choice(choices, lane_count, replace=False)))
    else:
        keys = [forced_port] + list(map(int, rng.choice([c for c in choices if c != forced_port], lane_count-1, replace=False)))
    n = len(node['children'])
    assignment = node['assignments']
    if assignment is None:
        assignment = list(range(lane_count)) + list(map(int, rng.integers(lane_count, size=n-lane_count)))
        rng.shuffle(assignment)
    shared = int(rng.choice([c for c in choices if c not in keys])) if node.get('shared_port') else None
    pairs = [_draw(child, rng, palette, keys[j], keys, compact,
                   forced_port=shared if child['kind'] == 'room' else None)
             for child, j in zip(node['children'], assignment)]
    content, placements = _pack(rng, [p[0] for p in pairs], bg, compact)
    padding = [int(rng.integers(2, 4 if compact else 6)) for _ in range(4)]
    top, bottom, left, right = padding
    h, w = content.shape[0]+top+bottom, content.shape[1]+left+right
    mask = _outline(rng, h, w, padding, compact)
    inp = np.full((h, w), bg, np.int8); inp[mask] = wall
    inp[top:top+content.shape[0], left:left+content.shape[1]] = content
    possibilities = np.arange(1, (h, w)[axis]-1)
    positions = sorted(map(int, rng.choice(possibilities, lane_count, replace=False)))
    rng.shuffle(keys)
    # Keep each child's assigned literal identity. Shuffling keys here changes
    # spatial guide order without changing the already visible child identities.
    guides = list(zip(positions, keys))
    for g, key in guides:
        pts = np.argwhere(mask)
        scan = pts[pts[:, axis] == g]
        a, b = scan[np.argmin(scan[:, 1-axis])], scan[np.argmax(scan[:, 1-axis])]
        inp[tuple(a)] = inp[tuple(b)] = key
    corner = node['corner'] if node['corner'] is not None else tuple(map(int, rng.integers(2, size=2)))
    source = None
    if incoming is not None:
        source = _corner_position(inp.shape, corner); inp[source] = incoming
    records = []
    # assignment used the original key order; retrieve identity from each child.
    for j, (child_in, child_out, child_source, child_handle) in enumerate(pairs):
        offset = np.array(placements[j]) + (top, left)
        reg = tuple(map(int, np.array(child_source) + offset))
        records.append(dict(key=int(child_in[child_source]), source=reg, out=child_out, handle=child_handle))
    out, handle = _model_output(records, guides, axis, bg, wall, incoming, corner)
    return inp, out, source, handle


def _key(a):
    return a.shape, np.asarray(a, np.int8).tobytes()


def _pair(rng, palette, plan, witness=None):
    errors = []
    for attempt in range(180):
        compact = attempt >= 8
        if witness is not None:
            node = _witness(witness, rng)
        elif plan == 'train':
            node = _random_tree(rng)
        else:
            node = _shift_tree(rng, plan)
        inp, out, _, _ = _draw(node, rng, palette, compact=compact)
        if max(*inp.shape, *out.shape) > 30:
            continue
        # Canvas margins are metric only. Root corner badges never exist, and
        # complete root walls may meet canvas edges without becoming anchors.
        available = [30 - d for d in inp.shape]
        extra = [int(rng.integers(0, min(6, v)+1)) for v in available]
        before = [int(rng.integers(v+1)) for v in extra]
        inp = np.pad(inp, tuple((a, e-a) for a, e in zip(before, extra)), constant_values=palette[0])
        if np.count_nonzero(inp == palette[0]) * 2 <= inp.size:
            continue
        try:
            decoded = _decode(inp)
            _require(np.array_equal(_predict(decoded), out), 'construction/decoder mismatch')
            if witness is not None:
                f = _features(decoded)
                _require(f['misaligned'] >= 1, 'witness needs genuine alignment')
                _require(max(f['areas']) > 1, 'witness needs a non-singleton blob')
        except ValueError as exc:
            errors.append(str(exc))
            continue
        return dict(input=inp, output=out)
    raise RuntimeError('bounded scene construction exhausted: ' + str(Counter(errors)))


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(name + ' outside the legal integer range')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    base = tuple(map(int, rng.permutation(10)))
    stable = bool(rng.random() < .3)
    shift = int(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', a), ('test', b)):
        for i in range(count):
            palette = base if stable else tuple(map(int, rng.permutation(10)))
            for _ in range(30):
                pair = _pair(rng, palette, 'train' if split == 'train' else shift,
                             witness=i if split == 'train' and i < 2 else None)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('duplicate input search exhausted')
            seen.add(_key(pair['input'])); puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    return puzzle


def _features(info):
    objs = info['objects']
    rooms = [o for o in objs if o['room']]
    depths = []
    for obj in rooms:
        depth, parent = 1, obj['parent']
        while parent is not None:
            depth += 1; parent = objs[parent]['parent']
        depths.append(depth)
    misaligned = 0
    for obj in rooms:
        guide_map = {g[1]: g[0] for g in obj['guides']}
        misaligned += sum(objs[j]['reg'][obj['axis']] != guide_map[objs[j]['key']] for j in obj['children'])
    def signature(i):
        obj = objs[i]
        if not obj['room']:
            return ('P',)
        return ('R', obj['axis'], obj.get('corner'), tuple(tuple(signature(j) for j in sorted(
            (j for j in obj['children'] if objs[j]['key'] == key),
            key=lambda j: (objs[j]['reg'][1-obj['axis']], objs[j]['reg'][obj['axis']])))
            for _, key, _ in obj['guides']))
    nonrect = 0
    for obj in rooms:
        local = obj['mask'][obj['lo'][0]:obj['hi'][0]+1, obj['lo'][1]:obj['hi'][1]+1]
        nonrect += not (local[0].all() and local[-1].all() and local[:, 0].all() and local[:, -1].all())
    key_axes = {}
    for obj in rooms:
        for _, key, axis in obj['guides']:
            key_axes.setdefault(key, set()).add(axis)
    scope_reuse = any(len(axes) > 1 for axes in key_axes.values())
    return dict(scope_reuse=scope_reuse, depth=max(depths), rooms=len(rooms), blobs=sum(not o['room'] for o in objs),
                axes={o['axis'] for o in rooms}, corners={o['corner'] for o in rooms if o['parent'] is not None},
                lanes=[len(o['guides']) for o in rooms], misaligned=misaligned,
                nonrect=nonrect, areas=[len(o['cells']) for o in objs if not o['room']],
                topology=signature(info['root']))


def validate_generated_puzzle(puzzle):
    """Raise ValueError on invalid grids, geometry, labels or teaching coverage."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, 'duplicate input'); seen.add(_key(inp))
            info = _decode(inp)
            _require(np.array_equal(_predict(info), out), 'output violates nested docking')
            normalized = _decode(out, info['background'])
            _require(_features(info)['topology'] == _features(normalized)['topology'], 'containment or guide assignment changed')
            _require(np.array_equal(_predict(normalized), out), 'output is not canonical')
            for obj in normalized['objects']:
                if not obj['room']:
                    _require(len(obj['cells']) == 1, 'a normalized blob is one point')
                    continue
                lo, hi = obj['lo'], obj['hi']
                _require(len(obj['cells']) == 2 * int(np.sum(hi-lo)), 'normalized contours are rectangular perimeters')
                children = [normalized['objects'][j] for j in obj['children']]
                _require(np.array_equal(np.min([o['lo'] for o in children], 0), lo+2), 'wrong top/left moat')
                _require(np.array_equal(np.max([o['hi'] for o in children], 0), hi-2), 'wrong bottom/right moat')
                gmap = {key: g for g, key, _ in obj['guides']}
                _require(all(o['reg'][obj['axis']] == gmap[o['key']] for o in children), 'handle not on local guide')
            root = normalized['objects'][normalized['root']]
            _require(tuple(root['lo']) == (0, 0) and tuple(root['hi']+1) == out.shape, 'wrong crop')
            if split == 'train':
                training.append((info, out))
    _require(set().union(*(_features(i)['axes'] for i, _ in training)) == {0, 1}, 'both orientations need demonstrations')
    _require(set().union(*(_features(i)['corners'] for i, _ in training)) == {(0, 0), (0, 1), (1, 0), (1, 1)}, 'all corner handles need demonstrations')
    _require(any(_features(i)['scope_reuse'] for i, _ in training), 'local guide reuse needs a training demonstration')
    for flags, message in ((dict(keep_blobs=True), 'blob abstraction'), (dict(flatten=True), 'containment'),
                           (dict(centered=True), 'reference rather than center'), (dict(ignore_ports=True), 'local port assignment')):
        _require(any(not np.array_equal(_predict(i, **flags), out) for i, out in training), 'training fails to distinguish ' + message)


def _relocated_ports(info, rng):
    # Preserve guide order and identity, change only their metric coordinates.
    grid, wall = info['grid'].copy(), info['wall']
    for obj in info['objects']:
        if not obj['room']:
            continue
        axis = obj['axis']
        for _, key, _ in obj['guides']:
            for p in obj['cells']:
                if int(grid[tuple(p)]) == key:
                    grid[tuple(p)] = wall
        choices = np.arange(obj['lo'][axis]+1, obj['hi'][axis])
        positions = sorted(map(int, rng.choice(choices, len(obj['guides']), replace=False)))
        for new_g, (_, key, _) in zip(positions, obj['guides']):
            scan = obj['cells'][obj['cells'][:, axis] == new_g]
            for p in (scan[np.argmin(scan[:, 1-axis])], scan[np.argmax(scan[:, 1-axis])]):
                grid[tuple(p)] = key
    return grid


def self_test():
    """Bounded 42-seed coverage, semantic ablations and interface checks."""
    fingerprints, shapes, topologies, outputs = set(), set(), set(), set()
    axes, corners, depths, lanes, areas = set(), set(), set(), set(), set()
    nonrect, rooms, palette_modes = 0, 0, set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        fingerprints.add(tuple((_key(pair['input']), _key(pair['output'])) for s in ('train', 'test') for pair in p[s]))
        palette_modes.add(len({_decode(pair['input'])['background'] for s in p for pair in p[s]}) == 1)
        for split in p:
            for pair in p[split]:
                info = _decode(pair['input']); f = _features(info)
                shapes.add(pair['input'].shape); topologies.add(f['topology'])
                bg, wall = info['background'], info['wall']
                norm = np.asarray(pair['output'])
                roles = {bg: 0, wall: 1}
                outputs.add((norm.shape, tuple(roles.setdefault(int(c), len(roles)) for c in norm.flat)))
                axes |= f['axes']; corners |= f['corners']; depths.add(f['depth'])
                lanes.update(f['lanes']); areas.update(f['areas'])
                nonrect += f['nonrect']; rooms += f['rooms']
                if split == 'test':
                    train_fs = [_features(_decode(q['input'])) for q in p['train']]
                    _require(f['depth'] > max(q['depth'] for q in train_fs) or max(f['lanes']) > max(max(q['lanes']) for q in train_fs),
                             'test lacks a structural shift')
        if seed < 5:
            _require(tuple((_key(q['input']), _key(q['output'])) for s in p for q in p[s]) ==
                     tuple((_key(q['input']), _key(q['output'])) for s in p for q in generate_puzzle(seed)[s]), 'nondeterministic puzzle')
        # Per-pair recoloring and transposition commute with the left/top
        # packing convention. Reflections need not commute with ragged lanes.
        pair = p['test'][0]
        rng = np.random.default_rng(seed + 31415)
        colors = rng.permutation(10)
        transformed = colors[pair['input'].T]
        expected = colors[pair['output'].T]
        _require(np.array_equal(solve(transformed), expected), 'transpose or palette equivariance failed')
        info = _decode(pair['input'])
        _require(np.array_equal(solve(_relocated_ports(info, rng)), pair['output']), 'metric guide coordinates leaked into compaction')
        root = info['objects'][info['root']]
        lo, hi = root['lo'], root['hi']
        cropped = info['grid'][lo[0]:hi[0]+1, lo[1]:hi[1]+1]
        translated = np.full(info['grid'].shape, info['background'], np.int8)
        start = [int(rng.integers(info['grid'].shape[k]-cropped.shape[k]+1)) for k in (0, 1)]
        y, x = start; h, w = cropped.shape
        translated[y:y+h, x:x+w] = cropped
        _require(np.array_equal(solve(translated), pair['output']), 'canvas placement leaked into answer')
    _require(len(fingerprints) == 42, 'seed diversity failed')
    _require(len(shapes) >= 60 and len(topologies) >= 80 and len(outputs) >= 80, 'structural variety too low')
    _require(axes == {0, 1} and len(corners) == 4 and {1, 2, 3} <= depths, 'geometry or nesting coverage missing')
    _require(lanes == {1, 2, 3} and len(areas) >= 8 and max(areas) >= 12, 'guide/blob variation missing')
    _require(nonrect > rooms*.45 and palette_modes == {False, True}, 'distortion or palette regime missing')
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(800+a*10+b, a, b)
            validate_generated_puzzle(p)
            _require(len(p['train']) == a and len(p['test']) == b, 'explicit count ignored')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid counts accepted')
    p = generate_puzzle(2718, 2, 1)
    info = _decode(p['train'][0]['input'])
    root = info['objects'][info['root']]
    # A genuine opening and a diagonal port pair must not be silently interpreted.
    for mode in ('opened_wall', 'diagonal_ports'):
        invalid = info['grid'].copy()
        y, x = map(int, root['lo'])
        if mode == 'opened_wall':
            invalid[y, x+1] = info['background']
        else:
            key = root['guides'][0][1]
            port = next(tuple(map(int, q)) for q in root['cells'] if invalid[tuple(q)] == key)
            invalid[port] = info['wall']
            invalid[y, x] = key
        try:
            _decode(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid room geometry accepted: '+mode)
    bad = {s: [{k: v.copy() for k, v in q.items()} for q in p[s]] for s in p}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('wrong output accepted')
    return dict(seeds=42, distinct_puzzles=len(fingerprints), input_shapes=len(shapes),
                topologies=len(topologies), normalized_outputs=len(outputs), nonrectangular_rooms=nonrect, rooms=rooms)
