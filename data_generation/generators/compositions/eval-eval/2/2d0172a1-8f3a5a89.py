"""Accessible chamber maps: gated containment, visibility, and ordinal redraw.

No parent module or stored puzzle is imported. The constructor labels examples
using a chamber graph; solve recovers them using pixel floods alone.
"""

from collections import deque
import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= d <= 30 for d in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("grids must be integer 2D arrays, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _flood(allowed, start):
    reached = np.zeros(allowed.shape, bool)
    if not allowed[start]:
        return reached
    reached[start] = True
    stack = [start]
    h, w = allowed.shape
    while stack:
        y, x = stack.pop()
        for dy, dx in N4:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and allowed[yy, xx] and not reached[yy, xx]:
                reached[yy, xx] = True
                stack.append((yy, xx))
    return reached


def _components(mask):
    remaining = mask.copy()
    result = []
    for y, x in zip(*np.nonzero(mask)):
        if remaining[y, x]:
            component = _flood(remaining, (int(y), int(x)))
            result.append(component)
            remaining[component] = False
    return result


def _holes(mask):
    padded = np.pad(mask, 1)
    outside = _flood(~padded, (0, 0))
    return (~padded & ~outside)[1:-1, 1:-1]


def _neighbors(mask, diagonal=False):
    padded = np.pad(mask, 1)
    h, w = mask.shape
    result = np.zeros_like(mask)
    offsets = [(y, x) for y in (-1, 0, 1) for x in (-1, 0, 1) if y or x] if diagonal else N4
    for y, x in offsets:
        result |= padded[1+y:1+y+h, 1+x:1+x+w]
    return result


def _roles(a):
    colors, counts = np.unique(a, return_counts=True)
    assert len(colors) == 3, "exactly background, wall, and door/marker colors required"
    order = np.argsort(counts)
    assert counts[order[0]] < counts[order[1]] < counts[order[2]], "role counts must be strictly ordered"
    return tuple(int(colors[i]) for i in (order[2], order[1], order[0]))


def _parse(a):
    bg, wall, ink = _roles(a)
    structural = a == wall
    colored = a == ink
    marker_mask = colored & ~_neighbors(structural)
    assert marker_mask.sum() == 1, "one ink cell without an orthogonal wall neighbor is the marker"
    marker = tuple(int(v) for v in np.argwhere(marker_mask)[0])
    gates = colored & ~marker_mask
    for y, x in np.argwhere(gates):
        around = [bool(0 <= y+dy < a.shape[0] and 0 <= x+dx < a.shape[1]
                       and structural[y+dy, x+dx]) for dy, dx in N4]
        assert around in ([True, True, False, False], [False, False, True, True]), "doors lie on straight wall segments"
    sealed = structural | gates
    assert not sealed[0].any() and not sealed[-1].any()
    assert not sealed[:, 0].any() and not sealed[:, -1].any(), "closed objects have an exterior margin"
    objects = []
    labels = np.zeros(a.shape, int)
    for index, mask in enumerate(_components(sealed)):
        coords = np.argwhere(mask)
        lo, hi = coords.min(0), coords.max(0)
        hole = _holes(mask)
        kind = 'ring' if hole.any() else 'blob'
        if kind == 'ring':
            assert len(_components(hole)) == 1, "each membrane encloses one chamber"
        else:
            assert not (mask & gates).any(), "solid blobs have no doors"
            assert mask.sum() >= np.prod(hi-lo+1) * .35, "blobs are compact"
        objects.append(dict(kind=kind, mask=mask, hole=hole, lo=lo, hi=hi,
                            center=tuple(int(v) for v in lo+hi),
                            gates=int((mask & gates).sum())))
        labels[mask] = index+1
    for dy, dx in ((1, 0), (0, 1), (1, 1), (1, -1)):
        if dy == 0:
            left, right = labels[:, :-1], labels[:, 1:]
        elif dx == 0:
            left, right = labels[:-1], labels[1:]
        elif dx == 1:
            left, right = labels[:-1, :-1], labels[1:, 1:]
        else:
            left, right = labels[:-1, 1:], labels[1:, :-1]
        assert not np.any((left > 0) & (right > 0) & (left != right)), "objects must not touch, even diagonally"
    assert any(o['kind'] == 'ring' for o in objects)
    parents = []
    for i, child in enumerate(objects):
        containing = []
        for j, parent in enumerate(objects):
            if i == j or parent['kind'] != 'ring':
                continue
            inside = parent['hole'][child['mask']]
            assert inside.all() or not inside.any(), "noncrossing membranes required"
            if inside.all():
                containing.append(j)
        parents.append(min(containing, key=lambda j: int(objects[j]['hole'].sum())) if containing else None)
    selected = _flood(a != wall, marker)
    complement = np.pad(~selected, 1, constant_values=True)
    exterior_padded = _flood(complement, (0, 0))
    exterior = exterior_padded[1:-1, 1:-1]
    rim = _neighbors(exterior_padded, True)[1:-1, 1:-1] & selected
    return dict(bg=bg, wall=wall, ink=ink, marker=marker, objects=objects,
                parents=parents, selected=selected, exterior=exterior, rim=rim)


def _frontier_indices(info, selected=None):
    """Objects seen from the reachable chamber, with sealed interiors opaque."""
    if selected is None:
        selected = info['selected']
        exterior = info['exterior']
    else:
        exterior = _flood(np.pad(~selected, 1, constant_values=True), (0, 0))[1:-1, 1:-1]
    touching = _neighbors(selected)
    indices = []
    for index, obj in enumerate(info['objects']):
        if not (obj['mask'] & touching).any():
            continue
        if obj['kind'] == 'ring':
            if obj['gates'] or (obj['mask'] & exterior).any():
                continue
        # A closed island retains only its hollow silhouette, never its contents.
        indices.append(index)
    return indices


def _frontier(info, selected=None):
    return [(info['objects'][i]['kind'], info['objects'][i]['center'])
            for i in _frontier_indices(info, selected)]


def _canonical(tokens, marker, palette):
    """Compress both coordinate orders using minimal separated 1/3-cell bands."""
    bg, wall, ink = palette
    tokens = list(tokens) + [('marker', tuple(2*int(v) for v in marker))]
    assert len({p for _, p in tokens}) == len(tokens), "visible token centers must be distinct"
    positions = []
    dimensions = []
    for axis in (0, 1):
        ranks = sorted({p[axis] for _, p in tokens})
        centers = {}
        start = 2  # external frame, then one background cell
        for i, rank in enumerate(ranks):
            size = max(3 if kind == 'ring' else 1 for kind, p in tokens if p[axis] == rank)
            centers[rank] = start + size//2
            start += size + (i < len(ranks)-1)
        dimensions.append(start+2)
        positions.append(centers)
    if max(dimensions) > 30:
        raise ValueError("the canonical chamber map must fit within 30 by 30")
    out = np.full(dimensions, bg, np.int8)
    out[0, :] = out[-1, :] = ink
    out[:, 0] = out[:, -1] = ink
    for kind, p in tokens:
        y, x = positions[0][p[0]], positions[1][p[1]]
        if kind == 'ring':
            out[y-1:y+2, x-1:x+2] = wall
            out[y, x] = bg
        else:
            out[y, x] = ink if kind == 'marker' else wall
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all roles, reachability, visibility, and layout from this input."""
    info = _parse(_grid(input_grid))
    return _canonical(_frontier(info), info['marker'], (info['bg'], info['wall'], info['ink']))


# Construction below uses an explicit containment/chamber graph, not _frontier.

def _blob(rng, small=False):
    choices = [1, 3] if small else [1, 3, 5]
    h, w = (int(rng.choice(choices)) for _ in range(2))
    a = np.ones((h, w), bool)
    kind = int(rng.integers(4))
    if kind == 1 and min(h, w) > 1:
        a[0, 0] = a[0, -1] = a[-1, 0] = a[-1, -1] = False
    elif kind == 2:
        a[:] = False
        a[h//2, :] = a[:, w//2] = True
    elif kind == 3 and h == w and h > 1:
        y, x = np.indices((h, w))
        a = abs(y-h//2)+abs(x-w//2) <= h//2
    return dict(kind='blob', shape=a)


def _ring(children, gates=0):
    return dict(kind='ring', children=children, gates=gates)


def _shift_records(records, offset):
    delta = np.array(offset)
    return [dict(rec, cells=rec['cells']+delta, doors=rec['doors']+delta,
                 center=tuple(int(v) for v in np.array(rec['center'])+2*delta)) for rec in records]


def _stack(tiles, axis, rng):
    if len(tiles) == 1:
        return tiles[0]
    cross = 1-axis
    # All dimensions are odd; odd gaps preserve odd group dimensions.
    gaps = [int(rng.choice([1, 3])) for _ in tiles[:-1]]
    extent = max(t[0].shape[cross] for t in tiles)
    offsets = [(extent-t[0].shape[cross])//2 for t in tiles]
    if rng.random() < .3:
        offsets = [v+2*int(rng.integers(2)) for v in offsets]
    shape = [0, 0]
    shape[axis] = sum(t[0].shape[axis] for t in tiles)+sum(gaps)
    shape[cross] = max(o+t[0].shape[cross] for o, t in zip(offsets, tiles))
    a = np.zeros(shape, np.int8)
    records = []
    position = 0
    for i, ((tile, recs), offset) in enumerate(zip(tiles, offsets)):
        start = [0, 0]
        start[axis], start[cross] = position, offset
        y, x = start
        h, w = tile.shape
        a[y:y+h, x:x+w] = tile
        records.extend(_shift_records(recs, start))
        position += tile.shape[axis]+(gaps[i] if i < len(gaps) else 0)
    return a, records


def _layout(tiles, rng):
    axis = int(rng.integers(2))
    if len(tiles) >= 3 and rng.random() < .65:
        cut = int(rng.integers(1, len(tiles)))
        return _stack([_stack(tiles[:cut], 1-axis, rng),
                       _stack(tiles[cut:], 1-axis, rng)], axis, rng)
    return _stack(tiles, axis, rng)


def _outline(h, w, pad, rng):
    a = np.zeros((h, w), bool)
    profiles = []
    for length in (w, h, w, h):
        p = np.zeros(length, int)
        if pad >= 3 and rng.random() < .8:
            low = pad
            high = length-pad
            if low < high:
                cuts = sorted(set([low, high]+[int(rng.integers(low, high+1)) for _ in range(3)]))
                for first, last in zip(cuts, cuts[1:]):
                    p[first:last] = int(rng.integers(pad-1))
        profiles.append(p)
    top, right, bottom, left = profiles
    path = [(int(top[x]), x) for x in range(w)]
    path += [(y, w-1-int(right[y])) for y in range(h)]
    path += [(h-1-int(bottom[x]), x) for x in range(w-1, -1, -1)]
    path += [(y, int(left[y])) for y in range(h-1, -1, -1)]
    path.append(path[0])
    for (y, x), (yy, xx) in zip(path, path[1:]):
        a[min(y, yy):max(y, yy)+1, x] = True
        a[yy, min(x, xx):max(x, xx)+1] = True
    return a


def _draw(node, path, rng, compact=False, outer=False):
    if node['kind'] == 'blob':
        mask = node['shape']
        h, w = mask.shape
        record = dict(path=path, parent=path[:-1] if len(path)>1 else None,
                      kind='blob', cells=np.argwhere(mask), doors=np.empty((0, 2), int),
                      center=(h-1, w-1))
        return mask.astype(np.int8), [record]
    children = [_draw(child, path+(i,), rng, compact) for i, child in enumerate(node['children'])]
    content, records = _layout(children, rng)
    pad = int(rng.choice([3, 4])) if outer else int(rng.choice([2, 3] if compact else [2, 3, 4]))
    h, w = (d+2*pad for d in content.shape)
    mask = _outline(h, w, pad, rng)
    a = mask.astype(np.int8)
    a[pad:pad+content.shape[0], pad:pad+content.shape[1]] = content
    hole = _holes(mask)
    assert len(_components(hole)) == 1
    candidates = []
    for y, x in np.argwhere(mask):
        around = [bool(0 <= y+dy < h and 0 <= x+dx < w and mask[y+dy, x+dx]) for dy, dx in N4]
        if around == [True, True, False, False]:
            separates = bool(x > 0 and hole[y, x-1]) != bool(x+1 < w and hole[y, x+1])
        elif around == [False, False, True, True]:
            separates = bool(y > 0 and hole[y-1, x]) != bool(y+1 < h and hole[y+1, x])
        else:
            separates = False
        if separates:
            candidates.append((int(y), int(x)))
    rng.shuffle(candidates)
    doors = []
    for cell in candidates:
        if all(max(abs(cell[0]-p[0]), abs(cell[1]-p[1])) >= 3 for p in doors):
            doors.append(cell)
        if len(doors) == node['gates']:
            break
    if not node['gates']:
        doors = []
    assert len(doors) == node['gates']
    for cell in doors:
        a[cell] = 2
    record = dict(path=path, parent=path[:-1] if len(path)>1 else None,
                  kind='ring', cells=np.argwhere(mask), doors=np.array(doors, int).reshape(-1, 2),
                  center=(h-1, w-1))
    return a, [record]+_shift_records(records, (pad, pad))


def _graph_tokens(records, room):
    graph = {None: []}
    for rec in records:
        if rec['kind'] == 'ring':
            graph[rec['path']] = []
    for rec in records:
        if rec['kind'] == 'ring' and len(rec['doors']):
            graph[rec['parent']].append(rec['path'])
            graph[rec['path']].append(rec['parent'])
    reached = {room}
    todo = deque([room])
    while todo:
        for neighbor in graph[todo.popleft()]:
            if neighbor not in reached:
                reached.add(neighbor)
                todo.append(neighbor)
    tokens = []
    for rec in records:
        if rec['parent'] in reached and (rec['kind'] == 'blob'
                or (not len(rec['doors']) and rec['path'] not in reached)):
            tokens.append((rec['kind'], rec['center']))
    return tokens, reached


def _room_mask(a, records, room):
    mask = np.ones(a.shape, bool)
    if room is not None:
        rec = next(rec for rec in records if rec['path'] == room)
        wall = np.zeros(a.shape, bool)
        wall[tuple(rec['cells'].T)] = True
        mask = _holes(wall)
    for rec in records:
        if rec['parent'] == room:
            child = np.zeros(a.shape, bool)
            child[tuple(rec['cells'].T)] = True
            mask &= ~(child | (_holes(child) if rec['kind'] == 'ring' else False))
    # Marker is visually separate from both walls and doors.
    return mask & (a == 0) & ~_neighbors(a != 0, True)


def _embed(core, records, rng):
    h, w = core.shape
    if max(h, w) > 28:
        raise ValueError("metric scene too large")
    height = int(rng.integers(h+2, min(30, h+6)+1))
    width = int(rng.integers(w+2, min(30, w+6)+1))
    y = int(rng.integers(1, height-h))
    x = int(rng.integers(1, width-w))
    a = np.zeros((height, width), np.int8)
    a[y:y+h, x:x+w] = core
    return a, _shift_records(records, (y, x))


def _materialize_pair(role, records, room, palette, rng, alignment=False):
    tokens, reached = _graph_tokens(records, room)
    assert len(tokens) <= 5, "keep the visible map legible"
    candidates = np.argwhere(_room_mask(role, records, room))
    assert len(candidates), "room has no clearly separate marker position"
    if alignment and tokens:
        aligned = [p for p in candidates if any(2*p[0] == c[0] or 2*p[1] == c[1] for _, c in tokens)]
        if aligned:
            candidates = np.array(aligned)
    marker = tuple(int(v) for v in candidates[int(rng.integers(len(candidates)))])
    role = role.copy()
    role[marker] = 2
    colors = np.array(palette, np.int8)
    inp = colors[role]
    out = _canonical(tokens, marker, palette)
    info = _parse(inp)
    assert sorted(_frontier(info)) == sorted(tokens), "pixel visibility must agree with the chamber graph"
    assert np.array_equal(solve(inp), out)
    return dict(input=inp, output=out)


def _witnesses(rng, palette):
    for attempt in range(180):
        small = True
        opened = _ring([_blob(rng, small) for _ in range(int(rng.choice([1, 2])))], int(rng.choice([1, 2])))
        opaque = _ring([_blob(rng, small) for _ in range(int(rng.choice([1, 2])))])
        children = [opened, opaque]
        if rng.random() < .4:
            children.append(_blob(rng, small))
        rng.shuffle(children)
        nodes = [_ring(children), _blob(rng, small)]
        try:
            main = _draw(nodes[0], (0,), rng, compact=True, outer=True)
            other = _draw(nodes[1], (1,), rng, compact=True)
            axis = int(np.argmin(main[0].shape))
            core, records = _stack([main, other] if rng.random() < .5 else [other, main], axis, rng)
            role, records = _embed(core, records, rng)
            inside = _materialize_pair(role, records, (0,), palette, rng, True)
            outside = _materialize_pair(role, records, None, palette, rng, True)
            if np.array_equal(inside['output'], outside['output']):
                continue
            features = [_features(p['input']) for p in (inside, outside)]
            if not features[0]['large_blob']:
                continue
            assert features[0]['opaque_hidden'] and features[0]['opened_visible']
            rotation = int(rng.integers(4))
            reflect = bool(rng.integers(2))
            for pair in (inside, outside):
                for key in pair:
                    a = np.rot90(pair[key], rotation)
                    pair[key] = (np.fliplr(a) if reflect else a).copy()
            return [inside, outside]
        except (AssertionError, ValueError):
            continue
    raise RuntimeError("bounded witness construction exhausted")


def _random_tree(rng, depth=1, gates=0, compact=False):
    children = []
    n = int(rng.choice([1, 2, 3], p=[.4, .45, .15]))
    for _ in range(n):
        if depth < 3 and rng.random() < (.6 if depth == 1 else .25):
            children.append(_random_tree(rng, depth+1, int(rng.choice([0, 1, 2, 3])), compact))
        else:
            children.append(_blob(rng, compact))
    return _ring(children, gates)


def _random_pair(rng, palette, test=False):
    for attempt in range(180):
        compact = attempt >= 12 or bool(rng.integers(2))
        if test:
            # A door at the root plus one inside contracts two levels, combining
            # the internal-door and exterior-selection evidence in the witnesses.
            inner = _random_tree(rng, 2, int(rng.choice([1, 2, 3])), compact)
            children = [inner] + [_blob(rng, compact) for _ in range(int(rng.integers(0, 3)))]
            if rng.random() < .5:
                children.append(_random_tree(rng, 2, 0, compact))
            rng.shuffle(children)
            main = _ring(children, int(rng.choice([1, 2, 3])))
        else:
            main = _random_tree(rng, compact=compact)
        roots = [main]
        for _ in range(int(rng.integers(0, 3))):
            roots.append(_random_tree(rng, 2, 0, compact) if rng.random() < .25 else _blob(rng, compact))
        rng.shuffle(roots)
        try:
            tiles = [_draw(node, (i,), rng, compact, outer=True) for i, node in enumerate(roots)]
            core, records = _layout(tiles, rng)
            role, records = _embed(core, records, rng)
            if test:
                # Select the component traversing the required two open levels.
                room = None
            else:
                rooms = [None]+[r['path'] for r in records if r['kind'] == 'ring']
                room = rooms[int(rng.integers(len(rooms)))]
            pair = _materialize_pair(role, records, room, palette, rng, bool(rng.integers(2)))
            if test and _features(pair['input'])['reachable_rooms'] < 3:
                continue
            rotation, reflect = int(rng.integers(4)), bool(rng.integers(2))
            for key in pair:
                a = np.rot90(pair[key], rotation)
                pair[key] = (np.fliplr(a) if reflect else a).copy()
            return pair
        except (AssertionError, ValueError):
            continue
    raise RuntimeError("bounded scene construction exhausted")


def _counts(num_train, num_test):
    for name, count, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or not low <= count <= high:
            raise ValueError(name+" is outside the allowed integer range")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _key(a):
    a = np.asarray(a, np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = tuple(int(v) for v in rng.choice(10, 3, replace=False))
    color_mode = int(rng.integers(3))  # stable, role swaps, or freely changing
    puzzle = dict(train=_witnesses(rng, palette), test=[])
    seen = {_key(p['input']) for p in puzzle['train']}
    for split, count in (('train', num_train-2), ('test', num_test)):
        for i in range(count):
            for retry in range(32):
                if color_mode == 0:
                    roles = palette
                elif color_mode == 1:
                    roles = tuple(np.roll(palette, i+1+retry))
                else:
                    roles = tuple(int(v) for v in rng.choice(10, 3, replace=False))
                pair = _random_pair(rng, roles, test=split == 'test')
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("distinct-input construction exhausted")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    return puzzle


def _features(a):
    info = _parse(a)
    objects = info['objects']
    frontier = _frontier(info)
    marker = info['marker']
    enclosing = [i for i, o in enumerate(objects) if o['kind'] == 'ring' and o['hole'][marker]]
    room = min(enclosing, key=lambda i: int(objects[i]['hole'].sum())) if enclosing else None
    records = [dict(path=i, parent=info['parents'][i], kind=o['kind'], center=o['center'],
                    doors=np.zeros((o['gates'], 2), int)) for i, o in enumerate(objects)]
    graph_tokens, reachable = _graph_tokens(records, room)
    assert sorted(graph_tokens) == sorted(frontier)
    visible_indices = _frontier_indices(info)
    opaque_hidden = any(objects[i]['kind'] == 'ring'
                        and any(parent == i for parent in info['parents']) for i in visible_indices)
    opened_visible = any(o['kind'] == 'ring' and o['gates'] and i in reachable for i, o in enumerate(objects))
    large_blob = any(objects[i]['kind'] == 'blob' and objects[i]['mask'].sum() > 1 for i in visible_indices)
    depth = 0
    for i, o in enumerate(objects):
        if o['kind'] == 'ring':
            d, parent = 1, info['parents'][i]
            while parent is not None:
                d += 1
                parent = info['parents'][parent]
            depth = max(depth, d)
    return dict(shape=a.shape, palette=(info['bg'], info['wall'], info['ink']),
                rings=sum(o['kind'] == 'ring' for o in objects), depth=depth,
                gates=sum(o['gates'] for o in objects), reachable_rooms=len(reachable),
                blobs=sum(k == 'blob' for k, _ in frontier), opaque=sum(k == 'ring' for k, _ in frontier),
                opaque_hidden=opaque_hidden, opened_visible=opened_visible, large_blob=large_blob,
                exterior_selected=None in reachable,
                blob_shapes=[tuple((o['hi']-o['lo']+1).tolist()) for o in objects if o['kind'] == 'blob'],
                ring_shapes=[tuple((o['hi']-o['lo']+1).tolist()) for o in objects if o['kind'] == 'ring'],
                shared_axis=any(c[0] == 2*marker[0] or c[1] == 2*marker[1] for _, c in frontier),
                frontier=tuple(sorted(frontier)))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(a) not in seen, "duplicate input"
            seen.add(_key(a))
            info = _parse(a)
            assert np.array_equal(out, solve(a)), "wrong accessible chamber map"
            bg, wall, ink = info['bg'], info['wall'], info['ink']
            assert np.all(out[0] == ink) and np.all(out[-1] == ink)
            assert np.all(out[:, 0] == ink) and np.all(out[:, -1] == ink)
            assert np.count_nonzero(out[1:-1, 1:-1] == ink) == 1, "marker remains one interior cell"
            assert np.all(out[1, 1:-1] == bg) and np.all(out[-2, 1:-1] == bg)
            assert np.all(out[1:-1, 1] == bg) and np.all(out[1:-1, -2] == bg)
            components = _components(out == wall)
            assert len(components) == len(_frontier(info))
            for component in components:
                coords = np.argwhere(component)
                lo, hi = coords.min(0), coords.max(0)
                if component.sum() == 1:
                    continue
                assert np.array_equal(hi-lo, [2, 2]) and component.sum() == 8
                assert out[tuple(lo+1)] == bg, "sealed chamber contents must be hidden"
            if split == 'train':
                training.append(_features(a))
    assert any(f['opaque_hidden'] and f['opened_visible'] and f['large_blob'] for f in training)
    assert any(f['exterior_selected'] for f in training)
    assert any(not f['exterior_selected'] for f in training)
    # The paired witnesses visibly change selection on the same walls and doors.
    found = False
    for i, first in enumerate(puzzle['train']):
        ai = _grid(first['input'])
        fi = _parse(ai)
        for second in puzzle['train'][i+1:]:
            aj = _grid(second['input'])
            fj = _parse(aj)
            if ai.shape != aj.shape or (fi['bg'], fi['wall'], fi['ink']) != (fj['bg'], fj['wall'], fj['ink']):
                continue
            clean_i, clean_j = ai.copy(), aj.copy()
            clean_i[fi['marker']], clean_j[fj['marker']] = fi['bg'], fj['bg']
            if np.array_equal(clean_i, clean_j) and _frontier(fi) != _frontier(fj):
                assert not np.array_equal(first['output'], second['output'])
                found = True
    assert found, "marker selection requires a paired demonstration"


def _puzzle_key(puzzle):
    return tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in puzzle[s])


def _fixture():
    a = np.zeros((17, 21), np.int8)
    a[2, 2:19] = a[14, 2:19] = 1
    a[2:15, 2] = a[2:15, 18] = 1
    a[5, 5:12] = a[11, 5:12] = 1
    a[5:12, 5] = a[5:12, 11] = 1
    a[8, 8] = 1
    a[7:10, 14:17] = 1
    a[0, 10] = 2
    return a


def self_test():
    """Bounded exact, metamorphic, corruption, count, and 42-seed checks."""
    arr = lambda text: np.array([[int(c) for c in row] for row in text.split()], np.int8)
    closed = _fixture()
    assert np.array_equal(solve(closed), arr('''2222222
2000002
2002002
2000002
2011102
2010102
2011102
2000002
2222222'''))
    opened = closed.copy()
    opened[2, 10] = 2
    assert np.array_equal(solve(opened), arr('''22222222222
20000000002
20000020002
20000000002
20111000002
20101000102
20111000002
20000000002
22222222222'''))
    both = opened.copy()
    both[5, 8] = 2
    assert np.array_equal(solve(both), arr('''222222222
200000002
200020002
200000002
201000102
200000002
222222222'''))
    # Filling or deleting hidden contents cannot affect the sealed exterior view.
    hidden = closed.copy()
    hidden[8, 8] = 0
    assert np.array_equal(solve(hidden), solve(closed))
    # A new same-center blob silhouette changes metric geometry, not its token.
    cross = both.copy()
    cross[7, 14] = cross[7, 16] = cross[9, 14] = cross[9, 16] = 0
    assert np.array_equal(solve(cross), solve(both))
    assert np.array_equal(solve(np.pad(both, 1)), solve(both)), "translation and blank margin do not encode the map"
    moved = opened.copy()
    moved[0, 10], moved[7, 7] = 0, 2
    assert _frontier(_parse(moved)) == [('blob', (16, 16))]
    assert not np.array_equal(solve(moved), solve(opened))
    stats, puzzles, signatures, outputs = [], [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_key(puzzle))
        puzzles.append(puzzle)
        for split in puzzle:
            for pair in puzzle[split]:
                stats.append(_features(pair['input']))
                info = _parse(pair['input'])
                normal = np.zeros(pair['output'].shape, np.int8)
                normal[pair['output'] == info['wall']] = 1
                normal[pair['output'] == info['ink']] = 2
                outputs.add(_key(normal))
        pair = puzzle['test'][0]
        original = pair['input'].copy()
        for rotation in range(4):
            for reflection in (False, True):
                inp, out = np.rot90(original, rotation), np.rot90(pair['output'], rotation)
                if reflection:
                    inp, out = np.fliplr(inp), np.fliplr(out)
                assert np.array_equal(solve(inp), out)
        permutation = np.random.default_rng(seed+42000).permutation(10)
        assert np.array_equal(solve(permutation[original]), permutation[pair['output']])
        assert np.array_equal(original, pair['input']), "solver must not mutate inputs"
    assert len(signatures) == 42 and len(outputs) >= 45
    assert len({f['shape'] for f in stats}) >= 65
    assert len({f['frontier'] for f in stats}) >= 100
    assert len({f['gates'] for f in stats}) >= 5
    assert {1, 2, 3}.issubset({f['reachable_rooms'] for f in stats})
    assert {0, 1, 2}.issubset({f['opaque'] for f in stats})
    assert {0, 1, 2, 3}.issubset({f['blobs'] for f in stats})
    assert {1, 2, 3}.issubset({f['depth'] for f in stats})
    assert len({s for f in stats for s in f['blob_shapes']}) >= 6
    assert len({s for f in stats for s in f['ring_shapes']}) >= 20
    assert all(len({f['palette'][role] for f in stats}) == 10 for role in range(3))
    assert _puzzle_key(generate_puzzle(0)) == _puzzle_key(puzzles[0])
    for nt, nq in ((2, 1), (10, 1), (10, 4), (7, 3)):
        p = generate_puzzle(590902775, nt, nq)
        validate_generated_puzzle(p)
        assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    corrupted = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    corrupted['test'][0]['output'][0, 0] = (int(corrupted['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return dict(seeds=42, pairs=len(stats), distinct_puzzles=len(signatures),
                normalized_outputs=len(outputs), input_shapes=len({f['shape'] for f in stats}),
                gate_counts=sorted({f['gates'] for f in stats}),
                reachable_room_counts=sorted({f['reachable_rooms'] for f in stats}),
                visible_blob_counts=sorted({f['blobs'] for f in stats}),
                visible_sealed_counts=sorted({f['opaque'] for f in stats}),
                depths=sorted({f['depth'] for f in stats}))
