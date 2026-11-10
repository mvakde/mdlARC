"""Gated rooms: topological normalization determines a rigid key's socket.

Only NumPy and the standard library are used. No parent code is imported.
The solver reconstructs containment, cardinal layout, gates and fits from pixels.
"""
from collections import deque
import numpy as np


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
INVERSE = (0, 3, 2, 1, 4, 5, 6, 7)


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _check(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'grid dimensions')
    _check(np.issubdtype(a.dtype, np.integer), 'integer cells required')
    _check(bool(np.all((0 <= a) & (a <= 9))), 'grid colors')
    return a.astype(np.int8, copy=False)


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, queue = {start}, [start]
        while queue:
            y, x = queue.pop()
            for dy, dx in DIRS:
                p = (y + dy, x + dx)
                if p in remaining:
                    remaining.remove(p)
                    part.add(p)
                    queue.append(p)
        result.append(frozenset(part))
    return result


def _cells(mask):
    return frozenset((int(y), int(x)) for y, x in np.argwhere(mask))


def _box(cells):
    return (min(y for y, x in cells), min(x for y, x in cells),
            max(y for y, x in cells), max(x for y, x in cells))


def _holes(cells, shape):
    """Actual enclosed space of one isolated component, including other objects."""
    h, w = shape
    blocked = {(y + 1, x + 1) for y, x in cells}
    seen, queue = {(0, 0)}, deque([(0, 0)])
    while queue:
        y, x = queue.popleft()
        for dy, dx in DIRS:
            p = (y + dy, x + dx)
            if (0 <= p[0] < h + 2 and 0 <= p[1] < w + 2
                    and p not in blocked and p not in seen):
                seen.add(p)
                queue.append(p)
    return frozenset((y, x) for y in range(h) for x in range(w)
                     if (y + 1, x + 1) not in seen and (y, x) not in cells)


def _parse(inp):
    values, counts = np.unique(inp, return_counts=True)
    order = np.argsort(counts)
    _check(len(values) >= 5 and counts[order[-1]] > counts[order[-2]], 'background')
    bg = int(values[order[-1]])
    parts = _components(_cells(inp != bg))
    mono = [p for p in parts if len({int(inp[q]) for q in p}) == 1]
    wall_colors = {int(inp[min(p)]) for p in mono}
    _check(len(wall_colors) == 1, 'monochrome scaffold role')
    wall = wall_colors.pop()
    scaffold, keys, gate_colors = [], [], set()
    for part in parts:
        colors = sorted({int(inp[p]) for p in part})
        _check(len(colors) in (1, 2), 'component color count')
        if wall in colors:
            hole = _holes(part, inp.shape)
            _check(not hole or len(_components(hole)) == 1, 'one hole per contour')
            gates = frozenset(p for p in part if int(inp[p]) != wall)
            if gates:
                _check(bool(hole), 'only contours can have gates')
                _check(len(_components(gates)) == 1, 'one gate segment')
                gate_colors.update(int(inp[p]) for p in gates)
            scaffold.append(dict(cells=part, box=_box(part), hole=hole, gates=gates))
        else:
            _check(len(colors) == 2, 'two-color key')
            regions = tuple(frozenset(p for p in part if int(inp[p]) == c) for c in colors)
            _check(all(len(_components(r)) == 1 for r in regions), 'connected key regions')
            keys.append(dict(colors=tuple(colors), regions=regions, cells=part))
    _check(len(gate_colors) == 1, 'shared visible gate role')
    gate = gate_colors.pop()
    _check(all(gate not in k['colors'] for k in keys), 'gate distinct from key colors')
    _check(1 <= len(keys) <= 2, 'one or two keys')
    # Distinct scaffold objects must have an 8-neighbor moat.
    owner = {p: i for i, obj in enumerate(scaffold) for p in obj['cells']}
    for (y, x), i in owner.items():
        _check(all(owner.get((y + dy, x + dx), i) == i
                   for dy in (-1, 0, 1) for dx in (-1, 0, 1)), 'touching scaffold')
    parents = []
    for i, obj in enumerate(scaffold):
        containers = []
        for j, candidate in enumerate(scaffold):
            if i == j:
                continue
            intersection = obj['cells'] & candidate['hole']
            _check(not intersection or intersection == obj['cells'], 'partial containment')
            if intersection:
                containers.append(j)
        parents.append(min(containers, key=lambda j: len(scaffold[j]['hole']))
                       if containers else None)
    children = {i: [] for i in range(len(scaffold))}
    roots = []
    for i, p in enumerate(parents):
        (roots if p is None else children[p]).append(i)
    for i, obj in enumerate(scaffold):
        _check(bool(children[i]) == bool(obj['hole']), 'nonempty rooms; terminal filled blobs')
    gated = [i for i, o in enumerate(scaffold) if o['gates']]
    _check(len(gated) == len(keys), 'one key per gate')
    for i in gated:
        p = parents[i]
        while p is not None:
            _check(p not in gated, 'gated rooms may not contain gated rooms')
            p = parents[p]
    for k in keys:
        _check(all(not (k['cells'] & o['hole']) for o in scaffold), 'keys are exterior')
    return dict(bg=bg, wall=wall, gate=gate, objects=scaffold, keys=keys,
                roots=roots, children=children, parents=parents, gated=gated)


def _tile(a, boxes, leaves):
    return dict(a=a, boxes=boxes, leaves=leaves)


def _combine(tiles, axis, offsets, gaps):
    cross = 1 - axis
    low = min(offsets)
    starts, size, position = [], [0, 0], 0
    for i, t in enumerate(tiles):
        start = [0, 0]
        start[axis], start[cross] = position, offsets[i] - low
        starts.append(start)
        position += t['a'].shape[axis] + (gaps[i] if i < len(gaps) else 0)
        size[cross] = max(size[cross], start[cross] + t['a'].shape[cross])
    size[axis] = position
    a = np.zeros(size, dtype=np.int8)
    boxes, leaves = {}, {}
    for t, (y, x) in zip(tiles, starts):
        h, w = t['a'].shape
        a[y:y+h, x:x+w] = t['a']
        boxes.update({k: (b[0]+y, b[1]+x, b[2]+y, b[3]+x)
                      for k, b in t['boxes'].items()})
        leaves.update({k: (p[0]+y, p[1]+x) for k, p in t['leaves'].items()})
    return _tile(a, boxes, leaves)


def _normalize(info):
    """Normalize only the scaffold, retaining gate-to-descendant alignment.

    Whitespace cuts describe cardinal groups. Projected leaf overlaps establish
    their offsets. All legal decompositions must agree, including object boxes.
    """
    objects, children, cache = info['objects'], info['children'], {}

    def align(tiles, axis):
        cross, edges = 1-axis, []
        for i in range(len(tiles)):
            for j in range(i+1, len(tiles)):
                shifts = set()
                for u, p in tiles[i]['leaves'].items():
                    for v, q in tiles[j]['leaves'].items():
                        a, b = objects[u]['box'], objects[v]['box']
                        if max(a[cross], b[cross]) <= min(a[cross+2], b[cross+2]):
                            shifts.add(p[cross] - q[cross])
                _check(len(shifts) <= 1, 'conflicting leaf alignments')
                if shifts:
                    edges.append((i, j, shifts.pop()))
        offsets = {0: 0}
        for _ in tiles:
            for i, j, delta in edges:
                if i in offsets and j in offsets:
                    _check(offsets[j] == offsets[i]+delta, 'alignment cycle')
                elif i in offsets:
                    offsets[j] = offsets[i]+delta
                elif j in offsets:
                    offsets[i] = offsets[j]-delta
        _check(len(offsets) == len(tiles), 'unanchored diagonal layout')
        return [offsets[i] for i in range(len(tiles))]

    def group(ids):
        ids = tuple(sorted(ids))
        if ids in cache:
            return cache[ids]
        if len(ids) == 1:
            i, = ids
            obj = objects[i]
            if not obj['hole']:
                result = _tile(np.ones((1, 1), dtype=np.int8), {i: (0, 0, 0, 0)}, {i: (0, 0)})
            else:
                content = group(children[i])
                a = np.pad(content['a'], 2)
                a[0, :] = a[-1, :] = a[:, 0] = a[:, -1] = 1
                boxes = {k: (b[0]+2, b[1]+2, b[2]+2, b[3]+2)
                         for k, b in content['boxes'].items()}
                boxes[i] = (0, 0, a.shape[0]-1, a.shape[1]-1)
                leaves = {k: (p[0]+2, p[1]+2) for k, p in content['leaves'].items()}
                if obj['gates']:
                    g = _box(obj['gates'])
                    all_children = [objects[j]['box'] for j in children[i]]
                    lo = [min(b[d] for b in all_children) for d in (0, 1)]
                    hi = [max(b[d+2] for b in all_children) for d in (0, 1)]
                    sides = []
                    for axis in (0, 1):
                        if g[axis+2] < lo[axis]:
                            sides.append((axis, 0))
                        if g[axis] > hi[axis]:
                            sides.append((axis, 1))
                    _check(len(sides) == 1, 'gate side')
                    axis, high = sides[0]
                    cross = 1-axis
                    _check(g[axis] == g[axis+2], 'straight gate segment')
                    positions = {p[cross] for k, p in leaves.items()
                                 if max(g[cross], objects[k]['box'][cross])
                                 <= min(g[cross+2], objects[k]['box'][cross+2])}
                    _check(len(positions) == 1, 'gate descendant anchor')
                    p = [0, 0]
                    p[axis] = a.shape[axis]-1 if high else 0
                    p[cross] = positions.pop()
                    a[tuple(p)] = 2
                result = _tile(a, boxes, leaves)
        else:
            candidates = []
            for axis in (0, 1):
                partitions, end = [], -1
                for i in sorted(ids, key=lambda k: objects[k]['box'][axis]):
                    b = objects[i]['box']
                    if not partitions or b[axis] > end+1:
                        partitions.append([])
                    partitions[-1].append(i)
                    end = max(end, b[axis+2])
                if len(partitions) < 2:
                    continue
                try:
                    tiles = [group(p) for p in partitions]
                    candidates.append(_combine(tiles, axis, align(tiles, axis), [1]*(len(tiles)-1)))
                except ValueError:
                    continue
            _check(bool(candidates), 'no cardinal decomposition')
            result = candidates[0]
            for t in candidates[1:]:
                _check(np.array_equal(t['a'], result['a']) and t['boxes'] == result['boxes']
                       and t['leaves'] == result['leaves'], 'ambiguous normalization')
        cache[ids] = result
        return result
    return group(info['roots'])


def _rooms(info, tile):
    a, result = tile['a'], []
    for i in info['gated']:
        y0, x0, y1, x1 = tile['boxes'][i]
        rectangle = frozenset((y, x) for y in range(y0, y1+1) for x in range(x0, x1+1))
        gates = rectangle & _cells(a == 2)
        _check(len(gates) == 1, 'one canonical gate')
        gate, = gates
        candidates = [(gate[0]+dy, gate[1]+dx) for dy, dx in DIRS
                      if y0 < gate[0]+dy < y1 and x0 < gate[1]+dx < x1]
        _check(len(candidates) == 1 and a[candidates[0]] == 0, 'gate interior')
        floor, stack = {candidates[0]}, [candidates[0]]
        while stack:
            y, x = stack.pop()
            for dy, dx in DIRS:
                p = (y+dy, x+dx)
                if y0 < p[0] < y1 and x0 < p[1] < x1 and a[p] == 0 and p not in floor:
                    floor.add(p)
                    stack.append(p)
        result.append(dict(id=i, box=(y0, x0, y1, x1), rectangle=rectangle,
                           floor=frozenset(floor), gate=gate))
    return result


def _raw(p, op):
    y, x = p
    return ((y, x), (x, -y), (-y, -x), (-x, y),
            (y, -x), (-x, -y), (-y, x), (x, y))[op]


def _transform(regions, op):
    raw = [[_raw(p, op) for p in r] for r in regions]
    y0 = min(y for r in raw for y, x in r)
    x0 = min(x for r in raw for y, x in r)
    return tuple(frozenset((y-y0, x-x0) for y, x in r) for r in raw)


def _shape(region):
    y0, x0, _, _ = _box(region)
    return tuple(sorted((y-y0, x-x0) for y, x in region))


def _paint(tile, info, moves):
    paint = {p: info['wall'] for p in _cells(tile['a'] == 1)}
    for m in moves:
        colors = m['colors'] if m['op'] < 4 else m['colors'][::-1]
        for region, color in zip(m['placed'], colors):
            for p in region:
                _check(p not in paint, 'overlapping assembly')
                paint[p] = color
    _check(bool(paint), 'empty output')
    y0, x0, y1, x1 = _box(paint)
    _check(y1-y0 < 30 and x1-x0 < 30, 'assembled bounds')
    out = np.full((y1-y0+1, x1-x0+1), info['bg'], dtype=np.int8)
    for (y, x), color in paint.items():
        out[y-y0, x-x0] = color
    return out


def _solve_details(input_grid, swap_policy=(0, 1)):
    inp = _grid(input_grid)
    info = _parse(inp)
    tile = _normalize(info)
    rooms = _rooms(info, tile)
    fixed = _cells(tile['a'] != 0)
    choices = []
    for k, key in enumerate(info['keys']):
        available = []
        for j, room in enumerate(rooms):
            sy, sx, _, _ = _box(room['floor'])
            for op in range(8):
                transformed = _transform(key['regions'], op)
                for plug in (0, 1):
                    py, px, _, _ = _box(transformed[plug])
                    dy, dx = sy-py, sx-px
                    placed = tuple(frozenset((y+dy, x+dx) for y, x in r) for r in transformed)
                    if placed[plug] != room['floor']:
                        continue
                    handle = placed[1-plug]
                    if handle & room['rectangle'] != {room['gate']}:
                        continue
                    if handle & fixed != {room['gate']}:
                        continue
                    effective_op = op if swap_policy[int(op >= 4)] == int(op >= 4) else (op+4) % 8
                    available.append(dict(key=k, room=j, plug=plug, op=effective_op,
                                          geometric_op=op, colors=key['colors'], placed=placed))
        _check(bool(available), 'key has no normalized room fit')
        choices.append(available)
    solutions = {}

    def search(k, used, occupied, moves):
        if len(solutions) > 1:
            return
        if k == len(choices):
            try:
                out = _paint(tile, info, moves)
            except ValueError:
                return
            solutions.setdefault(_key(out), (out, moves))
            return
        for m in choices[k]:
            full = m['placed'][0] | m['placed'][1]
            if m['room'] not in used and not full & occupied:
                search(k+1, used | {m['room']}, occupied | full, moves+[m])
    search(0, set(), set(), [])
    _check(len(solutions) == 1, 'assembly has no unique colored output')
    out, moves = next(iter(solutions.values()))
    return out, dict(info=info, tile=tile, rooms=rooms, moves=moves)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the unique assembled canonical floorplan, from this input alone."""
    return _solve_details(input_grid)[0]


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


# Generation uses abstract construction, then checks visible reconstruction.
# Internal node IDs are never returned or read by solve.
class _Builder:
    def __init__(self, rng):
        self.rng, self.serial = rng, 0

    def point(self):
        self.serial += 1
        return dict(kind='p', id=self.serial)

    def leaves(self, n):
        if n['kind'] == 'p':
            return [n['id']]
        if n['kind'] == 'r':
            return self.leaves(n['content'])
        return [k for c in n['items'] for k in self.leaves(c)]

    def group(self, items, axis=None):
        if len(items) == 1:
            return items[0]
        if axis is None:
            axis = int(self.rng.integers(2))
        return dict(kind='g', axis=axis, items=items,
                    anchors=[int(self.rng.choice(self.leaves(c))) for c in items])

    def room(self, content, gated=False):
        self.serial += 1
        return dict(kind='r', id=self.serial, content=content,
                    gate=(int(self.rng.integers(4)), int(self.rng.choice(self.leaves(content))))
                    if gated else None)

    def motif(self, nested=False):
        # Compose cardinal groups, rather than selecting saved output pictures.
        axis = int(self.rng.integers(2))
        counts = ([int(self.rng.integers(1, 3)), int(self.rng.integers(2, 4))]
                  if nested else [int(self.rng.integers(1, 4))
                                  for _ in range(int(self.rng.choice([2, 2, 3])))])
        if sum(counts) < 3:
            counts[-1] += 1
        rows = [[self.point() for _ in range(n)] for n in counts]
        if nested:
            index = int(self.rng.integers(len(rows[-1])))
            inner_points = [self.point() for _ in range(1 if self.rng.random() < .8 else 2)]
            rows[-1][index] = self.room(self.group(inner_points))
        groups = [self.group(row, axis) for row in rows]
        self.rng.shuffle(groups)
        return self.group(groups, 1-axis)


def _render(node, rng, metric=False, compact=False):
    kind = node['kind']
    if kind == 'p':
        h, w = ((1, 1) if compact == 2 else
                tuple(int(rng.integers(1, 3 if compact else 4)) for _ in range(2))) if metric else (1, 1)
        a = np.ones((h, w), dtype=np.int8)
        if metric and min(h, w) > 1 and rng.random() < .6:
            a[int(rng.choice([0, h-1])), int(rng.choice([0, w-1]))] = 0
        return _tile(a, {node['id']: (0, 0, h-1, w-1)},
                     {node['id']: ((h-1)//2, (w-1)//2)})
    if kind == 'g':
        tiles = [_render(n, rng, metric, compact) for n in node['items']]
        axis, cross = node['axis'], 1-node['axis']
        offsets = [-t['leaves'][k][cross] for t, k in zip(tiles, node['anchors'])]
        gaps = [int(rng.integers(1, 3 if compact else 5)) if metric and compact != 2 else 1 for _ in tiles[:-1]]
        return _combine(tiles, axis, offsets, gaps)
    content = _render(node['content'], rng, metric, compact)
    pads = [int(rng.integers(2, 4 if compact else 5)) for _ in range(4)] if metric and compact != 2 else [2]*4
    if metric and compact == 2:
        pads[int(rng.integers(4))] = 3
    top, bottom, left, right = pads
    a = np.pad(content['a'], ((top, bottom), (left, right)))
    child_mask = a != 0
    h, w = a.shape
    a[0, :] = a[-1, :] = a[:, 0] = a[:, -1] = 1
    # Stepped corner cuts change actual enclosure geometry without altering it.
    if metric:
        for cy, cx in ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1)):
            sy, sx = (1 if cy == 0 else -1), (1 if cx == 0 else -1)
            py, px = cy+sy, cx+sx
            if rng.random() < .55 and not child_mask[py-1:py+2, px-1:px+2].any():
                a[cy, cx] = 0
                a[py, px] = 1
    boxes = {k: (b[0]+top, b[1]+left, b[2]+top, b[3]+left)
             for k, b in content['boxes'].items()}
    boxes[node['id']] = (0, 0, h-1, w-1)
    leaves = {k: (p[0]+top, p[1]+left) for k, p in content['leaves'].items()}
    if node['gate'] is not None:
        side, anchor = node['gate']
        axis, high = side//2, side % 2
        cross = 1-axis
        p = [0, 0]
        p[axis] = a.shape[axis]-1 if high else 0
        p[cross] = leaves[anchor][cross]
        a[tuple(p)] = 2
    return _tile(a, boxes, leaves)


def _sample_scaffold(rng, mode):
    b = _Builder(rng)
    if mode == 'nested':
        root = b.room(b.motif(nested=True), True)
    elif mode == 'two_inside':
        rooms = [b.room(b.motif(), True) for _ in range(2)]
        root = b.room(b.group(rooms))
    elif mode == 'symmetric':
        point = b.point()
        room = b.room(point, True)
        axis, high = int(rng.integers(2)), int(rng.integers(2))
        room['gate'] = (2*axis+high, point['id'])
        upper = b.group([b.point(), b.point()], 1-axis)
        items = [upper, room] if high == 0 else [room, upper]
        content = b.group(items, axis)
        content['anchors'] = ([b.leaves(upper)[0], point['id']] if high == 0 else
                              [point['id'], b.leaves(upper)[0]])
        root = b.room(content)
    else:
        count = 2 if mode == 'two' else 1
        rooms = [b.room(b.motif(nested=mode == 'merged' or (mode == 'one' and rng.random() < .25)), True)
                 for _ in range(count)]
        root = b.group(rooms)
        if count == 1 and mode in ('inside', 'merged'):
            # An ungated surrounding room plus a sibling creates a corridor.
            root = b.room(b.group([root, b.point()]))
    return root


def _generated_rooms(tile):
    """Construction-side floor extraction, without using the input parser."""
    result = []
    for k, box in tile['boxes'].items():
        y0, x0, y1, x1 = box
        if y1-y0 < 4 or x1-x0 < 4:
            continue
        gates = [(y, x) for y in range(y0, y1+1) for x in range(x0, x1+1)
                 if tile['a'][y, x] == 2 and (y in (y0, y1) or x in (x0, x1))]
        if len(gates) != 1:
            continue
        gate = gates[0]
        rectangle = frozenset((y, x) for y in range(y0, y1+1) for x in range(x0, x1+1))
        floor = frozenset((y, x) for y, x in _cells(tile['a'] == 0)
                          if y0 < y < y1 and x0 < x < x1)
        # A child room's enclosed interior is not part of this room's floor.
        for other, (a, c, d, e) in tile['boxes'].items():
            if other != k and y0 < a < d < y1 and x0 < c < e < x1:
                floor -= frozenset((y, x) for y in range(a, d+1) for x in range(c, e+1))
        result.append(dict(id=k, box=box, rectangle=rectangle, floor=floor, gate=gate))
    return result


def _grow_handle(rng, room, tile, occupied):
    fixed = _cells(tile['a'] != 0)
    gate = room['gate']
    available = fixed - {gate}
    reach = int(rng.integers(2, 5))
    y0, x0, y1, x1 = room['box']
    handle = {gate}
    size = int(rng.integers(3, min(19, len(room['floor'])+5)))
    for _ in range(size-1):
        frontier = sorted({(y+dy, x+dx) for y, x in handle for dy, dx in DIRS}
                          - handle - room['rectangle'] - available - occupied)
        frontier = [p for p in frontier if y0-reach <= p[0] <= y1+reach
                    and x0-reach <= p[1] <= x1+reach]
        if not frontier:
            break
        # Variable compact, bent and branched attachments; all remain rigid.
        p = frontier[int(rng.integers(len(frontier)))]
        handle.add(p)
    _check(len(handle) >= 3, 'handle corridor too narrow')
    return frozenset(handle)


def _pack(rng, tiles, palette):
    """Pack independent rectangles; no key is assigned by source proximity."""
    for i, a in enumerate(tiles):
        for b in tiles[i+1:]:
            _check(a.shape[0]+b.shape[0]+1 <= 30 or a.shape[1]+b.shape[1]+1 <= 30,
                   'rectangles cannot coexist in 30 cells')
    for attempt in range(12):
        h, w = (30, 30) if attempt >= 5 else (
            int(rng.integers(max(18, max(a.shape[0] for a in tiles)), 31)),
            int(rng.integers(max(18, max(a.shape[1] for a in tiles)), 31)))
        positions, occupied = {}, np.zeros((h, w), dtype=bool)
        order = sorted(range(len(tiles)), key=lambda i: -tiles[i].size + rng.random())
        for i in order:
            th, tw = tiles[i].shape
            if th > h or tw > w:
                break
            options = [(y, x) for y in range(h-th+1) for x in range(w-tw+1)]
            rng.shuffle(options)
            for y, x in options:
                if not occupied[y:y+th, x:x+tw].any():
                    positions[i] = (y, x)
                    occupied[max(0, y-1):min(h, y+th+1), max(0, x-1):min(w, x+tw+1)] = True
                    break
            else:
                break
        else:
            inp = np.full((h, w), palette[0], dtype=np.int8)
            for i, a in enumerate(tiles):
                y, x = positions[i]
                inp[y:y+a.shape[0], x:x+a.shape[1]] = a
            if rng.random() < .85:
                y0, x0, y1, x1 = _box(_cells(inp != palette[0]))
                margins = rng.integers(0, 3, 4)
                inp = inp[max(0, y0-margins[0]):min(h, y1+1+margins[1]),
                          max(0, x0-margins[2]):min(w, x1+1+margins[3])].copy()
            return inp
    raise ValueError('packing exhausted')


def _make_pair(rng, palette, mode, witness_parity=None):
    last = None
    for attempt in range(160):
        try:
            # Compact realizations are used when the first larger attempts fail.
            node = _sample_scaffold(rng, mode)
            tile = _render(node, rng)
            raw_tile = _render(node, rng, True, 2 if attempt >= 30 else attempt >= 5)
            _check(max(raw_tile['a'].shape) <= 30, 'scaffold bounds')
            _check(not np.array_equal(raw_tile['a'], tile['a']), 'metric variation must require normalization')
            rooms = _generated_rooms(tile)
            _check(len(rooms) in (1, 2), 'construction gate count')
            bg, wall, gate, c1, c2 = palette[:5]
            occupied = set().union(*(r['floor'] for r in rooms))
            keys, moves = [], []
            for j, room in enumerate(rooms):
                handle = _grow_handle(rng, room, tile, occupied)
                occupied |= handle
                parity = witness_parity if j == 0 and witness_parity is not None else int(rng.integers(2))
                op = int(rng.integers(4)) + 4*parity
                regions = _transform((room['floor'], handle), INVERSE[op])
                pool = palette[3:] if j > 0 else (c1, c2)
                colors = tuple(int(c) for c in rng.choice(pool, 2, replace=False))
                cells = regions[0] | regions[1]
                _, _, y1, x1 = _box(cells)
                a = np.full((y1+1, x1+1), bg, dtype=np.int8)
                for region, color in zip(regions, colors):
                    for p in region:
                        a[p] = color
                keys.append(a)
                moves.append(dict(op=op, colors=colors, placed=(room['floor'], handle)))
            expected = _paint(tile, dict(bg=bg, wall=wall), moves)
            scaffold = np.choose(raw_tile['a'], [bg, wall, gate]).astype(np.int8)
            inp = _pack(rng, [scaffold]+keys, palette)
            inferred, details = _solve_details(inp)
            _check(np.array_equal(inferred, expected), 'visible normalization/assembly mismatch')
            if witness_parity is not None:
                _check(any(int(m['geometric_op'] >= 4) == witness_parity for m in details['moves']), 'parity witness')
            return dict(input=inp, output=expected)
        except ValueError as e:
            last = e
    raise RuntimeError('bounded pair construction exhausted: '+str(last))


def _count(value, low, high, name):
    _check(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
           and low <= value <= high, name+' must be a legal integer count')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    _check(num_train > num_test, 'num_train must exceed num_test')
    _check(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_))
           and seed >= 0, 'seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    stable = rng.random() < .3
    base_palette = tuple(int(c) for c in rng.choice(10, 5, replace=False))
    # Tests recombine enclosing corridors with nested islands, or put the two
    # familiar gated rooms in the exterior rather than under one shared roof.
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            palette = base_palette if stable else tuple(int(c) for c in rng.choice(10, 7, replace=False))
            if split == 'train' and i < 2:
                mode, parity = ('nested', 0) if i == 0 else ('two_inside', 1)
            else:
                modes = ('one', 'inside', 'nested', 'two_inside', 'symmetric') if split == 'train' else ('merged', 'two')
                mode, parity = str(rng.choice(modes)), None
            for _ in range(12):
                pair = _make_pair(rng, palette, mode, parity)
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct inputs exhausted')
    rng.shuffle(puzzle['train'])
    return puzzle


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _check(all(isinstance(puzzle[s], list) for s in puzzle), 'split lists')
    nt = _count(len(puzzle['train']), 2, 10, 'num_train')
    ne = _count(len(puzzle['test']), 1, 4, 'num_test')
    _check(nt > ne, 'split counts')
    seen, parities, training_details = set(), set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _check(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            actual, details = _solve_details(inp)
            _check(np.array_equal(actual, out), 'incorrect output')
            info, tile = details['info'], details['tile']
            room_depths = []
            for i, obj in enumerate(info['objects']):
                if obj['hole']:
                    depth, parent = 1, info['parents'][i]
                    while parent is not None:
                        depth += 1
                        parent = info['parents'][parent]
                    room_depths.append(depth)
            if split == 'train':
                _check(len(info['roots']) == 1 and max(room_depths) <= 2, 'training structural regime')
            else:
                _check((len(info['roots']) == 2 and len(details['rooms']) == 2 and max(room_depths) == 1)
                       or (len(info['roots']) == 1 and len(details['rooms']) == 1 and max(room_depths) == 3),
                       'held-out structural recombination')
            _check(info['gate'] not in np.unique(out), 'gates must be replaced')
            _check(set(np.unique(out)) == set(np.unique(inp)) - {info['gate']}, 'retained palette roles')
            # The output is cropped tightly; canonical fixed ink and key areas
            # are conserved, even though the scaffold's metric pixels shrink.
            _check(all(np.any(edge != info['bg']) for edge in (out[0], out[-1], out[:, 0], out[:, -1])), 'crop')
            _check(int(np.sum(out != info['bg'])) == int(np.sum(tile['a'] == 1))
                   + sum(len(m['placed'][0])+len(m['placed'][1]) for m in details['moves']), 'area conservation')
            if split == 'train':
                training_details.append(details)
                parities.update(int(m['geometric_op'] >= 4) for m in details['moves'])
    _check(parities == {0, 1}, 'both transformation parities must be demonstrated')
    _check(any(len(d['rooms']) == 2 for d in training_details), 'two-room training witness')
    _check(any(any(d['info']['children'][r['id']] and any(
        d['info']['objects'][j]['hole'] for j in d['info']['children'][r['id']])
        for r in d['rooms']) for d in training_details), 'nested obstacle witness')
    # Four plausible color policies are actually distinguished by labels.
    compatible = []
    for policy in ((0, 0), (0, 1), (1, 0), (1, 1)):
        try:
            if all(np.array_equal(_solve_details(p['input'], policy)[0], p['output']) for p in puzzle['train']):
                compatible.append(policy)
        except ValueError:
            pass
    _check(compatible == [(0, 1)], 'training color-policy identification')


def _symmetry_fixture(constrained=True, mirrored_key=False):
    """An orbit-1 floor: the surrounding point, not its socket, selects parity."""
    rng = np.random.default_rng(2307)
    b = _Builder(rng)
    point = b.point()
    room = b.room(point, True)
    room['gate'] = (0, point['id'])
    if constrained:
        upper = b.group([b.point(), b.point()], 1)
        content = b.group([upper, room], 0)
        content['anchors'] = [b.leaves(upper)[0], point['id']]
        root = b.room(content)
    else:
        root = room
    tile = _render(root, rng)
    raw_tile = _render(root, rng, True, 2)
    target, = _generated_rooms(tile)
    y, x = target['gate']
    handle = frozenset(((y, x), (y-1, x), (y-1, x-1), (y-2, x-1), (y-2, x-2)))
    op = 4 if mirrored_key else 0
    regions = _transform((target['floor'], handle), op)
    _, _, yy, xx = _box(regions[0] | regions[1])
    key = np.zeros((yy+1, xx+1), dtype=np.int8)
    for region, color in zip(regions, (3, 4)):
        for p in region:
            key[p] = color
    palette = (0, 1, 2, 3, 4)
    inp = _pack(rng, [raw_tile['a'], key], palette)
    out = _paint(tile, dict(bg=0, wall=1), [dict(op=op, colors=(3, 4),
                                                     placed=(target['floor'], handle))])
    return dict(input=inp, output=out)


def self_test():
    """Bounded 42-seed checks, structural coverage and counterfactuals."""
    structures, dimensions, slot_shapes, handle_shapes, operations = set(), set(), set(), set(), set()
    room_counts, floor_areas, leaf_counts, palettes, nonrect = set(), set(), set(), set(), 0
    area_relations, gate_sides, depths = set(), set(), set()
    puzzles = []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        for split in ('train', 'test'):
            for pair in p[split]:
                inp = pair['input']
                out, d = _solve_details(inp)
                info = d['info']
                dimensions.add(inp.shape)
                palettes.add((info['bg'], info['wall'], info['gate']))
                room_counts.add(len(d['rooms']))
                leaf_counts.add(len(d['tile']['leaves']))
                for i, obj in enumerate(info['objects']):
                    if obj['hole']:
                        depth, parent = 1, info['parents'][i]
                        while parent is not None:
                            depth += 1
                            parent = info['parents'][parent]
                        depths.add(depth)
                for r in d['rooms']:
                    y, x, yy, xx = r['box']
                    gy, gx = r['gate']
                    gate_sides.add('top' if gy == y else 'bottom' if gy == yy else
                                   'left' if gx == x else 'right')
                structures.add((d['tile']['a'].shape, d['tile']['a'].tobytes()))
                for obj in info['objects']:
                    if obj['hole']:
                        y, x, yy, xx = obj['box']
                        perimeter = {(r, c) for r in range(y, yy+1) for c in range(x, xx+1)
                                     if r in (y, yy) or c in (x, xx)}
                        nonrect += not perimeter <= obj['cells']
                for m in d['moves']:
                    operations.add(m['geometric_op'])
                    plug = m['placed'][m['plug']]
                    handle = m['placed'][1-m['plug']]
                    slot_shapes.add(min(_shape(_transform((plug,), op)[0]) for op in range(8)))
                    handle_shapes.add(min(_shape(_transform((handle,), op)[0]) for op in range(8)))
                    floor_areas.add(len(plug))
                    area_relations.add(int(np.sign(len(plug)-len(handle))))
                # Translation and semantic recoloring cannot affect the answer.
                if split == 'test' and seed < 12:
                    perm = np.random.default_rng(1000+seed).permutation(10)
                    _check(np.array_equal(solve(perm[inp]), perm[out]), 'recolor equivariance')
                    if max(inp.shape) <= 28:
                        shifted = np.pad(inp, 1, constant_values=info['bg'])
                        _check(np.array_equal(solve(shifted), out), 'source translation invariance')
                    for op in range(8):
                        transformed_input = np.rot90(inp[:, ::-1] if op >= 4 else inp, op % 4)
                        transformed_output = np.rot90(out[:, ::-1] if op >= 4 else out, op % 4)
                        _check(np.array_equal(solve(transformed_input), transformed_output), 'whole-scene symmetry')
    _check(len(structures) >= 35 and len(dimensions) >= 40, 'structural/canvas diversity')
    _check(len(slot_shapes) >= 12 and len(handle_shapes) >= 25,
           f'socket/handle diversity: {len(slot_shapes)}/{len(handle_shapes)}')
    _check(len(floor_areas) >= 8 and len(leaf_counts) >= 4,
           f'counts and topology variation: {len(floor_areas)}/{len(leaf_counts)}')
    _check(operations == set(range(8)) and room_counts == {1, 2}, 'operations and room counts')
    _check(len(palettes) >= 35 and nonrect >= 100, 'roles and deformed contours')
    _check(gate_sides == {'top', 'bottom', 'left', 'right'} and {1, 2, 3} <= depths,
           'gate sides and nesting depth')
    _check(all(_key(a['input']) == _key(b['input']) and _key(a['output']) == _key(b['output'])
               for s in ('train', 'test') for a, b in zip(puzzles[0][s], generate_puzzle(0)[s])), 'determinism')
    # Omitting B's reflection exchange fails a demonstrated reflected assembly.
    failed = False
    for pair in puzzles[0]['train']:
        try:
            wrong = _solve_details(pair['input'], (0, 0))[0]
            failed |= not np.array_equal(wrong, pair['output'])
        except ValueError:
            failed = True
    _check(failed, 'reflection counterfactual')
    # Omitting A's compression cannot fit keys to the distorted metric floor.
    metric_mismatch = False
    for pair in puzzles[0]['train']:
        _, d = _solve_details(pair['input'])
        for r in d['rooms']:
            obj = d['info']['objects'][r['id']]
            floor = obj['hole'] - set().union(*(o['cells'] | o['hole'] for i, o in enumerate(d['info']['objects'])
                                               if d['info']['parents'][i] == r['id']))
            metric_mismatch |= len(floor) != len(r['floor'])
    _check(metric_mismatch, 'metric-fit counterfactual')
    # Symmetric sockets are valid when the attached shape is constrained.
    for mirrored in (False, True):
        fixture = _symmetry_fixture(True, mirrored)
        actual, d = _solve_details(fixture['input'])
        _check(np.array_equal(actual, fixture['output']), 'constrained symmetric socket')
        _check(len(d['rooms'][0]['floor']) == 8, 'symmetric floor area')
        _check(int(d['moves'][0]['geometric_op'] >= 4) == int(mirrored), 'environment resolves parity')
    ambiguous = _symmetry_fixture(False)
    try:
        solve(ambiguous['input'])
    except ValueError as e:
        _check('unique colored output' in str(e), 'ambiguous fixture rejected for right reason')
    else:
        raise AssertionError('ambiguous symmetric socket accepted')
    # Damage labels: family validator must reject missing key ink.
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    out = damaged['test'][0]['output']
    bg = _parse(damaged['test'][0]['input'])['bg']
    p = next(p for p in _cells(out != bg))
    out[p] = bg
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError('damaged label accepted')
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid counts accepted')
    return {'seeds': 42, 'normalized_scaffolds': len(structures), 'input_shapes': len(dimensions),
            'socket_shapes': len(slot_shapes), 'handle_shapes': len(handle_shapes),
            'floor_areas': sorted(floor_areas), 'leaf_counts': sorted(leaf_counts),
            'operations': sorted(operations), 'nonrectangular_contours': nonrect,
            'area_relations': sorted(area_relations), 'depths': sorted(depths),
            'gate_sides': sorted(gate_sides)}
