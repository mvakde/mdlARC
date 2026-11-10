"""Scoped ribbon transcription: a coordinate key reads a containment forest.

No parent generator or evidence file is imported. Generation builds abstract
trees; solve reconstructs components, actual holes, and reading lanes from pixels.
"""
from __future__ import annotations

import numpy as np


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _components(mask):
    unseen = set(zip(*np.nonzero(mask)))
    result = []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        todo, cells = [p], []
        while todo:
            r, c = todo.pop()
            cells.append((int(r), int(c)))
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in unseen:
                    unseen.remove(q)
                    todo.append(q)
        result.append(np.array(cells, dtype=int))
    return result


def _holes(mask):
    # Only this component is an obstacle; enclosed children do not fill its hole.
    padded = np.pad(mask, 1)
    reached = np.zeros(padded.shape, dtype=bool)
    reached[0, 0] = True
    todo = [(0, 0)]
    h, w = padded.shape
    while todo:
        r, c = todo.pop()
        for y, x in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= y < h and 0 <= x < w and not padded[y, x] and not reached[y, x]:
                reached[y, x] = True
                todo.append((y, x))
    return (~padded & ~reached)[1:-1, 1:-1]


def _read_key(grid):
    """Return a proper rotation with fast pointing right; never reflect data."""
    candidates = []
    for rotation in range(4):
        a = np.rot90(grid, rotation)
        if min(a.shape) < 5:
            continue
        for bottom in (False, True):
            view = a[::-1] if bottom else a
            origin, fast, slow, bg = map(int, (view[0, 0], view[0, 1], view[1, 0], view[-1, -1]))
            if len({origin, fast, slow, bg}) != 4:
                continue
            if (view[0, -1] == origin and np.all(view[0, 1:-1] == fast)
                    and np.all(view[1:, 0] == slow)
                    and np.all(view[-1, 1:] == bg) and np.all(view[1:, -1] == bg)):
                candidates.append((a.copy(), bottom, bg, rotation, origin, fast, slow))
    assert len(candidates) == 1, 'One origin-echo L key is required'
    return candidates[0]


def _objects(a, bg, mask=None, output=False):
    if mask is None:
        mask = a != bg
    records = []
    labels = np.zeros(a.shape, dtype=int)
    for cells in _components(mask):
        lo, end = cells.min(axis=0), cells.max(axis=0)+1
        local = np.zeros(tuple(end-lo), dtype=bool)
        shifted = cells-lo
        local[shifted[:, 0], shifted[:, 1]] = True
        hole = _holes(local)
        values = a[cells[:, 0], cells[:, 1]]
        h, w = local.shape
        if hole.any():
            assert len(_components(hole)) == 1, 'One enclosed region per contour'
            kind = 'C'
            assert len(set(map(int, values))) == 1, 'Contours are monochrome'
        elif local.all() and h == 2 and w >= 3 and w % 2 == 1:
            kind = 'S'
        else:
            kind = 'B'
            assert len(set(map(int, values))) == 1, 'Blobs are monochrome'
            if output:
                assert h == w == len(cells) == 1, 'Normalized blobs are points'
            else:
                assert max(h, w) <= 4 and len(cells) >= h*w*.4, 'Compact hole-free blob required'
        full_hole = np.zeros(a.shape, dtype=bool)
        full_hole[lo[0]:end[0], lo[1]:end[1]] = hole
        records.append(dict(kind=kind, lo=lo, end=end, cells=cells, local=local,
                            hole=full_hole, color=int(values[0])))
        labels[cells[:, 0], cells[:, 1]] = len(records)
    assert records
    for first, second in ((labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:]),
                          (labels[:-1, :-1], labels[1:, 1:]), (labels[:-1, 1:], labels[1:, :-1])):
        assert not np.any((first != 0) & (second != 0) & (first != second)), 'Objects must not touch'
    parents = []
    for i, child in enumerate(records):
        containing = []
        for j, parent in enumerate(records):
            if i == j or parent['kind'] != 'C':
                continue
            inside = parent['hole'][child['cells'][:, 0], child['cells'][:, 1]]
            assert inside.all() or not inside.any(), 'Partial containment is forbidden'
            if inside.all():
                containing.append(j)
        parents.append(min(containing, key=lambda j: int(records[j]['hole'].sum())) if containing else None)
    children = {i: [] for i in range(len(records))}
    roots = []
    for i, p in enumerate(parents):
        (roots if p is None else children[p]).append(i)
    for i, obj in enumerate(records):
        assert bool(children[i]) == (obj['kind'] == 'C'), 'Contours must contain objects'
    return records, parents, children, roots


def _order(ids, objects, bottom):
    """Sibling boxes have aligned near edges in each separated lane band."""
    lanes = {}
    for i in ids:
        obj = objects[i]
        near = -int(obj['end'][0]) if bottom else int(obj['lo'][0])
        lanes.setdefault(near, []).append(i)
    bands, ordered = [], []
    for near in sorted(lanes):
        row = sorted(lanes[near], key=lambda i: int(objects[i]['lo'][1]))
        for i, j in zip(row, row[1:]):
            assert objects[j]['lo'][1] - objects[i]['end'][1] >= 1, 'Overlapping lane siblings'
        bands.append((min(int(objects[i]['lo'][0]) for i in row),
                      max(int(objects[i]['end'][0]) for i in row)))
        ordered.extend(row)
    physical = sorted(bands)
    assert all(b[0]-a[1] >= 1 for a, b in zip(physical, physical[1:])), 'Staggered lane bands'
    return ordered, tuple(len(lanes[k]) for k in sorted(lanes))


def _parse_input(value):
    grid = _grid(value)
    a, bottom, bg, rotation, origin, fast, slow = _read_key(grid)
    mask = a != bg
    mask[-1 if bottom else 0, :] = False
    mask[:, 0] = False
    objects, parents, children, roots = _objects(a, bg, mask)
    for obj in objects:
        assert obj['lo'][1] >= 2 and obj['end'][1] <= a.shape[1]-1, 'Clearance from key and opposite border'
        if bottom:
            assert obj['lo'][0] >= 1 and obj['end'][0] <= a.shape[0]-2
        else:
            assert obj['lo'][0] >= 2 and obj['end'][0] <= a.shape[0]-1
    assert any(o['kind'] == 'C' for o in objects), 'At least one reading scope'
    lanes = []

    def read(ids):
        ordered, populations = _order(ids, objects, bottom)
        lanes.append(populations)
        nodes = []
        for i in ordered:
            obj = objects[i]
            r, c = obj['lo']; y, x = obj['end']
            if obj['kind'] == 'S':
                nodes.append(('S', a[r:y, c:x].copy()))
            elif obj['kind'] == 'B':
                nodes.append(('B', obj['color']))
            else:
                nodes.append(('C', obj['color'], read(children[i])))
        return nodes

    forest = read(roots)
    return dict(forest=forest, background=bg, objects=objects, parents=parents,
                children=children, roots=roots, bottom=bottom, rotation=rotation,
                roles=(bg, origin, fast, slow), lanes=lanes, aligned=a)


def _stack(tiles, bg):
    width = max(t.shape[1] for t in tiles)
    assert all(t.shape[1] % 2 == width % 2 for t in tiles)
    out = np.full((sum(t.shape[0] for t in tiles)+len(tiles)-1, width), bg, dtype=np.int8)
    r = 0
    for t in tiles:
        c = (width-t.shape[1])//2
        out[r:r+t.shape[0], c:c+t.shape[1]] = t
        r += t.shape[0]+1
    return out


def _canonical(nodes, bg):
    tiles = []
    for node in nodes:
        if node[0] == 'S':
            tile = node[1].copy()
        elif node[0] == 'B':
            tile = np.array([[node[1]]], dtype=np.int8)
        else:
            content = _canonical(node[2], bg)
            tile = np.pad(content, 2, constant_values=bg)
            tile[0, :] = tile[-1, :] = node[1]
            tile[:, 0] = tile[:, -1] = node[1]
        tiles.append(tile)
    return _stack(tiles, bg)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """All conventions, colors, order, geometry and topology come from input."""
    info = _parse_input(input_grid)
    result = _canonical(info['forest'], info['background'])
    assert max(result.shape) <= 30, 'Canonical result exceeds the family bounds'
    return result


def _signature(nodes, payload=True):
    return tuple((n[0], _key(n[1]) if payload else n[1].shape[1]) if n[0] == 'S'
                 else ('B', n[1] if payload else None) if n[0] == 'B'
                 else ('C', n[1] if payload else None, _signature(n[2], payload)) for n in nodes)


def _stats(nodes):
    s = b = c = depth = 0
    maximum_siblings = len(nodes)
    for node in nodes:
        if node[0] == 'S':
            s += 1
        elif node[0] == 'B':
            b += 1
        else:
            ss, bb, cc, dd, mm = _stats(node[2])
            s += ss; b += bb; c += cc+1
            depth = max(depth, dd+1)
            maximum_siblings = max(maximum_siblings, mm)
    return s, b, c, depth, maximum_siblings


def _flatten_without_scopes(info):
    # Deliberate ablation: raw leaf near-edge raster, discarding all contours.
    a = []
    for obj in info['objects']:
        if obj['kind'] != 'C':
            near = -int(obj['end'][0]) if info['bottom'] else int(obj['lo'][0])
            a.append((near, int(obj['lo'][1]), obj))
    # Compare leaf identities, not just an obviously missing output frame.
    return tuple((o['kind'], tuple(o['cells'].min(axis=0))) for _, _, o in sorted(a, key=lambda t: t[:2]))


def _scope_leaf_order(info):
    order = []

    def visit(ids):
        ordered, _ = _order(ids, info['objects'], info['bottom'])
        for i in ordered:
            obj = info['objects'][i]
            if obj['kind'] == 'C':
                visit(info['children'][i])
            else:
                order.append((obj['kind'], tuple(obj['lo'])))
    visit(info['roots'])
    return tuple(order)


def _competing_output(info, mode):
    """Concrete simpler reading rules used only for demonstration checks."""
    def read(ids):
        ordered, counts = _order(ids, info['objects'], info['bottom'])
        rows, start = [], 0
        for count in counts:
            rows.append(ordered[start:start+count]); start += count
        if mode == 'reverse_fast':
            ordered = [i for row in rows for i in row[::-1]]
        elif mode == 'reverse_slow':
            ordered = [i for row in rows[::-1] for i in row]
        elif mode == 'fast_major':
            ordered = sorted(ordered, key=lambda i: (int(info['objects'][i]['lo'][1]),
                -int(info['objects'][i]['end'][0]) if info['bottom'] else int(info['objects'][i]['lo'][0])))
        nodes = []
        for i in ordered:
            o = info['objects'][i]
            if o['kind'] == 'C':
                nodes.append(('C', o['color'], read(info['children'][i])))
            elif o['kind'] == 'B':
                nodes.append(('B', o['color']))
            else:
                r, c = o['lo']; y, x = o['end']
                block = info['aligned'][r:y, c:x].copy()
                if mode == 'point_ribbons':
                    nodes.append(('B', int(block[0, 0])))
                else:
                    if mode == 'reflect_rows':
                        block = block[::-1]
                    elif mode == 'reflect_columns':
                        block = block[:, ::-1]
                    nodes.append(('S', block))
        return nodes
    return _canonical(read(info['roots']), info['background'])


def _check_output(out, info):
    bg = info['background']
    objects, _, children, roots = _objects(out, bg, output=True)

    def read(ids):
        ids = sorted(ids, key=lambda i: int(objects[i]['lo'][0]))
        tiles = [objects[i] for i in ids]
        lo = np.min([o['lo'] for o in tiles], axis=0)
        end = np.max([o['end'] for o in tiles], axis=0)
        center = int(lo[1]+end[1])
        assert all(int(o['lo'][1]+o['end'][1]) == center for o in tiles), 'Local symmetric centering'
        assert all(int(b['lo'][0]-a['end'][0]) == 1 for a, b in zip(tiles, tiles[1:])), 'One sibling gap'
        nodes = []
        for i in ids:
            obj = objects[i]
            r, c = obj['lo']; y, x = obj['end']
            if obj['kind'] == 'S':
                nodes.append(('S', out[r:y, c:x].copy()))
            elif obj['kind'] == 'B':
                nodes.append(('B', obj['color']))
            else:
                ring = obj['local']
                expected = np.zeros(ring.shape, dtype=bool)
                expected[0, :] = expected[-1, :] = True
                expected[:, 0] = expected[:, -1] = True
                assert np.array_equal(ring, expected), 'Every normalized contour is an exact frame'
                child_boxes = [objects[j] for j in children[i]]
                assert np.array_equal(np.min([o['lo'] for o in child_boxes], axis=0), obj['lo']+2)
                assert np.array_equal(np.max([o['end'] for o in child_boxes], axis=0), obj['end']-2)
                nodes.append(('C', obj['color'], read(children[i])))
        return nodes

    actual = read(roots)
    assert _signature(actual) == _signature(info['forest']), 'Containment, sequence, colors and payload preservation'
    assert np.array_equal(np.min([objects[i]['lo'] for i in roots], axis=0), [0, 0])
    assert np.array_equal(np.max([objects[i]['end'] for i in roots], axis=0), out.shape), 'Tight outer crop'
    s, b, c, _, _ = _stats(actual)
    assert out.shape[0] == 3*s+2*b+4*c-1
    assert out.shape[1] % 2 == 1


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training, testing = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'Duplicate input'
            seen.add(_key(inp))
            info = _parse_input(inp)
            assert np.array_equal(solve(inp), out), 'Wrong solution'
            _check_output(out, info)
            (training if split == 'train' else testing).append(info)
    assert {i['rotation'] % 2 for i in training} == {0, 1}, 'Train both fast axes'
    assert {i['bottom'] for i in training} == {False, True}, 'Train both key handednesses'
    assert any(_stats(i['forest'])[3] >= 2 for i in training), 'Teach recursive scopes'
    assert any(_scope_leaf_order(i) != _flatten_without_scopes(i) for i in training), 'Containment must alter leaf order'
    assert any(any(len(lane) > 1 and max(lane) > 1 for lane in i['lanes']) for i in training), 'Teach lane-major reading'
    assert any(any(o['kind'] == 'B' and len(o['cells']) > 1 for o in i['objects']) for i in training), 'Teach blob compression'
    assert any(any(o['kind'] == 'C' and not (o['local'][0].all() and o['local'][-1].all()
                   and o['local'][:, 0].all() and o['local'][:, -1].all()) for o in i['objects']) for i in training), 'Teach shape normalization'
    for mode in ('reverse_fast', 'reverse_slow', 'fast_major', 'point_ribbons', 'reflect_rows', 'reflect_columns'):
        assert any(not np.array_equal(_competing_output(i, mode), _canonical(i['forest'], i['background']))
                   for i in training), f'Train must distinguish {mode}'
    assert any(_stats(i['forest'])[3] > max(_stats(t['forest'])[3] for t in training)
               or _stats(i['forest'])[4] > max(_stats(t['forest'])[4] for t in training) for i in testing), 'Structural train/test shift'


def _blob(rng, color, compact, witness=False):
    h, w = map(int, rng.integers(2 if witness else 1, 3 if compact else 5, size=2))
    if h == 2 and w >= 3 and w % 2:
        h = 3  # That rectangle belongs to the ribbon vocabulary.
    mask = np.ones((h, w), dtype=bool)
    if h >= 2 and w >= 2 and rng.random() < .65:
        for r, c in ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1)):
            if rng.random() < .5 and mask.sum() > h*w*.5:
                candidate = mask.copy(); candidate[r, c] = False
                if len(_components(candidate)) == 1 and not _holes(candidate).any():
                    mask = candidate
    rows, cols = np.nonzero(mask)
    mask = mask[rows.min():rows.max()+1, cols.min():cols.max()+1]
    # Peeling can reveal a full ribbon-shaped box; keep it a visibly different blob.
    if mask.all() and mask.shape[0] == 2 and mask.shape[1] >= 3 and mask.shape[1] % 2:
        mask[0, 0] = False
    return mask, color


def _source_group(parts, groups, bg, rng, compact, lane_witness=False):
    rows, cursor = [], 0
    for count in groups:
        tiles = parts[cursor:cursor+count]; cursor += count
        gaps = [int(rng.integers(1, 3 if compact else 5)) for _ in tiles[:-1]]
        row = np.full((max(t.shape[0] for t in tiles), sum(t.shape[1] for t in tiles)+sum(gaps)), bg, dtype=np.int8)
        col = 0
        for j, tile in enumerate(tiles):
            row[:tile.shape[0], col:col+tile.shape[1]] = tile
            col += tile.shape[1]+(gaps[j] if j < len(gaps) else 0)
        rows.append(row)
    gaps = [int(rng.integers(1, 3 if compact else 5)) for _ in rows[:-1]]
    result = np.full((sum(t.shape[0] for t in rows)+sum(gaps), max(t.shape[1] for t in rows)), bg, dtype=np.int8)
    r = 0
    for j, row in enumerate(rows):
        offset = 0 if lane_witness and j == 1 else int(rng.integers(result.shape[1]-row.shape[1]+1))
        result[r:r+row.shape[0], offset:offset+row.shape[1]] = row
        r += row.shape[0]+(gaps[j] if j < len(gaps) else 0)
    return result


def _partition(rng, n):
    # A random subset of inter-child cuts supplies 1..n reading lanes.
    result, count = [], 1
    for _ in range(n-1):
        if rng.random() < .5:
            result.append(count); count = 1
        else:
            count += 1
    return result+[count]


def _outline(h, w, padding, rng, force=False):
    ring = np.zeros((h, w), dtype=bool)
    ring[0, :] = ring[-1, :] = True
    ring[:, 0] = ring[:, -1] = True
    choices = [j for j, p in enumerate(padding) if p >= 3]
    selected = [j for j in choices if rng.random() < .65]
    if force and choices and not selected:
        selected = [int(rng.choice(choices))]
    for side in selected:
        # Work in a rotated view whose chosen side is the top edge.
        before = ring.copy()
        turn = (0, 2, 3, 1)[side]  # top, bottom, left, right
        view = np.rot90(ring, turn)
        width = view.shape[1]
        if width < 7:
            continue
        start = int(rng.integers(2, width-4))
        end = int(rng.integers(start+2, width-2))
        depth = int(rng.integers(1, padding[side]-1))
        view[0, start+1:end] = False
        view[:depth+1, start] = view[:depth+1, end] = True
        view[depth, start:end+1] = True
        # Side deformations can approach a corner. Keep only deformations
        # that preserve one connected contour and one enclosed region.
        if len(_components(ring)) != 1 or len(_components(_holes(ring))) != 1:
            ring[:] = before
    return ring


def _draw(nodes, bg, rng, compact, mirror, root_groups=None, witness=False):
    tiles = []
    for node in nodes:
        if node[0] == 'S':
            tiles.append(node[1][::-1].copy() if mirror else node[1].copy())
        elif node[0] == 'B':
            mask, color = _blob(rng, node[1], compact, witness)
            tiles.append(np.where(mask, color, bg).astype(np.int8))
        else:
            groups = _partition(rng, len(node[2]))
            if witness and len(node[2]) == 3:
                groups = [2, 1]
            content = _draw(node[2], bg, rng, compact, mirror, groups, witness)
            padding = list(map(int, rng.integers(2, 4 if compact else 6, size=4)))
            if witness:
                padding[int(rng.integers(4))] = 3
            top, bottom, left, right = padding
            h, w = content.shape
            mask = _outline(h+top+bottom, w+left+right, padding, rng, witness)
            tile = np.where(mask, node[1], bg).astype(np.int8)
            tile[top:top+h, left:left+w] = content
            tiles.append(tile)
    groups = root_groups or _partition(rng, len(nodes))
    return _source_group(tiles, groups, bg, rng, compact, witness and len(nodes) == 3 and groups == [2, 1])


def _ribbon(rng, palette, length=None, asymmetric=False):
    length = int(length or rng.choice([3, 5, 7, 9]))
    mode = int(rng.integers(4))
    if asymmetric:
        colors = rng.choice(palette, 3, replace=False)
        block = rng.choice(colors, size=(2, length)).astype(np.int8)
        # Random sampled array, repaired only if the orientation is invisible.
        for _ in range(20):
            if not np.array_equal(block, block[::-1]) and not np.array_equal(block, block[:, ::-1]):
                break
            block = rng.choice(colors, size=(2, length)).astype(np.int8)
        else:
            block[:] = colors[0]; block[0, 0] = colors[1]; block[1, -1] = colors[2]
    elif mode == 0:
        block = np.full((2, length), rng.choice(palette), dtype=np.int8)
    elif mode == 1:
        block = np.repeat(rng.choice(palette, size=(2, 1)), length, axis=1).astype(np.int8)
    elif mode == 2:
        block = np.repeat(rng.choice(palette, size=(1, length)), 2, axis=0).astype(np.int8)
    else:
        motif = rng.choice(palette, size=(2, int(rng.integers(1, length+1))))
        block = np.tile(motif, (1, length))[:, :length].astype(np.int8)
    return ('S', block)


def _sample_tree(rng, palette, mode):
    dot = lambda: ('B', int(rng.choice(palette)))
    wrap = lambda nodes: ('C', int(rng.choice(palette)), nodes)
    if mode == 'scope_witness':
        lengths = rng.permutation([3, 5, 7])
        inside = [_ribbon(rng, palette, int(lengths[0]), True), dot(), _ribbon(rng, palette, int(lengths[1]))]
        rng.shuffle(inside)
        return [wrap(inside), _ribbon(rng, palette, int(lengths[2]), True)]
    if mode == 'nested_witness':
        inner = [dot(), _ribbon(rng, palette, int(rng.choice([3, 5])), True)]
        rng.shuffle(inner)
        outer = [_ribbon(rng, palette, int(rng.choice([3, 5, 7])), True), wrap(inner)]
        rng.shuffle(outer)
        return [wrap(outer), dot()] if rng.random() < .5 else [dot(), wrap(outer)]
    if mode == 'deep_test':
        deepest = [_ribbon(rng, palette, int(rng.choice([3, 5])), True)]
        middle = [wrap(deepest), dot()]; rng.shuffle(middle)
        outer = [wrap(middle), _ribbon(rng, palette, int(rng.choice([3, 5, 7])))]
        rng.shuffle(outer)
        root = [wrap(outer)]
        if rng.random() < .5:
            root.append(dot()); rng.shuffle(root)
        return root
    if mode == 'wide_test':
        children = [_ribbon(rng, palette, int(rng.choice([3, 5])), True), dot(),
                    wrap([_ribbon(rng, palette, int(rng.choice([3, 5])))]),
                    _ribbon(rng, palette, int(rng.choice([3, 5])))]
        rng.shuffle(children)
        return [wrap(children)]
    # Ordinary demonstrations: procedurally wrap contiguous slices of a leaf
    # sequence. These slices are logical, not contiguous global raster spans.
    n = int(rng.integers(2, 7))
    strips = int(rng.integers(1, n))
    nodes = [_ribbon(rng, palette) for _ in range(strips)]+[dot() for _ in range(n-strips)]
    rng.shuffle(nodes)
    for _ in range(int(rng.integers(1, 4))):
        start = int(rng.integers(len(nodes)))
        count = int(rng.integers(1, min(3, len(nodes)-start)+1))
        candidate = nodes[:start]+[wrap(nodes[start:start+count])]+nodes[start+count:]
        s, b, c, depth, sib = _stats(candidate)
        if depth <= 2 and 3*s+2*b+4*c-1 <= 30:
            nodes = candidate
    while len(nodes) > 3:
        nodes = [wrap(nodes[:3])]+nodes[3:]
    s, b, c, depth, sib = _stats(nodes)
    return nodes if c and depth <= 2 and 3*s+2*b+4*c-1 <= 30 else [wrap([_ribbon(rng, palette), dot()])]


def _make_pair(rng, roles, mode, config):
    bg, origin, fast, slow = roles
    pool = [c for c in range(10) if c != bg]
    palette = rng.choice(pool, size=int(rng.integers(3, 10)), replace=False)
    rotation, mirror = config
    for attempt in range(100):
        nodes = _sample_tree(rng, palette, mode)
        expected = _canonical(nodes, bg)
        if max(expected.shape) > 30:
            continue
        source = _draw(nodes, bg, rng, attempt >= 4, mirror,
                       [2] if mode == 'scope_witness' else None,
                       mode in ('scope_witness', 'nested_witness'))
        if max(source.shape) > 27:
            continue
        # Independent empty margins keep the L key separate from all objects.
        extra = [int(rng.integers(1, min(4, 28-d)+1)) for d in source.shape]
        h, w = source.shape[0]+extra[0]+1, source.shape[1]+extra[1]+1
        if h > 29 or w > 29:
            continue
        interior = np.full((h, w), bg, dtype=np.int8)
        y = int(rng.integers(1, h-source.shape[0]))
        x = int(rng.integers(1, w-source.shape[1]))
        interior[y:y+source.shape[0], x:x+source.shape[1]] = source
        grid = np.full((h+1, w+1), bg, dtype=np.int8)
        grid[1:, 1:] = interior
        grid[0, :] = fast
        grid[0, 0] = grid[0, -1] = origin
        grid[1:, 0] = slow
        if mirror:
            grid = grid[::-1]
        grid = np.rot90(grid, rotation).copy()
        info = _parse_input(grid)
        assert _signature(info['forest']) == _signature(nodes), 'Visible reading must recover the sampled tree'
        assert np.array_equal(solve(grid), expected)
        return {'input': grid, 'output': expected}
    raise RuntimeError('Bounded source packing exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    base_roles = tuple(map(int, rng.choice(10, 4, replace=False)))
    stable = bool(rng.random() < .35)
    k, mirror = int(rng.integers(4)), bool(rng.integers(2))
    configs = [(k, mirror), ((k+1) % 4, not mirror)]
    shift = 'deep_test' if rng.random() < .5 else 'wide_test'
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for i in range(count):
            mode = ('scope_witness' if i == 0 else 'nested_witness' if i == 1 else 'ordinary') if split == 'train' else shift
            config = configs[i] if split == 'train' and i < 2 else (
                configs[int(rng.integers(2))] if rng.random() < .5 else (int(rng.integers(4)), bool(rng.integers(2))))
            for _ in range(20):
                roles = base_roles if stable else tuple(map(int, rng.choice(10, 4, replace=False)))
                pair = _make_pair(rng, roles, mode, config)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('Distinct input sampling exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test():
    """Bounded 42-seed diversity, ablation, metamorphic and rejection checks."""
    puzzles, features, shapes, normalized = set(), [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        key = tuple(_key(p['input']) for split in ('train', 'test') for p in puzzle[split])
        puzzles.add(key)
        if seed in (0, 1, 41):
            again = generate_puzzle(seed)
            assert key == tuple(_key(p['input']) for split in ('train', 'test') for p in again[split])
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                info = _parse_input(inp)
                features.append((_stats(info['forest']), info['rotation'], info['bottom'], info['roles'],
                                 tuple(info['lanes']), _signature(info['forest'], False)))
                shapes.add((inp.shape, out.shape))
                roles = {}
                normalized.add((out.shape, tuple(roles.setdefault(int(v), len(roles)) for v in out.flat)))
        # A whole-input rotation carries the key and all payloads together.
        pair = puzzle['test'][0]
        for turn in (1, 2, 3):
            assert np.array_equal(solve(np.rot90(pair['input'], turn)), pair['output'])
        permutation = np.random.default_rng(seed+5000).permutation(10)
        assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
    assert len(puzzles) == 42 and len(shapes) >= 80 and len(normalized) >= 80
    assert len({f[1:3] for f in features}) == 8
    assert {f[3][0] for f in features} == set(range(10))
    assert len({f[0][:3] for f in features}) >= 12
    assert len({f[4] for f in features}) >= 30
    assert len({f[5] for f in features}) >= 70
    assert {f[0][3] for f in features} == {1, 2, 3}
    assert max(f[0][4] for f in features) >= 4
    # Two equal-sized contours with different content must have different output.
    a = np.zeros((15, 21), dtype=np.int8)
    a[0, :] = 1; a[0, 0] = a[0, -1] = 3; a[1:, 0] = 2
    a[2, 2:18] = a[12, 2:18] = 8; a[2:13, 2] = a[2:13, 17] = 8
    a[5:7, 5:8] = [[4, 5, 6], [6, 4, 5]]
    b = a.copy(); b[9, 6] = 7
    assert solve(a).shape != solve(b).shape
    # Proper rotation and reflection are visibly different operations.
    mirrored = a[::-1].copy()
    assert not np.array_equal(solve(mirrored), solve(a))
    # Redraw fixed logical forests with independent blobs, contours and gaps.
    # Different metric realizations must retain exactly the same transcript.
    redrawn = 0
    rng = np.random.default_rng(1708)
    for _ in range(6):
        forest = _sample_tree(rng, np.array([4, 5, 6, 7, 8, 9]), 'scope_witness')
        expected = _canonical(forest, 0)
        drawings = set()
        for _ in range(5):
            source = _draw(forest, 0, rng, True, False, [2], True)
            if max(source.shape) > 27:
                continue
            grid = np.zeros((source.shape[0]+3, source.shape[1]+3), dtype=np.int8)
            grid[2:-1, 2:-1] = source
            grid[0, :] = 1; grid[0, 0] = grid[0, -1] = 3; grid[1:, 0] = 2
            assert np.array_equal(solve(grid), expected)
            drawings.add(_key(grid))
        redrawn += len(drawings) >= 2
    assert redrawn == 6
    for train, test in ((2, 1), (10, 4), (6, 3)):
        validate_generated_puzzle(generate_puzzle(1200+train, train, test))
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid counts accepted')
    damaged = generate_puzzle(310)
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('Wrong output accepted')
    broken_key = a.copy(); broken_key[0, -1] = 1
    try:
        solve(broken_key)
    except AssertionError:
        pass
    else:
        raise AssertionError('Missing fast-arm echo accepted')
    return {'seeds': 42, 'pairs': len(features), 'distinct_puzzles': len(puzzles),
            'shape_pairs': len(shapes), 'normalized_outputs': len(normalized),
            'count_combinations': len({f[0][:3] for f in features}),
            'lane_structures': len({f[4] for f in features}),
            'tree_length_structures': len({f[5] for f in features}),
            'forests_with_multiple_metric_realizations': redrawn,
            'note': 'Author-owned semantic checks; no independent reference solver.'}
