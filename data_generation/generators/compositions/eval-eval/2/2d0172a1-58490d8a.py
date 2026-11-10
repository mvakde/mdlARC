"""Selective census chambers: color ownership through nested boundaries.

Standard library + NumPy only. No evidence files, latent state, or disk writes.
"""
from collections import deque
import hashlib
import json
import numpy as np

_STEPS4 = ((-1, 0), (0, -1), (0, 1), (1, 0))
_STEPS8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
_DIAMOND = ((0, 1), (1, 0), (1, 2), (2, 1))
_SQUARE = ((0, 0), (0, 1), (1, 0), (1, 1))
_DOT = ((0, 0),)
_X = ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2))
_HOST_SEEDS = (0, 361680857, 413533327, 829726607, 174460460, 704165343,
               438212679, 791823450, 638309824, 8206595, 433013017, 214969590,
               490577155, 111833014, 246001830, 384955380, 489168503, 792165964,
               897460579, 495971338, 712239339, 204563757, 633195394, 128041606,
               78285078, 838443718, 73369410, 554961183, 397007982, 508946375,
               26630404, 359552074, 485744203, 466429497, 820239721, 263633121,
               961281074, 918703819, 706407833, 562182569, 598143308, 542686799)


class _Retry(Exception):
    pass


def _components(mask, diagonal=True):
    h, w = mask.shape
    seen = np.zeros((h, w), bool)
    result = []
    steps = _STEPS8 if diagonal else _STEPS4
    for rr, cc in np.argwhere(mask):
        r, c = int(rr), int(cc)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in steps:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(cells)
    return result


def _hole(mask):
    """Enclosed space of this boundary alone, ignoring all other drawing."""
    padded = np.pad(mask, 1)
    h, w = padded.shape
    seen = np.zeros((h, w), bool)
    seen[0, 0] = True
    queue = deque([(0, 0)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in _STEPS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not padded[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                queue.append((rr, cc))
    return (~padded & ~seen)[1:-1, 1:-1]


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), 'grid dimensions'
    assert np.issubdtype(a.dtype, np.integer), 'integer cells required'
    assert np.all((a >= 0) & (a <= 9)), 'color range'
    return a.astype(np.int8, copy=False)


def _key(a):
    return tuple(a.shape), np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _box(cells):
    a = np.asarray(cells)
    return (int(a[:, 0].min()), int(a[:, 1].min()),
            int(a[:, 0].max()) + 1, int(a[:, 1].max()) + 1)


def _rows(boxes):
    """Transitive row overlap; touching intervals do not overlap."""
    rows, end = [], -1
    for i in sorted(range(len(boxes)), key=lambda j: (boxes[j][0], boxes[j][1])):
        top, left, bottom, right = boxes[i]
        if not rows or top >= end:
            rows.append([])
            end = bottom
        else:
            end = max(end, bottom)
        rows[-1].append(i)
    for row in rows:
        row.sort(key=lambda j: boxes[j][1])
        # Siblings in a row must have separated horizontal projections.
        for i, j in zip(row, row[1:]):
            assert boxes[i][3] < boxes[j][1], 'ambiguous sibling row'
    return rows


def _parse(input_grid, background=None):
    a = _grid(input_grid)
    bg = int(a[0, 0]) if background is None else int(background)
    if background is None:
        assert all(np.all(e == bg) for e in (a[0], a[-1], a[:, 0], a[:, -1])), 'blank border'
    walls, objects = [], []
    owner = np.full(a.shape, -1, int)
    for serial, cells in enumerate(_components(a != bg)):
        rr, cc = zip(*cells)
        owner[rr, cc] = serial
        values, counts = np.unique(a[rr, cc], return_counts=True)
        mask = np.zeros(a.shape, bool)
        mask[rr, cc] = True
        record = dict(cells=cells, mask=mask, box=_box(cells))
        if len(values) == 1:
            record['color'] = int(values[0])
            objects.append(record)
        else:
            assert len(values) == 2 and sorted(map(int, counts))[0] == 1, 'one badge per wall'
            assert len(_components(mask, False)) == 1, 'wall must be 4-connected'
            badge = int(values[np.argmin(counts)])
            wall = int(values[np.argmax(counts)])
            hole = _hole(mask)
            assert len(_components(hole, False)) == 1, 'wall must enclose one region'
            record.update(badge=badge, wall=wall, hole=hole, parent=None, children=[])
            walls.append(record)
    assert walls, 'no chambers'
    wall_color = walls[0]['wall']
    assert all(w['wall'] == wall_color for w in walls), 'one boundary color'
    assert wall_color != bg and all(o['color'] != wall_color for o in objects)
    assert all(w['badge'] not in (bg, wall_color) for w in walls)
    # Distinct foreground components must have a blank Chebyshev halo.
    # Same-color diagonal joins are already inside one component above.
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        r0, r1 = max(0, -dr), min(a.shape[0], a.shape[0] - dr)
        c0, c1 = max(0, -dc), min(a.shape[1], a.shape[1] - dc)
        p, q = owner[r0:r1, c0:c1], owner[r0+dr:r1+dr, c0+dc:c1+dc]
        assert not np.any((p >= 0) & (q >= 0) & (p != q))

    def parent_of(record, own=None):
        candidates = []
        rr, cc = zip(*record['cells'])
        for j, wall in enumerate(walls):
            if j == own:
                continue
            inside = wall['hole'][rr, cc]
            assert inside.all() or not inside.any(), 'crossing wall'
            if inside.all():
                candidates.append(j)
        return min(candidates, key=lambda j: int(walls[j]['hole'].sum())) if candidates else None

    roots = []
    for i, w in enumerate(walls):
        w['parent'] = parent_of(w, i)
    for i, w in enumerate(walls):
        if w['parent'] is None:
            roots.append(i)
        else:
            walls[w['parent']]['children'].append(i)
    for obj in objects:
        obj['parent'] = parent_of(obj)
        obj['owner'] = None
        j = obj['parent']
        while j is not None:
            if walls[j]['badge'] == obj['color']:
                obj['owner'] = j
                break
            j = walls[j]['parent']
    # This is the specified spatial domain, not a bounding-box containment test.
    _rows([walls[i]['box'] for i in roots])
    for w in walls:
        _rows([walls[i]['box'] for i in w['children']])
    return dict(background=bg, wall=wall_color, walls=walls, objects=objects, roots=roots)


def _counts(info, mode='correct'):
    walls = info['walls']
    counts = [0] * len(walls)
    for obj in info['objects']:
        if mode == 'global':
            for i, w in enumerate(walls):
                counts[i] += w['badge'] == obj['color']
        elif mode == 'direct':
            i = obj['parent']
            if i is not None and walls[i]['badge'] == obj['color']:
                counts[i] += 1
        elif mode == 'all_ancestors':
            i = obj['parent']
            while i is not None:
                counts[i] += walls[i]['badge'] == obj['color']
                i = walls[i]['parent']
        elif mode == 'ignore_color':
            if obj['parent'] is not None:
                counts[obj['parent']] += 1
        elif obj['owner'] is not None:
            amount = 1
            if mode == 'area':
                amount = len(obj['cells'])
            elif mode == 'four':
                amount = len(_components(obj['mask'], False))
            counts[obj['owner']] += amount
    return counts


def _compose(tiles, boxes, bg):
    if not tiles:
        return None
    rows = _rows(boxes)
    row_heights = [max(tiles[i].shape[0] for i in row) for row in rows]
    row_widths = [sum(tiles[i].shape[1] for i in row) + len(row) - 1 for row in rows]
    result = np.full((sum(row_heights) + len(rows) - 1, max(row_widths)), bg, np.int8)
    y = 0
    for row, height in zip(rows, row_heights):
        x = 0
        for i in row:
            tile = tiles[i]
            result[y:y+tile.shape[0], x:x+tile.shape[1]] = tile
            x += tile.shape[1] + 1
        y += height + 1
    return result


def _frame(badge, count, children, bg, wall):
    h = 5 if children is None else children.shape[0] + 6
    w = max(5, 2 * count + 3, 0 if children is None else children.shape[1] + 4)
    out = np.full((h, w), bg, np.int8)
    out[0, :] = out[-1, :] = wall
    out[:, 0] = out[:, -1] = wall
    out[0, 0] = badge
    out[2, 2:2+2*count:2] = badge
    if children is not None:
        out[4:4+children.shape[0], 2:2+children.shape[1]] = children
    return out


def _answer(info):
    counts = _counts(info)
    walls, bg, wall_color = info['walls'], info['background'], info['wall']

    def draw(i):
        ids = walls[i]['children']
        child = _compose([draw(j) for j in ids], [walls[j]['box'] for j in ids], bg)
        return _frame(walls[i]['badge'], counts[i], child, bg, wall_color)

    ids = info['roots']
    return _compose([draw(i) for i in ids], [walls[i]['box'] for i in ids], bg)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every boundary, badge, owner, tally, and placement from input only."""
    return _answer(_parse(input_grid))


def _normal(cells):
    y, x = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-y, c-x) for r, c in cells))


def _shape(rng, small=False):
    motifs = (_DOT, _DIAMOND, _SQUARE, _X,
              ((0, 0), (1, 1)), ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)))
    if rng.random() < .55:
        cells = motifs[int(rng.integers(len(motifs)))]
    else:
        side = 3 if small else int(rng.integers(3, 5))
        target = int(rng.integers(1, min(10, side * side) + 1))
        occupied = {(side // 2, side // 2)}
        while len(occupied) < target:
            frontier = sorted({(r+dr, c+dc) for r, c in occupied for dr, dc in _STEPS8
                               if 0 <= r+dr < side and 0 <= c+dc < side} - occupied)
            occupied.add(frontier[int(rng.integers(len(frontier)))])
        cells = tuple(occupied)
    if rng.integers(2):
        cells = tuple((r, -c) for r, c in cells)
    for _ in range(int(rng.integers(4))):
        cells = tuple((-c, r) for r, c in cells)
    return _normal(cells)


def _node(badge, objects=(), children=()):
    # An object's optional explicit shape is a training witness, not a lookup.
    return dict(badge=badge, objects=list(objects), children=list(children))


def _plans(rng, kind, colors):
    a, b, c = colors[:3]
    if kind == 'passage':
        inner = _node(b, [(a, _DIAMOND), (a, _SQUARE), (b, None)])
        roots = [_node(a, [(a, _DOT), (b, None)], [inner])]
        exterior = [(a, None)]
    elif kind == 'shadow':
        matched = _node(a, [(a, None), (a, None)])
        empty = _node(b, [(a, None)])
        roots = [_node(a, [(a, None), (c, None)], [matched, empty])]
        exterior = [(a, None)]
    else:
        total = int(rng.integers(1, 5)) if kind == 'ordinary' else int(rng.integers(3, 6))
        roots = [_node(int(rng.choice(colors[:int(rng.integers(1, 5))])))]
        nodes = [roots[0]]
        depths = [1]
        if kind == 'deep':
            roots[0]['badge'] = a
            mid = _node(b)
            leaf = _node(a if rng.random() < .7 else c)
            roots[0]['children'].append(mid)
            mid['children'].append(leaf)
            nodes += [mid, leaf]
            depths += [2, 3]
        while len(nodes) < total:
            new = _node(int(rng.choice(colors[:int(rng.integers(1, 5))])))
            if kind == 'ordinary' and rng.random() < .20 and len(roots) < 2:
                roots.append(new)
                depth = 1
            else:
                cap = 2 if kind == 'ordinary' else 3
                eligible = [i for i, d in enumerate(depths) if d < cap and len(nodes[i]['children']) < 3]
                if not eligible:
                    break
                parent = eligible[int(rng.integers(len(eligible)))]
                nodes[parent]['children'].append(new)
                depth = depths[parent] + 1
            nodes.append(new)
            depths.append(depth)
        ancestors = {}

        def collect(node, path):
            ancestors[id(node)] = [node['badge']] + path
            for child in node['children']:
                collect(child, [node['badge']] + path)

        for root in roots:
            collect(root, [])
        concentrated = rng.random() < .55
        for i, node in enumerate(nodes):
            cap = 6 if kind == 'ordinary' and total <= 2 else 4 if total == 3 else 3
            if node['children']:
                cap = max(2, cap - 2)
            n = int(rng.integers(0, cap + 1))
            # Each scene has visible claim evidence, but individual offices can be empty.
            if i == len(nodes) - 1:
                n = max(1, n)
            for j in range(n):
                if concentrated and rng.random() < .75:
                    color = int(rng.choice(ancestors[id(node)]))
                else:
                    color = node['badge'] if rng.random() < .35 else int(rng.choice(colors[:5]))
                node['objects'].append((color, None))
        if kind == 'deep':
            # A new depth combines passage and repeated-badge shadowing.
            nodes[2]['objects'].append((a, None))
            nodes[1]['objects'].append((a, None))
        exterior = [(int(rng.choice(colors[:5])), None) for _ in range(int(rng.integers(3)))]
    return roots, exterior


def _pack(tiles, rng, compact=False):
    """Bounded shelf search; no retry payloads or candidate histories retained."""
    if not tiles:
        return [], (1, 1)
    best = None
    target = float(rng.uniform(.65, 1.55))
    for _ in range(24):
        order = list(map(int, rng.permutation(len(tiles))))
        rows, row = [], []
        for i in order:
            row.append(i)
            if rng.random() < .43:
                rows.append(row)
                row = []
        if row:
            rows.append(row)
        gap = 1 if compact else int(rng.integers(1, 4))
        widths = [sum(tiles[i].shape[1] for i in row) + gap * (len(row)-1) for row in rows]
        heights = [max(tiles[i].shape[0] for i in row) for row in rows]
        h, w = sum(heights) + gap * (len(rows)-1), max(widths)
        if max(h, w) > 26:
            continue
        score = h * w + 10 * abs(w - target * h) + float(rng.uniform(0, 35))
        if best is None or score < best[0]:
            positions = [None] * len(tiles)
            y = 0
            for row, height in zip(rows, heights):
                x = 0
                for i in row:
                    # Top alignment makes row grouping unambiguous.
                    positions[i] = (y, x)
                    x += tiles[i].shape[1] + gap
                y += height + gap
            best = (score, positions, (h, w))
    if best is None:
        raise _Retry('packing does not fit')
    return best[1], best[2]


def _outline(h, w, pad, rng, compact):
    top, bottom, left, right = pad
    y0, y1, x0, x1 = top - 2, h - bottom + 1, left - 2, w - right + 1
    mask = np.zeros((h, w), bool)
    style = int(rng.integers(5))

    def profile(n, cap):
        if style == 0 or cap <= 0:
            return np.zeros(n, int)
        knots = rng.integers(0, cap + 1, min(n, 5))
        return np.rint(np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(knots)), knots)).astype(int)

    upper, lower = profile(w, top-2), h-1-profile(w, bottom-2)
    lhs, rhs = profile(h, left-2), w-1-profile(h, right-2)
    path = [(int(upper[x]), x) for x in range(x0, x1+1)]
    path += [(y0, x1), (y0, int(rhs[y0]))]
    path += [(y, int(rhs[y])) for y in range(y0, y1+1)]
    path += [(int(lower[x1]), int(rhs[y1])), (int(lower[x1]), x1)]
    path += [(int(lower[x]), x) for x in range(x1, x0-1, -1)]
    path += [(y1, x0), (y1, int(lhs[y1]))]
    path += [(y, int(lhs[y])) for y in range(y1, y0-1, -1)]
    path += [(int(upper[x0]), int(lhs[y0])), path[0]]
    for (r, c), (rr, cc) in zip(path, path[1:]):
        mask[min(r, rr):max(r, rr)+1, c] = True
        mask[rr, min(c, cc):max(c, cc)+1] = True
    if len(_components(mask, False)) != 1 or len(_components(_hole(mask), False)) != 1:
        raise _Retry('deformation pinched a wall')
    return mask


def _draw_plans(roots, exterior, rng, bg, wall, compact=False):
    serial = [0]
    records = []
    local_objects = []

    def blob(color, given, parent):
        shape = _shape(rng, compact) if given is None else given
        tile = np.full((1+max(r for r, c in shape), 1+max(c for r, c in shape)), bg, np.int8)
        for r, c in shape:
            tile[r, c] = color
        local_objects.append((parent, color, shape))
        return tile

    def draw(node, parent):
        own = serial[0]
        serial[0] += 1
        record = dict(badge=node['badge'], parent=parent, children=[], boxes=[])
        records.append(record)
        child_pairs = [draw(c, own) for c in node['children']]
        children = [pair[0] for pair in child_pairs]
        record['children'] = [pair[1] for pair in child_pairs]
        tiles = children + [blob(color, given, own) for color, given in node['objects']]
        positions, (ch, cw) = _pack(tiles, rng, compact)
        # Compact retries retain one genuinely stepped edge when possible.
        pad = [2 if compact else int(rng.integers(2, 5)) for _ in range(4)]
        if compact:
            pad[int(rng.integers(4))] = 3
        top, bottom, left, right = pad
        h, w = ch + top + bottom, cw + left + right
        if max(h, w) > 28:
            raise _Retry('wall exceeds canvas')
        mask = _outline(h, w, pad, rng, compact)
        grid = np.full((h, w), bg, np.int8)
        grid[mask] = wall
        points = np.argwhere(mask)
        badge_at = points[int(rng.integers(len(points)))]
        grid[tuple(badge_at)] = node['badge']
        for i, (tile, (y, x)) in enumerate(zip(tiles, positions)):
            y, x = y + top, x + left
            grid[y:y+tile.shape[0], x:x+tile.shape[1]] = tile
            if i < len(children):
                record['boxes'].append((y, x, y+tile.shape[0], x+tile.shape[1]))
        r0, c0, r1, c1 = _box(points)
        record['boxes'] = [(y-r0, x-c0, yy-r0, xx-c0) for y, x, yy, xx in record['boxes']]
        return grid[r0:r1, c0:c1].copy(), own

    root_pairs = [draw(node, None) for node in roots]
    root_tiles = [pair[0] for pair in root_pairs]
    ids = [pair[1] for pair in root_pairs]
    tiles = root_tiles + [blob(c, shape, None) for c, shape in exterior]
    positions, (h, w) = _pack(tiles, rng, compact)
    top, bottom, left, right = [int(rng.integers(1, 4)) for _ in range(4)]
    if h + top + bottom > 30 or w + left + right > 30:
        top = bottom = left = right = 1
    if h + top + bottom > 30 or w + left + right > 30:
        raise _Retry('no blank border fits')
    grid = np.full((h+top+bottom, w+left+right), bg, np.int8)
    root_boxes = []
    for i, (tile, (y, x)) in enumerate(zip(tiles, positions)):
        y, x = y+top, x+left
        grid[y:y+tile.shape[0], x:x+tile.shape[1]] = tile
        if i < len(root_tiles):
            root_boxes.append((y, x, y+tile.shape[0], x+tile.shape[1]))
    # Ground truth ownership comes from the construction tree, not segmentation.
    counts = [0] * len(records)
    for parent, color, shape in local_objects:
        i = parent
        while i is not None:
            if records[i]['badge'] == color:
                counts[i] += 1
                break
            i = records[i]['parent']

    def expected(i):
        record = records[i]
        contents = _compose([expected(j) for j in record['children']], record['boxes'], bg)
        return _frame(record['badge'], counts[i], contents, bg, wall)

    output = _compose([expected(i) for i in ids], root_boxes, bg)
    if max(output.shape) > 30:
        raise _Retry('answer exceeds grid bound')
    return dict(input=grid, output=output)


def _make_pair(rng, kind, palette):
    bg, wall, *colors = palette
    for attempt in range(96):
        # Retry the geometry of one sampled semantic plan before resampling.
        if attempt % 8 == 0:
            roots, exterior = _plans(rng, kind, colors)
        try:
            pair = _draw_plans(roots, exterior, rng, bg, wall, compact=attempt % 8 >= 3)
            assert np.array_equal(solve(pair['input']), pair['output']), 'construction/parser discrepancy'
            return pair
        except _Retry:
            continue
    # Proven-size fallback resamples a small scene under exactly the same rule.
    # It retains both mandatory witnesses, or a depth-three transfer for tests.
    a, b, c = colors[:3]
    if kind == 'passage':
        roots = [_node(a, [(a, _DOT), (b, _DOT)],
                       [_node(b, [(a, _DIAMOND), (a, _SQUARE), (b, _DOT)])])]
        exterior = [(a, _DOT)]
    elif kind == 'shadow':
        roots = [_node(a, [(a, _DOT), (c, _DOT)],
                       [_node(a, [(a, _DOT), (a, _DOT)]), _node(b, [(a, _DOT)])])]
        exterior = [(a, _DOT)]
    elif kind == 'deep':
        roots = [_node(a, [(a, _DOT)], [_node(b, [(a, _DOT)], [_node(a, [(a, _DOT)])])])]
        exterior = []
    else:
        roots, exterior = [_node(a, [(a, _DOT)] * int(rng.integers(1, 7)))], []
    pair = _draw_plans(roots, exterior, rng, bg, wall, compact=True)
    assert np.array_equal(solve(pair['input']), pair['output'])
    return pair


def _check_counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(name + ' must be an integer in its legal range')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    stable = rng.random() < .2
    base = list(map(int, rng.permutation(10)))
    result, seen = {'train': [], 'test': []}, set()
    for split, n in (('train', num_train), ('test', num_test)):
        kinds = ['passage', 'shadow'] + ['ordinary'] * (n-2) if split == 'train' else ['deep'] * n
        for kind in kinds:
            for _ in range(16):
                palette = base if stable else list(map(int, rng.permutation(10)))
                pair = _make_pair(rng, kind, palette)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('distinct input search exhausted')
            seen.add(_key(pair['input']))
            result[split].append(pair)
        rng.shuffle(result[split])
    return result


def _features(info):
    walls, objects = info['walls'], info['objects']
    counts = _counts(info)
    depth = []
    for i, w in enumerate(walls):
        d, parent = 1, w['parent']
        while parent is not None:
            d += 1
            parent = walls[parent]['parent']
        depth.append(d)
    passage = any(o['owner'] is not None and o['owner'] != o['parent'] for o in objects)
    shadow = False
    for o in objects:
        i = o['owner']
        if i is not None:
            i = walls[i]['parent']
            while i is not None:
                shadow |= walls[i]['badge'] == o['color']
                i = walls[i]['parent']
    mixed_area = any(len({len(o['cells']) for o in objects if o['owner'] == i}) > 1 for i in range(len(walls)))
    diagonal = _counts(info, 'four') != counts
    nonrect = sum(len(w['cells']) != 2 * ((w['box'][2]-w['box'][0]) + (w['box'][3]-w['box'][1])) - 4 for w in walls)
    layouts = []
    for ids in [info['roots']] + [w['children'] for w in walls]:
        if len(ids) > 1:
            rows = _rows([walls[i]['box'] for i in ids])
            layouts.append(tuple(len(row) for row in rows))
    return dict(depth=max(depth), chambers=len(walls), objects=len(objects),
                counts=tuple(counts), passage=passage, shadow=shadow,
                diagonal=diagonal, mixed_area=mixed_area, nonrect=nonrect,
                zero=0 in counts, ignored=any(o['owner'] is None for o in objects),
                layouts=tuple(layouts))


def _audit_output(info, actual):
    """Check geometry, census cells, and child placement in the returned output."""
    bg, wall = info['background'], info['wall']
    parsed = _parse(actual, bg)
    assert parsed['background'] == bg and parsed['wall'] == wall
    assert len(parsed['walls']) == len(info['walls'])
    # Output has canonical NW badges, true rectangular perimeters, and point leaves.
    for w in parsed['walls']:
        y, x, yy, xx = w['box']
        assert actual[y, x] == w['badge']
        expected_area = 2 * ((yy-y) + (xx-x)) - 4
        assert len(w['cells']) == expected_area
        assert all(np.all(e != bg) for e in (actual[y, x:xx], actual[yy-1, x:xx],
                                             actual[y:yy, x], actual[y:yy, xx-1]))
    assert all(len(o['cells']) == 1 for o in parsed['objects'])
    # Output row (2,2+2k) per frame is its census, not a second raw scene census.
    counts = _counts(info)

    def signature(meta, ids, census):
        result = []
        boxes = [meta['walls'][i]['box'] for i in ids]
        for row in _rows(boxes):
            entries = []
            for j in row:
                i = ids[j]
                w = meta['walls'][i]
                if census is None:
                    y, x, yy, xx = w['box']
                    row_values = actual[y+2, x+2:xx-1]
                    points = np.flatnonzero(row_values != bg)
                    assert all(int(row_values[p]) == w['badge'] and p == 2*k for k, p in enumerate(points))
                    n = len(points)
                else:
                    n = census[i]
                entries.append((w['badge'], n, signature(meta, w['children'], census)))
            result.append(tuple(entries))
        return tuple(result)

    assert signature(info, info['roots'], counts) == signature(parsed, parsed['roots'], None)
    assert np.array_equal(actual, _answer(info)), 'canonical crop, moat, or spacing'


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, train_features, test_features = set(), [], []
    failures = {mode: False for mode in ('global', 'direct', 'all_ancestors', 'ignore_color', 'area', 'four')}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            info = _parse(inp)
            assert np.array_equal(out, _answer(info)), 'wrong answer'
            _audit_output(info, out)
            f = _features(info)
            (train_features if split == 'train' else test_features).append(f)
            if split == 'train':
                counts = _counts(info)
                for mode in failures:
                    failures[mode] |= _counts(info, mode) != counts
    assert any(f['passage'] and f['diagonal'] and f['mixed_area'] for f in train_features), 'passage/count witness'
    assert any(f['shadow'] and f['zero'] for f in train_features), 'shadow/zero witness'
    assert all(failures.values()), 'training admits a tested shortcut'
    assert all(f['depth'] <= 2 for f in train_features)
    assert all(f['depth'] == 3 for f in test_features), 'depth recombination shift'
    return True


def self_test():
    """Bounded 42-seed semantic/variation checks, count boundaries, and mutations."""
    shapes, structures, topologies, palettes, layouts, glyphs, tallies = set(), set(), set(), set(), set(), set(), set()
    signatures = set()
    totals = dict(pairs=0, passage=0, shadow=0, nonrectangular_walls=0, zero_pairs=0)
    first = None
    for seed in _HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        if first is None:
            first = puzzle
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                digest.update(inp.tobytes())
                digest.update(out.tobytes())
                info = _parse(inp)
                f = _features(info)
                totals['pairs'] += 1
                totals['passage'] += f['passage']
                totals['shadow'] += f['shadow']
                totals['nonrectangular_walls'] += f['nonrect']
                totals['zero_pairs'] += f['zero']
                shapes.add(inp.shape)
                palettes.add((info['background'], info['wall']))
                layouts.update(f['layouts'])
                tallies.update(f['counts'])
                def topo(i):
                    return tuple(sorted(topo(j) for j in info['walls'][i]['children']))
                topologies.add(tuple(sorted(topo(i) for i in info['roots'])))
                mapping = {}
                norm = tuple(mapping.setdefault(int(c), len(mapping)) for c in out.flat)
                structures.add((out.shape, norm))
                for o in info['objects']:
                    glyphs.add(_normal(o['cells']))
        signatures.add(digest.digest())
        # Palette permutation and translation must preserve visible semantics.
        pair = puzzle['test'][0]
        permutation = np.random.default_rng(seed + 701).permutation(10)
        assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
        a = pair['input']
        if max(a.shape) <= 28:
            moved = np.pad(a, 1, constant_values=int(a[0, 0]))
            assert np.array_equal(solve(moved), pair['output'])
    assert len(signatures) == 42
    assert len(shapes) >= 50 and len(structures) >= 65
    assert len(topologies) >= 10 and len(palettes) >= 40 and len(glyphs) >= 50
    assert (2,) in layouts and (1, 1) in layouts
    assert {0, 1, 2, 3} <= tallies
    assert totals['nonrectangular_walls'] >= 80
    assert totals['passage'] >= 42 and totals['shadow'] >= 42 and totals['zero_pairs'] >= 42
    # Determinism, legal count boundaries, and invalid count types.
    repeat = generate_puzzle(_HOST_SEEDS[0])
    assert all(_key(p[k]) == _key(q[k]) for s in ('train', 'test')
               for p, q in zip(first[s], repeat[s]) for k in ('input', 'output'))
    for n, m in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(902 + 10*n + m, n, m)
        assert len(puzzle['train']) == n and len(puzzle['test']) == m
        validate_generated_puzzle(puzzle)
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4., 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Same abstract chain, independently sampled shapes, distances, and walls.
    rng = np.random.default_rng(87102)
    roots = [_node(2, [(2, None), (4, None)], [_node(3, [(2, None), (3, None)])])]
    answers, inputs = set(), set()
    for _ in range(12):
        try:
            pair = _draw_plans(roots, [], rng, 0, 1, compact=True)
        except _Retry:
            continue
        assert np.array_equal(solve(pair['input']), pair['output'])
        answers.add(_key(pair['output']))
        inputs.add(_key(pair['input']))
    assert len(answers) == 1 and len(inputs) >= 5
    # Changed tally, missing zero frame, and extra crop margin must be detected.
    for mode in range(3):
        bad = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in first[s]] for s in ('train', 'test')}
        pair = bad['test'][0]
        info = _parse(pair['input'])
        bg = info['background']
        if mode == 0:
            pair['output'][2, 2] = (int(pair['output'][2, 2]) + 1) % 10
        elif mode == 1:
            w = _parse(pair['output'], bg)['walls'][-1]
            for r, c in w['cells']:
                pair['output'][r, c] = bg
        else:
            pair['output'] = np.pad(pair['output'], 1, constant_values=bg)
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError('corrupted output accepted')
    return dict(seeds=42, distinct_puzzles=len(signatures), input_shapes=len(shapes),
                normalized_outputs=len(structures), containment_topologies=len(topologies),
                ordered_background_wall_palettes=len(palettes), glyph_geometries=len(glyphs),
                sibling_layouts=sorted(layouts), tally_values=sorted(tallies), **totals)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
