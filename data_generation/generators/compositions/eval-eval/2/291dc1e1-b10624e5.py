"""Socketed ribbon kits: ordered, articulated, scaled and reflected assembly.

Only standard-library modules and NumPy are used. solve() observes input pixels
alone. Construction uses array pasting; prediction uses individual logical cells.
"""
from __future__ import annotations

import copy
import json
import numpy as np


_CORNERS = {(0, 0): 0, (1, 0): 1, (1, 1): 2, (0, 1): 3}
_CORNER_POINTS = ((0, 0), (1, 0), (1, 1), (0, 1))


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    return int(value)


def _counts(num_train, num_test):
    a, b = _integer(num_train, 'num_train'), _integer(num_test, 'num_test')
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError('need 2..10 training, 1..4 tests, and num_train > num_test')
    return a, b


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), 'bad grid dimensions'
    assert np.issubdtype(a.dtype, np.integer), 'grid must contain integers'
    assert np.all((0 <= a) & (a <= 9)), 'bad colors'
    return a.astype(np.int64, copy=True)


def _key(grid):
    candidates = []
    for k in range(4):
        g = np.rot90(grid, k)
        if min(g.shape) < 5:
            continue
        for down in (True, False):
            row = 0 if down else g.shape[0]-1
            other = g.shape[0]-1-row
            origin, fast, slow, bg = map(int, (g[row, 0], g[row, 2], g[other, 0], g[other, -1]))
            arm = np.delete(g[:, 0], row)
            if (len({origin, fast, slow, bg}) == 4 and g[row, 1] == origin
                    and np.all(g[row, 2:] == fast) and np.all(arm == slow)):
                candidates.append((g.copy(), k, down, row, origin, fast, slow, bg))
    assert len(candidates) == 1, 'expected one notched L-key'
    return candidates[0]


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    boxes = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, cells = [first], [first]
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    cells.append(q)
                    stack.append(q)
        r0, c0 = np.min(cells, axis=0).tolist()
        r1, c1 = (np.max(cells, axis=0)+1).tolist()
        assert len(cells) == (r1-r0)*(c1-c0), 'kit pieces must be full rectangles'
        boxes.append((r0, r1, c0, c1))
    return boxes


def _logical(block, unit):
    h, w = block.shape
    assert h % unit == w % unit == 0
    tiles = block.reshape(h//unit, unit, w//unit, unit)
    lo, hi = tiles.min(axis=(1, 3)), tiles.max(axis=(1, 3))
    assert np.array_equal(lo, hi), 'nonuniform lattice pixels'
    return lo.copy()


def _socket(square, marker, body=None):
    side = square.shape[0]
    assert square.shape == (side, side) and side >= 2 and side % 2 == 0, 'bad socket size'
    quarters = _logical(square, side//2)
    marks = np.argwhere(quarters == marker)
    assert len(marks) == 1, 'socket needs exactly one marked quarter'
    remaining = quarters[quarters != marker]
    assert np.all(remaining == remaining[0]), 'socket body must be uniform'
    actual_body = int(remaining[0])
    if body is not None:
        assert actual_body == body, 'mismatching socket body'
    return actual_body, _CORNERS[tuple(map(int, marks[0]))], side//2


def _decode(input_grid):
    original = _grid(input_grid)
    g, rotation, down, key_row, marker, fast, slow, bg = _key(original)
    h, w = g.shape
    body_rows = np.arange(1, h) if down else np.arange(h-1)
    wall_candidates = []
    for c in range(2, w-3):
        color = int(g[body_rows[0], c])
        if color not in (bg, marker, fast, slow) and np.all(g[body_rows, c] == color):
            wall_candidates.append((c, color))
    assert len(wall_candidates) == 1, 'expected one full library wall'
    divider, wall = wall_candidates[0]
    first, last = int(body_rows[0]), int(body_rows[-1])+1
    bank = g[first:last, 1:divider]
    boxes = [(a+first, b+first, c+1, d+1) for a, b, c, d in _components(bank != bg)]
    assert boxes, 'empty kit'
    thicknesses = {b-a for a, b, c, d in boxes}
    assert len(thicknesses) == 1 and next(iter(thicknesses)) % 2 == 0
    unit = next(iter(thicknesses))//2
    assert unit >= 1
    lanes = {}
    socket_body = None
    for a, b, c, d in boxes:
        assert first < a < b < last and 1 < c < d < divider, 'library needs background margins'
        assert d-c >= 6*unit and (d-c) % (2*unit) == 0
        piece = _logical(g[a:b, c:d], unit)
        body, head, _ = _socket(piece[:, :2], marker)
        assert head == 0, 'entry corner must be TL after proper rotation'
        if socket_body is None:
            socket_body = body
        assert body == socket_body and body not in (bg, marker, fast, slow, wall)
        _socket(piece[:, -2:], marker, body)
        assert not np.any(np.isin(piece[:, 2:-2], (bg, marker, body))), 'reserved interior color'
        lanes.setdefault(a, []).append((c, (a, b, c, d), piece))
    natural = sorted(lanes)
    assert all(b-a > 2*unit for a, b in zip(natural, natural[1:])), 'overlapping lane bands'
    groups, ordered_boxes, blocks = [], [], []
    for r in sorted(lanes, reverse=not down):
        line = sorted(lanes[r], key=lambda item: item[0])
        assert all(second[1][2] > first_item[1][3]
                   for first_item, second in zip(line, line[1:])), 'missing strip gap'
        groups.append(list(range(len(blocks), len(blocks)+len(line))))
        for _, box, block in line:
            ordered_boxes.append(box)
            blocks.append(block)
    separators = [r for r in range(first, last) if np.all(g[r, divider+1:] == wall)]
    edges = [first-1]+separators+[last]
    rooms = []
    for lo, hi in zip(edges, edges[1:]):
        a, b = lo+1, hi
        assert b > a, 'empty target chamber'
        panel = g[a:b, divider+1:]
        foreground = np.argwhere(panel != bg)
        assert len(foreground), 'chamber needs one socket'
        r0, c0 = foreground.min(axis=0).tolist()
        r1, c1 = (foreground.max(axis=0)+1).tolist()
        assert len(foreground) == (r1-r0)*(c1-c0), 'unrelated target pixels'
        assert 0 < r0 < r1 < panel.shape[0] and 0 < c0 < c1 < panel.shape[1], 'root needs a margin'
        _, corner, target_unit = _socket(panel[r0:r1, c0:c1], marker, socket_body)
        rooms.append({'bounds': (a, b, divider+1, w), 'root': (r0, c0),
                      'unit': target_unit, 'corner': corner})
    return {'grid': g, 'rotation': rotation, 'down': down, 'key_row': key_row,
            'marker': marker, 'fast': fast, 'slow': slow, 'background': bg,
            'wall': wall, 'body': socket_body, 'divider': divider, 'unit': unit,
            'blocks': blocks, 'groups': groups, 'boxes': ordered_boxes, 'rooms': rooms}


def _rot_point(r, c, turns):
    for _ in range(turns % 4):
        r, c = 1-c, r
    return r, c


def _exit(block, marker):
    point = np.argwhere(block[:, -2:] == marker)
    assert len(point) == 1
    return _CORNERS[tuple(map(int, point[0]))]


def _chain_cells(blocks, marker, turns='encoded'):
    """Cell-wise docking: the solver never uses construction's array raster."""
    cells = {}
    joint_r = joint_c = orientation = 0
    overwritten = 0
    for block in blocks:
        for r in range(2):
            for c in range(block.shape[1]):
                dr, dc = _rot_point(r, c, orientation)
                point = (joint_r+dr, joint_c+dc)
                value = int(block[r, c])
                overwritten += point in cells and cells[point] != value
                cells[point] = value
        distance = block.shape[1]-2
        dr, dc = ((0, distance), (-distance, 0), (0, -distance), (distance, 0))[orientation]
        joint_r, joint_c = joint_r+dr, joint_c+dc
        delta = _exit(block, marker) if turns == 'encoded' else 0 if turns == 'straight' else 3
        orientation = (orientation+delta) % 4
    return cells, overwritten


def _order(data, alternative):
    groups = data['groups']
    if alternative == 'fast_reverse':
        return [i for group in groups for i in group[::-1]]
    if alternative == 'slow_reverse':
        return [i for group in groups[::-1] for i in group]
    if alternative == 'fast_major':
        return sorted(range(len(data['blocks'])),
                      key=lambda i: (data['boxes'][i][2],
                                     data['boxes'][i][0] if data['down'] else -data['boxes'][i][0]))
    if alternative == 'repeat_first':
        return [0]*len(data['blocks'])
    return list(range(len(data['blocks'])))


def _predict(data, alternative=None):
    g, bg = data['grid'], data['background']
    blocks = [data['blocks'][i] for i in _order(data, alternative)]
    turn_mode = 'straight' if alternative == 'no_turns' else 'clockwise' if alternative == 'all_clockwise' else 'encoded'
    cells, overwritten = _chain_cells(blocks, data['marker'], turn_mode)
    out = g.copy()
    records = []
    for room in data['rooms']:
        a, b, c, d = room['bounds']
        root_r, root_c = room['root']
        actual_unit = room['unit']
        unit = data['unit'] if alternative == 'no_scale' else actual_unit
        corner = 0 if alternative == 'no_reflection' else room['corner']
        vertical = corner in (1, 2)
        horizontal = corner in (2, 3)
        lost = visible_detail = 0
        for (r, col), value in cells.items():
            if alternative == 'rotate_root':
                r, col = _rot_point(r, col, corner)
            else:
                r = 1-r if vertical else r
                col = 1-col if horizontal else col
            top, left = a+root_r+unit*r, c+root_c+unit*col
            r0, r1 = max(a, top), min(b, top+unit)
            c0, c1 = max(c, left), min(d, left+unit)
            area = max(0, r1-r0)*max(0, c1-c0)
            lost += unit*unit-area
            if area:
                out[r0:r1, c0:c1] = value
                if value not in (bg, data['marker'], data['body']):
                    visible_detail += area
        # The original root is the reference frame, and is always foreground.
        rr, cc = a+root_r, c+root_c
        side = 2*actual_unit
        out[rr:rr+side, cc:cc+side] = g[rr:rr+side, cc:cc+side]
        records.append({'lost': lost, 'visible_detail': visible_detail})
    return np.rot90(out, -data['rotation']).copy(), records, overwritten


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the complete scene using only the supplied grid."""
    return _predict(_decode(input_grid))[0]


def _array_chain(blocks, marker, bg):
    """Construction raster: np.rot90, bounding rectangles and whole-array paste."""
    placements = []
    jr = jc = orientation = 0
    for block in blocks:
        length = block.shape[1]
        raster = np.rot90(block, orientation).copy()
        head_offsets = ((0, 0), (length-2, 0), (0, length-2), (0, 0))
        tail_offsets = ((0, length-2), (0, 0), (0, 0), (length-2, 0))
        hr, hc = head_offsets[orientation]
        tr, tc = tail_offsets[orientation]
        placements.append((jr-hr, jc-hc, raster))
        jr, jc = jr+tr-hr, jc+tc-hc
        orientation = (orientation+_exit(block, marker)) % 4
    top = min(r for r, c, array in placements)
    left = min(c for r, c, array in placements)
    bottom = max(r+array.shape[0] for r, c, array in placements)
    right = max(c+array.shape[1] for r, c, array in placements)
    canvas = np.full((bottom-top, right-left), bg, dtype=np.int64)
    for r, c, array in placements:
        canvas[r-top:r-top+array.shape[0], c-left:c-left+array.shape[1]] = array
    return canvas, (-top, -left)


def _frame_raster(raster, offset, unit, corner):
    arr = raster
    row, col = offset
    if corner in (1, 2):
        arr = arr[::-1]
        row = arr.shape[0]-row-2
    if corner in (2, 3):
        arr = arr[:, ::-1]
        col = arr.shape[1]-col-2
    return np.repeat(np.repeat(arr, unit, axis=0), unit, axis=1), (row*unit, col*unit)


def _paste(panel, raster, top, left, bg):
    a, b = max(0, top), min(panel.shape[0], top+raster.shape[0])
    c, d = max(0, left), min(panel.shape[1], left+raster.shape[1])
    if a < b and c < d:
        tile = raster[a-top:b-top, c-left:d-left]
        target = panel[a:b, c:d]
        mask = tile != bg
        target[mask] = tile[mask]


def _spaces(rng, extent, occupied, intervals):
    minimum = occupied+intervals+1
    if minimum > extent:
        return None
    extra = extent-minimum
    weights = rng.dirichlet(np.ones(intervals+1))
    return (1+rng.multinomial(extra, weights)).tolist()


def _block(rng, length, roles, previous):
    bg, marker, fast, slow, wall, body = roles
    palette = [x for x in range(10) if x not in (bg, marker, body)]
    size = int(rng.integers(1, len(palette)+1))
    local = rng.choice(palette, size=size, replace=False)
    old = [b for b in previous if b.shape[1] == length]
    if old and rng.random() < .18:
        return old[int(rng.integers(len(old)))].copy()
    mode = int(rng.integers(4))
    if mode == 0:
        block = np.full((2, length), rng.choice(local), dtype=np.int64)
    elif mode == 1:
        block = np.repeat(rng.choice(local, size=(2, 1)), length, axis=1)
    elif mode == 2:
        block = np.repeat(rng.choice(local, size=(1, length)), 2, axis=0)
    else:
        block = rng.choice(local, size=(2, length))
    block[:, :2] = body
    block[0, 0] = marker
    block[:, -2:] = body
    r, c = _CORNER_POINTS[int(rng.integers(4))]
    block[r, length-2+c] = marker
    return block.astype(np.int64)


def _visible_modules(blocks, marker):
    """Count modules retaining payload after all later layers are painted."""
    owner = {}
    jr = jc = orientation = 0
    for index, block in enumerate(blocks):
        for r in range(2):
            for c in range(block.shape[1]):
                dr, dc = _rot_point(r, c, orientation)
                owner[jr+dr, jc+dc] = index if 2 <= c < block.shape[1]-2 else None
        distance = block.shape[1]-2
        dr, dc = ((0, distance), (-distance, 0), (0, -distance), (distance, 0))[orientation]
        jr, jc = jr+dr, jc+dc
        orientation = (orientation+_exit(block, marker)) % 4
    return len(set(owner.values())-{None})


def _build_pair(rng, configuration, kind='ordinary', count_limit=7, force_count=None, force_unit3=False):
    rotation, down = configuration
    roles = tuple(map(int, rng.permutation(10)[:6]))
    bg, marker, fast, slow, wall, body = roles
    if kind == 'first':
        source_unit = 1
        counts = ([2, 2], [2, 1, 1], [2, 2, 1])[int(rng.integers(3))]
        corners, units = [0, 3], [1, 1]
    elif kind == 'second':
        source_unit = 2
        counts = [1]*int(rng.integers(3, 5))
        corners, units = [1, 2], [1, 2]
    else:
        source_unit = int(rng.choice([1, 1, 2]))
        n = force_count or int(rng.integers(1, min(count_limit, 5 if source_unit == 2 else 7)+1))
        if n > 5:
            source_unit = 1
        counts = [1]*n
        if source_unit == 1:
            while len(counts) > 1 and rng.random() < .65:
                eligible = [i for i in range(len(counts)-1) if counts[i] == counts[i+1] == 1]
                if not eligible:
                    break
                i = int(rng.choice(eligible))
                counts[i:i+2] = [2]
        rooms = int(rng.integers(1, 4))
        corners = rng.integers(0, 4, size=rooms).tolist()
        units = rng.integers(1, 3 if kind == 'ordinary' else 4, size=rooms).tolist()
        if force_unit3:
            units[int(rng.integers(rooms))] = 3
    # Queries sometimes request a complete small-scale view as well. This
    # preference changes only placement/fit, never any solving convention.
    full_query = kind == 'query' and rng.random() < .4
    if full_query and force_unit3 and len(units) == 1:
        full_query = False
    if full_query:
        units[0] = 1
        if force_unit3:
            units[int(rng.integers(1, len(units)))] = 3
    blocks, groups, lengths = [], [], []
    for count in counts:
        group = []
        for _ in range(count):
            ceiling = 8 if count == 2 or source_unit == 2 else 14
            length = int(rng.choice(np.arange(6, ceiling+1, 2)))
            group.append(len(blocks))
            blocks.append(_block(rng, length, roles, blocks))
            lengths.append(length)
        groups.append(group)
    base, offset = _array_chain(blocks, marker, bg)
    teaching = kind in ('first', 'second')
    if teaching:
        # Both dimensions must be visible, rather than merely a clipped bar.
        # Retain several module interiors while allowing informative overlap.
        if (min(base.shape) < 4 or _visible_modules(blocks, marker) < (3 if kind == 'first' else 2)
                or not any(_exit(block, marker) in (1, 3) for block in blocks[:-1])):
            return None
    # Multi-lane teaching kits have interleaved fast coordinates. Geometry is
    # otherwise sampled by budgets rather than a fixed first-strip signature.
    bank_min = max(sum(lengths[i]*source_unit for i in group)+len(group)+1 for group in groups)
    room_min_width = max(8 if kind in ('first', 'second') else 6, 2*max(units)+2)
    if teaching or full_query:
        room_min_width = max(room_min_width, base.shape[1]+2)
    width_min = 1+bank_min+1+room_min_width
    if width_min > 30:
        return None
    width = int(rng.integers(width_min, 31))
    bank_width = bank_min+int(rng.integers(width-width_min+1))
    divider = 1+bank_width
    lane_min = len(groups)*2*source_unit+len(groups)+1
    room_min = [max(2*t+2, 7 if kind == 'first' else 2*t+2) for t in units]
    if teaching or full_query:
        room_min[0] = max(room_min[0], base.shape[0]+2)
    if kind == 'first':
        room_min[1] = max(room_min[1], base.shape[0]+2)
    body_min = max(lane_min, sum(room_min)+len(units)-1)
    if body_min > 29:
        return None
    body_height = int(rng.integers(body_min, 30))
    height = body_height+1
    row0 = 1 if down else 0
    g = np.full((height, width), bg, dtype=np.int64)
    key_row = 0 if down else height-1
    g[:, 0] = slow
    g[key_row] = fast
    g[key_row, :2] = marker
    g[row0:row0+body_height, divider] = wall
    lane_gaps = _spaces(rng, body_height, len(groups)*2*source_unit, len(groups))
    starts = []
    r = row0+lane_gaps[0]
    for i in range(len(groups)):
        starts.append(r)
        r += 2*source_unit+lane_gaps[i+1]
    if not down:
        starts.reverse()
    for lane, group in enumerate(groups):
        occupied = sum(lengths[i]*source_unit for i in group)
        gaps = _spaces(rng, bank_width, occupied, len(group))
        if kind == 'first':
            # Preserve ample gap variation but put each lane's first piece
            # before the preceding lane's second piece.
            old = gaps[0]
            gaps[0] = 1+lane % 2
            if old < gaps[0]:
                gaps[0] = old
            gaps[-1] += old-gaps[0]
        c = 1+gaps[0]
        for j, i in enumerate(group):
            physical = np.repeat(np.repeat(blocks[i], source_unit, axis=0), source_unit, axis=1)
            g[starts[lane]:starts[lane]+physical.shape[0], c:c+physical.shape[1]] = physical
            c += physical.shape[1]+gaps[j+1]
    extra = body_height-sum(room_min)-(len(units)-1)
    allocations = rng.multinomial(extra, np.full(len(units), 1/len(units)))
    room_heights = [a+int(b) for a, b in zip(room_min, allocations)]
    out = g.copy()
    r = row0
    for index, (ph, t, corner) in enumerate(zip(room_heights, units, corners)):
        panel = g[r:r+ph, divider+1:]
        target = out[r:r+ph, divider+1:]
        pw = panel.shape[1]
        side = 2*t
        raster, off = _frame_raster(base, offset, t, corner)
        candidates = [(y, x) for y in range(1, ph-side) for x in range(1, pw-side)]
        rng.shuffle(candidates)
        accepted = None
        for y, x in candidates:
            if ((teaching and (index == 0 or kind == 'first')) or (full_query and index == 0)):
                top, left = y-off[0], x-off[1]
                if not (top >= 1 and left >= 1 and top+raster.shape[0] <= ph-1
                        and left+raster.shape[1] <= pw-1):
                    continue
            trial = np.full(panel.shape, bg, dtype=np.int64)
            _paste(trial, raster, y-off[0], x-off[1], bg)
            trial[y:y+side, x:x+side] = bg
            if np.any(~np.isin(trial, (bg, marker, body))):
                accepted = (y, x)
                break
        if accepted is None:
            return None
        y, x = accepted
        root = np.full((2, 2), body, dtype=np.int64)
        mr, mc = _CORNER_POINTS[corner]
        root[mr, mc] = marker
        root = np.repeat(np.repeat(root, t, axis=0), t, axis=1)
        panel[y:y+side, x:x+side] = root
        _paste(target, raster, y-off[0], x-off[1], bg)
        target[y:y+side, x:x+side] = root
        r += ph
        if index+1 < len(units):
            g[r, divider+1:] = wall
            out[r, divider+1:] = wall
            r += 1
    # Rotating both arrays is a proper scene rotation, not a mirror of payloads.
    return {'input': np.rot90(g, rotation).copy(), 'output': np.rot90(out, rotation).copy()}


_ALTERNATIVES = ('fast_reverse', 'slow_reverse', 'fast_major', 'repeat_first',
                 'no_turns', 'all_clockwise', 'no_reflection', 'rotate_root', 'no_scale')


def _teaching_evidence(train):
    data = [_decode(p['input']) for p in train]
    outcomes = {}
    for name in _ALTERNATIVES:
        outcomes[name] = any(not np.array_equal(_predict(d, name)[0], p['output'])
                             for d, p in zip(data, train))
    clipped = any(any(r['lost'] > 0 for r in _predict(d)[1]) for d in data)
    complete_units = set()
    complete_multilane = False
    for d in data:
        cells, _ = _chain_cells(d['blocks'], d['marker'])
        rs, cs = zip(*cells)
        articulated = max(rs)-min(rs)+1 >= 4 and max(cs)-min(cs)+1 >= 4
        full = any(record['lost'] == 0 for record in _predict(d)[1])
        if full and articulated and _visible_modules(d['blocks'], d['marker']) >= 2:
            complete_units.add(d['unit'])
            complete_multilane |= (len(d['groups']) >= 2 and max(map(len, d['groups'])) >= 2
                                   and _visible_modules(d['blocks'], d['marker']) >= 3)
    return (all(outcomes.values()) and clipped and complete_units == {1, 2}
            and complete_multilane), outcomes, clipped


def _signature(puzzle):
    return json.dumps({s: [{k: np.asarray(v).tolist() for k, v in p.items()} for p in puzzle[s]]
                       for s in ('train', 'test')}, separators=(',', ':'))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed')
    if seed < 0:
        raise ValueError('seed must be nonnegative')
    a, b = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    curriculum = int(rng.integers(3))  # held-out key / count / target lattice unit
    first_config = (int(rng.integers(4)), bool(rng.integers(2)))
    second_config = ((first_config[0]+int(rng.integers(1, 4))) % 4, not first_config[1])
    for _ in range(64):
        teaching = []
        for configuration, kind in ((first_config, 'first'), (second_config, 'second')):
            for attempt in range(512):
                pair = _build_pair(rng, configuration, kind)
                if pair is not None:
                    teaching.append(pair)
                    break
            else:
                raise RuntimeError('bounded complete-chain construction exhausted')
        one, two = teaching
        if _teaching_evidence([one, two])[0]:
            break
    else:
        raise RuntimeError('bounded teaching-kit construction exhausted')
    puzzle = {'train': [one, two], 'test': []}
    seen = {(p['input'].shape, p['input'].tobytes()) for p in puzzle['train']}
    unused_configs = [(k, down) for k in range(4) for down in (True, False)
                      if (k, down) not in (first_config, second_config)]
    for split, count in (('train', a-2), ('test', b)):
        for i in range(count):
            for _ in range(128):
                if split == 'train' and curriculum == 0:
                    config = (first_config, second_config)[int(rng.integers(2))]
                elif split == 'test' and curriculum == 0 and i == 0:
                    config = unused_configs[int(rng.integers(len(unused_configs)))]
                else:
                    config = (int(rng.integers(4)), bool(rng.integers(2)))
                pair = _build_pair(rng, config, 'ordinary' if split == 'train' else 'query',
                                   count_limit=5 if curriculum == 1 and split == 'train' else 7,
                                   force_count=int(rng.integers(6, 8)) if curriculum == 1 and split == 'test' and i == 0 else None,
                                   force_unit3=curriculum == 2 and split == 'test' and i == 0)
                if pair is None:
                    continue
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded distinct-input construction exhausted')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = (inp.shape, inp.tobytes())
            assert key not in seen, 'duplicate input'
            seen.add(key)
            d = _decode(inp)
            assert 1 <= len(d['blocks']) <= 7 and d['unit'] in (1, 2)
            assert 1 <= len(d['rooms']) <= 3 and all(1 <= r['unit'] <= 3 for r in d['rooms'])
            predicted, records, _ = _predict(d)
            assert np.array_equal(predicted, out), 'incorrect docking, transfer or clipping'
            assert not np.array_equal(inp, out), 'identity pair'
            protected = np.ones(d['grid'].shape, dtype=bool)
            normalized_out = np.rot90(out, d['rotation'])
            for room, record in zip(d['rooms'], records):
                a, b, c, e = room['bounds']
                protected[a:b, c:e] = False
                r, col = room['root']
                side = room['unit']*2
                protected[a+r:a+r+side, c+col:c+col+side] = True
                region = normalized_out[a:b, c:e]
                assert np.any(~np.isin(region, (d['background'], d['body'], d['marker']))), 'invisible completion'
                assert record['visible_detail'] > 0
            assert np.array_equal(normalized_out[protected], d['grid'][protected]), 'changed kit, key, wall or root'
            if split == 'train':
                training.append(d)
    assert {d['unit'] for d in training} == {1, 2}, 'teach both reference resolutions'
    assert {d['down'] for d in training} == {True, False}, 'teach both slow directions'
    assert len({d['rotation'] for d in training}) >= 2, 'teach proper key rotation'
    assert set.union(*(set(r['corner'] for r in d['rooms']) for d in training)) == {0, 1, 2, 3}
    passed, contrasts, clipped = _teaching_evidence(puzzle['train'])
    assert passed, f'inadequate training contrasts: {contrasts}, clipped={clipped}'
    # A shift is observed from returned pixels, without reading a curriculum flag.
    queries = [_decode(p['input']) for p in puzzle['test']]
    train_configs = {(d['rotation'], d['down']) for d in training}
    largest = max(len(d['blocks']) for d in training)
    train_units = {r['unit'] for d in training for r in d['rooms']}
    assert any((d['rotation'], d['down']) not in train_configs or len(d['blocks']) > largest
               or any(r['unit'] not in train_units for r in d['rooms']) for d in queries), 'missing train/test shift'


def _expect_invalid(puzzle):
    try:
        validate_generated_puzzle(puzzle)
    except (AssertionError, ValueError):
        return
    raise AssertionError('invalid puzzle accepted')


def self_test():
    """Bounded 42-seed suite plus concrete semantic regressions."""
    signatures, shapes, layouts, palettes, directions, lengths, exits, scales = [set() for _ in range(8)]
    chamber_counts, counts, overlaps, clip_counts, source_units = set(), set(), 0, set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        if seed < 3:
            assert _signature(puzzle) == _signature(generate_puzzle(seed))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                d = _decode(pair['input'])
                shapes.add(d['grid'].shape)
                directions.add((d['rotation'], d['down']))
                counts.add(len(d['blocks']))
                source_units.add(d['unit'])
                chamber_counts.add(len(d['rooms']))
                layouts.add((tuple(len(group) for group in d['groups']), tuple(d['boxes']),
                             tuple(tuple(room['bounds']) for room in d['rooms'])))
                palettes.add((d['background'], d['marker'], d['fast'], d['slow'], d['wall'], d['body']))
                lengths.update(b.shape[1] for b in d['blocks'])
                exits.update(_exit(b, d['marker']) for b in d['blocks'])
                scales.update((room['unit'], d['unit']) for room in d['rooms'])
                prediction, records, overwritten = _predict(d)
                assert np.array_equal(prediction, pair['output'])
                overlaps += overwritten > 0
                clip_counts.add(sum(r['lost'] > 0 for r in records))
                # Proper whole-scene rotation changes neither logical rule nor frame.
                if seed < 2:
                    for k in range(4):
                        assert np.array_equal(solve(np.rot90(pair['input'], k)), np.rot90(pair['output'], k))
    assert len(signatures) == 42
    assert len(shapes) >= 30 and len(layouts) >= 100 and len(palettes) >= 100
    assert len(directions) == 8 and counts >= {1, 3, 4, 5, 6, 7}
    assert chamber_counts == {1, 2, 3} and source_units == {1, 2}
    assert exits == {0, 1, 2, 3} and len(lengths) >= 4
    assert scales >= {(1, 1), (2, 1), (1, 2), (2, 2), (3, 2)}
    assert overlaps >= 40 and len(clip_counts) >= 3
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(1900+a*10+b, a, b)
            assert (len(p['train']), len(p['test'])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid counts accepted')
    for seed in (-1, True, 1.5):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid seed accepted')
    damaged = copy.deepcopy(generate_puzzle(702, 2, 1))
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    _expect_invalid(damaged)
    duplicate = copy.deepcopy(generate_puzzle(703, 2, 1))
    duplicate['test'][0] = copy.deepcopy(duplicate['train'][0])
    _expect_invalid(duplicate)
    # Five rows suffice for a key, two margin rows and one two-row piece.
    # This compact case must not be excluded by an arbitrary parser size floor.
    small = np.full((5, 16), 9, dtype=np.int64)
    small[:, 0] = 3
    small[0] = 1
    small[0, :2] = 0
    small[1:, 9] = 4
    small[2:4, 2:8] = np.array([[0, 8, 2, 2, 0, 8], [8, 8, 2, 2, 8, 8]])
    small[2:4, 11:13] = np.array([[0, 8], [8, 8]])
    expected = small.copy()
    expected[2:4, 13:15] = 2
    expected[2:4, 15] = [0, 8]
    assert np.array_equal(solve(small), expected)
    # Docking orientation regression: TL, BL, BR, TR are 0, 1, 2, 3 turns.
    for delta in range(4):
        first = np.full((2, 6), 7, dtype=np.int64)
        second = np.full((2, 8), 8, dtype=np.int64)
        for block in (first, second):
            block[:, :2] = block[:, -2:] = 6
            block[0, 0] = 9
            block[0, -2] = 9
        first[:, -2:] = 6
        r, c = _CORNER_POINTS[delta]
        first[r, 4+c] = 9
        cells, _ = _chain_cells([first, second], 9)
        raster, offset = _array_chain([first, second], 9, 0)
        expected = {(r-offset[0], c-offset[1]): int(raster[r, c])
                    for r, c in np.argwhere(raster != 0)}
        assert cells == expected
        for k in range(4):
            array, off = _frame_raster(raster, offset, 2, k)
            for (y, x), value in cells.items():
                yy = 1-y if k in (1, 2) else y
                xx = 1-x if k in (2, 3) else x
                assert np.all(array[off[0]+yy*2:off[0]+yy*2+2,
                                    off[1]+xx*2:off[1]+xx*2+2] == value)
    # A reversal sends the next material over the first strip and its exit;
    # the last layer wins except for the explicitly preserved input root.
    first = np.array([[9, 6, 7, 7, 6, 6], [6, 6, 7, 7, 6, 9]])
    second = np.array([[9, 6, 8, 8, 8, 8, 8, 8, 6, 9], [6, 6, 8, 8, 8, 8, 8, 8, 6, 6]])
    cells, overwritten = _chain_cells([first, second], 9)
    assert overwritten > 0 and cells[(0, 2)] == 8
    # A third arm passes through an earlier exit socket. Only the root is
    # protected in the final output; this generated socket becomes detail.
    crossing = []
    for length, value, corner in ((6, 7, 3), (6, 8, 2), (10, 5, 0)):
        b = np.full((2, length), value)
        b[:, :2] = b[:, -2:] = 6
        b[0, 0] = 9
        r, c = _CORNER_POINTS[corner]
        b[r, length-2+c] = 9
        crossing.append(b)
    cells, _ = _chain_cells(crossing, 9)
    assert cells[(0, 4)] == 5
    # Clip after assembly: an unseen joint outside a small panel can send the
    # following strip back in. Omitting that third strip loses visible cells.
    blocks = []
    for length, value, corner in ((6, 7, 0), (8, 8, 2), (14, 5, 0)):
        b = np.full((2, length), value)
        b[:, :2] = b[:, -2:] = 6
        b[0, 0] = 9
        r, c = _CORNER_POINTS[corner]
        b[r, length-2+c] = 9
        blocks.append(b)
    complete, _ = _chain_cells(blocks, 9)
    truncated, _ = _chain_cells(blocks[:2], 9)
    assert complete[(0, 3)] == 5 and truncated[(0, 3)] == 7
    raster, offset = _array_chain(blocks, 9, 0)
    panel = np.zeros((4, 8), dtype=np.int64)
    _paste(panel, raster, -offset[0], -offset[1], 0)
    assert panel[0, 3] == 5
    # Corrupt a source lattice quarter: parse must reject without any label.
    damaged = copy.deepcopy(generate_puzzle(704, 2, 1))
    p = next(p for p in damaged['train'] if _decode(p['input'])['unit'] == 2)
    d = _decode(p['input'])
    a, b, c, e = d['boxes'][0]
    bad = d['grid'].copy()
    bad[a, c] = d['body']
    p['input'] = np.rot90(bad, -d['rotation']).copy()
    _expect_invalid(damaged)
    return {'seeds': 42, 'distinct_puzzles': len(signatures), 'shapes': len(shapes),
            'layouts': len(layouts), 'palettes': len(palettes), 'directions': len(directions),
            'strip_counts': sorted(counts), 'lengths': sorted(lengths), 'scale_units': sorted(scales),
            'overlapping_pairs': overlaps, 'chamber_counts': sorted(chamber_counts)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
