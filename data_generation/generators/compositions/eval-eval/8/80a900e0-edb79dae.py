"""Diagonal ribbon loom: input-only geometric stencil continuation.

Only Python and NumPy are required. All coordinates below refer to the frame's
interior, so frame cells and exterior clues never participate in propagation.
"""
from __future__ import annotations

import numpy as np

# Directions clockwise from north-east. Each side is traversed clockwise.
DIRECTIONS = ((-1, 1), (1, 1), (1, -1), (-1, -1))
TIPS = ((-1, 0), (0, 1), (1, 0), (0, -1))
TANGENTS = ((1, 1), (1, -1), (-1, -1), (-1, 1))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _outline(center, radius):
    r, c = center
    return [(r + y, c + x) for y in range(-radius, radius + 1)
            for x in range(-radius, radius + 1) if abs(y) + abs(x) == radius]


def _cap(center, radius, direction):
    dr, dc = DIRECTIONS[direction]
    ar, ac = TIPS[direction]
    tr, tc = TANGENTS[direction]
    return [(center[0] + dr + radius * ar + j * tr,
             center[1] + dc + radius * ac + j * tc) for j in range(radius + 1)]


def _lanes(center, radius, direction, shape):
    dr, dc = DIRECTIONS[direction]
    result = []
    for r, c in _cap(center, radius, direction):
        lane = []
        r, c = r + dr, c + dc
        while _inside((r, c), shape):
            lane.append((r, c))
            r, c = r + dr, c + dc
        result.append(lane)
    return result


def _base(shape, colors):
    return np.asarray(colors, dtype=np.int8)[np.indices(shape).sum(0) % 2]


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, cells = [p], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _box(cells):
    a = np.asarray(cells)
    return (int(a[:, 0].min()), int(a[:, 0].max()) + 1,
            int(a[:, 1].min()), int(a[:, 1].max()) + 1)


def _parse(input_grid):
    """Recover the frame, legend and every emitter from visible pixels only."""
    grid = _grid(input_grid)
    frames = []
    for color in np.unique(grid):
        cells = np.argwhere(grid == color)
        a, b, x, y = _box(cells)
        h, w = b - a, y - x
        if h < 9 or w < 9 or len(cells) != 2 * h + 2 * w - 4:
            continue
        if (np.all(grid[a, x:y] == color) and np.all(grid[b - 1, x:y] == color)
                and np.all(grid[a:b, x] == color) and np.all(grid[a:b, y - 1] == color)):
            frames.append(((a, b, x, y), int(color)))
    assert len(frames) == 1, 'the work frame must be unique and have an exclusive color'
    frame, frame_color = frames[0]
    a, b, x, y = frame
    board = grid[a + 1:b - 1, x + 1:y - 1]
    parity = np.indices(board.shape).sum(0) % 2
    backgrounds = []
    for p in (0, 1):
        counts = np.bincount(board[parity == p], minlength=10)
        bg = int(counts.argmax())
        assert counts[bg] * 2 > int((parity == p).sum())
        backgrounds.append(bg)
    assert len(set(backgrounds)) == 2 and frame_color not in backgrounds
    base = _base(board.shape, backgrounds)
    exterior = np.ones(grid.shape, dtype=bool)
    exterior[a:b, x:y] = False
    counts = np.bincount(grid[exterior], minlength=10)
    outer_bg = int(counts.argmax())
    assert outer_bg in backgrounds
    clue_mask = exterior & (grid != outer_bg)
    mapping, key_boxes = {}, {}
    consumed = np.zeros(grid.shape, dtype=bool)
    # Two-color solid 2x2 keys: three source cells, one replacement cell.
    for cells in _components(clue_mask):
        values, counts = np.unique([grid[p] for p in cells], return_counts=True)
        if len(values) == 1:
            continue
        aa, bb, xx, yy = _box(cells)
        assert (bb - aa, yy - xx) == (2, 2) and len(cells) == 4
        assert sorted(map(int, counts)) == [1, 3]
        source = int(values[np.argmax(counts)])
        target = int(values[np.argmin(counts)])
        assert source not in mapping
        assert source not in backgrounds and target not in backgrounds
        assert frame_color not in (source, target)
        mapping[source] = target
        key_boxes[source] = (aa, bb, xx, yy)
        consumed[aa:bb, xx:yy] = True
    assert 1 <= len(mapping) <= 3
    remainder = clue_mask & ~consumed
    assert set(map(int, np.unique(grid[remainder]))) == set(mapping)
    stencils, stencil_boxes, sizes = {}, {}, set()
    for source in sorted(mapping):
        aa, bb, xx, yy = _box(np.argwhere(remainder & (grid == source)))
        n = bb - aa
        assert n == yy - xx and 2 <= n <= 4
        tile = grid[aa:bb, xx:yy]
        assert np.all(np.isin(tile, [outer_bg, source]))
        mask = tile == source
        assert mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()
        stencils[source] = mask.copy()
        stencil_boxes[source] = (aa, bb, xx, yy)
        sizes.add(n)
    assert len(sizes) == 1
    n = sizes.pop()
    radius = n + 1
    boxes = list(stencil_boxes.values()) + list(key_boxes.values())
    for i, (aa, bb, xx, yy) in enumerate(boxes):
        assert bb <= a or aa >= b or yy <= x or xx >= y
        for aaa, bbb, xxx, yyy in boxes[i + 1:]:
            assert bb < aaa or bbb < aa or yy < xxx or yyy < xx
    # Complete uniform diamond outlines. Their structural color is never a
    # source or replacement color, preventing cap/stencil role ambiguity.
    role_colors = set(backgrounds) | set(mapping) | set(mapping.values()) | {frame_color}
    emitters = []
    h, w = board.shape
    for r in range(radius, h - radius):
        for c in range(radius, w - radius):
            cells = _outline((r, c), radius)
            color = int(board[cells[0]])
            if color not in role_colors and all(board[p] == color for p in cells):
                emitters.append(dict(center=(r, c), struct=color, caps={}))
    assert 1 <= len(emitters) <= 3
    covered = set()
    for e in emitters:
        cells = set(_outline(e['center'], radius))
        assert not cells & covered
        covered |= cells
    for e in emitters:
        for d in range(4):
            cells = _cap(e['center'], radius, d)
            if not all(_inside(p, board.shape) for p in cells):
                continue
            color = int(board[cells[0]])
            if color in mapping and all(board[p] == color for p in cells):
                assert not covered.intersection(cells)
                e['caps'][d] = color
                covered.update(cells)
        assert e['caps']
    assert covered == set(map(tuple, np.argwhere(board != base)))
    assert {s for e in emitters for s in e['caps'].values()} == set(mapping)
    reserved = set()
    for e in emitters:
        for d in e['caps']:
            lanes = _lanes(e['center'], radius, d, board.shape)
            for lane in lanes:
                for p in lane:
                    assert p not in covered and p not in reserved, 'ribbons must have unique unobstructed ownership'
                    reserved.add(p)
    return dict(grid=grid, frame=frame, frame_color=frame_color, board=board,
                base=base, backgrounds=tuple(backgrounds), outer_bg=outer_bg,
                mapping=mapping, stencils=stencils, n=n, radius=radius,
                emitters=emitters, key_boxes=key_boxes, stencil_boxes=stencil_boxes,
                footprint=reserved)


def _paint(analysis, mode='full'):
    """Walking-lane implementation used by solve; ablations aid validation."""
    a = analysis
    r, e, c, f = a['frame']
    result = a['grid'][r:e, c:f].copy()
    board = result[1:-1, 1:-1]
    for emitter in a['emitters']:
        for direction, source in emitter['caps'].items():
            color = source if mode == 'identity_map' else a['mapping'][source]
            lanes = _lanes(emitter['center'], a['radius'], direction, board.shape)
            for j, lane in enumerate(lanes):
                for t, p in enumerate(lane):
                    edge = j in (0, a['radius'])
                    mask = a['stencils'][source]
                    if mode == 'transpose':
                        mask = mask.T
                    elif mode == 'reverse_width':
                        mask = mask[:, ::-1]
                    elif mode == 'reverse_depth':
                        mask = mask[::-1]
                    active = edge or (mode != 'rails_only' and mask[t % a['n'], j - 1])
                    if mode == 'pattern_only' and edge:
                        active = False
                    if mode == 'solid':
                        active = True
                    if mode == 'single_tile' and t >= a['n'] and not edge:
                        active = False
                    if active:
                        board[p] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """No training data, RNG state, palette tables or generation metadata."""
    return _paint(_parse(input_grid))


def _coordinate_prediction(a):
    """Check labels by projecting every board cell into the ribbon basis.

    Shares visible-input parsing, but does not walk lanes or reuse _paint.
    """
    r, e, c, f = a['frame']
    result = a['grid'][r:e, c:f].copy()
    target = result[1:-1, 1:-1]
    yy, xx = np.indices(target.shape)
    for emitter in a['emitters']:
        for d, source in emitter['caps'].items():
            dr, dc = DIRECTIONS[d]
            tr, tc = TANGENTS[d]
            ar, ac = TIPS[d]
            origin_r = emitter['center'][0] + dr + a['radius'] * ar
            origin_c = emitter['center'][1] + dc + a['radius'] * ac
            dy, dx = yy - origin_r, xx - origin_c
            along = dy * dr + dx * dc
            across = dy * tr + dx * tc
            valid = ((along % 2) == 0) & ((across % 2) == 0)
            t, j = along // 2, across // 2
            valid &= (t >= 1) & (j >= 0) & (j <= a['radius'])
            edge = (j == 0) | (j == a['radius'])
            inside = (j > 0) & (j < a['radius'])
            stamp = a['stencils'][source][(t - 1) % a['n'], np.clip(j - 1, 0, a['n'] - 1)]
            target[valid & (edge | (inside & stamp))] = a['mapping'][source]
    return result


def _mask(rng, n, informative=False):
    for _ in range(100):
        kind = int(rng.integers(7))
        yy, xx = np.indices((n, n))
        if informative or kind == 0:
            mask = rng.random((n, n)) < rng.uniform(.25, .8)
        elif kind == 1:
            mask = (xx == yy) | (xx + yy == n - 1)
        elif kind == 2:
            mask = (xx == 0) | (yy == 0) | (xx == n - 1) | (yy == n - 1)
        elif kind == 3:
            mask = (xx + yy) % 2 == 0
        elif kind == 4:
            mask = (yy % 2 == 0) | (yy == n - 1)
        elif kind == 5:
            mask = (xx == n // 2) | (yy == n // 2)
        else:
            mask = np.ones((n, n), dtype=bool)
        for edge in (mask[0], mask[-1], mask[:, 0], mask[:, -1]):
            if not edge.any():
                edge[int(rng.integers(n))] = True
        if informative:
            if (mask.all() or np.array_equal(mask, mask.T)
                    or np.array_equal(mask, mask[:, ::-1]) or np.array_equal(mask, mask[::-1])):
                continue
            if len({row.tobytes() for row in mask}) < 2:
                continue
        return mask
    # Deterministic, non-transpose-symmetric and row-varying fallback.
    mask = np.zeros((n, n), dtype=bool)
    mask[:, 0] = True
    mask[-1, :] = True
    return mask


def _footprint(center, radius, caps, shape):
    objects = set(_outline(center, radius))
    rays = set()
    for d in caps:
        objects.update(_cap(center, radius, d))
        rays.update(p for lane in _lanes(center, radius, d, shape) for p in lane)
    if not all(_inside(p, shape) for p in objects):
        return None
    return objects, rays


def _geometry(rng, shape, n, count, required=None):
    radius = n + 1
    for _ in range(12):
        emitters, occupied, rays = [], set(), set()
        for i in range(count):
            for _ in range(60):
                caps = list(required) if i == 0 and required is not None else [d for d in range(4) if rng.random() < .43]
                if not caps:
                    caps = [int(rng.integers(4))]
                if required is not None and i == 0:
                    margin = 2 * n + 2
                    center = tuple(int(rng.integers(margin, extent - margin)) for extent in shape)
                else:
                    center = tuple(int(rng.integers(radius + 1, extent - radius - 1)) for extent in shape)
                if any(sum(abs(v - w) for v, w in zip(center, e['center'])) < 2 * radius + 3
                       for e in emitters):
                    continue
                fp = _footprint(center, radius, caps, shape)
                if fp is None:
                    continue
                obj, path = fp
                if obj & (occupied | rays) or path & (occupied | rays):
                    continue
                if not path:
                    continue
                emitters.append(dict(center=center, caps=caps))
                occupied |= obj
                rays |= path
                break
            else:
                break
        if len(emitters) == count:
            return emitters
    # A guaranteed pair of separated diamonds facing opposite outer corners.
    # Count-three packing may reduce to two, never below the multiplex shift.
    if count >= 2:
        centers = [(radius + 3, radius + 3),
                   (shape[0] - radius - 4, shape[1] - radius - 4)]
        result = [dict(center=centers[0], caps=[3]), dict(center=centers[1], caps=[1])]
        first = _footprint(centers[0], radius, [3], shape)
        second = _footprint(centers[1], radius, [1], shape)
        assert not (first[0] | first[1]) & (second[0] | second[1])
        return result
    raise RuntimeError('single-emitter construction failed')


def _draw_frame(grid, frame, color):
    a, b, x, y = frame
    grid[a, x:y] = grid[b - 1, x:y] = color
    grid[a:b, x] = grid[a:b, y - 1] = color


def _generate_pair(rng, n, count, required, side, compact=False, longrun=False):
    # At least n+1 complete interior rows in teaching caps expose repetition.
    minimum = 4 * n + 7 if required is not None else max(2 * n + 11, 3 * (n + 1) + 10 if count > 1 else 0)
    max_short = 29 - n
    minimum = min(minimum, max_short)
    bh = int(rng.integers(minimum, max_short + 1))
    bw = int(rng.integers(minimum, 31))
    if compact:
        bh = minimum
        bw = max(minimum, min(bw, minimum + 2))
    if longrun:
        bh, bw = max_short, 30
    # For vertical clue bands exchange board dimensions, not clue orientation.
    if side in (1, 3):
        bh, bw = bw, bh
    shape = (bh - 2, bw - 2)
    if longrun:
        d = int(rng.integers(4))
        dr, dc = DIRECTIONS[d]
        center = (n + 2 if dr > 0 else shape[0] - n - 3,
                  n + 2 if dc > 0 else shape[1] - n - 3)
        emitters = [dict(center=center, caps=[d])]
    else:
        emitters = _geometry(rng, shape, n, count, required)
    cap_slots = [(i, d) for i, emitter in enumerate(emitters) for d in emitter['caps']]
    k = 2 if required is not None else int(rng.integers(1, min(3, len(cap_slots)) + 1))
    colors = list(map(int, rng.permutation(10)))
    bg, other_bg, frame_color, structural = colors[:4]
    sources = colors[4:4 + k]
    choices = colors[4:]
    map_kind = int(rng.integers(4))
    if required is not None:
        map_kind = int(rng.integers(2))
    if map_kind == 0 and k > 1:
        order = list(map(int, rng.permutation(sources)))
        mapping = {s: order[(i + 1) % k] for i, s in enumerate(order)}
    elif map_kind == 1 and k > 1:
        order = list(map(int, rng.permutation(sources)))
        mapping = {s: order[i + 1] if i + 1 < k else colors[4 + k] for i, s in enumerate(order)}
    else:
        mapping = {s: int(rng.choice([v for v in choices if v != s])) for s in sources}
    masks = {s: _mask(rng, n, informative=required is not None) for s in sources}
    assignments = sources + list(map(int, rng.choice(sources, len(cap_slots) - k)))
    rng.shuffle(assignments)
    board = _base(shape, (bg, other_bg))
    for emitter in emitters:
        for p in _outline(emitter['center'], n + 1):
            board[p] = structural
    for (i, d), source in zip(cap_slots, assignments):
        for p in _cap(emitters[i]['center'], n + 1, d):
            board[p] = source
    # Shuffle all clue rectangles independently of source/key association.
    entries = [('stencil', s, n) for s in sources] + [('key', s, 2) for s in sources]
    rng.shuffle(entries)
    used = sum(e[2] for e in entries) + len(entries) - 1
    long = max(bw if side in (0, 2) else bh, used)
    assert long <= 30
    short = (bh if side in (0, 2) else bw) + n + 1
    canvas_shape = (short, long) if side in (0, 2) else (long, short)
    grid = np.full(canvas_shape, bg, dtype=np.int8)
    offset = int(rng.integers(long - (bw if side in (0, 2) else bh) + 1))
    if side == 0:  # top
        frame = (n + 1, n + 1 + bh, offset, offset + bw)
    elif side == 2:
        frame = (0, bh, offset, offset + bw)
    elif side == 3:  # left
        frame = (offset, offset + bh, n + 1, n + 1 + bw)
    else:
        frame = (offset, offset + bh, 0, bw)
    _draw_frame(grid, frame, frame_color)
    a, b, x, y = frame
    grid[a + 1:b - 1, x + 1:y - 1] = board
    gaps = rng.multinomial(long - used, np.full(len(entries) + 1, 1 / (len(entries) + 1)))
    along = int(gaps[0])
    for index, (kind, source, size) in enumerate(entries):
        across = int(rng.integers(n - size + 1))
        if side == 0:
            rr, cc = across, along
        elif side == 2:
            rr, cc = bh + 1 + across, along
        elif side == 3:
            rr, cc = along, across
        else:
            rr, cc = along, bw + 1 + across
        if kind == 'stencil':
            grid[rr:rr + n, cc:cc + n][masks[source]] = source
        else:
            tile = np.full((2, 2), source, dtype=np.int8)
            corner = int(rng.integers(4))
            tile[corner // 2, corner % 2] = mapping[source]
            grid[rr:rr + 2, cc:cc + 2] = tile
        along += size + 1 + int(gaps[index + 1])
    analysis = _parse(grid)
    assert len(analysis['emitters']) == len(emitters)
    return {'input': grid, 'output': _coordinate_prediction(analysis)}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    nmax = int(rng.choice([2, 3, 4], p=[.25, .4, .35]))
    # A puzzle holds out either multiplicity, longer runs, or a new clue side.
    shift = int(rng.integers(3))
    sides = list(map(int, rng.permutation(4)))
    partition = list(map(int, rng.permutation(4)))
    witnesses = [partition[:2], partition[2:]]
    teaching_slots = list(map(int, rng.choice(num_train, 2, replace=False)))
    teaching = dict(zip(teaching_slots, witnesses))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            required = teaching.get(index) if split == 'train' else None
            longrun = split == 'test' and shift == 1
            n = nmax if required is not None or longrun else int(rng.integers(2, nmax + 1))
            if required is not None or (split == 'train' and shift == 0):
                emit_count = 1
            elif split == 'test' and shift == 0:
                emit_count = 2 if n >= 3 else int(rng.integers(2, 4))
            else:
                emit_count = int(rng.integers(1, (2 if n >= 3 else 3) + 1))
            if shift == 1:
                emit_count = 1
            side = sides[2 + index % 2] if split == 'test' and shift == 2 else int(rng.choice(sides[:2] if shift == 2 else sides))
            compact = split == 'train' and shift == 1
            for _ in range(20):
                pair = _generate_pair(rng, n, emit_count, required, side, compact, longrun)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct-input construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _evidence(a, out):
    """Directions with a full tile, its next row, and continuing endpoint rails."""
    evidence = set()
    for emitter in a['emitters']:
        for d, source in emitter['caps'].items():
            lanes = _lanes(emitter['center'], a['radius'], d, a['board'].shape)
            mask = a['stencils'][source]
            if (min(len(lane) for lane in lanes[1:-1]) >= a['n'] + 1
                    and min(len(lanes[0]), len(lanes[-1])) >= a['n']
                    and not mask.all() and mask.any()
                    and not np.array_equal(mask, mask.T)
                    and not np.array_equal(mask, mask[:, ::-1])):
                evidence.add((d, a['n']))
    return evidence


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, evidence, test_sizes = set(), set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            a = _parse(inp)
            assert np.array_equal(out, _paint(a))
            assert np.array_equal(out, _coordinate_prediction(a))
            r, e, c, f = a['frame']
            original = inp[r:e, c:f]
            assert out.shape == original.shape
            assert np.array_equal(out[0], original[0]) and np.array_equal(out[-1], original[-1])
            assert np.array_equal(out[:, 0], original[:, 0]) and np.array_equal(out[:, -1], original[:, -1])
            foreground = a['board'] != a['base']
            assert np.array_equal(out[1:-1, 1:-1][foreground], a['board'][foreground])
            assert not np.array_equal(out, original)
            assert not np.array_equal(out, _paint(a, 'identity_map')), 'mapping must affect the answer'
            if split == 'train':
                evidence |= _evidence(a, out)
            else:
                test_sizes.add(a['n'])
    for d in range(4):
        assert any(dd == d and n >= max(test_sizes) for dd, n in evidence), 'all directions need informative training'
    # Both inherited mechanisms have explicit counterexamples in training.
    for mode in ('pattern_only', 'rails_only', 'solid', 'single_tile', 'transpose', 'reverse_width', 'reverse_depth'):
        assert any(not np.array_equal(np.asarray(pair['output']), _paint(_parse(pair['input']), mode))
                   for pair in puzzle['train']), f'training fails to exclude {mode}'


def measure_puzzle(puzzle):
    records = []
    for split in ('train', 'test'):
        for index, pair in enumerate(puzzle[split]):
            a = _parse(pair['input'])
            lengths = [len(lane) for e in a['emitters'] for d in e['caps']
                       for lane in _lanes(e['center'], a['radius'], d, a['board'].shape)]
            records.append(dict(split=split, index=index, n=a['n'], emitters=len(a['emitters']),
                                caps=sum(len(e['caps']) for e in a['emitters']),
                                directions=sorted({d for e in a['emitters'] for d in e['caps']}),
                                categories=len(a['mapping']), overlap=bool(set(a['mapping']) & set(a['mapping'].values())),
                                disconnected=sum(len(_components(m)) > 1 for m in a['stencils'].values()),
                                longest=max(lengths), shortest=min(lengths),
                                input_shape=list(np.asarray(pair['input']).shape),
                                output_shape=list(np.asarray(pair['output']).shape),
                                masks=[m.astype(int).tolist() for m in a['stencils'].values()],
                                geometry=(a['board'].shape, tuple((e['center'], tuple(e['caps'])) for e in a['emitters']))))
    return records


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records = [r for p in puzzles for r in measure_puzzle(p)]
    assert len({tuple(_key(pair['input']) for s in ('train', 'test') for pair in p[s]) for p in puzzles}) == 42
    assert {r['n'] for r in records} == {2, 3, 4}
    assert {r['emitters'] for r in records} >= {1, 2, 3}
    assert {r['categories'] for r in records} >= {1, 2, 3}
    assert {d for r in records for d in r['directions']} == set(range(4))
    assert len({r['geometry'] for r in records}) >= 120
    assert len({tuple(r['input_shape']) for r in records}) >= 35
    assert len({str(r['masks']) for r in records}) >= 60
    assert sum(r['disconnected'] for r in records) >= 10
    assert sum(r['overlap'] for r in records) >= 20
    assert max(r['longest'] for r in records) >= 15
    assert min(r['shortest'] for r in records) <= 1
    assert all(_key(a[k]) == _key(b[k]) for s in ('train', 'test')
               for a, b in zip(puzzles[0][s], generate_puzzle(0)[s]) for k in ('input', 'output'))
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(10000 + 10 * train + test, train, test)
            assert (len(p['train']), len(p['test'])) == (train, test)
    for tr, te in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    # Arbitrary palette permutation preserves every semantic role.
    permutation = np.array([8, 3, 0, 7, 5, 9, 2, 1, 6, 4], dtype=np.int8)
    for p in puzzles[:4]:
        for s in ('train', 'test'):
            for pair in p[s]:
                assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
    # A visible, original source assignment selects only one substitution,
    # even when the destination color is another source (including cycles).
    for p in puzzles[:5]:
        for pair in p['train']:
            a = _parse(pair['input'])
            if len(a['mapping']) < 2:
                continue
            changed = pair['input'].copy()
            sources = sorted(a['mapping'])
            for i, source in enumerate(sources):
                r, e, c, f = a['key_boxes'][source]
                block = changed[r:e, c:f]
                block[block != source] = sources[(i + 1) % len(sources)]
            assert np.array_equal(solve(changed), _coordinate_prediction(_parse(changed)))
    # Reject label corruption, and check the solver never modifies its input.
    pair = puzzles[0]['test'][0]
    original = pair['input'].copy()
    solve(pair['input'])
    assert np.array_equal(original, pair['input'])
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in ('train', 'test')}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged label accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds, geometry/stencil/role variation, all legal counts')
