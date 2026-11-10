"""Count-addressed catalogue: object counts select ordinal strip occurrences.

Only standard-library and NumPy dependencies. No files, seed, or examples are
consulted by solve(). Diagnostics in self_test() are bounded and kept in memory.
"""
from collections import Counter
import hashlib
import numpy as np


_DIAMOND = ((0, 1), (1, 0), (1, 2), (2, 1))
_SQUARE = ((0, 0), (0, 1), (1, 0), (1, 1))
_CROSS = ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))
_X = ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2))
_SINGLE = ((0, 0),)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and not np.issubdtype(a.dtype, np.bool_)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _fingerprint(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _components(mask, diagonal=True):
    todo = set(map(tuple, np.argwhere(mask)))
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        pending, cells = [start], []
        while pending:
            r, c = pending.pop()
            cells.append((int(r), int(c)))
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in todo:
                    todo.remove(q)
                    pending.append(q)
        result.append(cells)
    return result


def _normalize(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _stack(blocks, field, width=None):
    if not blocks:
        return np.empty((0, 0), dtype=np.int8)
    width = max(b.shape[1] for b in blocks) if width is None else width
    a = np.full((2 * len(blocks), width), field, dtype=np.int8)
    for i, b in enumerate(blocks):
        assert width >= b.shape[1] and (width - b.shape[1]) % 2 == 0
        left = (width - b.shape[1]) // 2
        a[2*i:2*i+2, left:left+b.shape[1]] = b
    return a


def _board_candidate(grid, field, bounds, bottom, right, bg):
    r0, r1, c0, c1 = bounds
    assert 1 <= r0 < r1 <= grid.shape[0]-1 and 1 <= c0 < c1 <= grid.shape[1]-1
    board = grid[r0:r1, c0:c1]
    assert min(board.shape) >= 5
    aligned = board[::-1] if bottom else board
    aligned = aligned[:, ::-1] if right else aligned
    origin, horizontal, vertical = map(int, (aligned[0, 0], aligned[0, 1], aligned[1, 0]))
    assert len({bg, field, origin, horizontal, vertical}) == 5
    assert np.all(aligned[0, 1:] == horizontal) and np.all(aligned[1:, 0] == vertical)
    assert np.all(aligned[-1, 1:] == field) and np.all(aligned[1:, -1] == field)
    ring = grid[r0-1:r1+1, c0-1:c1+1]
    assert all(np.all(edge == bg) for edge in (ring[0], ring[-1], ring[:, 0], ring[:, -1]))
    mask = board != field
    mask[-1 if bottom else 0, :] = False
    mask[:, -1 if right else 0] = False
    boxes, axes = [], set()
    for cells in _components(mask):
        rr, cc = zip(*cells)
        a, b, c, d = min(rr), max(rr)+1, min(cc), max(cc)+1
        assert len(cells) == (b-a)*(d-c), 'Tickets must be full rectangles'
        assert 0 < a < b < board.shape[0] and 0 < c < d < board.shape[1]
        if b-a == 2 and d-c >= 3:
            axes.add(0)
        elif d-c == 2 and b-a >= 3:
            axes.add(1)
        else:
            raise AssertionError('Tickets have thickness two and length at least three')
        boxes.append((a+r0, b+r0, c+c0, d+c0))
    assert 2 <= len(boxes) <= 15 and len(axes) == 1
    axis = next(iter(axes))
    lane_reverse, fast_reverse = (right, bottom) if axis else (bottom, right)
    lane_coordinate, fast_coordinate = (2, 0) if axis else (0, 2)
    lanes = {}
    for box in boxes:
        lanes.setdefault(box[lane_coordinate], []).append(box)
    v = sorted(lanes)
    assert all(b-a >= 3 for a, b in zip(v, v[1:])), 'Lane bands must not overlap'
    ordered_boxes, populations = [], []
    for lane in sorted(lanes, reverse=lane_reverse):
        group = sorted(lanes[lane], key=lambda b: b[fast_coordinate], reverse=fast_reverse)
        intervals = sorted((b[fast_coordinate], b[fast_coordinate+1]) for b in group)
        assert all(b[0]-a[1] >= 1 for a, b in zip(intervals, intervals[1:]))
        ordered_boxes.extend(group)
        populations.append(len(group))
    rotation = (3 if bottom else 1) if axis else (2 if right else 0)
    blocks = [np.rot90(grid[a:b, c:d], rotation).copy() for a, b, c, d in ordered_boxes]
    assert len({b.shape[1] % 2 for b in blocks}) == 1
    assert all(b[0, 0] == b[1, 0] for b in blocks), 'Leading column is the category cap'
    keys = [int(b[0, 0]) for b in blocks]
    assert 1 <= len(set(keys)) <= 4 and not set(keys) & {field, bg}
    outside = np.ones(grid.shape, dtype=bool)
    outside[r0:r1, c0:c1] = False
    # A panel field is local: it cannot also occur in the exterior scene.
    assert not np.any((grid == field) & outside)
    return dict(field=field, background=bg, bounds=bounds, axis=axis, bottom=bottom,
                right=right, rotation=rotation, blocks=blocks, keys=keys,
                boxes=ordered_boxes, populations=populations, outside=outside,
                lane_coordinate=lane_coordinate, fast_coordinate=fast_coordinate,
                lane_reverse=lane_reverse, fast_reverse=fast_reverse,
                markers=(origin, horizontal, vertical))


def _inspect(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    assert all(np.all(edge == bg) for edge in (grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    candidates = []
    for field in map(int, np.unique(grid)):
        if field == bg:
            continue
        rr, cc = np.where(grid == field)
        fr0, fr1, fc0, fc1 = int(rr.min()), int(rr.max())+1, int(cc.min()), int(cc.max())+1
        for bottom in (False, True):
            for right in (False, True):
                # Field occupies every interior margin; one L-key row/column
                # lies just beyond its bounding box on the marked sides.
                bounds = (fr0-int(not bottom), fr1+int(bottom),
                          fc0-int(not right), fc1+int(right))
                try:
                    candidates.append(_board_candidate(grid, field, bounds, bottom, right, bg))
                except (AssertionError, IndexError):
                    pass
    assert len(candidates) == 1, 'Exactly one visibly keyed catalogue is required'
    m = candidates[0]
    objects, owner, serial = {}, np.full(grid.shape, -1, dtype=int), 0
    for color in sorted(set(map(int, grid[m['outside']])) - {bg}):
        objects[color] = _components((grid == color) & m['outside'])
        for cells in objects[color]:
            for r, c in cells:
                owner[r, c] = serial
            serial += 1
    for r, c in np.argwhere(owner >= 0):
        near = owner[max(0, r-1):r+2, max(0, c-1):c+2]
        assert np.all((near == -1) | (near == owner[r, c])), 'Exterior objects need a halo'
    counts = {color: len(parts) for color, parts in objects.items()}
    occurrence_counts = Counter(m['keys'])
    assert all(1 <= counts.get(c, 0) <= n for c, n in occurrence_counts.items()), 'Ordinal out of range'
    m.update(objects=objects, counts=counts,
             ignored=set(counts)-set(m['keys']), grid=grid)
    return m


def _chosen_indices(keys, counts, order=None):
    order = list(range(len(keys))) if order is None else order
    seen, chosen = Counter(), []
    for index in order:
        c = keys[index]
        seen[c] += 1
        if seen[c] == counts.get(c, 0):
            chosen.append(index)
    return chosen


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from the visible input, including all direction conventions."""
    m = _inspect(input_grid)
    indices = _chosen_indices(m['keys'], m['counts'])
    assert len(indices) == len(set(m['keys']))
    return _stack([m['blocks'][i] for i in indices], m['field'])


def _alternative(m, name):
    """Counterfactuals for evidence checks, never used to choose ground truth."""
    keys, blocks = m['keys'], m['blocks']
    counts, order = dict(m['counts']), list(range(len(keys)))
    if name == 'first':
        counts = dict.fromkeys(keys, 1)
    elif name == 'last':
        counts = Counter(keys)
    elif name in ('four_connected', 'area'):
        for c in set(keys):
            mask = (m['grid'] == c) & m['outside']
            counts[c] = len(_components(mask, False)) if name == 'four_connected' else int(mask.sum())
    elif name in ('reverse_fast', 'reverse_slow'):
        groups, start = [], 0
        for n in m['populations']:
            groups.append(order[start:start+n])
            start += n
        order = ([i for g in groups for i in g[::-1]] if name == 'reverse_fast'
                 else [i for g in groups[::-1] for i in g])
    elif name == 'fast_major':
        coord = m['fast_coordinate']
        order.sort(key=lambda i: (m['boxes'][i][coord], i), reverse=m['fast_reverse'])
    indices = _chosen_indices(keys, counts, order)
    if len(indices) != len(set(keys)):
        return np.empty((0, 0), dtype=np.int8)
    if name == 'sort_color':
        indices.sort(key=lambda i: keys[i])
    elif name == 'first_key_order':
        indices.sort(key=lambda i: keys.index(keys[i]))
    selected = [blocks[i][::-1] if name == 'reflect_rows' else blocks[i] for i in indices]
    width = max(b.shape[1] for b in blocks) if name == 'all_ticket_width' else None
    return _stack(selected, m['field'], width)


_SHORTCUTS = ('first', 'last', 'four_connected', 'area', 'reverse_fast',
              'reverse_slow', 'fast_major', 'sort_color', 'first_key_order',
              'reflect_rows', 'all_ticket_width')


def _evidence(training):
    for name in _SHORTCUTS:
        assert any(not np.array_equal(_alternative(m, name),
                                     _stack([m['blocks'][i] for i in _chosen_indices(m['keys'], m['counts'])],
                                            m['field'])) for m in training), 'Unexposed shortcut: '+name
    assert {m['axis'] for m in training} == {0, 1}
    assert any(any(1 < m['counts'][c] < m['keys'].count(c) for c in set(m['keys'])) for m in training)
    assert any(set(_chosen_indices(m['keys'], m['counts'])) !=
               set(_chosen_indices(m['keys'], m['counts'], list(range(len(m['keys'])))[::-1]))
               for m in training), 'Traversal must affect ticket identities as well as order'
    assert any(set(a['keys']) & b['ignored'] for a in training for b in training)
    assert any(len({b.shape[1] for b in [m['blocks'][i] for i in _chosen_indices(m['keys'], m['counts'])]}) > 1
               for m in training), 'Centering needs selected tickets of unequal length'


def _glyph(rng):
    mode = int(rng.integers(8))
    if mode < 5:
        cells = (_DIAMOND, _SQUARE, _CROSS, _X, _SINGLE)[mode]
    elif mode == 5:
        cells = tuple((0, i) for i in range(int(rng.integers(2, 6))))
    else:
        side = int(rng.integers(3, 5))
        target = int(rng.integers(2, min(10, side*side)+1))
        cells = {(side//2, side//2)}
        while len(cells) < target:
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                               if 0 <= r+dr < side and 0 <= c+dc < side} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    if rng.integers(2):
        cells = [(r, -c) for r, c in cells]
    for _ in range(int(rng.integers(4))):
        cells = [(-c, r) for r, c in cells]
    return _normalize(cells)


def _gaps(rng, occupied, items, maximum):
    minimum = occupied + items + 1
    assert minimum <= maximum
    extra = int(rng.integers(min(5, maximum-minimum)+1))
    gaps = 1 + rng.multinomial(extra, rng.dirichlet(np.ones(items+1)))
    return gaps.tolist(), minimum + extra


def _body(rng, length, color, pool):
    local = rng.choice(pool, size=int(rng.integers(1, min(5, len(pool))+1)), replace=False)
    mode = int(rng.integers(4))
    if mode == 0:
        a = np.full((2, length), rng.choice(local), dtype=np.int8)
    elif mode == 1:
        a = np.repeat(rng.choice(local, size=(2, 1)), length, axis=1).astype(np.int8)
    elif mode == 2:
        motif = rng.choice(local, size=(2, int(rng.integers(2, min(4, length)+1))))
        a = np.tile(motif, (1, length))[:, :length].astype(np.int8)
    else:
        a = rng.choice(local, size=(2, length)).astype(np.int8)
    a[:, 0] = color
    return a


def _make_catalogue(rng, k, witness, rank_four, config, ignored_color=None):
    structural_pool = [c for c in range(10) if c != ignored_color]
    bg, field = map(int, rng.choice(structural_pool, 2, replace=False))
    origin, horizontal, vertical = map(int, rng.choice([c for c in range(10) if c not in (bg, field)], 3, replace=False))
    category_pool = [c for c in range(10) if c not in (bg, field, ignored_color)]
    colors = list(map(int, rng.choice(category_pool, k, replace=False)))
    occurrences = [int(rng.integers(2, 5)) for _ in range(k)]
    if witness == 1:
        occurrences[0] = 4
    elif witness == 2:
        occurrences[0] = max(3, occurrences[0])
    if rank_four:
        occurrences[0] = int(rng.integers(4, 6))
    while sum(occurrences) > 12:
        eligible = [i for i, n in enumerate(occurrences) if n > (4 if i == 0 and (rank_four or witness == 1) else 2)]
        occurrences[int(rng.choice(eligible))] -= 1
    counts = [int(rng.integers(1, min(n, 3)+1)) for n in occurrences]
    if witness:
        counts[0] = 3 if witness == 1 else 2
    if rank_four:
        counts[0] = 4
    while sum(counts) > 11:
        eligible = [i for i, n in enumerate(counts) if n > 1 and not (rank_four and i == 0)]
        counts[int(rng.choice(eligible))] -= 1
    ordinal_counts = dict(zip(colors, counts))
    sequence = [c for c, n in zip(colors, occurrences) for _ in range(n)]
    for _ in range(128):
        rng.shuffle(sequence)
        chosen = _chosen_indices(sequence, ordinal_counts)
        first_order = list(dict.fromkeys(sequence))
        winning_order = [sequence[i] for i in chosen]
        if not witness or (winning_order != first_order and winning_order != sorted(colors)):
            break
    else:
        # There is always an interleaving with the first color's third ticket
        # after another color's winner. A fresh palette/sequence is simpler.
        raise ValueError('Rare catalogue ordering collision')
    n = len(sequence)
    base = int(rng.choice((3, 4)))
    capacity = 3 if witness else (5 if base == 3 else 4)
    low = max(2 if witness else 1, (n+capacity-1)//capacity)
    lane_count = int(rng.integers(low, min(4, n)+1))
    populations = [1]*lane_count
    for _ in range(n-lane_count):
        eligible = [i for i, c in enumerate(populations) if c < capacity]
        populations[int(rng.choice(eligible))] += 1
    lengths = [base]*n
    if witness:
        # A discarded global maximum and unequal retained widths jointly
        # expose filtering-before-centering and symmetric padding.
        lengths[int(rng.choice(chosen))] += 2
        lengths[int(rng.choice([i for i in range(n) if i not in chosen]))] += 4
    start = 0
    for pop in populations:
        for _ in range(int(rng.integers(0, 5))):
            eligible = [i for i in range(start, start+pop)
                        if lengths[i] < base+6 and (not witness or i not in chosen)]
            if eligible and sum(lengths[start:start+pop])+pop+3 <= 24:
                lengths[int(rng.choice(eligible))] += 2
        start += pop
    # Use one common fast extent, while distributing each lane's free space.
    groups, start = [], 0
    for pop in populations:
        groups.append(lengths[start:start+pop])
        start += pop
    fast_extent = max(sum(g)+len(g)+1 for g in groups)
    fast_extent += int(rng.integers(min(4, 24-fast_extent)+1))
    slow_gaps = [int(rng.integers(1, 3)) for _ in range(lane_count+1)]
    slow_extent = 2*lane_count + sum(slow_gaps)
    assert slow_extent <= 18 and fast_extent <= 24
    pool = [c for c in range(10) if c != field]
    blocks = []
    for length, color in zip(lengths, sequence):
        for _ in range(32):
            b = _body(rng, length, color, pool)
            # Different occurrences of a category should be addressable.
            if all(not np.array_equal(b, old) for old, old_c in zip(blocks, sequence) if old_c == color):
                break
        else:
            b = rng.choice(pool, size=(2, length)).astype(np.int8)
            b[:, 0] = color
        blocks.append(b)
    if witness:
        i = int(rng.choice(chosen))
        b = blocks[i]
        a, z = map(int, rng.choice([c for c in pool if c != sequence[i]], 2, replace=False))
        b[0, 1], b[1, 1], b[0, -1] = a, z, z
    board = np.full((slow_extent+1, fast_extent+1), field, dtype=np.int8)
    board[0, 0], board[0, 1:], board[1:, 0] = origin, horizontal, vertical
    rotation, reflection = config
    v, index = 1+slow_gaps[0], 0
    for lane, group in enumerate(groups):
        minimum = sum(group)+len(group)+1
        extra = fast_extent-minimum
        gaps = 1+rng.multinomial(extra, rng.dirichlet(np.ones(len(group)+1)))
        u = 1+int(gaps[0])
        for length in group:
            b = blocks[index]
            board[v:v+2, u:u+length] = b[::-1] if reflection else b
            u += length+int(gaps[index-sum(populations[:lane])+1])
            index += 1
        v += 2+slow_gaps[lane+1]
    if reflection:
        board = board[::-1]
    board = np.rot90(board, rotation).copy()
    labels = _stack([blocks[i] for i in chosen], field)
    return board, labels, bg, field, colors, ordinal_counts


def _place(grid, blocked, color, shape, rng):
    sh, sw = 1+max(r for r, c in shape), 1+max(c for r, c in shape)
    nh, nw = grid.shape[0]-sh+1, grid.shape[1]-sw+1
    if min(nh, nw) <= 0:
        return False
    allowed = np.ones((nh, nw), dtype=bool)
    for r, c in shape:
        allowed &= ~blocked[r:r+nh, c:c+nw]
    candidates = np.argwhere(allowed)
    if not len(candidates):
        return False
    r0, c0 = map(int, candidates[int(rng.integers(len(candidates)))])
    for dr, dc in shape:
        r, c = r0+dr, c0+dc
        grid[r, c] = color
        blocked[max(0, r-1):r+2, max(0, c-1):c+2] = True
    return True


def _scene(rng, board, bg, objects):
    bh, bw = board.shape
    for attempt in range(9):
        fallback = attempt == 8
        if attempt >= 6:
            h = w = 30
        else:
            h = int(rng.integers(max(16, bh+2), 31))
            w = int(rng.integers(max(16, bw+2), 31))
        r0 = 1 if fallback else int(rng.integers(1, h-bh))
        c0 = 1 if fallback else int(rng.integers(1, w-bw))
        grid = np.full((h, w), bg, dtype=np.int8)
        grid[r0:r0+bh, c0:c0+bw] = board
        blocked = np.zeros((h, w), dtype=bool)
        blocked[0] = blocked[-1] = True
        blocked[:, 0] = blocked[:, -1] = True
        blocked[r0-1:r0+bh+1, c0-1:c0+bw+1] = True
        if fallback:
            # A <=25 by <=19 board at (1,1) leaves >=14 separated
            # three-by-three slots; the exterior budget is at most 13 objects.
            slots = [(r, c) for r in range(1, 26, 4) for c in range(1, 26, 4)
                     if not blocked[r:r+3, c:c+3].any()]
            assert len(slots) >= len(objects)
            rng.shuffle(slots)
            for (color, shape), (r, c) in zip(objects, slots):
                if max(a for a, b in shape) >= 3 or max(b for a, b in shape) >= 3:
                    shape = _CROSS
                for dr, dc in shape:
                    grid[r+dr, c+dc] = color
            return grid
        if all(_place(grid, blocked, color, shape, rng) for color, shape in objects):
            return grid
    raise AssertionError('Unreachable packing failure')


def _example(rng, k, witness, rank_four, config, ignored_color=None):
    board, output, bg, field, colors, counts = _make_catalogue(rng, k, witness, rank_four, config, ignored_color)
    objects = []
    for i, c in enumerate(colors):
        for j in range(counts[c]):
            if witness == 1 and i == 0:
                shape = (_DIAMOND, _SQUARE, _SINGLE)[j]
            elif witness == 2 and i == 0:
                shape = (_X, _SINGLE)[j]
            else:
                shape = _glyph(rng)
            objects.append((c, shape))
    available = [c for c in range(10) if c not in (bg, field) and c not in colors]
    rng.shuffle(available)
    if ignored_color is not None:
        available.remove(ignored_color)
        available.insert(0, ignored_color)
    nd = int(rng.integers(0, 3)) if ignored_color is None else int(rng.integers(1, 3))
    for j, c in enumerate(available[:nd]):
        amount = int(rng.integers(1, min(4, 13-len(objects)-(nd-j-1))+1))
        for _ in range(amount):
            objects.append((c, _glyph(rng)))
    assert len(objects) <= 13
    rng.shuffle(objects)
    objects.sort(key=lambda item: -(max(r for r, c in item[1])+1)*(max(c for r, c in item[1])+1))
    grid = _scene(rng, board, bg, objects)
    assert np.array_equal(solve(grid), output), 'Visible scene differs from construction'
    return {'input': grid, 'output': output}, colors[0]


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < low or (high is not None and value > high)):
        raise ValueError(name+' must be an integer in its legal range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    nt = _integer(num_train, 'num_train', 2, 10)
    nq = _integer(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must be greater than num_test')
    rng = np.random.default_rng(seed)
    # Sampling choices affect examples, never the interpretation of an input.
    shift = int(rng.integers(3))  # ordinal extrapolation, categories, new frame combination
    first = (int(rng.integers(4)), bool(rng.integers(2)))
    second = ((first[0]+int(rng.choice((1, 3)))) % 4, not first[1])
    for _ in range(64):
        try:
            a, color = _example(rng, int(rng.integers(2, 4)), 1, False, first)
            b, _ = _example(rng, int(rng.integers(2, 4)), 2, False, second, color)
            training = [_inspect(a['input']), _inspect(b['input'])]
            _evidence(training)
            break
        except (AssertionError, ValueError):
            continue
    else:
        raise RuntimeError('Could not expose all catalogue conventions within the retry bound')
    puzzle, seen = {'train': [a, b], 'test': []}, {_fingerprint(a['input']), _fingerprint(b['input'])}
    all_configs = [(r, f) for r in range(4) for f in (False, True)]
    held = [c for c in all_configs if c not in (first, second)]
    if shift == 2:
        rng.shuffle(held)
        held = held[:int(rng.integers(1, 4))]
    for split, amount in (('train', nt-2), ('test', nq)):
        for index in range(amount):
            query = split == 'test'
            k = 4 if query and shift == 1 else int(rng.integers(2, 4))
            configs = all_configs
            if shift == 2:
                configs = held if query else [c for c in all_configs if c not in held]
            config = configs[int(rng.integers(len(configs)))]
            for _ in range(24):
                try:
                    pair, _ = _example(rng, k, 0, query and shift == 0, config)
                except ValueError:
                    continue
                identity = _fingerprint(pair['input'])
                if identity not in seen:
                    break
            else:
                raise RuntimeError('Distinct example retry bound exhausted')
            seen.add(identity)
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check schema, visible geometry, count addressing, and training evidence."""
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
        assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
        nt, nq = len(puzzle['train']), len(puzzle['test'])
        assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
        seen, training = set(), []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
                inp, out = _grid(pair['input']), _grid(pair['output'])
                identity = _fingerprint(inp)
                assert identity not in seen, 'Repeated input'
                seen.add(identity)
                m = _inspect(inp)
                indices = _chosen_indices(m['keys'], m['counts'])
                assert len(indices) == len(set(m['keys']))
                expected = _stack([m['blocks'][i] for i in indices], m['field'])
                assert np.array_equal(expected, out), 'Wrong ordinal, order, rotation, or centering'
                assert out.shape == (2*len(set(m['keys'])), max(m['blocks'][i].shape[1] for i in indices))
                assert not np.array_equal(inp, out)
                if split == 'train':
                    training.append(m)
        _evidence(training)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError('Malformed puzzle: '+str(exc)) from exc


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds, every explicit count combination, mutations, and equivariance.

    One puzzle is retained at a time; no execution traces or sample files.
    Returns compact, reproducible diversity measurements for optional reporting.
    """
    # Direct boundary members outside the sampler's catalogue/count range.
    # Fifteen same-cap tickets include repeated payloads; fifteen external
    # singletons address the last occurrence without deduplication.
    manual = np.zeros((30, 30), dtype=np.int8)
    board = np.full((14, 18), 8, dtype=np.int8)
    board[0, 0], board[0, 1:], board[1:, 0] = 9, 1, 2
    palette = [0, 1, 2, 3, 4, 5, 6, 7, 9]
    for j in range(15):
        r, c = 2+3*(j//4), 2+4*(j % 4)
        board[r:r+2, c:c+3] = [[3, palette[(j+1) % 9], palette[(j+2) % 9]],
                              [3, palette[(j+3) % 9], palette[(j+5) % 9]]]
    manual[1:15, 1:19] = board
    for r in (19, 22, 25):
        for c in (2, 5, 8, 11, 14):
            manual[r, c] = 3
    expected = np.array([[3, 6, 7], [3, 9, 1]], dtype=np.int8)
    for rotation in range(4):
        assert np.array_equal(solve(np.rot90(manual, rotation)), expected)
    # A long retained strip requires eleven padding cells on each side of
    # another retained ticket. Input dimensions attain the 30-cell boundary.
    manual.fill(0)
    board = np.full((8, 28), 8, dtype=np.int8)
    board[0, 0], board[0, 1:], board[1:, 0] = 9, 1, 2
    board[2:4, 2:27] = [[3]+[5]*24, [3]+[6]*24]
    board[5:7, 2:5] = [[4, 7, 1], [4, 2, 9]]
    manual[1:9, 1:29] = board
    manual[15, 5], manual[20, 10] = 3, 4
    expected = np.full((4, 25), 8, dtype=np.int8)
    expected[:2] = [[3]+[5]*24, [3]+[6]*24]
    expected[2:, 11:14] = [[4, 7, 1], [4, 2, 9]]
    assert np.array_equal(solve(manual), expected)
    signatures, input_shapes, output_shapes, configs = set(), set(), set(), set()
    layouts, counts, categories, backgrounds, fields, glyphs, widths = set(), set(), set(), set(), set(), set(), set()
    pairs = 0
    for seed in range(42):
        nt = 2+seed % 9
        nq = 1+seed % min(4, nt-1)
        p = generate_puzzle(seed, nt, nq)
        signatures.add(_puzzle_digest(p))
        for split in ('train', 'test'):
            for pair in p[split]:
                m = _inspect(pair['input'])
                input_shapes.add(pair['input'].shape)
                output_shapes.add(pair['output'].shape)
                configs.add((m['axis'], m['bottom'], m['right']))
                layouts.add((tuple(m['populations']), tuple(b.shape[1] for b in m['blocks'])))
                categories.add(len(set(m['keys'])))
                counts.update(m['counts'][c] for c in set(m['keys']))
                backgrounds.add(m['background'])
                fields.add(m['field'])
                widths.update(b.shape[1] for b in m['blocks'])
                glyphs.update(_normalize(cells) for parts in m['objects'].values() for cells in parts)
                assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
                pairs += 1
        if seed < 3:
            assert _puzzle_digest(p) == _puzzle_digest(generate_puzzle(seed, nt, nq))
            query = p['test'][0]
            for rotation in range(4):
                assert np.array_equal(solve(np.rot90(query['input'], rotation)), query['output'])
            # Any color permutation preserves geometric roles and equality.
            palette = np.random.default_rng(seed+100).permutation(10)
            assert np.array_equal(solve(palette[query['input']]), palette[query['output']])
    assert len(signatures) == 42 and len(configs) == 8
    assert len(input_shapes) >= 35 and len(output_shapes) >= 10 and len(layouts) >= 100
    assert categories == {2, 3, 4} and counts == {1, 2, 3, 4}
    assert backgrounds == fields == set(range(10))
    assert len(glyphs) >= 50 and len(widths) >= 7
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(12345, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (3.0, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError('Illegal counts accepted')
    bad = generate_puzzle(101, 2, 1)
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Damaged output accepted')
    bad = generate_puzzle(102, 2, 1)
    bad['test'][0] = bad['train'][0]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Duplicate input accepted')
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(signatures), configurations=len(configs),
                input_shapes=len(input_shapes), output_shapes=len(output_shapes), layouts=len(layouts),
                glyphs=len(glyphs), ticket_lengths=sorted(widths), category_counts=sorted(categories),
                ordinal_counts=sorted(counts))
