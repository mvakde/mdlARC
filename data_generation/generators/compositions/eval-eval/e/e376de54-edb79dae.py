"""Elastic Stencil Loom: input-only parsing and bounded structural sampling.

Only standard-library modules and NumPy are used. No parent code, stored pairs,
seed state or training labels are consulted by solve.
"""
from __future__ import annotations

import numpy as np


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "bad dimensions"
    assert np.issubdtype(a.dtype, np.integer), "noninteger cells"
    assert np.all((0 <= a) & (a <= 9)), "bad palette"
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _puzzle_key(puzzle):
    return tuple((_key(p['input']), _key(p['output']))
                 for split in ('train', 'test') for p in puzzle[split])


def _box(cells):
    a = np.asarray(cells)
    return (int(a[:, 0].min()), int(a[:, 0].max()) + 1,
            int(a[:, 1].min()), int(a[:, 1].max()) + 1)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        stack, cells = [first], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in unseen:
                    unseen.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _frame(grid, color, box):
    r, e, c, d = box
    grid[r, c:d] = grid[e-1, c:d] = color
    grid[r:e, c] = grid[r:e, d-1] = color


def _separated(a, b):
    r, e, c, d = a
    rr, ee, cc, dd = b
    return e < rr or ee < r or d < cc or dd < c


def _outside(box, frame):
    r, e, c, d = box
    rr, ee, cc, dd = frame
    return e <= rr or r >= ee or d <= cc or c >= dd


def _strokes(interior, bg):
    cells = np.argwhere(interior != bg)
    assert len(cells), "empty board"
    candidates = []
    # axis=1 means horizontal (axial coordinate is column).
    for axis in (1, 0):
        transverse = 1-axis
        levels = sorted(set(map(int, cells[:, transverse])))
        if len(levels) not in (3, 5, 7):
            continue
        if min(np.diff(levels)) < 2:
            continue
        groups, valid = [], True
        for level in levels:
            group = cells[cells[:, transverse] == level]
            group = group[np.argsort(group[:, axis])]
            values = interior[group[:, 0], group[:, 1]]
            coords = group[:, axis]
            if len(set(map(int, values))) != 1 or np.any(np.diff(coords) != 1):
                valid = False
                break
            groups.append(dict(level=level, start=int(coords[0]),
                               end=int(coords[-1])+1, length=len(group),
                               source=int(values[0])))
        if not valid:
            continue
        left = len({g['start'] for g in groups}) == 1
        right = len({g['end'] for g in groups}) == 1
        if left == right:
            continue
        length = groups[len(groups)//2]['length']
        if length < 3:
            continue
        for g in groups:
            g['anchor'] = g['start'] if left else g['end']-1
        candidates.append(dict(axis=axis, forward=left, strokes=groups,
                               reference=length))
    assert len(candidates) == 1, "ambiguous skeleton array"
    return candidates[0]


def _parse(input_grid):
    grid = _grid(input_grid)
    frames = []
    for color in np.unique(grid):
        cells = np.argwhere(grid == color)
        r, e, c, d = _box(cells)
        h, w = e-r, d-c
        if h < 5 or w < 5 or len(cells) != 2*h+2*w-4:
            continue
        if (np.all(grid[r, c:d] == color) and np.all(grid[e-1, c:d] == color)
                and np.all(grid[r:e, c] == color)
                and np.all(grid[r:e, d-1] == color)):
            frames.append(((r, e, c, d), int(color)))
    assert len(frames) == 1, "need an exclusive complete frame"
    frame, frame_color = frames[0]
    r, e, c, d = frame
    corners = [grid[r+1, c+1], grid[r+1, d-2],
               grid[e-2, c+1], grid[e-2, d-2]]
    assert len(set(map(int, corners))) == 1
    bg = int(corners[0])
    assert bg != frame_color
    a = _strokes(grid[r+1:e-1, c+1:d-1], bg)
    # Convert stroke coordinates from interior to inclusive cropped-board space.
    for g in a['strokes']:
        for name in ('level', 'start', 'end', 'anchor'):
            g[name] += 1
    sources = {g['source'] for g in a['strokes']}
    assert 1 <= len(sources) <= 4 and not sources & {bg, frame_color}
    exterior = grid != bg
    exterior[r:e, c:d] = False
    key_cells = np.zeros(grid.shape, bool)
    mapping, key_boxes = {}, {}
    for cells in _components(exterior):
        values, counts = np.unique([grid[x, y] for x, y in cells], return_counts=True)
        if len(values) == 1:
            continue
        box = _box(cells)
        rr, ee, cc, dd = box
        assert (ee-rr, dd-cc) == (2, 2) and len(cells) == 4
        assert len(values) == 2 and sorted(map(int, counts)) == [1, 3]
        source = int(values[np.argmin(counts)])
        target = int(values[np.argmax(counts)])
        assert source in sources and source not in mapping
        assert target not in (bg, frame_color)
        assert _outside(box, frame)
        mapping[source], key_boxes[source] = target, box
        key_cells[rr:ee, cc:dd] = True
    assert set(mapping) == sources, "incomplete direct paint table"
    remaining = exterior & ~key_cells
    assert set(map(int, np.unique(grid[remaining]))) == sources
    prototypes, prototype_boxes = {}, {}
    for source in sorted(sources):
        box = _box(np.argwhere(remaining & (grid == source)))
        rr, ee, cc, dd = box
        mask = grid[rr:ee, cc:dd] == source
        assert np.all(np.isin(grid[rr:ee, cc:dd], [source, bg]))
        assert mask.shape[a['axis']] == 3
        assert mask.shape[1-a['axis']] in (1, 3, 5)
        assert mask[0].any() and mask[-1].any()
        assert mask[:, 0].any() and mask[:, -1].any()
        assert _outside(box, frame)
        prototypes[source], prototype_boxes[source] = mask.copy(), box
    boxes = list(key_boxes.values()) + list(prototype_boxes.values())
    for i, box in enumerate(boxes):
        # Rectangle separation is stronger than active-pixel separation.
        assert _separated(box, frame)
        assert all(_separated(box, b) for b in boxes[i+1:])
    a.update(background=bg, frame_color=frame_color, frame_box=frame,
             shape=(e-r, d-c), mapping=mapping, prototypes=prototypes,
             prototype_boxes=prototype_boxes, key_boxes=key_boxes)
    footprints = [_footprint(a, g, a['reference']) for g in a['strokes']]
    h, w = a['shape']
    for i, (rr, ee, cc, dd) in enumerate(footprints):
        assert 2 <= rr < ee <= h-2 and 2 <= cc < dd <= w-2, "ribbon boundary"
        assert all(_separated(footprints[i], b) for b in footprints[i+1:]), "ribbon collision"
    a['footprints'] = footprints
    return a


def _footprint(a, stroke, length):
    width = a['prototypes'][stroke['source']].shape[1-a['axis']]
    t = stroke['level'] - width//2
    x = stroke['anchor'] if a['forward'] else stroke['anchor']-length+1
    return (t, t+width, x, x+length) if a['axis'] == 1 else (x, x+length, t, t+width)


def _elastic(mask, length, axis):
    assert length >= 3
    return np.concatenate((np.take(mask, [0], axis=axis),
                           np.repeat(np.take(mask, [1], axis=axis), length-2, axis=axis),
                           np.take(mask, [2], axis=axis)), axis=axis)


def _render(a, *, own_lengths=False, thin=False, source_paint=False):
    out = np.full(a['shape'], a['background'], dtype=np.int8)
    _frame(out, a['frame_color'], (0, out.shape[0], 0, out.shape[1]))
    for g in a['strokes']:
        length = max(3, g['length']) if own_lengths else a['reference']
        target = g['source'] if source_paint else a['mapping'][g['source']]
        if thin:
            axial = g['anchor'] if a['forward'] else g['anchor']-length+1
            if a['axis'] == 1:
                out[g['level'], axial:axial+length] = target
            else:
                out[axial:axial+length, g['level']] = target
        else:
            r, e, c, d = _footprint(a, g, length)
            assert 0 <= r < e <= out.shape[0] and 0 <= c < d <= out.shape[1]
            mask = _elastic(a['prototypes'][g['source']], length, a['axis'])
            out[r:e, c:d] = np.where(mask, target, a['background'])
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every convention and geometric parameter from this input only."""
    return _render(_parse(input_grid))


def _mask(rng, width, informative=False):
    if informative:
        # Randomly reflected version of a legible asymmetric elastic grammar.
        m = np.array([[1, 0, 0], [1, 1, 1], [0, 1, 1]], bool)
        return (np.flipud(m) if rng.integers(2) else m).copy()
    kind = int(rng.integers(7))
    y, x = np.indices((width, 3))
    if kind == 0:
        m = rng.random((width, 3)) < rng.uniform(.2, .85)
    elif kind == 1:
        m = (x == 0) | (x == 2) | (y == width//2)
    elif kind == 2:
        m = (x+y) % 2 == 0
    elif kind == 3:
        m = (y == 0) | (y == width-1) | (x == 1)
    elif kind == 4:
        m = np.ones((width, 3), bool)
    elif kind == 5:
        m = np.zeros((width, 3), bool)
        m[0, 0] = m[-1, 2] = True  # Disconnected caps; empty elastic core.
    else:
        m = (y == width//2) | ((x == 1) & (y % 2 == 0))
    for edge in (m[0], m[-1], m[:, 0], m[:, -1]):
        if not edge.any():
            edge[int(rng.integers(len(edge)))] = True
    return m


def _mapping(rng, sources, available, kind):
    order = list(map(int, rng.permutation(sources)))
    other = [c for c in available if c not in sources]
    if len(sources) == 1:
        kind = 'disjoint'
    if kind == 'disjoint':
        targets = list(map(int, rng.choice(other, len(sources), replace=len(sources) > len(other))))
        return dict(zip(sources, targets))
    if kind == 'chain':
        return {s: order[i+1] if i+1 < len(order) else int(rng.choice(other))
                for i, s in enumerate(order)}
    if kind == 'cycle':
        return {s: order[(i+1) % len(order)] for i, s in enumerate(order)}
    if kind == 'many':
        target = int(rng.choice(other))
        return {s: target for s in sources}
    return {s: int(rng.choice([c for c in available if c != s])) for s in sources}


def _reserve(occupied, box):
    r, e, c, d = box
    occupied[max(0, r-1):min(len(occupied), e+1),
             max(0, c-1):min(occupied.shape[1], d+1)] = True


def _free_positions(occupied, h, w):
    sums = np.pad(occupied.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    totals = sums[h:, w:] - sums[:-h, w:] - sums[h:, :-w] + sums[:-h, :-w]
    return np.argwhere(totals == 0)


def _pack(rng, board, tiles, bg):
    bh, bw = board.shape
    if rng.random() < .70:
        for _ in range(8):
            fr, fc = int(rng.integers(31-bh)), int(rng.integers(31-bw))
            occupied = np.zeros((30, 30), bool)
            _reserve(occupied, (fr, fr+bh, fc, fc+bw))
            grid = np.full((30, 30), bg, np.int8)
            grid[fr:fr+bh, fc:fc+bw] = board
            order = list(map(int, rng.permutation(len(tiles))))
            order.sort(key=lambda i: -tiles[i][2].size)
            placed = True
            for i in order:
                tile = tiles[i][2]
                h, w = tile.shape
                places = _free_positions(occupied, h, w)
                if not len(places):
                    placed = False
                    break
                r, c = map(int, places[int(rng.integers(len(places)))])
                grid[r:r+h, c:c+w] = tile
                _reserve(occupied, (r, r+h, c, c+w))
            if placed:
                r, e, c, d = _box(np.argwhere(grid != bg))
                r = max(0, r-int(rng.integers(3)))
                c = max(0, c-int(rng.integers(3)))
                e = min(30, e+int(rng.integers(3)))
                d = min(30, d+int(rng.integers(3)))
                return grid[r:e, c:d].copy()
    # Guaranteed side-bank packing; prototype and key order are independent.
    prototypes = [x for x in tiles if x[0] == 'prototype']
    keys = [x for x in tiles if x[0] == 'key']
    rng.shuffle(prototypes)
    rng.shuffle(keys)
    if len(keys) > 1 and [x[1] for x in prototypes] == [x[1] for x in keys]:
        keys = keys[1:] + keys[:1]
    proto_height = sum(t.shape[0] for _, _, t in prototypes) + len(prototypes)-1
    h = min(30, max(bh, proto_height, 3*len(keys)-1) + int(rng.integers(3)))
    left = int(rng.integers(min(2, 30-bw-7)+1))
    w = min(30, left+bw+7+int(rng.integers(3)))
    grid = np.full((h, w), bg, np.int8)
    top = int(rng.integers(h-bh+1))
    grid[top:top+bh, left:left+bw] = board
    for bank, col in ((prototypes, left+bw+1), (keys, left+bw+5)):
        used = sum(t.shape[0] for _, _, t in bank) + len(bank)-1
        slack = rng.multinomial(h-used, np.full(len(bank)+1, 1/(len(bank)+1)))
        row = int(slack[0])
        for i, (_, _, tile) in enumerate(bank):
            th, tw = tile.shape
            grid[row:row+th, col:col+tw] = tile
            row += th+1+int(slack[i+1])
    return grid


def _make_pair(rng, shift, split, index, base, avoid_reference=None):
    training = split == 'train'
    witness = training and index == 0
    second = training and index == 1
    count = 5 if witness else 3 if second else 7 if not training and shift == 'count' else int(rng.choice([3, 5]))
    if not training and shift == 'thickness':
        count = 3
    refs = list(range(4, 7)) if witness or (training and shift == 'length') else list(range(3, 10))
    if not training and shift == 'length':
        refs = list(range(8, 12))
    if second:
        refs = [x for x in refs if x != avoid_reference]
    reference = int(rng.choice(refs))
    colors = list(map(int, rng.permutation(10)))
    bg, frame_color = colors[:2]
    available = colors[2:]
    categories = int(rng.integers(2, 4)) if witness else 2 if second else int(rng.integers(1, min(4, count)+1))
    if not training and shift == 'mapping':
        categories = int(rng.integers(2, min(4, count)+1))
    sources = available[:categories]
    assignments = sources + list(map(int, rng.choice(sources, count-categories)))
    rng.shuffle(assignments)
    kind = 'disjoint' if witness else 'chain' if second else str(rng.choice(['disjoint', 'chain', 'cycle', 'many', 'mixed']))
    if training and shift == 'mapping' and not (witness or second):
        kind = str(rng.choice(['disjoint', 'chain']))
    if not training and shift == 'mapping':
        kind = str(rng.choice(['cycle', 'many']))
    mapping = _mapping(rng, sources, available, kind)
    width_pool = [1, 3] if count == 7 or (training and shift == 'thickness') else [1, 3, 5]
    widths = {s: 3 if witness else int(rng.choice(width_pool)) for s in sources}
    if not training and shift == 'thickness':
        widths[int(rng.choice(sources))] = 5
    # Fitting is based on expanded footprints, not just thin skeleton spacing.
    while sum(widths[s] for s in assignments) + count+3 > 28:
        s = max(sources, key=lambda s: widths[s])
        assert widths[s] > 1
        widths[s] -= 2
    masks = {}
    for i, s in enumerate(sources):
        reusable = [t for t in sources[:i] if widths[t] == widths[s]]
        if reusable and not witness and rng.random() < .20:
            masks[s] = masks[int(rng.choice(reusable))].copy()
        else:
            masks[s] = _mask(rng, widths[s], informative=witness and i == 0)
    middle = count//2
    lengths = [reference]*count
    others = [int(i) for i in rng.permutation(count) if i != middle]
    if witness:
        for i in others:
            lengths[i] = int(rng.integers(3, reference))
        lengths[others[0]] = int(rng.integers(reference+1, min(15, reference+4)+1))
    else:
        mode = str(rng.choice(['mixed', 'extend', 'trim', 'one']))
        for i in others:
            if mode == 'one' and i != others[0] or rng.random() < .25:
                continue
            if mode == 'extend' or (mode != 'trim' and rng.random() < .5):
                lengths[i] = int(rng.integers(1, reference))
            else:
                lengths[i] = int(rng.integers(reference+1, min(15, reference+4)+1))
        if all(x == reference for x in lengths):
            lengths[others[0]] = reference+1
    halves = [widths[s]//2 for s in assignments]
    gaps = [halves[i]+halves[i+1]+2 for i in range(count-1)]
    if witness:
        gaps[0 if rng.integers(2) else -1] += 5
    min_h = sum(gaps) + halves[0]+halves[-1]+5
    assert min_h <= 30
    extra = int(rng.integers(min(7, 30-min_h)+1))
    pad = rng.multinomial(extra, np.full(count+1, 1/(count+1)))
    # Keep the witness's decisive gap vector intact, varying exterior margins.
    if witness:
        pad = np.zeros(count+1, int)
        pad[0] = extra//2
        pad[-1] = extra-extra//2
    positions = [2+halves[0]+int(pad[0])]
    for i, gap in enumerate(gaps):
        positions.append(positions[-1]+gap+int(pad[i+1]))
    bh = positions[-1]+halves[-1]+3+int(pad[-1])
    anchor = 2+int(rng.integers(3))
    bw = anchor+max(lengths)+2+int(rng.integers(2))
    assert bh <= 30 and bw <= 22
    board = np.full((bh, bw), bg, np.int8)
    output = board.copy()
    _frame(board, frame_color, (0, bh, 0, bw))
    _frame(output, frame_color, (0, bh, 0, bw))
    for row, s, length in zip(positions, assignments, lengths):
        board[row, anchor:anchor+length] = s
        # Labels come directly from sampled geometry, not from input parsing.
        mask = masks[s]
        expanded = np.column_stack((mask[:, 0],
                                    np.repeat(mask[:, 1:2], reference-2, axis=1),
                                    mask[:, 2]))
        top = row-widths[s]//2
        output[top:top+widths[s], anchor:anchor+reference] = np.where(expanded, mapping[s], bg)
    tiles = []
    for s in sources:
        tiles.append(('prototype', s, np.where(masks[s], s, bg).astype(np.int8)))
        tile = np.full((2, 2), mapping[s], np.int8)
        tile.flat[int(rng.integers(4))] = s
        tiles.append(('key', s, tile))
    inp = _pack(rng, board, tiles, bg)
    # Transform the whole scene, including displayed prototypes, and the crop.
    rotation = base if shift == 'direction' and training else (base+int(rng.choice([1, 2, 3]))) % 4 if shift == 'direction' else int(rng.integers(4))
    inp, output = np.rot90(inp, rotation), np.rot90(output, rotation)
    if shift != 'direction' and rng.integers(2):
        inp, output = np.fliplr(inp), np.fliplr(output)
    return {'input': inp.copy(), 'output': output.copy()}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    train_count = _count(num_train, 'num_train', 2, 10)
    test_count = _count(num_test, 'num_test', 1, 4)
    if train_count <= test_count:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(['length', 'count', 'thickness', 'mapping', 'direction']))
    base = int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()
    first_reference = None
    for split, amount in (('train', train_count), ('test', test_count)):
        for i in range(amount):
            for _ in range(32):
                pair = _make_pair(rng, shift, split, i, base, first_reference)
                fingerprint = _key(pair['input'])
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    puzzle[split].append(pair)
                    if split == 'train' and i == 0:
                        first_reference = _parse(pair['input'])['reference']
                    break
            else:
                raise RuntimeError('bounded unique-input sampling failed')
    validate_generated_puzzle(puzzle)
    return puzzle


def _geometric_witness(a):
    levels = np.array([g['level'] for g in a['strokes']])
    distances = abs(2*levels-levels[0]-levels[-1])
    near = np.flatnonzero(distances == distances.min())
    return (len(near) == 1 and near[0] != len(levels)//2
            and a['strokes'][int(near[0])]['length'] != a['reference'])


def _informative(a):
    axis = a['axis']
    for mask in a['prototypes'].values():
        core = np.take(mask, 1, axis=axis)
        if (mask.shape[1-axis] > 1 and core.any() and not core.all()
                and not np.array_equal(np.take(mask, 0, axis=axis), np.take(mask, 2, axis=axis))):
            return True
    return False


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    tr, te = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= tr <= 10 and 1 <= te <= 4 and tr > te
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            a = _parse(inp)
            assert np.array_equal(out, _render(a)), 'wrong elastic substitution'
            assert out.shape == a['shape']
            r, e, c, d = a['frame_box']
            crop = inp[r:e, c:d]
            assert np.array_equal(out[[0, -1]], crop[[0, -1]])
            assert np.array_equal(out[:, [0, -1]], crop[:, [0, -1]])
            assert set(map(int, np.unique(out))) <= set(map(int, np.unique(inp)))
            assert not np.array_equal(crop, out)
            analyses[split].append(a)
    train = analyses['train']
    assert len({a['reference'] for a in train}) >= 2
    assert len({len(a['strokes']) for a in train}) >= 2
    assert any(_geometric_witness(a) for a in train), 'missing ordinal/geometric witness'
    assert any(sorted(g['length'] for g in a['strokes'])[len(a['strokes'])//2] != a['reference'] for a in train)
    assert any(any(g['length'] < a['reference'] for g in a['strokes']) and
               any(g['length'] > a['reference'] for g in a['strokes']) for a in train)
    assert any(_informative(a) and a['reference'] >= 4 for a in train)
    assert any(any(t in a['mapping'] and a['mapping'][t] != t for t in a['mapping'].values()) for a in train)
    assert any(len(a['strokes']) > len(a['mapping']) for a in train)
    # Explicit parent ablations: own-length grammar and painted thin lines.
    assert any(not np.array_equal(_render(a, own_lengths=True), _render(a)) for a in train)
    assert any(not np.array_equal(_render(a, thin=True), _render(a)) for a in train)
    assert any(not np.array_equal(_render(a, source_paint=True), _render(a)) for a in train)


def measure_puzzle(puzzle):
    records = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            a = _parse(pair['input'])
            masks = list(a['prototypes'].values())
            sources = set(a['mapping'])
            targets = list(a['mapping'].values())
            overlap = any(t in sources for t in targets)
            kind = ('cycle' if set(targets) == sources else 'overlap') if overlap else 'disjoint'
            if len(set(targets)) < len(targets):
                kind += '_many'
            records.append(dict(split=split, axis=a['axis'], forward=a['forward'],
                                count=len(a['strokes']), reference=a['reference'],
                                lengths=tuple(g['length'] for g in a['strokes']),
                                gaps=tuple(map(int, np.diff([g['level'] for g in a['strokes']]))),
                                widths=tuple(m.shape[1-a['axis']] for m in masks),
                                input_shape=tuple(np.asarray(pair['input']).shape),
                                output_shape=a['shape'], background=a['background'],
                                categories=len(sources), mapping_kind=kind,
                                disconnected=sum(len(_components(m)) > 1 for m in masks),
                                empty_core=sum(not np.take(m, 1, axis=a['axis']).any() for m in masks),
                                repeated_geometry=len(masks)-len({_key(m) for m in masks}),
                                mask_signatures=tuple((m.shape, m.tobytes()) for m in masks)))
    return records


def self_test():
    """Finite checks: 42 seeds, count boundaries, equivariance and causal probes."""
    rng = np.random.default_rng(417)
    signatures, records = set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_key(puzzle))
        records.extend(measure_puzzle(puzzle))
        if seed < 3:
            assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        pair = puzzle['test'][0]
        inp, out = pair['input'], pair['output']
        palette = rng.permutation(10)
        assert np.array_equal(solve(palette[inp]), palette[out])
        for k in range(4):
            for reflect in (False, True):
                left, right = np.rot90(inp, k), np.rot90(out, k)
                if reflect:
                    left, right = np.fliplr(left), np.fliplr(right)
                assert np.array_equal(solve(left), right)
        # Causal reference probe: shorten the central stroke in the first witness.
        witness = puzzle['train'][0]
        a = _parse(witness['input'])
        altered = witness['input'].copy()
        g = a['strokes'][len(a['strokes'])//2]
        free = g['end']-1 if a['forward'] else g['start']
        rr, _, cc, _ = a['frame_box']
        row, col = (g['level'], free) if a['axis'] == 1 else (free, g['level'])
        altered[rr+row, cc+col] = a['background']
        changed = _parse(altered)
        assert changed['reference'] == a['reference']-1
        assert not np.array_equal(solve(altered), witness['output'])
        # Toggle a core pixel without hiding the stencil bounds. It must change
        # exactly that source's ribbons, not the count, lengths or other sources.
        source = next(s for s, m in a['prototypes'].items()
                      if m.shape[1-a['axis']] == 3 and
                      np.take(m, 1, axis=a['axis']).any() and
                      not np.take(m, 1, axis=a['axis']).all())
        mask = a['prototypes'][source]
        core = np.take(mask, 1, axis=a['axis'])
        trans = int(np.flatnonzero(~core)[0])
        local = (trans, 1) if a['axis'] == 1 else (1, trans)
        pr, _, pc, _ = a['prototype_boxes'][source]
        altered = witness['input'].copy()
        altered[pr+local[0], pc+local[1]] = source
        predicted = solve(altered)
        difference = predicted != witness['output']
        expected_change = np.zeros(a['shape'], bool)
        for stroke in a['strokes']:
            if stroke['source'] != source:
                continue
            r, e, c, d = _footprint(a, stroke, a['reference'])
            if a['axis'] == 1:
                expected_change[r+trans, c+1:d-1] = True
            else:
                expected_change[r+1:e-1, c+trans] = True
        assert np.array_equal(difference, expected_change)
    assert len(signatures) == 42
    assert {r['count'] for r in records} == {3, 5, 7}
    assert {w for r in records for w in r['widths']} == {1, 3, 5}
    assert {(r['axis'], r['forward']) for r in records} == {(0, False), (0, True), (1, False), (1, True)}
    assert len({r['reference'] for r in records}) >= 7
    assert len({r['input_shape'] for r in records}) >= 15
    assert len({r['output_shape'] for r in records}) >= 35
    assert len({(r['count'], r['reference'], r['gaps'], r['lengths'], r['widths']) for r in records}) >= 120
    assert len({m for r in records for m in r['mask_signatures']}) >= 30
    assert sum(r['disconnected'] for r in records) >= 20
    assert sum(r['empty_core'] for r in records) >= 5
    assert sum(r['repeated_geometry'] for r in records) >= 3
    assert {'disjoint', 'overlap', 'cycle'} <= {r['mapping_kind'] for r in records}
    for tr in range(2, 11):
        for te in range(1, min(4, tr-1)+1):
            p = generate_puzzle(812+tr*10+te, tr, te)
            assert (len(p['train']), len(p['test'])) == (tr, te)
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid counts accepted')
    damaged = generate_puzzle(0, 2, 1)
    a = _parse(damaged['test'][0]['input'])
    damaged['test'][0]['output'][0, 0] = a['background']
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged frame was accepted')
    return dict(seeds=42, pairs=len(records), distinct_puzzles=len(signatures),
                input_shapes=len({r['input_shape'] for r in records}),
                output_shapes=len({r['output_shape'] for r in records}),
                structural_signatures=len({(r['count'], r['reference'], r['gaps'],
                                           r['lengths'], r['widths']) for r in records}),
                distinct_masks=len({m for r in records for m in r['mask_signatures']}),
                disconnected_entries=sum(r['disconnected'] for r in records),
                empty_core_entries=sum(r['empty_core'] for r in records))


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
