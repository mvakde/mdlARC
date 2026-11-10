"""Reachability-gated stencil delivery on a miniature blueprint.

Only NumPy and the standard library are used.  Input parsing, scale recovery,
shape matching, and collision tests never receive a seed or an output label.
"""
from collections import Counter, deque
from math import isqrt

import numpy as np


FLOOR, SOCKET, BAY = 0, 5, 8
STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), 'grid dimensions')
    _require(np.issubdtype(a.dtype, np.integer), 'integer cells required')
    _require(bool(np.all((0 <= a) & (a <= 9))), 'color range')
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        queue, cells = [p], []
        for r, c in queue:
            cells.append((int(r), int(c)))
            for dr, dc in STEPS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    queue.append(q)
        result.append(tuple(cells))
    return result


def _bounds(cells):
    return (min(r for r, c in cells), max(r for r, c in cells) + 1,
            min(c for r, c in cells), max(c for r, c in cells) + 1)


def _normal(cells):
    r0, _, c0, _ = _bounds(cells)
    return tuple(sorted((int(r-r0), int(c-c0)) for r, c in cells))


def _rotate(shape):
    return _normal([(c, -r) for r, c in shape])


def _orientations(shape, reflection=True):
    result = set()
    for base in (shape, _normal([(r, -c) for r, c in shape])) if reflection else (shape,):
        for _ in range(4):
            result.add(base)
            base = _rotate(base)
    return tuple(sorted(result))


def _canonical(shape):
    return min(_orientations(_normal(shape)))


def _size(shape):
    return max(r for r, c in shape) + 1, max(c for r, c in shape) + 1


def _mask(shape):
    a = np.zeros(_size(shape), dtype=bool)
    for p in shape:
        a[p] = True
    return a


def _expand(a, scale):
    return np.repeat(np.repeat(a, scale, axis=0), scale, axis=1)


def _decode(cells, scale):
    r0, r1, c0, c1 = _bounds(cells)
    if (r1-r0) % scale or (c1-c0) % scale:
        return None
    blocks = Counter(((r-r0)//scale, (c-c0)//scale) for r, c in cells)
    if any(n != scale**2 for n in blocks.values()):
        return None
    return tuple(sorted(blocks))


def _placements(allowed, footprint):
    h, w = footprint.shape
    vh, vw = allowed.shape[0]-h+1, allowed.shape[1]-w+1
    if min(vh, vw) <= 0:
        return np.zeros((0, 0), dtype=bool)
    valid = np.ones((vh, vw), dtype=bool)
    for r, c in np.argwhere(footprint):
        valid &= allowed[r:r+vh, c:c+vw]
    return valid


def _can_deliver(board, wall, cells, scale, *, solid_box=False):
    """Unit-pixel translations, in the pose visibly specified by this socket."""
    shape = _normal(cells)
    footprint = _expand(_mask(shape), scale)
    if solid_box:
        footprint[:] = True
    enlarged = _expand(board, scale)
    valid = _placements(enlarged != wall, footprint)
    starts = _placements(enlarged == BAY, footprint)
    _require(starts.size and bool(starts.any()), 'piece does not fit the launch bay')
    r0, _, c0, _ = _bounds(cells)
    target = r0 * scale, c0 * scale
    seen = starts.copy()
    queue = deque(map(tuple, np.argwhere(starts)))
    while queue:
        p = queue.popleft()
        if p == target:
            return True
        for dr, dc in STEPS:
            q = p[0]+dr, p[1]+dc
            if 0 <= q[0] < valid.shape[0] and 0 <= q[1] < valid.shape[1] and valid[q] and not seen[q]:
                seen[q] = True
                queue.append(q)
    return False


def _render_deliveries(board, wall, sockets, colors, scale):
    output = _expand(board, scale)
    output[output == BAY] = FLOOR
    delivered = []
    for cells, color in zip(sockets, colors):
        success = _can_deliver(board, wall, cells, scale)
        delivered.append(success)
        if success:
            for r, c in cells:
                output[r*scale:(r+1)*scale, c*scale:(c+1)*scale] = color
    return output.astype(np.int8), tuple(delivered)


def _recover(inp):
    """Identify the one framed map, then recover scale and color by shape."""
    candidates = []
    for wall in range(1, 10):
        if wall in (SOCKET, BAY):
            continue
        for component in _components(inp == wall):
            r0, r1, c0, c1 = _bounds(component)
            board = inp[r0:r1, c0:c1]
            if min(board.shape) < 5:
                continue
            if not (np.all(board[0] == wall) and np.all(board[-1] == wall)
                    and np.all(board[:, 0] == wall) and np.all(board[:, -1] == wall)):
                continue
            if not np.all(np.isin(board, (FLOOR, SOCKET, BAY, wall))):
                continue
            bays = _components(board == BAY)
            sockets = _components(board == SOCKET)
            if len(bays) != 1 or len(sockets) < 2:
                continue
            br0, br1, bc0, bc1 = _bounds(bays[0])
            q = br1-br0
            if not (2 <= q <= 4 and bc1-bc0 == q and len(bays[0]) == q*q):
                continue
            outside = np.ones(inp.shape, dtype=bool)
            outside[r0:r1, c0:c1] = False
            moat = np.zeros(inp.shape, dtype=bool)
            moat[max(0, r0-1):r1+1, max(0, c0-1):c1+1] = True
            if np.any(outside & moat & (inp != FLOOR)):
                continue
            if np.any(outside & np.isin(inp, (wall, SOCKET, BAY))):
                continue
            external = [(color, cells) for color in range(1, 10)
                        for cells in _components(outside & (inp == color))]
            if len(_components(outside & (inp != FLOOR))) != len(external):
                continue
            pieces = [(color, cells) for color, cells in external if len(cells) >= 4]
            noise = [(color, cells) for color, cells in external if len(cells) < 4]
            if len(pieces) != len(sockets):
                continue
            area = sum(len(c) for c in sockets)
            enlarged_area = sum(len(c) for _, c in pieces)
            scale = isqrt(enlarged_area // area)
            if not (2 <= scale <= 4 and enlarged_area == area*scale*scale
                    and max(board.shape)*scale <= 30):
                continue
            decoded = [(color, _decode(cells, scale)) for color, cells in pieces]
            if any(shape is None for _, shape in decoded):
                continue
            groups = {}
            for color, shape in decoded:
                groups.setdefault(_canonical(shape), []).append((color, shape))
            holes = Counter(_canonical(cells) for cells in sockets)
            if Counter({key: len(items) for key, items in groups.items()}) != holes:
                continue
            _require(all(len({c for c, _ in items}) == 1 for items in groups.values()),
                     'congruent differently colored pieces give an ambiguous assignment')
            _require(all(max(_size(_normal(cells))) <= q for cells in sockets), 'bay too small')
            _require(len(_components(board != wall)) == 1, 'floor map is disconnected')
            colors = [groups[_canonical(cells)][0][0] for cells in sockets]
            output, delivered = _render_deliveries(board, wall, sockets, colors, scale)
            reflection = rotation = delivered_reflection = delivered_rotation = False
            for cells, success in zip(sockets, delivered):
                group = groups[_canonical(cells)]
                if len(group) == 1:
                    loose = group[0][1]
                    target_shape = _normal(cells)
                    needs_reflection = target_shape not in _orientations(loose, False)
                    needs_rotation = target_shape != loose and target_shape in _orientations(loose, False)
                    reflection |= needs_reflection
                    rotation |= needs_rotation
                    delivered_reflection |= success and needs_reflection
                    delivered_rotation |= success and needs_rotation
            candidates.append((output, {
                'wall': wall, 'board': board.copy(), 'bounds': (r0, r1, c0, c1),
                'q': q, 'scale': scale, 'sockets': sockets, 'colors': tuple(colors),
                'delivered': delivered, 'reflection': bool(reflection), 'rotation': bool(rotation),
                'delivered_reflection': bool(delivered_reflection), 'delivered_rotation': bool(delivered_rotation),
                'noise': sum(len(c) for _, c in noise),
                'classes': tuple(sorted(holes)), 'piece_count': len(sockets),
            }))
    _require(len(candidates) == 1, 'input needs exactly one map/scale/color interpretation')
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _recover(_grid(input_grid))[0]


def _grow_shape(rng, q, area):
    cells = {(0, 0)}
    while len(cells) < area:
        options = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in STEPS} - cells)
        options = [p for p in options if max(_size(_normal(cells | {p}))) <= q]
        cells.add(options[int(rng.integers(len(options)))])
    return _normal(cells)


def _layout(rng, scale, witness):
    if witness == 'reflection':
        # A controlled teaching contrast, with randomized lengths, offsets,
        # target positions, palette, packing and global square symmetry.
        # The reflected L is delivered through a full-height horizontal gate;
        # the equal-area T cannot traverse a long vertical width-2 gate in
        # its specified width-3 pose, although a rotated T could traverse it.
        q = 3
        gap_h = int(rng.integers(1, 5)) if scale == 2 else 1
        gap_v = int(rng.integers(2, 5)) if scale == 2 else 2
        rows, cols = [1, 4+gap_v], [1, 4+gap_h]
        allowed = np.zeros((rows[-1]+4, cols[-1]+4), dtype=bool)
        for r in rows:
            for c in cols:
                allowed[r:r+3, c:c+3] = True
        allowed[1:4, 4:cols[1]] = True
        offset = int(rng.integers(2))
        allowed[4:rows[1], 1+offset:3+offset] = True
        width = int(rng.integers(1, 4))
        offset = int(rng.integers(4-width))
        allowed[4:rows[1], cols[1]+offset:cols[1]+offset+width] = True
        l = ((0, 0), (1, 0), (2, 0), (2, 1))
        if rng.integers(2):
            l = _normal([(r, -c) for r, c in l])
        t = ((0, 0), (0, 1), (0, 2), (1, 1))
        offsets = [(1, cols[1]+int(rng.integers(2))),
                   (rows[1]+int(rng.integers(2)), 1),
                   (rows[1]+int(rng.integers(3)), cols[1]+int(rng.integers(3)))]
        sockets = [tuple((r+a, c+b) for r, c in shape)
                   for shape, (a, b) in zip((l, t, ((0, 0),)), offsets)]
        return allowed, (1, 1, 3), sockets
    q = 3 if witness else int(rng.choice([2, 3, 4] if scale == 2 else [2, 3] if scale == 3 else [2]))
    limit = 30 // scale
    nr, nc = 2, 2
    if not witness and (3*q + 4 <= limit) and rng.random() < .5:
        nr, nc = (2, 3) if rng.integers(2) else (3, 2)
    def coordinates(n):
        budget = limit - (n*q + n-1 + 2)
        gaps = [1]* (n-1)
        for i in rng.permutation(n-1):
            extra = int(rng.integers(min(3, budget)+1))
            gaps[int(i)] += extra
            budget -= extra
        return [1] + [1 + i*q + sum(gaps[:i]) for i in range(1, n)]
    rows, cols = coordinates(nr), coordinates(nc)
    h, w = rows[-1]+q+1, cols[-1]+q+1
    allowed = np.zeros((h, w), dtype=bool)
    nodes = [(r, c) for r in range(nr) for c in range(nc)]
    for r, c in nodes:
        allowed[rows[r]:rows[r]+q, cols[c]:cols[c]+q] = True
    bay_node = nodes[int(rng.integers(len(nodes)))]
    # A randomly grown spanning tree supplies branches and unequal route lengths.
    visited, edges = {bay_node}, []
    while len(visited) < len(nodes):
        frontier = [(p, (p[0]+dr, p[1]+dc)) for p in sorted(visited) for dr, dc in STEPS
                    if (p[0]+dr, p[1]+dc) in nodes and (p[0]+dr, p[1]+dc) not in visited]
        a, b = frontier[int(rng.integers(len(frontier)))]
        visited.add(b)
        edges.append((a, b))
    # Reachability has no ambiguous 'furthest stop' even when routes form a loop.
    # A loop is shown in the second teaching map before tests can use one.
    all_edges = [(a, b) for a in nodes for b in nodes
                 if a < b and abs(a[0]-b[0])+abs(a[1]-b[1]) == 1]
    existing = {tuple(sorted(e)) for e in edges}
    for edge in all_edges:
        if edge not in existing and (witness == 'rotation' or rng.random() < .25):
            edges.append(edge)
    for a, b in edges:
        width = int(rng.integers(1, q+1))
        offset = int(rng.integers(q-width+1))
        if a[0] == b[0]:
            lo, hi = sorted((cols[a[1]], cols[b[1]]))
            r = rows[a[0]]+offset
            allowed[r:r+width, lo+q:hi] = True
        else:
            lo, hi = sorted((rows[a[0]], rows[b[0]]))
            c = cols[a[1]]+offset
            allowed[lo+q:hi, c:c+width] = True
    others = [p for p in nodes if p != bay_node]
    rng.shuffle(others)
    count = 3 if witness else int(rng.integers(2, len(others)+1))
    selected = others[:count]
    shapes = []
    if witness == 'rotation':
        shapes = [((0, 0), (0, 1), (1, 0)), ((0, 0),),
                  _grow_shape(rng, q, int(rng.integers(4, 8)))]
    else:
        for i in range(count):
            if shapes and rng.random() < .22:
                shapes.append(shapes[int(rng.integers(len(shapes)))])
            else:
                shapes.append(_grow_shape(rng, q, int(rng.integers(1, min(q*q, 8)+1))))
    sockets = []
    for node, shape in zip(selected, shapes):
        poses = _orientations(shape)
        shape = poses[int(rng.integers(len(poses)))]
        sh, sw = _size(shape)
        r0 = rows[node[0]] + int(rng.integers(q-sh+1))
        c0 = cols[node[1]] + int(rng.integers(q-sw+1))
        sockets.append(tuple((r+r0, c+c0) for r, c in shape))
    return allowed, (rows[bay_node[0]], cols[bay_node[1]], q), sockets


def _pack(rng, board, loose, colors, scale):
    objects = [board] + [np.where(_expand(_mask(s), scale), color, FLOOR).astype(np.int8)
                         for s, color in zip(loose, colors)]
    for attempt in range(32):
        h, w = (int(rng.integers(22, 31)), int(rng.integers(22, 31))) if attempt < 20 else (30, 30)
        canvas = np.zeros((h, w), dtype=np.int8)
        blocked = np.zeros((h, w), dtype=bool)
        order = sorted(range(len(objects)), key=lambda i: (-objects[i].size, float(rng.random())))
        for i in order:
            obj = objects[i]
            oh, ow = obj.shape
            if oh > h or ow > w:
                break
            sums = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
            totals = sums[oh:, ow:] - sums[:-oh, ow:] - sums[oh:, :-ow] + sums[:-oh, :-ow]
            positions = np.argwhere(totals == 0)
            if len(positions) == 0:
                break
            r, c = map(int, positions[int(rng.integers(len(positions)))])
            canvas[r:r+oh, c:c+ow] = obj
            blocked[max(0, r-1):r+oh+1, max(0, c-1):c+ow+1] = True
        else:
            # Optional isolated specks are below the area of one scale-2 block.
            count = int(rng.integers(6))
            for _ in range(count):
                positions = np.argwhere(~blocked)
                if not len(positions):
                    break
                r, c = map(int, positions[int(rng.integers(len(positions)))])
                canvas[r, c] = colors[int(rng.integers(len(colors)))]
                blocked[max(0, r-1):r+2, max(0, c-1):c+2] = True
            r0, r1, c0, c1 = _bounds(tuple(map(tuple, np.argwhere(canvas != FLOOR))))
            r0, c0 = max(0, r0-int(rng.integers(4))), max(0, c0-int(rng.integers(4)))
            r1, c1 = min(h, r1+int(rng.integers(4))), min(w, c1+int(rng.integers(4)))
            return canvas[r0:r1, c0:c1].copy()
    return None


def _make_pair(rng, scale, witness=None, need_mixed=False):
    for _ in range(480):
        allowed, (r, c, q), sockets = _layout(rng, scale, witness)
        palette = list(map(int, rng.permutation([1, 2, 3, 4, 6, 7, 9])))
        wall, available = palette[0], palette[1:]
        board = np.where(allowed, FLOOR, wall).astype(np.int8)
        board[r:r+q, c:c+q] = BAY
        classes = list(dict.fromkeys(_canonical(cells) for cells in sockets))
        # The equal-area training shapes have different colors. Extra examples
        # also allow distinct classes to share colors.
        cardinality = len(classes) if witness else int(rng.integers(1, len(classes)+1))
        color_map = {key: available[i % cardinality] for i, key in enumerate(classes)}
        colors = [color_map[_canonical(cells)] for cells in sockets]
        for cells in sockets:
            for p in cells:
                board[p] = SOCKET
        if witness == 'rotation':
            # A wall peg occupies the missing corner of the L-triomino socket.
            # Its true silhouette can dock here; its solid bounding box cannot.
            a, b, c0, d = _bounds(sockets[0])
            missing = [(x, y) for x in range(a, b) for y in range(c0, d)
                       if (x, y) not in sockets[0]]
            board[missing[0]] = wall
        elif not witness:
            # Irregular chamber edges and pillars, without damaging any marker.
            for _ in range(int(rng.integers(4))):
                options = np.argwhere(board == FLOOR)
                if not len(options):
                    break
                p = tuple(options[int(rng.integers(len(options)))])
                board[p] = wall
                if len(_components(board != wall)) != 1:
                    board[p] = FLOOR
        if len(_components(board != wall)) != 1:
            continue
        turns, reflected = int(rng.integers(4)), bool(rng.integers(2))
        def transform(a):
            a = np.rot90(a, turns)
            return np.fliplr(a) if reflected else a
        original_shape = board.shape
        transformed_sockets = []
        for cells in sockets:
            marker = np.zeros(original_shape, dtype=bool)
            for p in cells:
                marker[p] = True
            transformed_sockets.append(tuple(map(tuple, np.argwhere(transform(marker)))))
        board = transform(board).copy()
        sockets = transformed_sockets
        expected, delivered = _render_deliveries(board, wall, sockets, colors, scale)
        if need_mixed and not (any(delivered) and not all(delivered)):
            continue
        if witness == 'rotation' and not delivered[0]:
            continue
        loose = []
        for i, cells in enumerate(sockets):
            shape = _normal(cells)
            poses = _orientations(shape)
            if i == 0 and witness == 'reflection':
                poses = tuple(p for p in poses if p not in _orientations(shape, False))
            elif i == 0 and witness == 'rotation':
                poses = tuple(p for p in _orientations(shape, False) if p != shape)
            loose.append(poses[int(rng.integers(len(poses)))])
        inp = _pack(rng, board, loose, colors, scale)
        if inp is None:
            continue
        output, info = _recover(inp)
        _require(np.array_equal(expected, output), 'construction and recovery disagree')
        _require(info['scale'] == scale, 'scale recovery')
        if witness and not info[witness]:
            continue
        return {'input': inp, 'output': expected}
    raise AssertionError('bounded pair construction exhausted')


def _check_counts(num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, 'num_train'), (num_test, 1, 4, 'num_test')):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    # A larger block scale, reverse scale, or a familiar-scale new topology.
    if regime == 0:
        train_scales, test_scales = [2]*num_train, [3]*num_test
    elif regime == 1:
        train_scales, test_scales = [3]*num_train, [2]*num_test
    else:
        train_scales = [2, 3] + [int(rng.choice([2, 3])) for _ in range(num_train-2)]
        test_scales = [4]*num_test
    result, seen = {'train': [], 'test': []}, set()
    for split, scales in (('train', train_scales), ('test', test_scales)):
        for i, scale in enumerate(scales):
            witness = ('reflection' if i == 0 else 'rotation' if i == 1 else None) if split == 'train' else None
            for _ in range(12):
                pair = _make_pair(rng, scale, witness, need_mixed=split == 'train' and i < 2)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise AssertionError('distinct input construction exhausted')
            seen.add(_key(pair['input']))
            result[split].append(pair)
    # Witnesses have no fixed presentation position.
    result['train'] = [result['train'][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(result)
    return result


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split lists')
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            expected, info = _recover(inp)
            _require(np.array_equal(out, expected), 'incorrect deliveries, scale, or preserved structure')
            board = info['board']
            s = info['scale']
            _require(out.shape == (board.shape[0]*s, board.shape[1]*s), 'output dimensions')
            _require(not np.any(out == BAY), 'launch bay must become floor')
            for cells, color, delivered in zip(info['sockets'], info['colors'], info['delivered']):
                for r, c in cells:
                    _require(np.all(out[r*s:(r+1)*s, c*s:(c+1)*s] == (color if delivered else SOCKET)),
                             'socket must be filled wholly or stay neutral')
            scenery = (board != SOCKET) & (board != BAY)
            _require(np.array_equal(out[_expand(scenery, s)], _expand(board, s)[_expand(scenery, s)]),
                     'scenery altered')
            infos[split].append(info)
    train = infos['train']
    _require(any(i['delivered_reflection'] for i in train), 'training lacks a delivered reflected match')
    _require(any(i['delivered_rotation'] for i in train), 'training lacks a delivered rotated match')
    _require(sum(any(i['delivered']) and not all(i['delivered']) for i in train) >= 2,
             'two training maps must contrast reachable and blocked sockets')
    _require(any(any(len(a) == len(b) and _canonical(a) != _canonical(b)
                         and ca != cb and da != db
                         for a, ca, da in zip(i['sockets'], i['colors'], i['delivered'])
                         for b, cb, db in zip(i['sockets'], i['colors'], i['delivered'])) for i in train),
             'training needs differently colored equal-area shapes with contrasting access')
    _require(any(success and not _can_deliver(i['board'], i['wall'], cells, i['scale'], solid_box=True)
                 for i in train for cells, success in zip(i['sockets'], i['delivered'])),
             'training lacks an output-visible silhouette/bounding-box contrast')
    return None


def self_test():
    """Bounded semantic diagnostics and a 42-seed structural diversity audit."""
    fingerprints, board_patterns, classes, scales, sizes, counts, outcomes = set(), set(), set(), set(), set(), set(), set()
    pair_count = 0
    reflections = rotations = repeated = bbox_contrasts = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(tuple(_key(p['input']) for split in puzzle for p in puzzle[split]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                pair_count += 1
                out, info = _recover(pair['input'])
                _require(np.array_equal(out, pair['output']), 'prediction mismatch')
                board = info['board']
                board_patterns.add((board.shape, (board == info['wall']).tobytes(), (board == SOCKET).tobytes()))
                classes.update(info['classes'])
                scales.add(info['scale'])
                sizes.add(board.shape)
                counts.add(info['piece_count'])
                outcomes.add(info['delivered'])
                reflections += info['reflection']
                rotations += info['rotation']
                repeated += len(info['classes']) < info['piece_count']
                bbox_contrasts += sum(success and not _can_deliver(board, info['wall'], cells, info['scale'], solid_box=True)
                                     for cells, success in zip(info['sockets'], info['delivered']))
    _require(len(fingerprints) == 42 and len(board_patterns) >= 150, 'insufficient structural variation')
    _require(len(classes) >= 20 and len(sizes) >= 12, 'shape/geometry diversity')
    _require(scales == {2, 3, 4} and counts >= {2, 3, 4}, 'scale/count diversity')
    _require(reflections >= 42 and rotations >= 42 and repeated > 0, 'matching variation')
    _require(bbox_contrasts > 0, 'full silhouettes never matter beyond bounding boxes')
    _require(any(all(v) for v in outcomes) and any(not any(v) for v in outcomes), 'outcome spectrum')
    # A direct diagnostic fixes its expected delivery outcomes, independent of
    # sampling: horizontal width-1 throat passes a horizontal domino, not a square.
    board = np.full((7, 11), 2, dtype=np.int8)
    board[1:6, 1:4] = FLOOR
    board[1:6, 7:10] = FLOOR
    board[3, 4:7] = FLOOR
    board[1:4, 1:4] = BAY
    domino, square = ((3, 7), (3, 8)), ((3, 7), (3, 8), (4, 7), (4, 8))
    _require(_can_deliver(board, 2, domino, 2), 'domino throat diagnostic')
    _require(not _can_deliver(board, 2, square, 2), 'square throat diagnostic')
    first = generate_puzzle(0)
    _require(tuple(_key(p['input']) for split in first for p in first[split]) ==
             tuple(_key(p['input']) for split in first for p in generate_puzzle(0)[split]), 'nondeterminism')
    corrupted = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in first[s]] for s in first}
    corrupted['test'][0]['output'][0, 0] = (int(corrupted['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError('incorrect output accepted')
    # Break one full macrocell in a genuine piece: it must not become accepted
    # noise or cause a silently chosen alternative scale.
    sample = first['train'][0]['input'].copy()
    _, info = _recover(sample)
    r0, r1, c0, c1 = info['bounds']
    outside = np.ones(sample.shape, dtype=bool)
    outside[r0:r1, c0:c1] = False
    piece = next(cells for color in range(1, 10)
                 for cells in _components(outside & (sample == color)) if len(cells) > 4)
    sample[piece[0]] = FLOOR
    try:
        solve(sample)
    except AssertionError:
        pass
    else:
        raise AssertionError('broken macrocell accepted')
    # Identical-color repeated pieces are interchangeable; recoloring one
    # member of that class makes the target color assignment ambiguous.
    repeated_input = None
    for seed in range(42):
        for pair in generate_puzzle(seed)['train']:
            _, info = _recover(pair['input'])
            if len(info['classes']) < info['piece_count']:
                repeated_input = pair['input'].copy()
                break
        if repeated_input is not None:
            break
    _require(repeated_input is not None, 'repeated-class diagnostic unavailable')
    r0, r1, c0, c1 = info['bounds']
    outside = np.ones(repeated_input.shape, dtype=bool)
    outside[r0:r1, c0:c1] = False
    objects = [(color, cells, _canonical(_decode(cells, info['scale'])))
               for color in range(1, 10)
               for cells in _components(outside & (repeated_input == color)) if len(cells) >= 4]
    frequencies = Counter(key for color, cells, key in objects)
    color, cells, key = next(item for item in objects if frequencies[item[2]] > 1)
    replacement = next(c for c in (1, 2, 3, 4, 6, 7, 9) if c not in (info['wall'], color))
    for p in cells:
        repeated_input[p] = replacement
    try:
        solve(repeated_input)
    except AssertionError:
        pass
    else:
        raise AssertionError('differently colored congruent pieces accepted')
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            puzzle = generate_puzzle(1900+a*10+b, a, b)
            _require(len(puzzle['train']) == a and len(puzzle['test']) == b, 'explicit counts ignored')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return {'seeds': 42, 'pairs': pair_count, 'distinct_puzzles': len(fingerprints),
            'board_patterns': len(board_patterns), 'shape_classes': len(classes),
            'board_dimensions': len(sizes), 'scales': sorted(scales),
            'socket_counts': sorted(counts), 'delivery_patterns': len(outcomes),
            'reflection_pairs': reflections, 'rotation_pairs': rotations,
            'repeated_shape_pairs': repeated, 'bounding_box_counterexamples': bbox_contrasts}
