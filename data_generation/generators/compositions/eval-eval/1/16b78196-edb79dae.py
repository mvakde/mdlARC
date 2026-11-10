"""Interlocking print plates: visible stencil inventory and exact-cover printing.

Only Python's standard library and NumPy are used. No generation state is read
by solve. Generation labels are painted from a sampled partition; a separate
input-derived placement search must uniquely recover the same colored grid.
"""

from collections import Counter
import hashlib

import numpy as np


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= d <= 30 for d in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any(grid < 0) or np.any(grid > 9)):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return grid.astype(np.int8, copy=True)


def _components(mask):
    pending = set(map(tuple, np.argwhere(mask)))
    result = []
    while pending:
        first = min(pending)
        pending.remove(first)
        stack, cells = [first], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in pending:
                    pending.remove(p)
                    stack.append(p)
        result.append(np.asarray(sorted(cells), dtype=int))
    return result


def _box(cells):
    lo, hi = cells.min(axis=0), cells.max(axis=0)+1
    return int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1])


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _bits(cells, width):
    result = 0
    for r, c in cells:
        result |= 1 << (int(r)*width+int(c))
    return result


def _separated(a, b):
    r, e, c, d = a
    rr, ee, cc, dd = b
    return e < rr or ee < r or d < cc or dd < c


def _analyze(input_grid):
    """Decode only visible frame, coupons, keys, and stencil coordinates."""
    g = _grid(input_grid)
    bg = int(g[0, 0])
    if not (np.all(g[0] == bg) and np.all(g[-1] == bg)
            and np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg)):
        raise ValueError("family inputs have a blank outer rim")
    frames = []
    for col in np.unique(g):
        if col == bg:
            continue
        cells = np.argwhere(g == col)
        r, e, c, d = _box(cells)
        if e-r < 5 or d-c < 5:
            continue
        tile = g[r:e, c:d]
        payload = set(map(int, np.unique(tile[1:-1, 1:-1]))) - {bg, int(col)}
        if (len(payload) >= 2 and np.all(tile[0] == col)
                and np.all(tile[-1] == col) and np.all(tile[:, 0] == col)
                and np.all(tile[:, -1] == col)):
            frames.append((int(col), (r, e, c, d)))
    if len(frames) != 1:
        raise ValueError("expected one frame enclosing at least two coupon colors")
    mold, frame = frames[0]
    r, e, c, d = frame
    board = g[r:e, c:d].copy()
    if len(_components(board == mold)) != 1:
        raise ValueError("all mold cells must be connected to the frame")
    interior = board[1:-1, 1:-1]
    coupons = []
    for source in sorted(set(map(int, np.unique(interior))) - {bg, mold}):
        for cells in _components(interior == source):
            a, b, x, y = _box(cells)
            if (b-a, y-x, len(cells)) != (2, 2, 4):
                raise ValueError("each inventory coupon must be a solid 2x2 square")
            coupons.append((source, a+1, x+1))
    counts = Counter(s for s, _, _ in coupons)
    if not 2 <= len(counts) <= 4 or not 2 <= len(coupons) <= 8:
        raise ValueError("expected two to four types and at most eight coupons")
    for i, (_, a, x) in enumerate(coupons):
        for _, b, y in coupons[i+1:]:
            if not _separated((a, a+2, x, x+2), (b, b+2, y, y+2)):
                raise ValueError("coupons require a blank-cell gap from each other")
    outside = g != bg
    outside[r:e, c:d] = False
    remaining = outside.copy()
    mapping, key_boxes = {}, {}
    for cells in _components(outside):
        colors, numbers = np.unique(g[cells[:, 0], cells[:, 1]], return_counts=True)
        if len(colors) == 1:
            continue
        a, b, x, y = _box(cells)
        if (b-a, y-x, len(cells)) != (2, 2, 4) or sorted(numbers) != [1, 3]:
            raise ValueError("keys must contain three source cells and one paint cell")
        source = int(colors[np.argmax(numbers)])
        target = int(colors[np.argmin(numbers)])
        if source not in counts or source in mapping or target in (bg, mold):
            raise ValueError("key is not a functional mapping for a coupon type")
        mapping[source], key_boxes[source] = target, (a, b, x, y)
        remaining[a:b, x:y] = False
    if set(mapping) != set(counts) or len(set(mapping.values())) < 2:
        raise ValueError("every type needs a key; at least two paint colors are used")
    if set(map(int, np.unique(g[remaining]))) != set(counts):
        raise ValueError("one exterior stencil is required for every source")
    stencils, stencil_boxes = {}, {}
    for source in counts:
        cells = np.argwhere(remaining & (g == source))
        a, b, x, y = _box(cells)
        tile = g[a:b, x:y]
        if max(tile.shape) > 8 or not np.all(np.isin(tile, [bg, source])):
            raise ValueError("stencil bounds are at most 8x8 and contain only their source")
        stencils[source] = tile == source
        stencil_boxes[source] = (a, b, x, y)
    boxes = list(key_boxes.values())+list(stencil_boxes.values())
    for i, box in enumerate(boxes):
        if not _separated(box, frame):
            raise ValueError("clue rectangles require a blank-cell gap to the frame")
        for other in boxes[i+1:]:
            if not _separated(box, other):
                raise ValueError("clue rectangles require a blank-cell gap")
    free = board != mold
    free[0] = free[-1] = False
    free[:, 0] = free[:, -1] = False
    if int(free.sum()) != sum(counts[s]*int(m.sum()) for s, m in stencils.items()):
        raise ValueError("the inventory area must exactly equal the open mold area")
    return dict(grid=g, background=bg, mold=mold, frame=frame, board=board,
                coupons=coupons, counts=counts, mapping=mapping, free=free,
                stencils=stencils, stencil_boxes=stencil_boxes, key_boxes=key_boxes)


def _assemblies(scene):
    """All translations; multiset exact cover; deduplicate actual painted answers.

    Stops only after finding two distinct colored answers. Identical-color
    assignments of indistinguishable impressions are not false ambiguities.
    """
    free, masks = scene['free'], scene['stencils']
    height, width = free.shape
    sources = sorted(masks)
    amounts = [scene['counts'][s] for s in sources]
    whole = _bits(np.argwhere(free), width)
    by_cell = [[] for _ in range(height*width)]
    for index, source in enumerate(sources):
        mask = masks[source]
        local = np.argwhere(mask)
        h, w = mask.shape
        for r in range(1, height-h):
            for c in range(1, width-w):
                cells = local+(r, c)
                if np.all(free[cells[:, 0], cells[:, 1]]):
                    bitset = _bits(cells, width)
                    candidate = (index, bitset, cells, (r, r+h, c, c+w))
                    for a, b in cells:
                        by_cell[int(a)*width+int(b)].append(candidate)
    solutions = {}

    def visit(occupied, left, chosen):
        if len(solutions) >= 2:
            return
        if occupied == whole:
            if any(left):
                return
            answer = scene['board'].copy()
            answer[free] = scene['background']
            placements = []
            for i, _, cells, box in chosen:
                source = sources[i]
                answer[cells[:, 0], cells[:, 1]] = scene['mapping'][source]
                placements.append((source, cells, box))
            solutions.setdefault(answer.tobytes(), (answer, placements))
            return
        uncovered = whole ^ occupied
        options = None
        while uncovered:
            low = uncovered & -uncovered
            cell = low.bit_length()-1
            uncovered ^= low
            available = [p for p in by_cell[cell] if left[p[0]] and not p[1] & occupied]
            if not available:
                return
            if options is None or len(available) < len(options):
                options = available
                if len(options) == 1:
                    break
        for p in options:
            i = p[0]
            next_left = list(left)
            next_left[i] -= 1
            visit(occupied | p[1], next_left, chosen+[p])

    visit(0, amounts, [])
    return list(solutions.values())


def _solve_scene(scene):
    answers = _assemblies(scene)
    if len(answers) != 1:
        raise ValueError("mold must have exactly one distinct painted completion")
    return answers[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode and pack using this input alone; returns the inclusive frame crop."""
    return _solve_scene(_analyze(input_grid))[0].copy()


def _well(rng, h, w, k):
    free = np.ones((h, w), dtype=bool)
    for _ in range(int(rng.integers(2, 6))):
        side = int(rng.integers(4))
        along = w if side < 2 else h
        start = int(rng.integers(along))
        end = min(along, start+int(rng.integers(1, max(2, along//2+1))))
        depth = int(rng.integers(1, 3))
        if side == 0:
            free[:depth, start:end] = False
        elif side == 1:
            free[-depth:, start:end] = False
        elif side == 2:
            free[start:end, :depth] = False
        else:
            free[start:end, -depth:] = False
    if (free.sum() < max(4*k+3, int(.62*h*w)) or len(_components(free)) != 1
            or not all(edge.any() for edge in (free[0], free[-1], free[:, 0], free[:, -1]))):
        raise ValueError("well clearance")
    return free


def _partition(rng, free, k, islands):
    cells = np.argwhere(free)
    seeds = [cells[int(rng.integers(len(cells)))]]
    for _ in range(k-1):
        distances = np.min(np.array([np.abs(cells-p).sum(axis=1) for p in seeds]), axis=0)
        scores = distances+rng.random(len(cells))*1.8
        seeds.append(cells[int(np.argmax(scores))])
    labels = np.full(free.shape, -1, dtype=int)
    for i, p in enumerate(seeds):
        labels[tuple(p)] = i
    sizes = [1]*k
    while np.any(free & (labels < 0)):
        frontiers = [set() for _ in range(k)]
        for r, c in np.argwhere(free & (labels < 0)):
            for a, b in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= a < free.shape[0] and 0 <= b < free.shape[1] and labels[a, b] >= 0:
                    frontiers[int(labels[a, b])].add((int(r), int(c)))
        live = [i for i in range(k) if frontiers[i]]
        if not live:
            raise ValueError("partition connectivity")
        i = min(live, key=lambda j: sizes[j]+float(rng.uniform(0, 3)))
        choices = sorted(frontiers[i])
        labels[choices[int(rng.integers(len(choices)))]] = i
        sizes[i] += 1
    if min(sizes) < 4:
        raise ValueError("tiny plate")
    if islands:
        candidates = []
        for p in cells:
            i = labels[tuple(p)]
            adjacent = {labels[a, b] for a, b in ((p[0]-1, p[1]), (p[0]+1, p[1]),
                                                 (p[0], p[1]-1), (p[0], p[1]+1))
                        if 0 <= a < free.shape[0] and 0 <= b < free.shape[1]}
            for j in range(k):
                if i != j and j not in adjacent:
                    candidates.append((tuple(p), int(j)))
        if not candidates:
            raise ValueError("no isolated island position")
        p, j = candidates[int(rng.integers(len(candidates)))]
        labels[p] = j
        if min(np.count_nonzero(labels == i) for i in range(k)) < 4:
            raise ValueError("island made a tiny plate")
        if not any(len(_components(labels == i)) > 1 for i in range(k)):
            raise ValueError("island did not stay separate")
    masks = []
    for i in range(k):
        a, b, x, y = _box(np.argwhere(labels == i))
        masks.append((labels[a:b, x:y] == i).copy())
    # Bounding-box overlap is the geometric point, not a row of solid blocks.
    boxes = [_box(np.argwhere(labels == i)) for i in range(k)]
    if not any(max(a[0], b[0]) < min(a[1], b[1]) and max(a[2], b[2]) < min(a[3], b[3])
               for i, a in enumerate(boxes) for b in boxes[i+1:]):
        raise ValueError("no interlock")
    return labels, masks


def _mapping(rng, sources, available, kind):
    order = [int(sources[i]) for i in rng.permutation(len(sources))]
    external = [c for c in available if c not in sources]
    if kind == 'disjoint':
        targets = list(map(int, rng.choice(external, len(sources), replace=False)))
        return dict(zip(order, targets))
    if kind == 'cycle':
        return {s: order[(i+1) % len(order)] for i, s in enumerate(order)}
    if kind == 'chain':
        return {s: order[i+1] if i+1 < len(order) else external[0] for i, s in enumerate(order)}
    for _ in range(20):
        result = {s: int(rng.choice([c for c in available if c != s])) for s in order}
        if len(set(result.values())) >= 2:
            return result
    return {s: order[(i+1) % len(order)] for i, s in enumerate(order)}


def _coupons(rng, board, free, sources, copies):
    h, w = board.shape
    for _ in range(32):
        result = board.copy()
        blocked = np.zeros(board.shape, dtype=bool)
        assignments = [s for s in sources for _ in range(copies)]
        rng.shuffle(assignments)
        for source in assignments:
            candidates = [(r, c) for r in range(1, h-2) for c in range(1, w-2)
                          if free[r:r+2, c:c+2].all() and not blocked[r:r+2, c:c+2].any()]
            if not candidates:
                break
            r, c = candidates[int(rng.integers(len(candidates)))]
            result[r:r+2, c:c+2] = source
            blocked[r-1:r+3, c-1:c+3] = True
        else:
            return result
    raise ValueError("coupon packing")


def _available(occupied, h, w):
    if h > occupied.shape[0] or w > occupied.shape[1]:
        return []
    prefix = np.pad(occupied.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = prefix[h:, w:]-prefix[:-h, w:]-prefix[h:, :-w]+prefix[:-h, :-w]
    return np.argwhere(sums == 0)


def _reserve(occupied, r, c, h, w):
    occupied[max(0, r-1):r+h+1, max(0, c-1):c+w+1] = True


def _canvas(rng, board, masks, sources, mapping, bg):
    h, w = board.shape
    items = [('stencil', s, m.shape) for s, m in zip(sources, masks)]
    items += [('key', s, (2, 2)) for s in sources]
    for attempt in range(40):
        if attempt % 4:
            r = int(rng.choice([1, 29-h]))
            c = int(rng.choice([1, 29-w]))
        else:
            r, c = int(rng.integers(1, 30-h)), int(rng.integers(1, 30-w))
        occupied = np.zeros((30, 30), dtype=bool)
        occupied[0] = occupied[-1] = True
        occupied[:, 0] = occupied[:, -1] = True
        _reserve(occupied, r, c, h, w)
        positions = {}
        # Large-first rectangle packing with independently randomized ties.
        ordered = sorted(items, key=lambda it: -it[2][0]*it[2][1]+float(rng.random())*.5)
        for kind, source, (hh, ww) in ordered:
            candidates = _available(occupied, hh, ww)
            if not len(candidates):
                break
            a, b = map(int, candidates[int(rng.integers(len(candidates)))])
            positions[kind, source] = (a, b)
            _reserve(occupied, a, b, hh, ww)
        else:
            grid = np.full((30, 30), bg, dtype=np.int8)
            grid[r:r+h, c:c+w] = board
            for source, mask in zip(sources, masks):
                a, b = positions['stencil', source]
                grid[a:a+mask.shape[0], b:b+mask.shape[1]][mask] = source
                a, b = positions['key', source]
                key = np.full((2, 2), source, dtype=np.int8)
                corner = int(rng.integers(4))
                key[corner//2, corner % 2] = mapping[source]
                grid[a:a+2, b:b+2] = key
            # Keep one or two background rows/columns around the occupied box.
            a, b, x, y = _box(np.argwhere(grid != bg))
            a, x = max(0, a-int(rng.integers(1, 3))), max(0, x-int(rng.integers(1, 3)))
            b, y = min(30, b+int(rng.integers(1, 3))), min(30, y+int(rng.integers(1, 3)))
            return grid[a:b, x:y].copy()
    raise ValueError("exterior legend packing")


def _make_pair(rng, h, w, copies, vertical, islands, map_kind, rotation):
    kmax = 3 if min(h, w) <= 5 or max(h, w) == 8 else 4
    k = int(rng.integers(2, kmax+1))
    free = _well(rng, h, w, k)
    labels, masks = _partition(rng, free, k, islands)
    palette = list(map(int, rng.permutation(10)))
    bg, mold = palette[:2]
    sources = palette[2:2+k]
    mapping = _mapping(rng, sources, palette[2:], map_kind)
    top, bottom, left, right = map(int, rng.integers(0, 2, size=4))
    gap = int(rng.integers(1, 4)) if copies == 2 else 0
    rows, cols = (copies, 1) if vertical else (1, copies)
    bh, bw = 2+top+bottom+rows*h+(rows-1)*gap, 2+left+right+cols*w+(cols-1)*gap
    board = np.full((bh, bw), mold, dtype=np.int8)
    target = board.copy()
    all_free = np.zeros(board.shape, dtype=bool)
    for j in range(copies):
        r = 1+top+(j*(h+gap) if vertical else 0)
        c = 1+left+(j*(w+gap) if not vertical else 0)
        board[r:r+h, c:c+w][free] = bg
        all_free[r:r+h, c:c+w] = free
        for i, source in enumerate(sources):
            target[r:r+h, c:c+w][labels == i] = mapping[source]
    inp_board = _coupons(rng, board, all_free, sources, copies)
    inp = _canvas(rng, inp_board, masks, sources, mapping, bg)
    inp, target = np.rot90(inp, rotation).copy(), np.rot90(target, rotation).copy()
    scene = _analyze(inp)
    predicted, _ = _solve_scene(scene)
    if not np.array_equal(predicted, target):
        raise ValueError("constructed partition is not the unique printed answer")
    if np.array_equal(_in_place_shortcut(scene), target):
        raise ValueError("coupon positions accidentally permit in-place replacement")
    return {'input': inp, 'output': target}


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    train = _count(num_train, 'num_train', 2, 10)
    test = _count(num_test, 'num_test', 1, 4)
    if train <= test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    shift, rotation = int(rng.integers(2)), int(rng.integers(4))
    result, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', train), ('test', test)):
        for index in range(count):
            training = split == 'train'
            if shift == 0:
                low, high = (5, 7) if training else (7, 9)
                copies = (1 if index == 0 else 2) if training and index < 2 else int(rng.integers(1, 3))
                vertical = bool(rng.integers(2))
            else:
                low, high = 5, 8
                copies = 1 if training and index == 0 else 2
                vertical = not training
            h, w = int(rng.integers(low, high)), int(rng.integers(low, high))
            islands = (training and index == 1) or bool(rng.random() < .35)
            kind = 'disjoint' if training and index == 0 else ('chain' if training and index == 1
                    else str(rng.choice(['disjoint', 'chain', 'cycle', 'mixed'])))
            for _ in range(96):
                try:
                    pair = _make_pair(rng, h, w, copies, vertical, islands, kind, rotation)
                except ValueError:
                    continue
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("bounded construction exhausted; no rule-changing fallback")
            seen.add(_key(pair['input']))
            result[split].append(pair)
    validate_generated_puzzle(result)
    return result


def _in_place_shortcut(scene):
    """Ignore packing: stamp each decoded plate at its coupon's top-left."""
    answer = scene['board'].copy()
    answer[scene['free']] = scene['background']
    for source, r, c in scene['coupons']:
        for a, b in np.argwhere(scene['stencils'][source]):
            x, y = int(r+a), int(c+b)
            if x < answer.shape[0] and y < answer.shape[1] and scene['free'][x, y]:
                answer[x, y] = scene['mapping'][source]
    return answer


def _validate_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
    inp, out = _grid(pair['input']), _grid(pair['output'])
    scene = _analyze(inp)
    answer, placements = _solve_scene(scene)
    assert np.array_equal(answer, out), "label disagrees with unique visible completion"
    assert out.shape == scene['board'].shape
    assert np.array_equal(out == scene['mold'], scene['board'] == scene['mold'])
    assert not np.any(out == scene['background']), "all open cells must be printed"
    used, occupied = Counter(), np.zeros(out.shape, dtype=bool)
    overlaps = False
    for i, (source, cells, box) in enumerate(placements):
        used[source] += 1
        normalized = cells-cells.min(axis=0)
        assert np.array_equal(normalized, np.argwhere(scene['stencils'][source]))
        assert not occupied[cells[:, 0], cells[:, 1]].any()
        occupied[cells[:, 0], cells[:, 1]] = True
        assert np.all(out[cells[:, 0], cells[:, 1]] == scene['mapping'][source])
        for _, _, other in placements[:i]:
            overlaps |= (max(box[0], other[0]) < min(box[1], other[1])
                         and max(box[2], other[2]) < min(box[3], other[3]))
    assert used == scene['counts'] and np.array_equal(occupied, scene['free'])
    assert overlaps, "examples must show an actual bounding-box interlock"
    assert scene['free'].sum() > 4*len(scene['coupons']), "packing solid coupons must fail by area"
    assert not np.array_equal(_in_place_shortcut(scene), out), "in-place replacement must fail"
    assert all(s != t for s, t in scene['mapping'].items())
    scene['placements'] = placements
    return scene


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            scene = _validate_pair(pair)
            signature = _key(pair['input'])
            assert signature not in seen, "inputs must be distinct"
            seen.add(signature)
            if split == 'train':
                training.append(scene)
    assert any(len(_components(m)) > 1 for s in training for m in s['stencils'].values())
    assert any(max(s['counts'].values()) == 2 for s in training)
    assert any(any(t in s['mapping'] for t in s['mapping'].values()) for s in training)
    assert any(not m.all() for s in training for m in s['stencils'].values())


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                shape, raw = _key(pair[role])
                digest.update(bytes(shape))
                digest.update(raw)
    return digest.hexdigest()


def _movement_tests(puzzle):
    moved = set()
    for pair in puzzle['train']:
        scene = _analyze(pair['input'])
        original = scene['grid']
        bg = scene['background']
        for kind in ('key', 'stencil'):
            if kind in moved:
                continue
            boxes = scene['key_boxes'] if kind == 'key' else scene['stencil_boxes']
            for source, box in boxes.items():
                a, b, x, y = box
                occupied = np.zeros(original.shape, dtype=bool)
                occupied[0] = occupied[-1] = True
                occupied[:, 0] = occupied[:, -1] = True
                r, e, c, d = scene['frame']
                _reserve(occupied, r, c, e-r, d-c)
                for other in list(scene['key_boxes'].values())+list(scene['stencil_boxes'].values()):
                    if other != box:
                        aa, bb, xx, yy = other
                        _reserve(occupied, aa, xx, bb-aa, yy-xx)
                choices = [(int(rr), int(cc)) for rr, cc in _available(occupied, b-a, y-x)
                           if abs(int(rr)-a)+abs(int(cc)-x) > 4]
                if not choices:
                    continue
                rr, cc = choices[-1]
                changed = original.copy()
                changed[a:b, x:y] = bg
                changed[rr:rr+b-a, cc:cc+y-x] = original[a:b, x:y]
                assert np.array_equal(solve(changed), pair['output'])
                moved.add(kind)
                break
        if 'coupon' not in moved:
            for index, (source, a, c) in enumerate(scene['coupons']):
                blocked = ~scene['free'].copy()
                for j, (_, r, x) in enumerate(scene['coupons']):
                    if index != j:
                        _reserve(blocked, r, x, 2, 2)
                choices = [(int(r), int(x)) for r, x in _available(blocked, 2, 2)
                           if (int(r), int(x)) != (a, c)]
                if not choices:
                    continue
                rr, cc = choices[-1]
                r0, _, c0, _ = scene['frame']
                changed = original.copy()
                changed[r0+a:r0+a+2, c0+c:c0+c+2] = bg
                changed[r0+rr:r0+rr+2, c0+cc:c0+cc+2] = source
                assert np.array_equal(solve(changed), pair['output'])
                moved.add('coupon')
                break
    assert moved == {'key', 'stencil', 'coupon'}
    # Change just one visible key, leaving all packing geometry untouched.
    pair = puzzle['train'][0]
    scene = _analyze(pair['input'])
    source = min(scene['mapping'])
    old = scene['mapping'][source]
    fresh = next(c for c in range(10) if c not in
                 {scene['background'], scene['mold'], *scene['mapping'], *scene['mapping'].values()})
    a, b, x, y = scene['key_boxes'][source]
    changed = pair['input'].copy()
    tile = changed[a:b, x:y]
    tile[tile == old] = fresh
    expected = pair['output'].copy()
    expected[expected == old] = fresh  # First demonstration has disjoint injective targets.
    assert np.array_equal(solve(changed), expected)


def self_test():
    """42 streamed seeds, metamorphic checks, count contract and bad-label checks.

    Keeps only hashes and small feature sets, writes no files, and retains at
    most one puzzle (plus a single-pair mutation). Returns descriptive metrics.
    """
    digests, shapes, templates, geometry = set(), set(), set(), set()
    categories, multiplicities, mask_sizes, colors = set(), set(), set(), set()
    disconnected = overlapping_maps = many_to_one = interlocks = 0
    pairs = large_shifts = arrangement_shifts = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digests.add(_puzzle_digest(puzzle))
        if seed < 2:
            assert _puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed))
        well_sizes, axes = {'train': [], 'test': []}, {'train': set(), 'test': set()}
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                pairs += 1
                scene = _validate_pair(pair)
                wells = _components(scene['free'])
                for cells in wells:
                    a, b, x, y = _box(cells)
                    well_sizes[split].extend((b-a, y-x))
                if len(wells) == 2:
                    delta = wells[1].min(axis=0)-wells[0].min(axis=0)
                    axes[split].add(int(np.argmax(np.abs(delta))))
                assert np.array_equal(solve(pair['input'].copy()), pair['output'])
                categories.add(len(scene['counts']))
                multiplicities.update(scene['counts'].values())
                shapes.add(pair['input'].shape)
                colors.update((scene['background'], scene['mold']))
                overlapping_maps += any(t in scene['mapping'] for t in scene['mapping'].values())
                many_to_one += len(set(scene['mapping'].values())) < len(scene['mapping'])
                interlocks += 1
                for mask in scene['stencils'].values():
                    templates.add(_key(mask.astype(np.int8)))
                    mask_sizes.add(mask.shape)
                    disconnected += len(_components(mask)) > 1
                roles, normalized = {}, []
                for col in pair['input'].flat:
                    normalized.append(roles.setdefault(int(col), len(roles)))
                geometry.add((pair['input'].shape, bytes(normalized)))
        if max(well_sizes['train']) < min(well_sizes['test']):
            large_shifts += 1
        else:
            assert axes['train'] and axes['test'] and not axes['train'] & axes['test']
            arrangement_shifts += 1
        if seed < 6:
            pair = puzzle['test'][0]
            # All ten colors may be permuted; all four rotations preserve the rule.
            permutation = np.random.default_rng(9000+seed).permutation(10)
            changed = permutation[pair['input']]
            assert np.array_equal(solve(changed), permutation[pair['output']])
            for turn in (1, 2, 3):
                assert np.array_equal(solve(np.rot90(pair['input'], turn)), np.rot90(pair['output'], turn))
            assert np.array_equal(solve(np.fliplr(pair['input'])), np.fliplr(pair['output']))
        if seed == 0:
            _movement_tests(puzzle)
            scene = _analyze(puzzle['train'][0]['input'])
            # Removing a coupon changes inventory area, even though the open mold is unchanged.
            broken = puzzle['train'][0]['input'].copy()
            source, a, b = scene['coupons'][0]
            r, _, c, _ = scene['frame']
            broken[r+a:r+a+2, c+b:c+b+2] = scene['background']
            try:
                solve(broken)
            except ValueError:
                pass
            else:
                raise AssertionError("missing inventory was accepted")
            altered = {s: [dict(p) for p in puzzle[s]] for s in ('train', 'test')}
            altered['test'][0] = dict(altered['test'][0])
            altered['test'][0]['output'] = altered['test'][0]['output'].copy()
            altered['test'][0]['output'][0, 0] = scene['background']
            try:
                validate_generated_puzzle(altered)
            except (AssertionError, ValueError):
                pass
            else:
                raise AssertionError("damaged frame label was accepted")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(70000+10*a+b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    assert len(digests) == 42 and len(geometry) == pairs
    assert categories == {2, 3, 4} and multiplicities == {1, 2}
    assert len(shapes) >= 15 and len(mask_sizes) >= 20 and len(templates) >= 100
    assert colors == set(range(10)) and large_shifts >= 10 and arrangement_shifts >= 10
    assert disconnected >= 42 and overlapping_maps >= 42 and many_to_one >= 1
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(digests),
                color_normalized_inputs=len(geometry), input_shapes=len(shapes),
                stencil_shapes=len(mask_sizes), distinct_stencils=len(templates),
                disconnected_stencils=disconnected, overlapping_maps=overlapping_maps,
                many_to_one_maps=many_to_one, interlocking_pairs=interlocks,
                larger_well_shifts=large_shifts, perpendicular_arrangement_shifts=arrangement_shifts)
