"""Interference looms: legend stencils compete as periodic Manhattan fields.

Only NumPy and the standard library are used. solve() reads a single input.
The fixed right-pointing key has seven source and five paint cells; frame ink
also marks coincidence of at least two active, equally nearest emitters.
"""
from __future__ import annotations

import itertools
import numpy as np


_ARROW = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0]], bool)
_HOST_SEEDS = (0, 509339928, 938455582, 402197125, 68055968, 43359190,
               251205548, 159688893, 328187835, 511816349, 83342592,
               425357052, 579809444, 12724113, 862693133, 89617312,
               798029923, 610351286, 273064044, 457291888, 312267253,
               283628220, 391429521, 958965190, 352774592, 394420553,
               665992450, 787288404, 316327521, 376783789, 283091458,
               520704835, 969725449, 983871636, 700292921, 485549649,
               9077685, 885319235, 121527086, 63969280, 753448819, 237136250)


def _check_grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _fingerprint(puzzle):
    return tuple((_key(p['input']), _key(p['output']))
                 for split in ('train', 'test') for p in puzzle[split])


def _box(cells):
    a = np.asarray(cells)
    r, c = a.min(axis=0)
    e, d = a.max(axis=0) + 1
    return int(r), int(e), int(c), int(d)


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
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _separate(a, b):
    r, e, c, d = a
    rr, ee, cc, dd = b
    return e < rr or ee < r or d < cc or dd < c


def _side(box, frame):
    r, e, c, d = box
    rr, ee, cc, dd = frame
    if e < rr:
        return 'above'
    if r > ee:
        return 'below'
    if d < cc:
        return 'left'
    if c > dd:
        return 'right'
    raise ValueError("clue needs a blank cell between its bounds and the frame")


def _parse_input(input_grid):
    """Recover all conventions from pixels, including disconnected stencils."""
    grid = _check_grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[np.argmax(counts)])
    if int(counts.max()) * 2 <= grid.size:
        raise ValueError("background must be the strict majority color")
    frames = []
    for color in colors:
        if color == background:
            continue
        cells = np.argwhere(grid == color)
        r, e, c, d = _box(cells)
        h, w = e-r, d-c
        if (h >= 8 and w >= 8 and len(cells) == 2*h + 2*w - 4
                and np.all(grid[r, c:d] == color)
                and np.all(grid[e-1, c:d] == color)
                and np.all(grid[r:e, c] == color)
                and np.all(grid[r:e, d-1] == color)):
            frames.append(((r, e, c, d), int(color)))
    if len(frames) != 1:
        raise ValueError("expected one exclusive complete rectangular frame")
    frame, frame_color = frames[0]
    r, e, c, d = frame
    interior = grid[r+1:e-1, c+1:d-1]
    emitters, sizes = [], set()
    for cells in _components(interior != background):
        a, b, x, y = _box(cells)
        tile = interior[a:b, x:y]
        if (b-a != y-x or not 2 <= b-a <= 5 or len(cells) != (b-a)**2
                or len(np.unique(tile)) != 1):
            raise ValueError("emitters must be separated solid same-size squares")
        sizes.add(b-a)
        emitters.append((a, x, int(tile[0, 0])))
    if len(sizes) != 1 or not 1 <= len(emitters) <= 8:
        raise ValueError("expected one to eight common-period emitters")
    n = next(iter(sizes))
    sources = {s for _, _, s in emitters}
    if not 1 <= len(sources) <= (3 if n == 5 else 4):
        raise ValueError("source-category bound exceeded")
    for a, b in itertools.combinations(emitters, 2):
        if not _separate((a[0], a[0]+n, a[1], a[1]+n),
                         (b[0], b[0]+n, b[1], b[1]+n)):
            raise ValueError("emitter squares need a blank-cell gap")
    exterior = grid != background
    exterior[r:e, c:d] = False
    key_cells = np.zeros(grid.shape, bool)
    mapping, key_boxes = {}, {}
    for cells in _components(exterior):
        values = {int(grid[a, b]) for a, b in cells}
        if len(values) == 1:
            continue
        a, b, x, y = _box(cells)
        if (b-a, y-x) != (3, 4) or len(cells) != 12 or len(values) != 2:
            raise ValueError("unknown exterior key")
        tile = grid[a:b, x:y]
        src, dst = int(tile[_ARROW][0]), int(tile[~_ARROW][0])
        if (src == dst or src not in sources or dst in (background, frame_color)
                or not np.all(tile[_ARROW] == src)
                or not np.all(tile[~_ARROW] == dst) or src in mapping):
            raise ValueError("key must encode one direct source-to-paint arrow")
        mapping[src], key_boxes[src] = dst, (a, b, x, y)
        key_cells[a:b, x:y] = True
    if set(mapping) != sources:
        raise ValueError("each board source needs exactly one key")
    remaining = exterior & ~key_cells
    if set(map(int, np.unique(grid[remaining]))) != sources:
        raise ValueError("unexplained exterior pixels")
    masks, stencil_boxes = {}, {}
    for source in sorted(sources):
        cells = np.argwhere(remaining & (grid == source))
        a, b, x, y = _box(cells)
        if (b-a, y-x) != (n, n):
            raise ValueError("stencil extrema must recover the common square")
        tile = grid[a:b, x:y]
        mask = tile == source
        if not np.all(np.isin(tile, [source, background])) or mask.all():
            raise ValueError("stencils are non-solid binary patches")
        masks[source], stencil_boxes[source] = mask.copy(), (a, b, x, y)
    boxes = list(stencil_boxes.values()) + list(key_boxes.values())
    for a, b in itertools.combinations(boxes, 2):
        if not _separate(a, b):
            raise ValueError("complete clue bounds need blank-cell gaps")
    sides = {_side(box, frame) for box in boxes}
    return dict(background=background, frame_color=frame_color, frame=frame,
                shape=interior.shape, n=n, emitters=emitters, masks=masks,
                mapping=mapping, stencil_boxes=stencil_boxes,
                key_boxes=key_boxes, sides=sides)


def _fields(info, metric='manhattan', phase='local', solid=False):
    y, x = np.indices(info['shape'])
    n = info['n']
    distances, active = [], []
    for r, c, source in info['emitters']:
        dy, dx = abs(2*y-(2*r+n-1)), abs(2*x-(2*c+n-1))
        distances.append(dy + dx if metric == 'manhattan' else dy*dy + dx*dx)
        mask = np.ones((n, n), bool) if solid else info['masks'][source]
        rr, cc = (y-r) % n, (x-c) % n
        if phase == 'global':
            rr, cc = y % n, x % n
        active.append(mask[rr, cc])
    distances, active = np.array(distances), np.array(active)
    nearest = distances == distances.min(axis=0)
    return distances, active, nearest


def _render(info, variant=None):
    if variant == 'swapped_stencils':
        sources = sorted(info['masks'])
        info = dict(info, masks={s: info['masks'][sources[(i+1) % len(sources)]]
                                for i, s in enumerate(sources)})
    metric = 'euclidean' if variant == 'euclidean' else 'manhattan'
    phase = 'global' if variant == 'global_phase' else 'local'
    distances, active, nearest = _fields(info, metric, phase, variant == 'solid')
    if variant == 'farther_fills_holes':
        eligible = np.where(active, distances, np.iinfo(np.int32).max)
        nearest = eligible == eligible.min(axis=0)
    participants = active & nearest
    number = participants.sum(axis=0)
    result = np.full(info['shape'], info['background'], np.int8)
    for i, (_, _, source) in enumerate(info['emitters']):
        paint = source if variant == 'no_key' else info['mapping'][source]
        if variant == 'recursive':
            # A concrete erroneous two-pass recoloring, also defined for cycles.
            paint = info['mapping'].get(paint, paint)
        result[participants[i] & (number == 1)] = paint
    result[number >= 2] = info['frame_color']
    if variant == 'all_ties':
        result[nearest.sum(axis=0) >= 2] = info['frame_color']
    elif variant == 'no_collisions':
        for i, (_, _, source) in reversed(list(enumerate(info['emitters']))):
            result[participants[i]] = info['mapping'][source]
    elif variant == 'different_inks_only':
        for ink in set(info['mapping'].values()):
            ink_count = sum(participants[i] for i, (_, _, s) in enumerate(info['emitters'])
                            if info['mapping'][s] == ink)
            cells = (number >= 2) & (ink_count == number)
            result[cells] = ink
    elif variant == 'local_stamps':
        result[:] = info['background']
        n = info['n']
        for r, c, source in info['emitters']:
            result[r:r+n, c:c+n] = np.where(info['masks'][source],
                                           info['mapping'][source], info['background'])
    h, w = info['shape']
    output = np.full((h+2, w+2), info['frame_color'], np.int8)
    output[1:-1, 1:-1] = result
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """No seed, examples, hidden palette, or generator state is consulted."""
    return _render(_parse_input(input_grid))


def _asymmetric(mask):
    others = [np.rot90(mask, k) for k in (1, 2, 3)]
    others += [np.rot90(np.fliplr(mask), k) for k in range(4)]
    return all(not np.array_equal(mask, other) for other in others)


def _sample_mask(rng, n, density_range, asymmetric=False):
    low, high = density_range
    y, x = np.indices((n, n))
    for _ in range(250):
        kind = int(rng.integers(8))
        if kind < 4 or asymmetric:
            mask = rng.random((n, n)) < rng.uniform(low, high)
        elif kind == 4:
            mask = (y == x) | (y+x == n-1)
        elif kind == 5:
            mask = (y == 0) | (y == n-1) | (x == 0) | (x == n-1)
        elif kind == 6:
            mask = (y == n//2) | (x == n//2)
        else:
            mask = (y+x) % 2 == 0
        for edge in (mask[0], mask[-1], mask[:, 0], mask[:, -1]):
            if not edge.any():
                edge[int(rng.integers(n))] = True
        if (low <= float(mask.mean()) <= high and not mask.all()
                and (not asymmetric or _asymmetric(mask))):
            return mask.copy()
    # Exhaustive finite fallback for small patterns; no invisible border cells.
    if n <= 3:
        candidates = []
        for code in range(1, (1 << (n*n))-1):
            mask = np.array([(code >> i) & 1 for i in range(n*n)], bool).reshape(n, n)
            if (low <= mask.mean() <= high and mask[0].any() and mask[-1].any()
                    and mask[:, 0].any() and mask[:, -1].any()
                    and (not asymmetric or _asymmetric(mask))):
                candidates.append(mask)
        if candidates:
            return candidates[int(rng.integers(len(candidates)))].copy()
    raise RuntimeError("bounded mask sampling exhausted")


def _mapping(rng, sources, extra, kind):
    k = len(sources)
    order = list(map(int, rng.permutation(sources)))
    if k == 1 or kind == 'disjoint':
        return {s: int(rng.choice(extra)) for s in sources}
    if kind == 'cycle':
        return {s: order[(i+1) % k] for i, s in enumerate(order)}
    if kind == 'chain':
        return {s: order[i+1] if i+1 < k else int(rng.choice(extra))
                for i, s in enumerate(order)}
    if kind == 'many':
        target = int(rng.choice(sources + extra))
        return {s: target if s != target else int(rng.choice(extra)) for s in sources}
    return {s: int(rng.choice([t for t in sources + extra if t != s])) for s in sources}


def _reserve(occupied, r, c, h, w):
    occupied[max(0, r-1):r+h+1, max(0, c-1):c+w+1] = True


def _available(occupied, h, w):
    integral = np.pad(occupied.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    totals = integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]
    return np.argwhere(totals == 0)


def _emitter_positions(rng, shape, n, count):
    for _ in range(24):
        occupied = np.zeros(shape, bool)
        spots = []
        for i in range(count):
            candidates = _available(occupied, n, n)
            if i == 1:
                parity = sum(spots[0]) % 2
                candidates = candidates[(candidates.sum(axis=1) % 2) == parity]
            if not len(candidates):
                break
            r, c = map(int, candidates[int(rng.integers(len(candidates)))])
            spots.append((r, c))
            _reserve(occupied, r, c, n, n)
        if len(spots) == count:
            return spots
    # Guaranteed separated lattice packing, with randomly allocated offsets.
    rows, cols = ((d+1)//(n+1) for d in shape)
    ry = int(rng.integers(shape[0] - ((rows-1)*(n+1)+n) + 1))
    cx = int(rng.integers(shape[1] - ((cols-1)*(n+1)+n) + 1))
    slots = [(ry+i*(n+1), cx+j*(n+1)) for i in range(rows) for j in range(cols)]
    rng.shuffle(slots)
    first, second = next((a, b) for a, b in itertools.combinations(slots, 2)
                         if sum(a) % 2 == sum(b) % 2)
    slots.remove(first)
    slots.remove(second)
    return [first, second] + slots[:count-2]


def _board(rng, n, count):
    limit = min(24, 29-n)
    choices = [(h, w) for h in range(max(10, n+4), limit+1)
               for w in range(max(10, n+4), 25)
               if ((h-1)//(n+1))*((w-1)//(n+1)) >= max(3, count)]
    weights = np.array([1 / (h*w) for h, w in choices])
    h, w = choices[int(rng.choice(len(choices), p=weights/weights.sum()))]
    return (h-2, w-2), _emitter_positions(rng, (h-2, w-2), n, count)


def _pack_clues(rng, shape, n, sources):
    h, w = shape[0]+2, shape[1]+2
    for attempt in range(24):
        if attempt % 2:
            r, c = int(rng.choice([0, 30-h])), int(rng.choice([0, 30-w]))
        else:
            r, c = int(rng.integers(31-h)), int(rng.integers(31-w))
        frame = (r, r+h, c, c+w)
        occupied = np.zeros((30, 30), bool)
        _reserve(occupied, r, c, h, w)
        items = [('stencil', s, n, n) for s in sources] + [('key', s, 3, 4) for s in sources]
        rng.shuffle(items)
        positions = {}
        for kind, source, hh, ww in items:
            candidates = _available(occupied, hh, ww)
            if not len(candidates):
                break
            # All candidates already lie a blank cell away from the whole board.
            rr, cc = map(int, candidates[int(rng.integers(len(candidates)))])
            positions[kind, source] = (rr, cc)
            _reserve(occupied, rr, cc, hh, ww)
        if len(positions) == len(items):
            return frame, positions
    # Top stencil bank plus right key bank. Orders are independent of maps.
    frame = (30-h, 30, 0, w)
    positions = {}
    for i, source in enumerate(rng.permutation(sources)):
        positions['stencil', int(source)] = (0, i*(n+1))
    for i, source in enumerate(rng.permutation(sources)):
        positions['key', int(source)] = (n+1+i*4, w+1)
    return frame, positions


def _draw_input(rng, info):
    frame, positions = _pack_clues(rng, info['shape'], info['n'], list(info['mapping']))
    r, e, c, d = frame
    grid = np.full((30, 30), info['background'], np.int8)
    grid[r, c:d] = grid[e-1, c:d] = info['frame_color']
    grid[r:e, c] = grid[r:e, d-1] = info['frame_color']
    n = info['n']
    for source, mask in info['masks'].items():
        a, b = positions['stencil', source]
        grid[a:a+n, b:b+n][mask] = source
        a, b = positions['key', source]
        grid[a:a+3, b:b+4] = np.where(_ARROW, source, info['mapping'][source])
    for a, b, source in info['emitters']:
        grid[r+1+a:r+1+a+n, c+1+b:c+1+b+n] = source
    a, b, x, y = _box(np.argwhere(grid != info['background']))
    a, x = max(0, a-int(rng.integers(3))), max(0, x-int(rng.integers(3)))
    b, y = min(30, b+int(rng.integers(3))), min(30, y+int(rng.integers(3)))
    trimmed = grid[a:b, x:y].copy()
    # A structural background convention, not a guessing heuristic.
    return trimmed if 2*np.count_nonzero(trimmed == info['background']) > trimmed.size else grid


def _witnesses(info):
    _, active, nearest = _fields(info)
    participants = active & nearest
    number, ties = participants.sum(axis=0), nearest.sum(axis=0) >= 2
    own_count = {}
    for i, (_, _, source) in enumerate(info['emitters']):
        own_count[source] = own_count.get(source, 0) + participants[i]
    return dict(tie_zero=bool(np.any(ties & (number == 0))),
                tie_one=bool(np.any(ties & (number == 1))),
                tie_many=bool(np.any(number >= 2)),
                same_source_collision=any(np.any(a >= 2) for a in own_count.values()),
                farther_hole=bool(np.any((number == 0) & (active & ~nearest).any(axis=0))))


def _make_pair(rng, mode, training, index):
    diagnostic = training and index < 2
    for _ in range(160):
        if mode == 'scale':
            n = int(rng.choice([2, 3] if training else [4, 5]))
        else:
            n = int(rng.choice([2, 3, 4] if mode == 'count' else [3, 4, 5]))
        if training and index == 0:
            n = 3 if mode == 'scale' else int(rng.choice([3, 4]))
        if mode == 'count':
            count = 2 if training else int(rng.integers(4, 9))
        else:
            count = int(rng.integers(2, 7))
        if diagnostic:
            count = 2 if index == 0 or mode == 'count' else 3
            k = 2 if index == 0 or mode != 'count' else 1
        else:
            k = int(rng.integers(1, min(count, 3 if n == 5 else 4)+1))
        colors = list(map(int, rng.permutation(10)))
        background, frame_color = colors[:2]
        sources, extra = colors[2:2+k], colors[2+k:]
        density = ((.56, .85) if training else (.16, .44)) if mode == 'density' else (.16, .88)
        masks = {s: _sample_mask(rng, n, density, asymmetric=diagnostic and index == 0 and i == 0)
                 for i, s in enumerate(sources)}
        if not diagnostic and k > 1 and rng.random() < .16:
            masks[sources[-1]] = masks[sources[0]].copy()
        kind = str(rng.choice(['disjoint', 'chain', 'cycle', 'many', 'mixed']))
        if training and index == 0:
            kind = 'chain' if mode == 'count' else 'disjoint'
        elif training and index == 1:
            kind = 'cycle' if k > 1 else 'disjoint'
        mapping = _mapping(rng, sources, extra, kind)
        shape, spots = _board(rng, n, count)
        assignments = sources + list(map(int, rng.choice(sources, count-k)))
        rng.shuffle(assignments)
        info = dict(background=background, frame_color=frame_color, n=n,
                    shape=shape, emitters=[(*p, s) for p, s in zip(spots, assignments)],
                    masks=masks, mapping=mapping)
        output = _render(info)
        if diagnostic:
            witnesses = _witnesses(info)
            if index == 0:
                if not all(witnesses[t] for t in ('tie_zero', 'tie_one', 'tie_many', 'farther_hole')):
                    continue
                if any(np.array_equal(output, _render(info, v))
                       for v in ('global_phase', 'euclidean', 'local_stamps', 'no_key',
                                 'swapped_stencils')):
                    continue
            elif not witnesses['same_source_collision']:
                continue
        grid = _draw_input(rng, info)
        # The layout parser also rejects accidentally touching/ambiguous clues.
        parsed = _parse_input(grid)
        if not np.array_equal(output, _render(parsed)):
            raise AssertionError("input construction lost emitter or legend data")
        return {'input': grid, 'output': output}
    raise RuntimeError("bounded diagnostic sampling exhausted")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    mode = str(rng.choice(['scale', 'count', 'density']))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            for _ in range(12):
                pair = _make_pair(rng, mode, split == 'train', i)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct-input sampling exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _audit_cells(info, output):
    """Cell-level invariants use scalar distances and stencil coordinates."""
    h, w = info['shape']
    assert output.shape == (h+2, w+2)
    ink, bg, n = info['frame_color'], info['background'], info['n']
    assert np.all(output[0] == ink) and np.all(output[-1] == ink)
    assert np.all(output[:, 0] == ink) and np.all(output[:, -1] == ink)
    for y in range(h):
        for x in range(w):
            best, paints = None, []
            for r, c, source in info['emitters']:
                dist = abs(2*y-(2*r+n-1)) + abs(2*x-(2*c+n-1))
                if best is None or dist < best:
                    best, paints = dist, []
                if dist == best and info['masks'][source][(y-r) % n, (x-c) % n]:
                    paints.append(info['mapping'][source])
            expected = bg if not paints else paints[0] if len(paints) == 1 else ink
            assert int(output[y+1, x+1]) == expected, 'territory/phase/coincidence invariant failed'


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            grid, output = _check_grid(pair['input']), _check_grid(pair['output'])
            key = _key(grid)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            info = _parse_input(grid)
            assert np.array_equal(output, solve(grid))
            _audit_cells(info, output)
            assert set(map(int, np.unique(output))).issubset(set(map(int, np.unique(grid))))
            analyses[split].append(info)
    train = analyses['train']
    assert any(len(a['mapping']) >= 2 for a in train)
    assert any(_asymmetric(m) for a in train for m in a['masks'].values())
    evidence = [_witnesses(a) for a in train]
    assert all(any(e[k] for e in evidence) for k in
               ('tie_zero', 'tie_one', 'tie_many', 'same_source_collision', 'farther_hole'))
    # These are concrete simpler rules, each contradicted by a demonstration.
    alternatives = ('solid', 'no_key', 'swapped_stencils', 'local_stamps', 'all_ties', 'no_collisions',
                    'different_inks_only', 'farther_fills_holes', 'global_phase',
                    'euclidean', 'recursive')
    for variant in alternatives:
        assert any(not np.array_equal(p['output'], _render(a, variant))
                   for p, a in zip(puzzle['train'], train)), 'undistinguished shortcut: ' + variant
    te = analyses['test']
    scale = max(a['n'] for a in train) < min(a['n'] for a in te)
    count = max(len(a['emitters']) for a in train) < min(len(a['emitters']) for a in te)
    density = (min(float(m.mean()) for a in train for m in a['masks'].values())
               > max(float(m.mean()) for a in te for m in a['masks'].values()))
    assert scale or count or density, 'missing visible structural train/test shift'


def measure_puzzle(puzzle):
    """Small JSON-friendly metrics derived from returned grids, not RNG metadata."""
    records = []
    for split in ('train', 'test'):
        for i, pair in enumerate(puzzle[split]):
            a = _parse_input(pair['input'])
            _, active, nearest = _fields(a)
            number, owner_count = (active & nearest).sum(axis=0), nearest.sum(axis=0)
            records.append(dict(split=split, pair=i+1, n=a['n'], emitters=len(a['emitters']),
                                categories=len(a['mapping']), input_shape=list(np.shape(pair['input'])),
                                output_shape=list(np.shape(pair['output'])), sides=sorted(a['sides']),
                                background=a['background'], frame_color=a['frame_color'],
                                masks=[m.astype(int).tolist() for m in a['masks'].values()],
                                densities=[round(float(m.mean()), 4) for m in a['masks'].values()],
                                disconnected=sum(len(_components(m)) > 1 for m in a['masks'].values()),
                                centers=[[2*r+a['n']-1, 2*c+a['n']-1] for r, c, _ in a['emitters']],
                                overlap=any(t in a['mapping'] for t in a['mapping'].values()),
                                many_to_one=len(set(a['mapping'].values())) < len(a['mapping']),
                                tie_cells=int(np.count_nonzero(owner_count >= 2)),
                                triple_ties=int(np.count_nonzero(owner_count >= 3)),
                                highlights=int(np.count_nonzero(number >= 2)), **_witnesses(a)))
    return records


def self_test():
    """Bounded 42-seed coverage, explicit counts, ablations, and metamorphisms."""
    fixture_rng = np.random.default_rng(911)
    # Solver breadth beyond the sampler: a single edge-touching emitter and
    # a disconnected diagonal stencil, including even periods and negative phases.
    for n in (2, 3, 4, 5):
        a = dict(background=0, frame_color=9, n=n, shape=(10, 12),
                 emitters=[(0, 12-n, 2)], masks={2: np.eye(n, dtype=bool)}, mapping={2: 3})
        grid, expected = _draw_input(fixture_rng, a), _render(a)
        assert np.array_equal(solve(grid), expected)
        _audit_cells(_parse_input(grid), expected)
    # Three active nearest copies coincide, including same-source copies.
    cross = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], bool)
    a = dict(background=0, frame_color=9, n=3, shape=(12, 12),
             emitters=[(0, 0, 2), (0, 8, 2), (8, 4, 2)], masks={2: cross}, mapping={2: 3})
    grid = _draw_input(fixture_rng, a)
    assert int(solve(grid)[4, 6]) == 9
    # Rounding side-four centers down falsely makes this a distance tie. Exact
    # half-cell centers select the first emitter's hole; the farther ring loses.
    ring = np.ones((4, 4), bool)
    ring[1:3, 1:3] = False
    a = dict(background=0, frame_color=9, n=4, shape=(10, 12),
             emitters=[(0, 0, 2), (2, 6, 4)],
             masks={2: np.eye(4, dtype=bool), 4: ring}, mapping={2: 3, 4: 5})
    grid = _draw_input(fixture_rng, a)
    assert int(solve(grid)[1, 6]) == 0
    puzzles = [generate_puzzle(s) for s in _HOST_SEEDS]
    records = [r for p in puzzles for r in measure_puzzle(p)]
    assert len({_fingerprint(p) for p in puzzles}) == 42
    assert {r['n'] for r in records} == {2, 3, 4, 5}
    assert {r['categories'] for r in records} == {1, 2, 3, 4}
    assert {r['emitters'] for r in records} >= {2, 3, 4, 5, 6, 7, 8}
    assert {r['background'] for r in records} == set(range(10))
    assert {r['frame_color'] for r in records} == set(range(10))
    assert {s for r in records for s in r['sides']} == {'above', 'below', 'left', 'right'}
    assert len({tuple(r['output_shape']) for r in records}) >= 45
    assert len({_key(np.asarray(m)) for r in records for m in r['masks']}) >= 100
    assert sum(r['disconnected'] for r in records) >= 30
    assert any(r['triple_ties'] for r in records)
    assert any(r['many_to_one'] for r in records) and any(r['overlap'] for r in records)
    assert _fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0))
    for tr in range(2, 11):
        for te in range(1, min(4, tr-1)+1):
            p = generate_puzzle(3000+tr*10+te, tr, te)
            assert len(p['train']) == tr and len(p['test']) == te
    for tr, te in ((1, 1), (11, 1), (2, 2), (3, 4), (4, 0), (6, 5),
                   (True, 1), (4.0, 1), (4, False), (4, None), (None, 1)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    rng = np.random.default_rng(1979)
    for p in puzzles[:6]:
        for pair in p['train'][:2] + p['test']:
            grid, output = pair['input'], pair['output']
            before = grid.copy()
            assert np.array_equal(solve(grid), output) and np.array_equal(grid, before)
            perm = rng.permutation(10)
            assert np.array_equal(solve(perm[grid]), perm[output]), 'palette covariance'
            a = _parse_input(grid)
            # Moving the independent key bank does not alter the board answer.
            s = next(iter(a['mapping']))
            r, e, c, d = a['key_boxes'][s]
            rr, ee, cc, dd = a['frame']
            occupied = np.zeros(grid.shape, bool)
            _reserve(occupied, rr, cc, ee-rr, dd-cc)
            for box in list(a['stencil_boxes'].values()) + list(a['key_boxes'].values()):
                y, z, x, v = box
                _reserve(occupied, y, x, z-y, v-x)
            locations = _available(occupied, 3, 4)
            if len(locations):
                y, x = map(int, locations[0])
                moved = grid.copy()
                moved[r:e, c:d] = a['background']
                moved[y:y+3, x:x+4] = grid[r:e, c:d]
                assert np.array_equal(solve(moved), output), 'key location should be irrelevant'
    corrupted = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in puzzles[0][s]]
                 for s in ('train', 'test')}
    a = _parse_input(corrupted['test'][0]['input'])
    out = corrupted['test'][0]['output']
    out[1, 1] = (int(out[1, 1])+1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output was accepted')


if __name__ == '__main__':
    self_test()
    print('Interference looms self_test passed (42 seeds).')
