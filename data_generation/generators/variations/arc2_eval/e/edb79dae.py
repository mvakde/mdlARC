"""Framed-board stencil substitution with a training-learned legend convention."""
import numpy as np


# True and False are the two ordered roles; polarity is learned from training.
_GLYPHS = (
    np.array([[1, 0], [0, 0]], dtype=bool),
    np.array([[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0]], dtype=bool),
)
_SIDES = ('above', 'left', 'below', 'right')


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for point in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
        result.append(cells)
    return result


def _box(cells):
    a = np.asarray(cells)
    return int(a[:, 0].min()), int(a[:, 0].max()) + 1, int(a[:, 1].min()), int(a[:, 1].max()) + 1


def _array_key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _puzzle_key(puzzle):
    return tuple((split, _array_key(p['input']), _array_key(p['output']))
                 for split in ('train', 'test') for p in puzzle[split])


def _validate_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert all(1 <= d <= 30 for d in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((grid >= 0) & (grid <= 9))


def _draw_frame(grid, r, c, h, w, color):
    grid[r, c:c+w] = grid[r+h-1, c:c+w] = color
    grid[r:r+h, c] = grid[r:r+h, c+w-1] = color


def _glyph(convention):
    family, rotation, polarity = convention
    mask = np.rot90(_GLYPHS[family], rotation)
    return mask if polarity else ~mask


def _key_tile(source, target, convention):
    return np.where(_glyph(convention), source, target).astype(np.int8)


def _side(box, frame):
    r, e, c, d = box
    fr, fe, fc, fd = frame
    if e <= fr:
        return 'above'
    if r >= fe:
        return 'below'
    if d <= fc:
        return 'left'
    if c >= fd:
        return 'right'
    raise AssertionError('clue square intersects the board')


def _parse_input(grid):
    """Recover visible objects. No labels, RNG state or sampled metadata are used."""
    _validate_grid(grid)
    frames = []
    for color in np.unique(grid):
        cells = np.argwhere(grid == color)
        r, e, c, d = _box(cells)
        h, w = e-r, d-c
        if h < 5 or w < 5 or len(cells) != 2*h + 2*w - 4:
            continue
        if (np.all(grid[r, c:d] == color) and np.all(grid[e-1, c:d] == color)
                and np.all(grid[r:e, c] == color) and np.all(grid[r:e, d-1] == color)):
            frames.append(((r, e, c, d), int(color)))
    assert len(frames) == 1, 'need a unique complete frame with its own color'
    frame, frame_color = frames[0]
    r, e, c, d = frame
    # The empty inner corners identify background without any outer canvas rim.
    corners = [grid[r+1, c+1], grid[r+1, d-2], grid[e-2, c+1], grid[e-2, d-2]]
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[np.argmax(counts)])
    if len(set(map(int, corners))) == 1:
        background = int(corners[0])
    assert background != frame_color
    interior = grid[r+1:e-1, c+1:d-1]
    places = []
    sizes = set()
    for cells in _components(interior != background):
        a, b, x, y = _box(cells)
        block = interior[a:b, x:y]
        assert b-a == y-x and 2 <= b-a <= 6
        assert len(cells) == (b-a)**2 and len(np.unique(block)) == 1
        sizes.add(b-a)
        places.append((a+r+1, x+c+1, int(block[0, 0])))
    assert len(sizes) == 1 and places
    n = next(iter(sizes))
    sources = {s for _, _, s in places}
    assert 1 <= len(sources) <= 5
    exterior = grid != background
    exterior[r:e, c:d] = False
    keys = []
    key_cells = np.zeros(grid.shape, dtype=bool)
    # A disconnected single-color stencil is deliberately not a key component.
    for cells in _components(exterior):
        values = {int(grid[a, b]) for a, b in cells}
        if len(values) == 1:
            continue
        assert len(values) == 2
        box = _box(cells)
        a, b, x, y = box
        assert len(cells) == (b-a)*(y-x)
        tile = grid[a:b, x:y]
        candidates = {}
        for family in range(len(_GLYPHS)):
            for rotation in range(4):
                for polarity in (0, 1):
                    conv = (family, rotation, polarity)
                    mask = _glyph(conv)
                    if mask.shape != tile.shape:
                        continue
                    src, dst = int(tile[mask][0]), int(tile[~mask][0])
                    if src != dst and np.all(tile[mask] == src) and np.all(tile[~mask] == dst):
                        candidates[conv] = (src, dst)
        assert candidates, 'unknown two-color key glyph'
        _side(box, frame)
        keys.append((box, candidates))
        key_cells[a:b, x:y] = True
    assert len(keys) == len(sources)
    remaining = exterior & ~key_cells
    assert set(map(int, np.unique(grid[remaining]))) == sources
    stencils, stencil_boxes = {}, {}
    for source in sorted(sources):
        cells = np.argwhere(remaining & (grid == source))
        box = _box(cells)
        a, b, x, y = box
        assert (b-a, y-x) == (n, n), 'all pieces must recover one size-matched square'
        assert np.all(np.isin(grid[a:b, x:y], [background, source]))
        mask = grid[a:b, x:y] == source
        assert mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()
        _side(box, frame)
        stencils[source], stencil_boxes[source] = mask.copy(), box
    # Every complete clue rectangle has a blank-cell gap to every other clue.
    boxes = list(stencil_boxes.values()) + [box for box, _ in keys]
    for i, (a, b, x, y) in enumerate(boxes):
        for aa, bb, xx, yy in boxes[i+1:]:
            assert b < aa or bb < a or y < xx or yy < x
    conventions = set.intersection(*(set(k[1]) for k in keys))
    mappings, key_boxes = {}, {}
    for conv in conventions:
        entries = [k[1][conv] for k in keys]
        if len({s for s, _ in entries}) == len(entries) and {s for s, _ in entries} == sources:
            mappings[conv] = dict(entries)
            key_boxes[conv] = {k[1][conv][0]: k[0] for k in keys}
    assert mappings, 'legend has no functional interpretation for all board sources'
    return dict(background=background, frame=frame_color, frame_box=frame,
                n=n, placeholders=places, stencils=stencils, stencil_boxes=stencil_boxes,
                mappings=mappings, key_boxes=key_boxes,
                sides=frozenset(_side(box, frame) for box in boxes))


def _predict(grid, analysis, convention):
    mapping = analysis['mappings'][convention]
    r, e, c, d = analysis['frame_box']
    result = grid[r:e, c:d].copy()
    n = analysis['n']
    for a, b, source in analysis['placeholders']:
        result[a-r:a-r+n, b-c:b-c+n] = np.where(
            analysis['stencils'][source], mapping[source], analysis['background'])
    return result


def _learn_convention(train_pairs):
    possible = None
    for pair in train_pairs:
        analysis = _parse_input(pair['input'])
        fits = {conv for conv in analysis['mappings']
                if np.array_equal(_predict(pair['input'], analysis, conv), pair['output'])}
        possible = fits if possible is None else possible & fits
    assert possible and len(possible) == 1, 'training must uniquely establish key direction'
    return next(iter(possible))


def solve(train_pairs, test_input):
    """Training-conditioned prediction, including cycles ambiguous in isolation."""
    convention = _learn_convention(train_pairs)
    analysis = _parse_input(test_input)
    assert convention in analysis['mappings']
    return _predict(test_input, analysis, convention)


def _asymmetric(mask):
    return all(not np.array_equal(mask, other)
               for other in [np.rot90(mask, k) for k in (1, 2, 3)]
               + [np.rot90(np.fliplr(mask), k) for k in range(4)])


def _random_stencil(rng, n, informative=False):
    if informative:
        for _ in range(100):
            mask = rng.random((n, n)) < rng.uniform(.3, .8)
            if (_asymmetric(mask) and mask[0].any() and mask[-1].any()
                    and mask[:, 0].any() and mask[:, -1].any()):
                return mask
        mask = np.zeros((n, n), dtype=bool)
        mask[0, :] = mask[:, 0] = True
        mask[-1, 1] = True
        return mask
    kind = int(rng.choice(9, p=[.27, .13, .09, .09, .09, .1, .09, .09, .05]))
    y, x = np.indices((n, n))
    if kind == 0:
        mask = rng.random((n, n)) < rng.uniform(.12, .94)
    elif kind == 1:
        mask = np.zeros((n, n), dtype=bool)
        mask[0, :] = mask[:, 0] = True
        for _ in range(int(rng.integers(0, n*n))):
            frontier = [(r, c) for r in range(n) for c in range(n) if not mask[r, c]
                        and any(0 <= a < n and 0 <= b < n and mask[a, b]
                                for a, b in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)))]
            if not frontier:
                break
            mask[frontier[int(rng.integers(len(frontier)))]] = True
    elif kind == 2:
        mask = (y == 0) | (y == n-1) | (x == 0) | (x == n-1)
    elif kind == 3:
        mask = (x == n//2) | (y == n//2)
        if n % 2 == 0:
            mask |= (x == n//2-1) | (y == n//2-1)
    elif kind == 4:
        mask = x == y
        if rng.random() < .5:
            mask |= x+y == n-1
    elif kind == 5:
        mask = ((x+y) % int(rng.integers(2, 4))) == 0
    elif kind == 6:
        mask = (y % 2 == 0) | (y == n-1)
    elif kind == 7:
        mask = rng.random((n, n)) < rng.uniform(.75, .97)
    else:
        mask = np.ones((n, n), dtype=bool)
    # Expose all four square bounds without insisting on connectivity or asymmetry.
    for edge in (mask[0], mask[-1], mask[:, 0], mask[:, -1]):
        if not edge.any():
            edge[int(rng.integers(n))] = True
    # Small rings/crosses can collapse to solid squares. Keep plain recoloring
    # occasional instead of letting those degeneracies dominate the 2x2 case.
    if kind != 8 and mask.all() and rng.random() < .85:
        mask[int(rng.integers(n)), int(rng.integers(n))] = False
    return np.rot90(mask, int(rng.integers(4))).copy()


def _sample_mapping(rng, palette, k, kind):
    sources = list(map(int, palette[:k]))
    external = list(map(int, palette[k:]))
    order = list(map(int, rng.permutation(sources)))
    if k == 1:
        kind = 'disjoint'
    if kind == 'disjoint':
        targets = list(map(int, rng.choice(external, k, replace=k > len(external))))
        return dict(zip(sources, targets))
    if kind == 'many_to_one':
        target = int(rng.choice(palette))
        # No identity edge: two-color keys remain visibly ordered.
        return {s: (external[0] if s == target else target) for s in sources}
    if kind == 'cycle':
        return {s: order[(i+1) % k] for i, s in enumerate(order)}
    if kind == 'several_chains':
        cut = int(rng.integers(2, k-1)) if k >= 4 else int(rng.integers(1, k))
        cuts = {cut, k}
    elif kind == 'partial_chain':
        length = int(rng.integers(2, k+1)) if k < 3 else int(rng.integers(2, k))
        cuts = set(range(length, k+1))
    elif kind == 'chain':
        cuts = {k}
    else:
        return {s: int(rng.choice([t for t in palette if t != s])) for s in sources}
    mapping = {}
    terminal_index = 0
    for i, source in enumerate(order):
        if i+1 in cuts:
            mapping[source] = external[terminal_index % len(external)]
            terminal_index += 1
        else:
            mapping[source] = order[i+1]
    return mapping


def _available(occupied, h, w):
    if h > occupied.shape[0] or w > occupied.shape[1]:
        return np.empty((0, 2), dtype=int)
    summed = np.pad(occupied.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    totals = summed[h:, w:] - summed[:-h, w:] - summed[h:, :-w] + summed[:-h, :-w]
    return np.argwhere(totals == 0)


def _reserve(occupied, r, c, h, w):
    occupied[max(0, r-1):r+h+1, max(0, c-1):c+w+1] = True


def _board(rng, n, k):
    limit = min(26, 29-n)
    compact = rng.random() < .25
    counts = [(a, b) for a in range(1, min(5, (limit-3)//(n+1))+1)
              for b in range(1, min(5, (limit-3)//(n+1))+1) if a*b >= k]
    if compact:
        smallest = min(a*b for a, b in counts)
        counts = [(a, b) for a, b in counts if a*b == smallest]
    weights = np.array([1 / (1 + .15*(a*b-k)) for a, b in counts])
    rows, cols = counts[int(rng.choice(len(counts), p=weights/weights.sum()))]
    min_h, min_w = rows*(n+1)+3, cols*(n+1)+3
    h = int(rng.integers(min_h, min(limit, min_h+2) + 1 if compact else limit+1))
    w = int(rng.integers(min_w, min(limit, min_w+2) + 1 if compact else limit+1))
    count = int(rng.integers(k, min(20, rows*cols)+1))
    mode = int(rng.integers(3))
    if mode == 2:
        for _ in range(6):
            occupied = np.zeros((h-4, w-4), dtype=bool)
            spots = []
            for _ in range(count):
                candidates = _available(occupied, n, n)
                if not len(candidates):
                    break
                r, c = map(int, candidates[int(rng.integers(len(candidates)))])
                spots.append((r+2, c+2))
                _reserve(occupied, r, c, n, n)
            if len(spots) == count:
                return h, w, spots
    # Guaranteed packing; slack is allocated independently among unequal gaps.
    def starts(length, amount):
        slack = length - (amount*(n+1)+3)
        gaps = rng.multinomial(slack, np.full(amount+1, 1/(amount+1))) if mode else np.zeros(amount+1, int)
        offset = 2 + (int(rng.integers(slack+1)) if not mode else int(gaps[0]))
        result = []
        for i in range(amount):
            result.append(offset)
            offset += n+1 + int(gaps[i+1])
        return result
    rs, cs = starts(h, rows), starts(w, cols)
    slots = [(r, c) for r in rs for c in cs]
    spots = [slots[int(i)] for i in rng.permutation(len(slots))[:count]]
    if mode:
        # Move individual squares off the shared lattice while preserving spacing.
        for i in rng.permutation(len(spots)):
            r, c = spots[int(i)]
            for _ in range(10):
                a = int(np.clip(r + rng.integers(-3, 4), 2, h-n-2))
                b = int(np.clip(c + rng.integers(-3, 4), 2, w-n-2))
                if all(j == i or a+n < x or x+n < a or b+n < y or y+n < b
                       for j, (x, y) in enumerate(spots)):
                    spots[int(i)] = (a, b)
                    break
    return h, w, spots


def _clue_positions(rng, h, w, n, sources, convention, allowed, mode):
    kh, kw = _glyph(convention).shape
    for attempt in range(48):
        # Interior, edge and corner frames are all possible. Large boards often
        # need an edge position to leave a whole stencil-wide exterior band.
        if attempt % 3:
            r = int(rng.choice([0, 30-h]))
            c = int(rng.choice([0, 30-w]))
        else:
            r, c = int(rng.integers(31-h)), int(rng.integers(31-w))
        frame = (r, r+h, c, c+w)
        occupied = np.zeros((30, 30), dtype=bool)
        _reserve(occupied, r, c, h, w)
        stencil_order = list(map(int, rng.permutation(sources)))
        key_order = list(map(int, rng.permutation(sources)))
        preferred = [str(rng.choice(allowed)), str(rng.choice(allowed))]
        positions = {}
        failed = False
        for kind, order, oh, ow in (('stencil', stencil_order, n, n), ('key', key_order, kh, kw)):
            for source in order:
                candidates = _available(occupied, oh, ow)
                candidates = [(int(a), int(b)) for a, b in candidates
                              if ((a+oh <= r and 'above' in allowed) or (a >= r+h and 'below' in allowed)
                                  or (b+ow <= c and 'left' in allowed) or (b >= c+w and 'right' in allowed))]
                if not candidates:
                    failed = True
                    break
                if mode == 'nearby' and kind == 'key':
                    sr, sc = positions['stencil', source]
                    distances = [abs(a-sr)+abs(b-sc) for a, b in candidates]
                    near = min(distances)
                    candidates = [p for p, dist in zip(candidates, distances) if dist <= near+1]
                elif mode == 'banks':
                    side = preferred[kind == 'key']
                    favored = [p for p in candidates if _side((p[0], p[0]+oh, p[1], p[1]+ow), frame) == side]
                    if favored:
                        candidates = favored
                a, b = candidates[int(rng.integers(len(candidates)))]
                positions[kind, source] = (a, b)
                _reserve(occupied, a, b, oh, ow)
            if failed:
                break
        if not failed:
            return frame, positions
    return None


def _fallback(rng, n, sources, stencils, mapping, background, frame_color, convention, allowed):
    """Guaranteed shelf packing in any requested exterior direction."""
    q = _SIDES.index(str(rng.choice(allowed)))
    key = np.rot90(_key_tile(sources[0], mapping[sources[0]], convention), -q)
    kh, kw = key.shape
    k = len(sources)
    cols = min(k, (27//(n+1)))
    rows = (k+cols-1)//cols
    bh, bw = rows*(n+1)+3, cols*(n+1)+3
    h = n+1+kh+1+bh
    w = max(bw, k*(n+1)-1, k*(kw+1)-1)
    assert h <= 30 and w <= 30
    grid = np.full((h, w), background, dtype=np.int8)
    top = n+kh+2
    left = int(rng.integers(w-bw+1))
    _draw_frame(grid, top, left, bh, bw, frame_color)
    for i, source in enumerate(rng.permutation(sources)):
        tile = grid[:n, i*(n+1):i*(n+1)+n]
        tile[np.rot90(stencils[int(source)], -q)] = source
    for i, source in enumerate(rng.permutation(sources)):
        grid[n+1:n+1+kh, i*(kw+1):i*(kw+1)+kw] = np.rot90(_key_tile(int(source), mapping[int(source)], convention), -q)
    output = grid[top:top+bh, left:left+bw].copy()
    for i, source in enumerate(rng.permutation(sources)):
        a, b = 2+(i//cols)*(n+1), 2+(i % cols)*(n+1)
        grid[top+a:top+a+n, left+b:left+b+n] = source
        output[a:a+n, b:b+n] = np.where(np.rot90(stencils[int(source)], -q), mapping[int(source)], background)
    return np.rot90(grid, q).copy(), np.rot90(output, q).copy()


def _generate_pair(rng, regime, params, index):
    training = regime == 'source_train'
    assert training or regime == 'target_test'
    shift, convention = params['shift'], params['convention']
    if shift == 'simple':
        n = params['base']
    elif shift in ('scale', 'combined'):
        n = int(rng.choice([2, 3, 4])) if training else int(rng.choice([5, 6]))
    else:
        n = int(rng.choice([2, 3, 4, 5, 6], p=[.18, .26, .26, .20, .10]))
    if training and index == 0:
        n = max(3, n)  # A non-symmetric demonstration teaches orientation.
    k = int(rng.integers(1, (3 if n == 6 else 5)+1))
    if training and index == 1 and shift not in ('mapping', 'combined'):
        k = max(2, k)
    family = str(rng.choice(['disjoint', 'partial_chain', 'several_chains', 'cycle', 'many_to_one', 'chain', 'mixed'],
                            p=[.25, .20, .13, .13, .12, .08, .09]))
    if (training and index == 0) or (training and shift in ('mapping', 'combined')):
        family = 'disjoint'
    elif training and index == 1:
        family = 'partial_chain'
    elif not training and shift in ('mapping', 'combined'):
        k = max(2, k)
        family = str(rng.choice(['partial_chain', 'cycle', 'chain', 'several_chains']))
    colors = rng.permutation(10)
    background, frame_color = map(int, colors[:2])
    # Visible roles must actually vary within a puzzle, even at minimal counts.
    if index == 1 and training:
        old_bg, old_frame = params['first_colors']
        if background == old_bg:
            background = next(int(x) for x in colors if x not in (old_bg, frame_color))
        if frame_color == old_frame:
            frame_color = next(int(x) for x in colors if x not in (old_frame, background))
    palette = np.array([int(x) for x in colors if x not in (background, frame_color)])
    mapping = _sample_mapping(rng, palette, k, family)
    sources = list(mapping)
    stencils = {}
    for i, source in enumerate(sources):
        if i and rng.random() < .18:
            stencils[source] = stencils[int(rng.choice(sources[:i]))].copy()
        else:
            stencils[source] = _random_stencil(rng, n, informative=training and index == 0 and i == 0)
    allowed = list(_SIDES)
    if shift in ('side', 'combined'):
        allowed = params['train_sides'] if training else params['test_sides']
    if shift == 'simple':
        allowed = params['train_sides']
    h, w, spots = _board(rng, n, k)
    mode = str(rng.choice(['nearby', 'banks', 'dispersed'], p=[.23, .37, .40]))
    packed = _clue_positions(rng, h, w, n, sources, convention, allowed, mode)
    if packed is None:
        grid, output = _fallback(rng, n, sources, stencils, mapping, background, frame_color, convention, allowed)
    else:
        frame, positions = packed
        r, e, c, d = frame
        grid = np.full((30, 30), background, dtype=np.int8)
        _draw_frame(grid, r, c, h, w, frame_color)
        for source in sources:
            a, b = positions['stencil', source]
            grid[a:a+n, b:b+n][stencils[source]] = source
            a, b = positions['key', source]
            tile = _key_tile(source, mapping[source], convention)
            grid[a:a+tile.shape[0], b:b+tile.shape[1]] = tile
        assignments = sources + list(map(int, rng.choice(sources, len(spots)-k)))
        rng.shuffle(assignments)
        output = grid[r:e, c:d].copy()
        for (a, b), source in zip(spots, assignments):
            grid[r+a:r+a+n, c+b:c+b+n] = source
            output[a:a+n, b:b+n] = np.where(stencils[source], mapping[source], background)
        # Trim unused canvas, retaining independently varied outer margins.
        a, b, x, y = _box(np.argwhere(grid != background))
        a, x = max(0, a-int(rng.integers(4))), max(0, x-int(rng.integers(4)))
        b, y = min(30, b+int(rng.integers(4))), min(30, y+int(rng.integers(4)))
        grid = grid[a:b, x:y].copy()
    if training and index == 0:
        params['first_colors'] = (background, frame_color)
    return {'input': grid, 'output': output}


def _validate_optional_count(value, name, low, high):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer or None')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be between {low} and {high}')


def _resolve_counts(seed, rng, num_train, num_test):
    _validate_optional_count(num_train, 'num_train', 2, 10)
    _validate_optional_count(num_test, 'num_test', 1, 4)
    if num_train is None and num_test is None:
        test_count = 1 + int(seed) % 4
        train_count = test_count + 1 + ((int(seed)//4) % (10-test_count))
    elif num_train is None:
        test_count = int(num_test)
        train_count = int(rng.integers(test_count+1, 11))
    elif num_test is None:
        train_count = int(num_train)
        test_count = int(rng.integers(1, min(4, train_count-1)+1))
    else:
        train_count, test_count = int(num_train), int(num_test)
    if train_count <= test_count:
        raise ValueError('num_train must be greater than num_test')
    return train_count, test_count


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    side_order = list(map(str, rng.permutation(_SIDES)))
    params = dict(convention=(int(rng.integers(3)), int(rng.integers(4)), int(rng.integers(2))),
                  shift=str(rng.choice(['overlap', 'simple', 'scale', 'side', 'mapping', 'combined'],
                                       p=[.36, .10, .16, .14, .14, .10])),
                  base=int(rng.choice([3, 4])), train_sides=side_order[:int(rng.integers(1, 3))],
                  test_sides=side_order[2:], first_colors=None)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, regime, count in (('train', 'source_train', train_count), ('test', 'target_test', test_count)):
        for index in range(count):
            for _ in range(16):
                pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                pair = _generate_pair(pair_rng, regime, params, index)
                key = _array_key(pair['input'])
                if key not in seen and (_array_key(pair['input']), _array_key(pair['output'])) not in _source_keys():
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('could not construct a distinct pair')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                _validate_grid(grid)
            key = _array_key(pair['input'])
            assert key not in seen
            seen.add(key)
            assert (key, _array_key(pair['output'])) not in _source_keys()
            assert not np.array_equal(pair['input'], pair['output'])
            analyses[split].append(_parse_input(pair['input']))
    convention = _learn_convention(puzzle['train'])
    for split in ('train', 'test'):
        for pair, analysis in zip(puzzle[split], analyses[split]):
            assert convention in analysis['mappings']
            assert np.array_equal(_predict(pair['input'], analysis, convention), pair['output'])
            assert set(map(int, np.unique(pair['output']))).issubset(set(map(int, np.unique(pair['input']))))
    assert any(_asymmetric(mask) and not mask.all() for a in analyses['train'] for mask in a['stencils'].values())
    assert len({a['background'] for a in analyses['train']}) > 1
    assert len({a['frame'] for a in analyses['train']}) > 1


# Exact source fixtures are used only for regressions and content-copy exclusion.
_SOURCE_ROWS = (
    (
        '114488888888888888888888 111488181883838888888888 114488111883338888888888 888888818883338888888888 332288888888888888888888 333288888888888888888888 332288888888888888888888 888888888888888888888888 888555555555555555555588 888588888888888888888588 888583338111833381118588 888583338111833381118588 888583338111833381118588 888588888888888888888588 888583338111888883338588 888583338111888883338588 888583338111888883338588 888588888888888888888588 888588888333811181118588 888588888333811181118588 888588888333811181118588 888588888888888888888588 888555555555555555555588',
        '5555555555555555555 5888888888888888885 5828284848282848485 5822284448222844485 5822288488222884885 5888888888888888885 5828284848888828285 5822284448888822285 5822288488888822285 5888888888888888885 5888882828484848485 5888882228444844485 5888882228848884885 5888888888888888885 5555555555555555555',
    ),
    (
        '22441111111111111111111 22241122221333311666611 22441122221133111661111 11111121221333311661111 33881121221311311666611 33381111111111111111111 33881111111111111111111 11111111111111111111111 66771555555555555555555 66671511111111111111115 66771512222166661333315 11111512222166661333315 11111512222166661333315 11111512222166661333315 11111511111111111111115 11111513333122221666615 11111513333122221666615 11111513333122221666615 11111513333122221666615 11111511111111111111115 11111555555555555555555 11111111111111111111111',
        '555555555555555555 511111111111111115 514444177771888815 514444177111188115 514144177111888815 514144177771811815 511111111111111115 518888144441777715 511881144441771115 518888141441771115 518118141441777715 511111111111111115 555555555555555555',
    ),
    (
        '339949922444444444444444422114 333949992443344344884484422214 339949922443444344884484422114 444444444443433344844444444444 116648877443333344884484444444 111648887443344344888884444444 116648877444444444444444444444 444444444444444444444449999944 444444444444444444444449444444 555555555555555555555449999944 544444444444444444445444499444 543333344444449999945449999944 543333344444449999945444444444 543333344444449999945442222244 543333344444449999945442222244 543333344444449999945442444244 544444444444444444445442444244 543333342222241111145442422244 543333342222241111145444444444 543333342222241111145444444444 543333342222241111145441444444 543333342222241111145441111144 544444444444444444445441144144 544444448888842222245441111144 544444448888842222245441444444 544444448888842222245444444444 544444448888842222245444444444 544444448888842222245444444444 544444444444444444445444444444 555555555555555555555444444444',
        '555555555555555555555 544444444444444444445 549944944444442222245 549444944444442444445 549499944444442222245 549999944444444422445 549944944444442222245 544444444444444444445 549944941111146444445 549444941111146666645 549499941444146644645 549999941444146666645 549944941411146444445 544444444444444444445 544444447744741111145 544444447744741111145 544444447444441444145 544444447744741444145 544444447777741411145 544444444444444444445 555555555555555555555',
    ),
)


def _source_pairs():
    return [{role: np.array([[int(v) for v in row] for row in text.split()], dtype=np.int8)
             for role, text in zip(('input', 'output'), rows)} for rows in _SOURCE_ROWS]


_SOURCE_KEYS = frozenset((_array_key(p['input']), _array_key(p['output'])) for p in _source_pairs())


def _source_keys():
    return _SOURCE_KEYS


def _mapping_features(mapping):
    sources, targets = set(mapping), list(mapping.values())
    cycles = set()
    for source in sources:
        path, current = [], source
        while current in sources and current not in path:
            path.append(current)
            current = mapping[current]
        if current in path:
            cycle = path[path.index(current):]
            cycles.add(tuple(sorted(cycle)))
    roots = sources - set(targets)
    lengths = []
    for source in roots:
        visited, current = set(), source
        while current in sources and current not in visited:
            visited.add(current)
            current = mapping[current]
        lengths.append(len(visited))
    overlap = sum(t in sources for t in targets)
    many = len(set(targets)) < len(targets)
    if not overlap:
        kind = 'disjoint_many' if many else 'disjoint'
    elif cycles:
        kind = 'cycle' if sum(map(len, cycles)) == len(sources) else 'cycle_mixed'
    elif many:
        kind = 'overlap_many'
    elif len(lengths) == 1:
        kind = 'chain'
    elif lengths.count(1) == len(lengths)-1:
        kind = 'chain_and_independent'
    else:
        kind = 'several_chains'
    return dict(map_kind=kind, overlap=overlap, many_to_one=many,
                cycle_lengths=sorted(map(len, cycles)), chain_lengths=sorted(lengths))


def _motif_features(masks):
    counts = dict(rings=0, crosses=0, stripes=0, diagonals=0, checkerboards=0)
    for mask in masks:
        if mask.all():
            continue
        n = mask.shape[0]
        y, x = np.indices(mask.shape)
        cross = (x == n//2) | (y == n//2)
        if n % 2 == 0:
            cross |= (x == n//2-1) | (y == n//2-1)
        prototypes = dict(rings=[(x == 0) | (y == 0) | (x == n-1) | (y == n-1)],
                          crosses=[cross], stripes=[(y % 2 == 0) | (y == n-1)],
                          diagonals=[x == y, (x == y) | (x+y == n-1)],
                          checkerboards=[(x+y) % 2 == 0, (x+y) % 3 == 0])
        for name, patterns in prototypes.items():
            counts[name] += any(np.array_equal(mask, np.rot90(p, k))
                                for p in patterns for k in range(4))
    return counts


def measure_puzzle(puzzle):
    """Review metrics reconstructed from returned grids, not construction choices."""
    convention = _learn_convention(puzzle['train'])
    records = []
    for split in ('train', 'test'):
        for index, pair in enumerate(puzzle[split]):
            a = _parse_input(pair['input'])
            mapping = a['mappings'][convention]
            boxes, keys = a['stencil_boxes'], a['key_boxes'][convention]
            order = sorted(boxes, key=lambda s: (boxes[s][0], boxes[s][2]))
            key_order = sorted(keys, key=lambda s: (keys[s][0], keys[s][2]))
            gap = []
            for source in mapping:
                r, e, c, d = boxes[source]
                rr, ee, cc, dd = keys[source]
                gap.append(max(r-ee, rr-e, 0) + max(c-dd, cc-d, 0))
            masks = list(a['stencils'].values())
            r, e, c, d = a['frame_box']
            n = a['n']
            places = [(x-r, y-c) for x, y, _ in a['placeholders']]
            shortcut = {s: order[i+1] if i+1 < len(order) else mapping[s] for i, s in enumerate(order)}
            record = dict(split=split, pair=index, n=n, categories=len(mapping), blocks=len(places),
                          background=a['background'], frame=a['frame'], convention=list(convention),
                          input_shape=list(pair['input'].shape), output_shape=list(pair['output'].shape),
                          frame_box=list(a['frame_box']), sides=sorted(a['sides']),
                          stencil_sides=sorted({_side(b, a['frame_box']) for b in boxes.values()}),
                          key_sides=sorted({_side(b, a['frame_box']) for b in keys.values()}),
                          edge_touch=r == 0 or c == 0 or e == pair['input'].shape[0] or d == pair['input'].shape[1],
                          disconnected=sum(len(_components(m)) > 1 for m in masks),
                          components=[len(_components(m)) for m in masks],
                          densities=[round(float(m.mean()), 4) for m in masks],
                          symmetric=sum(any(np.array_equal(m, np.rot90(m, i)) for i in (1, 2, 3)) for m in masks),
                          asymmetric=sum(_asymmetric(m) for m in masks),
                          solids=sum(bool(m.all()) for m in masks),
                          duplicate_geometry=len(masks)-len({m.tobytes() for m in masks}),
                          key_gaps=gap, order_matches=order == key_order,
                          shortcut_success=shortcut == mapping,
                          off_lattice=any((x-1) % (n+1) or (y-1) % (n+1) for x, y in places),
                          uneven=sum(len({p[axis] % (n+1) for p in places}) > 1 for axis in (0, 1)) > 0,
                          mask_signatures=[(m.shape, m.tobytes().hex()) for m in masks])
            record.update(_mapping_features(mapping))
            record.update(_motif_features(masks))
            records.append(record)
    return records


def _regression_tests(puzzles):
    source = _source_pairs()
    for pair in source:
        a = _parse_input(pair['input'])
        assert np.array_equal(_predict(pair['input'], a, (1, 0, 1)), pair['output'])
    assert np.array_equal(solve(source[:2], source[2]['input'].copy()), source[2]['output'])
    cyan = _parse_input(source[2]['input'])['stencils'][8]
    assert sorted(map(len, _components(cyan))) == [2, 13]
    assert np.array_equal(source[2]['output'][14:19, 8:13] == 7, cyan)

    moved_key = moved_stencil = moved_block = cycled = False
    rng = np.random.default_rng(930117)
    for puzzle in puzzles[:12]:
        convention = _learn_convention(puzzle['train'])
        # Recolor every role, including background and the frame.
        permutation = rng.permutation(10)
        recolored = {s: [{k: permutation[v] for k, v in p.items()} for p in puzzle[s]] for s in ('train', 'test')}
        for p in recolored['test']:
            assert np.array_equal(solve(recolored['train'], p['input'].copy()), p['output'])
        for pair in puzzle['test']:
            grid = pair['input']
            a = _parse_input(grid)
            sources = sorted(a['stencils'])
            n, bg = a['n'], a['background']
            if not cycled and len(sources) >= 3:
                changed = grid.copy()
                mapping = {s: sources[(i+1) % len(sources)] for i, s in enumerate(sources)}
                for s, box in a['key_boxes'][convention].items():
                    r, e, c, d = box
                    changed[r:e, c:d] = _key_tile(s, mapping[s], convention)
                analysis = _parse_input(changed)
                assert len(analysis['mappings']) == 2, 'cycle should need training direction'
                expected = pair['output'].copy()
                r0, _, c0, _ = a['frame_box']
                for r, c, s in a['placeholders']:
                    expected[r-r0:r-r0+n, c-c0:c-c0+n] = np.where(a['stencils'][s], mapping[s], bg)
                assert np.array_equal(solve(puzzle['train'], changed), expected)
                cycled = True
            for kind in ('key', 'stencil'):
                if (kind == 'key' and moved_key) or (kind == 'stencil' and moved_stencil):
                    continue
                boxes = a['key_boxes'][convention] if kind == 'key' else a['stencil_boxes']
                occupied = np.zeros(grid.shape, dtype=bool)
                r0, e0, c0, d0 = a['frame_box']
                _reserve(occupied, r0, c0, e0-r0, d0-c0)
                for box in list(a['stencil_boxes'].values()) + list(a['key_boxes'][convention].values()):
                    r, e, c, d = box
                    _reserve(occupied, r, c, e-r, d-c)
                for box in boxes.values():
                    r, e, c, d = box
                    positions = _available(occupied, e-r, d-c)
                    if not len(positions):
                        continue
                    rr, cc = map(int, positions[len(positions)//2])
                    changed = grid.copy()
                    tile = grid[r:e, c:d].copy()
                    changed[r:e, c:d] = bg
                    changed[rr:rr+e-r, cc:cc+d-c] = tile
                    assert np.array_equal(solve(puzzle['train'], changed), pair['output'])
                    if kind == 'key':
                        moved_key = True
                    else:
                        moved_stencil = True
                    break
            if not moved_block:
                r0, e0, c0, d0 = a['frame_box']
                occupied = grid[r0+2:e0-2, c0+2:d0-2] != bg
                blocked = np.zeros(occupied.shape, bool)
                for r, c in np.argwhere(occupied):
                    _reserve(blocked, int(r), int(c), 1, 1)
                positions = _available(blocked, n, n)
                if len(positions):
                    rr, cc = map(int, positions[0])
                    rr, cc = rr+r0+2, cc+c0+2
                    r, c, s = a['placeholders'][0]
                    changed, expected = grid.copy(), pair['output'].copy()
                    changed[r:r+n, c:c+n] = bg
                    changed[rr:rr+n, cc:cc+n] = s
                    expected[r-r0:r-r0+n, c-c0:c-c0+n] = bg
                    expected[rr-r0:rr-r0+n, cc-c0:cc-c0+n] = np.where(a['stencils'][s], a['mappings'][convention][s], bg)
                    assert np.array_equal(solve(puzzle['train'], changed), expected)
                    moved_block = True
    assert moved_key and moved_stencil and moved_block and cycled


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert _puzzle_key(generate_puzzle(17)) == _puzzle_key(puzzles[17])
    assert _puzzle_key(generate_puzzle(2027, 7, 3)) == _puzzle_key(generate_puzzle(2027, 7, 3))
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    for i, (tr, te) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(3000+i, tr, te)
        assert tr is None or len(p['train']) == tr
        assert te is None or len(p['test']) == te
    for tr, te in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (2, 3), (4, 4), (True, 1), (2.5, 1)):
        try:
            generate_puzzle(5000, tr, te)
        except ValueError:
            continue
        raise AssertionError('invalid count request was accepted')
    records = []
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        # The prediction function receives training labels and the test input only.
        train = [{k: v.copy() for k, v in p.items()} for p in puzzle['train']]
        for pair in puzzle['test']:
            prediction = solve(train, pair['input'].copy())
            assert np.array_equal(prediction, pair['output'])
        records.extend(measure_puzzle(puzzle))
    assert {r['n'] for r in records} == {2, 3, 4, 5, 6}
    assert {r['categories'] for r in records} == {1, 2, 3, 4, 5}
    assert len({tuple(r['input_shape']) for r in records}) >= 30
    assert len({tuple(r['output_shape']) for r in records}) >= 70
    assert len({tuple(r['convention']) for r in records}) >= 16
    assert {r['convention'][0] for r in records} == {0, 1, 2}
    assert {r['convention'][1] for r in records} == {0, 1, 2, 3}
    assert {r['convention'][2] for r in records} == {0, 1}
    assert {r['background'] for r in records} == set(range(10))
    assert {r['frame'] for r in records} == set(range(10))
    assert len({(tuple(r['frame_box']), tuple(r['sides']), r['n'], r['categories']) for r in records}) >= 200
    assert len({str(mask) for r in records for mask in r['mask_signatures']}) >= 100
    assert sum(r['disconnected'] for r in records) >= 50
    assert sum(r['symmetric'] for r in records) >= 50
    assert sum(r['solids'] for r in records) >= 10
    assert sum(r['duplicate_geometry'] for r in records) >= 30
    for name in ('rings', 'crosses', 'stripes', 'diagonals', 'checkerboards'):
        assert sum(r[name] for r in records) >= 10, name
    assert any(d <= .35 for r in records for d in r['densities'])
    assert any(.8 <= d < 1 for r in records for d in r['densities'])
    assert sum(r['uneven'] for r in records) >= 50
    assert max(r['blocks'] for r in records if r['split'] == 'train') >= 10
    assert sum(r['edge_touch'] for r in records) >= 20
    assert sum(max(r['output_shape']) <= 12 for r in records) >= 3
    assert all(any(side in r['sides'] for r in records) for side in _SIDES)
    assert sum(len(r['sides']) >= 3 for r in records) >= 15
    assert sum(not r['order_matches'] for r in records) >= 60
    assert sum(g >= 8 for r in records for g in r['key_gaps']) >= 100
    assert sum(g <= 2 for r in records for g in r['key_gaps']) >= 10
    assert sum(set(r['stencil_sides']).isdisjoint(r['key_sides']) for r in records) >= 10
    for split in ('train', 'test'):
        subset = [r for r in records if r['split'] == split]
        assert {'disjoint', 'chain_and_independent', 'cycle', 'several_chains'}.issubset({r['map_kind'] for r in subset})
        assert any(r['many_to_one'] for r in subset)
        assert all(any(r['categories'] == k for r in subset) for k in range(1, 6))
    eligible = [r for r in records if r['split'] == 'test' and r['categories'] >= 3]
    assert sum(r['shortcut_success'] for r in eligible) < .3*len(eligible)
    assert len({(tuple(r['sides']), r['map_kind'], r['categories']) for r in records}) >= 80
    overlaps = larger = simple = map_shift = trained_overlap = side_shift = 0
    for puzzle in puzzles:
        measured = measure_puzzle(puzzle)
        tr = [r for r in measured if r['split'] == 'train']
        te = [r for r in measured if r['split'] == 'test']
        sizes_tr, sizes_te = {r['n'] for r in tr}, {r['n'] for r in te}
        overlaps += bool(sizes_tr & sizes_te)
        larger += max(sizes_tr) < min(sizes_te)
        simple += len(sizes_tr | sizes_te) == 1
        map_shift += all(r['overlap'] == 0 for r in tr) and any(r['overlap'] > 0 for r in te)
        trained_overlap += any(r['overlap'] > 0 for r in tr)
        side_shift += bool(set.union(*(set(r['sides']) for r in te)) - set.union(*(set(r['sides']) for r in tr)))
        assert len({r['background'] for r in measured}) >= 2
        assert len({r['frame'] for r in measured}) >= 2
    assert overlaps >= 10 and larger >= 3 and simple >= 1
    assert map_shift >= 3 and trained_overlap >= 10 and side_shift >= 3
    _regression_tests(puzzles)
