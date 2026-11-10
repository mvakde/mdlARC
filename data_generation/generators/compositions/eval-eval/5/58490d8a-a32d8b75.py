"""Census wavefront mosaics: object counts set graph radii; waves toggle tiles.

No parent module, files, training labels, seed, or generated state is used by solve.
The only persistent artifacts produced by this module are those a caller requests.
"""
from collections import deque
import hashlib
import numpy as np

BACKGROUND, FRAME, TRACK = 0, 5, 7
INK = (1, 2, 3, 4, 6, 8, 9)
ORTH = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIAG = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
ARROW = ((0, 1), (1, 2), (2, 1), (1, 0))
DIAMOND = ((0, 1), (1, 0), (1, 2), (2, 1))
SQUARE = ((0, 0), (0, 1), (1, 0), (1, 1))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    return tuple(a.shape), np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _components(mask, diagonal=True):
    todo = set(map(tuple, np.argwhere(mask)))
    groups = []
    steps = DIAG if diagonal else ORTH
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, group = [start], []
        while stack:
            r, c = stack.pop()
            group.append((int(r), int(c)))
            for dr, dc in steps:
                p = (r + dr, c + dc)
                if p in todo:
                    todo.remove(p)
                    stack.append(p)
        groups.append(group)
    return groups


def _decode(value):
    """Identify all roles by complete visible frames and their contents."""
    a = _grid(value)
    rings = _components(a == FRAME, diagonal=False)
    assert len(rings) == 3, 'Exactly three isolated rectangular frames are required'
    outside = np.ones(a.shape, bool)
    cards, bounds = {}, {}
    for ring in rings:
        points = np.asarray(ring)
        r, c = map(int, points.min(0))
        r1, c1 = map(int, points.max(0) + 1)
        h, w = r1 - r, c1 - c
        expected = {(y, x) for y in range(r, r1) for x in range(c, c1)
                    if y in (r, r1 - 1) or x in (c, c1 - 1)}
        assert h >= 4 and w >= 4 and set(ring) == expected
        assert outside[max(0, r - 1):r1 + 1, max(0, c - 1):c1 + 1].all(), 'Frames need a blank separating halo'
        outside[r:r1, c:c1] = False
        b = a[r + 1:r1 - 1, c + 1:c1 - 1].copy()
        assert FRAME not in b
        vals = set(map(int, np.unique(b)))
        if b.shape[0] == b.shape[1] and 2 <= len(b) <= 5 and len(vals) == 2 and vals <= set(INK):
            kind = 'tile'
        elif b.shape == (3, 3) and vals == {BACKGROUND, TRACK} and np.count_nonzero(b) == 2:
            points = set(map(tuple, np.argwhere(b == TRACK)))
            assert (1, 1) in points
            k = [i for i, p in enumerate(ARROW) if points == {(1, 1), p}]
            assert len(k) == 1
            kind = 'arrow'
        else:
            assert min(b.shape) >= 4 and max(b.shape) <= 10
            assert BACKGROUND in vals and TRACK in vals and vals <= {BACKGROUND, TRACK, *INK}
            kind = 'network'
        assert kind not in cards
        cards[kind], bounds[kind] = b, (r, r1, c, c1)
    assert set(cards) == {'tile', 'arrow', 'network'}
    network = cards['network']
    sources = []
    for color in sorted(set(map(int, np.unique(network))) & set(INK)):
        positions = np.argwhere(network == color)
        assert len(positions) == 1, 'One source per category color'
        sources.append((color, tuple(map(int, positions[0]))))
    assert 1 <= len(sources) <= 3
    counts, areas, components = {}, {}, {}
    owner = np.full(a.shape, -1, int)
    serial = 0
    assert not np.any(outside & np.isin(a, (FRAME, TRACK)))
    for color in sorted(set(map(int, a[outside])) - {BACKGROUND}):
        assert color in INK
        groups = _components((a == color) & outside)
        counts[color], areas[color], components[color] = len(groups), sum(map(len, groups)), groups
        for group in groups:
            for r, c in group:
                owner[r, c] = serial
            serial += 1
    # Distinct objects, including different colors, have a blank Chebyshev halo.
    for r, c in np.argwhere(owner >= 0):
        hood = owner[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
        assert np.all((hood < 0) | (hood == owner[r, c]))
        assert np.all(outside[max(0, r - 1):r + 2, max(0, c - 1):c + 2])
    assert all(1 <= counts.get(color, 0) <= 8 for color, _ in sources)
    k = next(i for i, p in enumerate(ARROW) if cards['arrow'][p] == TRACK)
    assert max(network.shape) * len(cards['tile']) <= 30
    return dict(tile=cards['tile'], network=network, mask=network != BACKGROUND,
                sources=sources, counts=counts, areas=areas, components=components,
                turn=k, outside=outside, bounds=bounds)


def _distances(mask, source, steps=ORTH):
    distance = np.full(mask.shape, 999, np.int16)
    distance[source] = 0
    queue = deque([source])
    while queue:
        r, c = queue.popleft()
        for dr, dc in steps:
            y, x = r + dr, c + dc
            if (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]
                    and distance[y, x] == 999):
                distance[y, x] = distance[r, c] + 1
                queue.append((y, x))
    return distance


def _hits(meta, behavior='normal'):
    hits = np.zeros(meta['mask'].shape, np.int8)
    rr, cc = np.indices(hits.shape)
    for color, source in meta['sources']:
        count = meta['counts'][color]
        if behavior == 'one_per_source':
            count = 1
        elif behavior == 'count_pixels':
            count = meta['areas'][color]
        elif behavior == 'include_source':
            count += 1
        elif behavior == 'four_connected_objects':
            count = sum(len(_components(_object_mask(g), False)) for g in meta['components'][color])
        elif behavior == 'unlimited':
            count = 998
        if behavior == 'manhattan':
            distance = abs(rr - source[0]) + abs(cc - source[1])
        else:
            distance = _distances(meta['mask'], source, DIAG if behavior == 'diagonal_travel' else ORTH)
        hits += meta['mask'] & (distance < count)
    return hits


def _object_mask(group):
    p = np.asarray(group)
    p = p - p.min(0)
    mask = np.zeros(tuple(p.max(0) + 1), bool)
    mask[p[:, 0], p[:, 1]] = True
    return mask


def _render(hits, tile, turn, behavior='normal'):
    # Only macro coordinates rotate. Tile orientation always remains as shown.
    k = 0 if behavior == 'no_rotation' else (-turn) % 4 if behavior == 'reverse_rotation' else turn
    state = np.rot90(hits, -k)
    t = np.rot90(tile, -k) if behavior == 'rotate_tile' else tile
    colors = np.unique(t)
    complement = colors[0] + colors[1] - t
    n = len(t)
    out = np.zeros((state.shape[0] * n, state.shape[1] * n), np.int8)
    for r, c in np.argwhere(state > 0):
        h = int(state[r, c])
        if behavior == 'erase_even' and h % 2 == 0:
            continue
        block = t if behavior == 'no_exchange' or (h % 2 == 0 and behavior != 'always_exchange') else complement
        out[r * n:(r + 1) * n, c * n:(c + 1) * n] = block
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Count external components, propagate distinct sources, then expand blocks."""
    meta = _decode(input_grid)
    return _render(_hits(meta), meta['tile'], meta['turn'])


def _alternative(meta, behavior):
    return _render(_hits(meta, behavior), meta['tile'], meta['turn'], behavior)


ALTERNATIVES = ('one_per_source', 'count_pixels', 'include_source', 'four_connected_objects',
                'unlimited', 'manhattan', 'diagonal_travel', 'no_exchange',
                'always_exchange', 'erase_even', 'rotate_tile', 'no_rotation', 'reverse_rotation')


def _make_mask(rng, h, w):
    """Mix maze corridors, free branches, loops, and connected random patches."""
    style = int(rng.integers(4))
    mask = np.zeros((h, w), bool)
    if style == 0:
        # DFS on a coarse lattice: unit-width paths with real detours and forks.
        nr, nc = (h + 1) // 2, (w + 1) // 2
        current = (int(rng.integers(nr)), int(rng.integers(nc)))
        seen, stack = {current}, [current]
        mask[2 * current[0], 2 * current[1]] = True
        while stack:
            r, c = stack[-1]
            options = [(r + dr, c + dc) for dr, dc in ORTH
                       if 0 <= r + dr < nr and 0 <= c + dc < nc and (r + dr, c + dc) not in seen]
            if not options:
                stack.pop()
                continue
            y, x = options[int(rng.integers(len(options)))]
            mask[r + y, c + x] = mask[2 * y, 2 * x] = True
            seen.add((y, x))
            stack.append((y, x))
        if rng.random() < .4:
            missing = np.argwhere(~mask)
            rng.shuffle(missing)
            for r, c in missing[:int(rng.integers(1, 4))]:
                if any(0 <= r + dr < h and 0 <= c + dc < w and mask[r + dr, c + dc] for dr, dc in ORTH):
                    mask[r, c] = True
    elif style == 1:
        # A variable U and optional outward twig preserve a barrier shortcut.
        uh, uw = int(rng.integers(3, h + 1)), int(rng.integers(3, w + 1))
        r, c = int(rng.integers(h - uh + 1)), int(rng.integers(w - uw + 1))
        mask[r:r + uh, c] = mask[r:r + uh, c + uw - 1] = True
        mask[r + uh - 1, c:c + uw] = True
        if uh < h:
            twig_row = r - 1 if r > 0 else r + uh
            mask[twig_row, c:c + uw] = True
    else:
        start = (int(rng.integers(h)), int(rng.integers(w)))
        mask[start] = True
        target = int(rng.integers(max(9, h * w // 3), max(10, h * w * 3 // 4)))
        for _ in range(target - 1):
            options = sorted({(int(r + dr), int(c + dc)) for r, c in np.argwhere(mask) for dr, dc in ORTH
                              if 0 <= r + dr < h and 0 <= c + dc < w and not mask[r + dr, c + dc]})
            if not options:
                break
            if style == 2:
                thin = [p for p in options if sum(0 <= p[0] + dr < h and 0 <= p[1] + dc < w
                                                 and mask[p[0] + dr, p[1] + dc] for dr, dc in ORTH) == 1]
                options = thin or options
            mask[options[int(rng.integers(len(options)))]] = True
    # Random reflection/half-turn changes path placement without changing shape.
    if rng.random() < .5:
        mask = np.flipud(mask)
    if rng.random() < .5:
        mask = np.fliplr(mask)
    return mask.copy()


def _network(rng, h, w, nsource, counts, witness):
    for _ in range(160):
        mask = _make_mask(rng, h, w)
        points = np.argwhere(mask)
        if len(points) < max(9, nsource):
            continue
        sources = [tuple(map(int, p)) for p in points[rng.choice(len(points), nsource, replace=False)]]
        meta = dict(mask=mask, sources=list(zip(range(nsource), sources)), counts=dict(enumerate(counts)))
        hits = _hits(meta)
        if not np.any(hits == 1):
            continue
        if witness == 1:
            # The same radius must distinguish detours, diagonal travel, and n+1.
            if any(np.array_equal(hits, _hits(meta, b)) for b in
                   ('manhattan', 'diagonal_travel', 'include_source', 'unlimited', 'one_per_source')):
                continue
        if nsource >= 2 and not np.any(hits == 2):
            continue
        if witness == 3 and nsource == 3 and not np.any(hits == 3):
            continue
        return mask, sources, hits
    # Guaranteed path, bent into a U. Count-three source has a visible detour.
    mask = np.zeros((h, w), bool)
    if nsource == 1:
        path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)]
        path += [(0, c) for c in range(3, w)]
    else:
        # A ten-or-more-cell folded path preserves single/double/triple bands,
        # including when all three radii are six in a 4x4 network.
        path = [(0, c) for c in range(w)] + [(1, w - 1)]
        path += [(2, c) for c in range(w - 1, -1, -1)] + [(3, 0)]
    for p in path:
        mask[p] = True
    if nsource == 1:
        sources = [path[0]]
    elif nsource == 2:
        sources = [path[0], path[3]]
    else:
        sources = [path[1], path[2], path[3]]
    meta = dict(mask=mask, sources=list(zip(range(nsource), sources)), counts=dict(enumerate(counts)))
    return mask, sources, _hits(meta)


def _glyph(rng):
    choices = [DIAMOND, SQUARE, ((0, 0),), ((0, 0), (1, 1), (2, 2)),
               ((0, 0), (0, 1), (0, 2)), ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))]
    if rng.random() < .55:
        cells = choices[int(rng.integers(len(choices)))]
    else:
        occupied = {(1, 1)}
        for _ in range(int(rng.integers(1, 8))):
            frontier = sorted({(r + dr, c + dc) for r, c in occupied for dr, dc in DIAG
                               if 0 <= r + dr < 3 and 0 <= c + dc < 3} - occupied)
            if frontier:
                occupied.add(frontier[int(rng.integers(len(frontier)))])
        cells = tuple(occupied)
    mask = _object_mask(cells)
    mask = np.rot90(mask, int(rng.integers(4)))
    return tuple(map(tuple, np.argwhere(mask)))


def _put(a, blocked, color, cells, rng):
    cells = np.asarray(cells)
    h, w = map(int, cells.max(0) + 1)
    nh, nw = a.shape[0] - h + 1, a.shape[1] - w + 1
    available = np.ones((nh, nw), bool)
    for r, c in cells:
        available &= ~blocked[r:r + nh, c:c + nw]
    options = np.argwhere(available)
    if not len(options):
        return False
    r0, c0 = options[int(rng.integers(len(options)))]
    for dr, dc in cells:
        r, c = int(r0 + dr), int(c0 + dc)
        a[r, c] = color
        blocked[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
    return True


def _scene(rng, cards, objects):
    """At most 12 random layouts and one guaranteed 30x30 slot layout."""
    framed = [np.pad(card, 1, constant_values=FRAME) for card in cards]
    for attempt in range(13):
        fallback = attempt == 12
        if fallback:
            h = w = 30
        else:
            low = 20 if len(objects) < 13 else 25
            h, w = map(int, rng.integers(low, 31, size=2))
        a = np.zeros((h, w), np.int8)
        blocked = np.zeros(a.shape, bool)
        success = True
        column = 1
        for card in framed:
            ch, cw = card.shape
            if fallback:
                r, c = 1, column
                column += cw + 1
            else:
                options = [(r, c) for r in range(h - ch + 1) for c in range(w - cw + 1)
                           if not blocked[r:r + ch, c:c + cw].any()]
                if not options:
                    success = False
                    break
                r, c = options[int(rng.integers(len(options)))]
            a[r:r + ch, c:c + cw] = card
            blocked[max(0, r - 1):r + ch + 1, max(0, c - 1):c + cw + 1] = True
        if not success:
            continue
        if fallback:
            slots = [(r, c) for r in (14, 18, 22, 26) for c in (1, 5, 9, 13, 17, 21, 25)]
            rng.shuffle(slots)
            assert len(objects) <= len(slots)
            for (color, cells), (r, c) in zip(objects, slots):
                for dr, dc in cells:
                    a[r + dr, c + dc] = color
        else:
            for color, cells in sorted(objects, key=lambda o: -len(o[1])):
                if not _put(a, blocked, color, cells, rng):
                    success = False
                    break
            if not success:
                continue
        # Layout and screen orientation are arbitrary; arrows are screen-relative.
        return a
    raise AssertionError('Unreachable packing failure')


def _make_pair(rng, mode, split, index, quarter):
    witness = (index + 1) if split == 'train' and index < 2 else 0
    if witness == 1:
        nsource, counts = 1, [3]
    elif witness == 2:
        nsource, counts = 2, [3, 4]
    else:
        nsource = 3 if split == 'test' and mode == 0 else int(rng.integers(1, 3 if split == 'train' and mode == 0 else 4))
        cap = 4 if split == 'train' and mode == 1 else 6
        counts = list(map(int, rng.integers(1, cap + 1, size=nsource)))
        if split == 'test' and mode == 1:
            counts[0] = int(rng.integers(5, 7))
        if nsource >= 2:
            counts[:2] = [max(3, counts[0]), max(3, counts[1])]
    n = int(rng.choice((2, 3) if split == 'train' and mode == 2 else
                       (4, 5) if split == 'test' and mode == 2 else (2, 3, 4, 5)))
    limit = min(9, 30 // n)
    h, w = map(int, rng.integers(4, limit + 1, size=2))
    # A rectangle exposes macro rotation even when the active subset is simple.
    if witness == 1 and h == w:
        w = 4 if h != 4 else 5
    k = quarter if witness == 1 else (quarter + 1) % 4 if witness == 2 else int(rng.integers(4))
    colors = list(map(int, rng.permutation(INK)))
    source_colors = colors[:nsource]
    tile_colors = list(map(int, rng.choice(INK, 2, replace=False)))
    for _ in range(40):
        bits = rng.random((n, n)) < rng.uniform(.2, .8)
        if bits.any() and not bits.all() and all(not np.array_equal(bits, np.rot90(bits, j)) for j in (1, 2, 3)):
            break
    else:
        bits = np.zeros((n, n), bool)
        bits[0, 0] = True
    tile = np.where(bits, *tile_colors).astype(np.int8)
    need_triple = split == 'test' and mode == 0
    if need_triple:
        counts = [max(3, c) for c in counts]
    mask, positions, hits = _network(rng, h, w, nsource, counts, 3 if need_triple else witness)
    network = np.where(mask, TRACK, BACKGROUND).astype(np.int8)
    for color, pos in zip(source_colors, positions):
        network[pos] = color
    arrow = np.zeros((3, 3), np.int8)
    arrow[1, 1] = arrow[ARROW[k]] = TRACK
    objects = []
    for i, (color, count) in enumerate(zip(source_colors, counts)):
        for j in range(count):
            glyph = (DIAMOND, SQUARE, ((0, 0),))[j] if witness == 1 and i == 0 and j < 3 else _glyph(rng)
            objects.append((color, glyph))
    nd = int(rng.integers(1 if witness else 0, 3))
    for color in colors[nsource:nsource + nd]:
        for _ in range(int(rng.integers(1, 5))):
            objects.append((color, _glyph(rng)))
    assert len(objects) <= 26
    rng.shuffle(objects)
    inp = _scene(rng, [network, tile, arrow], objects)
    # Forward labels use sampled counts/positions, never a parser-derived relabel.
    out = _render(hits, tile, k)
    return {'input': inp, 'output': out}


def _integer(name, value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError(name + ' must be a legal integer')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _integer('num_train', num_train, 2, 10)
    nq = _integer('num_test', num_test, 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    mode, quarter = int(rng.integers(3)), int(rng.choice((1, 3)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            for _ in range(24):
                pair = _make_pair(rng, mode, split, i, quarter)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise AssertionError('Failed bounded uniqueness construction')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_label(meta, output):
    """Audit with synchronous wave sets and per-block inverse rotation.

    This does not call the BFS, hit counter, renderer, or solve used above.
    """
    h, w = meta['mask'].shape
    reached_sets = []
    for color, source in meta['sources']:
        reached, frontier = {source}, {source}
        for _ in range(meta['counts'][color] - 1):
            nxt = {(r + dr, c + dc) for r, c in frontier for dr, dc in ORTH
                   if 0 <= r + dr < h and 0 <= c + dc < w and meta['mask'][r + dr, c + dc]} - reached
            reached |= nxt
            frontier = nxt
        reached_sets.append(reached)
    n, k = len(meta['tile']), meta['turn']
    oh, ow = (w, h) if k % 2 else (h, w)
    assert output.shape == (oh * n, ow * n)
    colors = np.unique(meta['tile'])
    swapped = np.where(meta['tile'] == colors[0], colors[1], colors[0])
    for y in range(oh):
        for x in range(ow):
            if k == 0:
                p = y, x
            elif k == 1:
                p = h - 1 - x, y
            elif k == 2:
                p = h - 1 - y, w - 1 - x
            else:
                p = x, w - 1 - y
            waves = sum(p in reached for reached in reached_sets)
            expected = np.zeros((n, n), np.int8) if waves == 0 else swapped if waves % 2 else meta['tile']
            assert np.array_equal(output[y * n:(y + 1) * n, x * n:(x + 1) * n], expected), 'Incorrect wavefront tile block'


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, train = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, out = _grid(pair['input']), _grid(pair['output'])
            key = _key(a)
            assert key not in seen
            seen.add(key)
            meta = _decode(a)
            _check_label(meta, out)
            assert np.any(out != BACKGROUND)
            if split == 'train':
                train.append((meta, out))
    # Require evidence for the actual inherited mechanisms and their connection.
    assert any(len(m['sources']) == 1 for m, _ in train)
    assert any(len(m['sources']) >= 2 and np.any(_hits(m) == 2) for m, _ in train)
    assert any(any(len({len(g) for g in m['components'][color]}) > 1 and
                       any(len(_components(_object_mask(g), False)) > 1 for g in m['components'][color])
                       for color, _ in m['sources']) for m, _ in train)
    for behavior in ALTERNATIVES:
        assert any(not np.array_equal(_alternative(m, behavior), out) for m, out in train), 'Training fails to distinguish ' + behavior
    # Rotation direction must be uniquely demonstrated within the cyclic maps.
    survivors = []
    for sign in (1, -1):
        for offset in range(4):
            if all(np.array_equal(_render(_hits(m), m['tile'], (sign * m['turn'] + offset) % 4), out)
                   for m, out in train):
                survivors.append((sign, offset))
    assert survivors == [(1, 0)], 'Ambiguous direction convention in training'


def self_test():
    """42 streamed seeds, bounded fingerprints/statistics, count edges and mutations."""
    signatures, shapes, geometries, glyphs, sizes, turns, sources, maxima = set(), set(), set(), set(), set(), set(), set(), set()
    shifts = set()
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        train_meta, test_meta = [], []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                for a in pair.values():
                    digest.update(bytes(a.shape))
                    digest.update(a.tobytes())
                    shapes.add(tuple(a.shape))
                m = _decode(pair['input'])
                geometries.add((m['mask'].shape, m['mask'].tobytes()))
                sizes.add(len(m['tile']))
                turns.add(m['turn'])
                sources.add(len(m['sources']))
                maxima.update(m['counts'][c] for c, _ in m['sources'])
                for groups in m['components'].values():
                    glyphs.update(_key(_object_mask(g)) for g in groups)
                (train_meta if split == 'train' else test_meta).append(m)
                assert np.array_equal(solve(pair['input'].copy()), pair['output'])
        signatures.add(digest.digest())
        if max(len(m['sources']) for m in test_meta) > max(len(m['sources']) for m in train_meta):
            shifts.add('sources')
        if max(len(m['tile']) for m in test_meta) > max(len(m['tile']) for m in train_meta):
            shifts.add('scale')
        if max(m['counts'][c] for m in test_meta for c, _ in m['sources']) > max(m['counts'][c] for m in train_meta for c, _ in m['sources']):
            shifts.add('reach')
        if seed == 0:
            first = puzzle
    assert len(signatures) == 42 and len(shapes) >= 35 and len(geometries) >= 100 and len(glyphs) >= 50
    assert sizes == {2, 3, 4, 5} and turns == {0, 1, 2, 3} and sources == {1, 2, 3}
    assert maxima >= {1, 2, 3, 4, 5, 6} and shifts == {'sources', 'scale', 'reach'}
    repeated = generate_puzzle(0)
    assert all(np.array_equal(p[r], q[r]) for split in first for p, q in zip(first[split], repeated[split]) for r in p)
    # Direct interventions: neither population nor binary tile content is inert.
    witness = next(p for p in first['train'] if len(_decode(p['input'])['sources']) == 1
                   and _decode(p['input'])['counts'][_decode(p['input'])['sources'][0][0]] == 3
                   and any(len(_components(_object_mask(g), False)) > 1
                           for groups in _decode(p['input'])['components'].values() for g in groups))
    m = _decode(witness['input'])
    color = m['sources'][0][0]
    altered = witness['input'].copy()
    for r, c in m['components'][color][-1]:
        altered[r, c] = BACKGROUND
    assert not np.array_equal(solve(altered), witness['output'])
    altered = witness['input'].copy()
    r, r1, c, c1 = m['bounds']['tile']
    colors = np.unique(m['tile'])
    altered[r + 1:r1 - 1, c + 1:c1 - 1] = colors.sum() - m['tile']
    assert not np.array_equal(solve(altered), witness['output'])
    # Hand-specified networks exercise supported extremes outside the sampler:
    # count eight, symmetric tile, a source-free island, and a two-path cycle.
    fixture_rng = np.random.default_rng(7281)
    tile = np.array([[2, 3], [3, 2]], np.int8)
    for cyclic in (False, True):
        if cyclic:
            network = np.array([[1, 7, 2, 0], [7, 0, 7, 0], [7, 7, 7, 0], [0, 0, 0, 7]], np.int8)
            active = network != 0
            active[3, 3] = False
            expected_hits = active.astype(np.int8) * 2
            objects = [(color, SQUARE) for color in (1, 2) for _ in range(8)]
        else:
            network = np.array([[1, 7, 7, 0, 0], [0, 0, 7, 0, 7], [7, 7, 7, 0, 7], [7, 0, 0, 0, 0]], np.int8)
            active = network != 0
            active[:, 4] = False
            expected_hits = active.astype(np.int8)
            objects = [(1, SQUARE) for _ in range(8)]
        for k in range(4):
            arrow = np.zeros((3, 3), np.int8)
            arrow[1, 1] = arrow[ARROW[k]] = TRACK
            a = _scene(fixture_rng, [network, tile, arrow], objects)
            expected = np.kron(np.rot90(expected_hits > 0, -k), tile if cyclic else 5 - tile)
            assert np.array_equal(solve(a.tolist()), expected)
            _check_label(_decode(a), expected)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(7123, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal count accepted')
    damaged = {s: [{r: a.copy() for r, a in p.items()} for p in first[s]] for s in first}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('Corrupted label accepted')
