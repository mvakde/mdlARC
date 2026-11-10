"""Count-weighted stencil contests, with all conventions visible in each input.

Only standard-library Python and NumPy are used. No parent generator is imported.
"""
from __future__ import annotations

import copy
import numpy as np


# The source occupies the seven-cell arrow body; the other five cells are paint.
_ARROW = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0]], dtype=bool)
_DIAMOND = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _check_grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _components(mask, diagonal=True):
    remaining = set(map(tuple, np.argwhere(mask)))
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in steps:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _box(cells):
    a = np.asarray(cells)
    return (int(a[:, 0].min()), int(a[:, 0].max()) + 1,
            int(a[:, 1].min()), int(a[:, 1].max()) + 1)


def _transform(a, rotation, reflection):
    return np.rot90(np.fliplr(a) if reflection else a, rotation).copy()


def _untransform(a, rotation, reflection):
    a = np.rot90(a, -rotation)
    return (np.fliplr(a) if reflection else a).copy()


def _asymmetric(mask):
    return all(not np.array_equal(mask, _transform(mask, r, f))
               for r in range(4) for f in (False, True) if r or f)


def _outlines(grid):
    """Each structural color occurs only in its rectangular border(s)."""
    found = []
    for color in np.unique(grid):
        comps = _components(grid == color)
        records = []
        for cells in comps:
            r, e, c, d = _box(cells)
            h, w = e - r, d - c
            if min(h, w) < 5 or len(cells) != 2*h + 2*w - 4:
                break
            a = grid[r:e, c:d]
            if not all(np.all(edge == color) for edge in (a[0], a[-1], a[:, 0], a[:, -1])):
                break
            records.append((int(color), (r, e, c, d)))
        else:
            found.extend(records)
    return found


def _decode_card(tile, border, background):
    """Recognize a card in any rigid orientation, and undo that orientation on its mask."""
    choices = []
    for rotation in range(4):
        for reflection in (False, True):
            a = _transform(tile, rotation, reflection)
            n = a.shape[0] - 6
            if n not in (3, 5) or a.shape[1] != max(4, n) + 2:
                continue
            header = a[1:4, 1:5]
            source, target = int(header[0, 0]), int(header[0, 2])
            if (len({source, target, border, background}) != 4
                    or not np.array_equal(header, np.where(_ARROW, source, target))):
                continue
            stencil = a[5:5+n, 1:1+n]
            if not np.all(np.isin(stencil, [background, source])) or not np.any(stencil == source):
                continue
            expected = np.full(a.shape, background, np.int8)
            _draw_frame(expected, 0, 0, *a.shape, border)
            expected[1:4, 1:5] = header
            expected[5:5+n, 1:1+n] = stencil
            if np.array_equal(expected, a):
                choices.append(dict(source=source, target=target, n=n,
                                    mask=_untransform(stencil == source, rotation, reflection)))
    assert len(choices) <= 1, 'ambiguous catalog orientation'
    return choices[0] if choices else None


def _analyze_input(value):
    grid = _check_grid(value)
    colors, frequencies = np.unique(grid, return_counts=True)
    assert np.count_nonzero(frequencies == frequencies.max()) == 1
    background = int(colors[np.argmax(frequencies)])
    outlines = _outlines(grid)
    cards, boards = [], []
    excluded = np.zeros(grid.shape, bool)
    for color, box in outlines:
        r, e, c, d = box
        card = _decode_card(grid[r:e, c:d], color, background)
        if card is None:
            boards.append((color, box))
        else:
            card.update(border=color, box=box)
            cards.append(card)
        excluded[r:e, c:d] = True
    assert 2 <= len(cards) <= 4 and len(boards) == 1
    frame, bounds = boards[0]
    assert len({a['border'] for a in cards}) == 1
    catalog = cards[0]['border']
    assert len({background, frame, catalog}) == 3
    assert len({a['n'] for a in cards}) == 1
    n = cards[0]['n']
    assert n == 3 or len(cards) <= 3
    assert len({a['source'] for a in cards}) == len(cards)
    table = {a['source']: a for a in cards}
    assert not ({a['target'] for a in cards} | set(table)) & {frame, catalog, background}
    r, e, c, d = bounds
    assert min(e-r, d-c) >= n + 4
    board = grid[r:e, c:d]
    assert all(int(board[rr, cc]) == background for rr, cc in
               ((1, 1), (1, board.shape[1]-2), (board.shape[0]-2, 1),
                (board.shape[0]-2, board.shape[1]-2)))
    # Complete rectangular reservations have a blank-cell halo.
    boxes = [bounds] + [a['box'] for a in cards]
    for i, (a, b, x, y) in enumerate(boxes):
        for aa, bb, xx, yy in boxes[i+1:]:
            assert b < aa or bb < a or y < xx or yy < x
        halo = np.zeros(grid.shape, bool)
        halo[max(0, a-1):b+1, max(0, x-1):y+1] = True
        halo[a:b, x:y] = False
        assert np.all(grid[halo] == background)
    anchors = {s: [] for s in table}
    inner = board[1:-1, 1:-1]
    radius = n // 2
    for cells in _components(inner != background):
        assert len(cells) == 1, 'board keys must be separated singleton anchors'
        rr, cc = cells[0]
        rr, cc = rr + 1, cc + 1
        source = int(board[rr, cc])
        assert source in table
        assert radius + 1 <= rr < board.shape[0] - radius - 1
        assert radius + 1 <= cc < board.shape[1] - radius - 1
        anchors[source].append((rr, cc))
    assert all(anchors.values())
    loose = (~excluded) & (grid != background)
    objects, counts, counts4 = {}, {}, {}
    owner = np.full(grid.shape, -1, int)
    serial = 0
    for color in sorted(set(map(int, grid[loose]))):
        comps = _components(loose & (grid == color))
        objects[color] = comps
        counts[color] = len(comps)
        counts4[color] = len(_components(loose & (grid == color), False))
        for comp in comps:
            a, b, x, y = _box(comp)
            assert b-a <= 3 and y-x <= 3
            for rr, cc in comp:
                owner[rr, cc] = serial
            serial += 1
    for rr, cc in np.argwhere(owner >= 0):
        neighbors = owner[max(0, rr-1):rr+2, max(0, cc-1):cc+2]
        assert np.all((neighbors < 0) | (neighbors == owner[rr, cc]))
    assert set(table) <= set(counts)
    assert all(1 <= counts[s] <= 6 for s in table)
    layers = {}
    for source, points in anchors.items():
        layer = np.zeros(board.shape, bool)
        for rr, cc in points:
            layer[rr-radius:rr+radius+1, cc-radius:cc+radius+1] |= table[source]['mask']
        layers[source] = layer
    return dict(grid=grid, background=background, frame=frame, catalog=catalog,
                bounds=bounds, board=board, table=table, anchors=anchors, n=n,
                objects=objects, counts=counts, counts4=counts4, layers=layers)


def _predict(meta, strengths=None, solid=False, global_winner=False, add_anchor_strength=False,
             recursive=False, rotate_masks=False):
    """Local competition between source categories, not between individual anchors."""
    sources = sorted(meta['table'])
    strengths = meta['counts'] if strengths is None else strengths
    layers = []
    radius = meta['n'] // 2
    for s in sources:
        layer = meta['layers'][s].copy()
        if rotate_masks:
            layer[:] = False
            for r, c in meta['anchors'][s]:
                layer[r-radius:r+radius+1, c-radius:c+radius+1] |= np.rot90(meta['table'][s]['mask'])
        if solid:
            for r, c in meta['anchors'][s]:
                layer[r-radius:r+radius+1, c-radius:c+radius+1] = True
        if global_winner and strengths[s] != max(strengths[t] for t in sources):
            layer[:] = False
        layers.append(layer)
    if add_anchor_strength:
        votes = []
        for s in sources:
            multiplicity = np.zeros(meta['board'].shape, int)
            for r, c in meta['anchors'][s]:
                multiplicity[r-radius:r+radius+1, c-radius:c+radius+1] += meta['table'][s]['mask']
            votes.append(multiplicity * strengths[s])
        scores = np.stack(votes)
    else:
        scores = np.stack([np.where(layer, strengths[s], 0) for s, layer in zip(sources, layers)])
    best = scores.max(axis=0)
    unique = ((scores == best).sum(axis=0) == 1) & (best > 0)
    out = np.full(meta['board'].shape, meta['background'], np.int8)
    _draw_frame(out, 0, 0, *out.shape, meta['frame'])
    for i, s in enumerate(sources):
        paint = meta['table'][s]['target']
        if recursive:
            visited = {s}
            while paint in meta['table'] and paint not in visited:
                visited.add(paint)
                paint = meta['table'][paint]['target']
        out[unique & (scores[i] == best)] = paint
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer from this grid alone, without examples or generation state."""
    return _predict(_analyze_input(input_grid))


def _draw_frame(grid, r, c, h, w, color):
    grid[r, c:c+w] = grid[r+h-1, c:c+w] = color
    grid[r:r+h, c] = grid[r:r+h, c+w-1] = color


def _card(source, target, mask, background, border):
    n = len(mask)
    tile = np.full((n+6, max(4, n)+2), background, np.int8)
    _draw_frame(tile, 0, 0, *tile.shape, border)
    tile[1:4, 1:5] = np.where(_ARROW, source, target)
    tile[5:5+n, 1:1+n] = np.where(mask, source, background)
    return tile


def _reserve(blocked, r, c, h, w):
    blocked[max(0, r-1):r+h+1, max(0, c-1):c+w+1] = True


def _available(blocked, h, w):
    if h > blocked.shape[0] or w > blocked.shape[1]:
        return np.empty((0, 2), int)
    sums = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    totals = sums[h:, w:] - sums[:-h, w:] - sums[h:, :-w] + sums[:-h, :-w]
    return np.argwhere(totals == 0)


def _object_shape(rng):
    kind = int(rng.integers(8))
    y, x = np.indices((3, 3))
    if kind == 0:
        mask = _DIAMOND.copy()
    elif kind == 1:
        mask = (x == y) | (x+y == 2)
    elif kind == 2:
        mask = (x == 1) | (y == 1)
    elif kind == 3:
        mask = np.ones((int(rng.integers(1, 4)), int(rng.integers(1, 4))), bool)
    else:
        mask = np.zeros((3, 3), bool)
        mask[1, 1] = True
        for _ in range(int(rng.integers(0, 8))):
            options = [(r, c) for r in range(3) for c in range(3) if not mask[r, c]
                       and mask[max(0, r-1):r+2, max(0, c-1):c+2].any()]
            if options:
                r, c = options[int(rng.integers(len(options)))]
                mask[r, c] = True
    mask = _transform(mask, int(rng.integers(4)), bool(rng.integers(2)))
    r, e, c, d = _box(np.argwhere(mask))
    return mask[r:e, c:d].copy()


def _stencil(rng, n):
    y, x = np.indices((n, n))
    kind = int(rng.integers(8))
    if kind == 0:
        mask = (y == 0) | (x == 0) | (x == n-1) | (y == n-1)
    elif kind == 1:
        mask = (x == n//2) | (y == n//2)
    elif kind == 2:
        mask = x == y
    elif kind == 3:
        mask = (x+y) % 2 == 0
    elif kind == 4:
        mask = y % 2 == 0
    elif kind == 5:
        mask = np.ones((n, n), bool)
    else:
        mask = rng.random((n, n)) < rng.uniform(.18, .86)
    if not mask.any():
        mask[n//2, n//2] = True
    return _transform(mask, int(rng.integers(4)), bool(rng.integers(2)))


def _anchor_positions(rng, h, w, n, sources, repetitions, dedupe_witness=False):
    radius = n // 2
    choices = [(r, c) for r in range(radius+1, h-radius-1)
               for c in range(radius+1, w-radius-1)]
    # A close pair gives a genuine shared region. Its direction varies.
    starts = []
    for p in choices:
        for q in choices:
            dr, dc = abs(p[0]-q[0]), abs(p[1]-q[1])
            if max(dr, dc) in range(2, n) and min(dr, dc) <= 2:
                starts.append((p, q))
    first, second = starts[int(rng.integers(len(starts)))] if len(sources) > 1 else (choices[0], choices[0])
    anchors = {sources[0]: [first]}
    used = [first]
    if len(sources) > 1:
        anchors[sources[1]] = [second]
        used.append(second)
    order = [(s, i) for s in sources for i in range(repetitions[s])
             if not (i == 0 and s in anchors)]
    rng.shuffle(order)
    for source, _ in order:
        options = [p for p in choices if all(max(abs(p[0]-q[0]), abs(p[1]-q[1])) >= 2 for q in used)]
        if dedupe_witness:
            # The two copies of the first category and the competing category
            # have a shared stencil pixel; duplicate stamps are still one vote.
            options = [p for p in options
                       if all(max(p[axis], first[axis], second[axis])
                              - min(p[axis], first[axis], second[axis]) < n for axis in (0, 1))]
        if not options:
            return None
        p = options[int(rng.integers(len(options)))]
        anchors.setdefault(source, []).append(p)
        used.append(p)
    return anchors


def _informative_masks(rng, n, anchors, sources, witness, dedupe_witness=False):
    masks = {s: _stencil(rng, n) for s in sources}
    if len(sources) >= 2:
        a, b = sources[:2]
        pa, pb = anchors[a][0], anchors[b][0]
        radius = n // 2
        squares = [{(r, c) for r in range(p[0]-radius, p[0]+radius+1)
                    for c in range(p[1]-radius, p[1]+radius+1)} for p in (pa, pb)]
        overlap = sorted(squares[0] & squares[1])
        common = overlap[int(rng.integers(len(overlap)))]
        protected_a = None
        for s, p, region in ((a, pa, squares[0]-squares[1]), (b, pb, squares[1]-squares[0])):
            exclusive = sorted(region)[int(rng.integers(len(region)))]
            for rr, cc in (common, exclusive):
                masks[s][rr-p[0]+radius, cc-p[1]+radius] = True
            if s == a:
                protected_a = {(rr-pa[0]+radius, cc-pa[1]+radius) for rr, cc in (common, exclusive)}
        if witness:
            # Preserve the shared/exclusive cells while making orientation informative.
            for _ in range(80):
                if _asymmetric(masks[a]) and not masks[a].all():
                    break
                r, c = map(int, rng.integers(0, n, size=2))
                if (r, c) not in protected_a:
                    masks[a][r, c] = not masks[a][r, c]
        if dedupe_witness:
            points = [anchors[a][0], anchors[b][0], anchors[a][1]]
            common_pixels = [(r, c)
                             for r in range(max(p[0] for p in points)-radius,
                                            min(p[0] for p in points)+radius+1)
                             for c in range(max(p[1] for p in points)-radius,
                                            min(p[1] for p in points)+radius+1)]
            r, c = common_pixels[int(rng.integers(len(common_pixels)))]
            for source, p in ((a, points[0]), (b, points[1]), (a, points[2])):
                masks[source][r-p[0]+radius, c-p[1]+radius] = True
    return masks


def _construction_output(shape, anchors, masks, mapping, counts, background, frame):
    """Labels use placed category counts and explicit translated pixel ownership."""
    voters = {}
    radius = next(iter(masks.values())).shape[0] // 2
    for source, points in anchors.items():
        for r, c in points:
            for dr, dc in np.argwhere(masks[source]):
                p = (r-radius+int(dr), c-radius+int(dc))
                voters.setdefault(p, set()).add(source)
    out = np.full(shape, background, np.int8)
    _draw_frame(out, 0, 0, *shape, frame)
    for (r, c), sources in voters.items():
        strength = max(counts[s] for s in sources)
        winners = [s for s in sources if counts[s] == strength]
        if len(winners) == 1:
            out[r, c] = mapping[winners[0]]
    return out


def _layout(rng, h, w, cards, fallback=False):
    blocked = np.zeros((30, 30), bool)
    if fallback:
        top, left = cards[0].shape[0]+1, 0
        _reserve(blocked, top, left, h, w)
        positions = []
        c = 0
        for tile in cards:
            positions.append((0, c))
            _reserve(blocked, 0, c, *tile.shape)
            c += tile.shape[1]+1
        return top, left, positions, blocked
    for _ in range(20):
        blocked[:] = False
        if rng.random() < .6:
            top, left = int(rng.choice([0, 30-h])), int(rng.choice([0, 30-w]))
        else:
            top, left = int(rng.integers(31-h)), int(rng.integers(31-w))
        _reserve(blocked, top, left, h, w)
        positions = []
        for tile in cards:
            options = _available(blocked, *tile.shape)
            if not len(options):
                break
            r, c = map(int, options[int(rng.integers(len(options)))])
            positions.append((r, c))
            _reserve(blocked, r, c, *tile.shape)
        if len(positions) == len(cards):
            return top, left, positions, blocked
    return None


def _place_loose(grid, blocked, objects, rng):
    for source, mask in sorted(objects, key=lambda item: -item[1].size):
        h, w = mask.shape
        nh, nw = grid.shape[0]-h+1, grid.shape[1]-w+1
        allowed = np.ones((nh, nw), bool)
        for r, c in np.argwhere(mask):
            allowed &= ~blocked[r:r+nh, c:c+nw]
        options = np.argwhere(allowed)
        if not len(options):
            return False
        r, c = map(int, options[int(rng.integers(len(options)))])
        grid[r:r+h, c:c+w][mask] = source
        for dr, dc in np.argwhere(mask):
            _reserve(blocked, r+int(dr), c+int(dc), 1, 1)
    return True


def _place_loose_fallback(grid, blocked, objects, rng, top, board_h, board_w):
    # The fixed shelf leaves at least nineteen separated 3x3 slots, even
    # with three side-five cards. There are never more than eighteen objects.
    slots = [(r, c) for r in range(top, 28, 4) for c in range(board_w+1, 28, 4)]
    slots += [(r, c) for r in range(top+board_h+1, 28, 4) for c in range(0, board_w-2, 4)]
    slots = [(r, c) for r, c in slots if not blocked[r:r+3, c:c+3].any()]
    assert len(slots) >= len(objects)
    rng.shuffle(slots)
    for (source, mask), (r, c) in zip(objects, slots):
        h, w = mask.shape
        r += int(rng.integers(4-h))
        c += int(rng.integers(4-w))
        grid[r:r+h, c:c+w][mask] = source


def _make_pair(rng, shift, training, index):
    witness, tie_witness = training and index == 0, training and index == 1
    n = (3 if training else 5) if shift == 'stencil_size' else int(rng.choice([3, 5], p=[.7, .3]))
    if witness:
        n = 3
    k = 2 if witness or tie_witness else int(rng.integers(2, (4 if n == 3 else 3)+1))
    palette = list(map(int, rng.permutation(10)))
    background, frame, catalog = palette[:3]
    sources = palette[3:3+k]
    external = palette[3+k:]
    map_kind = str(rng.choice(['disjoint', 'chain', 'cycle', 'many', 'mixed']))
    if witness:
        map_kind = 'disjoint'
    elif tie_witness:
        map_kind = 'chain'
    if k == 1:
        map_kind = 'disjoint'
    order = list(map(int, rng.permutation(sources)))
    if map_kind == 'disjoint':
        if witness:
            mapping = dict(zip(sources, map(int, rng.choice(external, k, replace=False))))
        else:
            mapping = {s: int(rng.choice(external)) for s in sources}
    elif map_kind == 'chain':
        if tie_witness:
            order = sources
        mapping = {s: order[i+1] if i+1 < k else external[0] for i, s in enumerate(order)}
    elif map_kind == 'cycle':
        mapping = {s: order[(i+1) % k] for i, s in enumerate(order)}
    elif map_kind == 'many':
        mapping = {s: external[0] for s in sources}
    else:
        mapping = {s: int(rng.choice([t for t in sources+external if t != s])) for s in sources}
    cap = 4 if training and shift == 'count' else 6
    counts = {s: int(rng.integers(1, cap+1)) for s in sources}
    if witness:
        counts.update({sources[0]: 3, sources[1]: 4})
    elif tie_witness:
        counts.update({sources[0]: 2, sources[1]: 2})
    elif not training and shift == 'count':
        counts[sources[0]] = int(rng.integers(5, 7))
    while sum(counts.values()) > 15:
        options = [s for s in sources[1:] if counts[s] > 1]
        counts[options[int(rng.integers(len(options)))]] -= 1
    max_repeats = (2 if training else 3) if shift == 'anchor_count' else 2
    repetitions = {s: int(rng.integers(1, max_repeats+1)) for s in sources}
    if witness:
        repetitions = {s: 1 for s in sources}
    if tie_witness:
        repetitions = {sources[0]: 2, sources[1]: 1}
    if not training and shift == 'anchor_count':
        repetitions[sources[0]] = 3
    distractors = [s for s in external if s not in sources]
    rng.shuffle(distractors)
    nd = int(rng.integers(0, min(2, len(distractors))+1))
    objects = []
    for source in sources:
        for j in range(counts[source]):
            mask = _object_shape(rng)
            if witness:
                if source == sources[0]:
                    mask = (_DIAMOND, np.ones((2, 2), bool), np.ones((1, 1), bool))[j]
                else:
                    mask = (np.ones((2, 2), bool), np.ones((1, 3), bool),
                            np.array([[1, 0], [1, 1]], bool), np.ones((1, 1), bool))[j]
            elif tie_witness:
                # Equal object counts but different areas expose census vs area.
                mask = ((np.ones((3, 3), bool), np.ones((1, 1), bool)) if source == sources[0]
                        else (np.ones((2, 2), bool), np.ones((1, 2), bool)))[j]
            objects.append((source, mask.copy()))
    for source in distractors[:nd]:
        if len(objects) >= 18:
            break
        for _ in range(int(rng.integers(1, min(5, 18-len(objects))+1))):
            objects.append((source, _object_shape(rng)))
    rng.shuffle(objects)
    # Retry geometry and packing, retaining the sampled counts and actual shapes.
    for attempt in range(16):
        fallback = attempt >= 12
        if fallback:
            h = w = 13
        else:
            h, w = map(int, rng.integers(n+6, 19, size=2))
        anchors = _anchor_positions(rng, h, w, n, sources, repetitions, tie_witness)
        if anchors is None:
            continue
        masks = _informative_masks(rng, n, anchors, sources, witness, tie_witness)
        card_order = list(map(int, rng.permutation(sources)))
        cards = []
        for s in card_order:
            rotation = 0 if fallback else int(rng.integers(4))
            reflection = False if fallback else bool(rng.integers(2))
            # A card's displayed stencil is masks[s], regardless of where its
            # header lies. Rotating a card never secretly rotates the board.
            canonical_mask = _untransform(masks[s], rotation, reflection)
            cards.append(_transform(_card(s, mapping[s], canonical_mask, background, catalog),
                                    rotation, reflection))
        layout = _layout(rng, h, w, cards, fallback)
        if layout is None:
            continue
        top, left, positions, blocked = layout
        grid = np.full((30, 30), background, np.int8)
        _draw_frame(grid, top, left, h, w, frame)
        for source, points in anchors.items():
            for r, c in points:
                grid[top+r, left+c] = source
        for tile, (r, c) in zip(cards, positions):
            grid[r:r+tile.shape[0], c:c+tile.shape[1]] = tile
        if fallback:
            _place_loose_fallback(grid, blocked, objects, rng, top, h, w)
        elif not _place_loose(grid, blocked, objects, rng):
            continue
        output = _construction_output((h, w), anchors, masks, mapping, counts, background, frame)
        r, e, c, d = _box(np.argwhere(grid != background))
        r, c = max(0, r-int(rng.integers(3))), max(0, c-int(rng.integers(3)))
        e, d = min(30, e+int(rng.integers(3))), min(30, d+int(rng.integers(3)))
        grid = grid[r:e, c:d].copy()
        rotation, reflection = int(rng.integers(4)), bool(rng.integers(2))
        grid = _transform(grid, rotation, reflection)
        output = _transform(output, rotation, reflection)
        meta = _analyze_input(grid)
        assert np.array_equal(_predict(meta), output), 'construction/visible-input disagreement'
        if witness:
            checks = _witnesses(meta)
            if not all(checks[name] for name in ('no_a', 'no_b', 'strict', 'connectivity', 'local', 'not_rotated')):
                continue
        if tie_witness:
            checks = _witnesses(meta)
            if not all(checks[name] for name in ('tie', 'no_dup_votes', 'nonrecursive', 'not_area')):
                continue
        if witness and not any(_asymmetric(a['mask']) for a in meta['table'].values()):
            continue
        return {'input': grid, 'output': output}
    raise RuntimeError('bounded scene packing or witness construction exhausted')


def _witnesses(meta):
    expected = _predict(meta)
    sources = sorted(meta['table'])
    scores = np.stack([np.where(meta['layers'][s], meta['counts'][s], 0) for s in sources])
    maximum = scores.max(0)
    cover = (scores > 0).sum(0)
    ties = (maximum > 0) & ((scores == maximum).sum(0) > 1)
    return dict(no_a=not np.array_equal(expected, _predict(meta, {s: 1 for s in sources})),
                no_b=not np.array_equal(expected, _predict(meta, solid=True)),
                connectivity=not np.array_equal(expected, _predict(meta, meta['counts4'])),
                tie=bool(ties.any()), strict=bool(((cover > 1) & ~ties).any()),
                local=not np.array_equal(expected, _predict(meta, global_winner=True)),
                no_dup_votes=not np.array_equal(expected, _predict(meta, add_anchor_strength=True)),
                nonrecursive=not np.array_equal(expected, _predict(meta, recursive=True)),
                not_area=not np.array_equal(expected, _predict(meta, {
                    s: sum(map(len, meta['objects'][s])) for s in sources})),
                not_rotated=not np.array_equal(expected, _predict(meta, rotate_masks=True)),
                overlap=int((cover > 1).sum()), tie_pixels=int(ties.sum()))


def _legal_count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _legal_count(num_train, 'num_train', 2, 10)
    nq = _legal_count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    shift = str(rng.choice(['count', 'stencil_size', 'anchor_count']))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            for _ in range(10):
                pair = _make_pair(rng, shift, split == 'train', index)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('unable to construct distinct inputs')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, training, testing = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _check_grid(pair['input']), _check_grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            meta = _analyze_input(inp)
            assert np.array_equal(_predict(meta), out)
            assert set(map(int, out.flat)) <= set(map(int, inp.flat))
            assert not np.array_equal(inp, out)
            (training if split == 'train' else testing).append(meta)
    witnesses = [_witnesses(m) for m in training]
    assert any(w['no_a'] and w['no_b'] and w['strict'] and w['connectivity'] and w['local']
               and w['not_rotated'] for w in witnesses)
    assert any(w['tie'] and w['no_dup_votes'] and w['nonrecursive'] and w['not_area'] for w in witnesses)
    assert any(_asymmetric(a['mask']) and not a['mask'].all() for m in training for a in m['table'].values())
    assert any(any(a['target'] in m['table'] for a in m['table'].values()) for m in training)
    # At least one meaningful factor is held out; other factors may overlap.
    assert (max(m['n'] for m in training) < min(m['n'] for m in testing)
            or max(m['counts'][s] for m in training for s in m['table'])
            < min(max(m['counts'][s] for s in m['table']) for m in testing)
            or max(len(p) for m in training for p in m['anchors'].values())
            < min(max(len(p) for p in m['anchors'].values()) for m in testing))


def self_test():
    """Bounded 42-seed coverage, ablations, metamorphic checks, and count contracts."""
    # Hand-specified local contest: the answer below is not generated by solve.
    mask1 = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], bool)
    mask2 = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]], bool)
    fixture = np.zeros((23, 25), np.int8)
    fixture[0:9, 0:6] = _card(1, 3, mask1, 0, 8)
    fixture[0:9, 7:13] = _card(2, 4, mask2, 0, 8)
    _draw_frame(fixture, 10, 0, 9, 9, 9)
    fixture[14, 3], fixture[14, 5] = 1, 2
    fixture[11:14, 11:14] = np.where(_DIAMOND, 1, 0)
    fixture[15:17, 11:13] = 1
    fixture[20, 11] = 1
    for r, c in ((11, 16), (14, 17), (18, 16), (21, 20)):
        fixture[r, c] = 2
    expected = np.array([[int(c) for c in row] for row in
                         ('999999999', '900000009', '900000009', '903340409',
                          '900334009', '903044009', '900000009', '900000009', '999999999')], np.int8)
    assert np.array_equal(solve(fixture), expected)
    assert _predict(_analyze_input(fixture), {1: 6, 2: 4})[5, 4] == 3
    tied = fixture.copy()
    tied[21, 20] = 0
    tied_expected = expected.copy()
    tied_expected[5, 4] = 0
    assert np.array_equal(solve(tied), tied_expected)
    duplicate = tied.copy()
    duplicate[16, 5] = 1
    assert solve(duplicate)[5, 4] == 0
    assert _predict(_analyze_input(duplicate), add_anchor_strength=True)[5, 4] == 3
    larger_object = fixture.copy()
    larger_object[11, 17] = 2
    assert np.array_equal(solve(larger_object), expected)
    ignored = fixture.copy()
    ignored[20:22, 23] = 5
    assert np.array_equal(solve(ignored), expected)
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures = {tuple(_key(p['input']) for s in ('train', 'test') for p in q[s]) for q in puzzles}
    assert len(signatures) == 42
    assert all(np.array_equal(a[k], b[k]) for s in ('train', 'test')
               for a, b in zip(puzzles[0][s], generate_puzzle(0)[s]) for k in ('input', 'output'))
    metas = [_analyze_input(p['input']) for q in puzzles for s in ('train', 'test') for p in q[s]]
    assert {m['n'] for m in metas} == {3, 5}
    assert {len(m['table']) for m in metas} == {2, 3, 4}
    assert {m['background'] for m in metas} == set(range(10))
    assert {m['frame'] for m in metas} == set(range(10))
    assert {m['catalog'] for m in metas} == set(range(10))
    assert {m['counts'][s] for m in metas for s in m['table']} == set(range(1, 7))
    assert len({m['board'].shape for m in metas}) >= 30
    assert len({m['grid'].shape for m in metas}) >= 10
    assert len({(a['mask'].shape, a['mask'].tobytes()) for m in metas for a in m['table'].values()}) >= 60
    assert any(len(_components(a['mask'])) > 1 for m in metas for a in m['table'].values())
    assert any(a['mask'].all() for m in metas for a in m['table'].values())
    assert any(len({a['target'] for a in m['table'].values()}) < len(m['table']) for m in metas)
    assert any(len(m['objects']) > len(m['table']) for m in metas)
    assert {len(p) for m in metas for p in m['anchors'].values()} >= {1, 2, 3}
    rng = np.random.default_rng(88713)
    for puzzle in puzzles[:3]:
        for pair in puzzle['train'] + puzzle['test']:
            original = pair['input'].copy()
            assert np.array_equal(solve(original), pair['output'])
            assert np.array_equal(original, pair['input']), 'solve mutated its input'
            # Reassign every literal color, including structural roles.
            permutation = rng.permutation(10)
            assert np.array_equal(solve(permutation[original]), permutation[pair['output']])
            for rotation, reflection in ((1, False), (0, True)):
                assert np.array_equal(solve(_transform(original, rotation, reflection)),
                                      _transform(pair['output'], rotation, reflection))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            q = generate_puzzle(8100+nt*10+nq, nt, nq)
            assert len(q['train']) == nt and len(q['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    damaged = copy.deepcopy(puzzles[0])
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('incorrect output accepted')
