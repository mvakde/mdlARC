"""Circuit jigsaw: rigid silhouettes and local stencils jointly close one loop.

No parent generator, file, RNG, or demonstration is consulted by solve().
The construction starts with a square partition and an induced pixel cycle;
solving instead parses cards/keys and searches decorated rigid placements.
"""
from __future__ import annotations

import numpy as np

_EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
_EIGHT = _EDGE + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), 'Invalid grid dimensions.')
    _require(np.issubdtype(a.dtype, np.integer), 'Grid must contain integers.')
    _require(bool(np.all((0 <= a) & (a <= 9))), 'Invalid grid colors.')
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask, neighbors=_EDGE):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in neighbors:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _connected(mask):
    return bool(mask.any()) and len(_components(mask)) == 1


def _box(cells):
    a = np.asarray(cells)
    r, c = a.min(axis=0)
    e, d = a.max(axis=0) + 1
    return int(r), int(e), int(c), int(d)


def _cycle(mask):
    """Ink cells, ignoring paint colors, form one induced four-neighbor cycle."""
    padded = np.pad(mask.astype(np.int8), 1)
    degrees = (padded[:-2, 1:-1] + padded[2:, 1:-1]
               + padded[1:-1, :-2] + padded[1:-1, 2:])
    return (int(mask.sum()) >= 4 and bool(np.all(degrees[mask] == 2))
            and _connected(mask))


def _turns(mask):
    p = np.pad(mask, 1)
    vertical = p[:-2, 1:-1] | p[2:, 1:-1]
    horizontal = p[1:-1, :-2] | p[1:-1, 2:]
    return int(np.sum(mask & vertical & horizontal))


def _parse(input_grid):
    g = _grid(input_grid)
    bg = int(g[0, 0])
    _require(all(np.all(edge == bg) for edge in (g[0], g[-1], g[:, 0], g[:, -1])),
             'The empty outer rim identifies the background.')
    frame_candidates = []
    for color in sorted(set(map(int, g.flat)) - {bg}):
        boxes = []
        valid = True
        for cells in _components(g == color):
            r, e, c, d = _box(cells)
            h, w = e-r, d-c
            border = np.zeros((h, w), bool)
            border[0] = border[-1] = True
            border[:, 0] = border[:, -1] = True
            if (h < 3 or w < 3 or len(cells) != int(border.sum())
                    or not np.array_equal(g[r:e, c:d] == color, border)):
                valid = False
                break
            boxes.append((r, e, c, d))
        if valid and 3 <= len(boxes) <= 5:
            frame_candidates.append((color, boxes))
    _require(len(frame_candidates) == 1, 'Need one exclusive stencil-card border color.')
    frame, boxes = frame_candidates[0]
    cards, card_boxes = {}, {}
    remaining = g != bg
    for r, e, c, d in boxes:
        tile = g[r+1:e-1, c+1:d-1]
        sources = set(map(int, tile.flat)) - {bg}
        _require(len(sources) == 1, 'Each card has one source-colored stencil.')
        source = sources.pop()
        _require(source != frame and source not in cards, 'Card identities must be distinct.')
        mask = tile == source
        _require(int(mask.sum()) >= 2, 'Each stencil must expose at least two active cells.')
        cards[source] = mask.copy()
        card_boxes[source] = (r, e, c, d)
        remaining[r:e, c:d] = False
    mappings, key_boxes = {}, {}
    # Card sources make a marked 2x2 square distinguishable from a key:
    # the key's singleton upper-left cell MUST be a card's source color.
    for cells in _components(remaining, _EIGHT):
        r, e, c, d = _box(cells)
        if (e-r, d-c) != (2, 2) or len(cells) != 4:
            continue
        tile = g[r:e, c:d]
        source, target = int(tile[0, 0]), int(tile[1, 1])
        if (source in cards and source != target
                and tile[0, 1] == tile[1, 0] == target):
            _require(source not in mappings, 'Multiple keys for one source.')
            _require(target not in (bg, frame), 'Paint cannot be a structural color.')
            mappings[source] = target
            key_boxes[source] = (r, e, c, d)
            remaining[r:e, c:d] = False
    _require(set(mappings) == set(cards), 'One directed key is required per card.')
    colors, counts = np.unique(g[remaining], return_counts=True)
    singleton = [int(v) for v, n in zip(colors, counts) if n == 1]
    _require(len(singleton) == 1, 'Need one singleton anchor on a piece.')
    marker = singleton[0]
    _require(marker not in cards and marker not in mappings.values() and marker != frame,
             'Anchor is a separate color role.')
    pieces = []
    for cells in _components(remaining, _EIGHT):
        r, e, c, d = _box(cells)
        values = {int(g[a, b]) for a, b in cells} - {marker}
        _require(len(values) == 1, 'Separate pieces must not touch, even diagonally.')
        source = values.pop()
        _require(source in cards, 'Every piece needs its own card.')
        foot = np.zeros((e-r, d-c), bool)
        anchor = np.zeros_like(foot)
        for a, b in cells:
            foot[a-r, b-c] = True
            anchor[a-r, b-c] = g[a, b] == marker
        _require(_connected(foot) and int(foot.sum()) >= 3, 'Pieces are edge-connected, area at least three.')
        ink = cards[source]
        _require(ink.shape == foot.shape and not np.any(ink & ~foot),
                 'Card coordinates must fit the matching piece silhouette.')
        _require(not anchor.any() or bool(ink[anchor].all()), 'The anchor is an active circuit pixel.')
        if anchor.any():
            ar, ac = map(int, np.argwhere(anchor)[0])
            _require(min(foot.shape) >= 2 and ar in (0, foot.shape[0]-1)
                     and ac in (0, foot.shape[1]-1),
                     'The marked piece has two-dimensional extent and a corner anchor.')
        pieces.append(dict(source=source, foot=foot, ink=ink.copy(), anchor=anchor,
                           box=(r, e, c, d)))
    _require(len(pieces) == len(cards) and {p['source'] for p in pieces} == set(cards),
             'There is exactly one rigid piece per source.')
    _require(sum(bool(p['anchor'].any()) for p in pieces) == 1, 'Anchor belongs to one piece.')
    area = sum(int(p['foot'].sum()) for p in pieces)
    side = int(area ** .5)
    _require(side * side == area and 5 <= side <= 9, 'Piece area determines a side-five-to-nine square.')
    return dict(grid=g, background=bg, frame=frame, marker=marker, side=side,
                pieces=pieces, mapping=mappings, cards=cards,
                card_boxes=card_boxes, key_boxes=key_boxes)


def _neighbor_bits(side):
    return [sum(1 << ((r+dr)*side+c+dc) for dr, dc in _EDGE
                if 0 <= r+dr < side and 0 <= c+dc < side)
            for r in range(side) for c in range(side)]


def _placements(facts, geometry_only=False, rotations=True):
    s = facts['side']
    options, anchor_index = [], None
    for i, piece in enumerate(facts['pieces']):
        marked = bool(piece['anchor'].any())
        if marked:
            anchor_index = i
        choices = {}
        for k in range(4 if rotations else 1):
            foot, ink, anchor = [np.rot90(piece[name], k) for name in ('foot', 'ink', 'anchor')]
            h, w = foot.shape
            if h > s or w > s:
                continue
            if marked:
                offsets = [0] if anchor[0, 0] else []
            else:
                offsets = [r*s+c for r in range(s-h+1) for c in range(s-w+1)]
            base = sum(1 << (int(r)*s+int(c)) for r, c in np.argwhere(foot))
            trace = sum(1 << (int(r)*s+int(c)) for r, c in np.argwhere(ink))
            paint = tuple((int(r)*s+int(c), facts['marker'] if anchor[r, c]
                           else facts['mapping'][piece['source']]) for r, c in np.argwhere(ink))
            for offset in offsets:
                mask, active = base << offset, trace << offset
                cells = tuple((pos+offset, color) for pos, color in paint)
                # Preserve distinct stencil rotations on a symmetric silhouette.
                signature = mask if geometry_only else (mask, active, cells)
                choices.setdefault(signature, (mask, active, cells))
        options.append(list(choices.values()))
    return options, anchor_index


def _search(facts, *, circuit=True, geometry_only=False, rotations=True, limit=2):
    """Exhaustive rigid exact cover, with safe partial circuit-degree pruning.

    A geometric symmetry is harmless only when it has the same decorated result.
    With circuit=False this is also a counterfactual ablation, not the rule.
    """
    s = facts['side']
    full = (1 << (s*s)) - 1
    neighbors = _neighbor_bits(s)
    options, anchor_index = _placements(facts, geometry_only, rotations)
    placed = [None] * len(options)
    answers, signatures = [], set()

    def possible(ink, occupied):
        if not circuit:
            return True
        bits = ink
        while bits:
            bit = bits & -bits
            pos = bit.bit_length()-1
            degree = (neighbors[pos] & ink).bit_count()
            if degree > 2 or degree + (neighbors[pos] & (full ^ occupied)).bit_count() < 2:
                return False
            bits ^= bit
        return True

    def record(ink):
        if geometry_only:
            result = tuple(p[0] for p in placed)
            signature = result
        else:
            result = np.full((s, s), facts['background'], dtype=np.int8)
            for _, _, cells in placed:
                for pos, color in cells:
                    result.flat[pos] = color
            if circuit and not _cycle(result != facts['background']):
                return
            signature = result.tobytes()
        if signature not in signatures:
            signatures.add(signature)
            answers.append(result)

    def search(occupied, ink, remaining):
        if len(answers) >= limit:
            return
        if not remaining:
            if occupied == full:
                record(ink)
            return
        first = (full ^ occupied) & -(full ^ occupied)
        for i in remaining:
            rest = tuple(j for j in remaining if j != i)
            for candidate in options[i]:
                mask, active, _ = candidate
                if mask & first and not mask & occupied:
                    joined = ink | active
                    filled = occupied | mask
                    if possible(joined, filled):
                        placed[i] = candidate
                        search(filled, joined, rest)
                        if len(answers) >= limit:
                            return
            placed[i] = None

    if anchor_index is None or any(not choices for choices in options):
        return []
    rest = tuple(i for i in range(len(options)) if i != anchor_index)
    for candidate in options[anchor_index]:
        if possible(candidate[1], candidate[0]):
            placed[anchor_index] = candidate
            search(candidate[0], candidate[1], rest)
            if len(answers) >= limit:
                break
    return answers


def solve(input_grid: np.ndarray) -> np.ndarray:
    facts = _parse(input_grid)
    answers = _search(facts)
    if len(answers) != 1:
        raise ValueError('Input does not have exactly one painted circuit assembly.')
    return answers[0]


def _make_loop(rng, side, partial=False):
    if partial:
        h, w = side, side
        if rng.integers(2):
            h = int(rng.integers(4, side))
        else:
            w = int(rng.integers(4, side))
    elif rng.random() < .65:
        h = w = side
    else:
        h, w = map(int, rng.integers(4, side+1, size=2))
    mask = np.zeros((side, side), bool)
    mask[0, :w] = mask[h-1, :w] = True
    mask[:h, 0] = mask[:h, w-1] = True
    bounds = np.zeros_like(mask)
    bounds[:h, :w] = True
    for _ in range(25):
        k = int(rng.integers(4))
        a = np.rot90(mask, k).copy()
        segments = [(r, c, d) for r in range(side) for c in range(side-2)
                    for d in range(c+2, side) if a[r, c:d+1].all()]
        if not segments:
            break
        r, c, d = segments[int(rng.integers(len(segments)))]
        rr = r + int(rng.choice([-2, -1, 1, 2]))
        if not 0 <= rr < side:
            continue
        trial = a.copy()
        trial[r, c+1:d] = False
        trial[min(r, rr):max(r, rr)+1, c] = True
        trial[min(r, rr):max(r, rr)+1, d] = True
        trial[rr, c:d+1] = True
        trial = np.rot90(trial, -k).copy()
        if trial[0, 0] and not np.any(trial & ~bounds) and _cycle(trial):
            mask = trial
    return mask


def _grow(rng, side, ids, reserved=None, balanced=False):
    labels = np.full((side, side), -1, dtype=np.int8) if reserved is None else reserved.copy()
    pool = list(map(int, np.flatnonzero(labels.ravel() < 0)))
    if len(pool) < len(ids) or 0 not in pool:
        return None
    if balanced and len(ids) == 4 and reserved is None:
        seeds = [0, side-1, side*(side-1), side*side-1]
    else:
        seeds = [0] + list(map(int, rng.choice([p for p in pool if p != 0], len(ids)-1, replace=False)))
    frontiers = {i: set() for i in ids}
    areas = {i: 1 for i in ids}
    weights = dict(zip(ids, np.exp(rng.uniform(-1, 1, len(ids)))))
    neighbors = [[(r+dr)*side+c+dc for dr, dc in _EDGE
                  if 0 <= r+dr < side and 0 <= c+dc < side]
                 for r in range(side) for c in range(side)]
    for i, pos in zip(ids, seeds):
        labels.flat[pos] = i
    for i, pos in zip(ids, seeds):
        frontiers[i].update(v for v in neighbors[pos] if labels.flat[v] < 0)
    for _ in range(len(pool)-len(ids)):
        eligible = [i for i in ids if frontiers[i]]
        if not eligible:
            return None
        i = min(eligible, key=lambda j: areas[j]/weights[j] + float(rng.random())*.2)
        cells = sorted(frontiers[i])
        pos = cells[int(rng.integers(len(cells)))]
        labels.flat[pos] = i
        areas[i] += 1
        for frontier in frontiers.values():
            frontier.discard(pos)
        frontiers[i].update(v for v in neighbors[pos] if labels.flat[v] < 0)
    return labels


def _paired_inserts(rng, side, count, loop):
    """Identical silhouettes with incongruent ink force stencil-dependent packing."""
    for _ in range(16):
        h, w = [(2, 2), (2, 3), (1, 3), (2, 4)][int(rng.integers(4))]
        proto = np.ones((h, w), bool)
        if h == 2 and w > 2 and rng.random() < .5:
            proto[0, 0] = False
        candidates = []
        for k in range(4):
            foot = np.rot90(proto, k).copy()
            h, w = foot.shape
            for r in range(side-h+1):
                for c in range(side-w+1):
                    if r == 0 and c == 0 and foot[0, 0]:
                        continue
                    ink = loop[r:r+h, c:c+w] & foot
                    if int(ink.sum()) < 2:
                        continue
                    mask = np.zeros((side, side), bool)
                    mask[r:r+h, c:c+w] = foot
                    candidates.append((mask, foot, ink))
        order = rng.permutation(len(candidates))
        for ii in order[:24]:
            a, f, t = candidates[int(ii)]
            for jj in rng.permutation(len(candidates))[:24]:
                b, g, u = candidates[int(jj)]
                if np.any(a & b):
                    continue
                if any(np.array_equal(f, np.rot90(g, k)) and np.array_equal(t, np.rot90(u, k))
                       for k in range(4)):
                    continue
                if not _connected(~(a | b)):
                    continue
                labels = np.full((side, side), -1, dtype=np.int8)
                labels[a], labels[b] = 1, 2
                result = _grow(rng, side, [0] + list(range(3, count)), labels)
                if result is not None:
                    return result
    return None


def _cuts(rng, side, count):
    rectangles = [(0, side, 0, side)]
    while len(rectangles) < count:
        choices = []
        for i, (r, e, c, d) in enumerate(rectangles):
            choices.extend((i, 0, v) for v in range(r+1, e) if min(v-r, e-v)*(d-c) >= 3)
            choices.extend((i, 1, v) for v in range(c+1, d) if min(v-c, d-v)*(e-r) >= 3)
        if not choices:
            return None
        i, axis, v = choices[int(rng.integers(len(choices)))]
        r, e, c, d = rectangles.pop(i)
        rectangles.extend([(r, v, c, d), (v, e, c, d)] if axis == 0
                          else [(r, e, c, v), (r, e, v, d)])
    labels = np.empty((side, side), dtype=np.int8)
    for i, (r, e, c, d) in enumerate(rectangles):
        labels[r:e, c:d] = i
    for _ in range(side*2):
        r, c = map(int, rng.integers(side, size=2))
        dr, dc = _EDGE[int(rng.integers(4))]
        rr, cc = r+dr, c+dc
        if not (0 <= rr < side and 0 <= cc < side) or (r, c) == (0, 0):
            continue
        a, b = int(labels[r, c]), int(labels[rr, cc])
        if a == b or int(np.sum(labels == a)) <= 3 or rng.random() < .35:
            continue
        labels[r, c] = b
        if not _connected(labels == a):
            labels[r, c] = a
    return labels


def _mapping(rng, sources, extras, kind):
    order = list(map(int, rng.permutation(sources)))
    if kind == 'chain':
        return {v: order[i+1] if i+1 < len(order) else int(rng.choice(extras))
                for i, v in enumerate(order)}
    if kind == 'cycle':
        return {v: order[(i+1) % len(order)] for i, v in enumerate(order)}
    if kind == 'many':
        target = int(rng.choice(extras))
        return {v: target for v in sources}
    if kind == 'disjoint':
        return {v: int(rng.choice(extras)) for v in sources}
    palette = sources + extras
    return {v: int(rng.choice([c for c in palette if c != v])) for v in sources}


def _objects(facts):
    bg, frame = facts['background'], facts['frame']
    objects = []
    for p in facts['pieces']:
        tile = np.full(p['foot'].shape, bg, dtype=np.int8)
        tile[p['foot']] = p['source']
        tile[p['anchor']] = facts['marker']
        objects.append((tile, p['foot'].copy()))
        h, w = p['ink'].shape
        card = np.full((h+2, w+2), bg, dtype=np.int8)
        card[0] = card[-1] = frame
        card[:, 0] = card[:, -1] = frame
        card[1:-1, 1:-1][p['ink']] = p['source']
        objects.append((card, np.ones(card.shape, bool)))
        key = np.full((2, 2), facts['mapping'][p['source']], dtype=np.int8)
        key[0, 0] = p['source']
        objects.append((key, np.ones((2, 2), bool)))
    return objects


def _scatter(rng, facts):
    objects = _objects(facts)
    for attempt in range(16):
        order = sorted(range(len(objects)), key=lambda i: -int(objects[i][1].sum())*float(rng.uniform(.8, 1.2)))
        canvas = np.full((30, 30), facts['background'], dtype=np.int8)
        blocked = 0
        maxr = maxc = 1
        placed = True
        aspect = float(rng.uniform(.75, 1.4))
        loose = attempt < 2 and rng.random() < .3
        for i in order:
            tile, reserved = objects[i]
            h, w = tile.shape
            base = sum(1 << (int(r)*30+int(c)) for r, c in np.argwhere(reserved))
            options = []
            for r in range(1, 30-h):
                for c in range(1, 30-w):
                    mask = base << (r*30+c)
                    if mask & blocked:
                        continue
                    hr, wc = max(maxr, r+h), max(maxc, c+w)
                    score = (float(rng.random()) if loose else
                             hr*wc + abs(hr-aspect*wc)*2 + float(rng.random())*45)
                    options.append((score, r, c, mask))
            if not options:
                placed = False
                break
            _, r, c, mask = min(options)
            view = canvas[r:r+h, c:c+w]
            view[tile != facts['background']] = tile[tile != facts['background']]
            horizontal = mask | (mask << 1) | (mask >> 1)
            blocked |= horizontal | (horizontal << 30) | (horizontal >> 30)
            maxr, maxc = max(maxr, r+h), max(maxc, c+w)
        if placed:
            r, e, c, d = _box(np.argwhere(canvas != facts['background']))
            r, c = max(0, r-1-int(rng.integers(3))), max(0, c-1-int(rng.integers(3)))
            e, d = min(30, e+1+int(rng.integers(3))), min(30, d+1+int(rng.integers(3)))
            return canvas[r:e, c:d].copy()
    return None


def _pair(rng, side, count, rotation, colors, *, ambiguous=False, partial=False, chain=False, bent=False):
    bg, frame, marker = colors[:3]
    sources = colors[3:3+count]
    extras = colors[3+count:]
    mapping = _mapping(rng, sources, extras, 'chain' if chain else
                       str(rng.choice(['chain', 'cycle', 'many', 'disjoint', 'mixed'])))
    for attempt in range(300):
        loop = _make_loop(rng, side, partial)
        if bent and _turns(loop) <= 4:
            continue
        if ambiguous or attempt > 160:
            labels = _paired_inserts(rng, side, count, loop)
        else:
            style = int(rng.integers(4))
            labels = (_cuts(rng, side, count) if style == 0 else
                      _paired_inserts(rng, side, count, loop) if style == 1 else
                      _grow(rng, side, list(range(count)), balanced=style == 2))
        if labels is None:
            continue
        anchor_label = int(labels[0, 0])
        order = [anchor_label] + [i for i in range(count) if i != anchor_label]
        pieces = []
        output = np.full((side, side), bg, dtype=np.int8)
        good = True
        for index, label in enumerate(order):
            cells = labels == label
            r, e, c, d = _box(np.argwhere(cells))
            foot = cells[r:e, c:d].copy()
            ink = (loop & cells)[r:e, c:d].copy()
            if int(foot.sum()) < 3 or int(ink.sum()) < 2 or not _connected(foot):
                good = False
                break
            anchor = np.zeros_like(foot)
            if index == 0:
                anchor[0, 0] = True
                if min(foot.shape) < 2:
                    good = False
                    break
            source = sources[index]
            output[loop & cells] = mapping[source]
            k = rotation if index == 0 else int(rng.integers(4))
            pieces.append(dict(source=source, foot=np.rot90(foot, k).copy(),
                               ink=np.rot90(ink, k).copy(), anchor=np.rot90(anchor, k).copy()))
        if not good:
            continue
        output[0, 0] = marker
        facts = dict(background=bg, frame=frame, marker=marker, side=side,
                     mapping=mapping, pieces=pieces)
        answers = _search(facts)
        if len(answers) != 1 or not np.array_equal(answers[0], output):
            continue
        if ambiguous and len(_search(facts, circuit=False, geometry_only=True)) < 2:
            continue
        inp = _scatter(rng, facts)
        if inp is not None:
            return dict(input=inp, output=output)
    raise RuntimeError('Bounded construction could not find a unique circuit jigsaw.')


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f'{name} must be an integer in {low}..{high}.')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test.')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer.')
    rng = np.random.default_rng(seed)
    mode = str(rng.choice(['scale', 'orientation', 'count']))
    stable_palette = bool(rng.random() < .2)
    palette = list(map(int, rng.permutation(10)))
    puzzle, seen = dict(train=[], test=[]), set()
    for split, n in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(n):
            train = split == 'train'
            if mode == 'scale':
                side = int(rng.integers(6 if train and index < 2 else 5, 8)) if train else int(rng.integers(8, 10))
            else:
                side = int(rng.integers(6 if train and index < 2 else 5, 10))
            if train and index == 0:
                side = 7 if mode == 'scale' else int(rng.integers(7, 10))
            if mode == 'count':
                count = (3 if index == 0 else 4 if index == 1 else int(rng.choice([3, 4]))) if train else 5
            else:
                count = int(rng.choice([3, 4] if train and index == 0 else [3, 4, 5]))
            if mode == 'orientation':
                rotation = ([1, 3][index] if train and index < 2 else int(rng.choice([0, 1, 3]))) if train else 2
            else:
                rotation = int(rng.choice([1, 2, 3])) if train and index == 0 else int(rng.integers(4))
            colors = palette if stable_palette else list(map(int, rng.permutation(10)))
            for _ in range(8):
                pair = _pair(rng, side, count, rotation, colors, ambiguous=index == 0,
                             partial=train and index == 1, chain=train and index == 1,
                             bent=train and index == 0)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('Could not create distinct inputs.')
    return puzzle


def _features(pair):
    f = _parse(pair['input'])
    out = _grid(pair['output'])
    mask = out != f['background']
    anchor = next(p for p in f['pieces'] if p['anchor'].any())
    r, c = np.argwhere(anchor['anchor'])[0]
    corner = ('T' if r == 0 else 'B') + ('L' if c == 0 else 'R')
    f.update(count=len(f['pieces']), turns=_turns(mask), corner=corner,
             partial=not mask[-1].any() or not mask[:, -1].any(),
             disconnected=sum(len(_components(p['ink'])) > 1 for p in f['pieces']),
             symmetric=sum(any(np.array_equal(p['foot'], np.rot90(p['foot'], k)) for k in (1, 2, 3))
                           for p in f['pieces']),
             overlap=any(t in f['mapping'] for t in f['mapping'].values()))
    return f


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Invalid puzzle keys.')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'Splits must be lists.')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, features = set(), dict(train=[], test=[])
    for split in ('train', 'test'):
        for index, pair in enumerate(puzzle[split]):
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Invalid pair keys.')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            f = _features(pair)
            _require(_key(inp) not in seen, 'Duplicate input.')
            seen.add(_key(inp))
            _require(out.shape == (f['side'], f['side']), 'Output size must follow silhouette area.')
            _require(int(out[0, 0]) == f['marker'] and int(np.sum(out == f['marker'])) == 1,
                     'Preserve the anchor at the upper-left.')
            _require(_cycle(out != f['background']), 'The painted answer must be one closed branchless loop.')
            _require(int(np.sum(out != f['background'])) == sum(int(p['ink'].sum()) for p in f['pieces']),
                     'Transfer every stencil pixel, including the anchor, once.')
            _require(np.any(out == f['background']), 'The carriers must be cleared outside their stencils.')
            _require(set(map(int, out.flat)) <= set(map(int, inp.flat)), 'Output uses visible colors only.')
            answers = _search(f)
            _require(len(answers) == 1 and np.array_equal(answers[0], out), 'Incorrect or ambiguous decorated packing.')
            for source, target in f['mapping'].items():
                amount = sum(int(p['ink'].sum())-int(p['anchor'].sum()) for p in f['pieces'] if p['source'] == source)
                expected = sum(int(p['ink'].sum())-int(p['anchor'].sum()) for p in f['pieces']
                               if f['mapping'][p['source']] == target)
                _require(amount >= 1 and int(np.sum(out == target)) == expected,
                         'Apply the original source-to-target map once, without recursive remapping.')
            if index == 0:
                shape_answers = _search(f, circuit=False, geometry_only=True)
                raw_answers = _search(f, circuit=False)
                _require(len(shape_answers) == 2 and len(raw_answers) == 2,
                         'The first pair of each split must require stencil-dependent assembly.')
                _require(any(not _cycle(a != f['background']) for a in raw_answers),
                         'Shape-only packing must admit an answer that breaks the loop.')
            features[split].append(f)
    train, test = features['train'], features['test']
    _require(train[0]['turns'] > 4, 'Demonstrate closure of a bent circuit, beyond rectangular outlines.')
    _require(train[1]['partial'], 'Demonstrate empty square margins rather than cropping to the circuit.')
    _require(train[1]['overlap'], 'Demonstrate nonrecursive substitution with a color chain.')
    _require(len(_search(train[0], rotations=False)) == 0, 'First training pair must demonstrate rigid rotation.')
    size_shift = min(f['side'] for f in test) > max(f['side'] for f in train)
    corner_shift = bool({f['corner'] for f in test} - {f['corner'] for f in train})
    count_shift = bool({f['count'] for f in test} - {f['count'] for f in train})
    _require(size_shift or corner_shift or count_shift, 'Require an observable structural train/test shift.')
    return None


def self_test():
    """Bounded 42-seed diversity audit plus semantic counterfactual mutations."""
    shapes, contours, signatures, records = set(), set(), set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = tuple((_key(p['input']), _key(p['output']))
                          for split in ('train', 'test') for p in puzzle[split])
        signatures.add(signature)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _features(pair)
                records.append(f)
                shapes.add(pair['input'].shape)
                contours.add((pair['output'].shape, (pair['output'] != f['background']).tobytes()))
        if seed < 3:
            _require(signature == tuple((_key(p['input']), _key(p['output']))
                                       for split in ('train', 'test') for p in generate_puzzle(seed)[split]),
                     'Seed reproducibility failed.')
    _require(len(signatures) == 42, 'Different seeds must produce different puzzles.')
    _require({f['side'] for f in records} == set(range(5, 10)), 'Exercise every supported square size.')
    _require({f['count'] for f in records} == {3, 4, 5}, 'Exercise all piece counts.')
    _require({f['corner'] for f in records} == {'TL', 'TR', 'BL', 'BR'}, 'Exercise all anchor orientations.')
    _require(len(shapes) >= 25 and len(contours) >= 50, 'Require geometry and contour variation.')
    _require(sum(f['disconnected'] for f in records) >= 40, 'Exercise disconnected registered stencil fragments.')
    _require(sum(f['symmetric'] for f in records) >= 40, 'Symmetric silhouettes must coexist with oriented ink.')
    _require(sum(f['turns'] > 4 for f in records) >= 20, 'Do not collapse the family to rectangular outlines.')
    _require(sum(f['partial'] for f in records) >= 42, 'Exercise foreground-independent square dimensions.')
    rng = np.random.default_rng(94181)
    pair = generate_puzzle(0)['train'][0]
    f = _parse(pair['input'])
    permutation = rng.permutation(10)
    _require(np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']]),
             'Semantic role recoloring failed.')
    # Change the scattered reference orientation of each silhouette AND its card.
    # This represents the same decorated rigid objects, so their answer is invariant.
    for p in f['pieces']:
        k = int(rng.integers(4))
        for name in ('foot', 'ink', 'anchor'):
            p[name] = np.rot90(p[name], k).copy()
    moved = _scatter(rng, f)
    _require(moved is not None and np.array_equal(solve(moved), pair['output']),
             'Card/key relocation and rigid object reorientation must preserve the answer.')
    # Change only the paint keys, including a many-to-one map. Geometry and loop stay.
    f = _parse(pair['input'])
    target = next(c for c in range(10) if c not in (f['background'], f['frame'], f['marker'])
                  and c not in f['mapping'])
    changed = pair['input'].copy()
    for source, (r, e, c, d) in f['key_boxes'].items():
        changed[r:e, c:d] = target
        changed[r, c] = source
    expected = pair['output'].copy()
    expected[(expected != f['background']) & (expected != f['marker'])] = target
    _require(np.array_equal(solve(changed), expected), 'Direct paint keys must change the answer independently of packing.')
    for a in range(2, 11):
        for b in range(1, 5):
            if a > b:
                _counts(a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('Illegal counts accepted.')
    bad = generate_puzzle(0, 2, 1)
    bad['test'][0]['output'][0, 0] = int(bad['test'][0]['input'][0, 0])
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Corrupted output accepted.')
    return dict(seeds=42, pairs=len(records), distinct_puzzles=len(signatures),
                input_shapes=len(shapes), circuit_shapes=len(contours),
                disconnected_stencils=sum(f['disconnected'] for f in records),
                symmetric_silhouettes=sum(f['symmetric'] for f in records),
                nonrectangular_circuits=sum(f['turns'] > 4 for f in records),
                empty_margin_examples=sum(f['partial'] for f in records))
