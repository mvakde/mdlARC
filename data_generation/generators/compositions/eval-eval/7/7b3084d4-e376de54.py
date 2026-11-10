"""Broken painted tablets: joint jigsaw recovery and substrate-aware calibration.

The solver reads only its grid. Generation constructs labels and paint directly;
it imports neither parent and uses no stored puzzle or solution cache.
"""
from __future__ import annotations

from math import isqrt
import numpy as np


EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS = EDGE + ((-1, -1), (-1, 1), (1, -1), (1, 1))
AXES = ((0, 1), (1, 0), (1, 1), (1, -1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("Expected a 1..30 by 1..30 integer grid with colors 0..9.")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _connected(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    if not cells:
        return False
    todo = [cells.pop()]
    while todo:
        r, c = todo.pop()
        for dr, dc in EDGE:
            p = (r + dr, c + dc)
            if p in cells:
                cells.remove(p)
                todo.append(p)
    return not cells


def _components(a, background):
    cells = set(map(tuple, np.argwhere(a != background)))
    pieces = []
    while cells:
        start = min(cells)
        cells.remove(start)
        todo, component = [start], [start]
        while todo:
            r, c = todo.pop()
            for dr, dc in NEIGHBORS:
                p = (r + dr, c + dc)
                if p in cells:
                    cells.remove(p)
                    todo.append(p)
                    component.append(p)
        rr, cc = zip(*component)
        p = np.full((max(rr) - min(rr) + 1, max(cc) - min(cc) + 1), -1,
                    dtype=np.int8)
        for r, c in component:
            p[r - min(rr), c - min(cc)] = a[r, c]
        pieces.append(p)
    return pieces


def _read(a):
    colors, counts = np.unique(a, return_counts=True)
    _require(np.count_nonzero(counts == counts.max()) == 1,
             "Background must be the unique most frequent color.")
    background = int(colors[np.argmax(counts)])
    singletons = [int(c) for c, n in zip(colors, counts) if n == 1 and c != background]
    _require(len(singletons) == 1, "One unique singleton anchor color is required.")
    marker = singletons[0]
    area = int(np.count_nonzero(a != background))
    side = isqrt(area)
    _require(side * side == area, "The tablet area must be a square.")
    pieces = _components(a, background)
    _require(3 <= len(pieces) <= 5, "Expected three to five separated pieces.")
    occurrences = {}
    for p in pieces:
        for c in np.unique(p[p >= 0]):
            occurrences[int(c)] = occurrences.get(int(c), 0) + 1
    ink = tuple(sorted(c for c, n in occurrences.items() if n >= 2))
    _require(1 <= len(ink) <= 2, "Paint colors must be shared across pieces.")
    carriers = []
    for p in pieces:
        local = set(map(int, p[p >= 0])) - set(ink) - {marker}
        _require(len(local) == 1, "Each piece must expose exactly one private carrier.")
        carrier = local.pop()
        _require(np.count_nonzero(p == carrier) >= 3,
                 "A carrier needs at least three exposed cells.")
        _require(np.count_nonzero(p == carrier) > np.count_nonzero(np.isin(p, ink)),
                 "Exposed substrate must outweigh paint on each piece.")
        carriers.append(carrier)
    _require(len(set(carriers)) == len(pieces), "Carrier colors must be distinct.")
    return dict(background=background, marker=marker, side=side, pieces=pieces,
                carriers=carriers, ink=ink)


def _strokes(painted, ink, marker):
    """Infer the full pre-calibration stroke array on a candidate assembly."""
    points = [(int(r), int(c), int(painted[r, c]))
              for r, c in np.argwhere(np.isin(painted, ink))]
    interpretations = []
    for dr, dc in AXES:
        groups = {}
        for r, c, color in points:
            line, axial = dc * r - dr * c, dr * r + dc * c
            groups.setdefault(line, []).append((axial, r, c, color))
        if len(groups) not in (3, 5, 7):
            continue
        keys = sorted(groups)
        step = dr * dr + dc * dc
        gaps = [b - a for a, b in zip(keys, keys[1:])]
        if min(gaps) < 2 * step or any(g % step for g in gaps):
            continue
        ordered = [sorted(groups[k]) for k in keys]
        if any(len({x[3] for x in g}) != 1 or
               any(b[0] - a[0] != step for a, b in zip(g, g[1:]))
               for g in ordered):
            continue
        lo = {g[0][0] for g in ordered}
        hi = {g[-1][0] for g in ordered}
        if (len(lo) == 1) == (len(hi) == 1):
            continue
        sign = 1 if len(lo) == 1 else -1
        oriented = [g if sign == 1 else g[::-1] for g in ordered]
        reference = len(oriented[len(oriented) // 2])
        if reference < 2:
            continue
        targets, valid = [], True
        for g in oriented:
            _, r, c, color = g[0]
            cells = [(r + sign * dr * t, c + sign * dc * t, color)
                     for t in range(reference)]
            if any(not (0 <= rr < painted.shape[0] and 0 <= cc < painted.shape[1])
                   or painted[rr, cc] == marker for rr, cc, _ in cells):
                valid = False
                break
            targets.extend(cells)
        if valid:
            interpretations.append(dict(groups=oriented, keys=keys, gaps=gaps,
                                        direction=(sign * dr, sign * dc),
                                        lengths=tuple(map(len, oriented)),
                                        reference=reference, targets=targets))
    return interpretations[0] if len(interpretations) == 1 else None


def _orientations(piece):
    seen = set()
    for k in range(4):
        p = np.rot90(piece, k).copy()
        key = _key(p)
        if key not in seen:
            seen.add(key)
            yield p


def _assemble(facts, use_paint=True, limit=2):
    """Enumerate rigid exact covers; paint grammar filters the complete covers.

    Unlike monochrome jigsaws, equal occupied masks with differently oriented
    paint must remain separate candidates. The uniqueness key includes both
    the observed paint and the carrier underneath it.
    """
    side, pieces, marker = facts['side'], facts['pieces'], facts['marker']
    anchor = next(i for i, p in enumerate(pieces) if np.any(p == marker))
    options = []
    for i, piece in enumerate(pieces):
        placements = []
        for p in _orientations(piece):
            h, w = p.shape
            if h > side or w > side:
                continue
            coords = np.argwhere(p >= 0)
            local = tuple((int(r) * side + int(c), int(p[r, c])) for r, c in coords)
            base = sum(1 << pos for pos, _ in local)
            if i == anchor:
                offsets = [0] if p[0, 0] == marker else []
            else:
                offsets = [r * side + c for r in range(side - h + 1)
                           for c in range(side - w + 1)]
            placements.extend((base << off, off, local) for off in offsets)
        options.append(placements)
    full = (1 << (side * side)) - 1
    placed = [None] * len(pieces)
    answers, seen = [], set()

    def search(occupied, remaining):
        if len(answers) >= limit:
            return
        if not remaining:
            if occupied != full:
                return
            painted = np.empty((side, side), dtype=np.int8)
            owners = np.empty_like(painted)
            for i, (_, offset, cells) in enumerate(placed):
                for pos, color in cells:
                    painted.flat[pos + offset] = color
                    owners.flat[pos + offset] = facts['carriers'][i]
            stroke = _strokes(painted, facts['ink'], marker) if use_paint else None
            if use_paint and stroke is None:
                return
            signature = (_key(painted), _key(owners)) if use_paint else _key(owners)
            if signature not in seen:
                seen.add(signature)
                answers.append((painted, owners, stroke))
            return
        first = (full ^ occupied) & -(full ^ occupied)
        for i in remaining:
            rest = tuple(j for j in remaining if j != i)
            for p in options[i]:
                if p[0] & first and not p[0] & occupied:
                    placed[i] = p
                    search(occupied | p[0], rest)
                    if len(answers) >= limit:
                        return
            placed[i] = None

    for p in options[anchor]:
        placed[anchor] = p
        search(p[0], tuple(i for i in range(len(pieces)) if i != anchor))
        if len(answers) >= limit:
            break
    return answers


def _normalize(painted, owners, stroke, marker):
    output = owners.copy()
    output[0, 0] = marker
    for r, c, color in stroke['targets']:
        output[r, c] = color
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the unique painted tablet, then calibrate on its substrate."""
    a = _grid(input_grid)
    facts = _read(a)
    answers = _assemble(facts)
    _require(len(answers) == 1, "Expected one assembly satisfying geometry AND stroke grammar.")
    painted, owners, stroke = answers[0]
    return _normalize(painted, owners, stroke, facts['marker'])


def _partition(rng, side, count, style):
    if style == 'growth':
        labels = np.full((side, side), -1, dtype=np.int8)
        seeds = list(map(int, rng.choice(side * side, count, replace=False)))
        frontiers = [set() for _ in range(count)]
        sizes = [1] * count
        weights = np.exp(rng.uniform(-.8, .8, count))
        for i, pos in enumerate(seeds):
            labels.flat[pos] = i
        def neighbors(pos):
            r, c = divmod(pos, side)
            return [rr * side + cc for dr, dc in EDGE
                    if 0 <= (rr := r + dr) < side and 0 <= (cc := c + dc) < side]
        for i, pos in enumerate(seeds):
            frontiers[i].update(x for x in neighbors(pos) if labels.flat[x] < 0)
        for _ in range(side * side - count):
            eligible = [i for i in range(count) if frontiers[i]]
            i = min(eligible, key=lambda j: sizes[j] / weights[j] + float(rng.random()))
            cells = sorted(frontiers[i])
            pos = cells[int(rng.integers(len(cells)))]
            labels.flat[pos] = i
            sizes[i] += 1
            for f in frontiers:
                f.discard(pos)
            frontiers[i].update(x for x in neighbors(pos) if labels.flat[x] < 0)
        return labels
    rectangles = [(0, 0, side, side)]
    while len(rectangles) < count:
        candidates = []
        for i, (r, c, h, w) in enumerate(rectangles):
            candidates.extend((i, 0, k) for k in range(2, h - 1) if min(k, h-k) * w >= 9)
            candidates.extend((i, 1, k) for k in range(2, w - 1) if min(k, w-k) * h >= 9)
        if not candidates:
            return None
        i, axis, k = candidates[int(rng.integers(len(candidates)))]
        r, c, h, w = rectangles.pop(i)
        rectangles.extend([(r, c, k, w), (r+k, c, h-k, w)] if axis == 0 else
                          [(r, c, h, k), (r, c+k, h, w-k)])
    labels = np.empty((side, side), dtype=np.int8)
    for i, (r, c, h, w) in enumerate(rectangles):
        labels[r:r+h, c:c+w] = i
    # Irregular interfaces with variable notch depths, retaining rigid pieces.
    for _ in range(side * 4):
        r, c = map(int, rng.integers(side, size=2))
        dr, dc = EDGE[int(rng.integers(4))]
        rr, cc = r + dr, c + dc
        if not (0 <= rr < side and 0 <= cc < side) or (r, c) == (0, 0):
            continue
        a, b = int(labels[r, c]), int(labels[rr, cc])
        if a == b or np.count_nonzero(labels == a) <= 9:
            continue
        labels[r, c] = b
        if not _connected(labels == a):
            labels[r, c] = a
    return labels


def _paint_layout(rng, side, count, direction, mode, avoid_reference=None, witness=False):
    dr, dc = direction
    diagonal = bool(dr and dc)
    if witness:
        rows = [0, 2, 4, 6, 11] if side == 12 else [1, 3, 5, 7, 12]
        width = side // 3
        reference = int(rng.integers(width + 1, 2 * width + 1))
        lengths = [int(rng.integers(1, reference)) for _ in range(5)]
        lengths[2] = reference
        lengths[int(rng.choice([0, 1, 3, 4]))] = int(rng.integers(2*width + 1, side))
        anchors = [(r, 1) for r in rows]
        return anchors, lengths, reference
    gaps = [2] * (count - 1)
    budget = side - (5 if diagonal else 2) - sum(gaps)
    if budget < 0:
        return None
    for _ in range(int(rng.integers(budget + 1))):
        gaps[int(rng.integers(count - 1))] += 1
    span = sum(gaps)
    capacity = side - span if diagonal else side - 1
    choices = [x for x in range(2, capacity) if x != avoid_reference]
    if not choices:
        return None
    reference = int(rng.choice(choices))
    lengths = [reference] * count
    other = [int(x) for x in rng.permutation(count) if x != count // 2]
    def short():
        return int(rng.integers(1, reference))
    def long():
        return int(rng.integers(reference + 1, capacity + 1))
    if mode == 'mixed':
        lengths[other[0]], lengths[other[1]] = short(), long()
        for i in other[2:]:
            lengths[i] = reference if rng.random() < .3 else short() if rng.random() < .5 else long()
    elif mode in ('extend', 'trim'):
        for i in other:
            lengths[i] = reference if rng.random() < .3 else short() if mode == 'extend' else long()
        lengths[other[0]] = short() if mode == 'extend' else long()
    else:
        lengths[other[0]] = short() if mode == 'one_extend' else long()
    perpendicular = (dc, -dr)
    positions = [0] + list(map(int, np.cumsum(gaps)))
    anchors = [(p * perpendicular[0], p * perpendicular[1]) for p in positions]
    cells = [(r + t * dr, c + t * dc) for (r, c), length in zip(anchors, lengths)
             for t in range(max(length, reference))]
    low = [min(p[k] for p in cells) for k in (0, 1)]
    extent = [max(p[k] for p in cells) - low[k] + 1 for k in (0, 1)]
    if max(extent) > side:
        return None
    offsets = [int(rng.integers(side - extent[k] + 1)) - low[k] for k in (0, 1)]
    anchors = [(r + offsets[0], c + offsets[1]) for r, c in anchors]
    if any((r + t * dr, c + t * dc) == (0, 0)
           for (r, c), length in zip(anchors, lengths) for t in range(max(length, reference))):
        return None
    return anchors, lengths, reference


def _extract(labels, painted, count):
    pieces = []
    for i in range(count):
        points = np.argwhere(labels == i)
        r0, c0 = points.min(axis=0)
        r1, c1 = points.max(axis=0) + 1
        pieces.append(np.where(labels[r0:r1, c0:c1] == i,
                               painted[r0:r1, c0:c1], -1).astype(np.int8))
    return pieces


def _scatter(rng, pieces, background):
    rotations = list(map(int, rng.integers(4, size=len(pieces))))
    # A changed asymmetric colored object is always visible in the input.
    eligible = [i for i, p in enumerate(pieces) if len(list(_orientations(p))) == 4]
    if eligible:
        rotations[eligible[0]] = int(rng.integers(1, 4))
    oriented = [np.rot90(p, k).copy() for p, k in zip(pieces, rotations)]
    options = []
    full = (1 << 900) - 1
    first_col = sum(1 << (r * 30) for r in range(30))
    last_col = first_col << 29
    for p in oriented:
        h, w = p.shape
        base = sum(1 << (int(r) * 30 + int(c)) for r, c in np.argwhere(p >= 0))
        placements = []
        for r in range(31 - h):
            for c in range(31 - w):
                mask = base << (r * 30 + c)
                horizontal = mask | ((mask & ~first_col) >> 1) | ((mask & ~last_col) << 1)
                dilation = (horizontal | horizontal << 30 | horizontal >> 30) & full
                placements.append((mask, dilation, r, c))
        options.append(placements)
    for attempt in range(10):
        order = sorted(range(len(pieces)), key=lambda i: -oriented[i].size)
        if attempt > 0:
            order = list(map(int, rng.permutation(len(pieces))))
        blocked, placed, maxr, maxc = 0, {}, 0, 0
        for i in order:
            h, w = oriented[i].shape
            candidates = [p for p in options[i] if not p[0] & blocked]
            if not candidates:
                break
            scores = [max(maxr, p[2]+h) * max(maxc, p[3]+w)
                      + float(rng.uniform(0, 15 + 10*attempt)) for p in candidates]
            p = candidates[int(np.argmin(scores))]
            placed[i] = p
            blocked |= p[1]
            maxr, maxc = max(maxr, p[2]+h), max(maxc, p[3]+w)
        if len(placed) != len(pieces):
            continue
        canvas = np.full((30, 30), background, dtype=np.int8)
        for i, (_, _, r, c) in placed.items():
            p = oriented[i]
            view = canvas[r:r+p.shape[0], c:c+p.shape[1]]
            view[p >= 0] = p[p >= 0]
        rr, cc = np.nonzero(canvas != background)
        cropped = canvas[rr.min():rr.max()+1, cc.min():cc.max()+1]
        h, w = cropped.shape
        H, W = h + int(rng.integers(min(5, 30-h)+1)), w + int(rng.integers(min(5, 30-w)+1))
        fg = int(np.count_nonzero(cropped != background))
        largest = max(int(np.count_nonzero(cropped == c)) for c in np.unique(cropped) if c != background)
        while H * W - fg <= largest:
            if H < 30 and (H <= W or W == 30):
                H += 1
            elif W < 30:
                W += 1
            else:
                break
        if H * W - fg <= largest:
            continue
        out = np.full((H, W), background, dtype=np.int8)
        r, c = int(rng.integers(H-h+1)), int(rng.integers(W-w+1))
        out[r:r+h, c:c+w] = cropped
        return out
    return None


def _seam_edits(owners, stroke):
    count = 0
    dr, dc = stroke['direction']
    reference = stroke['reference']
    for g in stroke['groups']:
        _, r, c, _ = g[0]
        changed = range(min(len(g), reference), max(len(g), reference))
        count += any(owners[r+t*dr, c+t*dc] != owners[r, c] for t in changed)
    return count


def _make_pair(rng, descriptor, palette):
    side, count = descriptor['side'], descriptor['pieces']
    bg, marker = palette[:2]
    carriers = palette[2:2+count]
    inks = palette[2+count:2+count+descriptor['inks']]
    for attempt in range(320):
        if descriptor['witness']:
            labels = np.tile(np.arange(side) // (side // 3), (side, 1)).astype(np.int8)
        else:
            style = descriptor['style'] if attempt < 200 else 'cuts'
            labels = _partition(rng, side, count, style)
        if labels is None or any(np.count_nonzero(labels == i) < 9 for i in range(count)):
            continue
        layout = _paint_layout(rng, side, descriptor['strokes'], descriptor['direction'],
                               descriptor['mode'], descriptor.get('avoid'), descriptor['witness'])
        if layout is None:
            continue
        anchors, lengths, reference = layout
        owners = np.asarray(carriers, dtype=np.int8)[labels]
        painted = owners.copy()
        colors = [int(inks[i % len(inks)]) for i in range(len(anchors))]
        if len(inks) > 1:
            rng.shuffle(colors)
        dr, dc = descriptor['direction']
        for (r, c), length, color in zip(anchors, lengths, colors):
            for t in range(length):
                painted[r+t*dr, c+t*dc] = color
        painted[0, 0] = marker
        stroke = _strokes(painted, inks, marker)
        if stroke is None:
            continue
        ref_group = stroke['groups'][len(anchors)//2]
        if len({int(owners[r, c]) for _, r, c, _ in ref_group}) < 2:
            continue
        if _seam_edits(owners, stroke) == 0:
            continue
        pieces = _extract(labels, painted, count)
        if any(np.count_nonzero(p == carrier) < 3 or
               np.count_nonzero(p == carrier) <= np.count_nonzero(np.isin(p, inks))
               for p, carrier in zip(pieces, carriers)):
            continue
        if any(sum(np.any(p == ink) for p in pieces) < 2 for ink in inks):
            continue
        facts = dict(side=side, pieces=pieces, marker=marker, carriers=carriers, ink=tuple(inks))
        answers = _assemble(facts)
        if len(answers) != 1 or not np.array_equal(answers[0][0], painted):
            continue
        inp = _scatter(rng, pieces, bg)
        if inp is None:
            continue
        return dict(input=inp, output=_normalize(painted, owners, stroke, marker))
    raise RuntimeError(f"Could not generate descriptor {descriptor!r} within the bounded search.")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer.")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}.")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if ntrain <= ntest:
        raise ValueError("num_train must exceed num_test.")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    stable = bool(rng.integers(2))
    base_palette = list(map(int, rng.permutation(10)))
    first_reference = None
    # A randomly held-out directed axis supplies a visible geometry shift.
    axes = [(dr*s, dc*s) for dr, dc in AXES for s in (1, -1)]
    rng.shuffle(axes)
    held = next(d for d in axes if d != (0, 1))
    train_axes = [d for d in axes if d != held]
    for split, total in (('train', ntrain), ('test', ntest)):
        for i in range(total):
            witness = split == 'train' and i == 0
            if witness:
                descriptor = dict(side=int(rng.choice([12, 15])), pieces=3, strokes=5,
                                  direction=(0, 1), mode='mixed', inks=1, style='strips', witness=True)
            else:
                direction = (held if split == 'test' else
                             (-held[0], -held[1]) if i == 1 else train_axes[(i - 1) % 7])
                strokes = 3 if split == 'train' and i == 1 else int(rng.choice([3, 5, 7]))
                if all(direction) and strokes == 7:
                    strokes = 5  # Seven diagonals leave too little length evidence here.
                minimum = (14 if strokes == 7 else 13 if all(direction) and strokes == 5
                           else 10 if strokes == 5 else 9)
                side = int(rng.integers(minimum, 17))
                descriptor = dict(side=side, pieces=int(rng.choice([3, 4, 5])), strokes=strokes,
                                  direction=direction, mode='mixed' if i == 1 and split == 'train' else
                                  str(rng.choice(['mixed', 'extend', 'trim', 'one_extend', 'one_trim'])),
                                  inks=2 if split == 'train' and i == 1 else int(rng.choice([1, 2], p=[.7, .3])),
                                  style=str(rng.choice(['growth', 'cuts'])), witness=False,
                                  avoid=first_reference if split == 'train' and i == 1 else None)
            palette = base_palette if stable else list(map(int, rng.permutation(10)))
            for _ in range(8):
                pair = _make_pair(rng, descriptor, palette)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("Distinct input sampling exhausted.")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
            if witness:
                facts = _read(pair['input'])
                first_reference = _assemble(facts)[0][2]['reference']
    return puzzle


def _features(inp):
    facts = _read(inp)
    answers = _assemble(facts)
    _require(len(answers) == 1, "Painted assembly must be unique.")
    painted, owners, stroke = answers[0]
    reference = stroke['reference']
    lengths = stroke['lengths']
    distances = [abs(2*k - stroke['keys'][0] - stroke['keys'][-1]) for k in stroke['keys']]
    nearest = [i for i, d in enumerate(distances) if d == min(distances)]
    marked = next(p for p in facts['pieces'] if np.any(p == facts['marker']))
    marker_pos = tuple(map(int, np.argwhere(marked == facts['marker'])[0]))
    geometry_ambiguous = len(_assemble(facts, use_paint=False)) > 1
    carrier_shapes = []
    rotated = False
    for p, carrier in zip(facts['pieces'], facts['carriers']):
        mask = owners == carrier
        rr, cc = np.nonzero(mask)
        tight = painted[rr.min():rr.max()+1, cc.min():cc.max()+1]
        tight = np.where(mask[rr.min():rr.max()+1, cc.min():cc.max()+1], tight, -1)
        rotated |= not np.array_equal(tight, p)
        carrier_shapes.append((p.shape, int(np.count_nonzero(p >= 0))))
    return dict(facts=facts, painted=painted, owners=owners, stroke=stroke,
                side=facts['side'], pieces=len(facts['pieces']), inks=len(facts['ink']),
                count=len(lengths), reference=reference, direction=stroke['direction'],
                lengths=lengths, gaps=tuple(stroke['gaps']), marker_pos=marker_pos,
                geometry_ambiguous=geometry_ambiguous, rotation_required=rotated,
                nonmedian=sorted(lengths)[len(lengths)//2] != reference,
                geometric_witness=len(nearest) == 1 and nearest[0] != len(lengths)//2
                and lengths[nearest[0]] != reference,
                seam_edits=_seam_edits(owners, stroke),
                extend=any(x < reference for x in lengths), trim=any(x > reference for x in lengths),
                shape_signature=tuple(carrier_shapes))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "Invalid puzzle schema.")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists.")
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "Illegal split counts.")
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "Invalid pair schema.")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, "Duplicate inputs.")
            seen.add(_key(inp))
            f = _features(inp)
            facts, painted, owners, stroke = (f[k] for k in ('facts', 'painted', 'owners', 'stroke'))
            expected = _normalize(painted, owners, stroke, facts['marker'])
            _require(np.array_equal(out, expected), "Incorrect substrate-aware normalization.")
            _require(out.shape == (f['side'], f['side']), "Output size must follow total piece area.")
            _require(out[0, 0] == facts['marker'] and np.count_nonzero(out == facts['marker']) == 1,
                     "Preserve the upper-left anchor.")
            _require(not np.any(out == facts['background']), "No scatter background belongs in the tablet.")
            _require(f['seam_edits'] > 0, "Calibration must cross a piece seam.")
            ref = stroke['groups'][f['count']//2]
            _require(len({int(owners[r, c]) for _, r, c, _ in ref}) >= 2,
                     "The reference must be fragmented across pieces.")
            # Restored cells use their own substrate, not a single erase color.
            dr, dc = stroke['direction']
            for g in stroke['groups']:
                _, r, c, color = g[0]
                for t in range(min(len(g), f['reference'])):
                    _require(out[r+t*dr, c+t*dc] == color, "Keep each anchored prefix.")
                for t in range(f['reference'], len(g)):
                    _require(out[r+t*dr, c+t*dc] == owners[r+t*dr, c+t*dc], "Restore local substrate.")
            _require(np.array_equal(solve(inp), out), "Input-only solver disagrees.")
            features[split].append(f)
    train = features['train']
    _require(any(f['geometry_ambiguous'] for f in train), "Teach paint-constrained assembly.")
    _require(any(f['nonmedian'] and f['geometric_witness'] for f in train), "Teach ordinal centrality.")
    _require(any(f['extend'] and f['trim'] for f in train), "Teach both correction operations.")
    _require(any(f['rotation_required'] for f in train), "Teach rigid quarter-turn recovery.")
    _require(len({f['reference'] for f in train}) > 1, "Demonstrate variable reference lengths.")
    _require(len({f['count'] for f in train}) > 1, "Demonstrate variable stroke counts.")
    _require({f['inks'] for f in train} == {1, 2}, "Teach both monochrome and multicolor paint arrays.")
    return None


def self_test():
    """Bounded 42-seed campaign, invariants, metamorphic and negative checks."""
    puzzles, all_features, shapes, normalized_outputs = set(), [], set(), set()
    rng = np.random.default_rng(91280)
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signature = tuple(_key(pair[k]) for s in ('train', 'test') for pair in p[s]
                          for k in ('input', 'output'))
        puzzles.add(signature)
        for s in ('train', 'test'):
            for pair in p[s]:
                f = _features(pair['input'])
                all_features.append((s, f))
                shapes.add(pair['input'].shape)
                out = pair['output']
                roles = {}
                normalized_outputs.add((out.shape, tuple(roles.setdefault(int(c), len(roles)) for c in out.flat)))
        if seed < 3:
            q = generate_puzzle(seed)
            _require(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s])
                         for k in ('input', 'output')), "Seed determinism failed.")
            pair = p['test'][0]
            permutation = rng.permutation(10)
            _require(np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']]),
                     "Palette roles must commute with solving.")
            for k in range(4):
                _require(np.array_equal(solve(np.rot90(pair['input'], k)), pair['output']),
                         "Rotating the scatter must preserve the anchored answer.")
            f = _features(pair['input'])
            # Omitting B leaves the uncalibrated painted assembly.
            _require(not np.array_equal(f['painted'], pair['output']), "B must change the answer.")
    _require(len(puzzles) == 42, "Seeds must yield distinct puzzles.")
    _require(len(shapes) >= 25 and len(normalized_outputs) >= 150, "Need geometric, not just palette variation.")
    for field, minimum in (('side', 7), ('reference', 8), ('direction', 8), ('shape_signature', 100)):
        _require(len({f[field] for _, f in all_features}) >= minimum, f"Insufficient {field} variation.")
    _require({f['pieces'] for _, f in all_features} == {3, 4, 5}, "Exercise all piece counts.")
    _require({f['count'] for _, f in all_features} == {3, 5, 7}, "Exercise all stroke counts.")
    _require({f['inks'] for _, f in all_features} == {1, 2}, "Exercise shared paint palettes.")
    for split in ('train', 'test'):
        fs = [f for s, f in all_features if s == split]
        _require(any(f['extend'] and not f['trim'] for f in fs), "Exercise extension-only arrays.")
        _require(any(f['trim'] and not f['extend'] for f in fs), "Exercise truncation-only arrays.")
        _require(any(len(set(f['gaps'])) > 1 for f in fs), "Exercise unequal ordinal spacing.")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(1091, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted.")
    bad = generate_puzzle(77, 2, 1)
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("A corrupted output passed validation.")


if __name__ == '__main__':
    self_test()
    print('self_test passed')
