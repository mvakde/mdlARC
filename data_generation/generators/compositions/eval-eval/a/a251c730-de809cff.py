"""Material engraving: latent material keys, reference stencils, and XOR strokes.

Only the input is needed to solve.  The sample cards encode both stencils and
the highlight color; the damaged large rectangles encode the engraving sites.
"""

from itertools import permutations
import numpy as np


_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _pick(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    answer = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, cells = [p], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in _D4:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        answer.append(cells)
    return answer


def _bounds(cells):
    rows, cols = zip(*cells)
    return min(rows), min(cols), max(rows) + 1, max(cols) + 1


def _slice(box):
    r0, c0, r1, c1 = box[:4]
    return np.s_[r0:r1, c0:c1]


def _find_cards(grid):
    cards = []
    for color in map(int, np.unique(grid)):
        for cells in _components(grid == color):
            r0, c0, r1, c1 = _bounds(cells)
            h, w = r1-r0, c1-c0
            if min(h, w) < 5 or len(cells) != 2*h + 2*w - 4:
                continue
            if not all(r in (r0, r1-1) or c in (c0, c1-1) for r, c in cells):
                continue
            inner = grid[r0+1:r1-1, c0+1:c1-1]
            values, counts = np.unique(inner, return_counts=True)
            key = int(values[np.argmax(counts)])
            if key == color or counts.max() <= inner.size / 2:
                continue
            if not all(np.all(edge == key) for edge in
                       (inner[0], inner[-1], inner[:, 0], inner[:, -1])):
                continue
            cards.append((r0, c0, r1, c1, color, key))
    assert len(cards) == 2, 'Exactly two intact framed material samples are required'
    assert cards[0][4] == cards[1][4] and cards[0][5] != cards[1][5]
    highlight = cards[0][4]
    fills = tuple(sorted(card[5] for card in cards))
    mask = np.zeros(grid.shape, dtype=bool)
    templates, repeats = {}, {}
    for r0, c0, r1, c1, _, key in cards:
        assert not np.any(mask[r0:r1, c0:c1]), 'Samples must be disjoint'
        mask[r0:r1, c0:c1] = True
        inner = grid[r0+1:r1-1, c0+1:c1-1]
        other = next(f for f in fills if f != key)
        assert set(map(int, np.unique(inner))) == {key, other, highlight}
        signatures = []
        for cells in _components(inner != key):
            anchors = [p for p in cells if inner[p] == highlight]
            assert len(anchors) == 1, 'Every sample glyph has one highlight center'
            ar, ac = anchors[0]
            assert len(cells) >= 4
            assert all(inner[p] == other for p in cells if p != (ar, ac))
            signatures.append(tuple(sorted((r-ar, c-ac) for r, c in cells)))
        assert signatures and len(set(signatures)) == 1, 'Conflicting sample glyphs'
        templates[key] = signatures[0]
        repeats[key] = len(signatures)
    remaining = set(map(int, np.unique(grid))) - {highlight, *fills}
    assert len(remaining) == 1, 'One visibly distinct background is required'
    background = remaining.pop()
    assert not np.any((grid == highlight) & ~mask)
    return cards, mask, templates, repeats, fills, background, highlight


def _support(grid, color):
    same = grid == color
    count = np.zeros(grid.shape, dtype=np.int8)
    count[1:] += same[:-1]
    count[:-1] += same[1:]
    count[:, 1:] += same[:, :-1]
    count[:, :-1] += same[:, 1:]
    return same & (count >= 2)


def _overlap(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))


def _paint(shape, background, boxes):
    grid = np.full(shape, background, dtype=np.int8)
    for box in boxes:
        grid[_slice(box)] = box[4]
    return grid


def _chain_ok(boxes):
    if len(boxes) not in (2, 3):
        return False
    degrees = [0] * len(boxes)
    for i, a in enumerate(boxes):
        if min(a[2]-a[0], a[3]-a[1]) < 5:
            return False
        for j, b in enumerate(boxes[:i]):
            overlap = _overlap(a, b)
            if overlap:
                if overlap < 4 or a[4] == b[4]:
                    return False
                degrees[i] += 1
                degrees[j] += 1
    return sorted(degrees) == [1, 1] + [2] * (len(boxes)-2)


def _supported_boxes(grid, fills):
    boxes = []
    for color in fills:
        for cells in _components(_support(grid, color)):
            if len(cells) >= 18:
                boxes.append((*_bounds(cells), color))
    return tuple(sorted(boxes))


def _recover_scene(grid, fills, background, card_mask):
    """Use intact material support for bounds; compare every covering order.

    The family retains each visible rectangle's full bounding box in its dense
    support component. Isolated errors have no separate large support component.
    This is an evidence restriction, not a general arbitrary-image denoiser.
    """
    boxes = _supported_boxes(grid, fills)
    assert _chain_ok(boxes), 'Dense surviving evidence must identify a rectangle chain'
    scenes = {}
    for order in permutations(boxes):
        clean = _paint(grid.shape, background, order)
        if np.any((clean != background) & card_mask):
            continue
        valid = True
        for box in boxes:
            visible = clean[_slice(box)] == box[4]
            if visible.sum() < 24 or _bounds(list(map(tuple, np.argwhere(visible)))) != (0, 0, *visible.shape):
                valid = False
                break
            observed = grid[_slice(box)] == box[4]
            if np.count_nonzero(visible & observed) < .82 * visible.sum():
                valid = False
                break
        if valid:
            scenes[clean.tobytes()] = (int(np.count_nonzero(grid != clean)), clean, order)
    ranked = sorted(scenes.values(), key=lambda x: (x[0], x[1].tobytes()))
    assert ranked, 'No supported visible scene'
    loss, clean, order = ranked[0]
    assert loss <= max(12, int(.10 * grid.size)), 'Too much damage'
    assert len(ranked) == 1 or ranked[1][0] >= loss + 3, 'Ambiguous covering order'
    assert _supported_boxes(clean, fills) == boxes, 'Insufficient surviving bounds'
    return clean, tuple(order)


def _footprint(point, template, shape):
    r, c = point
    return {(r+dr, c+dc) for dr, dc in template
            if 0 <= r+dr < shape[0] and 0 <= c+dc < shape[1]}


def _compose(clean, centers, templates, fills, highlight, *, parity=True,
             mask_background=True, anchor_color=False):
    """Non-center strokes invert the original destination material, simultaneously."""
    centers = tuple(centers)
    assert len(set(centers)) == len(centers)
    counts = np.zeros(clean.shape, dtype=np.int16)
    fixed = {}
    for p in centers:
        key = int(clean[p])
        for q in _footprint(p, templates[key], clean.shape) - {p}:
            counts[q] += 1
            fixed[q] = next(f for f in fills if f != key)
    out = clean.copy()
    active = counts % 2 == 1 if parity else counts > 0
    if mask_background:
        active &= np.isin(clean, fills)
    if anchor_color:
        for q in map(tuple, np.argwhere(active)):
            out[q] = fixed[q]
    else:
        out[active & (clean == fills[0])] = fills[1]
        out[active & (clean == fills[1])] = fills[0]
        if not mask_background:
            for q in map(tuple, np.argwhere(active & ~np.isin(clean, fills))):
                out[q] = fixed[q]
    for p in centers:
        out[p] = highlight
    return out, counts


def _analyze(input_grid):
    grid = _grid(input_grid)
    cards, card_mask, templates, repeats, fills, background, highlight = _find_cards(grid)
    scene_input = grid.copy()
    scene_input[card_mask] = background
    clean, boxes = _recover_scene(scene_input, fills, background, card_mask)
    centers = [tuple(map(int, p)) for p in np.argwhere((scene_input == background) & np.isin(clean, fills))]
    assert centers, 'At least one genuine erasure is required'
    output, counts = _compose(clean, centers, templates, fills, highlight)
    output[card_mask] = grid[card_mask]
    return dict(grid=grid, cards=cards, card_mask=card_mask, templates=templates,
                repeats=repeats, fills=fills, background=background, highlight=highlight,
                scene_input=scene_input, clean=clean, boxes=boxes, centers=centers,
                output=output, counts=counts)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return a new output array without a seed, demonstrations, or hidden state."""
    return _analyze(input_grid)['output']


def _new_template(rng, small):
    for _ in range(40):
        h, w = (int(rng.integers(2, 5 if small else 6)) for _ in range(2))
        n = int(rng.integers(4, min(11, h*w)+1))
        cells = {(int(rng.integers(h)), int(rng.integers(w)))}
        compact = bool(rng.integers(2))
        while len(cells) < n:
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _D4
                               if 0 <= r+dr < h and 0 <= c+dc < w} - cells)
            weights = np.array([sum((r+dr, c+dc) in cells for dr, dc in _D4)
                                for r, c in frontier], dtype=float)
            weights = weights**2 if compact else 1 / weights
            cells.add(frontier[int(rng.choice(len(frontier), p=weights/weights.sum()))])
        ar, ac = _pick(rng, sorted(cells))
        shape = tuple(sorted((r-ar, c-ac) for r, c in cells))
        r0, c0, r1, c1 = _bounds(shape)
        if (r1-r0)*(c1-c0) > n and set(shape) != {(-c, r) for r, c in shape}:
            return shape
    return ((-1, 0), (0, 0), (0, 1), (1, 0), (1, 1))


def _layout(rng, templates, fills, background, highlight, repetitions):
    """Build two small swatches in a band, leaving a free large scene area."""
    dimensions = []
    for key in fills:
        r0, c0, r1, c1 = _bounds(templates[key])
        bh, bw = r1-r0, c1-c0
        dimensions.append((bh+4, repetitions*bw + repetitions+3))
    gap = int(rng.integers(1, 3))
    required_w = sum(w for h, w in dimensions) + gap
    assert required_w <= 30
    width = int(rng.integers(max(22, required_w), 31))
    band_h = max(h for h, w in dimensions) + int(rng.integers(0, 2))
    scene_gap = int(rng.integers(1, 3))
    scene_h = int(rng.integers(14, 31-band_h-scene_gap))
    height = band_h + scene_gap + scene_h
    grid = np.full((height, width), background, dtype=np.int8)
    card_mask = np.zeros(grid.shape, dtype=bool)
    order = list(rng.permutation(2))
    left = int(rng.integers(0, width-required_w+1))
    for i in order:
        key = fills[i]
        h, w = dimensions[i]
        top = int(rng.integers(0, band_h-h+1))
        view = grid[top:top+h, left:left+w]
        view[:] = highlight
        view[1:-1, 1:-1] = key
        card_mask[top:top+h, left:left+w] = True
        r0, c0, r1, c1 = _bounds(templates[key])
        for j in range(repetitions):
            ar, ac = 2-r0, 2-c0+j*(c1-c0+1)
            for dr, dc in templates[key]:
                view[ar+dr, ac+dc] = highlight if (dr, dc) == (0, 0) else next(f for f in fills if f != key)
        left += w+gap
    return grid, card_mask, band_h+scene_gap


def _sample_scene(rng, shape, count, fills, fallback=False):
    h, w = shape
    if fallback:
        if count == 2:
            boxes = [(0, 0, 2*h//3, 2*w//3, fills[0]),
                     (h//3, w//3, h, w, fills[1])]
        else:
            boxes = [(0, 0, 2*h//3, 43*w//100, fills[0]),
                     (h//3, 30*w//100, h, 78*w//100, fills[1]),
                     (0, 65*w//100, 2*h//3, w, fills[0])]
    else:
        boxes = []
        for _ in range(180):
            bh = int(rng.integers(6, h+1))
            bw = int(rng.integers(6, w+1))
            r0 = int(rng.integers(h-bh+1))
            c0 = int(rng.integers(w-bw+1))
            if rng.random() < .35:
                r0 = int(_pick(rng, (0, h-bh)))
            if rng.random() < .35:
                c0 = int(_pick(rng, (0, w-bw)))
            box = (r0, c0, r0+bh, c0+bw, fills[len(boxes) % 2])
            if boxes:
                overlap = _overlap(box, boxes[-1])
                old_area = (boxes[-1][2]-boxes[-1][0])*(boxes[-1][3]-boxes[-1][1])
                if not 4 <= overlap <= .55 * min(bh*bw, old_area):
                    continue
                if any(_overlap(box, other) for other in boxes[:-1]):
                    continue
            boxes.append(box)
            if len(boxes) == count:
                break
        if len(boxes) != count:
            return None
    order = [boxes[int(i)] for i in rng.permutation(count)]
    clean = _paint(shape, -1, order)
    # Full visible bounds and connected support are the reconstruction evidence.
    if _supported_boxes(clean, fills) != tuple(sorted(boxes)):
        return None
    for box in boxes:
        visible = clean[_slice(box)] == box[4]
        if visible.sum() < 40:
            return None
        if min(visible[0].sum(), visible[-1].sum(), visible[:, 0].sum(), visible[:, -1].sum()) < 3:
            return None
    return clean


def _choose_centers(rng, clean, templates, fills, card_mask, style):
    points = [tuple(map(int, p)) for p in np.argwhere(np.isin(clean, fills))]
    rng.shuffle(points)
    fps = {p: _footprint(p, templates[int(clean[p])], clean.shape) for p in points}
    full = {key: [p for p in points if clean[p] == key and
                  len(fps[p]) == len(templates[key]) and all(clean[q] == key for q in fps[p])]
            for key in fills}
    centers = []

    def add_separate(candidates):
        used = set().union(*(fps[p] for p in centers)) if centers else set()
        for p in candidates:
            if p not in centers and not (fps[p] & used):
                centers.append(p)
                return True
        return False

    if style == 'individual':
        for key in fills:
            if not add_separate(full[key]):
                return None
        seam = [p for p in points if any(clean[q] in fills and clean[q] != clean[p] for q in fps[p])]
        exterior = [p for p in points if any(clean[q] not in fills and not card_mask[q] for q in fps[p])]
        if not add_separate(seam) or not add_separate(exterior):
            return None
        if rng.random() < .5:
            add_separate(points)
    elif style == 'overlap':
        for key in _pick(rng, (fills, fills[::-1])):
            available = set(full[key])
            for p in full[key]:
                for q in sorted(fps[p] - {p}):
                    if q in available and (fps[p] & fps[q]) - {p, q}:
                        centers = [p, q]
                        break
                if centers:
                    break
            if centers:
                break
        if not centers:
            return None
        for _ in range(int(rng.integers(0, 4))):
            add_separate(points)
    elif style == 'mixed_overlap':
        for p in points:
            key = int(clean[p])
            other = next(f for f in fills if f != key)
            deltas = sorted({(dr-er, dc-ec) for dr, dc in templates[key]
                             for er, ec in templates[other]})
            rng.shuffle(deltas)
            for dr, dc in deltas:
                q = p[0]+dr, p[1]+dc
                if q in fps and clean[q] == other and any(clean[x] in fills for x in (fps[p] & fps[q]) - {p, q}):
                    centers = [p, q]
                    break
            if centers:
                break
        if not centers:
            return None
        for _ in range(int(rng.integers(1, 6))):
            add_separate(points)
    else:
        total = int(rng.integers(2, 10))
        pool = points
        if style == 'single_material':
            focus = int(_pick(rng, fills))
            pool = [p for p in points if clean[p] == focus]
        centers = pool[:total]
        if rng.random() < .5:
            p = centers[0]
            neighbors = [q for q in points if q not in centers and clean[q] == clean[p]
                         and max(abs(p[0]-q[0]), abs(p[1]-q[1])) <= 2]
            if neighbors:
                centers[-1] = neighbors[0]
    return centers


def _corrupt(rng, clean, centers, templates, fills, background, card_mask, demonstrate):
    corrupted = clean.copy()
    for p in centers:
        corrupted[p] = background
    # Some ordinary errors remain outside strokes, distinguishing repair from
    # annotation. Others may be near a stencil, where clean-before-XOR matters.
    stroke = set().union(*(_footprint(p, templates[int(clean[p])], clean.shape) for p in centers))
    wrong = [tuple(map(int, p)) for p in np.argwhere(np.isin(clean, fills)) if tuple(p) not in centers]
    rng.shuffle(wrong)
    favor_seams = rng.random() < .65
    wrong.sort(key=lambda p: (p in stroke, -int(favor_seams) * sum(
        0 <= p[0]+dr < clean.shape[0] and 0 <= p[1]+dc < clean.shape[1]
        and clean[p[0]+dr, p[1]+dc] in fills and clean[p[0]+dr, p[1]+dc] != clean[p] for dr, dc in _D4)))
    wrong_target = int(rng.integers(1 if demonstrate else 0, 4))
    accepted = 0
    for p in wrong:
        if accepted >= wrong_target:
            break
        other = next(f for f in fills if f != clean[p])
        neighbors = [(p[0]+dr, p[1]+dc) for dr, dc in _D4
                     if 0 <= p[0]+dr < clean.shape[0] and 0 <= p[1]+dc < clean.shape[1]]
        if sum(corrupted[q] == other for q in neighbors) <= 1:
            corrupted[p] = other
            accepted += 1
    exterior = [tuple(map(int, p)) for p in np.argwhere((clean == background) & ~card_mask)
                if tuple(p) not in stroke]
    rng.shuffle(exterior)
    speckle_target = int(rng.integers(1 if demonstrate else 0, 5))
    speckles = 0
    for p in exterior:
        if speckles >= speckle_target:
            break
        if all(corrupted[q] == background for q in _footprint(p, ((0, 0),) + _D4, clean.shape)):
            corrupted[p] = int(_pick(rng, fills))
            speckles += 1
    return corrupted


def _features(a):
    clean, centers, templates, fills = (a[k] for k in ('clean', 'centers', 'templates', 'fills'))
    fps = {p: _footprint(p, templates[int(clean[p])], clean.shape) for p in centers}
    center_set = set(centers)
    noncenter = np.ones(clean.shape, dtype=bool)
    for p in centers:
        noncenter[p] = False
    noise = (a['scene_input'] != clean) & (a['scene_input'] != a['background'])
    individual_keys = set()
    for p in centers:
        key = int(clean[p])
        if len(fps[p]) == len(templates[key]) and all(clean[q] == key for q in fps[p]) and all(
                not (fps[p] & fps[q]) for q in centers if q != p):
            individual_keys.add(key)
    return dict(
        shape=clean.shape, glyphs=tuple(sorted(templates.values())),
        glyph_sizes=tuple(sorted(map(len, templates.values()))), rectangles=len(a['boxes']),
        boxes=tuple(sorted(box[:4] for box in a['boxes'])), holes=len(centers),
        palette=(a['background'], *fills, a['highlight']), individual_keys=individual_keys,
        seam=any(clean[q] in fills and clean[q] != clean[p] for p in centers for q in fps[p]),
        exterior=any(clean[q] == a['background'] and not a['card_mask'][q] for p in centers for q in fps[p]),
        clipped=any(len(fps[p]) < len(templates[int(clean[p])]) for p in centers),
        even=bool(np.any((a['counts'] > 0) & (a['counts'] % 2 == 0) & np.isin(clean, fills) & noncenter)),
        center_hit=any(q in fps[p] for p in centers for q in center_set - {p}),
        mixed_overlap=any(clean[p] != clean[q] and any(clean[x] in fills for x in
                          (fps[p] & fps[q]) - center_set) for i, p in enumerate(centers) for q in centers[:i]),
        ordinary_noise=int(noise.sum()),
        seam_noise=any(any(0 <= r+dr < clean.shape[0] and 0 <= c+dc < clean.shape[1]
                           and clean[r+dr, c+dc] in fills and clean[r+dr, c+dc] != clean[r, c]
                           for dr, dc in _D4) for r, c in np.argwhere(noise & np.isin(clean, fills))),
    )


def _make_pair(rng, palette, count, style):
    background, first, second, highlight = palette
    fills = (first, second)
    for attempt in range(100):
        repetitions = 2 if style == 'individual' else int(rng.choice([1, 1, 1, 2]))
        if attempt >= 65:
            templates = {first: ((-1, 0), (0, 0), (0, 1), (1, 0), (1, 1)),
                         second: ((-1, -1), (-1, 0), (0, 0), (0, 1), (1, 0))}
        else:
            templates = {key: _new_template(rng, repetitions == 2) for key in fills}
        if templates[first] == templates[second]:
            continue
        grid, card_mask, scene_start = _layout(rng, templates, fills, background, highlight, repetitions)
        local = _sample_scene(rng, (grid.shape[0]-scene_start, grid.shape[1]), count, fills, attempt >= 45)
        if local is None:
            continue
        clean = np.full(grid.shape, background, dtype=np.int8)
        clean[scene_start:] = np.where(local == -1, background, local)
        centers = _choose_centers(rng, clean, templates, fills, card_mask, style)
        if centers is None:
            continue
        corrupted = _corrupt(rng, clean, centers, templates, fills, background, card_mask, style == 'individual')
        corrupted[card_mask] = grid[card_mask]
        output, _ = _compose(clean, centers, templates, fills, highlight)
        output[card_mask] = grid[card_mask]
        turns = int(rng.integers(4))
        reflected = bool(rng.integers(2))
        corrupted, output = np.rot90(corrupted, turns), np.rot90(output, turns)
        if reflected:
            corrupted, output = np.fliplr(corrupted), np.fliplr(output)
        pair = {'input': corrupted.copy(), 'output': output.copy()}
        try:
            a = _analyze(pair['input'])
            if not np.array_equal(a['output'], pair['output']):
                continue
            f = _features(a)
            if style == 'individual' and not (f['individual_keys'] == set(fills) and f['seam'] and f['exterior']
                                               and f['ordinary_noise'] >= 2):
                continue
            if style == 'overlap' and not (f['even'] and f['center_hit']):
                continue
            if style == 'mixed_overlap' and not (f['even'] and f['mixed_overlap']):
                continue
            return pair
        except AssertionError:
            continue
    raise RuntimeError('Bounded construction could not retain sufficient visible evidence')


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
            raise ValueError('Training count must be 2..10 and test count 1..4, both integers')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    return int(num_train), int(num_test)


def _signature(grid):
    grid = np.asarray(grid)
    return grid.shape, grid.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    base = tuple(map(int, rng.choice(10, 4, replace=False)))
    palette_policy = int(rng.integers(3))  # fixed, fixed materials, or independent
    shift = int(rng.integers(2))  # new rectangle count, or mixed-key overlap
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, total in (('train', num_train), ('test', num_test)):
        for index in range(total):
            if split == 'train':
                extra_style = 'random' if shift == 0 else 'single_material'
                style = 'individual' if index == 0 else 'overlap' if index == 1 else _pick(rng, (extra_style, 'overlap'))
                count = 2 if shift == 0 else int(rng.choice([2, 2, 3]))
            else:
                style = 'random' if shift == 0 else 'mixed_overlap'
                count = 3 if shift == 0 else int(rng.choice([2, 3]))
            palette = base
            if palette_policy == 1:
                remaining = [c for c in range(10) if c not in base[1:3]]
                bg, hi = map(int, rng.choice(remaining, 2, replace=False))
                palette = (bg, *base[1:3], hi)
            elif palette_policy == 2:
                palette = tuple(map(int, rng.choice(10, 4, replace=False)))
            for _ in range(5):
                pair = _make_pair(rng, palette, count, style)
                key = _signature(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('Could not make a distinct input')
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _ablation(a, templates=None, **options):
    out, _ = _compose(a['clean'], a['centers'], templates or a['templates'], a['fills'], a['highlight'], **options)
    out[a['card_mask']] = a['grid'][a['card_mask']]
    return out


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            assert _signature(inp) not in seen
            seen.add(_signature(inp))
            a = _analyze(inp)
            assert np.array_equal(a['output'], out), 'Input-inferred engraving disagrees with label'
            assert np.array_equal(inp[a['card_mask']], out[a['card_mask']]), 'Reference samples changed'
            assert all(out[p] == a['highlight'] for p in a['centers'])
            assert not np.any((out == a['highlight']) & ~a['card_mask'] & ~np.isin(a['scene_input'], [a['background']]))
            assert not np.array_equal(inp, out)
            if split == 'train':
                training.append((a, _features(a)))
    assert any(f['individual_keys'] == set(a['fills']) and f['seam'] and f['exterior'] and f['ordinary_noise'] >= 2
               for a, f in training), 'Training must expose both keys, local inversion, masking, and repair'
    assert any(f['even'] and f['center_hit'] for _, f in training), 'Training must expose parity and center precedence'
    ring = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1))
    # These are alternative-rule checks, not independent semantic certification.
    assert any(not np.array_equal(_ablation(a, {k: ring for k in a['fills']}), a['output']) for a, _ in training)
    assert any(not np.array_equal(_ablation(a, {a['fills'][0]: a['templates'][a['fills'][1]],
                                               a['fills'][1]: a['templates'][a['fills'][0]]}), a['output']) for a, _ in training)
    for option in ({'parity': False}, {'mask_background': False}, {'anchor_color': True}):
        assert any(not np.array_equal(_ablation(a, **option), a['output']) for a, _ in training), option
    # Omit B's repair even with oracle centers/keys: unmarked ordinary noise remains.
    assert any(np.any((a['scene_input'] != a['clean']) & (a['scene_input'] != a['background'])
                         & (a['counts'] == 0)) for a, _ in training)


def self_test() -> dict:
    """Bounded, author-owned semantic probes and structural diversity over 42 seeds."""
    records, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_signature(p['input']) for p in puzzle['train'] + puzzle['test']))
        for pair in puzzle['train'] + puzzle['test']:
            a = _analyze(pair['input'])
            records.append(_features(a))
            # Stroke order must not decide the answer, even for different keys.
            reverse, _ = _compose(a['clean'], list(reversed(a['centers'])), a['templates'], a['fills'], a['highlight'])
            reverse[a['card_mask']] = a['grid'][a['card_mask']]
            assert np.array_equal(reverse, pair['output'])
        if seed == 0:
            again = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for split in puzzle for p, q in zip(puzzle[split], again[split])
                       for k in ('input', 'output'))
    assert len(signatures) == 42
    assert len({f['shape'] for f in records}) >= 30
    assert len({f['boxes'] for f in records}) >= 100
    assert len({g for f in records for g in f['glyphs']}) >= 100
    assert {f['rectangles'] for f in records} == {2, 3}
    assert len({f['holes'] for f in records}) >= 5
    assert len({f['palette'] for f in records}) >= 30
    for feature in ('seam', 'exterior', 'clipped', 'even', 'center_hit', 'mixed_overlap', 'seam_noise'):
        assert sum(bool(f[feature]) for f in records) >= 3, feature
    # A concrete parity/seam/center probe with hand-written expected cells.
    clean = np.ones((7, 8), dtype=np.int8)
    clean[:, 4:] = 2
    clean[0] = 0
    templates = {1: ((0, 0), (0, 1), (1, 0), (1, 1)),
                 2: ((0, 0), (0, -1), (1, 0), (1, -1))}
    out, _ = _compose(clean, [(2, 3), (2, 4)], templates, (1, 2), 8)
    assert out[2, 3] == out[2, 4] == 8
    assert out[3, 3] == 1 and out[3, 4] == 2  # two strokes cancel across the seam
    one, _ = _compose(clean, [(2, 3)], templates, (1, 2), 8)
    assert one[2, 4] == 1 and one[3, 3] == 2 and one[3, 4] == 1
    edge_templates = {1: ((0, 0), (-1, 0), (-2, 0), (0, -1)), 2: templates[2]}
    edge, _ = _compose(clean, [(1, 0)], edge_templates, (1, 2), 8)
    assert np.all(edge[0] == 0) and edge[1, 0] == 8
    broken = generate_puzzle(42, 2, 1)
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('A corrupted label was accepted')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    return {'seeds': 42, 'pairs': len(records), 'distinct_shapes': len({f['shape'] for f in records}),
            'distinct_rectangle_geometries': len({f['boxes'] for f in records}),
            'distinct_glyphs': len({g for f in records for g in f['glyphs']}),
            'feature_pair_counts': {k: sum(bool(f[k]) for f in records) for k in
                                    ('seam', 'exterior', 'clipped', 'even', 'center_hit', 'mixed_overlap', 'seam_noise')}}
