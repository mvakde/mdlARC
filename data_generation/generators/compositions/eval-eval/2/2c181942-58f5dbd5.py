"""Docked punches: rotate keyed silhouettes, subtract their union, release cutouts.

All semantics are visible in one input. The outer background border registers the
canvas; a filled rectangle with four colored two-cell ports registers the plate.
The generator does not import or execute either parent generator.
"""
from collections import deque
import numpy as np


_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _connected(mask, neighbors=_N8):
    cells = set(map(tuple, np.argwhere(mask)))
    if not cells:
        return False
    stack = [cells.pop()]
    while stack:
        r, c = stack.pop()
        for dr, dc in neighbors:
            p = r + dr, c + dc
            if p in cells:
                cells.remove(p)
                stack.append(p)
    return not cells


def _canonical(mask):
    """Collapse symmetry-equivalent centered two-cell extreme attachments."""
    results = {}
    for turn in range(4):
        m = np.rot90(mask, turn).copy()
        h, w = m.shape
        if h < 2 or w < 2 or w % 2:
            continue
        edge = np.zeros(w, dtype=bool)
        edge[w // 2 - 1:w // 2 + 1] = True
        if np.array_equal(m[-1], edge):
            results.setdefault(_key(m), (m, []))[1].append(turn)
    assert len(results) == 1, 'missing or competing attachment edges'
    return next(iter(results.values()))


def _ports(h, w):
    return (((0, w // 2 - 1), (0, w // 2)),
            ((h // 2 - 1, w - 1), (h // 2, w - 1)),
            ((h - 1, w // 2 - 1), (h - 1, w // 2)),
            ((h // 2 - 1, 0), (h // 2, 0)))


def _plate(h, w, sheet, labels):
    a = np.full((h, w), sheet, dtype=np.int8)
    for color, cells in zip(labels, _ports(h, w)):
        for r, c in cells:
            a[r, c] = color
    return a


def _components(a, bg):
    seen = np.zeros(a.shape, dtype=bool)
    for r, c in np.argwhere(a != bg):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        color, stack, cells = int(a[r, c]), [(r, c)], []
        seen[r, c] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in _N4:
                yy, xx = y + dy, x + dx
                if (0 <= yy < a.shape[0] and 0 <= xx < a.shape[1]
                        and not seen[yy, xx] and a[yy, xx] == color):
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        yield color, np.array(cells, dtype=int)


def _cut_coords(mask, side, h, w):
    """A bottom-facing neck points OUT of the plate, toward its chosen port."""
    marker = np.zeros(mask.shape, dtype=bool)
    marker[-1, mask.shape[1] // 2 - 1:mask.shape[1] // 2 + 1] = True
    turn = (2 - side) % 4
    oriented, neck = np.rot90(mask, turn), np.argwhere(np.rot90(marker, turn))
    step = ((1, 0), (0, -1), (-1, 0), (0, 1))[side]
    target = np.array(_ports(h, w)[side]) + step
    shift = target.min(axis=0) - neck.min(axis=0)
    assert set(map(tuple, neck + shift)) == set(map(tuple, target))
    return np.argwhere(oriented) + shift


def _rim_connected(material, neighbors=_N4):
    """Four-neighbor reachability from every surviving perimeter cell."""
    h, w = material.shape
    keep = np.zeros_like(material)
    queue = deque()
    for r, c in np.argwhere(material):
        if r in (0, h - 1) or c in (0, w - 1):
            keep[r, c] = True
            queue.append((int(r), int(c)))
    while queue:
        r, c = queue.popleft()
        for dr, dc in neighbors:
            y, x = r + dr, c + dc
            if (0 <= y < h and 0 <= x < w and material[y, x]
                    and not keep[y, x]):
                keep[y, x] = True
                queue.append((y, x))
    return keep


def _decode(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    assert all(np.all(edge == bg) for edge in (a[0], a[-1], a[:, 0], a[:, -1]))
    candidates = []
    for color, cells in _components(a, bg):
        lo, hi = cells.min(axis=0), cells.max(axis=0) + 1
        h, w = map(int, hi - lo)
        if h < 8 or w < 8 or h % 2 or w % 2 or len(cells) != h * w - 8:
            continue
        y, x = map(int, lo)
        block = a[y:y + h, x:x + w]
        labels = tuple(int(block[p[0]]) for p in _ports(h, w))
        if len(set(labels)) != 4 or bg in labels or color in labels:
            continue
        if np.array_equal(block, _plate(h, w, color, labels)):
            candidates.append((y, x, h, w, color, labels))
    assert len(candidates) == 1, 'one filled plate with four distinct ports required'
    y, x, h, w, sheet, labels = candidates[0]
    outside = np.ones(a.shape, dtype=bool)
    outside[y:y + h, x:x + w] = False
    assert y >= 1 and x >= 1 and y + h < a.shape[0] and x + w < a.shape[1]
    assert np.all(a[y - 1:y + h + 1, x - 1:x + w + 1][
        outside[y - 1:y + h + 1, x - 1:x + w + 1]] == bg)
    assert not np.any(outside & (a == sheet)), 'sheet color is reserved for plate'
    masks, boxes = {}, []
    cuts = np.zeros((h, w), dtype=bool)
    multiplicity = np.zeros((h, w), dtype=int)
    matches = []
    for color in sorted(map(int, np.unique(a[outside]))):
        if color == bg:
            continue
        cells = np.argwhere(outside & (a == color))
        lo, hi = cells.min(axis=0), cells.max(axis=0) + 1
        r, c, b, d = map(int, (*lo, *hi))
        register = a[r:b, c:d]
        assert outside[r:b, c:d].all()
        assert np.all((register == bg) | (register == color))
        mask = register == color
        assert _connected(mask), 'each punch is one eight-connected silhouette'
        assert not (y - 1 < b and r < y + h + 1 and x - 1 < d and c < x + w + 1)
        boxes.append((r, c, b, d))
        masks[color] = mask
        if color not in labels:
            continue
        canonical, undo = _canonical(mask)
        side = labels.index(color)
        coords = _cut_coords(canonical, side, h, w)
        rr, cc = coords.T
        assert np.all((rr >= 1) & (rr < h - 1) & (cc >= 1) & (cc < w - 1)), 'punch must fit interior'
        cuts[rr, cc] = True
        multiplicity[rr, cc] += 1
        matches.append(dict(color=color, side=side, source=mask, canonical=canonical,
                            turns={(t + 2 - side) % 4 for t in undo}, coords=coords))
    for i, first in enumerate(boxes):
        for second in boxes[i + 1:]:
            assert (first[2] < second[0] or second[2] < first[0]
                    or first[3] < second[1] or second[3] < first[1]), 'punch registers need one-cell separation'
    assert masks and matches
    material = ~cuts
    kept = _rim_connected(material)
    released = material & ~kept
    plate = a[y:y + h, x:x + w].copy()
    plate[~kept] = bg
    out = np.pad(plate, 1, constant_values=bg)
    return out, dict(bg=bg, sheet=sheet, labels=labels, origin=(y, x), shape=(h, w),
                     masks=masks, matches=matches, cuts=cuts, released=released,
                     overlap=int(np.count_nonzero(multiplicity > 1)))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer every cut, orientation and released component from this input alone."""
    return _decode(input_grid)[0]


def _u_mask(width, stem, left, right):
    """Open goal with independent tine lengths and a centered handle."""
    tines = max(left, right)
    m = np.zeros((tines + 1 + stem, width), dtype=bool)
    m[tines] = True
    m[tines - left:tines, 0] = True
    m[tines - right:tines, -1] = True
    m[tines + 1:, width // 2 - 1:width // 2 + 1] = True
    return m


def _loop_pair(rng, radial, width, close):
    # Radial is the interior dimension. Both cut bars stay away from the rim.
    for _ in range(100):
        s1, s2 = map(int, rng.integers(1, 3, size=2))
        gap = radial - 1 - s1 - s2
        if gap < 3:
            continue
        ts = [int(rng.integers(1, 7 - s)) for s in (s1, s1, s2, s2)]
        # Opposite placements reverse the transverse order of the first U.
        joined = ts[0] + ts[3] >= gap - 1 and ts[1] + ts[2] >= gap - 1
        if joined != close:
            continue
        a, b = _u_mask(width, s1, *ts[:2]), _u_mask(width, s2, *ts[2:])
        if close:
            # A diagonal crack is a visible witness for four-neighbor anchoring.
            # The punch remains eight-connected through the diagonal corner.
            a[max(ts[:2]), int(rng.choice((0, width - 1)))] = False
        if max(a.shape[0], b.shape[0]) > radial:
            continue
        try:
            _canonical(a); _canonical(b)
        except AssertionError:
            continue
        return a, b
    # Width>=4, radial>=6: these have a unique geometric neck.
    t = (radial - 3 + 1) // 2 if close else 1
    a, b = _u_mask(width, 1, t, t), _u_mask(width, 1, t, t)
    if close:
        a[t, 0] = False
    return a, b


def _random_mask(rng, radial, transverse, simple=False):
    for _ in range(100):
        w = int(rng.choice([q for q in (2, 4, 6) if q <= transverse]))
        d = int(rng.integers(2, min(7, radial) + 1))
        center = w // 2
        m = np.zeros((d, w), dtype=bool)
        m[:, center - 1:center + 1] = True
        kind = int(rng.integers(2 if simple else 6))
        if kind == 0:  # bar or independently thickened T
            body = int(rng.integers(1, d))
            m[:body] = True
        elif kind == 1:  # irregular branching shoulders with holes
            for row in range(d - 1):
                l, r = int(rng.integers(center)), int(rng.integers(center + 1, w + 1))
                m[row, l:r] = True
            m[0] = True
            for row, col in np.argwhere(m[:-1]):
                if col not in (center - 1, center) and rng.random() < .25:
                    m[row, col] = False
        elif kind == 2 and w >= 4 and d >= 3:
            s = int(rng.integers(1, d - 1))
            t = d - s - 1
            m = _u_mask(w, s, int(rng.integers(1, t + 1)), t)
        elif kind == 3 and w >= 4 and d >= 4:  # closed loop on a stem
            body = int(rng.integers(3, d))
            m = np.zeros((d, w), dtype=bool)
            m[0] = m[body - 1] = True
            m[:body, 0] = m[:body, -1] = True
            m[body:, center - 1:center + 1] = True
        elif kind == 4 and w >= 4:  # diagonal wings; eight-connectivity matters
            m[:] = False
            t = min(d - 1, center - 1)
            for row in range(t):
                m[row, center - 1 - (t - row)] = True
                m[row, center + (t - row)] = True
            m[t:, center - 1:center + 1] = True
        else:  # comb of varying transverse teeth
            for row in range(0, d - 1, 2):
                m[row] = True
        # Trim unused extremes; alignment must follow the visible bounding box.
        rr, cc = np.nonzero(m)
        m = m[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
        try:
            canonical, _ = _canonical(m)
        except AssertionError:
            continue
        if (_connected(m) and canonical.shape[0] <= radial
                and canonical.shape[1] <= transverse and max(m.shape) <= 7):
            return canonical
    return np.array([[1, 1, 1, 1], [0, 1, 1, 0]], dtype=bool)


def _ring_mask(rng, radial, transverse):
    width = int(rng.choice([q for q in (4, 6) if q <= transverse]))
    body = int(rng.integers(3, min(5, radial - 1) + 1))
    stem = int(rng.integers(1, min(2, radial - body, 7 - body) + 1))
    m = np.zeros((body + stem, width), dtype=bool)
    m[0] = m[body - 1] = True
    m[:body, 0] = m[:body, -1] = True
    m[body:, width // 2 - 1:width // 2 + 1] = True
    return m


def _fallback_layout(rng, plate, objects, bg):
    """Pack any six supported registers without changing their orientations."""
    wide = sum(mask.shape[1] > 6 for _, mask in objects)
    transpose = wide >= 4
    local_plate = plate.T if transpose else plate
    local_objects = [(c, m.T if transpose else m) for c, m in objects]
    ph, pw = local_plate.shape
    wide = sum(m.shape[1] > 6 for _, m in local_objects)
    assert ph <= 14 and pw <= 14 and len(objects) <= 6 and wide <= 3
    assert all(max(m.shape) <= 7 and min(m.shape) <= 6 for _, m in local_objects)
    grid = np.full((30, 30), bg, dtype=np.int8)
    y, x = 8 + int(rng.integers(15 - ph)), 8 + int(rng.integers(15 - pw))
    grid[y:y + ph, x:x + pw] = local_plate
    if wide == 3:
        # Three wide boxes above; the remaining boxes occupy the side strips.
        horizontal = [(1, c) for c in (1, 11, 21)]
        vertical = [(r, c) for r in (9, 19) for c in (1, 23)]
    else:
        # Centered horizontal boxes cannot intersect the six side-strip boxes.
        horizontal = [(1, 9), (23, 9)]
        vertical = [(r, c) for r in (1, 11, 21) for c in (1, 23)]
    rng.shuffle(vertical); rng.shuffle(horizontal)
    for color, mask in sorted(local_objects, key=lambda p: -(p[1].shape[1] > 6)):
        slots = horizontal if mask.shape[1] > 6 else vertical
        r, c = slots.pop()
        grid[r:r + mask.shape[0], c:c + mask.shape[1]][mask] = color
    return grid.T.copy() if transpose else grid


def _layout(rng, plate, objects, bg):
    ph, pw = plate.shape
    for _ in range(16):
        h, w = map(int, rng.integers(22, 31, size=2))
        y, x = int(rng.integers(2, h - ph - 1)), int(rng.integers(2, w - pw - 1))
        grid = np.full((h, w), bg, dtype=np.int8)
        grid[y:y + ph, x:x + pw] = plate
        blocked = np.zeros((h, w), dtype=bool)
        blocked[y - 1:y + ph + 1, x - 1:x + pw + 1] = True
        success = True
        for color, mask in sorted(objects, key=lambda p: -p[1].size):
            mh, mw = mask.shape
            options = [(r, c) for r in range(1, h - mh) for c in range(1, w - mw)
                       if not blocked[r:r + mh, c:c + mw].any()]
            if not options:
                success = False
                break
            r, c = options[int(rng.integers(len(options)))]
            grid[r:r + mh, c:c + mw][mask] = color
            blocked[r - 1:r + mh + 1, c - 1:c + mw + 1] = True
        if success:
            break
    else:
        grid = _fallback_layout(rng, plate, objects, bg)
    # Geometry-preserving crop plus variable padding keeps a visible blank border.
    rr, cc = np.nonzero(grid != bg)
    grid = grid[rr.min() - 1:rr.max() + 2, cc.min() - 1:cc.max() + 2]
    pads = []
    for d in grid.shape:
        extra = int(rng.integers(min(4, 30 - d) + 1))
        start = int(rng.integers(extra + 1))
        pads.append((start, extra - start))
    return np.pad(grid, pads, constant_values=bg)


def _make_pair(rng, shape, sides, bg, *, axis=None, close=None, deltas=None,
               distractors=0, forced=None, simple=False):
    h, w = shape
    palette = list(map(int, rng.permutation([c for c in range(10) if c != bg])))
    sheet, labels = palette[0], tuple(palette[1:5])
    masks = {} if forced is None else dict(forced)
    if axis is not None:
        radial, cross = (h - 2, w - 2) if axis == 0 else (w - 2, h - 2)
        width = int(rng.choice([q for q in (4, 6) if q <= cross]))
        first, second = _loop_pair(rng, radial, width, close)
        masks[axis], masks[axis + 2] = first, second
    for side in sorted(sides):
        if side not in masks:
            radial, cross = (h - 2, w - 2) if side % 2 == 0 else (w - 2, h - 2)
            masks[side] = _random_mask(rng, radial, cross, simple=simple or close is False)
    objects = []
    for side, mask in sorted(masks.items()):
        turn = (2 - side - deltas[side]) % 4 if deltas and side in deltas else int(rng.integers(4))
        objects.append((labels[side], np.rot90(mask, turn).copy()))
    for color in palette[5:5 + distractors]:
        # Same silhouettes as valid punches; they are discarded only by color.
        mask = _random_mask(rng, 7, 6)
        objects.append((color, np.rot90(mask, int(rng.integers(4))).copy()))
    inp = _layout(rng, _plate(h, w, sheet, labels), objects, bg)
    out, info = _decode(inp)
    return dict(input=inp, output=out), info


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def _individual_release_counts(info):
    counts = []
    for p in info['matches']:
        cut = np.zeros(info['shape'], dtype=bool)
        rr, cc = p['coords'].T
        cut[rr, cc] = True
        counts.append(int(np.count_nonzero((~cut) & ~_rim_connected(~cut))))
    return counts


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    axis = int(rng.integers(2))
    slots = list(map(int, rng.choice(nt, 2, replace=False)))
    # Two opposite-punch contrasts, on complementary axes, cover every port.
    rotations = list(map(int, rng.permutation(4)))
    mixed_bg = rng.random() < .8
    fixed_bg = int(rng.integers(10))
    regime = int(rng.integers(3))  # count combination, visible scale, single-punch closure
    for i in range(nt + nq):
        split = 'train' if i < nt else 'test'
        witness = slots.index(i) if i in slots else None
        dimensions = (8, 10, 12) if regime == 1 and split == 'train' else (8, 10, 12, 14)
        if witness is not None:
            dimensions = tuple(d for d in dimensions if d >= 10)
        h, w = int(rng.choice(dimensions)), int(rng.choice(dimensions))
        if regime == 1 and split == 'test':
            if rng.random() < .5:
                h = 14
            else:
                w = 14
        bg = int(rng.integers(10)) if mixed_bg else fixed_bg
        ring_axis, close, deltas, forced = None, None, None, None
        if witness is not None:
            ring_axis, close = (axis + witness) % 2, bool(witness)
            sides = {ring_axis, ring_axis + 2}
            deltas = {s: rotations[2 * witness + j] for j, s in enumerate(sorted(sides))}
            distractors = 1 + int(rng.integers(2)) if witness == 0 else int(rng.integers(3))
        else:
            n = int(rng.integers(1, 5))
            if split == 'test' and regime == 0:
                n = int(rng.integers(3, 5))
            sides = set(map(int, rng.choice(4, n, replace=False)))
            distractors = int(rng.integers(3))
            if split == 'test' and regime == 2:
                side = int(rng.choice(sorted(sides)))
                radial, cross = (h - 2, w - 2) if side % 2 == 0 else (w - 2, h - 2)
                forced = {side: _ring_mask(rng, radial, cross)}
        for _ in range(100):
            pair, info = _make_pair(rng, (h, w), sides, bg, axis=ring_axis,
                                    close=close, deltas=deltas, distractors=distractors,
                                    forced=forced, simple=regime == 2 and split == 'train')
            # Contrast witnesses must survive the geometric and topology checks.
            if witness is not None and bool(info['released'].any()) != close:
                continue
            if witness == 1:
                material = ~info['cuts']
                if not info['overlap'] or np.array_equal(_rim_connected(material), _rim_connected(material, _N8)):
                    continue
            if forced is not None and not info['released'].any():
                continue
            if regime == 2 and split == 'train' and any(_individual_release_counts(info)):
                continue
            key = _key(pair['input'])
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError('bounded construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _without_rotation(info):
    """Shortcut counterexample: flush-align displayed masks without turning them."""
    h, w = info['shape']
    cuts = np.zeros((h, w), dtype=bool)
    for p in info['matches']:
        mask, side = p['source'], p['side']
        mh, mw = mask.shape
        r = 1 if side == 0 else h - 1 - mh if side == 2 else (h - mh) // 2
        c = 1 if side == 3 else w - 1 - mw if side == 1 else (w - mw) // 2
        for y, x in np.argwhere(mask) + (r, c):
            if 0 < y < h - 1 and 0 < x < w - 1:
                cuts[y, x] = True
    plate = _plate(h, w, info['sheet'], info['labels'])
    plate[~_rim_connected(~cuts)] = info['bg']
    return np.pad(plate, 1, constant_values=info['bg'])


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, train = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            expected, info = _decode(inp)
            assert np.array_equal(out, expected), 'incorrect cuts, release, rim or crop'
            h, w = info['shape']
            assert out.shape == (h + 2, w + 2)
            assert all(np.all(e == info['bg']) for e in (out[0], out[-1], out[:, 0], out[:, -1]))
            base = _plate(h, w, info['sheet'], info['labels'])
            rim = np.ones((h, w), dtype=bool); rim[1:-1, 1:-1] = False
            assert np.array_equal(out[1:-1, 1:-1][rim], base[rim])
            assert np.count_nonzero(out[1:-1, 1:-1] == info['bg']) == int(info['cuts'].sum() + info['released'].sum())
            assert _connected(out[1:-1, 1:-1] != info['bg'], _N4)
            if split == 'train':
                train.append((out, info))
    assert set(p['side'] for _, info in train for p in info['matches']) == set(range(4))
    assert set(t for _, info in train for p in info['matches'] for t in p['turns']) == set(range(4))
    assert any(len(info['masks']) > len(info['matches']) for _, info in train), 'need color filtering witness'
    assert any(not info['released'].any() for _, info in train), 'need open-cut contrast'
    assert any(info['released'].any() for _, info in train), 'need closed-cut release witness'
    assert any(not np.array_equal(_rim_connected(~info['cuts']), _rim_connected(~info['cuts'], _N8))
               for _, info in train), 'need diagonal contact convention witness'
    assert any(info['overlap'] for _, info in train), 'need idempotent overlapping cuts witness'
    assert any(info['released'].any() and not any(_individual_release_counts(info))
               for _, info in train), 'need release that depends on the interaction of punches'
    assert any(not np.array_equal(_without_rotation(info), out) for out, info in train), 'rotation shortcut survived'
    # Dropping B leaves the original solid plate; dropping release leaves islands.
    assert all(info['cuts'].any() for _, info in train)
    assert any(np.any(info['released'] & ~info['cuts']) for _, info in train)


def self_test():
    """Bounded semantic regressions, 42-seed diversity, counts and mutation checks."""
    signatures, shapes, sizes, masks, counts, bgs, subsets = set(), set(), set(), set(), set(), set(), set()
    overlap = release = diagonal = asymmetric = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                expected, info = _decode(pair['input'])
                assert np.array_equal(solve(pair['input']), expected)
                shapes.add(pair['input'].shape); sizes.add(info['shape']); bgs.add(info['bg'])
                counts.add((len(info['matches']), len(info['masks']) - len(info['matches'])))
                subsets.add(tuple(sorted(p['side'] for p in info['matches'])))
                overlap += info['overlap'] > 0
                release += info['released'].any()
                for p in info['matches']:
                    m = p['canonical']
                    masks.add(_key(m)); diagonal += not _connected(m, _N4)
                    asymmetric += not np.array_equal(m, np.fliplr(m))
    assert len(signatures) == 42 and len(shapes) >= 30 and len(sizes) >= 10
    assert len(masks) >= 60 and len(counts) >= 9 and len(subsets) >= 12
    assert bgs == set(range(10)) and overlap >= 20 and release >= 42
    assert diagonal >= 3 and asymmetric >= 15
    # A jointly closed ring: neither U alone releases anything.
    a, b = _u_mask(6, 1, 4, 4), _u_mask(6, 1, 4, 4)
    ca, cb = np.zeros((12, 12), bool), np.zeros((12, 12), bool)
    for cut, mask, side in ((ca, a, 0), (cb, b, 2)):
        rr, cc = _cut_coords(mask, side, 12, 12).T
        cut[rr, cc] = True
        assert not np.any((~cut) & ~_rim_connected(~cut))
    union = ca | cb
    assert np.any((~union) & ~_rim_connected(~union))
    # Connectivity is four-neighbor: a diagonal material contact does not anchor.
    material = np.zeros((5, 5), bool)
    material[0, 0] = material[1, 1] = True
    assert not _rim_connected(material)[1, 1]
    assert len(_canonical(np.ones((2, 2), bool))[1]) == 4
    # Explicitly exercise the rare layout fallback for every orientation mixture.
    tall = np.zeros((7, 6), dtype=bool)
    tall[0] = True; tall[1:, 2:4] = True
    wide = np.rot90(tall)
    for nwide in range(7):
        objects = [(c + 1, wide if c < nwide else tall) for c in range(6)]
        inp = _fallback_layout(np.random.default_rng(500 + nwide),
                               _plate(14, 14, 8, (1, 2, 3, 4)), objects, 0)
        _, info = _decode(inp)
        assert len(info['masks']) == 6 and len(info['matches']) == 4
    original = generate_puzzle(991, 2, 1)
    again = generate_puzzle(991, 2, 1)
    assert all(np.array_equal(p[k], q[k]) for s in original for p, q in zip(original[s], again[s]) for k in p)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + nt * 5 + nq, nt, nq)
            assert (len(p['train']), len(p['test'])) == (nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    damaged = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in original[s]] for s in original}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('bad output accepted')
    return {'seeds': 42, 'distinct_puzzles': len(signatures), 'input_shapes': len(shapes),
            'plate_shapes': len(sizes), 'canonical_silhouettes': len(masks),
            'count_combinations': len(counts), 'side_subsets': len(subsets),
            'overlap_pairs': int(overlap), 'release_pairs': int(release),
            'diagonal_punches': int(diagonal), 'asymmetric_punches': int(asymmetric)}


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
