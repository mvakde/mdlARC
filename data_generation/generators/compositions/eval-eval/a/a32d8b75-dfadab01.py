"""Parity printing: registered stencil impressions repeatedly exchange a tile.

Only Python's standard library and NumPy are used. No parent module is imported.
The visible legend is sufficient for solve(input_grid); generation metadata and
other examples are never consulted by the solver.
"""
from __future__ import annotations

import copy
import json
import numpy as np


_CORNERS = ((0, 0), (0, 1), (1, 1), (1, 0))
_DIRS = ((0, 1), (1, 2), (2, 1), (1, 0))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(int, copy=False)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _runs(bits):
    padded = np.r_[False, bits, False].astype(int)
    return list(zip(np.flatnonzero(np.diff(padded) == 1),
                    np.flatnonzero(np.diff(padded) == -1)))


def _box(a, r, c, content, frame):
    h, w = content.shape
    a[r:r + h + 2, c:c + w + 2] = frame
    a[r + 1:r + h + 1, c + 1:c + w + 1] = content


def _legend(tile, book, bg, frame, width, order):
    n = len(tile)
    height = max(n + 2, 5, *(v['mask'].shape[0] + 2 for v in book.values()))
    a = np.full((height + 1, width), bg, int)
    _box(a, 0, 0, tile, frame)
    c = n + 3
    direction_ink = int(np.unique(tile)[0])
    for code in order:
        ins = book[code]
        mask = ins['mask']
        _box(a, 0, c, np.where(mask, code, bg), frame)
        control = np.full((3, 3), bg, int)
        ar, ac = _CORNERS[ins['corner']]
        control[2 * ar, 2 * ac] = code
        control[1, 1] = direction_ink
        control[_DIRS[ins['turn']]] = direction_ink
        _box(a, 0, c + mask.shape[1] + 1, control, frame)
        c += mask.shape[1] + 7
    assert c <= width  # includes one final ground cell
    a[height] = frame
    return a


def _decode(input_grid):
    """Decode complete framed content; no coordinates depend on seed or split."""
    g = _grid(input_grid)
    frame = int(g[0, 0])
    runs = _runs(g[0] == frame)
    assert 2 <= len(runs) <= 3 and runs[0][0] == 0
    first_end = int(runs[0][1])
    assert first_end < g.shape[1]
    bg = int(g[0, first_end])
    assert bg != frame and np.all(g[0, g[0] != frame] == bg)
    n = first_end - 2
    assert 2 <= n <= 4
    tile = g[1:n + 1, 1:n + 1].copy()
    colors = set(map(int, np.unique(tile)))
    assert len(colors) == 2 and not colors.intersection({bg, frame})
    book = {}
    for start, end in runs[1:]:
        start, end = int(start), int(end)
        mw = end - start - 6
        assert 2 <= mw <= 4
        bottoms = [r for r in range(3, min(6, len(g)))
                   if np.all(g[r, start:start + mw + 2] == frame)]
        assert bottoms
        mh = bottoms[0] - 1
        assert 2 <= mh <= 4
        raw = g[1:mh + 1, start + 1:start + mw + 1]
        vals = set(map(int, np.unique(raw))) - {bg}
        assert len(vals) == 1
        code = vals.pop()
        assert code not in colors | {bg, frame} and code not in book
        mask = raw == code
        assert mask[0].any() and mask[-1].any()
        assert mask[:, 0].any() and mask[:, -1].any()
        control = g[1:4, start + mw + 2:start + mw + 5]
        dots = list(map(tuple, np.argwhere(control == code)))
        assert len(dots) == 1
        corners = [(2 * r, 2 * c) for r, c in _CORNERS]
        assert dots[0] in corners
        arrow = set(map(tuple, np.argwhere((control != bg) & (control != code))))
        assert (1, 1) in arrow and len(arrow) == 2
        neighbor = next(iter(arrow - {(1, 1)}))
        assert neighbor in _DIRS
        arrow_colors = set(int(control[q]) for q in arrow)
        assert len(arrow_colors) == 1 and arrow_colors <= colors
        book[code] = dict(mask=mask.copy(), turn=_DIRS.index(neighbor),
                          corner=corners.index(dots[0]))
    header = _legend(tile, book, bg, frame, g.shape[1], list(book))
    assert header.shape[0] < g.shape[0]
    assert np.array_equal(g[:len(header)], header), 'malformed legend/frame'
    canvas = g[len(header):].copy()
    assert set(map(int, np.unique(canvas))) <= {bg} | colors | set(book)
    return tile, book, bg, frame, canvas


def _components(a, bg):
    remaining = set(map(tuple, np.argwhere(a != bg)))
    objects = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in remaining:
                        remaining.remove(q)
                        stack.append(q)
        objects.append(cells)
    return objects


def _members(canvas, codes, bg):
    live, rejected, bare, painted = [], [], [], []
    for cells in _components(canvas, bg):
        points = [(r, c, int(canvas[r, c])) for r, c in cells
                  if int(canvas[r, c]) in codes]
        origin = min(r for r, _ in cells), min(c for _, c in cells)
        if len(points) == 1 and points[0][:2] == origin:
            live.append(points[0])
            if len(cells) > 1:
                painted.append(points[0])
        elif points:
            rejected.extend(points)
        else:
            bare.append(cells)
    return sorted(live), sorted(rejected), bare, sorted(painted)


def _transform_corner(corner, transform):
    r, c = _CORNERS[corner]
    if transform >= 4:
        c = 1 - c
    for _ in range(transform % 4):
        r, c = c, 1 - r
    return r, c


def _forward_hits(shape, n, records, book, convention=(1, 0, 0), mode='normal'):
    """Construction: rotate small masks and accumulate registered impressions."""
    H, W = shape
    hits = np.zeros(((H + n - 1) // n, (W + n - 1) // n), int)
    sign, offset, transform = convention
    for r, c, code in records:
        ins = book[code]
        turn = (sign * ins['turn'] + offset) % 4
        if mode == 'unrotated':
            turn = 0
        mask = np.rot90(ins['mask'], -turn)
        if mode == 'fill_holes':
            mask = np.ones_like(mask)
        ar, ac = _transform_corner(ins['corner'], transform)
        if mode == 'top_left':
            ar = ac = 0
        y = r // n - ar * (mask.shape[0] - 1)
        x = c // n - ac * (mask.shape[1] - 1)
        if mode == 'contained_only' and not (0 <= y and 0 <= x and
                (y + mask.shape[0]) * n <= H and (x + mask.shape[1]) * n <= W):
            continue
        if mode == 'unit_stamp':
            y, x, mask = r // n, c // n, np.ones((1, 1), bool)
        for a, b in np.argwhere(mask):
            yy, xx = y + int(a), x + int(b)
            if 0 <= yy < hits.shape[0] and 0 <= xx < hits.shape[1]:
                hits[yy, xx] += 1
    return hits


def _inverse_hits(shape, n, records, book):
    """Validation/solver path: each coarse cell queries unrotated source masks."""
    H, W = shape
    rr, cc = np.indices(((H + n - 1) // n, (W + n - 1) // n))
    hits = np.zeros(rr.shape, int)
    for r, c, code in records:
        ins = book[code]
        mask, k = ins['mask'], ins['turn']
        sh, sw = mask.shape
        rh, rw = (sw, sh) if k % 2 else (sh, sw)
        ar, ac = _CORNERS[ins['corner']]
        y = rr - (r // n - ar * (rh - 1))
        x = cc - (c // n - ac * (rw - 1))
        inside = (y >= 0) & (y < rh) & (x >= 0) & (x < rw)
        if k == 0:
            sy, sx = y, x
        elif k == 1:
            sy, sx = sh - 1 - x, y
        elif k == 2:
            sy, sx = sh - 1 - y, sw - 1 - x
        else:
            sy, sx = x, sw - 1 - y
        hits += inside & mask[np.clip(sy, 0, sh - 1), np.clip(sx, 0, sw - 1)]
    return hits


def _render(shape, tile, bg, hits, mode='normal'):
    rr, cc = np.indices(shape)
    n = len(tile)
    counts = hits[rr // n, cc // n]
    raw = tile[rr % n, cc % n]
    vals = np.unique(tile)
    swapped = int(vals[0]) + int(vals[1]) - raw
    active = counts > 0
    exchanged = counts % 2 == 1
    if mode == 'no_exchange':
        exchanged[:] = False
    elif mode == 'no_parity':
        exchanged[:] = True
    elif mode == 'cancel_overlap':
        active &= counts % 2 == 1
    elif mode == 'overlap_raw':
        exchanged = counts == 1
    return np.where(active, np.where(exchanged, swapped, raw), bg).astype(int)


def solve(input_grid: np.ndarray) -> np.ndarray:
    tile, book, bg, _, canvas = _decode(input_grid)
    records = _members(canvas, set(book), bg)[0]
    n = len(tile)
    assert all(r % n == 0 and c % n == 0 for r, c, _ in records)
    hits = _inverse_hits(canvas.shape, n, records, book)
    return _render(canvas.shape, tile, bg, hits)


def _alternative(decoded, behavior):
    tile, book, bg, _, canvas = decoded
    records, _, _, painted = _members(canvas, set(book), bg)
    if behavior == 'all_codes':
        records = [(int(r), int(c), int(canvas[r, c]))
                   for r, c in np.argwhere(np.isin(canvas, list(book)))]
    elif behavior == 'singleton_only':
        records = [v for v in records if v not in painted]
    elif behavior == 'ignore_selector':
        records = [(r, c, min(book)) for r, c, _ in records]
    hits = _forward_hits(canvas.shape, len(tile), records, book, mode=behavior)
    out = _render(canvas.shape, tile, bg, hits, behavior)
    if behavior == 'keep_material':
        rr, cc = np.indices(canvas.shape)
        exposed = (hits[rr // len(tile), cc // len(tile)] == 0) & np.isin(canvas, np.unique(tile))
        out[exposed] = canvas[exposed]
    return out


_ALTERNATIVES = ('no_exchange', 'no_parity', 'cancel_overlap', 'overlap_raw',
                 'fill_holes', 'unrotated', 'top_left', 'contained_only',
                 'all_codes', 'singleton_only', 'unit_stamp', 'keep_material')


def _rotated_tile_witness(decoded):
    tile, book, bg, _, canvas = decoded
    n = len(tile)
    records = _members(canvas, set(book), bg)[0]
    all_hits = _forward_hits(canvas.shape, n, records, book)
    rr, cc = np.indices(canvas.shape)
    for record in records:
        code = record[2]
        k = book[code]['turn']
        one = _forward_hits(canvas.shape, n, [record], book)
        sole = (all_hits[rr // n, cc // n] == 1) & (one[rr // n, cc // n] == 1)
        rotated = np.rot90(tile, -k)
        if np.any(sole & (tile[rr % n, cc % n] != rotated[rr % n, cc % n])):
            return True
    return False


def _training_checks(pairs):
    decoded = [_decode(p['input']) for p in pairs]
    for behavior in _ALTERNATIVES:
        assert any(not np.array_equal(_alternative(d, behavior), p['output'])
                   for d, p in zip(decoded, pairs)), 'untaught alternative: ' + behavior
    if len(decoded[0][1]) > 1:
        assert any(not np.array_equal(_alternative(d, 'ignore_selector'), p['output'])
                   for d, p in zip(decoded, pairs)), 'selector is immaterial'
    assert any(_rotated_tile_witness(d) for d in decoded), 'no stencil-only rotation witness'
    survivors = []
    for sign in (1, -1):
        for offset in range(4):
            for transform in range(8):
                convention = sign, offset, transform
                if all(np.array_equal(_render(d[4].shape, d[0], d[2],
                        _forward_hits(d[4].shape, len(d[0]),
                                      _members(d[4], set(d[1]), d[2])[0], d[1], convention)),
                        p['output']) for d, p in zip(decoded, pairs)):
                    survivors.append(convention)
    assert survivors == [(1, 0, 0)], 'ambiguous geometric convention'


def _sample_mask(rng):
    for _ in range(100):
        h, w = map(int, rng.integers(2, 5, size=2))
        style = int(rng.integers(5))
        rr, cc = np.indices((h, w))
        if style == 0:
            m = rng.random((h, w)) < rng.uniform(.35, .8)
        elif style == 1:
            m = (rr == h - 1) | (cc == 0)
            m[int(rng.integers(h)), :] |= rng.random(w) < .7
        elif style == 2:
            m = (rr == 0) | (rr == h - 1) | (cc == 0) | (cc == w - 1)
            m[int(rng.integers(h)), int(rng.integers(w))] = False
        elif style == 3:
            m = ((rr + cc) % 2 == 0) | (rng.random((h, w)) < .15)
        else:
            m = rng.random((h, w)) < .72
        if (m.sum() >= 3 and not m.all() and m[0].any() and m[-1].any()
                and m[:, 0].any() and m[:, -1].any()
                and not np.array_equal(m, np.rot90(m, 2))):
            return m
    return np.array([[1, 0, 0], [1, 1, 1]], bool)


def _sample_tile(rng, n, colors):
    for _ in range(100):
        style = int(rng.integers(4))
        rr, cc = np.indices((n, n))
        if style == 0:
            bits = rng.random((n, n)) < rng.uniform(.2, .8)
        elif style == 1:
            bits = rr <= cc
            bits[int(rng.integers(n)), int(rng.integers(n))] ^= True
        elif style == 2:
            bits = (rr == int(rng.integers(n))) | (cc == int(rng.integers(n)))
        else:
            bits = (rr + cc) % 2 == 0
            bits[int(rng.integers(n)), int(rng.integers(n))] ^= True
        if bits.any() and not bits.all() and not np.array_equal(bits, np.rot90(bits, 2)):
            return np.where(bits, colors[0], colors[1]).astype(int)
    bits = np.zeros((n, n), bool)
    bits[0, 0] = True
    return np.where(bits, colors[0], colors[1]).astype(int)


def _material(canvas, tile, book, bg, rng, kind, records):
    n = len(tile)
    H, W = canvas.shape
    if kind == 'painted':
        candidates = [(r, c, code) for r, c, code in records]
    else:
        candidates = [(r, c, int(rng.choice(list(book))))
                      for r in range(0, H - n + 1, n)
                      for c in range(0, W - n + 1, n)]
    rng.shuffle(candidates)
    for r, c, code in candidates:
        extra = int(kind == 'attached')
        if r + n + extra > H or c + n + extra > W:
            continue
        old = canvas.copy()
        if kind == 'painted':
            if old[r, c] != code:
                continue
            old[r, c] = bg
        region = old[max(0, r - 1):min(H, r + n + extra + 1),
                     max(0, c - 1):min(W, c + n + extra + 1)]
        if np.any(region != bg):
            continue
        canvas[r:r + n, c:c + n] = tile
        if kind == 'painted':
            canvas[r, c] = code
        elif kind == 'attached':
            canvas[r + n, c + n] = code
        return True
    return False


def _clip_sides(shape, n, records, book):
    H, W = shape
    sides = set()
    full = set()
    for r, c, code in records:
        ins = book[code]
        sh, sw = ins['mask'].shape
        rh, rw = (sw, sh) if ins['turn'] % 2 else (sh, sw)
        ar, ac = _CORNERS[ins['corner']]
        y, x = r - ar * (rh - 1) * n, c - ac * (rw - 1) * n
        if y < 0:
            sides.add('top')
        if x < 0:
            sides.add('left')
        if y + rh * n > H:
            sides.add('bottom')
        if x + rw * n > W:
            sides.add('right')
        if 0 <= y and 0 <= x and y + rh * n <= H and x + rw * n <= W:
            full.add(code)
    return sides, full


def _make_pair(rng, params, controls, kind, density, required_codes=()):
    n, bg, frame = params['n'], params['bg'], params['frame']
    book = {code: dict(mask=mask, turn=k, corner=a)
            for (code, mask), (k, a) in zip(params['masks'].items(), controls)}
    header_h = max(n + 2, 5, *(v['mask'].shape[0] + 2 for v in book.values())) + 1
    minimum_width = n + 3 + sum(v['mask'].shape[1] + 7 for v in book.values())
    for _ in range(700):
        heights = list(range(max(4 * n, 12), 31 - header_h))
        widths = list(range(max(minimum_width, 4 * n, 16), 31))
        if kind == 'material_witness':
            heights = [v for v in heights if v % n]
            widths = [v for v in widths if v % n]
        H, W = int(rng.choice(heights)), int(rng.choice(widths))
        tile = _sample_tile(rng, n, params['colors'])
        positions = [(r, c) for r in range(0, H, n) for c in range(0, W, n)]
        if kind == 'clean_witness':
            count = int(rng.integers(max(2, len(book)), 5))
        elif kind == 'material_witness':
            count = int(rng.integers(7, 10))
        elif density == 'dense':
            count = int(rng.integers(7, 13))
        elif density == 'sparse':
            count = int(rng.integers(1, 5))
        else:
            count = int(rng.integers(2, 10))
        records, taken = [], set()
        if kind == 'material_witness':
            # A bottom-right registration can cross the top/left edges; the
            # last peg row/column exposes partial tiles at bottom/right too.
            br_codes = [code for code in book if book[code]['corner'] == 2]
            assert br_codes
            code = br_codes[int(rng.integers(len(br_codes)))]
            last_r, last_c = (H - 1) // n * n, (W - 1) // n * n
            rotated = np.rot90(book[code]['mask'], -book[code]['turn'])
            rh, rw = rotated.shape
            across_r, across_c = (rh - 1) * n, (rw - 1) * n
            boundary = [(0, across_c), (across_r, 0),
                        (last_r, across_c), (across_r, last_c)]
            # Register three occupied cells onto one interior target. This is
            # possible for every supported mask, including a tiny 2x2 L, and
            # avoids relying on an accidental three-way collision in a field.
            target_r = int(rng.integers(0, (H - 1) // n - rh + 2))
            target_c = int(rng.integers(0, (W - 1) // n - rw + 2))
            occupied = np.argwhere(rotated)
            selected = rng.choice(len(occupied), 3, replace=False)
            triple = [((target_r + rh - 1 - int(occupied[j, 0])) * n,
                       (target_c + rw - 1 - int(occupied[j, 1])) * n) for j in selected]
            for p in boundary + triple:
                if p not in taken:
                    taken.add(p)
                    records.append((*p, code))
            count = max(count, len(records))
        if kind == 'clean_witness' or required_codes:
            for code in (book if kind == 'clean_witness' else required_codes):
                candidates = [p for p in positions if p not in taken and
                              code in _clip_sides((H, W), n, [(*p, code)], book)[1]]
                if not candidates:
                    break
                p = candidates[int(rng.integers(len(candidates)))]
                taken.add(p)
                records.append((*p, code))
            needed = len(book) if kind == 'clean_witness' else len(required_codes)
            if len(records) != needed:
                continue
        rng.shuffle(positions)
        for p in positions:
            if len(records) >= count:
                break
            if p not in taken:
                records.append((*p, int(rng.choice(list(book)))))
                taken.add(p)
        canvas = np.full((H, W), bg, int)
        for r, c, code in records:
            canvas[r, c] = code
        hits = _forward_hits(canvas.shape, n, records, book)
        if not np.any(hits):
            continue
        if kind == 'clean_witness' and not (np.any(hits == 1) and np.any(hits == 2)):
            continue
        if kind == 'material_witness':
            # Every boundary direction and counts 1,2,3 are demonstrated before
            # any query. Rectangle clipping is visible even at partial tiles.
            if _clip_sides(canvas.shape, n, records, book)[0] != {'top', 'bottom', 'left', 'right'}:
                continue
            if not all(np.any(hits == j) for j in (1, 2, 3)):
                continue
            requested = ['painted', 'attached', 'bare']
        elif kind == 'clean_witness':
            requested = []
        else:
            requested = [name for name in ('painted', 'attached', 'bare') if rng.random() < .38]
        if not all(_material(canvas, tile, book, bg, rng, name, records) for name in requested):
            continue
        if _members(canvas, set(book), bg)[0] != sorted(records):
            continue
        order = list(book)
        rng.shuffle(order)
        header = _legend(tile, book, bg, frame, W, order)
        inp = np.concatenate([header, canvas])
        out = _render(canvas.shape, tile, bg, hits)
        decoded = tile, book, bg, frame, canvas
        if kind == 'material_witness':
            if any(np.array_equal(_alternative(decoded, b), out)
                   for b in ('all_codes', 'singleton_only', 'keep_material', 'contained_only', 'overlap_raw')):
                continue
        return {'input': inp, 'output': out}
    raise AssertionError('bounded geometric/material packing exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    n, classes = int(rng.integers(2, 5)), int(rng.integers(1, 3))
    masks = {palette[4 + i]: _sample_mask(rng) for i in range(classes)}
    if classes == 2:
        for _ in range(100):
            if not any(np.array_equal(masks[palette[4]], np.rot90(masks[palette[5]], k)) for k in range(4)):
                break
            masks[palette[5]] = _sample_mask(rng)
        else:
            # A different bounding area is a constructive guarantee, even if
            # all bounded random draws happened to repeat the first motif.
            h = 3 if masks[palette[4]].size != 6 else 4
            fallback = np.zeros((h, 2), bool)
            fallback[:, 0] = True
            fallback[-1, 1] = True
            masks[palette[5]] = fallback
    params = dict(n=n, bg=palette[0], frame=palette[1], colors=palette[2:4], masks=masks)
    q, even = int(rng.choice([1, 3])), int(rng.choice([0, 2]))
    # Adjacent corners determine the geometric map. The BR key also permits
    # one compact demonstration to show clipping on all four sides.
    a, b = int(rng.choice([1, 3])), 2
    first = [(q, a)] if classes == 1 else [(q, a), (q, b)]
    second = [(even, b)] if classes == 1 else [(even, a), (even, b)]
    taught = set(first + second)
    held_choices = sorted(set((k, c) for k in range(4) for c in range(4)) - taught)
    held = held_choices[int(rng.integers(len(held_choices)))]
    available = sorted(set((k, c) for k in range(4) for c in range(4)) - {held})
    for _ in range(40):
        witnesses = [_make_pair(rng, params, first, 'clean_witness', 'ordinary'),
                     _make_pair(rng, params, second, 'material_witness', 'ordinary')]
        try:
            _training_checks(witnesses)
        except AssertionError:
            continue
        break
    else:
        raise AssertionError('could not construct informative demonstrations')
    rng.shuffle(witnesses)
    puzzle = {'train': witnesses, 'test': []}
    seen = {_key(p['input']) for p in witnesses}
    test_density = str(rng.choice(['sparse', 'dense', 'ordinary']))
    for split, remaining in (('train', int(num_train) - 2), ('test', int(num_test))):
        for i in range(remaining):
            controls = [available[int(rng.integers(len(available)))] for _ in masks]
            required = ()
            if split == 'test':
                held_index = int(rng.integers(classes))
                controls[held_index] = held
                required = (list(masks)[held_index],)
            for _ in range(20):
                pair = _make_pair(rng, params, controls, 'ordinary',
                                  test_density if split == 'test' else 'ordinary', required)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise AssertionError('distinct input construction exhausted')
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_material(canvas, tile, book, bg):
    n = len(tile)
    categories = set()
    for cells in _components(canvas, bg):
        r, c = min(y for y, _ in cells), min(x for _, x in cells)
        points = [p for p in cells if int(canvas[p]) in book]
        if len(cells) == 1:
            assert len(points) == 1
            categories.add('singleton')
            continue
        expected = {(r + y, c + x): int(tile[y, x]) for y in range(n) for x in range(n)}
        if not points:
            categories.add('bare')
        else:
            assert len(points) == 1
            p = points[0]
            if p == (r, c):
                categories.add('painted')
            else:
                assert p == (r + n, c + n), 'unsupported attachment'
                categories.add('attached')
            expected[p] = int(canvas[p])
        assert {p: int(canvas[p]) for p in cells} == expected, 'material is not a raw tile'
    return categories


def _family_signature(decoded):
    tile, book, bg, frame, _ = decoded
    return (len(tile), bg, frame, tuple(map(int, np.unique(tile))),
            tuple((code, _key(ins['mask'])) for code, ins in sorted(book.items())))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, signatures, controls = set(), set(), {'train': set(), 'test': set()}
    test_live_controls = []
    train_categories, train_sides, train_counts, full_codes = set(), set(), set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            decoded = _decode(inp)
            signatures.add(_family_signature(decoded))
            tile, book, bg, _, canvas = decoded
            n = len(tile)
            live, rejected, _, _ = _members(canvas, set(book), bg)
            assert live
            assert all(r % n == 0 and c % n == 0 for r, c, _ in live + rejected)
            categories = _check_material(canvas, tile, book, bg)
            forward = _forward_hits(canvas.shape, n, live, book)
            inverse = _inverse_hits(canvas.shape, n, live, book)
            assert np.array_equal(forward, inverse), 'registration disagreement'
            assert out.shape == canvas.shape
            assert np.array_equal(out, solve(inp)), 'incorrect input-derived output'
            assert np.any(forward > 0)
            assert set(map(int, np.unique(out))) <= {bg} | set(map(int, np.unique(tile)))
            controls[split].update((v['turn'], v['corner']) for v in book.values())
            if split == 'train':
                train_categories |= categories
                sides, full = _clip_sides(canvas.shape, n, live, book)
                train_sides |= sides
                full_codes |= full
                train_counts.update(map(int, np.unique(forward)))
            else:
                _, full = _clip_sides(canvas.shape, n, live, book)
                test_live_controls.append({(book[code]['turn'], book[code]['corner']) for code in full})
    assert len(signatures) == 1, 'dictionary, pitch or palette changed silently'
    assert {'bare', 'painted', 'attached', 'singleton'} <= train_categories
    assert train_sides == {'top', 'bottom', 'left', 'right'}
    assert {1, 2, 3} <= train_counts
    assert set(_decode(puzzle['train'][0]['input'])[1]) <= full_codes
    assert controls['test'] - controls['train'], 'no held-out geometric recombination'
    assert all(used - controls['train'] for used in test_live_controls), 'held-out key has no full live impression'
    _training_checks(puzzle['train'])


def self_test():
    """Bounded 42-seed structural, semantic, ablation and mutation checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, structures, stencils, tile_patterns = set(), set(), set(), set()
    pitches, classes, hits_seen, live_counts, backgrounds, sides = set(), set(), set(), set(), set(), set()
    partial, material_types, control_pairs = 0, set(), set()
    for p in puzzles:
        for split in ('train', 'test'):
            for pair in p[split]:
                d = _decode(pair['input'])
                tile, book, bg, _, canvas = d
                n = len(tile)
                records = _members(canvas, set(book), bg)[0]
                hits = _inverse_hits(canvas.shape, n, records, book)
                assert np.array_equal(solve(pair['input']), pair['output'])
                pitches.add(n)
                classes.add(len(book))
                shapes.add((pair['input'].shape, canvas.shape))
                backgrounds.add(bg)
                structures.add((canvas.shape, tuple((r, c, sorted(book).index(code)) for r, c, code in records)))
                hits_seen.update(map(int, hits.ravel()))
                live_counts.add(len(records))
                stencils.update(_key(v['mask']) for v in book.values())
                tile_patterns.add(_key(tile == np.unique(tile)[0]))
                material_types |= _check_material(canvas, tile, book, bg)
                sides |= _clip_sides(canvas.shape, n, records, book)[0]
                control_pairs.update((v['turn'], v['corner']) for v in book.values())
                partial += int(canvas.shape[0] % n != 0 or canvas.shape[1] % n != 0)
    assert pitches == {2, 3, 4} and classes == {1, 2}
    assert len(shapes) >= 60 and len(structures) >= 180
    assert len(stencils) >= 30 and len(tile_patterns) >= 40
    assert len(live_counts) >= 9 and len(backgrounds) >= 8
    assert {0, 1, 2, 3, 4} <= hits_seen and partial >= 20
    assert len(control_pairs) == 16 and len(sides) == 4
    assert material_types == {'singleton', 'bare', 'painted', 'attached'}
    signature = lambda p: tuple((_key(v['input']), _key(v['output']))
                                for s in ('train', 'test') for v in p[s])
    assert len({signature(p) for p in puzzles}) == 42
    assert signature(puzzles[0]) == signature(generate_puzzle(0))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(91, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), ('4', 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    damaged = copy.deepcopy(puzzles[0])
    out = damaged['test'][0]['output']
    out[0, 0] = (int(out[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted test output accepted')
    # Changing only a visible control must change its predicted answer in a
    # demonstrated case; the solver must not remember a seed's dictionary.
    pair = puzzles[0]['train'][0]
    tile, book, bg, frame, canvas = _decode(pair['input'])
    changed = copy.deepcopy(book)
    for ins in changed.values():
        ins['turn'] = (ins['turn'] + 1) % 4
    edited = np.concatenate([_legend(tile, changed, bg, frame, canvas.shape[1], list(changed)), canvas])
    expected = _render(canvas.shape, tile, bg,
                       _forward_hits(canvas.shape, len(tile), _members(canvas, set(book), bg)[0], changed))
    assert np.array_equal(solve(edited), expected)
    assert not np.array_equal(expected, pair['output'])
    # Independent parity truth table at one tile: zero blank, one swap, two raw,
    # three swap, four raw. This also checks that zero and positive even differ.
    t = np.array([[2, 2], [2, 7]])
    for count in range(5):
        result = _render((2, 2), t, 0, np.array([[count]]))
        target = np.zeros((2, 2), int) if count == 0 else (9 - t if count % 2 else t)
        assert np.array_equal(result, target)
    return {'seeds': 42, 'pairs': 210, 'shapes': len(shapes),
            'marker_structures': len(structures), 'stencils': len(stencils),
            'tile_patterns': len(tile_patterns), 'live_counts': sorted(live_counts),
            'max_overlap': max(hits_seen), 'partial_tile_fields': partial}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
