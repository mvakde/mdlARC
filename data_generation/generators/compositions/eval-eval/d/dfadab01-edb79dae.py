"""Cancellation printing: component-qualified anchors, local stencils, odd ink.

Only standard-library modules and NumPy are used. No files, saved examples, seed
state, or training labels are consulted by solve(). See task_spec.md for the
complete input grammar and the intentionally restricted ink-overlap convention.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask, diagonal=True):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    neighbors = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)
                 if (a or b) and (diagonal or not (a and b))]
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in neighbors:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _box(cells):
    a = np.asarray(cells)
    return (int(a[:, 0].min()), int(a[:, 0].max()) + 1,
            int(a[:, 1].min()), int(a[:, 1].max()) + 1)


def _frame(a, bg):
    candidates = []
    for color in np.unique(a):
        if color == bg:
            continue
        points = np.argwhere(a == color)
        r, e, c, d = _box(points)
        if e - r < 7 or d - c < 7:
            continue
        if (len(points) == 2 * (e - r + d - c) - 4
                and np.all(a[r, c:d] == color) and np.all(a[e-1, c:d] == color)
                and np.all(a[r:e, c] == color) and np.all(a[r:e, d-1] == color)):
            candidates.append(((r, e, c, d), int(color)))
    assert len(candidates) == 1, 'one exclusive rectangular frame is required'
    return candidates[0]


def _read(input_grid):
    """Parse visible clues and input components, never an output or latent record."""
    a = _grid(input_grid)
    colors, amounts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(amounts)])
    assert np.count_nonzero(a == bg) > a.size // 2, 'background must be a strict majority'
    frame, border = _frame(a, bg)
    r, e, c, d = frame
    outside = a != bg
    outside[r:e, c:d] = False
    consumed = np.zeros(a.shape, bool)
    mapping, key_boxes = {}, {}
    # Fixed visible key grammar: source in the top-left, ink in the other cells.
    for cells in _components(outside):
        values = {int(a[q]) for q in cells}
        if len(values) == 1:
            continue
        x, y, u, v = _box(cells)
        tile = a[x:y, u:v]
        assert tile.shape == (2, 2) and len(cells) == 4
        source, ink = int(tile[0, 0]), int(tile[1, 1])
        assert source != ink and tile[0, 1] == tile[1, 0] == ink
        assert source not in mapping
        mapping[source], key_boxes[source] = ink, (x, y, u, v)
        consumed[x:y, u:v] = True
    assert 1 <= len(mapping) <= 3
    codes, inks = set(mapping), set(mapping.values())
    assert codes.isdisjoint(inks) and {bg, border}.isdisjoint(codes | inks)
    remaining = outside & ~consumed
    assert set(map(int, np.unique(a[remaining]))) == codes
    masks, stencil_boxes = {}, {}
    for source in sorted(codes):
        cells = np.argwhere(remaining & (a == source))
        x, y, u, v = _box(cells)
        assert y - x == v - u and 3 <= y - x <= 6
        assert np.all(np.isin(a[x:y, u:v], [bg, source]))
        masks[source] = (a[x:y, u:v] == source).copy()
        stencil_boxes[source] = (x, y, u, v)
        assert y <= r or x >= e or v <= c or u >= d
    assert len({m.shape for m in masks.values()}) == 1
    boxes = list(key_boxes.values()) + list(stencil_boxes.values())
    for i, (x, y, u, v) in enumerate(boxes):
        assert y <= r or x >= e or v <= c or u >= d
        for xx, yy, uu, vv in boxes[i+1:]:
            assert y < xx or yy < x or v < uu or vv < u, 'clue rectangles need a blank gap'
    sheet = a[r+1:e-1, c+1:d-1]
    assert set(map(int, np.unique(sheet))) <= {bg} | codes | inks
    live, rejected, painted, bare = [], [], [], []
    for cells in _components(sheet != bg):
        points = [(x, y, int(sheet[x, y])) for x, y in cells if int(sheet[x, y]) in codes]
        x, _, u, _ = _box(cells)
        if len(points) == 1 and points[0][:2] == (x, u):
            live.append(points[0])
            if len(cells) > 1:
                painted.append(points[0])
        elif points:
            rejected.extend(points)
        else:
            bare.append(cells)
    assert live
    return dict(grid=a, background=bg, frame=border, frame_box=frame, sheet=sheet,
                mapping=mapping, masks=masks, key_boxes=key_boxes,
                stencil_boxes=stencil_boxes, live=live, rejected=rejected,
                painted=painted, bare=bare, n=next(iter(masks.values())).shape[0])


def _hits(shape, records, masks, mapping):
    counts = {ink: np.zeros(shape, dtype=np.int16) for ink in set(mapping.values())}
    for r, c, code in records:
        mask = masks[code]
        h, w = min(mask.shape[0], shape[0] - r), min(mask.shape[1], shape[1] - c)
        if h > 0 and w > 0:
            counts[mapping[code]][r:r+h, c:c+w] += mask[:h, :w]
    return counts


def _render(shape, bg, border, records, masks, mapping, parity=True):
    counts = _hits((shape[0]-2, shape[1]-2), records, masks, mapping)
    # Different inks have disjoint active footprints, including even hits.
    assert np.all(sum((x > 0).astype(int) for x in counts.values()) <= 1), 'different inks overlap'
    out = np.full(shape, bg, dtype=np.int8)
    out[0, :] = out[-1, :] = border
    out[:, 0] = out[:, -1] = border
    for ink, hits in counts.items():
        out[1:-1, 1:-1][hits % 2 == 1 if parity else hits > 0] = ink
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only prediction for every member of the specified family."""
    a = _read(input_grid)
    r, e, c, d = a['frame_box']
    return _render((e-r, d-c), a['background'], a['frame'],
                   a['live'], a['masks'], a['mapping'])


def _mask(rng, n):
    y, x = np.indices((n, n))
    kind = int(rng.integers(8))
    if kind == 0:
        m = (x == 0) | (y == 0) | (x == n-1) | (y == n-1)
    elif kind == 1:
        m = (x == y) | (x + y == n-1)
    elif kind == 2:
        m = (x + y) % 2 == 0
    elif kind == 3:
        m = y % 2 == 0
    elif kind == 4:
        m = (x == n//2) | (y == n//2)
    elif kind == 5:
        m = (x == 0) | (y == n-1) | ((y == 1) & (x < n-1))
    else:
        m = rng.random((n, n)) < rng.uniform(.35, .82)
    for edge in (m[0], m[-1], m[:, 0], m[:, -1]):
        if not edge.any():
            edge[int(rng.integers(n))] = True
    return np.rot90(m, int(rng.integers(4))).copy()


def _available(blocked, h, w):
    if h > blocked.shape[0] or w > blocked.shape[1]:
        return np.empty((0, 2), int)
    s = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return np.argwhere(s[h:, w:] - s[:-h, w:] - s[h:, :-w] + s[:-h, :-w] == 0)


def _reserve(blocked, r, c, h, w):
    blocked[max(0, r-1):r+h+1, max(0, c-1):c+w+1] = True


def _pack_units(rng, units, rows, H, W):
    """Try arbitrary placement; a two-shelf construction guarantees completion."""
    for _ in range(10):
        blocked = np.zeros((H-3, W), bool)  # bottom two rows plus one gap
        spots = {}
        order = sorted(range(len(units)), key=lambda i: -units[i][1]*units[i][2] + rng.random())
        for i in order:
            _, h, w, _ = units[i]
            choices = _available(blocked, h, w)
            if not len(choices):
                break
            r, c = map(int, choices[int(rng.integers(len(choices)))])
            spots[i] = (r, c)
            _reserve(blocked, r, c, h, w)
        if len(spots) == len(units):
            return spots
    spots, top = {}, 0
    for row in rows:
        row = list(row)
        rng.shuffle(row)
        row_h = max(units[i][1] for i in row)
        base_w = sum(units[i][2] for i in row) + len(row) - 1
        gaps = rng.multinomial(W-base_w, np.full(len(row)+1, 1/(len(row)+1)))
        left = int(gaps[0])
        for j, i in enumerate(row):
            _, h, w, _ = units[i]
            spots[i] = (top + int(rng.integers(row_h-h+1)), left)
            left += w + 1 + int(gaps[j+1])
        top += row_h + 1
    assert top <= H-2
    return spots


def _legend_positions(rng, bh, bw, n, codes):
    # Arbitrary separated clues around a freely placed frame are preferred.
    for _ in range(20):
        fr, fc = int(rng.integers(31-bh)), int(rng.integers(31-bw))
        blocked = np.zeros((30, 30), bool)
        _reserve(blocked, fr, fc, bh, bw)
        items = [('stencil', code, n, n) for code in codes] + [('key', code, 2, 2) for code in codes]
        rng.shuffle(items)
        spots = {}
        for kind, code, h, w in items:
            choices = _available(blocked, h, w)
            if not len(choices):
                break
            r, c = map(int, choices[int(rng.integers(len(choices)))])
            spots[kind, code] = (r, c)
            _reserve(blocked, r, c, h, w)
        if len(spots) == len(items):
            return fr, fc, spots
    # A strip holds <=3 square masks followed by <=3 independently ordered keys.
    horizontal_ok, vertical_ok = bh+n+1 <= 30, bw+n+1 <= 30
    sides = ([0, 2] if horizontal_ok else []) + ([1, 3] if vertical_ok else [])
    assert sides
    side = int(rng.choice(sides))  # top, left, bottom, right
    length = len(codes)*(n+1)-1 + 1 + len(codes)*3-1
    assert length <= 29
    along = int(rng.integers(31-length))
    if side in (0, 2):
        fr = n+1 if side == 0 else 0
        fc = int(rng.integers(31-bw))
        cross = 0 if side == 0 else bh+1
    else:
        fc = n+1 if side == 1 else 0
        fr = int(rng.integers(31-bh))
        cross = 0 if side == 1 else bw+1
    spots = {}
    for kind, step in (('stencil', n+1), ('key', 3)):
        for code in rng.permutation(codes):
            jitter = int(rng.integers(n-1)) if kind == 'key' else 0
            spots[kind, int(code)] = ((cross+jitter, along) if side in (0, 2)
                                     else (along, cross+jitter))
            along += step
    return fr, fc, spots


def _make_pair(rng, n, count, two_dimensional, witness, transpose):
    k = int(rng.integers(2, 4)) if witness in (0, 1) else int(rng.integers(1, 4))
    palette = list(map(int, rng.permutation(10)))
    bg, border = palette[:2]
    codes, possible_inks = palette[2:2+k], palette[2+k:2+2*k]
    if witness == 0:
        mapping = dict(zip(codes, possible_inks))
    elif witness == 1:
        mapping = {code: possible_inks[0] for code in codes}
    else:
        group_count = int(rng.integers(1, k+1))
        mapping = {code: possible_inks[int(rng.integers(group_count))] for code in codes}
    group = [code for code in codes if mapping[code] == mapping[codes[0]]]
    assigned = [group[i % len(group)] for i in range(count)]
    rng.shuffle(assigned)
    if count == 2:
        origins = [(0, 0), (0, 2)]
    elif count == 3 and not two_dimensional:
        origins = [(0, 0), (0, 2), (0, 4)]
    elif count == 3:
        origins = [(0, 0), (0, 2), (2, 0)]
    else:
        origins = [(r, c) for r in (0, 2) for c in (0, 2, 4)][:count]
    # Transposing the cluster alone changes spatial arrangements without changing
    # the upper-left anchoring convention or rotating a key glyph.
    if max(c for _, c in origins) <= 2 and rng.random() < .5:
        origins = [(c, r) for r, c in origins]
    # Force an explicit count-way common hit; sample until count-two cancellation
    # and a surviving pixel are both present. The loop is small and bounded.
    common = (max(r for r, _ in origins), max(c for _, c in origins))
    assert max(common) < n
    cluster_records = [(r, c, code) for (r, c), code in zip(origins, assigned)]
    for _ in range(48):
        masks = {code: _mask(rng, n) for code in codes}
        for (r, c), code in zip(origins, assigned):
            masks[code][common[0]-r, common[1]-c] = True
        ch, cw = n+common[0], n+common[1]
        if witness == 1:
            # Two different codes with the same ink must cancel visibly. Choose
            # a cell inside exactly two of the three instruction windows.
            candidates = []
            for r in range(ch):
                for c in range(cw):
                    covering = [(rr, cc, code) for rr, cc, code in cluster_records
                                if rr <= r < rr+n and cc <= c < cc+n]
                    if len(covering) == 2 and covering[0][2] != covering[1][2]:
                        candidates.append((r, c, covering))
            assert candidates
            r, c, covering = candidates[0]
            for rr, cc, code in covering:
                masks[code][r-rr, c-cc] = True
        counts = _hits((ch, cw), cluster_records, masks, mapping)[mapping[codes[0]]]
        if np.any(counts == 2) and np.any(counts % 2 == 1) and any(not m.all() for m in masks.values()):
            break
    else:
        raise AssertionError('bounded stencil sampling failed')
    # Units reserve full output windows, not just sparse input points. An entire
    # hypothetical rejected-anchor stamp is also reserved, making the A ablation
    # a legal, non-conflicting alternative with a visibly different answer.
    units = [('cluster', ch, cw, None), ('single', n, n, codes[0]),
             ('bare', n, n, codes[0])]
    row0 = [0, 1, 2]
    row1 = []
    for code in codes[1:]:
        row1.append(len(units))
        units.append(('single', n, n, code))
    row1.append(len(units))
    units.append(('dead', n+2, n+2, codes[0]))
    rows = [row0, row1]
    min_h = sum(max(units[i][1] for i in row) for row in rows) + 6
    min_w = max(sum(units[i][2] for i in row)+len(row)-1 for row in rows) + 2
    bh = min_h + int(rng.integers(max(1, 24-min_h)))
    bw = min_w + int(rng.integers(max(1, 29-min_w)))
    # Leave a guaranteed exterior strip on at least one axis.
    if min(bh, bw)+n+1 > 30:
        bh = min_h
    H, W = bh-2, bw-2
    spots = _pack_units(rng, units, rows, H, W)
    sheet = np.full((H, W), bg, np.int8)
    records = []
    for i, (kind, _, _, code) in enumerate(units):
        r, c = spots[i]
        if kind == 'cluster':
            for dr, dc, source in cluster_records:
                sheet[r+dr, c+dc] = source
                records.append((r+dr, c+dc, source))
        elif kind == 'single':
            sheet[r, c] = code
            records.append((r, c, code))
            if i == 1 and (witness in (0, 1) or rng.random() < .65):
                # The carried ink has no authority of its own.
                sheet[r:r+2, c+1:c+3] = mapping[code]
        elif kind == 'bare':
            sheet[r:r+n, c:c+n][masks[code]] = mapping[code]
        else:
            sheet[r:r+2, c:c+2] = mapping[code]
            sheet[r+2, c+2] = code
    # One bottom-clipped and one corner-clipped live instruction. Their visible
    # windows have a gap to all other windows and to one another.
    first_c = int(rng.integers(W-n-2))
    for r, c, code in ((H-2, first_c, int(rng.choice(codes))),
                       (H-2, W-2, int(rng.choice(codes)))):
        sheet[r, c] = code
        records.append((r, c, code))
    out = _render((bh, bw), bg, border, records, masks, mapping)
    fr, fc, legend = _legend_positions(rng, bh, bw, n, codes)
    inp = np.full((30, 30), bg, np.int8)
    inp[fr:fr+bh, fc:fc+bw] = bg
    inp[fr, fc:fc+bw] = inp[fr+bh-1, fc:fc+bw] = border
    inp[fr:fr+bh, fc] = inp[fr:fr+bh, fc+bw-1] = border
    inp[fr+1:fr+bh-1, fc+1:fc+bw-1] = sheet
    for code in codes:
        r, c = legend['stencil', code]
        inp[r:r+n, c:c+n][masks[code]] = code
        r, c = legend['key', code]
        inp[r:r+2, c:c+2] = mapping[code]
        inp[r, c] = code
    r, e, c, d = _box(np.argwhere(inp != bg))
    r, c = max(0, r-int(rng.integers(3))), max(0, c-int(rng.integers(3)))
    e, d = min(30, e+int(rng.integers(3))), min(30, d+int(rng.integers(3)))
    inp = inp[r:e, c:d].copy()
    if transpose:
        inp, out = inp.T.copy(), out.T.copy()
    if np.count_nonzero(inp == bg) <= inp.size // 2:
        return None
    analysis = _read(inp)
    assert sorted(analysis['live']) == sorted((c, r, code) if transpose else (r, c, code)
                                               for r, c, code in records)
    assert np.array_equal(solve(inp), out), 'construction and visible reconstruction disagree'
    return {'input': inp, 'output': out}


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < low or (high is not None and value > high)):
        raise ValueError(f'illegal {name}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    nt = _integer(num_train, 'num_train', 2, 10)
    nq = _integer(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    mode = str(rng.choice(['scale', 'multiplicity', 'geometry']))
    witness_order = [0, 1] + [-1]*(nt-2)
    rng.shuffle(witness_order)
    # Opposite transpose choices expose both axial boundary intersections even
    # at the minimum legal training count.
    witness_transpose = {0: bool(rng.integers(2))}
    witness_transpose[1] = not witness_transpose[0]
    puzzle, seen = {'train': [], 'test': []}, set()
    for i in range(nt+nq):
        training = i < nt
        witness = witness_order[i] if training else -1
        if training:
            n = int(rng.choice([3, 4] if mode == 'scale' else [5, 6]))
            count = 2 if witness == 0 else 3 if witness == 1 else int(rng.integers(2, 4))
            two_d = mode != 'geometry'
        else:
            n = int(rng.choice([5, 6]))
            count = int(rng.integers(4, 7)) if mode == 'multiplicity' else 3
            two_d = True
        transpose = witness_transpose[witness] if witness in (0, 1) else bool(rng.integers(2))
        for _ in range(24):
            pair = _make_pair(rng, n, count, two_d, witness, transpose)
            if pair is not None and _key(pair['input']) not in seen:
                break
        else:
            raise AssertionError('could not construct a distinct majority-background pair')
        seen.add(_key(pair['input']))
        puzzle['train' if training else 'test'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _facts(pair):
    a = _read(pair['input'])
    counts = _hits(a['sheet'].shape, a['live'], a['masks'], a['mapping'])
    summed = sum(counts.values())
    n, H, W = a['n'], *a['sheet'].shape
    remaining, groups = set(range(len(a['live']))), []
    while remaining:
        group = {min(remaining)}
        remaining -= group
        todo = list(group)
        while todo:
            i = todo.pop()
            r, c, _ = a['live'][i]
            adjacent = {j for j in remaining
                        if max(r, a['live'][j][0]) < min(r+n, a['live'][j][0]+n, H)
                        and max(c, a['live'][j][1]) < min(c+n, a['live'][j][1]+n, W)}
            group |= adjacent
            remaining -= adjacent
            todo.extend(adjacent)
        groups.append(group)
    cluster = max(groups, key=len)
    collinear = len({a['live'][i][0] for i in cluster}) == 1 or len({a['live'][i][1] for i in cluster}) == 1
    by_source = {code: _hits(a['sheet'].shape, [p for p in a['live'] if p[2] == code],
                            a['masks'], a['mapping'])[a['mapping'][code]] for code in a['mapping']}
    cross_source = bool(np.any(sum((x > 0).astype(int) for x in by_source.values()) >= 2))
    cross_cancellation = any(np.any(np.logical_or.reduce(
        [by_source[code] % 2 == 1 for code in a['mapping'] if a['mapping'][code] == ink]) != (hits % 2 == 1))
        for ink, hits in counts.items())
    a['facts'] = dict(n=n, categories=len(a['mapping']), inks=len(set(a['mapping'].values())),
                      anchors=len(a['live']), maximum=int(summed.max()),
                      cluster_size=len(cluster), collinear=collinear, cross_source=cross_source,
                      cross_cancellation=cross_cancellation,
                      two=int(np.count_nonzero(summed == 2)), three=int(np.count_nonzero(summed == 3)),
                      bottom=any(r+n > H for r, c, code in a['live']),
                      right=any(c+n > W for r, c, code in a['live']),
                      corner=any(r+n > H and c+n > W for r, c, code in a['live']),
                      painted=len(a['painted']), rejected=len(a['rejected']), bare=len(a['bare']),
                      disconnected=sum(len(_components(m, False)) > 1 for m in a['masks'].values()),
                      input_shape=list(np.asarray(pair['input']).shape),
                      output_shape=list(np.asarray(pair['output']).shape))
    return a


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            a = _facts(pair)
            assert np.array_equal(solve(inp.copy()), out), 'incorrect output from input-only reconstruction'
            assert a['rejected'] and a['bare']
            assert {code for r, c, code in a['live']} == set(a['mapping'])
            assert a['facts']['two'] > 0, 'must expose a cancellation'
            # All original source-colored points have a visibly meaningful role.
            # Without A, accepting the one rejected point adds a nonempty stamp.
            r, e, c, d = a['frame_box']
            all_points = a['live'] + a['rejected']
            without_A = _render((e-r, d-c), a['background'], a['frame'],
                                all_points, a['masks'], a['mapping'])
            assert not np.array_equal(out, without_A), 'A must change the answer'
            solid_masks = {code: np.ones_like(mask) for code, mask in a['masks'].items()}
            without_B = _render((e-r, d-c), a['background'], a['frame'],
                                a['live'], solid_masks, a['mapping'])
            assert not np.array_equal(out, without_B), 'B stencil geometry must change the answer'
            union = _render((e-r, d-c), a['background'], a['frame'],
                            a['live'], a['masks'], a['mapping'], parity=False)
            assert not np.array_equal(out, union), 'union shortcut must fail'
            # At least one complete unoverlapped occurrence teaches every entry.
            counts = _hits(a['sheet'].shape, a['live'], a['masks'], a['mapping'])
            for code, mask in a['masks'].items():
                ink = a['mapping'][code]
                assert any(r+a['n'] <= a['sheet'].shape[0] and c+a['n'] <= a['sheet'].shape[1]
                           and np.all(counts[ink][r:r+a['n'], c:c+a['n']][mask] == 1)
                           for r, c, source in a['live'] if source == code), 'missing clear stencil occurrence'
            analyses[split].append(a)
    tr, te = analyses['train'], analyses['test']
    assert any(a['facts']['three'] for a in tr), 'triple overlap must distinguish parity from exactly-one'
    assert any(a['painted'] for a in tr), 'teach that a correctly anchored composite is live'
    assert any(len(set(a['mapping'].values())) < len(a['mapping']) for a in tr), 'teach shared ink cancellation'
    assert any(a['facts']['cross_cancellation'] for a in tr), 'shared ink must cancel across source categories'
    assert any(len(set(a['mapping'].values())) > 1 for a in tr), 'teach color-key reading'
    assert any(a['facts']['bottom'] for a in tr) and any(a['facts']['right'] for a in tr)
    assert all(a['facts']['corner'] for a in tr+te)
    assert (max(a['n'] for a in tr) < min(a['n'] for a in te)
            or max(a['facts']['maximum'] for a in tr) < min(a['facts']['maximum'] for a in te)
            or (all(a['facts']['collinear'] for a in tr) and all(not a['facts']['collinear'] for a in te))), 'missing structural train/test shift'


def _puzzle_digest(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _manual_regression():
    # Hand-computed one-color 3x3 solid impressions at (0,0), (0,2), (2,0).
    # A two-hit pixel vanishes, the shared three-hit pixel stays, and the frame
    # is never overwritten. This checks the law independently of sampling.
    m = {2: np.ones((3, 3), bool)}
    records = [(0, 0, 2), (0, 2, 2), (2, 0, 2)]
    actual = _render((7, 7), 0, 9, records, m, {2: 4})
    expected = np.array([[int(c) for c in row] for row in
                         ['9999999','9440449','9440449','9004449','9444009','9444009','9999999']], np.int8)
    assert np.array_equal(actual, expected)


def self_test():
    """42 seeds, all count combinations, bounded mutation and symmetry checks.

    Runs sequentially, retains only feature sets/hashes, writes no files, and
    holds at most two puzzles plus a small transformed copy at once.
    """
    _manual_regression()
    signatures, shapes, output_shapes, masks, normalized, maxima, sizes, categories = (set() for _ in range(8))
    feature_totals, palette_roles, pairs = Counter(), {name: set() for name in ('background', 'frame', 'code', 'ink')}, 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(_puzzle_digest(p))
        for split in ('train', 'test'):
            for pair in p[split]:
                a = _facts(pair)
                f = a['facts']
                pairs += 1
                shapes.add(tuple(f['input_shape']))
                output_shapes.add(tuple(f['output_shape']))
                sizes.add(f['n'])
                categories.add(f['categories'])
                maxima.add(f['maximum'])
                for name in ('two', 'three', 'painted', 'rejected', 'bare', 'disconnected', 'bottom', 'right', 'corner', 'cross_source', 'cross_cancellation'):
                    feature_totals[name] += bool(f[name])
                masks.update((m.shape, m.tobytes()) for m in a['masks'].values())
                palette_roles['background'].add(a['background'])
                palette_roles['frame'].add(a['frame'])
                palette_roles['code'].update(a['mapping'])
                palette_roles['ink'].update(a['mapping'].values())
                roles = {}
                norm = bytes(roles.setdefault(int(x), len(roles)) for x in pair['input'].ravel())
                normalized.add(hashlib.sha256(bytes(pair['input'].shape)+norm).hexdigest())
        if seed < 6:
            assert _puzzle_digest(p) == _puzzle_digest(generate_puzzle(seed))
            q = p['test'][0]
            inp, out = q['input'], q['output']
            original = inp.copy()
            assert np.array_equal(solve(inp), out) and np.array_equal(inp, original)
            assert np.array_equal(solve(inp.T), out.T)
            perm = np.random.default_rng(seed+970).permutation(10)
            assert np.array_equal(solve(perm[inp]), perm[out])
            a = _read(inp)
            # Move a complete key, preserving its colors and shape. Association
            # must be by source identity rather than proximity or bank order.
            blocked = np.zeros(inp.shape, bool)
            for box in [a['frame_box'], *a['key_boxes'].values(), *a['stencil_boxes'].values()]:
                rr, ee, cc, dd = box
                _reserve(blocked, rr, cc, ee-rr, dd-cc)
            choices = _available(blocked, 2, 2)
            assert len(choices)
            rr, cc = map(int, choices[0])
            x, y, u, v = next(iter(a['key_boxes'].values()))
            moved = inp.copy()
            tile = moved[x:y, u:v].copy()
            moved[x:y, u:v] = a['background']
            moved[rr:rr+2, cc:cc+2] = tile
            assert np.array_equal(solve(moved), out)
            # Moving the rejected code to its component's top-left makes a live
            # trigger. Its reserved output window guarantees a visible change.
            rr, cc, code = a['rejected'][0]
            r, e, c, d = a['frame_box']
            changed = inp.copy()
            changed[r+1+rr, c+1+cc] = a['background']
            changed[r+rr-1, c+cc-1] = code
            assert not np.array_equal(solve(changed), out)
            bad = {s: [{'input': pair['input'], 'output': pair['output'].copy()} for pair in p[s]]
                   for s in ('train', 'test')}
            bad['test'][0]['output'][1, 1] = (int(bad['test'][0]['output'][1, 1])+1) % 10
            try:
                validate_generated_puzzle(bad)
            except AssertionError:
                pass
            else:
                raise AssertionError('corrupted output accepted')
    assert len(signatures) == 42 and len(normalized) >= 180
    assert len(shapes) >= 20 and len(output_shapes) >= 60 and len(masks) >= 100
    assert sizes == {3, 4, 5, 6} and categories == {1, 2, 3}
    assert {2, 3, 4, 5, 6} <= maxima
    assert all(len(values) == 10 for values in palette_roles.values())
    assert feature_totals['disconnected'] >= 20
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(44000+nt*10+nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,True),
                   (2.0,1),(4,1.0),(None,1),(4,None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return {'seeds':42,'pairs':pairs,'puzzles':len(signatures),'input_shapes':len(shapes),
            'output_shapes':len(output_shapes),
            'color_normalized_inputs':len(normalized),'stencils':len(masks),
            'sizes':sorted(sizes),'categories':sorted(categories),'maximum_hit_counts':sorted(maxima),
            'feature_pairs':dict(feature_totals),'palette_role_coverage':{k:len(v) for k,v in palette_roles.items()}}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
