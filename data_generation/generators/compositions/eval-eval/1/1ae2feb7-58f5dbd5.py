"""Periodic, color-selected negative stencil impressions on bounded strips.

Only NumPy and the standard library are used. The constructor paints opaque
impressions; solve reconstructs input geometry and evaluates each interior cell.
No parent code, seed, or puzzle-level sampling parameter is used by solve.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


RIVALS = ('no_repeat', 'union', 'nearest_only', 'far', 'short', 'long',
          'copy', 'stripe', 'first_stencil', 'flip_rows', 'flip_cols',
          'rotate_180', 'gap_phase', 'launch_priority')


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError('expected an integer ARC grid, dimensions 1..30, colors 0..9')
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def _fingerprint(puzzle):
    return hashlib.sha256(b''.join(bytes(np.asarray(p[r]).shape)
        + np.asarray(p[r], dtype=np.uint8).tobytes()
        for s in ('train', 'test') for p in puzzle[s]
        for r in ('input', 'output'))).hexdigest()


def _decode_canonical(a, bg, marker):
    """Recognize a rightward scene after a candidate whole-grid quarter-turn."""
    yy, xx = np.nonzero(a == marker)
    top, bottom, sep = int(yy.min()), int(yy.max()) + 1, int(xx[0])
    assert len(set(map(int, xx))) == 1 and len(yy) == bottom - top
    assert bottom - top >= 4 and sep >= 2 and sep + 5 <= a.shape[1]
    assert bg != marker
    # All non-background on the destination side in this band belongs to strips.
    strips = []
    y = top
    while y < bottom:
        cols = np.flatnonzero(a[y, sep + 1:] != bg) + sep + 1
        if not len(cols):
            y += 1
            continue
        start = y
        while y < bottom and np.any(a[y, sep + 1:] != bg):
            y += 1
        assert y - start >= 4
        occupied = a[start:y, sep + 1:] != bg
        nonzero = np.argwhere(occupied)
        x0, x1 = int(nonzero[:, 1].min()) + sep + 1, int(nonzero[:, 1].max()) + sep + 2
        assert x0 == sep + 2 and x1 - x0 >= 4
        color = int(a[start, x0])
        assert color not in (bg, marker)
        assert np.all(a[start:y, x0:x1] == color)
        assert occupied.sum() == (y - start) * (x1 - x0)
        strips.append({'y': start, 'x': x0, 'height': y - start,
                       'width': x1 - x0 - 2, 'material': color})
    assert strips and strips[0]['y'] == top
    assert strips[-1]['y'] + strips[-1]['height'] == bottom
    assert len({s['height'] for s in strips}) == 1
    assert len({s['material'] for s in strips}) == 1
    k = strips[0]['height'] - 2
    assert 2 <= k <= 4 and 1 <= len(strips) <= 5
    material = strips[0]['material']
    allowed = np.zeros(a.shape, dtype=bool)
    allowed[top:bottom, sep] = True
    for s in strips:
        y, x = s['y'], s['x']
        allowed[y:y + k + 2, x:x + s['width'] + 2] = True
        cursor, blocks = sep - 1, []
        while cursor >= 0:
            color = int(a[y + 1, cursor])
            if color == bg:
                cursor -= 1
                continue
            assert color not in (marker, material)
            near, end = sep - cursor, cursor + 1
            while cursor >= 0 and a[y + 1, cursor] == color:
                cursor -= 1
            blocks.append({'color': color, 'period': end - cursor - 1,
                           'proximity': near})
            allowed[y + 1, cursor + 1:end] = True
        assert 1 <= len(blocks) <= 4
        s['blocks'] = blocks  # Already ordered from nearest to farthest.
    # No unregistered foreground is allowed in the transport/program band.
    assert np.all(a[top:bottom][~allowed[top:bottom]] == bg)
    external = ~allowed & (a != bg)
    assert not np.any(external[top:bottom])
    masks, boxes = {}, {}
    for color in sorted(int(c) for c in np.unique(a[external])):
        assert color not in (bg, marker, material)
        cells = np.argwhere(external & (a == color))
        y0, x0 = map(int, cells.min(axis=0))
        y1, x1 = map(int, cells.max(axis=0) + 1)
        assert y1 <= top or y0 >= bottom
        assert y1 - y0 == k and 2 <= x1 - x0 <= 4
        region = a[y0:y1, x0:x1]
        assert np.all((region == bg) | (region == color))
        m = region == color
        assert 0 < int(m.sum()) < m.size
        assert m[0].any() and m[-1].any() and m[:, 0].any() and m[:, -1].any()
        masks[color], boxes[color] = m, (y0, x0, y1, x1)
        allowed[y0:y1, x0:x1] |= m
    assert masks and len({m.shape for m in masks.values()}) == 1
    q = next(iter(masks.values())).shape[1]
    for i, b in enumerate(boxes.values()):
        for c in list(boxes.values())[i + 1:]:
            assert b[2] < c[0] or c[2] < b[0] or b[3] < c[1] or c[3] < b[1]
    used = {b['color'] for s in strips for b in s['blocks']}
    assert used <= set(masks)
    assert len(set(masks) - used) <= 2
    assert all(q <= b['period'] <= 8 for s in strips for b in s['blocks'])
    assert np.all(a[~allowed] == bg)
    # All material pixels have to be in solid strips, never the tool bank.
    assert int(np.count_nonzero(a == material)) == sum((k + 2) * (s['width'] + 2) for s in strips)
    x0, x1 = sep + 1, max(s['x'] + s['width'] + 2 for s in strips) + 1
    crop = (top - 1, x0, bottom + 1, x1)
    assert crop[0] >= 0 and crop[2] <= a.shape[0] and crop[3] <= a.shape[1]
    return {'canonical': a, 'background': bg, 'marker': marker, 'material': material,
            'separator': sep, 'strips': strips, 'k': k, 'q': q, 'masks': masks,
            'boxes': boxes, 'crop': crop, 'distractors': len(set(masks) - used)}


def _decode(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    assert all(np.all(edge == bg) for edge in (a[0], a[-1], a[:, 0], a[:, -1])), 'missing background border'
    candidates = []
    for rotation in range(4):
        canonical = np.rot90(a, rotation)
        for color in map(int, np.unique(canonical)):
            if color == bg:
                continue
            yy, xx = np.nonzero(canonical == color)
            # The reserved start-bar color occurs nowhere else.
            if len(yy) < 4 or int(xx.min()) != int(xx.max()) or len(yy) != int(yy.max() - yy.min() + 1):
                continue
            try:
                info = _decode_canonical(canonical, bg, color)
                info['rotation'] = rotation
                candidates.append(info)
            except (AssertionError, ValueError, IndexError):
                continue
    assert len(candidates) == 1, 'input must have one recognizable start bar and transport direction'
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode only the input, then choose the covering tool for every cell."""
    try:
        info = _decode(input_grid)
    except AssertionError as error:
        raise ValueError(str(error)) from error
    y0, x0, y1, x1 = info['crop']
    out = info['canonical'][y0:y1, x0:x1].copy()
    for s in info['strips']:
        for row in range(info['k']):
            for col in range(s['width']):
                # Since period >= footprint width, at most one copy of a given
                # run covers this cell. Blank cells still occupy the footprint.
                for b in s['blocks']:
                    local = col % b['period']
                    if local < info['q']:
                        if info['masks'][b['color']][row, local]:
                            out[s['y'] - y0 + 1 + row, s['x'] - x0 + 1 + col] = info['background']
                        break
    return np.rot90(out, -info['rotation']).copy()


def _paint_decoded(info, mode='correct'):
    """Audit hypotheses using whole-impression overwrites, not solve's formula."""
    y0, x0, y1, x1 = info['crop']
    out = info['canonical'][y0:y1, x0:x1].copy()
    bg, mat, q = info['background'], info['material'], info['q']
    first_mask = info['masks'][min(info['boxes'], key=info['boxes'].get)]
    for s in info['strips']:
        blocks = list(s['blocks'])
        if mode == 'nearest_only':
            blocks = blocks[:1]
        if mode == 'far':
            blocks.reverse()
        elif mode == 'short':
            blocks.sort(key=lambda b: b['period'])
        elif mode == 'long':
            blocks.sort(key=lambda b: -b['period'])
        stamps = []
        for rank, b in enumerate(blocks):
            period = b['period']
            start = b['proximity'] - 1 if mode == 'gap_phase' else 0
            anchors = [start] if mode == 'no_repeat' else range(start, s['width'], period)
            mask = first_mask if mode == 'first_stencil' else info['masks'][b['color']]
            if mode == 'flip_rows':
                mask = np.flipud(mask)
            elif mode == 'flip_cols':
                mask = np.fliplr(mask)
            elif mode == 'rotate_180':
                mask = np.rot90(mask, 2)
            for anchor in anchors:
                stamps.append((rank, anchor, mask))
        # Higher-priority blocks paint last, regardless of launch time.
        stamps.sort(key=(lambda t: (t[1], -t[0])) if mode == 'launch_priority'
                    else (lambda t: (-t[0], t[1])))
        for rank, anchor, mask in stamps:
            n = min(q, s['width'] - anchor)
            if n <= 0:
                continue
            yy, xx = s['y'] - y0 + 1, s['x'] - x0 + 1 + anchor
            window = out[yy:yy + info['k'], xx:xx + n]
            if mode == 'stripe':
                window[:, :1] = bg
            elif mode == 'union':
                window[mask[:, :n]] = bg
            else:
                window[:] = np.where(mask[:, :n], mat if mode == 'copy' else bg,
                                     bg if mode == 'copy' else mat)
    return np.rot90(out, -info['rotation']).copy()


def _transforms(mask):
    return [np.rot90(mask, r) for r in range(4)] + [np.rot90(np.fliplr(mask), r) for r in range(4)]


def _mask(rng, k, q):
    for _ in range(120):
        m = np.zeros((k, q), dtype=bool)
        motif = int(rng.integers(8))
        if motif == 0:
            for y in range(k):
                m[y, (y * (q - 1)) // (k - 1)] = True
        elif motif == 1:
            m[0, :] = m[-1, :] = True
        elif motif == 2:
            m[:, 0] = m[:, -1] = True
        elif motif == 3:
            m[int(rng.integers(k)), :] = True
            m[:, int(rng.integers(q))] = True
        elif motif == 4:
            m[:] = True
            m.flat[int(rng.integers(m.size))] = False
        elif motif == 5:
            m[0, 0] = m[-1, -1] = True
        else:
            m = rng.random((k, q)) < float(rng.uniform(.2, .8))
        if rng.random() < .5:
            m = np.flipud(m)
        if rng.random() < .5:
            m = np.fliplr(m)
        if (0 < int(m.sum()) < m.size and m[0].any() and m[-1].any()
                and m[:, 0].any() and m[:, -1].any()):
            return m.copy()
    m = np.zeros((k, q), dtype=bool)
    m[0, 0] = m[-1, -1] = True
    return m


def _witness_masks(rng, k, q):
    for _ in range(400):
        a, b = _mask(rng, k, q), _mask(rng, k, q)
        same_shape = [m for m in _transforms(a) if m.shape == a.shape]
        if (sum(np.array_equal(a, m) for m in same_shape) == 1
                and np.any(a & ~b) and np.any(b & ~a) and 2 * int(a.sum()) != a.size):
            return a, b
    # Explicit asymmetric, full-span rectangular/square fallback, k >= 3.
    a = np.zeros((k, q), dtype=bool)
    a[0, 0] = a[0, -1] = a[-1, 0] = a[1, -1] = True
    if 2 * int(a.sum()) == a.size:
        for y, x in np.argwhere(~a):
            changed = a.copy()
            changed[y, x] = True
            if sum(np.array_equal(changed, m) for m in _transforms(changed)
                   if m.shape == changed.shape) == 1:
                a = changed
                break
    assert 2 * int(a.sum()) != a.size
    assert sum(np.array_equal(a, m) for m in _transforms(a) if m.shape == a.shape) == 1
    # A half-turn preserves its full span and occupancy, but asymmetry ensures
    # that each of these equal-occupancy masks has ink absent from the other.
    return a, np.rot90(a, 2).copy()


def _choose(rng, items):
    return items[int(rng.integers(len(items)))]


def _scene(rng, k, q, programs, widths, turn, witness=False, gap_witness=False):
    """Construct directly from source programs and write negative impressions."""
    nlanes = len(programs)
    used_indices = sorted({c for lane in programs for p, c in lane})
    index_map = {c: i for i, c in enumerate(used_indices)}
    programs = [[(p, index_map[c]) for p, c in lane] for lane in programs]
    ntools = len(used_indices)
    max_registers = min(7, 29 // (q + 1))
    distractors = int(rng.integers(min(2, max_registers - ntools) + 1))
    colors = list(map(int, rng.choice(10, 3 + ntools + distractors, replace=False)))
    bg, marker, material = colors[:3]
    tool_colors = colors[3:]
    masks = [_mask(rng, k, q) for _ in tool_colors]
    if witness:
        masks[0], masks[1] = _witness_masks(rng, k, q)
    # Copying motifs in different colors is legal; matching still uses color.
    if not witness and len(masks) >= 2 and rng.random() < .18:
        masks[-1] = masks[0].copy()
    minimum_gaps = [[int(i > 0 and lane[i - 1][1] == c)
                     + int(gap_witness and i == 0)
                     for i, (p, c) in enumerate(lane)] for lane in programs]
    payload = max(sum(p for p, c in lane) + sum(gaps)
                  for lane, gaps in zip(programs, minimum_gaps))
    max_width = max(widths)
    budget = 24 - max_width  # source + interior + six border/control cells
    assert payload <= budget
    source = int(rng.integers(payload, min(budget, payload + 4) + 1))
    sep, strip_x = source + 1, source + 3
    # Vertical geometry reserves space for one bank, with optional banks on both sides.
    span_min = nlanes * (k + 2) + nlanes - 1
    spare = 30 - (span_min + k + 3)
    assert spare >= 0
    split_bank = len(tool_colors) >= 2 and spare >= k + 1 and rng.random() < .35
    above = int(rng.integers(1, len(tool_colors))) if split_bank else (
        len(tool_colors) if rng.random() < .5 else 0)
    below = len(tool_colors) - above
    banks = int(above > 0) + int(below > 0)
    vertical_budget = 30 - (span_min + banks * (k + 1) + 2)
    extra_gaps = [0] * max(0, nlanes - 1)
    for _ in range(int(rng.integers(min(vertical_budget, 2 * len(extra_gaps)) + 1))):
        extra_gaps[int(rng.integers(len(extra_gaps)))] += 1
    gaps = [1 + v for v in extra_gaps]
    span = nlanes * (k + 2) + sum(gaps)
    first_y = 1 + (k + 1 if above else 0)
    height = span + banks * (k + 1) + 2
    # Bank gaps vary independently, subject to the 30-cell canvas limit.
    groups = [n for n in (above, below) if n]
    bank_gap = int(rng.integers(1, min(3, min((28 - n * q) // max(1, n - 1)
                                              for n in groups)) + 1))
    bank_width = max(n * q + (n - 1) * bank_gap for n in groups)
    width = max(source + max_width + 6, bank_width + 2)
    inp = np.full((height, width), bg, dtype=np.int8)
    out = np.full((span + 2, max_width + 4), bg, dtype=np.int8)
    inp[first_y:first_y + span, sep] = marker
    y, oy = first_y, 1
    for lane, w, required, gap_after in zip(programs, widths, minimum_gaps, gaps + [0]):
        inp[y:y + k + 2, strip_x:strip_x + w + 2] = material
        out[oy:oy + k + 2, 1:w + 3] = material
        slack = source - sum(p for p, c in lane) - sum(required)
        # Keep source offsets visible; no offset is part of the stamping phase.
        additions = rng.multinomial(int(rng.integers(slack + 1)), [1 / len(lane)] * len(lane))
        cursor = sep
        for (period, ci), gap, extra in zip(lane, required, additions):
            cursor -= gap + int(extra)
            inp[y + 1, cursor - period:cursor] = tool_colors[ci]
            cursor -= period
        for period, ci in reversed(lane):
            for anchor in range(0, w, period):
                n = min(q, w - anchor)
                out[oy + 1:oy + 1 + k, 2 + anchor:2 + anchor + n] = np.where(
                    masks[ci][:, :n], bg, material)
        y += k + 2 + gap_after
        oy += k + 2 + gap_after
    permutation = list(map(int, rng.permutation(len(tool_colors))))
    groups_indices = [permutation[:above], permutation[above:]]
    bank_rows = [1, first_y + span + 1]
    for indices, bank_y in zip(groups_indices, bank_rows):
        if not indices:
            continue
        bw = len(indices) * q + (len(indices) - 1) * bank_gap
        bx = int(rng.integers(1, width - bw))
        for slot, ci in enumerate(indices):
            x = bx + slot * (q + bank_gap)
            inp[bank_y:bank_y + k, x:x + q][masks[ci]] = tool_colors[ci]
    # Optional asymmetric outer padding changes the canvas, never the output crop.
    pads = []
    for dim in inp.shape:
        total = int(rng.integers(min(3, 30 - dim) + 1))
        first = int(rng.integers(total + 1))
        pads.append((first, total - first))
    inp = np.pad(inp, pads, constant_values=bg)
    return {'input': np.rot90(inp, turn).copy(), 'output': np.rot90(out, turn).copy()}


def _random_pair(rng, max_width, turn, mode='ordinary', block_cap=4):
    k, q = int(rng.integers(2, 5)), int(rng.integers(2, 5))
    max_lanes = min(5, (28 - k) // (k + 3))
    nlanes = int(rng.integers(1, max_lanes + 1))
    if mode == 'longer':
        w = int(rng.integers(max_width + 1, min(22, max_width + 7) + 1))
    elif mode == 'clip':
        w = int(rng.integers(2, q + 2))
    elif mode == 'mixture':
        w = 12  # Three periods 3,4,5 leave each tool a visible impression.
    else:
        w = int(rng.integers(max(q + 2, 6), max_width + 1))
    budget = 24 - w
    q = min(q, budget)
    max_blocks = min(block_cap, budget // q)
    if mode == 'mixture':
        # Three keys genuinely compose; sampling uses q=2 to keep enough room.
        q, max_blocks = 2, min(4, budget // 2)
        nlanes = max(2, nlanes)
    nkeys = int(rng.integers(1, 5))
    programs = []
    for _ in range(nlanes):
        count = int(rng.integers(1, max_blocks + 1))
        if mode == 'mixture' and not programs:
            count, nkeys = 3, 3
        lane = []
        for i in range(count):
            room = budget - sum(p for p, c in lane) - (count - i - 1) * q
            # Reserve a potential gap so equal-color neighbors stay separate.
            color = int(rng.integers(nkeys))
            if lane and color == lane[-1][1]:
                available = [c for c in range(nkeys) if c != color]
                if available:
                    color = int(rng.choice(available))
                else:
                    room -= i
            period = int(rng.integers(q, min(8, room) + 1)) if room >= q else q
            lane.append((period, color))
        if mode == 'mixture' and not programs:
            lane = [(3, 0), (4, 1), (5, 2)]
        # Guarantee fit when one-key repeats need background separators.
        while sum(p for p, c in lane) + sum(lane[i][1] == lane[i - 1][1]
                                            for i in range(1, len(lane))) > budget:
            if len(lane) > 1:
                lane.pop()
            else:
                lane[0] = (q, lane[0][1])
        programs.append(lane)
    widths = [w] + [int(rng.integers(2 if mode == 'clip' else q + 1, w + 1))
                    for _ in range(nlanes - 1)]
    return _scene(rng, k, q, programs, widths, turn)


def _teaching_evidence(puzzle):
    infos = [_decode(p['input']) for p in puzzle['train']]
    rejected = {r for r in RIVALS if any(not np.array_equal(p['output'], _paint_decoded(i, r))
                for p, i in zip(puzzle['train'], infos))}
    singles, shifted, edges = set(), False, set()
    for p, info in zip(puzzle['train'], infos):
        for s in info['strips']:
            bs = s['blocks']
            if len(bs) == 1 and s['width'] >= bs[0]['period'] + info['q']:
                singles.add(bs[0]['period'])
                shifted |= bs[0]['proximity'] > 1
            if len(bs) >= 2:
                for far in bs[1:]:
                    near = bs[0]
                    if not np.array_equal(info['masks'][near['color']], info['masks'][far['color']]):
                        edges.add((near['color'], far['color']))
    return rejected, singles, shifted, any((b, a) in edges for a, b in edges)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen = set()
    orientations = set(range(8))
    non_half = False
    for split in ('train', 'test'):
        for p in puzzle[split]:
            assert isinstance(p, dict) and set(p) == {'input', 'output'}
            inp, out = _grid(p['input']), _grid(p['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            info = _decode(inp)
            assert np.array_equal(out, _paint_decoded(info)), 'incorrect footprint, subtraction, priority, or crop'
            assert np.array_equal(out, solve(inp)), 'cell-wise solve disagrees'
            canonical_out = np.rot90(out, info['rotation'])
            bg, mat = info['background'], info['material']
            assert set(map(int, np.unique(out))) <= {bg, mat}
            assert all(np.all(edge == bg) for edge in (canonical_out[0], canonical_out[-1],
                                                       canonical_out[:, 0], canonical_out[:, -1]))
            y0, x0, y1, x1 = info['crop']
            original = info['canonical'][y0:y1, x0:x1]
            # No hole outside interiors, and no new material outside original strips.
            interiors = np.zeros(original.shape, dtype=bool)
            for s in info['strips']:
                y, x = s['y'] - y0 + 1, s['x'] - x0 + 1
                interiors[y:y + info['k'], x:x + s['width']] = True
            assert np.array_equal(canonical_out[~interiors], original[~interiors])
            assert np.any(canonical_out[interiors] == bg), 'must perforate some material'
            if split == 'train':
                for s in info['strips']:
                    if len(s['blocks']) == 1 and s['width'] >= s['blocks'][0]['period'] + info['q']:
                        m = info['masks'][s['blocks'][0]['color']]
                        orientations &= {i for i, t in enumerate(_transforms(m))
                                         if t.shape == m.shape and np.array_equal(t, m)}
                        non_half |= 2 * int(m.sum()) != m.size
    rejected, singles, shifted, cycle = _teaching_evidence(puzzle)
    assert rejected == set(RIVALS), 'training admits rivals: ' + ','.join(sorted(set(RIVALS) - rejected))
    assert len(singles) >= 2 and shifted, 'missing numerical recurrence or gap-independent phase evidence'
    assert cycle, 'missing same-color priority reversal'
    assert orientations == {0} and non_half, 'missing visible orientation/polarity witness'


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count('num_train', num_train, 2, 10), _count('num_test', num_test, 1, 4)
    if nt <= ne:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    k, q = int(rng.integers(3, 5)), int(rng.integers(2, 5))
    a, b = q + 1, q + 2
    teaching_width = int(rng.integers(2 * q + 2, min(2 * q + 5, 24 - a - b) + 1))
    turn = int(rng.integers(4))
    regime = str(rng.choice(['longer', 'turn', 'mixture', 'clip']))
    # Mandatory lessons can occur anywhere in the requested training list.
    lessons = list(map(int, rng.choice(nt, 2, replace=False)))
    for attempt in range(40):
        puzzle, seen = {'train': [], 'test': []}, set()
        for i in range(nt + ne):
            split = 'train' if i < nt else 'test'
            for _ in range(40):
                if i == lessons[0]:
                    pair = _scene(rng, k, q, [[(a, 0)], [(b, 1)]],
                                  [teaching_width, teaching_width], turn, True, True)
                elif i == lessons[1]:
                    pair = _scene(rng, k, q, [[(a, 0), (b, 1)], [(b, 1), (a, 0)]],
                                  [teaching_width, teaching_width], turn, True)
                else:
                    mode = regime if split == 'test' and regime != 'turn' else 'ordinary'
                    pair_turn = (turn + int(rng.choice([1, 2, 3]))) % 4 if split == 'test' and regime == 'turn' else turn
                    cap = 2 if split == 'train' and regime == 'mixture' else 4
                    pair = _random_pair(rng, teaching_width, pair_turn, mode, cap)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('could not construct distinct inputs')
        rejected, singles, shifted, cycle = _teaching_evidence(puzzle)
        if rejected == set(RIVALS) and len(singles) >= 2 and shifted and cycle:
            validate_generated_puzzle(puzzle)
            return puzzle
    raise RuntimeError('could not construct a fully discriminating training set')


def self_test():
    """Bounded semantic, count, determinism and 42-seed diversity checks."""
    shapes, outputs, palettes, periods, dimensions, directions, lane_counts = set(), set(), set(), set(), set(), set(), set()
    structures, puzzles, multi_width, overlapping = set(), set(), 0, 0
    rectangular = symmetric = asymmetric = repeated_keys = distractors = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(_fingerprint(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                info = _decode(inp)
                shapes.add(inp.shape); outputs.add(out.shape)
                palettes.add((info['background'], info['marker'], info['material']))
                dimensions.add((info['k'], info['q']))
                directions.add(info['rotation']); lane_counts.add(len(info['strips']))
                periods.update(b['period'] for s in info['strips'] for b in s['blocks'])
                roles = {}
                norm = bytes(roles.setdefault(int(c), len(roles)) for c in inp.flat)
                structures.add((inp.shape, norm))
                multi_width += len({s['width'] for s in info['strips']}) > 1
                overlapping += any(len(s['blocks']) > 1 for s in info['strips'])
                repeated_keys += any(len({b['color'] for b in s['blocks']}) < len(s['blocks']) for s in info['strips'])
                distractors += info['distractors'] > 0
                rectangular += info['k'] != info['q']
                for m in info['masks'].values():
                    syms = sum(np.array_equal(m, t) for t in _transforms(m) if t.shape == m.shape)
                    symmetric += syms > 1; asymmetric += syms == 1
    assert len(puzzles) == 42 and len(structures) >= 180
    assert len(shapes) >= 45 and len(outputs) >= 30 and len(palettes) >= 80
    assert len(dimensions) >= 8 and directions == set(range(4))
    assert {1, 2, 3, 4, 5} <= lane_counts and set(range(2, 9)) <= periods
    assert min(multi_width, overlapping, rectangular, symmetric, asymmetric, distractors) >= 20
    assert repeated_keys >= 5
    # Hand-calculated diagnostic: 2-column masks, periods 3 and 4, width 10.
    # Nearest A = 10/01/10; farther B = 01/10/01. Farther B survives at
    # column 5, but union would also erase its initially occluded column 1.
    rng = np.random.default_rng(12345)
    pair = _scene(rng, 3, 2, [[(3, 0), (4, 1)]], [10], 0)
    info = _decode(pair['input'])
    color_a, color_b = [b['color'] for b in info['strips'][0]['blocks']]
    modified = pair['input'].copy()
    for color, digits in [(color_a, ('10', '01', '10')), (color_b, ('01', '10', '01'))]:
        y0, x0, y1, x1 = info['boxes'][color]
        modified[y0:y1, x0:x1] = info['background']
        mask = np.array([list(map(int, row)) for row in digits], dtype=bool)
        modified[y0:y1, x0:x1][mask] = color
    expected_rows = ('1001011001', '0100100110', '1001011001')
    result = solve(modified)
    holes = result[2:5, 2:12] == info['background']
    assert [''.join(map(str, map(int, row))) for row in holes] == list(expected_rows)
    assert not np.array_equal(result, _paint_decoded(_decode(modified), 'union'))
    # Input-only equivariance under all quarter-turns and arbitrary color permutations.
    for rotation in range(4):
        assert np.array_equal(solve(np.rot90(modified, rotation)), np.rot90(result, rotation))
    permutation = np.random.default_rng(7).permutation(10)
    assert np.array_equal(solve(permutation[modified]), permutation[result])
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(101 + nt * 5 + ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    assert _fingerprint(generate_puzzle(99, 2, 1)) == _fingerprint(generate_puzzle(99, 2, 1))
    damaged = generate_puzzle(0, 2, 1)
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged output accepted')


if __name__ == '__main__':
    self_test()
    print(json.dumps({'self_test': 'passed', 'seeds': 42}, separators=(',', ':')))
