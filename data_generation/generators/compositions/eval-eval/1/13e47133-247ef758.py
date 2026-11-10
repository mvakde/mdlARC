"""Cancelling stencils: coordinate copies define parity material, then contours.

No parent generator is imported. All solving conventions are visible or fixed:
the left header holds a terminated diagonal palette, and paired frame marks
address odd-bounding-box templates below that header. Stamps toggle material.
"""
from collections import deque

import numpy as np


_HEADER = 6
_SOURCE_START = 8
_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= s <= 30 for s in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _puzzle_key(puzzle):
    return tuple(tuple((_key(np.asarray(p['input'])), _key(np.asarray(p['output'])))
                       for p in puzzle[s]) for s in ('train', 'test'))


def _connected(mask):
    points = set(map(tuple, np.argwhere(mask)))
    if not points:
        return False
    seen = {min(points)}
    todo = list(seen)
    while todo:
        r, c = todo.pop()
        for dr, dc in _D8:
            q = (r + dr, c + dc)
            if q in points and q not in seen:
                seen.add(q)
                todo.append(q)
    return seen == points


def _components(mask):
    seen = np.zeros(mask.shape, bool)
    result = []
    h, w = mask.shape
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        todo = [(sr, sc)]
        seen[sr, sc] = True
        cells = []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for dr, dc in _D4:
                nr, nc = r + dr, c + dc
                if (0 <= nr < h and 0 <= nc < w and mask[nr, nc]
                        and not seen[nr, nc]):
                    seen[nr, nc] = True
                    todo.append((nr, nc))
        result.append(cells)
    return result


def _erosion(mask, diagonal=True):
    """Solver's simultaneous inward peeling; outside the field is empty."""
    current = mask.copy()
    result = np.full(mask.shape, -1, dtype=np.int16)
    h, w = mask.shape
    level = 0
    while current.any():
        padded = np.pad(current, 1)
        core = current.copy()
        for dr, dc in (_D8 if diagonal else _D4):
            core &= padded[1+dr:1+dr+h, 1+dc:1+dc+w]
        result[current & ~core] = level
        current = core
        level += 1
    return result


def _boundary_distance(mask):
    """Construction/check path: multi-source eight-neighbor distance BFS."""
    h, w = mask.shape
    depth = np.full(mask.shape, -1, dtype=np.int16)
    todo = deque()
    for rr, cc in np.argwhere(mask):
        r, c = int(rr), int(cc)
        if any(not (0 <= r+dr < h and 0 <= c+dc < w and mask[r+dr, c+dc])
               for dr, dc in _D8):
            depth[r, c] = 0
            todo.append((r, c))
    while todo:
        r, c = todo.popleft()
        for dr, dc in _D8:
            nr, nc = r+dr, c+dc
            if (0 <= nr < h and 0 <= nc < w and mask[nr, nc]
                    and depth[nr, nc] < 0):
                depth[nr, nc] = depth[r, c] + 1
                todo.append((nr, nc))
    return depth


def _parse(inp):
    """Recover roles, palette, source masks and coordinates from the input."""
    h, w = inp.shape
    candidates = []
    for d in range(7, min(10, w-5)):
        if h <= _SOURCE_START or not np.all(inp[:, d] == inp[0, d]):
            continue
        interior = inp[1:-1, d+2:-1]
        if not interior.size or not np.all(interior == interior[0, 0]):
            continue
        bg, divider = int(interior[0, 0]), int(inp[0, d])
        frame = int(inp[0, d+1])
        if len({bg, divider, frame}) != 3:
            continue
        if any(int(inp[r, c]) != frame for r, c in
               ((0, w-1), (h-1, d+1), (h-1, w-1))):
            continue
        if not np.all(inp[_HEADER, :d] == divider):
            continue
        candidates.append((d, bg, divider, frame))
    if len(candidates) != 1:
        raise ValueError("layout must have one visible divider/header/frame")
    d, bg, divider, frame = candidates[0]
    header = inp[:_HEADER, :d]
    ends = [i for i in range(2, 5) if int(header[i, i]) == frame]
    if len(ends) != 1:
        raise ValueError("palette must have one terminator after 1..3 entries")
    end = ends[0]
    palette = tuple(int(header[i, i]) for i in range(1, end))
    if (divider in palette or frame in palette or len(set(palette)) != len(palette)
            or palette == (bg,)):
        raise ValueError("invalid contour palette")
    expected_header = np.full(header.shape, bg, dtype=np.int8)
    for i, color in enumerate(palette, 1):
        expected_header[i, i] = color
    expected_header[end, end] = frame
    if not np.array_equal(header, expected_header) or np.any(inp[7, :d] != bg):
        raise ValueError("unexpected header marks")
    rows, cols = {}, {}
    for r in range(1, h-1):
        color = int(inp[r, d+1])
        if color != int(inp[r, -1]) or color in (bg, divider):
            raise ValueError("row marks must be opposite aligned pairs")
        if color != frame:
            rows.setdefault(color, []).append(r-1)
    for c in range(d+2, w-1):
        color = int(inp[0, c])
        if color != int(inp[-1, c]) or color in (bg, divider):
            raise ValueError("column marks must be opposite aligned pairs")
        if color != frame:
            cols.setdefault(color, []).append(c-d-2)
    templates = []
    source = inp[_SOURCE_START:, :d]
    forbidden = {bg, divider, frame, *palette}
    for color in np.unique(source):
        color = int(color)
        if color == bg:
            continue
        if color in forbidden:
            raise ValueError("template IDs must be distinct from structural/palette roles")
        cells = np.argwhere(source == color) + np.array([_SOURCE_START, 0])
        lo, hi = cells.min(axis=0), cells.max(axis=0)
        sh, sw = map(int, hi-lo+1)
        if sh % 2 != 1 or sw % 2 != 1 or max(sh, sw) > 9:
            raise ValueError("templates require tight odd boxes of size at most 9")
        mask = inp[lo[0]:hi[0]+1, lo[1]:hi[1]+1] == color
        if not _connected(mask):
            raise ValueError("each ID identifies one eight-connected template")
        rs, cs = rows.get(color, []), cols.get(color, [])
        center = (lo+hi)//2
        offsets = cells-center
        stamps = []
        if rs and cs:
            for r in rs:
                for c in cs:
                    placed = offsets + np.array([r, c])
                    if (np.any(placed[:, 0] < 0) or np.any(placed[:, 0] >= h-2)
                            or np.any(placed[:, 1] < 0) or np.any(placed[:, 1] >= w-d-3)):
                        raise ValueError("occupied stencil pixels must fit inside the field")
                    stamps.append(placed)
        state = 'A' if rs and cs else 'R' if rs else 'C' if cs else 'N'
        templates.append(dict(color=color, cells=cells, lo=tuple(map(int, lo)),
                              mask=mask, shape=(sh, sw), offsets=offsets,
                              rows=rs, cols=cs, stamps=stamps, state=state))
    if not templates or not any(t['state'] == 'A' for t in templates):
        raise ValueError("at least one complete stencil coordinate specification is required")
    templates.sort(key=lambda t: t['lo'])
    return dict(d=d, bg=bg, divider=divider, frame=frame, palette=palette,
                templates=templates, field_shape=(h-2, w-d-3))


def _coverage(p, mode='normal'):
    count = np.zeros(p['field_shape'], dtype=np.int16)
    for t in p['templates']:
        if t['state'] != 'A':
            continue
        if mode == 'ignore_coordinates':
            centers = [(count.shape[0]//2, count.shape[1]//2)]
        elif mode == 'zip':
            centers = list(zip(t['rows'], t['cols']))
        else:
            centers = [(r, c) for r in t['rows'] for c in t['cols']]
        offsets = t['offsets']
        if mode == 'boxes':
            sh, sw = t['shape']
            offsets = np.array([(r-sh//2, c-sw//2) for r in range(sh) for c in range(sw)])
        for r, c in centers:
            placed = offsets + np.array([r, c])
            if np.any(placed < 0) or np.any(placed >= np.array(count.shape)):
                raise ValueError("bounding box shortcut leaves field")
            count[placed[:, 0], placed[:, 1]] += 1
    return count


def _predict(inp, p, mode='normal'):
    out = inp.copy()
    for t in p['templates']:
        if t['state'] == 'A':
            out[t['cells'][:, 0], t['cells'][:, 1]] = p['bg']
    count = _coverage(p, mode)
    material = count > 0 if mode == 'union' else count == 1 if mode == 'single' else count % 2 == 1
    field = np.full(count.shape, p['bg'], dtype=np.int8)
    palette = np.asarray(p['palette'], dtype=np.int8)
    if mode == 'skip_background':
        palette = palette[palette != p['bg']]
    if mode == 'flat':
        field[material] = palette[0]
    elif mode == 'individual':
        local_depth = np.full(count.shape, 99, dtype=np.int16)
        for t in p['templates']:
            if t['state'] != 'A':
                continue
            source_depth = _erosion(t['mask'])[t['mask']]
            for placed in t['stamps']:
                rr, cc = placed[:, 0], placed[:, 1]
                local_depth[rr, cc] = np.minimum(local_depth[rr, cc], source_depth)
        field[material] = palette[local_depth[material] % len(palette)]
    else:
        depth = _erosion(material, diagonal=mode != 'manhattan')
        field[material] = palette[depth[material] % len(palette)]
    out[1:-1, p['d']+2:-1] = field
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only the visible input; no training data, seed, or generation state."""
    inp = _grid(input_grid)
    return _predict(inp, _parse(inp))


def _sample_mask(rng, h, w):
    """Connected tight rectangles, paths, diagonals, rings and grown polyominoes."""
    mode = int(rng.integers(8))
    mask = np.zeros((h, w), dtype=bool)
    if mode == 0 or min(h, w) == 1:
        mask[:] = True
    elif mode == 1:
        mask[[0, -1], :] = True
        mask[:, [0, -1]] = True
    elif mode == 2 and h == w:
        mask = np.eye(h, dtype=bool)
    elif mode == 3:
        mask[h//2, :] = True
        mask[:, w//2] = True
        probability = float(rng.uniform(.05, .45))
        for index in rng.permutation(h*w):
            r, c = divmod(int(index), w)
            if (rng.random() < probability
                    and mask[max(0, r-1):r+2, max(0, c-1):c+2].any()):
                mask[r, c] = True
    elif mode == 4:
        mask[:] = True
        nr, nc = max(1, h//3), max(1, w//3)
        mask[:nr, :nc] = False
    else:
        # A random walk joining independently chosen edge contacts makes the
        # bounding box tight, without forcing a visible center or a full cross.
        contacts = [(0, int(rng.integers(w))), (h-1, int(rng.integers(w))),
                    (int(rng.integers(h)), 0), (int(rng.integers(h)), w-1)]
        rng.shuffle(contacts)
        r, c = contacts[0]
        mask[r, c] = True
        for tr, tc in contacts[1:]:
            while (r, c) != (tr, tc):
                dr, dc = int(np.sign(tr-r)), int(np.sign(tc-c))
                if dr and dc and rng.random() < .6:
                    if rng.integers(2):
                        dr = 0
                    else:
                        dc = 0
                r, c = r+dr, c+dc
                mask[r, c] = True
        desired = int(rng.uniform(.25, .95)*h*w)
        while mask.sum() < desired:
            choices = [(r, c) for r in range(h) for c in range(w)
                       if not mask[r, c] and mask[max(0, r-1):r+2, max(0, c-1):c+2].any()]
            r, c = choices[int(rng.integers(len(choices)))]
            mask[r, c] = True
    if rng.integers(2):
        mask = mask[::-1]
    if rng.integers(2):
        mask = mask[:, ::-1]
    if h == w:
        mask = np.rot90(mask, int(rng.integers(4)))
    return mask.copy()


def _template(mask, top=0, left=0, state='A', rows=(), cols=()):
    return dict(mask=mask, top=int(top), left=int(left), state=state,
                rows=list(rows), cols=list(cols))


def _assign_axes(rng, ts, fh, fw, test=False):
    used_r = {v for t in ts for v in t['rows']}
    used_c = {v for t in ts for v in t['cols']}
    for i, t in enumerate(ts):
        for axis, (length, used, name) in enumerate(((fh, used_r, 'rows'), (fw, used_c, 'cols'))):
            needs = t['state'] in ('A', 'R' if axis == 0 else 'C')
            if not needs or t[name]:
                continue
            rad = t['mask'].shape[axis]//2 if t['state'] == 'A' else 0
            choices = [v for v in range(rad, length-rad) if v not in used]
            requested = int(rng.choice([1, 2], p=[.6, .4])) if t['state'] == 'A' else 1
            if test and i == 0 and t['state'] == 'A':
                requested = 3 if axis == t['repeat_axis'] else int(rng.choice([1, 2]))
            if len(choices) < requested:
                return False
            t[name] = sorted(map(int, rng.choice(choices, requested, replace=False)))
            used.update(t[name])
    return True


def _assemble(rng, h, sw, fw, ts, length, background_entry=None):
    """Record-based construction: toggle masks, then use boundary-distance BFS."""
    colors = list(map(int, rng.permutation(10)))
    bg, divider, frame = colors[:3]
    palette = list(colors[3:3+length])
    if length > 1 and (background_entry is True or (background_entry is None and rng.random() < .5)):
        palette[int(rng.integers(length))] = bg
    ids = [c for c in colors if c not in {bg, divider, frame, *palette}]
    inp = np.full((h, sw+3+fw), bg, dtype=np.int8)
    inp[:, sw] = divider
    inp[_HEADER, :sw] = divider
    for i, color in enumerate(palette, 1):
        inp[i, i] = color
    inp[length+1, length+1] = frame
    inp[[0, -1], sw+1:] = frame
    inp[:, [sw+1, inp.shape[1]-1]] = frame
    material = np.zeros((h-2, fw), dtype=bool)
    assert len(ids) >= len(ts)
    for t, color in zip(ts, ids):
        t['color'] = color
        sh, tw = t['mask'].shape
        view = inp[t['top']:t['top']+sh, t['left']:t['left']+tw]
        assert np.all(view[t['mask']] == bg)
        view[t['mask']] = color
        for r in t['rows']:
            inp[r+1, sw+1] = inp[r+1, -1] = color
        for c in t['cols']:
            inp[0, sw+2+c] = inp[-1, sw+2+c] = color
        if t['state'] == 'A':
            for r in t['rows']:
                for c in t['cols']:
                    view = material[r-sh//2:r+sh//2+1, c-tw//2:c+tw//2+1]
                    assert view.shape == t['mask'].shape
                    view ^= t['mask']
    # Optional complete coordinates for a color with no template are inert.
    if len(ids) > len(ts) and rng.random() < .35:
        absent = ids[len(ts)]
        free_r = [r for r in range(1, h-1) if inp[r, sw+1] == frame]
        free_c = [c for c in range(sw+2, inp.shape[1]-1) if inp[0, c] == frame]
        if free_r and free_c:
            r, c = int(rng.choice(free_r)), int(rng.choice(free_c))
            inp[r, sw+1] = inp[r, -1] = absent
            inp[0, c] = inp[-1, c] = absent
    out = inp.copy()
    for t in ts:
        if t['state'] == 'A':
            sh, tw = t['mask'].shape
            view = out[t['top']:t['top']+sh, t['left']:t['left']+tw]
            view[t['mask']] = bg
    field = np.full(material.shape, bg, dtype=np.int8)
    depth = _boundary_distance(material)
    field[material] = np.array(palette, dtype=np.int8)[depth[material] % length]
    out[1:-1, sw+2:-1] = field
    return {'input': inp, 'output': out}


def _witness_one(rng):
    h, sw = int(rng.integers(24, 31)), int(rng.integers(7, 10))
    fw = int(rng.integers(14, 31-sw-3))
    mask = np.ones((7, 7), bool)
    if rng.random() < .75:
        nr, nc = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        mask[:nr, :nc] = False
        mask = np.rot90(mask, int(rng.integers(4))).copy()
    br = int(rng.integers(3, h-2-3-6))
    bc = int(rng.integers(3, fw-3-6))
    top = int(rng.integers(8, h-13))
    ts = [_template(mask, top, int(rng.integers(sw-6)), rows=(br, br+6), cols=(bc, bc+6))]
    for i, state in enumerate(('N', 'R', 'C')):
        width = int(rng.choice([1, 3, 5]))
        ts.append(_template(np.ones((1, width), bool), top+8+2*i,
                            int(rng.integers(sw-width+1)), state))
    assert _assign_axes(rng, ts, h-2, fw)
    return _assemble(rng, h, sw, fw, ts, 2, background_entry=True)


def _witness_two(rng):
    h, sw = int(rng.integers(24, 31)), int(rng.integers(7, 10))
    fw = int(rng.integers(14, 31-sw-3))
    large = np.ones((7, 7), bool)
    large[5:, 4:] = False
    flip_r, flip_c = bool(rng.integers(2)), bool(rng.integers(2))
    if flip_r:
        large = large[::-1].copy()
    if flip_c:
        large = large[:, ::-1].copy()
    small_size = int(rng.choice([3, 5]))
    small = np.ones((small_size, small_size), bool)
    top = int(rng.integers(8, h-14))
    left = int(rng.integers(sw-6))
    first = _template(large, top, left)
    second = _template(small, top+8, int(rng.integers(sw-small_size+1)))
    nested = _template(np.ones((1, 3), bool), top+(1 if flip_r else 5),
                       left+(0 if flip_c else 4), str(rng.choice(['N', 'R', 'C'])))
    gap, offset = int(rng.choice([4, 6])), int(rng.choice([-3, -2, 2, 3]))
    axis = int(rng.integers(2))
    lengths = (h-2, fw)
    bases = []
    for a, length in enumerate(lengths):
        low, high = 3, length-4-(gap if a == axis else 0)
        if a != axis:
            rad = small_size//2
            low, high = max(low, rad-offset), min(high, length-1-rad-offset)
        bases.append(int(rng.integers(low, high+1)))
    first['rows'] = [bases[0], bases[0]+gap] if axis == 0 else [bases[0]]
    first['cols'] = [bases[1], bases[1]+gap] if axis == 1 else [bases[1]]
    second['rows'] = [bases[0]+(gap//2 if axis == 0 else offset)]
    second['cols'] = [bases[1]+(gap//2 if axis == 1 else offset)]
    ts = [first, second, nested]
    assert _assign_axes(rng, ts, h-2, fw)
    return _assemble(rng, h, sw, fw, ts, 3)


def _place_sources(rng, ts, h, sw):
    """Try occupied-cell interleaving, otherwise pack boxes with random slack."""
    if rng.random() < .4:
        occupied = np.zeros((h, sw), bool)
        tops = set()
        success = True
        for t in sorted(ts, key=lambda x: -x['mask'].sum()):
            sh, tw = t['mask'].shape
            choices = [(r, c) for r in range(_SOURCE_START, h-sh+1) if r not in tops
                       for c in range(sw-tw+1)
                       if not np.any(occupied[r:r+sh, c:c+tw] & t['mask'])]
            if not choices:
                success = False
                break
            r, c = choices[int(rng.integers(len(choices)))]
            t['top'], t['left'] = r, c
            occupied[r:r+sh, c:c+tw] |= t['mask']
            tops.add(r)
        if success:
            return
    order = list(map(int, rng.permutation(len(ts))))
    slack = h-_SOURCE_START-sum(t['mask'].shape[0] for t in ts)-(len(ts)-1)
    assert slack >= 0
    gaps = rng.multinomial(slack, np.full(len(ts)+1, 1/(len(ts)+1)))
    top = _SOURCE_START+int(gaps[0])
    for i, index in enumerate(order):
        t = ts[index]
        t['top'], t['left'] = top, int(rng.integers(sw-t['mask'].shape[1]+1))
        top += t['mask'].shape[0]+1+int(gaps[i+1])


def _general_pair(rng, test, test_lengths):
    h = int(rng.integers(26 if test else 16, 31))
    sw = 9 if test else int(rng.integers(7, 10))
    fw = int(rng.integers(14 if test else 10, 31-sw-3))
    n = int(rng.integers(2 if test else 1, 4))
    budget = h-_SOURCE_START-(n-1)
    ts = []
    for i in range(n):
        if test and i == 0:
            sh = 9
            tw = int(rng.choice([3, 5, 7, 9]))
        else:
            choices = [s for s in (1, 3, 5, 7) if s <= budget-(n-i-1)]
            sh = int(rng.choice(choices))
            tw = int(rng.choice([1, 3, 5, 7]))
        budget -= sh
        t = _template(_sample_mask(rng, sh, tw))
        # Eligibility is independent of appearance and source order.
        t['state'] = 'A' if test and i < 2 else str(rng.choice(['A', 'N', 'R', 'C'], p=[.65, .15, .1, .1]))
        if test and i == 0:
            t['repeat_axis'] = int(rng.integers(2))
        ts.append(t)
    if not any(t['state'] == 'A' for t in ts):
        ts[int(rng.integers(n))]['state'] = 'A'
    _place_sources(rng, ts, h, sw)
    if not _assign_axes(rng, ts, h-2, fw, test):
        return None
    length = int(rng.choice(test_lengths)) if test else int(rng.integers(1, 4))
    return _assemble(rng, h, sw, fw, ts, length)


def _features(inp):
    p = _parse(inp)
    count = _coverage(p)
    material = count % 2 == 1
    depth = _erosion(material)
    normal = _predict(inp, p)
    changes = {mode: int(np.count_nonzero(normal != _predict(inp, p, mode)))
               for mode in ('flat', 'union', 'single', 'zip', 'boxes', 'manhattan',
                            'ignore_coordinates', 'individual', 'skip_background')}
    by_color = []
    for t in p['templates']:
        own = np.zeros(count.shape, np.int16)
        for placed in t['stamps']:
            own[placed[:, 0], placed[:, 1]] += 1
        by_color.append(own)
    overlap_colors = sum((a > 0).astype(np.int16) for a in by_color)
    same_overlap = any(np.any(a >= 2) for a in by_color)
    source_box_erasure = inp.copy()
    for t in p['templates']:
        if t['state'] == 'A':
            r, c = t['lo']
            sh, sw = t['shape']
            source_box_erasure[r:r+sh, c:c+sw] = p['bg']
    expected_source = normal[:, :p['d']]
    holes = sum(all(r not in (0, count.shape[0]-1) and c not in (0, count.shape[1]-1)
                    for r, c in component) for component in _components(~material))
    return dict(parsed=p, count=count, material=material, depth=depth,
                differences=changes, same_overlap=same_overlap,
                cross_overlap=bool(np.any(overlap_colors >= 2)),
                triple=bool(np.any(count >= 3)), holes=holes,
                source_erasure_difference=int(np.count_nonzero(expected_source != source_box_erasure[:, :p['d']])),
                components=len(_components(material)), max_depth=int(depth.max()))


def _check_counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def _fallback_pair(rng, kind):
    """Constructive witnesses, used only at the bounded search limit."""
    if kind == 'first':
        ts = [_template(np.ones((7, 7), bool), 8, 0, rows=(5, 11), cols=(4, 10)),
              _template(np.ones((1, 3), bool), 16, 0, 'N'),
              _template(np.ones((1, 3), bool), 18, 1, 'R', rows=(2,)),
              _template(np.ones((1, 3), bool), 20, 3, 'C', cols=(1,))]
        return _assemble(rng, 27, 7, 17, ts, 2, background_entry=True)
    if kind == 'second':
        mask = np.ones((7, 7), bool)
        mask[:2, 4:] = False
        ts = [_template(mask, 13, 0, rows=(11,), cols=(3, 9)),
              _template(np.ones((3, 3), bool), 21, 0, rows=(14,), cols=(6,)),
              _template(np.ones((1, 3), bool), 14, 4, 'R', rows=(2,))]
        return _assemble(rng, 30, 7, 17, ts, 3, background_entry=False)
    if kind == 'test':
        ts = [_template(np.ones((9, 9), bool), 8, 0, rows=(5, 13, 21), cols=(5,)),
              _template(np.ones((3, 3), bool), 18, 0, rows=(8,), cols=(7,))]
        return _assemble(rng, 30, 9, 18, ts, 2)
    ts = [_template(np.ones((5, 5), bool), 8, 1, rows=(7,), cols=(5,))]
    return _assemble(rng, 20, 7, 14, ts, 2)


def _distinct_recolor(pair, seen):
    """Twenty bijections exceed the maximum thirteen preceding input grids."""
    p = _parse(pair['input'])
    for index in range(20):
        remap = np.arange(10, dtype=np.int8)
        if index >= 10:
            a, b = p['bg'], p['divider']
            remap[a], remap[b] = remap[b], remap[a]
        remap = (remap+index % 10) % 10
        result = {role: remap[grid] for role, grid in pair.items()}
        if _key(result['input']) not in seen:
            return result
    raise AssertionError('twenty distinct recolorings exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    puzzle = {'train': [], 'test': []}
    seen = set()
    for i in range(num_train+num_test):
        test = i >= num_train
        for attempt in range(120):
            if attempt == 119:
                kind = 'first' if i == 0 else 'second' if i == 1 else 'test' if test else 'general'
                pair = _distinct_recolor(_fallback_pair(rng, kind), seen)
            elif i == 0:
                pair = _witness_one(rng)
            elif i == 1:
                pair = _witness_two(rng)
            else:
                lengths = [len(_parse(p['input'])['palette']) for p in puzzle['train']
                           if len(_parse(p['input'])['palette']) >= 2]
                pair = _general_pair(rng, test, sorted(set(lengths)))
                if pair is None:
                    continue
            f = _features(pair['input'])
            if _key(pair['input']) in seen or not f['material'].any():
                continue
            if i == 0 and not (f['same_overlap'] and f['max_depth'] >= 2
                               and all(f['differences'][m] for m in ('flat', 'union', 'zip', 'ignore_coordinates', 'individual'))):
                continue
            if i == 1 and not (f['cross_overlap'] and f['triple'] and f['max_depth'] >= 2
                               and f['differences']['manhattan'] and f['differences']['single']
                               and f['source_erasure_difference']):
                continue
            # A test must exercise both inherited mechanisms and cancellation.
            if test and not all(f['differences'][m] for m in ('flat', 'union', 'ignore_coordinates')):
                continue
            if np.array_equal(pair['input'], pair['output']):
                continue
            break
        else:
            raise RuntimeError('bounded scene search exhausted')
        seen.add(_key(pair['input']))
        puzzle['test' if test else 'train'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check returned grids, geometry, parity, palette phases and evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen = set()
    features = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            f = _features(inp)
            p, d = f['parsed'], f['parsed']['d']
            assert f['material'].any()
            assert np.array_equal(solve(inp), out), 'wrong parity contour answer'
            assert not np.array_equal(inp, out)
            assert np.array_equal(_boundary_distance(f['material']), f['depth']), 'depth paths disagree'
            # The whole header, divider and addressed frame are immutable.
            assert np.array_equal(inp[:_SOURCE_START, :d], out[:_SOURCE_START, :d])
            assert np.array_equal(inp[:, d:d+2], out[:, d:d+2])
            assert np.array_equal(inp[[0, -1], d+1:], out[[0, -1], d+1:])
            assert np.array_equal(inp[:, -1], out[:, -1])
            for t in p['templates']:
                values = out[t['cells'][:, 0], t['cells'][:, 1]]
                assert np.all(values == (p['bg'] if t['state'] == 'A' else t['color']))
                # Generated placements fit their entire box, not just the mask.
                if t['state'] == 'A':
                    for axis, name in enumerate(('rows', 'cols')):
                        rad = t['shape'][axis]//2
                        assert all(rad <= v < p['field_shape'][axis]-rad for v in t[name])
            field = out[1:-1, d+2:-1]
            assert np.all(field[~f['material']] == p['bg']), 'even coverage must vanish'
            assert set(map(int, np.unique(field))) <= {p['bg'], *p['palette']}
            if split == 'train':
                assert all(max(t['shape']) <= 7 for t in p['templates'])
            else:
                assert any(t['state'] == 'A' and 9 in t['shape'] for t in p['templates'])
                assert any(t['state'] == 'A' and max(len(t['rows']), len(t['cols'])) >= 3 for t in p['templates'])
                assert all(f['differences'][m] for m in ('flat', 'union', 'ignore_coordinates'))
            features[split].append(f)
    train = features['train']
    assert any(f['same_overlap'] for f in train)
    assert any(f['cross_overlap'] and f['triple'] for f in train)
    assert any(f['max_depth'] >= len(f['parsed']['palette']) >= 2 for f in train)
    assert any(f['source_erasure_difference'] for f in train)
    assert {'N', 'R', 'C'} <= {t['state'] for f in train for t in f['parsed']['templates']}
    assert any(f['max_depth'] >= 2 and len(f['parsed']['palette']) == 3 for f in train)
    for mode in ('flat', 'union', 'single', 'zip', 'boxes', 'manhattan',
                 'ignore_coordinates', 'individual', 'skip_background'):
        assert any(f['differences'][mode] for f in train), f'missing witness against {mode}'
    assert {len(f['parsed']['palette']) for f in features['test']} <= {len(f['parsed']['palette']) for f in train}


def self_test():
    """Bounded 42-seed coverage plus determinism, recoloring and corruption."""
    shapes, masks, geometries, palettes, roles, products, topology = (set() for _ in range(7))
    puzzles = set()
    coverage_maxima = set()
    background_entries = 0
    empty_centers = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.add(_puzzle_key(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input']
                f = _features(inp)
                p = f['parsed']
                shapes.add(inp.shape)
                geometries.add(_key(f['material'].astype(np.int8)))
                roles.add((p['bg'], p['divider'], p['frame']))
                palettes.add(len(p['palette']))
                topology.add((f['components'], f['holes']))
                coverage_maxima.add(int(f['count'].max()))
                background_entries += p['bg'] in p['palette']
                for t in p['templates']:
                    masks.add(_key(t['mask'].astype(np.int8)))
                    if t['state'] == 'A':
                        products.add((len(t['rows']), len(t['cols'])))
                        sh, sw = t['shape']
                        empty_centers += not t['mask'][sh//2, sw//2]
    assert len(puzzles) == 42
    assert len(shapes) >= 50 and len(masks) >= 70 and len(geometries) >= 140
    assert len(roles) >= 60 and palettes == {1, 2, 3}
    assert {(1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (1, 3)} <= products
    assert len(topology) >= 12 and max(coverage_maxima) >= 4
    assert background_entries >= 50 and empty_centers >= 20
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(42+a*5+b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    p = generate_puzzle(912345, 2, 1)
    assert _puzzle_key(p) == _puzzle_key(generate_puzzle(912345, 2, 1))
    rng = np.random.default_rng(321)
    for pair in p['train']+p['test']:
        remap = rng.permutation(10).astype(np.int8)
        assert np.array_equal(solve(remap[pair['input']]), remap[pair['output']])
        parsed = _parse(pair['input'])
        # Reordering the stencil records cannot affect the parity material.
        rev = dict(parsed, templates=list(reversed(parsed['templates'])))
        assert np.array_equal(_predict(pair['input'], rev), pair['output'])
    for mode in ('flat', 'union', 'single', 'zip', 'boxes', 'manhattan',
                 'ignore_coordinates', 'individual', 'skip_background'):
        chosen = next(pair for pair in p['train'] if _features(pair['input'])['differences'][mode])
        bad = {s: [{k: np.asarray(v).copy() for k, v in x.items()} for x in p[s]] for s in p}
        index = next(i for i, pair in enumerate(p['train']) if pair is chosen)
        bad['train'][index]['output'] = _predict(chosen['input'], _parse(chosen['input']), mode)
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f'validator accepted {mode}')
    bad = {s: [{k: np.asarray(v).copy() for k, v in x.items()} for x in p[s]] for s in p}
    pair = bad['train'][1]
    parsed = _parse(pair['input'])
    for t in parsed['templates']:
        if t['state'] == 'A':
            r, c = t['lo']
            sh, sw = t['shape']
            pair['output'][r:r+sh, c:c+sw] = parsed['bg']
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted whole-source-box erasure')
    # Exercise the constructive last-resort scenes independently of their
    # probability of being reached in normal random sampling.
    for seed in range(8):
        rng = np.random.default_rng(seed)
        a = _features(_fallback_pair(rng, 'first')['input'])
        b = _features(_fallback_pair(rng, 'second')['input'])
        c = _features(_fallback_pair(rng, 'test')['input'])
        assert a['max_depth'] >= 2 and a['differences']['skip_background']
        assert all(a['differences'][m] for m in ('flat', 'union', 'zip', 'individual', 'ignore_coordinates'))
        assert b['triple'] and b['max_depth'] >= 2 and b['source_erasure_difference']
        assert b['differences']['single'] and b['differences']['manhattan']
        assert all(c['differences'][m] for m in ('flat', 'union', 'ignore_coordinates'))
    # Raw family geometry: diagonal holes change Chebyshev versus L1 depths.
    mask = np.ones((9, 9), bool)
    mask[4, 4] = False
    assert np.array_equal(_erosion(mask), _boundary_distance(mask))
    assert not np.array_equal(_erosion(mask), _erosion(mask, False))


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds, legal counts, invariants and mutations)')
