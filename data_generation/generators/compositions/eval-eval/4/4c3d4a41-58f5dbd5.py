"""Perforated slider press: negative space changes mechanical contact.

The solver reads only pixels. Construction uses direct destination coordinates;
the solver pushes the remaining pixels step by step, preserving internal gaps.
"""
from __future__ import annotations

import numpy as np


def _count(value, lower, upper, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not lower <= int(value) <= upper):
        raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a <= 9))
    return a


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _components(a, bg):
    visited = np.zeros(a.shape, dtype=bool)
    result = []
    for yy, xx in np.argwhere(a != bg):
        y, x = int(yy), int(xx)
        if visited[y, x]:
            continue
        color, pending, cells = int(a[y, x]), [(y, x)], []
        visited[y, x] = True
        while pending:
            y, x = pending.pop()
            cells.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if (0 <= ny < a.shape[0] and 0 <= nx < a.shape[1]
                        and not visited[ny, nx] and a[ny, nx] == color):
                    visited[ny, nx] = True
                    pending.append((ny, nx))
        result.append((color, np.array(cells, dtype=int)))
    return result


def _decode(input_grid):
    """Identify the array, registered masks and one common pressure key."""
    a = _grid(input_grid)
    bg = int(a[0, 0])
    assert all(np.all(edge == bg) for edge in
               (a[0], a[-1], a[:, 0], a[:, -1])), "uniform outer border required"
    rectangles = []
    for color, cells in _components(a, bg):
        y, x = cells.min(axis=0)
        y1, x1 = cells.max(axis=0) + 1
        h, w = int(y1-y), int(x1-x)
        if h >= 5 and w >= 4 and len(cells) == h*w:
            rectangles.append((h*w, h, w, int(y), int(x), color))
    assert rectangles, "missing solid plates"
    largest = max(t[0] for t in rectangles)
    tiles = sorted(t for t in rectangles if t[0] == largest)
    dimensions = {(t[1], t[2]) for t in tiles}
    assert len(dimensions) == 1
    th, tw = next(iter(dimensions))
    m, n = th-2, tw-2
    ys, xs = sorted({t[3] for t in tiles}), sorted({t[4] for t in tiles})
    assert all(b-a == step for coords, step in ((ys, th+1), (xs, tw+1))
               for a, b in zip(coords, coords[1:]))
    assert {(t[3], t[4]) for t in tiles} == {(y, x) for y in ys for x in xs}
    assert len({t[5] for t in tiles}) == len(tiles)
    y0, x0, y1, x1 = ys[0]-1, xs[0]-1, ys[-1]+th+1, xs[-1]+tw+1
    assert y0 >= 0 and x0 >= 0 and y1 <= a.shape[0] and x1 <= a.shape[1]
    area = np.zeros(a.shape, dtype=bool)
    area[y0:y1, x0:x1] = True
    tiles_area = np.zeros(a.shape, dtype=bool)
    for _, _, _, y, x, color in tiles:
        tiles_area[y:y+th, x:x+tw] = True
    assert np.all(a[area & ~tiles_area] == bg)
    external = ~area & (a != bg)
    patterns, boxes, keys = {}, {}, []
    for color in sorted(int(c) for c in np.unique(a[external])):
        cells = np.argwhere(external & (a == color))
        top, left = map(int, cells.min(axis=0))
        bottom, right = map(int, cells.max(axis=0)+1)
        assert not area[top:bottom, left:right].any()
        register = a[top:bottom, left:right]
        assert np.all((register == bg) | (register == color))
        box = (top, left, bottom, right)
        boxes[color] = box
        if register.shape == (m, n):
            mask = register == color
            assert 0 < mask.sum() < m*n
            assert all((mask[0].any(), mask[-1].any(),
                        mask[:, 0].any(), mask[:, -1].any()))
            patterns[color] = mask.copy()
        else:
            assert register.shape[1] == n and 2 <= register.shape[0] <= m-1
            assert np.all(register[-1] == color)
            h = []
            for j in range(n):
                ink = np.flatnonzero(register[:-1, j] == color)
                assert len(ink) >= 1
                assert np.array_equal(ink, np.arange(ink[0], register.shape[0]-1))
                h.append(len(ink))
            keys.append((color, tuple(h)))
    assert len(keys) == 1, "one pressure key required"
    structure, heights = keys[0]
    assert structure not in {t[5] for t in tiles}
    assert {t[5] for t in tiles} <= set(patterns)
    for i, first in enumerate(boxes.values()):
        for second in list(boxes.values())[i+1:]:
            assert (first[2] < second[0] or second[2] < first[0]
                    or first[3] < second[1] or second[3] < first[1]), "registers touch"
    # Every material column must already contact or overlap the inserted tooth.
    for _, _, _, _, _, color in tiles:
        for j, h in enumerate(heights):
            material = np.flatnonzero(~patterns[color][:, j])
            assert len(material) and material[-1] >= m-h-2, "unsupported contact gap"
    return dict(background=bg, structure=structure, m=m, n=n,
                rows=len(ys), cols=len(xs), heights=heights, patterns=patterns,
                colors=[t[5] for t in tiles], boxes=boxes,
                crop=(y0, x0, y1, x1),
                tile_positions=[(t[3]-y0, t[4]-x0, t[5]) for t in tiles])


def _render_output(f, variant="correct"):
    """Direct constructor; variants are explicit semantic ablations for QA."""
    m, n, bg, s = f['m'], f['n'], f['background'], f['structure']
    out = np.full((f['rows']*(m+3)+1, f['cols']*(n+3)+1), bg, dtype=int)
    for i, color in enumerate(f['colors']):
        y, x = 1+(i//f['cols'])*(m+3), 1+(i % f['cols'])*(n+3)
        out[y:y+m+2, x:x+n+2] = color
        interior = out[y+1:y+m+1, x+1:x+n+1]
        interior[:] = bg
        mask = f['patterns'][color]
        if variant == 'wrong_color' and len(f['colors']) >= 2 and i < 2:
            mask = f['patterns'][f['colors'][1-i]]
        if variant == 'ignore_masks':
            mask = np.zeros((m, n), dtype=bool)
        elif variant == 'positive':
            mask = ~mask
        elif variant == 'flip_rows':
            mask = np.flipud(mask)
        elif variant == 'flip_cols':
            mask = np.fliplr(mask)
        heights = f['heights']
        common_shift = max(int(np.flatnonzero(~mask[:, j])[-1])-(m-h-2)
                           for j, h in enumerate(heights)) if variant == 'uniform_pressure' else 0
        for j, h in enumerate(heights):
            material = np.flatnonzero(~mask[:, j])
            target = m-h-2
            if len(material):
                shift = max(0, int(material[-1])-target)
                if variant == 'ignore_motion':
                    shift = 0
                elif variant == 'register_contact':
                    shift = h+1
                elif variant == 'uniform_pressure':
                    shift = common_shift
                if variant == 'compact':
                    destinations = range(max(0, target-len(material)+1), target+1)
                else:
                    destinations = (int(r)-shift for r in material)
                for r in destinations:
                    if 0 <= r <= target:
                        interior[r, j] = color
            interior[m-1-h:m, j] = s
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Push perforated columns until their last material pixel contacts the key."""
    a = _grid(input_grid)
    f = _decode(a)
    y0, x0, y1, x1 = f['crop']
    out = a[y0:y1, x0:x1].copy()
    m, n, bg = f['m'], f['n'], f['background']
    for y, x, color in f['tile_positions']:
        interior = out[y+1:y+m+1, x+1:x+n+1]
        interior[:] = bg
        for j, height in enumerate(f['heights']):
            cells = [(int(r), color) for r in np.flatnonzero(~f['patterns'][color][:, j])]
            tooth_top = m-1-height
            while cells[-1][0] >= tooth_top:
                cells = [(r-1, c) for r, c in cells if r-1 >= 0]
            assert cells[-1][0] == tooth_top-1
            for r, c in cells:
                interior[r, j] = c
            interior[tooth_top:, j] = f['structure']
    return out


def _bank_options(config, objects):
    r, c, m, n = config
    ah, aw = r*(m+3)+1, c*(n+3)+1
    result = []
    for nc in range(1, objects+1):
        nr = (objects+nc-1)//nc
        bh, bw = nr*(m+1)+1, nc*(n+1)+1
        if max(ah, bh) <= 30 and aw+bw+1 <= 30:
            result.append(('side', nr, nc))
        if ah+bh+1 <= 30 and max(aw, bw) <= 30:
            result.append(('below', nr, nc))
    return result


_CONFIGS = tuple((r, c, m, n) for r in range(1, 5) for c in range(1, 5)
                 for m in range(4, 8) for n in range(3, 7)
                 if r*c <= 6 and _bank_options((r, c, m, n), r*c+1))


def _layout(rng, config, objects, scatter):
    r, c, m, n = config
    ah, aw = r*(m+3)+1, c*(n+3)+1
    candidates = []
    for mode, nr, nc in _bank_options(config, objects):
        for gy in (1, 2):
            for gx in (1, 2):
                bh, bw = nr*m+(nr-1)*gy+2, nc*n+(nc-1)*gx+2
                h, w = ((max(ah, bh), aw+bw+1) if mode == 'side'
                        else (ah+bh+1, max(aw, bw)))
                if h <= 30 and w <= 30:
                    candidates.append((mode, nr, nc, gy, gx, bh, bw, h, w))
    mode, nr, nc, gy, gx, bh, bw, h, w = _pick(rng, candidates)
    if mode == 'side':
        ay, ax = int(rng.integers(h-ah+1)), 0
        by, bx = int(rng.integers(h-bh+1))+1, aw+2
    else:
        ay, ax = 0, int(rng.integers(w-aw+1))
        by, bx = ah+2, int(rng.integers(w-bw+1))+1
    slots = [(by+i*(m+gy), bx+j*(n+gx)) for i in range(nr) for j in range(nc)]
    slots = [slots[int(i)] for i in rng.choice(len(slots), objects, replace=False)]
    if rng.random() < .5:
        ay = h-ah-ay
        slots = [(h-m-y, x) for y, x in slots]
    if rng.random() < .5:
        ax = w-aw-ax
        slots = [(y, w-n-x) for y, x in slots]
    # Random surrounding whitespace, with one background border retained.
    total_y = int(rng.integers(min(4, 30-h)+1))
    total_x = int(rng.integers(min(4, 30-w)+1))
    dy, dx = int(rng.integers(total_y+1)), int(rng.integers(total_x+1))
    h, w, ay, ax = h+total_y, w+total_x, ay+dy, ax+dx
    slots = [(y+dy, x+dx) for y, x in slots]
    if scatter:
        # Bounded search; the guaranteed bank above is the fallback.
        for _ in range(6):
            hh, ww = max(h, 27), max(w, 27)
            yy, xx = int(rng.integers(hh-ah+1)), int(rng.integers(ww-aw+1))
            occupied = np.zeros((hh, ww), dtype=bool)
            occupied[yy:yy+ah, xx:xx+aw] = True
            coords = [(y, x) for y in range(1, hh-m) for x in range(1, ww-n)]
            placed = []
            for i in rng.permutation(len(coords)):
                y, x = coords[int(i)]
                if occupied[y:y+m, x:x+n].any():
                    continue
                placed.append((y, x))
                occupied[y-1:y+m+1, x-1:x+n+1] = True
                if len(placed) == objects:
                    return hh, ww, yy, xx, placed
    return h, w, ay, ax, slots


def _profile(rng, m, n, flat=False, witness=False):
    lo, hi = 1, m-2
    mode = 'flat' if flat else _pick(rng, ('rising', 'falling', 'valley', 'plateau', 'irregular'))
    h = rng.integers(lo, hi+1, n).tolist()
    if mode == 'flat':
        h = [int(rng.integers(1, m-2))]*n
    elif mode in ('rising', 'falling'):
        h = sorted(h, reverse=mode == 'falling')
    elif mode == 'valley':
        h[0] = h[-1] = hi
        h[n//2] = 1
    elif mode == 'plateau':
        p = int(rng.integers(n-1))
        h[p:p+2] = [hi, hi]
        h[(p+2) % n] = 1
    if not flat and len(set(h)) == 1:
        h[int(rng.integers(n))] = 1 if h[0] != 1 else hi
    lanes = None
    if witness:
        a, b = map(int, rng.choice(n, 2, replace=False))
        h[a], h[b] = 2, 1
        lanes = (a, b)
    return tuple(h), lanes


def _mask(rng, m, n, heights, witness=None):
    # Upper slots can be arbitrary; the trailing cut controls physical contact.
    mode = int(rng.integers(6))
    if mode == 0:
        mask = np.indices((m, n)).sum(axis=0) % 2 == 0
    elif mode == 1:
        mask = np.zeros((m, n), dtype=bool)
        mask[int(rng.integers(m)), :] = True
        mask[:, int(rng.integers(n))] = True
    elif mode == 2:
        mask = np.ones((m, n), dtype=bool)
    elif mode == 3:
        mask = np.zeros((m, n), dtype=bool)
        for j in range(n):
            mask[(j*2) % m, j] = True
    else:
        mask = rng.random((m, n)) < float(rng.uniform(.15, .8))
    for j, height in enumerate(heights):
        trim = int(rng.integers(height+2))
        bottom = m-1-trim
        mask[bottom, j] = False
        mask[bottom+1:, j] = True
    if witness is not None:
        a, b = witness
        target = m-heights[a]-2
        mask[0, a] = False
        mask[target, a] = False
        mask[target+1:, a] = True  # stationary, with a different contact point
        mask[0, b] = mask[2, b] = mask[-1, b] = False
        mask[-2, b] = True  # an internal slot survives a moving column
        other = next(j for j in range(n) if j not in (a, b))
        mask[0, other] = True
    if not mask[-1].any():
        j = int(rng.integers(n))
        mask[-1, j] = True
        mask[-2, j] = False
    if not mask[0].any():
        j = int(rng.integers(n))
        mask[0, j] = True
    for j in (0, n-1):
        if not mask[:, j].any():
            mask[0, j] = True
    # Register-span repairs must not delete a column's sole contact material.
    for j, height in enumerate(heights):
        if not (~mask[m-height-2:, j]).any():
            mask[-2, j] = False
    return mask


def _make_pair(rng, config, palette, flat, witness, scatter):
    r, c, m, n = config
    bg, structure = palette
    possible = [d for d in range(min(3, 8-r*c)+1)
                if _bank_options(config, r*c+d+1)]
    distractors = int(_pick(rng, possible))
    colors = [int(c) for c in rng.choice([c for c in range(10) if c not in palette],
                                         r*c+distractors, replace=False)]
    heights, lanes = _profile(rng, m, n, flat, witness)
    patterns = {color: _mask(rng, m, n, heights, lanes if i == 0 and witness else None)
                for i, color in enumerate(colors)}
    # Contrasting patterns make color equality consequential even in two-example puzzles.
    if witness:
        for _ in range(16):
            if not np.array_equal(patterns[colors[0]], patterns[colors[1]]):
                break
            patterns[colors[1]] = _mask(rng, m, n, heights)
    f = dict(rows=r, cols=c, m=m, n=n, background=bg, structure=structure,
             colors=colors[:r*c], patterns=patterns, heights=heights)
    h, w, ay, ax, slots = _layout(rng, config, len(colors)+1, scatter)
    inp = np.full((h, w), bg, dtype=int)
    for i, color in enumerate(f['colors']):
        y, x = ay+1+(i//c)*(m+3), ax+1+(i % c)*(n+3)
        inp[y:y+m+2, x:x+n+2] = color
    order = rng.permutation(len(colors)+1)
    for (y, x), i in zip(slots, order):
        if i == len(colors):
            # The shared key's base is at the bottom of its visible bounding box.
            depth = max(heights)+1
            inp[y+depth-1, x:x+n] = structure
            for j, height in enumerate(heights):
                inp[y+depth-1-height:y+depth-1, x+j] = structure
        else:
            color = colors[int(i)]
            inp[y:y+m, x:x+n][patterns[color]] = color
    return {'input': inp, 'output': _render_output(f)}


def _evidence(f):
    shifts, cuts, clips, slots = [], [], 0, 0
    for color in f['colors']:
        for j, h in enumerate(f['heights']):
            material = np.flatnonzero(~f['patterns'][color][:, j])
            shift = int(material[-1])-(f['m']-h-2)
            shifts.append(shift)
            cuts.append(f['m']-1-int(material[-1]))
            clips += int(np.count_nonzero(material < shift))
            first = max(shift, int(material[0]))
            slots += int(np.count_nonzero(f['patterns'][color][first:int(material[-1])+1, j]))
    return dict(shifts=shifts, cuts=cuts, clips=clips, slots=slots,
                stationary=shifts.count(0), moving=sum(s > 0 for s in shifts),
                masks_matter=not np.array_equal(_render_output(f), _render_output(f, 'ignore_masks')),
                motion_matters=not np.array_equal(_render_output(f), _render_output(f, 'ignore_motion')),
                contact_matters=not np.array_equal(_render_output(f), _render_output(f, 'register_contact')))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count(num_train, 2, 10, 'num_train'), _count(num_test, 1, 4, 'num_test')
    if nt <= ne:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    regime = _pick(rng, ('scale', 'multiplicity', 'flat', 'array_axis'))
    fixed_palette = tuple(map(int, rng.choice(10, 2, replace=False)))
    changing_palette = rng.random() < .8
    scatter_probability = float(rng.uniform(.15, .8))
    result, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for i in range(count):
            witness = split == 'train' and i < 2
            choices = list(_CONFIGS)
            if split == 'train':
                choices = [q for q in choices if q[2] >= 5 and q[0]*q[1] in (2, 3)]
                if witness:
                    choices = [q for q in choices if q[0]*q[1] == 2]
            if regime == 'scale':
                choices = [q for q in choices if (q[2] <= 6 and q[3] <= 4)
                           ] if split == 'train' else [q for q in choices if q[2] == 7 and q[3] >= 5]
            elif regime == 'multiplicity' and split == 'test':
                choices = [q for q in choices if q[0]*q[1] >= 4]
            elif regime == 'array_axis':
                choices = [q for q in choices if (q[0] == 1 and q[1] >= 2)
                           ] if split == 'train' else [q for q in choices if q[1] == 1 and q[0] >= 2]
            config = _pick(rng, choices)
            palette = tuple(map(int, rng.choice(10, 2, replace=False))) if changing_palette else fixed_palette
            flat = split == 'test' and regime == 'flat'
            if not witness and regime != 'flat':
                flat = rng.random() < .25
            for _ in range(96):
                pair = _make_pair(rng, config, palette, flat, witness,
                                  rng.random() < scatter_probability)
                f = _decode(pair['input'])
                e = _evidence(f)
                if _key(pair['input']) in seen or not e['masks_matter'] or not e['motion_matters']:
                    continue
                if witness and (not e['contact_matters'] or not e['clips'] or not e['slots']
                                or not e['stationary'] or not e['moving']):
                    continue
                if witness and any(np.array_equal(pair['output'], _render_output(f, v))
                                   for v in ('wrong_color', 'positive', 'flip_rows', 'flip_cols', 'compact', 'uniform_pressure')):
                    continue
                seen.add(_key(pair['input']))
                result[split].append(pair)
                break
            else:
                raise RuntimeError('bounded generation could not find a distinct informative example')
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, evidence, distinguish = set(), [], set()
    alternatives = ('ignore_masks', 'ignore_motion', 'register_contact', 'wrong_color', 'positive',
                    'flip_rows', 'flip_cols', 'compact', 'uniform_pressure')
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            f = _decode(inp)
            assert np.array_equal(out, _render_output(f)), 'incorrect press result or crop'
            assert np.array_equal(out, solve(inp)), 'input-only pixel simulation disagrees'
            # Check frame and key independently of the output comparison.
            for y, x, color in f['tile_positions']:
                plate = out[y:y+f['m']+2, x:x+f['n']+2]
                assert all(np.all(edge == color) for edge in
                           (plate[0], plate[-1], plate[:, 0], plate[:, -1]))
                for j, h in enumerate(f['heights']):
                    assert np.all(plate[f['m']-h:f['m']+1, j+1] == f['structure'])
                    assert plate[f['m']-h-1, j+1] == color
            if split == 'train':
                evidence.append(_evidence(f))
                distinguish.update(v for v in alternatives if not np.array_equal(out, _render_output(f, v)))
    assert set(alternatives) <= distinguish, 'training admits a listed simpler alternative'
    assert any(e['stationary'] and e['moving'] and e['clips'] and e['slots'] for e in evidence)
    assert any(e['contact_matters'] for e in evidence)


def _fingerprint(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def self_test():
    """Bounded 42-seed structural audit plus contact and negative regressions."""
    patterns = {2: np.array([[0, 0, 1], [0, 1, 1], [0, 0, 0],
                             [1, 0, 1], [1, 1, 0], [1, 0, 1]], dtype=bool)}
    focused = dict(rows=1, cols=1, m=6, n=3, background=0, structure=9,
                   colors=[2], patterns=patterns, heights=(2, 1, 3))
    expected = np.array([[2, 2, 0], [2, 2, 2], [2, 0, 9],
                         [9, 2, 9], [9, 9, 9], [9, 9, 9]])
    assert np.array_equal(_render_output(focused)[2:8, 2:5], expected)
    # A concrete input exercises the solver without sampling or latent state.
    inp = np.zeros((16, 15), dtype=int)
    inp[2:10, 2:7] = 2
    inp[2:8, 9:12][patterns[2]] = 2
    inp[13, 9:12] = 9
    for j, height in enumerate(focused['heights']):
        inp[13-height:13, 9+j] = 9
    assert np.array_equal(solve(inp)[2:8, 2:5], expected)
    # Recolor all roles and translate the scene: output coordinates stay local.
    recolor = np.array([8, 0, 4, 1, 2, 3, 5, 6, 7, 9])
    shifted = np.pad(recolor[inp], ((2, 1), (1, 2)), constant_values=8)
    assert np.array_equal(solve(shifted), recolor[solve(inp)])
    # Semantic bounds extend beyond the ordinary sampling curriculum. Include
    # a fitting five-plate array, a 30-row crop and a 30-column crop.
    boundary_rng = np.random.default_rng(918)
    for config in ((1, 5, 3, 2), (1, 1, 26, 2), (1, 1, 3, 26)):
        pair = _make_pair(boundary_rng, config, (0, 9), False, False, False)
        assert np.array_equal(solve(pair['input']), pair['output'])
        _grid(pair['input'])
        _grid(pair['output'])
    fingerprints, shapes, profiles, sizes, arrays, backgrounds, structures, counts = (set() for _ in range(8))
    totals = dict(stationary=0, moving=0, clips=0, slots=0, bottom_cut=0, distractors=0)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _decode(pair['input'])
                e = _evidence(f)
                shapes.add(pair['input'].shape)
                profiles.add(f['heights'])
                sizes.add((f['m'], f['n']))
                arrays.add((f['rows'], f['cols']))
                backgrounds.add(f['background'])
                structures.add(f['structure'])
                counts.add(len(f['colors']))
                for name in ('stationary', 'moving', 'clips', 'slots'):
                    totals[name] += e[name]
                totals['bottom_cut'] += sum(c > 0 for c in e['cuts'])
                totals['distractors'] += len(f['patterns'])-len(f['colors'])
                assert e['masks_matter'] and e['motion_matters']
    assert len(fingerprints) == 42 and len(shapes) >= 25 and len(profiles) >= 35
    assert len(sizes) >= 10 and len(arrays) >= 6
    assert backgrounds == structures == set(range(10))
    assert {2, 4, 6} <= counts and min(totals.values()) > 0
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            puzzle = generate_puzzle(117+nt*10+ne, nt, ne)
            assert len(puzzle['train']) == nt and len(puzzle['test']) == ne
            validate_generated_puzzle(puzzle)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    malformed = inp.copy()
    malformed[2:8, 9] = 2  # remove the entire first material column
    try:
        solve(malformed)
    except AssertionError:
        pass
    else:
        raise AssertionError('empty material column accepted')
    damaged = generate_puzzle(0, 2, 1)
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged output accepted')
    return dict(seeds=42, pairs=210, distinct_puzzles=len(fingerprints),
                input_shapes=len(shapes), profiles=len(profiles), interior_sizes=len(sizes),
                arrays=sorted(arrays), plate_counts=sorted(counts), **totals)


if __name__ == '__main__':
    print(self_test())
