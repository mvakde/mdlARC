"""Coat and dock: ordinal matching of cargos to topological housings.

No files, global random state, or generation metadata are used by solve().
All construction retries and diagnostics are bounded; self_test stores no grids.
"""
from itertools import permutations

import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(((0 <= a) & (a <= 9)).all()), "colors outside 0..9")
    return a


def _components(mask, neighbors=N4):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    cells.append(q)
                    stack.append(q)
        result.append(cells)
    return result


def _holes(mask, neighbors=N4):
    padded = np.pad(mask, 1)
    outside = np.zeros(padded.shape, bool)
    outside[0, 0] = True
    stack = [(0, 0)]
    h, w = padded.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in neighbors:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not padded[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                stack.append((rr, cc))
    return (~padded & ~outside)[1:-1, 1:-1]


def _dilate(mask, neighbors=N8):
    h, w = mask.shape
    p = np.pad(mask, 1)
    result = mask.copy()
    for dr, dc in neighbors:
        result |= p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return result


def _crop(mask):
    rr, cc = np.nonzero(mask)
    return mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1].copy()


def _unambiguous(mask):
    a, b, c, d = mask[:-1, :-1], mask[:-1, 1:], mask[1:, :-1], mask[1:, 1:]
    return not np.any((a == d) & (b == c) & (a != b))


def _orient(rng, mask):
    mask = np.rot90(mask, int(rng.integers(4)))
    return _crop(np.fliplr(mask) if rng.integers(2) else mask)


def _placement(housing, cargo):
    """Center bounding boxes, biasing an indivisible spare cell south/east."""
    holes = housing['holes']
    rr, cc = np.nonzero(holes)
    h, w = cargo['mask'].shape
    ch, cw = int(rr.max() - rr.min() + 1), int(cc.max() - cc.min() + 1)
    _require(h <= ch and w <= cw, "cargo exceeds cavity box")
    r = int(rr.min()) + (ch - h) // 2
    c = int(cc.min()) + (cw - w) // 2
    stamp = np.zeros(holes.shape, bool)
    stamp[r:r + h, c:c + w] = cargo['mask']
    _require(not (stamp & ~holes).any(), "centered cargo intersects housing")
    return stamp, (r, c), (ch - h, cw - w)


def _analyze(input_grid):
    g = _grid(input_grid)
    bg = int(g[0, 0])
    _require(bool(np.all(g[0] == bg) and np.all(g[-1] == bg)
                  and np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg)), "blank outer border")
    objects = []
    for cells in _components(g != bg):
        rr, cc = np.array(cells).T
        r, c = int(rr.min()), int(cc.min())
        bottom, right = int(rr.max()) + 1, int(cc.max()) + 1
        color = int(g[rr[0], cc[0]])
        _require(np.all(g[rr, cc] == color), "monochrome components required")
        body = np.zeros((bottom - r, right - c), bool)
        body[rr - r, cc - c] = True
        _require(_unambiguous(body), "diagonal-only topology")
        holes = _holes(body)
        _require(np.array_equal(holes, _holes(body, N8)), "ambiguous cavity")
        _require(len(_components(holes)) <= 1, "one cavity per housing")
        neighborhood = g[r - 1:bottom + 1, c - 1:right + 1].copy()
        neighborhood[1:-1, 1:-1] = bg
        _require(np.all(neighborhood == bg), "separate bounding boxes by background")
        _require(np.all(g[r:bottom, c:right][~body] == bg), "nested input components")
        objects.append(dict(top=r, left=c, mask=body, holes=holes, color=color))
    housings = sorted((o for o in objects if o['holes'].any()), key=lambda o: o['top'])
    cargos = sorted((o for o in objects if not o['holes'].any()), key=lambda o: o['top'])
    _require(2 <= len(housings) <= 4 and len(housings) == len(cargos), "equal queues of 2..4")
    for queue in (housings, cargos):
        _require(len({o['top'] for o in queue}) == len(queue), "same-class top-row tie")
    _require(all(max(o['mask'].shape) <= 10 for o in housings), "housing exceeds 10x10")
    _require(all(max(o['mask'].shape) <= 5 for o in cargos), "cargo exceeds 5x5")
    for h in housings:
        for d in cargos:
            _placement(h, d)
    for h, d in zip(housings, cargos):
        _require(h['color'] != d['color'], "paired colors must differ")
        _placement(h, d)
    return dict(grid=g, background=bg, housings=housings, cargos=cargos)


def _render(analysis, cargos=None, neighbors=N8, align='center', coat=True, fill=False):
    out = analysis['grid'].copy()
    bg = analysis['background']
    for d in analysis['cargos']:
        r, c = d['top'], d['left']
        view = out[r:r + d['mask'].shape[0], c:c + d['mask'].shape[1]]
        view[d['mask']] = bg
    for h, d in zip(analysis['housings'], analysis['cargos'] if cargos is None else cargos):
        stamp, _, _ = _placement(h, d)
        if align == 'corner':
            rr, cc = np.nonzero(h['holes'])
            stamp[:] = False
            r, c = int(rr.min()), int(cc.min())
            dh, dw = d['mask'].shape
            stamp[r:r + dh, c:c + dw] = d['mask']
            stamp &= h['holes']
        r, c = h['top'], h['left']
        hh, hw = h['mask'].shape
        view = out[r:r + hh, c:c + hw]
        if fill:
            view[h['holes']] = h['color']
        if coat:
            view[_dilate(stamp, neighbors) & h['holes'] & ~stamp] = h['color']
        view[stamp] = d['color']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer background, topology, queues, translation and coat from input only."""
    return _render(_analyze(input_grid))


def _cargo(rng):
    h, w = int(rng.integers(1, 6)), int(rng.integers(1, 6))
    style = int(rng.integers(6))
    mask = np.ones((h, w), bool)
    if style == 0:
        mask = np.ones((1, int(rng.integers(1, 6))), bool)
    elif style == 1:
        mask[:] = False
        mask[:, 0] = True
        mask[-1, :] = True
    elif style in (2, 3):
        h, w = max(3, h), max(3, w)
        mask = np.ones((h, w), bool)
        mask[1:-1, 1:-1] = False
        if style == 2:
            mask[0, 1:-1] = False
        else:
            mask[0, w // 2] = False
    elif style == 4:
        mask[:] = False
        mask[h // 2, w // 2] = True
        for _ in range(int(rng.integers(3, 18))):
            r, c = np.argwhere(mask)[int(rng.integers(mask.sum()))]
            dr, dc = N4[int(rng.integers(4))]
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < h and 0 <= cc < w:
                proposal = mask.copy()
                proposal[rr, cc] = True
                if _unambiguous(proposal):
                    mask = proposal
        mask |= _holes(mask)
    return _orient(rng, mask)


def _housing(rng, cargos, hole_shape=None, irregular=False):
    dh = max(m.shape[0] for m in cargos)
    dw = max(m.shape[1] for m in cargos)
    if hole_shape is None:
        hh = max(3, dh + int(rng.integers(3)))
        hw = max(3, dw + int(rng.integers(3)))
    else:
        hh, hw = hole_shape
    hole = np.ones((hh, hw), bool)
    # Corner intrusions vary cavity geometry without giving each cargo its
    # own uniquely fitting socket. Every cargo must fit every housing.
    for corner in rng.permutation(4)[:int(rng.integers(1, 4)) if irregular else 0]:
        ah, aw = int(rng.integers(1, min(2, hh - 2) + 1)), int(rng.integers(1, min(2, hw - 2) + 1))
        rs = slice(0, ah) if corner < 2 else slice(hh - ah, hh)
        cs = slice(0, aw) if corner % 2 == 0 else slice(hw - aw, hw)
        candidate = hole.copy()
        candidate[rs, cs] = False
        fake = dict(holes=candidate)
        try:
            for m in cargos:
                _placement(fake, dict(mask=m))
        except AssertionError:
            continue
        hole = candidate
    # The mandatory first cavity has a nonrectangular corner, while leaving
    # the same bounding box and an unobstructed centered 3x3 placement.
    if irregular and hh >= dh + 2 and hw >= dw + 2:
        hole[0, 0] = False
    t, b = int(rng.integers(1, 3)), int(rng.integers(1, 3))
    l, r = int(rng.integers(1, 3)), int(rng.integers(1, 3))
    if hh + t + b > 10:
        b = 1
    if hw + l + r > 10:
        r = 1
    mask = np.ones((hh + t + b, hw + l + r), bool)
    mask[t:t + hh, l:l + hw] &= ~hole
    before = _holes(mask)
    # Exterior corner cuts create uneven walls and stepped silhouettes.
    for corner in rng.permutation(4)[:int(rng.integers(4))]:
        ah, aw = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        candidate = mask.copy()
        rs = slice(0, ah) if corner < 2 else slice(len(mask) - ah, len(mask))
        cs = slice(0, aw) if corner % 2 == 0 else slice(mask.shape[1] - aw, mask.shape[1])
        candidate[rs, cs] = False
        if (_unambiguous(candidate) and len(_components(candidate)) == 1
                and np.array_equal(_holes(candidate), before)):
            mask = candidate
    return _crop(mask)


def _fallback_positions(masks, n):
    # A two-column housing cluster plus a narrow cargo strip always fits:
    # four <=10x10 housings, four <=5x5 cargos, border and one-cell gaps.
    houses, cargos = masks[:n], masks[n:]
    col_width = max(m.shape[1] for m in houses[::2])
    positions, row = [], 1
    for start in range(0, n, 2):
        end = row
        for j in range(start, min(n, start + 2)):
            top = row + j % 2
            left = 1 if j % 2 == 0 else col_width + 2
            positions.append((top, left))
            end = max(end, top + houses[j].shape[0])
        row = end + 1
    right = max(c + m.shape[1] for (r, c), m in zip(positions, houses))
    top = 1
    for m in cargos:
        positions.append((top, right + 1))
        top += m.shape[0] + 1
    h = max(r + m.shape[0] for (r, c), m in zip(positions, masks)) + 1
    w = max(c + m.shape[1] for (r, c), m in zip(positions, masks)) + 1
    _require(h <= 30 and w <= 30, "fallback bounds")
    return (h, w), positions


def _pack(rng, masks, n):
    fallback_shape, fallback = _fallback_positions(masks, n)
    area = sum(m.size for m in masks)
    ratio = float(np.exp(rng.uniform(-.4, .4)))
    h = max(max(m.shape[0] for m in masks) + 2, int(np.sqrt(area * ratio / .45)))
    w = max(max(m.shape[1] for m in masks) + 2, int(np.sqrt(area / ratio / .45)))
    for extra in (0, 5, 30):
        shape = min(30, h + extra), min(30, w + extra)
        for _ in range(4):
            occupied = np.zeros(shape, bool)
            tops = [set(), set()]
            positions = [None] * len(masks)
            for i in sorted(range(len(masks)), key=lambda j: -masks[j].size):
                mh, mw = masks[i].shape
                sites = []
                for r in range(1, shape[0] - mh):
                    if r in tops[int(i >= n)]:
                        continue
                    for c in range(1, shape[1] - mw):
                        if not occupied[r - 1:r + mh + 1, c - 1:c + mw + 1].any():
                            sites.append((r, c))
                if not sites:
                    break
                r, c = sites[int(rng.integers(len(sites)))]
                positions[i] = r, c
                tops[int(i >= n)].add(r)
                occupied[r:r + mh, c:c + mw] = True
            else:
                return shape, positions
    return fallback_shape, fallback


def _nearest_order(a):
    hs, ds = a['housings'], a['cargos']
    # A bijective proximity competitor; lexical order resolves equal costs.
    order = min(permutations(range(len(ds))), key=lambda p: (
        sum(abs(h['top'] - ds[j]['top']) + abs(h['left'] - ds[j]['left'])
            for h, j in zip(hs, p)), p))
    return [ds[j] for j in order]


def _horizontal_order(a):
    hs = sorted(a['housings'], key=lambda o: (o['left'], o['top']))
    ds = sorted(a['cargos'], key=lambda o: (o['left'], o['top']))
    mapping = {id(h): d for h, d in zip(hs, ds)}
    return [mapping[id(h)] for h in a['housings']]


def _witness(a, mode):
    order = _nearest_order(a) if mode == 'nearest' else _horizontal_order(a)
    return not np.array_equal(_render(a, cargos=order), _render(a))


def _make_pair(rng, index, training, n, bg, identity):
    if training and index == 0:
        opened = np.ones((3, 3), bool)
        opened[1, 1] = False
        opened[0, 1] = False
        bent = np.zeros((3, 3), bool)
        bent[:, 0] = True
        bent[-1] = True
        cargos = [_orient(rng, opened), _orient(rng, bent)]
        houses = [_housing(rng, cargos, (5, 5), True),
                  _housing(rng, cargos, (6, 6), bool(rng.integers(2)))]
    elif training and index == 1:
        opened = np.ones((5, 5), bool)
        opened[1:-1, 1:-1] = False
        opened[0, 2] = False
        cargos = [_orient(rng, opened), np.ones((1, 1), bool), _cargo(rng)]
        houses = [_housing(rng, cargos, (5, 5)) for _ in range(n)]
    else:
        cargos = [_cargo(rng) for _ in range(n)]
        houses = [_housing(rng, cargos, irregular=bool(rng.random() < .6)) for _ in range(n)]
    masks = houses + cargos
    # All cargos fit all housings: geometry cannot replace ordinal matching.
    for h in houses:
        for d in cargos:
            _placement(dict(holes=_holes(h)), dict(mask=d))
    shape, positions = _pack(rng, masks, n)
    if training and index == 0:
        # Finite deterministic evidence layout, then bounded legal free moves.
        # H0,H1 precede both cargos; one class's left order is reversed.
        h0, h1 = houses
        d0, d1 = cargos
        positions = [(1, 13), (3, 1), (16, 3), (21, 15)]
        shape = (25 + identity % 3, 25 + (identity // 3) % 3)
        # Translation/reflection of geometry is safe only if the rank witnesses
        # survive. Random layouts are tried first to avoid a mandatory template.
        for _ in range(12):
            candidate_shape, candidate_positions = _pack(rng, masks, n)
            candidate = np.full(candidate_shape, bg, np.int8)
            for j, (m, (r, c)) in enumerate(zip(masks, candidate_positions)):
                candidate[r:r + len(m), c:c + m.shape[1]][m] = (bg + j + 1) % 10
            # Colors here are temporary, distinct, and do not affect geometry.
            a = _analyze(candidate)
            if _witness(a, 'nearest') and _witness(a, 'horizontal'):
                shape, positions = candidate_shape, candidate_positions
                break
    g = np.full(shape, bg, np.int8)
    for j, (m, (r, c)) in enumerate(zip(masks, positions)):
        g[r:r + len(m), c:c + m.shape[1]][m] = (bg + j + 1) % 10
    # Assign by the actual visible ranks, rather than construction indices.
    a = _analyze(g)
    available = [c for c in range(10) if c != bg]
    repeat = not (training and index < 2) and rng.random() < .35
    donor_colors = list(map(int, rng.choice(available, n, replace=repeat)))
    if training and index == 0:
        house_colors = donor_colors[::-1]
    else:
        house_colors = list(map(int, rng.permutation(available)[:n]))
        for _ in range(16):
            if all(h != d for h, d in zip(house_colors, donor_colors)):
                break
            house_colors = list(map(int, rng.permutation(available)[:n]))
        else:
            # Guaranteed color contrast even when cargo colors repeat.
            house_colors = [next(c for c in available if c != d) for d in donor_colors]
    for queue, colors in ((a['housings'], house_colors), (a['cargos'], donor_colors)):
        for o, color in zip(queue, colors):
            r, c = o['top'], o['left']
            g[r:r + len(o['mask']), c:c + o['mask'].shape[1]][o['mask']] = color
    return dict(input=g, output=solve(g))


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    value = int(value)
    if value < low or high is not None and value > high:
        raise ValueError(name + ' outside legal range')
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    nt = _integer(num_train, 'num_train', 2, 10)
    nq = _integer(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    puzzle, seen = dict(train=[], test=[]), set()
    fixed_bg = int(rng.integers(10))
    variable_bg = bool(rng.integers(2))
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            training = split == 'train'
            n = (2 if index == 0 else 3) if training and index < 2 else int(rng.integers(2, 4))
            if not training:
                n = 4 if index == 0 else int(rng.integers(2, 5))
            bg = int(rng.integers(10)) if variable_bg else fixed_bg
            for attempt in range(20):
                pair = _make_pair(rng, index, training, n, bg, index + 3 * (not training) + attempt)
                key = pair['input'].shape, pair['input'].tobytes()
                if key not in seen:
                    break
            else:
                raise RuntimeError('bounded uniqueness sampling exhausted')
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'pair counts')
    seen, analyses = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'unchanged canvas required')
            key = inp.shape, inp.astype(np.int8).tobytes()
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            a = _analyze(inp)
            changed_zone = np.zeros(inp.shape, bool)
            for d in a['cargos']:
                r, c = d['top'], d['left']
                view = out[r:r + len(d['mask']), c:c + d['mask'].shape[1]]
                _require(np.all(view[d['mask']] == a['background']), 'cargo source not erased')
                changed_zone[r:r + len(d['mask']), c:c + d['mask'].shape[1]] |= d['mask']
            for h, d in zip(a['housings'], a['cargos']):
                r, c = h['top'], h['left']
                hh, hw = h['mask'].shape
                patch = out[r:r + hh, c:c + hw]
                stamp, _, _ = _placement(h, d)
                coat = _dilate(stamp) & h['holes'] & ~stamp
                residual = h['holes'] & ~stamp & ~coat
                _require(np.all(patch[h['mask']] == h['color']), 'housing body changed')
                _require(np.all(patch[stamp] == d['color']), 'wrong cargo shape, color or centering')
                _require(np.all(patch[coat] == h['color']), 'wrong exterior coat')
                _require(np.all(patch[residual] == a['background']), 'residual cavity filled')
                changed_zone[r:r + hh, c:c + hw] |= h['holes']
            _require(np.array_equal(inp[~changed_zone], out[~changed_zone]), 'unrelated cells changed')
            if split == 'train':
                analyses.append(a)
    # Evidence conditions are measured from returned pixels, not sampler flags.
    for mode in ('nearest', 'horizontal'):
        _require(any(_witness(a, mode) for a in analyses), 'missing rank witness: ' + mode)
    _require(any(any(_separated_runs(d['mask']) for d in a['cargos']) for a in analyses), 'no open-bay cargo')
    _require(any(not np.array_equal(_render(a, neighbors=N4), _render(a)) for a in analyses), 'eight-neighbor coat not taught')
    _require(any(not np.array_equal(_render(a, coat=False), _render(a)) for a in analyses), 'coat invisible')
    _require(any(not np.array_equal(_render(a, align='corner'), _render(a)) for a in analyses), 'centering not taught')
    _require(any(not np.array_equal(_render(a, fill=True), _render(a)) for a in analyses), 'cavity-fill shortcut fits')
    _require(any(any(any(delta % 2 and delta >= 3 for delta in _placement(h, d)[2])
                             for h, d in zip(a['housings'], a['cargos'])) for a in analyses), 'odd centering tie not taught')
    _require(any(any(h['holes'].sum() < np.prod(_crop(h['holes']).shape) for h in a['housings']) for a in analyses), 'no irregular cavity demonstration')
    _require(any(any(_holes(np.pad(d['mask'], 1) | _dilate(np.pad(d['mask'], 1))).any()
                             for d in a['cargos']) for a in analyses), 'pre-outline topology not taught')


def _separated_runs(mask):
    return any(len(indices := np.flatnonzero(row)) > 1 and np.any(np.diff(indices) > 1)
               for rows in (mask, mask.T) for row in rows)


def _signature(p):
    return tuple((s, tuple((v['input'].shape, v['input'].tobytes(), v['output'].tobytes())
                           for v in p[s])) for s in ('train', 'test'))


def self_test():
    """42 streamed seeds, all legal count pairs, negative controls; no I/O."""
    shapes, cargo_masks, housing_masks, cavities, layouts, palettes, counts = (set() for _ in range(7))
    signatures = set()
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(hash(_signature(p)))
        for split in p:
            for pair in p[split]:
                a = _analyze(pair['input'])
                _require(np.array_equal(solve(pair['input'].copy()), pair['output']), 'solve disagrees')
                shapes.add(pair['input'].shape)
                counts.add((split, len(a['housings'])))
                palettes.add((a['background'], tuple(o['color'] for o in a['housings'] + a['cargos'])))
                layouts.add((pair['input'].shape, tuple((o['top'], o['left']) for o in a['housings'] + a['cargos'])))
                for objects, collection in ((a['housings'], housing_masks), (a['cargos'], cargo_masks)):
                    collection.update((o['mask'].shape, o['mask'].tobytes()) for o in objects)
                cavities.update((o['holes'].shape, o['holes'].tobytes()) for o in a['housings'])
    _require(len(signatures) == 42, 'seed diversity')
    _require(len(shapes) >= 15 and len(layouts) >= 180, 'layout diversity')
    _require(len(cargo_masks) >= 50 and len(housing_masks) >= 150 and len(cavities) >= 80, 'shape diversity')
    _require(len(palettes) >= 150 and {n for s, n in counts} == {2, 3, 4}, 'palette/count diversity')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(101 + nt * 4 + nq, nt, nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, 'explicit count')
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    _require(_signature(generate_puzzle(42)) == _signature(generate_puzzle(42)), 'determinism')
    # Every semantic zone is protected, including empty space left in a bay.
    p = generate_puzzle(17, 2, 1)
    a = _analyze(p['train'][1]['input'])
    h, d = next((h, d) for h, d in zip(a['housings'], a['cargos']) if d['mask'].shape == (5, 5))
    stamp, _, _ = _placement(h, d)
    coat = _dilate(stamp) & h['holes'] & ~stamp
    residual = h['holes'] & ~stamp & ~coat
    zones = [(d['mask'], d['top'], d['left']), (h['mask'], h['top'], h['left']),
             (stamp, h['top'], h['left']), (coat, h['top'], h['left']),
             (residual, h['top'], h['left']), (np.ones((1, 1), bool), 0, 0)]
    for mask, r, c in zones:
        _require(mask.any(), 'negative control zone absent')
        q = {s: [{k: np.array(v, copy=True) for k, v in pair.items()} for pair in p[s]] for s in p}
        rr, cc = np.argwhere(mask)[0]
        out = q['train'][1]['output']
        out[r + rr, c + cc] = (int(out[r + rr, c + cc]) + 1) % 10
        try:
            validate_generated_puzzle(q)
        except AssertionError:
            pass
        else:
            raise AssertionError('corrupt semantic zone accepted')
