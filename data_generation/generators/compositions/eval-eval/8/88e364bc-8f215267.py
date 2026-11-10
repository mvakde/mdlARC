"""Projected census: component-count-selected rays sample an irregular boundary.

Only standard library and NumPy are used. Nothing is read from disk at runtime.
All solving conventions are visible in one input; generation metadata is never
consulted by solve or validation. No execution traces or seed archives are saved.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np

CARDINAL = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIAGONAL = ((-1, -1), (-1, 1), (1, 1), (1, -1))
DIRECTIONS = CARDINAL + DIAGONAL
HOST_SEEDS = (0, 929560726, 941668306, 941390907, 826455248, 71707427,
              550615128, 705760961, 263973291, 525577939, 270452428,
              53291009, 472486471, 884971843, 862031495, 377269154,
              777092044, 185550616, 1696874, 268965732, 25389271,
              42614103, 508197227, 551415155, 717986574, 335393293,
              186957039, 902248538, 916920280, 502047524, 439899797,
              381310328, 25445291, 913621149, 79787029, 192892371,
              955675765, 512895434, 734056345, 835128566, 848192943,
              102095247)


def _check(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(name + ' outside legal range')
    return value


def _grid(value):
    g = np.asarray(value)
    _check(g.ndim == 2 and all(1 <= x <= 30 for x in g.shape), 'grid shape')
    _check(np.issubdtype(g.dtype, np.integer), 'grid dtype')
    _check(np.all((0 <= g) & (g <= 9)), 'grid colors')
    return g.astype(np.int64, copy=False)


def _parts(mask, eight=False):
    seen = ~mask.copy()
    result = []
    h, w = mask.shape
    for rr, cc in np.argwhere(mask):
        r, c = int(rr), int(cc)
        if seen[r, c]:
            continue
        cells, stack = [], [(r, c)]
        seen[r, c] = True
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in (DIRECTIONS if eight else CARDINAL):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        result.append(cells)
    return result


def _box(cells):
    rows, cols = zip(*cells)
    return min(rows), max(rows), min(cols), max(cols)


def _dilate(mask):
    h, w = mask.shape
    pad = np.pad(mask, 1)
    return np.logical_or.reduce([pad[1+dr:1+dr+h, 1+dc:1+dc+w]
                                 for dr, dc in ((0, 0),) + DIRECTIONS])


def _erode(mask, eight=False):
    h, w = mask.shape
    pad = np.pad(mask, 1)
    result = mask.copy()
    for dr, dc in (DIRECTIONS if eight else CARDINAL):
        result &= pad[1+dr:1+dr+h, 1+dc:1+dc+w]
    return result


def _ray(free, start, direction, corner=True):
    r, c = start
    dr, dc = direction
    path = [(r, c)]
    while (r+dr, c+dc) in free:
        if corner and dr and dc and ((r+dr, c) not in free or (r, c+dc) not in free):
            break
        r, c = r+dr, c+dc
        path.append((r, c))
    return path


def _keys(g, bg):
    keys = []
    for cells in _parts(g != bg):
        r0, r1, c0, c1 = _box(cells)
        h, w = r1-r0+1, c1-c0+1
        if min(h, w) < 4 or len(cells) != h*w:
            continue
        a = g[r0:r1+1, c0:c1+1]
        carrier = int(a[0, 0])
        if not (np.all(a[0] == carrier) and np.all(a[-1] == carrier)
                and np.all(a[:, 0] == carrier) and np.all(a[:, -1] == carrier)):
            continue
        other = set(map(int, np.unique(a))) - {carrier}
        if len(other) != 2:
            continue
        interpretations = []
        for tip in sorted(other):
            trail = next(iter(other - {tip}))
            ends = set(map(tuple, np.argwhere(a == tip)))
            ink = set(map(tuple, np.argwhere(a == trail)))
            if not 2 <= len(ends) <= 3:
                continue
            for d in DIRECTIONS:
                used, lengths = set(), []
                for rr, cc in sorted(ends):
                    stroke = []
                    rr, cc = rr-d[0], cc-d[1]
                    while (rr, cc) in ink:
                        stroke.append((rr, cc))
                        rr, cc = rr-d[0], cc-d[1]
                    if len(stroke) < 2 or used.intersection(stroke):
                        break
                    used.update(stroke)
                    lengths.append(len(stroke))
                if len(lengths) == len(ends) and used == ink:
                    interpretations.append((trail, tip, d))
        _check(len(interpretations) == 1, 'ambiguous legend')
        trail, tip, direction = interpretations[0]
        keys.append(dict(carrier=carrier, trail=trail, tip=tip,
                         direction=direction, cells=set(cells), box=(r0, r1, c0, c1)))
    _check(1 <= len(keys) <= 2, 'expected one or two legends')
    _check(len({k['carrier'] for k in keys}) == len(keys), 'duplicate carrier')
    _check(len({(k['trail'], k['tip']) for k in keys}) == 1, 'conflicting ink conventions')
    return keys


def _read(g):
    counts = np.bincount(g.ravel(), minlength=10)
    _check(np.count_nonzero(counts == counts.max()) == 1, 'ambiguous background')
    bg = int(counts.argmax())
    keys = _keys(g, bg)
    trail, tip = keys[0]['trail'], keys[0]['tip']
    carriers = {k['carrier'] for k in keys}
    _check(not carriers.intersection((bg, trail, tip)), 'color roles conflict')
    fixed = set().union(*(k['cells'] for k in keys))
    key_mask = np.zeros(g.shape, bool)
    for p in fixed:
        key_mask[p] = True
    passable = np.isin(g, (bg, trail, tip)) & ~key_mask
    systems, enclosing = [], set()
    for cells in _parts(passable):
        ink = [p for p in cells if g[p] != bg]
        if not ink:
            continue
        region = set(cells)
        _check(not any(r in (0, g.shape[0]-1) or c in (0, g.shape[1]-1)
                       for r, c in cells), 'ruler is not enclosed')
        border = {(r+dr, c+dc) for r, c in cells for dr, dc in CARDINAL
                  if (r+dr, c+dc) not in region}
        colors = {int(g[p]) for p in border}
        _check(len(colors) == 1 and colors <= carriers, 'mixed chamber boundary')
        color = next(iter(colors))
        key = next(k for k in keys if k['carrier'] == color)
        caps = [p for p in ink if g[p] == tip]
        _check(len(caps) == 1, 'ruler must have exactly one cap')
        cap = caps[0]
        _check(len(ink) >= 5 and len(ink) % 2 == 1, 'ruler length')
        ruler_set = set(ink)
        readings = []
        for pr, pc in DIRECTIONS:
            if pr*key['direction'][0] + pc*key['direction'][1] != 0:
                continue
            ordered = [(cap[0]+j*pr, cap[1]+j*pc) for j in range(len(ink))]
            if set(ordered) == ruler_set and all(g[p] == trail for p in ordered[1:]):
                readings.append((ordered, (pr, pc)))
        _check(len(readings) == 1, 'ruler reading not unique')
        ordered, step = readings[0]
        _check(all(all((r+dr, c+dc) in region for dr, dc in DIRECTIONS)
                   for r, c in ordered), 'ruler touches wall')
        wall = set()
        for component in _parts((g == color) & ~key_mask, eight=True):
            if border.intersection(component):
                wall.update(component)
        _check(border <= wall, 'incomplete wall association')
        paths = [_ray(region, p, key['direction']) for p in ordered[2::2]]
        _check(all(len(path) > 1 for path in paths), 'blocked launch site')
        _check(all(not (set(path[1:]) & ruler_set) for path in paths), 'ray crosses ruler')
        _check(sum(map(len, paths)) == len(set().union(*map(set, paths))), 'intersecting rays')
        systems.append(dict(color=color, key=key, free=region, wall=wall,
                            ruler=ordered, step=step, capacity=len(paths), paths=paths))
        fixed.update(wall | ruler_set)
        enclosing.update(region | wall)
    _check(len(systems) == len(keys), 'missing or duplicate chamber')
    _check({s['color'] for s in systems} == carriers, 'chamber/key mismatch')
    remaining = (g != bg).copy()
    for p in fixed:
        remaining[p] = False
    objects = []
    for color in sorted(set(map(int, g[remaining]))):
        _check(color not in (trail, tip), 'reference ink outside structures')
        for cells in _parts(remaining & (g == color)):
            _check(1 <= len(cells) <= 9 and not enclosing.intersection(cells), 'external object')
            r0, r1, c0, c1 = _box(cells)
            silhouette = np.zeros((r1-r0+3, c1-c0+3), bool)
            for r, c in cells:
                silhouette[r-r0+1, c-c0+1] = True
            _check(len(_parts(~silhouette)) == 1, 'hollow external object')
            objects.append(dict(color=color, cells=set(cells), area=len(cells)))
    # Distinct exterior components and fixed structures have an eight-neighbor moat.
    occupied = set(fixed)
    for o in objects:
        halo = {(r+dr, c+dc) for r, c in o['cells'] for dr, dc in ((0, 0),)+DIRECTIONS}
        _check(not occupied.intersection(halo), 'objects touch structures or one another')
        occupied.update(o['cells'])
    for s in systems:
        matched = [o for o in objects if o['color'] == s['color']]
        s['count'] = len(matched)
        s['area'] = sum(o['area'] for o in matched)
        _check(s['count'] <= s['capacity'], 'census exceeds ruler capacity')
    _check(sum(s['count'] for s in systems) > 0, 'no matched objects')
    return dict(background=bg, trail=trail, tip=tip, keys=keys, systems=systems,
                fixed=fixed, objects=objects)


def _answer(g, info, mode='normal'):
    out = g.copy()
    for o in info['objects']:
        for p in o['cells']:
            out[p] = info['background']
    for s in info['systems']:
        n = s['count']
        if mode == 'all_ticks':
            n = s['capacity']
        elif mode == 'area':
            n = min(s['area'], s['capacity'])
        starts = s['ruler'][2::2]
        if mode == 'reverse_ruler':
            starts = s['ruler'][-3::-2]
        for start in starts[:n]:
            d = s['key']['direction']
            if mode == 'reverse_arrow':
                d = (-d[0], -d[1])
            p = start if mode == 'no_motion' else _ray(s['free'], start, d,
                                                      corner=mode != 'no_corner')[-1]
            out[p] = info['tip']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode a single input. No examples, seed, files, or latent state are used."""
    g = _grid(input_grid)
    return _answer(g, _read(g))


def _polyomino(rng, area):
    # Independent of category count. Reject holes, and include thin/branched masks.
    for _ in range(20):
        cells = {(0, 0)}
        if rng.random() < .23:
            cells = {(0, j) for j in range(area)}
        while len(cells) < area:
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in CARDINAL} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
        for _ in range(int(rng.integers(4))):
            cells = {(c, -r) for r, c in cells}
        if rng.random() < .5:
            cells = {(r, -c) for r, c in cells}
        r0, r1, c0, c1 = _box(cells)
        mask = np.zeros((r1-r0+3, c1-c0+3), bool)
        for r, c in cells:
            mask[r-r0+1, c-c0+1] = True
        if len(_parts(~mask)) == 1:
            return tuple(sorted((r-r0, c-c0) for r, c in cells))
    raise AssertionError('bounded object construction failed')


def _legend(rng, d, carrier, trail, tip):
    length = int(rng.choice((2, 2, 3)))
    # Two parallel, optionally staggered strokes; spacing prevents accidental joins.
    dr, dc = d
    base_step = (2, 0) if dr == 0 else (0, 2)
    points = []
    for i in range(2):
        stagger = int(rng.integers(2))
        for j in range(length+1):
            points.append((i*base_step[0]+(j+stagger)*dr,
                           i*base_step[1]+(j+stagger)*dc,
                           tip if j == length else trail))
    r0, r1, c0, c1 = _box([(r, c) for r, c, _ in points])
    pads = [int(rng.choice((1, 1, 2))) for _ in range(4)]
    a = np.full((r1-r0+1+pads[0]+pads[1], c1-c0+1+pads[2]+pads[3]), carrier, np.int8)
    for r, c, color in points:
        a[r-r0+pads[0], c-c0+pads[2]] = color
    return a


def _chamber(rng, d, sign, capacity, count, corner=False, nonflat=False):
    pr, pc = -d[1]*sign, d[0]*sign
    length = 2*capacity+1
    min_h = max(9, abs(pr)*(length-1)+5)
    min_w = max(9, abs(pc)*(length-1)+5)
    for attempt in range(85):
        h = int(rng.integers(min_h, max(min_h+1, 15)))
        w = int(rng.integers(min_w, max(min_w+1, 15)))
        footprint = np.ones((h, w), bool)
        kind = int(rng.choice(3, p=(.20, .35, .45)))
        if kind == 1:
            for k in rng.choice(4, int(rng.integers(1, 4)), replace=False):
                dh, dw = int(rng.integers(1, 4)), int(rng.integers(1, 4))
                rs = slice(0, dh) if k < 2 else slice(h-dh, h)
                cs = slice(0, dw) if k % 2 == 0 else slice(w-dw, w)
                footprint[rs, cs] = False
        elif kind == 2:
            # Stepped shores with broad flat spans, rather than a finite shape bank.
            left, right = int(rng.integers(2)), int(rng.integers(2))
            for r in range(h):
                if rng.random() < .32:
                    left = int(rng.integers(0, 3))
                if rng.random() < .32:
                    right = int(rng.integers(0, 3))
                footprint[r, :left] = False
                if right:
                    footprint[r, w-right:] = False
        wall = footprint & ~_erode(footprint)
        if corner or rng.random() < .60:
            for _ in range(int(rng.integers(1, 3))):
                if rng.random() < .5:
                    c = int(rng.integers(3, w-3))
                    cut = int(rng.integers(2, h-2))
                    rs = slice(0, cut) if rng.random() < .5 else slice(cut, h)
                    wall[rs, c] |= footprint[rs, c]
                else:
                    r = int(rng.integers(3, h-3))
                    cut = int(rng.integers(2, w-2))
                    cs = slice(0, cut) if rng.random() < .5 else slice(cut, w)
                    wall[r, cs] |= footprint[r, cs]
        free_mask = footprint & ~wall
        safe = _erode(free_mask, eight=True)
        free = set(map(tuple, np.argwhere(free_mask)))
        candidates = []
        for rr, cc in np.argwhere(safe):
            r, c = int(rr), int(cc)
            ruler = [(r+j*pr, c+j*pc) for j in range(length)]
            if not all(0 <= a < h and 0 <= b < w and safe[a, b] for a, b in ruler):
                continue
            paths = [_ray(free, p, d) for p in ruler[2::2]]
            if not all(len(path) > 1 for path in paths):
                continue
            # A baffle may separate chambers; every reference cell must share a region.
            active = paths[:count]
            witness = any(_ray(free, p[0], d, corner=False)[-1] != p[-1] for p in active)
            if corner and not witness:
                continue
            if nonflat and len({len(p) for p in active}) < 2:
                continue
            candidates.append((ruler, paths))
        if candidates:
            ruler, paths = candidates[int(rng.integers(len(candidates)))]
            component = next(set(p) for p in _parts(free_mask) if ruler[0] in p)
            if not set(ruler) <= component:
                continue
            return dict(footprint=footprint, wall=wall, ruler=ruler, paths=paths,
                        direction=d, count=count, capacity=capacity)
    raise AssertionError('bounded chamber construction failed')


def _positions(mask, occupied):
    h, w = mask.shape
    H, W = occupied.shape
    if h > H or w > W:
        return np.empty((0, 2), int)
    valid = np.ones((H-h+1, W-w+1), bool)
    for dr, dc in np.argwhere(mask):
        valid &= ~occupied[dr:dr+H-h+1, dc:dc+W-w+1]
    return np.argwhere(valid)


def _place_scene(rng, systems, keys, object_specs, colors, bg, trail, tip):
    # One bounded scene in memory. No packing candidates or failed scenes are retained.
    n = len(systems)
    for attempt in range(22):
        H = 30 if attempt > 12 else int(rng.integers(25 if n == 1 else 28, 31))
        W = 30 if attempt > 12 else int(rng.integers(25 if n == 1 else 28, 31))
        occupied = np.zeros((H, W), bool)
        a = np.full((H, W), bg, np.int8)
        b = a.copy()
        objects = []
        for i, s in enumerate(systems):
            objects.append(dict(kind='chamber', index=i, mask=s['footprint']))
        # Independent placement order: no key above or aligned to its matching chamber.
        chambers = sorted(objects, key=lambda o: -int(o['mask'].sum()))
        objects = chambers + [dict(kind='key', index=int(i), mask=np.ones(keys[int(i)].shape, bool))
                              for i in rng.permutation(n)]
        for color, cells in sorted(object_specs, key=lambda spec: -len(spec[1])):
            h, w = 1+max(r for r, c in cells), 1+max(c for r, c in cells)
            mask = np.zeros((h, w), bool)
            for p in cells:
                mask[p] = True
            objects.append(dict(kind='object', color=color, mask=mask))
        success = True
        for o in objects:
            candidates = _positions(o['mask'], _dilate(occupied))
            if not len(candidates):
                success = False
                break
            r, c = map(int, candidates[int(rng.integers(len(candidates)))])
            h, w = o['mask'].shape
            occupied[r:r+h, c:c+w] |= o['mask']
            if o['kind'] == 'key':
                a[r:r+h, c:c+w] = keys[o['index']]
                b[r:r+h, c:c+w] = keys[o['index']]
            elif o['kind'] == 'object':
                a[r:r+h, c:c+w][o['mask']] = o['color']
            else:
                i = o['index']
                s = systems[i]
                a[r:r+h, c:c+w][s['wall']] = colors[i]
                b[r:r+h, c:c+w][s['wall']] = colors[i]
                for j, (rr, cc) in enumerate(s['ruler']):
                    a[r+rr, c+cc] = b[r+rr, c+cc] = tip if j == 0 else trail
                for path in s['paths'][:s['count']]:
                    rr, cc = path[-1]
                    b[r+rr, c+cc] = tip
        if success:
            # Optional tight cropping changes shape without altering the coordinate rule.
            if rng.random() < .65:
                r0, r1, c0, c1 = _box(np.argwhere(a != bg))
                # Retain an exterior border so closed-region recognition is unchanged.
                if r0 > 0 and c0 > 0 and r1 < H-1 and c1 < W-1:
                    r0, c0 = max(0, r0-int(rng.integers(1, 4))), max(0, c0-int(rng.integers(1, 4)))
                    r1, c1 = min(H-1, r1+int(rng.integers(1, 4))), min(W-1, c1+int(rng.integers(1, 4)))
                    a, b = a[r0:r1+1, c0:c1+1].copy(), b[r0:r1+1, c0:c1+1].copy()
            frequencies = np.bincount(a.ravel(), minlength=10)
            if int(frequencies.argmax()) != bg or np.count_nonzero(frequencies == frequencies.max()) != 1:
                continue
            return dict(input=a, output=b)
    return None


def _make_pair(rng, plans, colors, bg, forced_areas):
    pool = [c for c in range(10) if c not in colors and c != bg]
    trail, tip = map(int, rng.choice(pool, 2, replace=False))
    for _ in range(14):
        systems = [_chamber(rng, p['direction'], p['sign'], p['capacity'], p['count'],
                            p.get('corner', False), p.get('nonflat', False)) for p in plans]
        keys = [_legend(rng, p['direction'], color, trail, tip) for p, color in zip(plans, colors)]
        specs = []
        for i, (p, color) in enumerate(zip(plans, colors)):
            areas = forced_areas.get(i, [int(rng.integers(1, 10)) for _ in range(p['count'])])
            specs += [(color, _polyomino(rng, area)) for area in areas]
        dpool = [c for c in pool if c not in (trail, tip)]
        for _ in range(int(rng.integers(0, 4))):
            specs.append((int(rng.choice(dpool)), _polyomino(rng, int(rng.integers(1, 10)))))
        pair = _place_scene(rng, systems, keys, specs, colors, bg, trail, tip)
        if pair is not None:
            # Reject rare silhouettes with accidentally unassociated internal reference ink.
            try:
                detail = _read(pair['input'])
                if np.array_equal(_answer(pair['input'], detail), pair['output']):
                    return pair
            except AssertionError:
                pass
    raise AssertionError('bounded pair packing failed')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    nt = _integer(num_train, 'num_train', 2, 10)
    nq = _integer(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    anchor, novel = map(int, rng.choice(10, 2, replace=False))
    primary = CARDINAL[int(rng.integers(4))]
    diagonal = DIAGONAL[int(rng.integers(4))]
    polarity = int(rng.choice((-1, 1)))
    area = int(rng.integers(4, 10))
    split_area = int(rng.integers(1, area))
    other_total = int(rng.choice([a for a in range(4, 11) if a != area]))
    other_split = int(rng.integers(max(1, other_total-9), min(9, other_total-1)+1))
    puzzle, seen = {'train': [], 'test': []}, set()
    previous_bg = None
    for i in range(nt+nq):
        train = i < nt
        query_index = i-nt
        forced = {}
        if i == 0:
            plans = [dict(direction=primary, sign=polarity, capacity=3, count=1),
                     dict(direction=diagonal, sign=polarity, capacity=3, count=3,
                          nonflat=True)]
            forced[0] = [area]
        elif i == 1:
            plans = [dict(direction=primary, sign=-polarity, capacity=3, count=2),
                     dict(direction=diagonal, sign=-polarity, capacity=3, count=2,
                          corner=True, nonflat=True)]
            forced = {0: [split_area, area-split_area],
                      1: [other_split, other_total-other_split]}
        else:
            n = int(rng.choice((1, 2), p=(.48, .52)))
            plans = []
            for j in range(n):
                cap = int(rng.integers(2, 5 if n == 1 else 4))
                count = int(rng.integers(0, cap+1))
                d = DIRECTIONS[int(rng.integers(8))]
                if not train and query_index == 0 and j == 0:
                    d = (-primary[0], -primary[1])
                    count = cap if n == 1 else int(rng.integers(1, cap+1))
                plans.append(dict(direction=d, sign=-polarity if not train and query_index == 0
                                  else int(rng.choice((-1, 1))), capacity=cap, count=count))
            if not any(p['count'] for p in plans):
                plans[0]['count'] = 1
        n = len(plans)
        color_pool = [c for c in range(10) if not train or c != novel]
        if i < 2:
            colors = [anchor, int(rng.choice([c for c in color_pool if c != anchor]))]
        elif not train and query_index == 0:
            colors = [novel] + list(map(int, rng.choice([c for c in range(10) if c != novel], n-1, replace=False)))
        else:
            colors = list(map(int, rng.choice(color_pool, n, replace=False)))
        bg = int(rng.choice([c for c in range(10) if c not in colors and c != previous_bg]))
        for _ in range(5):
            pair = _make_pair(rng, plans, colors, bg, forced)
            fingerprint = (pair['input'].shape, pair['input'].tobytes())
            if fingerprint not in seen:
                break
        _check(fingerprint not in seen, 'duplicate input construction')
        seen.add(fingerprint)
        puzzle['train' if train else 'test'].append(pair)
        previous_bg = bg
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _check(all(isinstance(puzzle[s], list) for s in puzzle), 'split schema')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _check(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'pair counts')
    details, seen = {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _check(a.shape == b.shape, 'changed dimensions')
            fingerprint = (a.shape, a.astype(np.int8).tobytes())
            _check(fingerprint not in seen, 'duplicate input')
            seen.add(fingerprint)
            info = _read(a)
            _check(np.array_equal(_answer(a, info), b), 'wrong projection, census, or erasure')
            _check(all(a[p] == b[p] for p in info['fixed']), 'reference ink changed')
            _check(all(b[p] == info['background'] for o in info['objects'] for p in o['cells']), 'field not erased')
            count = sum(s['count'] for s in info['systems'])
            _check(np.count_nonzero(b == info['tip'])-np.count_nonzero(a == info['tip']) == count,
                   'one new dot per matched component required')
            _check(not np.array_equal(a, b), 'identity pair')
            details[split].append((a, b, info))
    training = details['train']
    categories = [(s['color'], s['count'], s['area']) for _, _, d in training for s in d['systems']]
    _check(any(c == cc and n != nn and a == aa for c, n, a in categories for cc, nn, aa in categories),
           'missing same-color area collision')
    _check(any(n == nn and a != aa for c, n, a in categories for cc, nn, aa in categories),
           'missing count/area independence')
    _check({1, 2, 3} <= {n for _, n, _ in categories}, 'missing unary progression')
    directions = {s['key']['direction'] for _, _, d in training for s in d['systems']}
    _check(bool(directions.intersection(CARDINAL)) and bool(directions.intersection(DIAGONAL)),
           'missing cardinal/diagonal teaching')
    _check(any(len({len(p) for p in s['paths'][:s['count']]}) > 1
               for _, _, d in training for s in d['systems']), 'missing contour witness')
    signs = {int(np.sign(s['step'][0]*(-s['key']['direction'][1])
                         + s['step'][1]*s['key']['direction'][0]))
             for _, _, d in training for s in d['systems']}
    _check(signs == {-1, 1}, 'missing ruler polarity teaching')
    for mode in ('all_ticks', 'area', 'no_motion', 'reverse_arrow', 'reverse_ruler', 'no_corner'):
        _check(any(not np.array_equal(_answer(a, d, mode), b) for a, b, d in training),
               'training shortcut survives: '+mode)
    train_colors = {c for c, _, _ in categories}
    _check(any(s['color'] not in train_colors for s in details['test'][0][2]['systems']),
           'missing query color-role generalization')
    return True


def puzzle_metrics(puzzle):
    """Small grid-derived records, for bounded review without retaining puzzles."""
    rows = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            a = np.asarray(pair['input'])
            d = _read(a)
            irregular = 0
            for s in d['systems']:
                r0, r1, c0, c1 = _box(s['wall'])
                perimeter = {(r, c) for r in range(r0, r1+1) for c in range(c0, c1+1)
                             if r in (r0, r1) or c in (c0, c1)}
                irregular += not perimeter <= s['wall']
            rows.append(dict(split=split, shape=a.shape, background=d['background'],
                             systems=len(d['systems']), irregular=irregular,
                             directions=[s['key']['direction'] for s in d['systems']],
                             counts=[s['count'] for s in d['systems']],
                             capacities=[s['capacity'] for s in d['systems']],
                             areas=[o['area'] for o in d['objects']],
                             masks=[tuple(sorted((r-min(x[0] for x in o['cells']),
                                                  c-min(x[1] for x in o['cells']))
                                                 for r, c in o['cells'])) for o in d['objects']],
                             key_shapes=[(k['box'][1]-k['box'][0]+1, k['box'][3]-k['box'][2]+1)
                                         for k in d['keys']]))
    return rows


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 fixed seeds; no files, growing logs, traces, or puzzle archive."""
    fingerprints, shapes, backgrounds, directions, masks, keys = (set() for _ in range(6))
    counts, capacities, systems, areas = (set() for _ in range(4))
    irregular, pairs = 0, 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        for r in puzzle_metrics(puzzle):
            pairs += 1
            shapes.add(r['shape'])
            backgrounds.add(r['background'])
            directions.update(r['directions'])
            masks.update(r['masks'])
            keys.update(r['key_shapes'])
            counts.update(r['counts'])
            capacities.update(r['capacities'])
            systems.add(r['systems'])
            areas.update(r['areas'])
            irregular += r['irregular']
        # Geometric equivariance exercises the input-only parser in new orientations.
        pair = puzzle['test'][0]
        for turn in (1, 2, 3):
            _check(np.array_equal(solve(np.rot90(pair['input'], turn)),
                                  np.rot90(pair['output'], turn)), 'rotation invariance')
        permutation = np.random.default_rng(seed+1).permutation(10)
        _check(np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']]),
               'palette equivariance')
    _check(len(fingerprints) == 42 and len(shapes) >= 12, 'seed/shape diversity')
    _check(backgrounds == set(range(10)) and directions == set(DIRECTIONS), 'palette/direction diversity')
    _check(systems == {1, 2} and counts >= {0, 1, 2, 3, 4}, 'count variation')
    _check(capacities == {2, 3, 4} and areas >= set(range(1, 10)), 'scale variation')
    _check(len(masks) >= 70 and len(keys) >= 12 and irregular > 35, 'structural variation')
    a, b = generate_puzzle(17, 2, 1), generate_puzzle(17, 2, 1)
    _check(_fingerprint(a) == _fingerprint(b), 'determinism')
    # Host separately exercises every legal count. These cover maximum memory and minima.
    generate_puzzle(23, 10, 4)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid counts accepted')
    a['test'][0]['output'] = a['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(a)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepts identity corruption')
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(fingerprints),
                grid_shapes=len(shapes), backgrounds=sorted(backgrounds),
                directions=sorted(directions), counts=sorted(counts),
                capacities=sorted(capacities), object_areas=sorted(areas),
                object_masks=len(masks), key_shapes=len(keys), irregular_systems=irregular)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
