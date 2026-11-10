"""Two-color spotlight, region stencil, and color-specific straight shadows.

Only standard-library modules and NumPy are used. All recognition in solve is
input-only. Construction renders from explicit masks; solve uses lane scans.
"""
from __future__ import annotations

import copy
import json
import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(np.all((0 <= a) & (a <= 9)), "ARC colors required")
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    cells = set(map(tuple, np.argwhere(mask)))
    steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
             if (dr or dc) and (diagonal or abs(dr) + abs(dc) == 1)]
    result = []
    while cells:
        start = min(cells)
        cells.remove(start)
        pending, group = [start], [start]
        while pending:
            r, c = pending.pop()
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in cells:
                    cells.remove(q)
                    pending.append(q)
                    group.append(q)
        result.append(group)
    return result


def _mask(shape, cells):
    m = np.zeros(shape, dtype=bool)
    for p in cells:
        m[p] = True
    return m


def _regions(cut):
    groups = _components(~cut)
    _require(len(groups) == 2, "cut must separate exactly two regions")
    return [_mask(cut.shape, g) for g in groups]


def _check_path(cut):
    cells = set(map(tuple, np.argwhere(cut)))
    graph = {}
    for r, c in cells:
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                q = r + dr, c + dc
                if not (dr or dc) or q not in cells:
                    continue
                if dr and dc and ((r + dr, c) in cells or (r, c + dc) in cells):
                    continue  # No diagonal shortcut across an orthogonal elbow.
                neighbors.append(q)
        graph[r, c] = neighbors
    ends = [p for p, n in graph.items() if len(n) == 1]
    _require(len(cells) >= 4 and len(ends) == 2
             and all(1 <= len(n) <= 2 for n in graph.values()), "simple cut path")
    h, w = cut.shape
    border = lambda p: p[0] in (0, h - 1) or p[1] in (0, w - 1)
    _require(all(border(p) for p in ends), "cut endpoints must touch canvas edge")
    _require(all(not border(p) for p in cells - set(ends)), "extra boundary contact")
    _require(len(_components(cut, True)) == 1, "disconnected cut")
    return _regions(cut)


def _near(mask):
    result = mask.copy()
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        result[max(0, r - 1):min(h, r + 2), max(0, c - 1):min(w, c + 2)] = True
    return result


def _fit_rail(rows, sign):
    """Longest terminal thin stair with period 1 or 2 and two outward steps."""
    xs = np.arange(len(rows))
    candidates = []
    for period in (1, 2):
        for phase in range(period):
            predicted = rows[-1] + sign * ((xs - xs[-1] + phase) // period)
            length = 0
            for same in (predicted == rows)[::-1]:
                if not same:
                    break
                length += 1
            if length >= 2 * period + 1 and abs(int(rows[-length] - rows[-1])) >= 2:
                candidates.append((length, -period, phase))
    _require(candidates, "terminal rail must expose two periodic outward steps")
    length, negative_period, phase = max(candidates)
    return -negative_period, phase, length


def _emitter(grid, bg):
    matches = []
    for rotation in range(4):
        a = np.rot90(grid, -rotation)
        h, w = a.shape
        for color in np.unique(a):
            wall = int(color)
            if wall == bg:
                continue
            wall_mask = a == wall
            positions = np.argwhere(wall_mask)
            if len(positions) < 13 or len(_components(wall_mask, True)) != 1:
                continue
            rear, mouth = map(int, (positions[:, 1].min(), positions[:, 1].max()))
            if mouth - rear < 4 or mouth >= w - 2:
                continue
            cap = np.flatnonzero(wall_mask[:, rear])
            if len(cap) < 5 or np.any(np.diff(cap) != 1):
                continue
            rails = [np.flatnonzero(wall_mask[:, x]) for x in range(rear + 1, mouth + 1)]
            if any(len(rows) != 2 or rows[1] - rows[0] < 3 for rows in rails):
                continue
            upper = np.array([rows[0] for rows in rails])
            lower = np.array([rows[1] for rows in rails])
            if (abs(int(upper[0] - cap[0])) > 1 or abs(int(lower[0] - cap[-1])) > 1
                    or np.any(abs(np.diff(upper)) > 1) or np.any(abs(np.diff(lower)) > 1)):
                continue
            try:
                fits = _fit_rail(upper, -1), _fit_rail(lower, 1)
            except AssertionError:
                continue
            chamber = np.zeros(a.shape, dtype=bool)
            footprint = np.zeros_like(chamber)
            rr = np.arange(h)
            for x in range(rear + 1, w):
                if x <= mouth:
                    lo, hi = int(upper[x - rear - 1]), int(lower[x - rear - 1])
                else:
                    lo = int(upper[-1]) - (x - mouth + fits[0][1]) // fits[0][0]
                    hi = int(lower[-1]) + (x - mouth + fits[1][1]) // fits[1][0]
                footprint[:, x] = (rr > lo) & (rr < hi)
                if x <= mouth:
                    chamber[:, x] = footprint[:, x]
            patches = chamber & (a != bg) & ~wall_mask
            colors = list(map(int, np.unique(a[patches])))
            if len(colors) != 2:
                continue
            patch_positions = [np.argwhere(chamber & (a == c)) for c in colors]
            # Two separated, connected payload patches; top is marked-side source.
            order = np.argsort([p[:, 0].min() for p in patch_positions])
            ca, cb = colors[int(order[0])], colors[int(order[1])]
            pa, pb = patch_positions[int(order[0])], patch_positions[int(order[1])]
            if pa[:, 0].max() >= pb[:, 0].min():
                continue
            if any(not 1 <= len(p) <= 4 or p[:, 1].min() != rear + 1 for p in (pa, pb)):
                continue
            if any(len(_components(chamber & (a == c))) != 1 for c in (ca, cb)):
                continue
            if len(_components(chamber)) != 1:
                continue
            remaining = set(map(int, np.unique(a))) - {bg, wall, ca, cb}
            if len(remaining) != 1:
                continue
            control = remaining.pop()
            groups = sorted(_components(a == control, True), key=len)
            if len(groups) != 2 or len(groups[0]) != 1 or len(groups[1]) < 4:
                continue
            marker = groups[0][0]
            cut = _mask(a.shape, groups[1])
            if np.any(cut & (chamber | wall_mask)) or chamber[marker]:
                continue
            try:
                regions = _check_path(cut)
            except AssertionError:
                continue
            if _near(cut)[marker]:
                continue
            side = int(not regions[0][marker])
            screens = ((a == ca) | (a == cb)) & ~chamber
            matches.append(dict(a=a, rotation=rotation, bg=bg, wall=wall, ca=ca, cb=cb,
                                control=control, rear=rear, mouth=mouth, fits=fits,
                                chamber=chamber, footprint=footprint, cut=cut, marker=marker,
                                regions=regions, side=side, screens=screens,
                                upper=upper, lower=lower))
    _require(len(matches) == 1, "input must identify a unique emitter and stencil")
    return matches[0]


def _decode(input_grid):
    grid = _grid(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(counts[bg] > grid.size // 2, "input background must dominate")
    return _emitter(grid, bg)


def _availability(d):
    """Each color remembers its own blockers, independent of the selected region."""
    a, cone = d['a'], d['footprint']
    available = [np.zeros(a.shape, dtype=bool), np.zeros(a.shape, dtype=bool)]
    for y in range(a.shape[0]):
        blocked = [False, False]
        for x in range(d['rear'] + 1, a.shape[1]):
            if not cone[y, x]:
                continue
            if d['screens'][y, x]:
                blocked[0 if a[y, x] == d['ca'] else 1] = True
            for i in range(2):
                available[i][y, x] = not blocked[i]
    return available


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recognize the scene and draw the stencil-selected surviving illumination."""
    d = _decode(input_grid)
    a, bg = d['a'], d['bg']
    va, vb = _availability(d)
    out = a.copy()
    for y, x in np.argwhere(d['footprint']):
        p = int(y), int(x)
        if p == d['marker'] or d['screens'][p] or d['chamber'][p] and a[p] != bg:
            continue
        if a[p] not in (bg, d['control']):
            continue
        if d['cut'][p]:
            out[p] = d['ca'] if va[p] else d['cb'] if vb[p] else bg
        elif d['regions'][d['side']][p]:
            out[p] = d['ca'] if va[p] else bg
        else:
            out[p] = d['cb'] if vb[p] else bg
    return np.rot90(out, d['rotation']).copy()


def _render_masks(a, bg, ca, cb, footprint, chamber, cut, marker, marked):
    # Construction uses unions of downstream masks, separately from solve's scan.
    screens = ((a == ca) | (a == cb)) & ~chamber
    shadow_a, shadow_b = np.zeros(a.shape, bool), np.zeros(a.shape, bool)
    for y, x in np.argwhere(screens & footprint):
        (shadow_a if a[y, x] == ca else shadow_b)[y, x + 1:] = True
    va, vb = footprint & ~shadow_a, footprint & ~shadow_b
    out = a.copy()
    eligible = footprint & ((a == bg) | cut)
    eligible[marker] = False
    out[eligible] = bg
    out[eligible & ~cut & marked & va] = ca
    out[eligible & ~cut & ~marked & vb] = cb
    out[eligible & cut & vb] = cb
    out[eligible & cut & va] = ca
    return out


def _sample_geometry(rng, long=False):
    periods = [int(rng.integers(1, 3)), int(rng.integers(1, 3))]
    phases = [int(rng.integers(p)) for p in periods]
    body_length = int(rng.integers(2, 4))
    starts = [int(rng.integers(2, 4)), int(rng.integers(2, 4))]
    profiles = []
    for start, period, phase in zip(starts, periods, phases):
        body = [start] * body_length
        if body_length == 3 and start == 3 and rng.random() < .5:
            body[1] = 4  # A reservoir bulge, independent of the final flare.
        extension = [body[-1] + (t + phase) // period
                     for t in range(1, 2 * max(periods) + 2)]
        profiles.append(body + extension)
    rear = int(rng.integers(0, 3))
    mouth = rear + len(profiles[0])
    projection_low = 14 if long else 10
    projection = int(rng.integers(projection_low, min(19 if long else 14, 30 - mouth)))
    w = mouth + 1 + projection
    hmin = max(16, max(profiles[0]) + max(profiles[1]) + 3)
    h = int(rng.integers(hmin, 31))
    center = int(rng.integers(max(profiles[0]) + 1, h - max(profiles[1]) - 1))
    wall, footprint, chamber = (np.zeros((h, w), bool) for _ in range(3))
    wall[center - starts[0]:center + starts[1] + 1, rear] = True
    for j, (up, down) in enumerate(zip(*profiles), 1):
        x = rear + j
        wall[center - up, x] = wall[center + down, x] = True
        chamber[center - up + 1:center + down, x] = True
    footprint[:] = chamber
    for x in range(mouth + 1, w):
        t = x - (rear + body_length)
        lo = center - starts[0] - (t + phases[0]) // periods[0]
        hi = center + starts[1] + (t + phases[1]) // periods[1]
        footprint[max(0, lo + 1):min(h, hi), x] = True
    return wall, footprint, chamber, rear, mouth, center


def _cut(rng, shape, mouth, center, style):
    h, w = shape
    cut = np.zeros(shape, bool)
    x0 = int(rng.integers(mouth + 2, mouth + 4))
    last = w - 3
    if style == 'vertical':
        cut[:, int(rng.integers(x0, last + 1))] = True
    elif style == 'diagonal':
        cols = np.rint(np.linspace(x0, last, h)).astype(int)
        if rng.random() < .5:
            cols = cols[::-1]
        cut[np.arange(h), cols] = True
    elif style == 'elbow':
        row = int(rng.integers(max(3, center - 2), min(h - 2, center + 3)))
        cut[:row + 1, x0] = True
        cut[row, x0:] = True
        if rng.random() < .5:
            cut = cut[::-1].copy()
    elif style == 'steps':
        rows = sorted(map(int, rng.choice(np.arange(3, h - 3), 2, replace=False)))
        xs = sorted(map(int, rng.choice(np.arange(x0, last + 1), 3, replace=False)))
        previous = 0
        for i, row in enumerate(rows):
            cut[previous:row + 1, xs[i]] = True
            cut[row, xs[i]:xs[i + 1] + 1] = True
            previous = row
        cut[previous:, xs[-1]] = True
    else:
        legs = 5 if style == 'fold5' else 3
        xs = np.rint(np.linspace(x0, last, legs)).astype(int)
        _require(np.all(np.diff(xs) >= 2), "fold needs separated legs")
        low = max(2, center - int(rng.integers(3, 6)))
        high = min(h - 3, center + int(rng.integers(3, 6)))
        previous = 0
        for i in range(legs - 1):
            row = high if i % 2 == 0 else low
            cut[min(previous, row):max(previous, row) + 1, xs[i]] = True
            cut[row, xs[i]:xs[i + 1] + 1] = True
            previous = row
        cut[previous:, xs[-1]] = True
    _check_path(cut)
    return cut


def _pick(rng, cells):
    return tuple(map(int, cells[int(rng.integers(len(cells)))]))


def _place_groups(rng, a, bg, ca, cb, free, mouth, count, rich):
    offsets = [((0, 0),), ((0, 0), (1, 0)), ((0, 0), (0, 1)),
               ((0, 0), (1, 0), (1, 1)), ((0, 0), (0, 1), (1, 0), (1, 1)),
               ((0, 0), (0, 3))]
    h, w = a.shape
    for _ in range(count):
        shape = offsets[int(rng.integers(len(offsets) if rich else 3))]
        candidates = []
        for y, x in np.argwhere(free & (a == bg)):
            if x <= mouth:
                continue
            points = [(int(y) + dy, int(x) + dx) for dy, dx in shape]
            if all(0 <= yy < h and 0 <= xx < w and free[yy, xx] and a[yy, xx] == bg
                   for yy, xx in points):
                candidates.append(points)
        if candidates:
            points = candidates[int(rng.integers(len(candidates)))]
            color = ca if rng.random() < .5 else cb
            for p in points:
                a[p] = color


def _make_pair(rng, palette, rotation, style, side, teaching=False, clean=False,
               long=False, rich=False):
    bg, wall_color, control, ca, cb = map(int, palette[:5])
    for _ in range(100):
        wall, footprint, chamber, rear, mouth, center = _sample_geometry(rng, long)
        try:
            cut = _cut(rng, wall.shape, mouth, center, style)
        except AssertionError:
            continue
        regions = _regions(cut)
        a = np.full(wall.shape, bg, np.int8)
        a[wall] = wall_color
        # Payload patches vary in area and axial extent, while keeping their order.
        for color, row in ((ca, center - 1), (cb, center + 1)):
            a[row, rear + 1] = color
            if rng.random() < .5:
                a[row, rear + 2] = color
            if rng.random() < .25 and chamber[row + (-1 if color == ca else 1), rear + 1]:
                a[row + (-1 if color == ca else 1), rear + 1] = color
        a[cut] = control
        sites = np.argwhere(regions[side] & ~_near(cut) & (a == bg) & ~chamber)
        if not len(sites):
            continue
        marker = _pick(rng, sites)
        a[marker] = control
        if teaching:
            # Three lanes: A-only opacity, B-only opacity, both streams opaque.
            first_cut = np.argmax(cut, axis=1)
            counts = cut.sum(axis=1)
            rows = [y for y in range(a.shape[0]) if counts[y] == 3
                    and np.count_nonzero(footprint[y, mouth + 1:first_cut[y]] &
                                         (a[y, mouth + 1:first_cut[y]] == bg)) >= 2]
            if len(rows) < 3:
                continue
            rows = rng.choice(rows, 3, replace=False)
            for i, y in enumerate(rows):
                choices = np.flatnonzero(footprint[y] & (a[y] == bg)
                                         & (np.arange(a.shape[1]) > mouth)
                                         & (np.arange(a.shape[1]) < first_cut[y]))
                xs = rng.choice(choices, 2, replace=False)
                if i in (0, 2):
                    a[y, xs[0]] = ca
                if i in (1, 2):
                    a[y, xs[1]] = cb
        elif not clean:
            free = ~wall & ~chamber & ~cut
            free[marker] = False
            _place_groups(rng, a, bg, ca, cb, free, mouth,
                          int(rng.integers(0, 6)), rich)
        out = _render_masks(a, bg, ca, cb, footprint, chamber, cut, marker, regions[side])
        pair = {'input': np.rot90(a, rotation).copy(), 'output': np.rot90(out, rotation).copy()}
        try:
            d = _decode(pair['input'])
            _require(np.array_equal(solve(pair['input']), pair['output']), "constructor disagreement")
            f = _evidence(d)
            needed = {'uses_a', 'uses_b', 'priority', 'marker_matters'} if clean else set()
            if teaching:
                needed = {'shadow_a', 'shadow_b', 'fallback_b', 'seam_dark', 'reentry', 'marker_matters'}
            if not needed <= f:
                continue
            return pair
        except AssertionError:
            continue
    raise AssertionError('bounded scene construction exhausted')


def _evidence(d):
    va, vb = _availability(d)
    a, bg = d['a'], d['bg']
    marked = d['regions'][d['side']]
    blank = (a == bg) & d['footprint']
    seam = d['cut'] & d['footprint']
    flags = set()
    tests = dict(uses_a=np.any(blank & marked & va), uses_b=np.any(blank & ~marked & vb),
                 shadow_a=np.any(blank & marked & ~va & vb),
                 shadow_b=np.any(blank & ~marked & ~vb & va),
                 priority=np.any(seam & va & vb), fallback_b=np.any(seam & ~va & vb),
                 seam_dark=np.any(seam & ~va & ~vb),
                 marker_matters=np.any(blank & (va | vb)))
    flags |= {name for name, present in tests.items() if present}
    for y in range(a.shape[0]):
        states = []
        for x in range(d['mouth'] + 1, a.shape[1]):
            if not blank[y, x] or d['cut'][y, x]:
                continue
            selected = 0 if marked[y, x] else 1
            states.append((selected, bool((va, vb)[selected][y, x])))
        # An inactive stream is still alive; crossing the stencil can reveal it.
        if any(not light and other_light and color != other_color
               for (color, light), (other_color, other_light) in zip(states, states[1:])):
            flags.add('reentry')
    return flags


def _opposite_marker(pair, rng):
    d = _decode(pair['input'])
    a = d['a'].copy()
    a[d['marker']] = d['bg']
    sites = np.argwhere(d['regions'][1 - d['side']] & ~_near(d['cut'])
                        & (a == d['bg']) & ~d['chamber'])
    p = _pick(rng, sites)
    a[p] = d['control']
    inp = np.rot90(a, d['rotation']).copy()
    return {'input': inp, 'output': solve(inp)}


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if not lo <= int(value) <= hi:
        raise ValueError(name + ' is out of range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Produce demonstrations that teach both inherited mechanisms jointly."""
    nt = _count(num_train, 'num_train', 2, 10)
    nq = _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a non-negative integer')
    rng = np.random.default_rng(seed)
    palette = rng.permutation(10)[:5]
    shared_palette = rng.random() < .25
    rotation = int(rng.integers(4))
    shift = int(rng.integers(3))
    first_side = int(rng.integers(2))
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            for _ in range(30):
                if split == 'train' and index == 2:
                    pair = _opposite_marker(puzzle['train'][0], rng)
                else:
                    colors = palette if shared_palette else rng.permutation(10)[:5]
                    teaching = split == 'train' and index == 1
                    clean = split == 'train' and index == 0
                    if clean:
                        style = str(rng.choice(['vertical', 'diagonal', 'steps']))
                    elif teaching:
                        style = 'fold3'
                    elif split == 'test' and shift == 0:
                        style = 'fold5'
                    else:
                        style = str(rng.choice(['vertical', 'diagonal', 'steps', 'elbow', 'fold3']))
                    if shift == 1:
                        turn = ((rotation + 1) % 4 if split == 'test' else
                                (rotation + 2 * int(rng.integers(2))) % 4)
                    else:
                        turn = rotation if rng.random() < .4 else int(rng.integers(4))
                    side = first_side if clean else 1 - first_side if teaching else int(rng.integers(2))
                    pair = _make_pair(rng, colors, turn, style, side, teaching, clean,
                                      long=split == 'test',
                                      rich=(shift != 2 or split == 'test'))
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError('distinct input construction exhausted')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'pair counts')
    seen, coverage, sides = set(), set(), set()
    clean_taught = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'shape preservation')
            key = _key(inp)
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            d = _decode(inp)
            expected = solve(inp)
            _require(np.array_equal(out, expected), 'incorrect illumination or preservation')
            _require(np.any(inp != out), 'identity pair')
            canonical_out = np.rot90(out, -d['rotation'])
            preserved = (d['a'] != d['bg']) & ~d['cut']
            _require(np.array_equal(canonical_out[preserved], d['a'][preserved]), 'material preservation')
            _require(np.array_equal(canonical_out[~d['footprint']], d['a'][~d['footprint']]),
                     'exterior preservation')
            _require(np.all(np.isin(canonical_out[d['cut'] & d['footprint']],
                                   [d['bg'], d['ca'], d['cb']])), 'erase illuminated stencil')
            if split == 'train':
                coverage |= _evidence(d)
                sides.add(d['side'])
                clean_taught |= not np.any(d['screens'] & d['footprint'])
    required = {'uses_a', 'uses_b', 'priority', 'shadow_a', 'shadow_b',
                'fallback_b', 'seam_dark', 'reentry', 'marker_matters'}
    _require(required <= coverage, 'training evidence missing: ' + str(required - coverage))
    _require(clean_taught and sides == {0, 1}, 'clean projection and both marker sides required')


def _signature(puzzle):
    return tuple((_key(pair['input']), _key(pair['output']))
                 for split in ('train', 'test') for pair in puzzle[split])


def _feature(d):
    cut = d['cut']
    crossing_runs = np.count_nonzero(cut & ~np.pad(cut[:, :-1], ((0, 0), (1, 0))), axis=1)
    grouped = any(len(g) > 1 for c in (d['ca'], d['cb'])
                  for g in _components(d['screens'] & (d['a'] == c)))
    return dict(shape=d['a'].shape, rotation=d['rotation'], side=d['side'],
                periods=tuple(f[0] for f in d['fits']),
                palette=(d['bg'], d['wall'], d['control'], d['ca'], d['cb']),
                cut=(cut.shape, cut.tobytes()),
                outline=(tuple(d['upper'] - d['upper'][0]), tuple(d['lower'] - d['lower'][0])),
                max_crossings=int(crossing_runs.max()),
                projection=d['a'].shape[1] - d['mouth'] - 1,
                screens=int(d['screens'].sum()), grouped=grouped,
                inert=int((d['screens'] & ~d['footprint']).sum()),
                patch_area=int(np.count_nonzero(d['chamber'] & np.isin(d['a'], [d['ca'], d['cb']]))))


def self_test():
    """Bounded 42-seed coverage, semantic counterfactuals, and count checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    features = []
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                d = _decode(pair['input'])
                features.append(_feature(d))
        # Recoloring all roles and rotating the entire scene commute with solving.
        pair = puzzle['train'][0]
        for k in (1, 2, 3):
            _require(np.array_equal(solve(np.rot90(pair['input'], k)), np.rot90(pair['output'], k)),
                     'rotation equivariance')
        perm = np.random.default_rng(19).permutation(10)
        _require(np.array_equal(solve(perm[pair['input']]), perm[pair['output']]), 'palette equivariance')
        # Omit A: choose the top source throughout. Omit B: disregard all shadows.
        a_changed = b_changed = False
        for pair in puzzle['train']:
            d = _decode(pair['input'])
            va, vb = _availability(d)
            blank = (d['a'] == d['bg']) & d['footprint']
            marked = d['regions'][d['side']]
            a_changed |= np.any(blank & ~marked & vb)
            b_changed |= np.any(blank & ((marked & ~va) | (~marked & ~vb)))
            swapped = _opposite_marker(pair, np.random.default_rng(33))
            sd = _decode(swapped['input'])
            answer = np.rot90(pair['output'], -d['rotation'])
            swapped_answer = np.rot90(swapped['output'], -sd['rotation'])
            _require(np.array_equal(answer[d['cut']], swapped_answer[d['cut']]), 'marker changed seam priority')
        _require(a_changed and b_changed, 'both parents must affect answers')
    _require(len({_signature(p) for p in puzzles}) == 42, 'distinct puzzles')
    _require(len({f['shape'] for f in features}) >= 45, 'canvas variation')
    _require(len({f['cut'] for f in features}) >= 100, 'stencil variation')
    _require(len({f['outline'] for f in features}) >= 30, 'emitter variation')
    _require({f['rotation'] for f in features} == set(range(4)), 'all directions')
    _require({f['periods'] for f in features} == {(1, 1), (1, 2), (2, 1), (2, 2)}, 'rail cadences')
    _require(len({f['patch_area'] for f in features}) >= 4, 'payload variation')
    _require(sum(f['grouped'] for f in features) >= 5, 'grouped screens')
    _require(sum(f['inert'] > 0 for f in features) >= 5, 'inert screens')
    _require(max(f['max_crossings'] for f in features) >= 5, 'folded stencils')
    _require(len({f['palette'] for f in features}) >= 60, 'role palette variation')
    for seed in (0, 13, 41):
        _require(_signature(puzzles[seed]) == _signature(generate_puzzle(seed)), 'determinism')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(510 + nt * 10 + nq, nt, nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, 'explicit counts')
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    damaged = copy.deepcopy(puzzles[0])
    p = damaged['test'][0]
    changed = np.argwhere(p['input'] != p['output'])[0]
    point = tuple(changed)
    p['output'][point] = p['input'][point]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('incorrect label accepted')
    # Return descriptive data for reproducible author measurements; no files written.
    return dict(seeds=42, pairs=len(features), shapes=len({f['shape'] for f in features}),
                stencils=len({f['cut'] for f in features}),
                emitters=len({f['outline'] for f in features}),
                palettes=len({f['palette'] for f in features}),
                directions=sorted({f['rotation'] for f in features}),
                grouped=sum(f['grouped'] for f in features),
                inert=sum(f['inert'] > 0 for f in features),
                projections=sorted({f['projection'] for f in features}))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
