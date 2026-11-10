"""A tip-led courier spends color-specific component inventories along a route.

Only NumPy and the standard library are used. The decoder accepts the visible
geometry, not a mask catalogue, seed, training set, or construction metadata.
"""
from collections import Counter

import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_DIAMOND = ((0, 1), (1, 0), (1, 2), (2, 1))
_SQUARE = ((0, 0), (0, 1), (1, 0), (1, 1))
_X = ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2))
_CROSS = ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape),
             "grid must be a nonempty rectangle with axes at most 30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "grid cells must be integer colors 0..9")
    return a


def _components(cells, diagonal=True):
    remaining = set(cells)
    result = []
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, found = [p], {p}
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    found.add(q)
        result.append(found)
    return result


def _normalize(cells):
    cells = tuple(cells)
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _rotate(p, turns):
    r, c = p
    for _ in range(turns % 4):
        r, c = c, -r
    return r, c


def _footprint(mask, tip, direction):
    return {(tip[0] + r, tip[1] + c)
            for r, c in (_rotate(p, direction - 1) for p in mask)}


def _arrow_geometry(cells):
    # A reflection axis and a single forward extreme identify the anchor.
    candidates = []
    for tip in sorted(cells):
        for direction in range(4):
            offsets = {_rotate((r - tip[0], c - tip[1]), 1 - direction)
                       for r, c in cells}
            if (all(c <= 0 for r, c in offsets)
                    and {p for p in offsets if p[1] == 0} == {(0, 0)}
                    and any(r != 0 for r, c in offsets)
                    and {(-r, c) for r, c in offsets} == offsets):
                candidates.append((tuple(sorted(offsets)), tip, direction))
    _require(len(candidates) == 1, "arrow must have one geometrically identifiable tip")
    return candidates[0]


def _direction(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    _require((dr == 0) != (dc == 0), "route references must be distinct and aligned")
    return (1 if dc > 0 else 3) if dr == 0 else (2 if dr > 0 else 0)


def _ordered_route(start, stations):
    remaining = set(stations)
    points, directions = [start], []
    while remaining:
        rays = {}
        for q in sorted(remaining):
            if q[0] == points[-1][0] or q[1] == points[-1][1]:
                d = _direction(points[-1], q)
                rays.setdefault(d, []).append(q)
        _require(len(rays) == 1, "each tip must have exactly one occupied remaining ray")
        d, choices = next(iter(rays.items()))
        p = points[-1]
        q = min(choices, key=lambda q: abs(q[0] - p[0]) + abs(q[1] - p[1]))
        points.append(q)
        directions.append(d)
        remaining.remove(q)
    return points, directions


def _panel_candidate(grid, field):
    rr, cc = np.where(grid == field)
    r0, r1, c0, c1 = int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1
    panel = grid[r0:r1, c0:c1]
    if min(panel.shape) < 5 or not all(np.all(edge == field) for edge in
                                     (panel[0], panel[-1], panel[:, 0], panel[:, -1])):
        return None
    outside = np.ones(grid.shape, bool)
    outside[r0:r1, c0:c1] = False
    if not outside.any():
        return None
    bg = int(np.argmax(np.bincount(grid[outside].astype(int), minlength=10)))
    if bg == field or np.mean(grid[outside] == bg) <= .6:
        return None
    ring = np.zeros(grid.shape, bool)
    ring[max(0, r0 - 1):r1 + 1, max(0, c0 - 1):c1 + 1] = True
    ring[r0:r1, c0:c1] = False
    if not np.all(grid[ring] == bg) or np.any(panel == bg):
        return None
    grouped = []
    for color in sorted(set(map(int, np.unique(panel))) - {int(field)}):
        grouped.extend((color, comp) for comp in
                       _components(set(map(tuple, np.argwhere(panel == color)))))
    large = [(color, comp) for color, comp in grouped if len(comp) > 1]
    singles = [(color, next(iter(comp))) for color, comp in grouped if len(comp) == 1]
    if len(large) != 1 or len(singles) < 2:
        return None
    ink, mover = large[0]
    if any(color == ink for color, p in singles) or np.any(grid[outside] == ink):
        return None
    try:
        mask, tip, initial = _arrow_geometry(mover)
        stations = {p: color for color, p in singles}
        points, ds = _ordered_route(tip, stations)
        _require(ds[0] == initial, "first leg must follow the arrow")
        # All potential arrivals, even after a fuel stop, must remain readable.
        for i, p in enumerate(points):
            cells = _footprint(mask, p, initial if i == 0 else ds[i - 1])
            _require(all(0 < r < panel.shape[0] - 1 and 0 < c < panel.shape[1] - 1
                         for r, c in cells), "arrow must fit inside the panel's blank border")
            _require(all(max(abs(q[0] - r), abs(q[1] - c)) > 1
                         for q in points[i + 1:] for r, c in cells),
                     "future stations must be separate from the arrow")
        _require(all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1
                     for i, a in enumerate(points[1:]) for b in points[i + 2:]),
                 "stations must be separate, including at corners")
    except ValueError:
        return None
    return dict(field=int(field), background=bg, bounds=(r0, r1, c0, c1),
                panel_shape=panel.shape, outside=outside, mask=mask, ink=ink,
                points=points, directions=ds, initial=initial,
                colors=[stations[p] for p in points[1:]])


def _read_scene(input_grid):
    grid = _grid(input_grid)
    candidates = [m for color in np.unique(grid)
                  if (m := _panel_candidate(grid, int(color))) is not None]
    _require(len(candidates) == 1, "route panel must be unique")
    m = candidates[0]
    outside, bg = m['outside'], m['background']
    objects = {}
    owner = np.full(grid.shape, -1, int)
    serial = 0
    for color in sorted(set(map(int, grid[outside])) - {bg}):
        components = _components(set(map(tuple, np.argwhere((grid == color) & outside))))
        objects[color] = components
        for component in components:
            for r, c in component:
                owner[r, c] = serial
            serial += 1
    for r, c in np.argwhere(owner >= 0):
        halo = owner[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
        _require(np.all((halo < 0) | (halo == owner[r, c])),
                 "separate exterior objects may not touch, even across colors")
    keys = set(m['colors'])
    _require(keys <= set(objects), "every station color needs a positive exterior inventory")
    m['objects'] = objects
    m['counts'] = {c: len(objects[c]) for c in keys}
    m['distractors'] = set(objects) - keys
    return m


def _prefix(colors, inventories):
    fuel = dict(inventories)
    accepted = 0
    for color in colors:
        if fuel[color] == 0:
            break
        fuel[color] -= 1
        accepted += 1
    return accepted, fuel


def _terminal(m, accepted, rotate=True):
    direction = m['initial'] if accepted == 0 or not rotate else m['directions'][accepted - 1]
    out = np.full(m['panel_shape'], m['field'], dtype=np.int8)
    for r, c in _footprint(m['mask'], m['points'][accepted], direction):
        out[r, c] = m['ink']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Count visible exterior objects and move the input arrow to its fuel stop."""
    m = _read_scene(input_grid)
    accepted, _ = _prefix(m['colors'], m['counts'])
    return _terminal(m, accepted)


def _make_arrow(span, shaft, solid, tails):
    cells = {(0, -c) for c in range(shaft)}
    for depth in range(1, span + 1):
        rows = range(-depth, depth + 1) if solid else (-depth, depth)
        cells.update((r, 1 - shaft - depth) for r in rows)
    for t in range(tails):
        cells.update((r, -shaft - span - t) for r in (-span, span))
    return tuple(sorted(cells))


_SMALL_ARROWS = tuple(_make_arrow(1, shaft, solid, tail)
                      for shaft in (1, 2) for solid in (False, True) for tail in (0, 1))
_LARGE_ARROWS = tuple(_make_arrow(2, shaft, solid, tail)
                      for shaft in (1, 2) for solid in (False, True) for tail in (0, 1))


def _fits(cells, shape):
    return all(0 < r < shape[0] - 1 and 0 < c < shape[1] - 1 for r, c in cells)


def _appendable(points, footprints, mask, shape, q, d):
    cells = _footprint(mask, q, d)
    if not _fits(cells, shape):
        return None
    past = footprints or [_footprint(mask, points[0], d)]
    if not _fits(past[0], shape):
        return None
    if any(max(abs(q[0] - r), abs(q[1] - c)) <= 1 for f in past for r, c in f):
        return None
    for p, nxt in zip(points[:-1], points[1:]):
        if p[0] == q[0] or p[1] == q[1]:
            if _direction(p, q) != _direction(p, nxt):
                return None
            if abs(q[0] - p[0]) + abs(q[1] - p[1]) <= abs(nxt[0] - p[0]) + abs(nxt[1] - p[1]):
                return None
    return cells


def _random_route(rng, mask, n, shape, rotate_at=None):
    # Coordinate proposals, not a finite route bank or a regular lattice.
    for _ in range(24):
        points = [(int(rng.integers(2, shape[0] - 2)), int(rng.integers(2, shape[1] - 2)))]
        footprints, ds = [], []
        for step in range(n):
            options = {}
            r, c = points[-1]
            for d, (dr, dc) in enumerate(_DIRS):
                for length in range(2, max(shape)):
                    q = (r + dr * length, c + dc * length)
                    if not (0 < q[0] < shape[0] - 1 and 0 < q[1] < shape[1] - 1):
                        continue
                    cells = _appendable(points, footprints, mask, shape, q, d)
                    if cells is not None:
                        options.setdefault(d, []).append((q, cells))
            if not options:
                break
            choices = sorted(options)
            weights = np.array([1.1 if ds and d == ds[-1] else 1.0 for d in choices])
            d = int(rng.choice(choices, p=weights / weights.sum()))
            q, cells = options[d][int(rng.integers(len(options[d])))]
            if not footprints:
                footprints.append(_footprint(mask, points[0], d))
            points.append(q)
            footprints.append(cells)
            ds.append(d)
        if len(ds) == n and (rotate_at is None or ds[rotate_at - 1] != ds[0]):
            return points, ds
    return None


def _route(rng, mask, n, short=False, rotate_at=None):
    low = 9 if short else 16 if n >= 7 else 12
    high = 16 if short else 23
    for _ in range(3):
        shape = (int(rng.integers(low, high + 1)), int(rng.integers(low, high + 1)))
        found = _random_route(rng, mask, n, shape, rotate_at)
        if found is not None:
            return shape, *found
    # Guaranteed bounded fallback, preserving requested length and arrow mask.
    # Three collinear stations precede a bend; <=11 stations fit in 23x23.
    shape = (23, 23)
    span, depth = max(abs(r) for r, c in mask), -min(c for r, c in mask)
    p = (span + 1, depth + 1)
    points = [p] + [(p[0], p[1] + 2 * i) for i in range(1, min(n, 3) + 1)]
    for i in range(n - 3):
        points.append((p[0] + span + 2 + 2 * i, p[1] + 6))
    ds = [_direction(a, b) for a, b in zip(points, points[1:])]
    return shape, points, ds


def _glyph(rng):
    mode = int(rng.integers(7))
    if mode < 4:
        cells = (_DIAMOND, _SQUARE, _X, _CROSS)[mode]
    elif mode == 4:
        cells = tuple((0, i) for i in range(int(rng.integers(1, 4))))
    elif mode == 5:
        cells = tuple((i, i) for i in range(int(rng.integers(2, 4))))
    else:
        cells = {(1, 1)}
        target = int(rng.integers(2, 10))
        while len(cells) < target:
            frontier = sorted({(r + dr, c + dc) for r, c in cells
                               for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                               if 0 <= r + dr < 3 and 0 <= c + dc < 3} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    turns, reflection = int(rng.integers(4)), int(rng.choice((-1, 1)))
    return _normalize(_rotate((r, reflection * c), turns) for r, c in cells)


def _place(grid, blocked, color, cells, rng):
    h, w = 1 + max(r for r, c in cells), 1 + max(c for r, c in cells)
    nh, nw = grid.shape[0] - h + 1, grid.shape[1] - w + 1
    allowed = np.ones((nh, nw), bool)
    for r, c in cells:
        allowed &= ~blocked[r:r + nh, c:c + nw]
    positions = np.argwhere(allowed)
    if len(positions) == 0:
        return False
    r0, c0 = positions[int(rng.integers(len(positions)))]
    for dr, dc in cells:
        r, c = int(r0 + dr), int(c0 + dc)
        grid[r, c] = color
        blocked[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
    return True


def _scene(rng, mask, station_colors, counts, palette, distractors,
           short=False, witness=False, rotate_at=None):
    shape, points, ds = _route(rng, mask, len(station_colors), short, rotate_at)
    bg, field, ink = palette
    panel = np.full(shape, field, np.int8)
    for p, color in zip(points[1:], station_colors):
        panel[p] = color
    for p in _footprint(mask, points[0], ds[0]):
        panel[p] = ink
    accepted, _ = _prefix(station_colors, counts)
    output = np.full(shape, field, np.int8)
    for p in _footprint(mask, points[accepted], ds[accepted - 1]):
        output[p] = ink
    objects = []
    for color, count in list(counts.items()) + list(distractors.items()):
        prototype = _glyph(rng)
        for i in range(count):
            glyph = prototype if rng.random() < .3 else _glyph(rng)
            if witness and color == station_colors[0] and i < 3:
                glyph = (_DIAMOND, _SQUARE, ((0, 0),))[i]
            objects.append((color, glyph))
    _require(len(objects) <= 13 and max(shape) <= 23, "internal packing budget exceeded")
    rng.shuffle(objects)
    objects.sort(key=lambda x: -len(x[1]))
    for attempt in range(10):
        fallback = attempt == 9
        if fallback:
            h, w, r0, c0 = 30, 30, 0, 0
        else:
            h = int(rng.integers(max(shape[0] + 2, 21), 31))
            w = int(rng.integers(max(shape[1] + 2, 21), 31))
            if attempt >= 5:
                h = w = 30
            r0, c0 = int(rng.integers(h - shape[0] + 1)), int(rng.integers(w - shape[1] + 1))
        grid = np.full((h, w), bg, np.int8)
        grid[r0:r0 + shape[0], c0:c0 + shape[1]] = panel
        blocked = np.zeros(grid.shape, bool)
        blocked[max(0, r0 - 1):r0 + shape[0] + 1, max(0, c0 - 1):c0 + shape[1] + 1] = True
        if fallback:
            slots = [(r, c) for r in range(0, 28, 4) for c in range(0, 28, 4)
                     if not blocked[r:r + 3, c:c + 3].any()]
            _require(len(slots) >= len(objects), "fallback slots insufficient")
            rng.shuffle(slots)
            for (color, glyph), (r, c) in zip(objects, slots):
                for dr, dc in glyph:
                    grid[r + dr, c + dc] = color
        elif not all(_place(grid, blocked, color, glyph, rng) for color, glyph in objects):
            continue
        # Geometric transforms move both arrays; palette roles are pair-local.
        if rng.random() < .5:
            grid, output = np.fliplr(grid), np.fliplr(output)
        turns = int(rng.integers(4))
        grid, output = np.rot90(grid, turns).copy(), np.rot90(output, turns).copy()
        return {'input': grid, 'output': output}
    raise RuntimeError("unreachable packing failure")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    regime = int(rng.integers(3))  # bigger budgets, extra category, new arrow span
    tracked_color = int(rng.integers(10))
    first_mask = _SMALL_ARROWS[int(rng.integers(len(_SMALL_ARROWS)))]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            witness = split == 'train' and index == 0
            role_witness = split == 'train' and index == 1
            completion = split == 'train' and index == 2
            for retry in range(12):
                palette = list(map(int, rng.permutation(10)))
                if witness:
                    palette.remove(tracked_color)
                    palette.insert(3, tracked_color)
                elif role_witness:
                    palette.remove(tracked_color)
                    palette.insert(8, tracked_color)
                bg, field, ink = palette[:3]
                k = 2 if witness or role_witness or completion else 3 if split == 'test' and regime == 1 else int(rng.integers(1, 3))
                keys = palette[3:3 + k]
                if witness:
                    n = int(rng.integers(7, 10))
                    x, y = keys
                    colors = [x, y, x, x, y, x] + list(map(int, rng.choice(keys, n - 6)))
                    counts = {x: 3, y: 3}
                    mask = first_mask
                elif role_witness:
                    n = int(rng.integers(4, 7))
                    x, y = keys
                    colors = [x, y, x] + list(map(int, rng.choice(keys, n - 3)))
                    counts = {x: 1, y: int(rng.integers(2, 4))}
                    mask_pool = [m for m in _SMALL_ARROWS if m != first_mask]
                    mask = mask_pool[int(rng.integers(len(mask_pool)))]
                elif completion:
                    n = int(rng.integers(3, 6))
                    colors = [keys[0], keys[1]] + list(map(int, rng.choice(keys, n - 2)))
                    rng.shuffle(colors)
                    counts = dict(Counter(colors))
                    # Preserve completion even when a budget exceeds the sampling cap.
                    # Balance the short itinerary to at most three visits per color.
                    if max(counts.values()) > 3:
                        continue
                    counts[keys[int(rng.integers(k))]] += int(rng.integers(2))
                    if regime == 0 and max(counts.values()) > 3:
                        continue
                    mask_pool = [m for m in _SMALL_ARROWS if m != first_mask]
                    mask = mask_pool[int(rng.integers(len(mask_pool)))]
                elif split == 'test' and regime == 0:
                    n = int(rng.integers(8, 12))
                    k, keys = 2, palette[3:5]
                    x, y = keys
                    if rng.random() < .2:
                        # A larger inventory can also complete the whole journey.
                        n = 8
                        counts = {x: 4, y: 4}
                        colors = [x] * 4 + [y] * 4
                        rng.shuffle(colors)
                    else:
                        counts = {x: 4, y: int(rng.integers(2, 4))}
                        ny = int(rng.integers(1, min(counts[y], n - 6) + 1))
                        prefix = [x] * 4 + [y] * ny
                        rng.shuffle(prefix)
                        colors = prefix + [x] + list(map(int, rng.choice(keys, n - len(prefix) - 1)))
                    mask = _SMALL_ARROWS[int(rng.integers(len(_SMALL_ARROWS)))]
                elif split == 'test' and regime == 1:
                    n = int(rng.integers(8, 12))
                    counts = {c: int(rng.integers(1, 4)) for c in keys}
                    bottleneck = keys[int(rng.integers(k))]
                    used = {c: counts[c] if c == bottleneck else int(rng.integers(1, counts[c] + 1))
                            for c in keys}
                    while sum(used.values()) > n - 2:
                        choices = [c for c in keys if c != bottleneck and used[c] > 1]
                        used[int(rng.choice(choices))] -= 1
                    prefix = [c for c in keys for _ in range(used[c])]
                    rng.shuffle(prefix)
                    colors = prefix + [bottleneck] + list(map(int, rng.choice(keys, n - len(prefix) - 1)))
                    mask = _SMALL_ARROWS[int(rng.integers(len(_SMALL_ARROWS)))]
                else:
                    n = int(rng.integers(4, 10) if split == 'test' else rng.integers(max(4, k), 10))
                    colors = keys.copy() + list(map(int, rng.choice(keys, n - k)))
                    rng.shuffle(colors)
                    cap = 4 if regime == 2 else 3
                    counts = {c: int(rng.integers(1, cap + 1)) for c in keys}
                    mask_pool = _LARGE_ARROWS if split == 'test' and regime == 2 else _SMALL_ARROWS
                    mask = mask_pool[int(rng.integers(len(mask_pool)))]
                free = [c for c in palette[3:] if c not in keys]
                nd = 1 if witness or role_witness else int(rng.integers(3))
                distractors = {c: 1 for c in free[:nd]}
                if witness:
                    distractors[free[0]] = 3
                elif role_witness:
                    distractors = {tracked_color: int(rng.integers(1, 4))}
                elif nd:
                    # At most three ignored objects total; comparable multiplicities.
                    distractors[free[0]] += int(rng.integers(4 - nd))
                pair = _scene(rng, mask, colors, counts, (bg, field, ink), distractors,
                              short=role_witness or completion, witness=witness, rotate_at=5 if witness else None)
                if role_witness and pair['output'].shape == puzzle['train'][0]['output'].shape:
                    continue
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("failed to produce a distinct example")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(grid):
    m = _read_scene(grid)
    accepted, remaining = _prefix(m['colors'], m['counts'])
    m.update(accepted=accepted, remaining=remaining, completed=accepted == len(m['colors']))
    m['diagonal'] = any(len(_components(comp, False)) > 1
                        for c in m['counts'] for comp in m['objects'][c])
    m['mixed_area'] = any(len({len(comp) for comp in m['objects'][c]}) > 1 for c in m['counts'])
    m['equal_area_connectivity'] = any(
        len(a) == len(b) and len(_components(a, False)) != len(_components(b, False))
        for c in m['counts'] for a in m['objects'][c] for b in m['objects'][c])
    return m


def validate_generated_puzzle(puzzle):
    """Check labels, visible constraints, teaching witnesses and input uniqueness."""
    try:
        _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "incorrect puzzle keys")
        _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
        nt, nq = len(puzzle['train']), len(puzzle['test'])
        _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal split counts")
        seen, training, testing = set(), [], []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "incorrect pair keys")
                inp, out = _grid(pair['input']), _grid(pair['output'])
                _require(_key(inp) not in seen, "duplicate input")
                seen.add(_key(inp))
                m = _features(inp)
                _require(np.array_equal(out, _terminal(m, m['accepted'])), "incorrect terminal arrow")
                _require(int(np.sum(out == m['ink'])) == len(m['mask']), "arrow area changed")
                _require(set(map(int, np.unique(out))) == {m['field'], m['ink']}, "output has extra content")
                if split == 'train':
                    training.append(m)
                else:
                    testing.append(m)
        if nt >= 3:
            _require(any(m['completed'] for m in training), "training must teach route completion")
        depleted = [m for m in training if not m['completed'] and any(m['remaining'].values())]
        _require(bool(depleted), "training must teach a bottleneck despite spare fuel")
        _require(any(m['diagonal'] and m['mixed_area'] and m['equal_area_connectivity'] for m in depleted),
                 "depletion must demonstrate component identity rather than area")
        _require(any(m['directions'][m['accepted'] - 1] != m['initial'] for m in depleted),
                 "training must demonstrate rotation at the affordable stop")
        _require(len({m['mask'] for m in training}) >= 2, "training must preserve different arrow masks")
        _require(len({m['panel_shape'] for m in training}) >= 2, "training must vary panel geometry")
        _require(len({m['accepted'] for m in training if not m['completed']}) >= 2,
                 "training must defeat a fixed stopping depth")
        _require(any(not m['completed'] and m['counts'][m['colors'][m['accepted']]] == 1 for m in training),
                 "training must demonstrate depletion of a smaller inventory")
        _require(any(set(a['counts']) & b['distractors'] for a in training for b in training),
                 "training must teach local station-color filtering")
        # Actual wrong-answer witnesses, not merely the presence of ingredients.
        _require(any(_prefix(m['colors'], {c: sum(len(_components(comp, False)) for comp in m['objects'][c])
                                           for c in m['counts']})[0] != m['accepted'] for m in depleted),
                 "four-connectivity shortcut must change a training answer")
        _require(any(min(sum(m['counts'].values()), len(m['colors'])) != m['accepted'] for m in depleted),
                 "pooled-inventory shortcut must change a training answer")
        max_categories = max(len(m['counts']) for m in training)
        max_budget = max(n for m in training for n in m['counts'].values())
        max_span = max(max(abs(r) for r, c in m['mask']) for m in training)
        for m in testing:
            visits = Counter(m['colors'][:m['accepted']])
            if len(m['counts']) > max_categories:
                _require(set(visits) == set(m['counts']), "new test categories must participate before the stop")
            # A geometry-transfer test may contain an unused larger inventory.
            # Without new geometry, the larger inventory is the selected shift.
            if max(m['counts'].values()) > max_budget and max(abs(r) for r, c in m['mask']) <= max_span:
                _require(any(n > max_budget for n in visits.values()),
                         "test budget extrapolation must affect actual visits")
    except Exception as exc:
        raise AssertionError(f"invalid generated puzzle: {exc}") from exc


def _puzzle_key(puzzle):
    return tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in puzzle[s])


def _route_signature(m):
    start = m['points'][0]
    return min(tuple(_rotate((r - start[0], reflect * (c - start[1])), turn)
                     for r, c in m['points']) for reflect in (-1, 1) for turn in range(4))


def self_test():
    """Bounded 42-seed sweep, family counterfactuals, counts and determinism."""
    records, signatures = [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(_puzzle_key(p))
        train_meta = []
        for split in ('train', 'test'):
            for pair in p[split]:
                m = _features(pair['input'])
                _require(np.array_equal(solve(pair['input'].tolist()), pair['output']), "list input solve failed")
                records.append(m)
                if split == 'train':
                    train_meta.append((pair, m))
        # Removing B (inexhaustible fuel) and removing A's rotation both change
        # the designated depleted training witness's answer.
        pair, witness = next((pair, m) for pair, m in train_meta
                             if m['accepted'] == 5 and m['diagonal'] and m['equal_area_connectivity']
                             and len(m['colors']) >= 7 and m['colors'][0] != m['colors'][1]
                             and m['colors'][:6] == [m['colors'][0], m['colors'][1], m['colors'][0],
                                                     m['colors'][0], m['colors'][1], m['colors'][0]]
                             and m['directions'][4] != m['initial'])
        a = witness['accepted']
        _require(not np.array_equal(_terminal(witness, a), _terminal(witness, len(witness['colors']))),
                 "unlimited fuel failed to change the answer")
        actual_cells = _footprint(witness['mask'], witness['points'][a], witness['directions'][a - 1])
        unturned_cells = _footprint(witness['mask'], witness['points'][a], witness['initial'])
        _require(actual_cells != unturned_cells, "ignoring rotation failed to change the answer")
        area_counts = {c: sum(len(comp) for comp in witness['objects'][c]) for c in witness['counts']}
        _require(_prefix(witness['colors'], area_counts)[0] != a, "area counting shortcut survived")
        # Spend only within contiguous color runs: this wrong rule refreshes fuel.
        previous, run_used, reset_accepted = None, 0, 0
        for color in witness['colors']:
            run_used = run_used + 1 if color == previous else 1
            previous = color
            if run_used > witness['counts'][color]:
                break
            reset_accepted += 1
        _require(reset_accepted != a, "refresh-on-color-change shortcut survived")
        _require(any(_prefix(m['colors'], {c: 3 for c in m['counts']})[0] != m['accepted']
                     for _, m in train_meta), "fixed capacity of three survived")
        # Same positions, same exterior inventories, same color multiset.
        swapped = pair['input'].copy()
        r0, _, c0, _ = witness['bounds']
        left = witness['points'][2]
        right = witness['points'][6]
        p0, p1 = (r0 + left[0], c0 + left[1]), (r0 + right[0], c0 + right[1])
        swapped[p0], swapped[p1] = int(swapped[p1]), int(swapped[p0])
        changed = _features(swapped)
        _require(changed['counts'] == witness['counts'] and changed['points'] == witness['points']
                 and changed['accepted'] == 3, "order-sensitive inventory interaction failed")
        _require(not np.array_equal(solve(swapped), pair['output']), "color-order swap did not change the answer")
    _require(len(signatures) == 42, "seed diversity failed")
    stats = dict(seeds=42, pairs=len(records),
                 panel_shapes=len({m['panel_shape'] for m in records}),
                 routes=len({_route_signature(m) for m in records}),
                 arrow_geometries=len({m['mask'] for m in records}),
                 glyph_geometries=len({_normalize(comp) for m in records for comps in m['objects'].values() for comp in comps}),
                 stopped=sum(not m['completed'] for m in records), completed=sum(m['completed'] for m in records),
                 lengths=sorted({len(m['colors']) for m in records}),
                 budgets=sorted({n for m in records for n in m['counts'].values()}),
                 category_counts=sorted({len(m['counts']) for m in records}),
                 stopping_depths=sorted({m['accepted'] for m in records if not m['completed']}),
                 initial_directions=sorted({m['initial'] for m in records}),
                 final_directions=sorted({m['directions'][m['accepted'] - 1] for m in records}),
                 mixed_turn_routes=sum({(b - a) % 4 for a, b in zip(m['directions'], m['directions'][1:])} >= {1, 3}
                                      for m in records))
    _require(stats['panel_shapes'] >= 40 and stats['routes'] >= 80 and stats['arrow_geometries'] >= 12,
             "insufficient structural variation")
    _require(stats['glyph_geometries'] >= 40 and stats['mixed_turn_routes'] >= 12, "insufficient shape/turn variation")
    _require(stats['initial_directions'] == list(range(4)) and stats['final_directions'] == list(range(4)),
             "direction variation failed")
    _require(set(stats['budgets']) >= {1, 2, 3, 4} and len(stats['lengths']) >= 7, "count variation failed")
    _require({m['field'] for m in records} == set(range(10)) and
             {m['background'] for m in records} == set(range(10)), "palette role variation failed")
    # A compatible mask beyond the sampler catalogue, with a wider hollow head
    # and longer shaft/tails, must be read from its cells by the same solver.
    larger = _scene(np.random.default_rng(314159), _make_arrow(3, 3, False, 2),
                    [3, 4, 3, 5, 3, 6, 3, 3, 3], {3: 5, 4: 1, 5: 2, 6: 4}, (0, 1, 2), {})
    _require(np.array_equal(solve(larger['input']), larger['output']), "non-catalogue arrow failed")
    visible = _features(larger['input'])
    _require(len(visible['counts']) == 4 and visible['counts'][3] == 5 and visible['accepted'] == 8,
             "larger inventory/category decoder coverage failed")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(123, nt, nq)
            _require((len(p['train']), len(p['test'])) == (nt, nq), "explicit counts changed")
    _require(_puzzle_key(generate_puzzle(999, 5, 3)) == _puzzle_key(generate_puzzle(999, 5, 3)),
             "whole-puzzle determinism failed")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0)
    p['test'][0]['output'][0, 0] = (int(p['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect label accepted")
    try:
        _ordered_route((3, 3), {(3, 6), (6, 3)})
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous junction accepted")
    return stats
