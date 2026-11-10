"""Registered stamp clouds docking independently on corrugated, color-coded floors.

Only NumPy and the standard library are used. Generation keeps no solver state,
writes no files, and never imports either parent generator.
"""
from __future__ import annotations

import hashlib
import numpy as np

BACKGROUND = 0
STENCIL = 8
FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
COLORS = (1, 2, 3, 4, 5, 6, 7, 9)


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "grids must contain integer colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _components(grid, cells, diagonal=False):
    remaining = set(cells)
    result = []
    neighbors = EIGHT if diagonal else FOUR
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        color = int(grid[start])
        stack, part = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                p = r + dr, c + dc
                if p in remaining and grid[p] == color:
                    remaining.remove(p)
                    part.add(p)
                    stack.append(p)
        result.append((color, frozenset(part)))
    return result


def _normalized(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _effective(offsets):
    """Vertical registration cancels in docking; keep lateral registration."""
    top = min(r for r, c in offsets)
    return tuple(sorted((r - top, c) for r, c in offsets))


def _find_bands(a):
    """Read two rails and a columnwise route-prefix / solid-floor-suffix."""
    h, w = a.shape
    bands = []
    r = 1
    while r < h:
        nz = np.flatnonzero(a[r] != BACKGROUND)
        if len(nz) < 11 or np.any(a[r - 1] != BACKGROUND):
            r += 1
            continue
        left, right = int(nz[0]), int(nz[-1])
        rail, route = int(a[r, left]), int(a[r, left + 1])
        if (right - left < 10 or rail == route or rail in (0, 8) or route in (0, 8)
                or a[r, right] != rail or not np.all(a[r, left + 1:right] == route)):
            r += 1
            continue
        bottom = r
        while bottom + 1 < h and a[bottom + 1, left] == rail and a[bottom + 1, right] == rail:
            bottom += 1
        _require(bottom >= r + 2, "tray is too shallow")
        box = a[r:bottom + 1, left:right + 1]
        _require(np.all(box[-1] == rail), "tray needs a solid base")
        _require(not np.any(a[r:bottom + 1, :left]) and
                 not np.any(a[r:bottom + 1, right + 1:]), "content beside a tray")
        floors = {}
        for c in range(left + 1, right):
            column = a[r:bottom + 1, c]
            hits = np.flatnonzero(column == rail)
            _require(len(hits) > 0, "missing floor")
            first = int(hits[0])
            _require(first >= 1 and np.all(column[:first] == route) and
                     np.all(column[first:] == rail), "floor must be solid below its skyline")
            floors[c] = r + first
        bands.append(dict(top=r, bottom=bottom, left=left, right=right,
                          route=route, rail=rail, floor=floors))
        r = bottom + 1
    _require(1 <= len(bands) <= 3, "expected one to three trays")
    _require(len({b['route'] for b in bands}) == len(bands), "duplicate routing color")
    _require(all(b['top'] >= p['bottom'] + 2 for p, b in zip(bands, bands[1:])),
             "trays need blank separation")
    allowed = np.zeros(a.shape, dtype=bool)
    for b in bands:
        allowed[b['top']:b['bottom'] + 1, b['left']:b['right'] + 1] = True
    _require(not np.any((a != 0) & ~allowed & (np.indices(a.shape)[0] >= bands[0]['top'])),
             "all workpieces and instructions must be on the near side")
    return bands


def _label_options(a, mask, routes):
    """Parent B's source, target, next-cell registration, constrained by trays."""
    near = {(r + dr, c + dc) for r, c in mask
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
    options = []
    h, w = a.shape
    for qr, qc in sorted(near):
        if not (0 <= qr < h and 0 <= qc < w) or a[qr, qc] not in (0, 8):
            continue
        for dr, dc in FOUR:
            s, t = (qr - 2 * dr, qc - 2 * dc), (qr - dr, qc - dc)
            if not (0 <= s[0] < h and 0 <= s[1] < w and 0 <= t[0] < h and 0 <= t[1] < w):
                continue
            source, target = int(a[s]), int(a[t])
            if source in routes and target == routes[source]:
                options.append(dict(source=source, target=target, q=(qr, qc),
                                    labels=(s, t), mask=mask,
                                    offsets=tuple(sorted((r - qr, c - qc) for r, c in mask))))
    return options


def _parse_canonical(a):
    bands = _find_bands(a)
    stop = bands[0]['top']
    routes = {b['route']: b['rail'] for b in bands}
    neutral = {tuple(map(int, p)) for p in np.argwhere(a[:stop] == STENCIL)}
    masks = _components(a, neutral, True)
    _require(len(masks) == len(bands), "one stencil per routing color is required")
    records, labels = {}, set()
    for _, mask in masks:
        options = _label_options(a[:stop], mask, routes)
        _require(len(options) == 1, "stencil needs exactly one label/registration reading")
        rec = options[0]
        _require(rec['source'] not in records and not labels.intersection(rec['labels']),
                 "shared labels or multiple stencils for a route")
        records[rec['source']] = rec
        labels.update(rec['labels'])
    _require(set(records) == set(routes), "missing route instruction")
    staging = {tuple(map(int, p)) for p in np.argwhere(a[:stop] != 0)} - neutral - labels
    _require(staging and all(int(a[p]) in routes for p in staging), "unroutable workpiece cell")
    components = _components(a, staging)
    _require({color for color, cells in components} == set(routes), 'each tray needs a workpiece')
    _require(len(components) == len(_components(a, staging, True)),
             "diagonal-only same-color contacts are excluded")
    return dict(grid=a, bands=bands, records=records, components=components, stop=stop)


def _cloud(cells, offsets):
    return {(r + dr, c + dc) for r, c in cells for dr, dc in offsets}


def _dock(cloud, band, flat=False):
    _require(cloud and all(band['left'] < c < band['right'] for r, c in cloud),
             "footprint crosses a lateral rail")
    # The tightest column clearance controls the rigid body's first contact.
    floor = band['floor']
    level = band['bottom']  # flat alternative: remove all raised teeth
    shift = min((level if flat else floor[c]) - r - 1 for r, c in cloud)
    result = {(r + shift, c) for r, c in cloud}
    _require(all(band['top'] <= r < (level if flat else floor[c]) for r, c in result),
             "docked footprint does not fit in the tray")
    return result


def _render(analysis, variant=None):
    """Variants are explicit wrong hypotheses used only for teaching checks."""
    a = analysis['grid']
    result = a.copy()
    result[:analysis['stop']] = 0
    lookup = {b['route']: b for b in analysis['bands']}
    components = analysis['components']
    if variant == 'merge_color':
        components = [(color, frozenset().union(*(cells for col, cells in components if col == color)))
                      for color in sorted({col for col, cells in components})]
    paints = {}
    for color, cells in components:
        b, rec = lookup[color], analysis['records'][color]
        offsets = rec['offsets'] if variant != 'no_stencil' else ((0, 0),)
        if variant == 'center':
            r0 = (min(r for r, c in rec['mask']) + max(r for r, c in rec['mask'])) // 2
            c0 = (min(c for r, c in rec['mask']) + max(c for r, c in rec['mask'])) // 2
            offsets = tuple((r - r0, c - c0) for r, c in rec['mask'])
        if variant == 'solid_box':
            offsets = tuple((r, c) for r in range(min(r for r, c in offsets), max(r for r, c in offsets) + 1)
                            for c in range(min(c for r, c in offsets), max(c for r, c in offsets) + 1))
        groups = [{p} for p in sorted(cells)] if variant == 'per_pixel' else [cells]
        for group in groups:
            cloud = _cloud(group, offsets)
            if variant == 'no_docking':
                target = {(r, c) for r, c in cloud if 0 <= r < a.shape[0] and 0 <= c < a.shape[1]}
            else:
                target = _dock(cloud, b, flat=variant == 'flat_floor')
            for p in target:
                _require(p not in paints or paints[p] == b['rail'], "incompatible paint collision")
                paints[p] = b['rail']
    for p, color in paints.items():
        result[p] = color
    return result


def _analyze(input_grid):
    a = _grid(input_grid)
    candidates = []
    for turn in range(4):
        canonical = np.rot90(a, turn)
        try:
            analysis = _parse_canonical(canonical)
            analysis['output'] = _render(analysis)
            analysis['turn'] = turn
            candidates.append(analysis)
        except AssertionError:
            continue
    _require(len(candidates) == 1, "input must have one staging-to-tray orientation")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode every needed role, registration, component and floor from the input."""
    a = _analyze(input_grid)
    return np.rot90(a['output'], -a['turn']).copy()


def _differs(a, variant):
    try:
        return not np.array_equal(a['output'], _render(a, variant))
    except AssertionError:
        return False  # Teaching evidence must be an actual, in-bounds alternative output.


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _stencil(rng, max_height, hole=False):
    """Procedural geometry; returns a mask plus a visible label ray registration."""
    kind = 'c' if hole else _pick(rng, ('line', 'l', 'c', 'growth', 'fork', 'outline'))
    h = int(rng.integers(2, max_height + 1)) if max_height >= 2 else 1
    w = int(rng.integers(2, 6))
    if kind in ('c', 'outline') and max_height < 3:
        kind = 'l'
    if kind == 'line':
        cells = {(0, c) for c in range(int(rng.integers(2, 7)))}
    elif kind == 'c':
        h = int(rng.integers(3, max_height + 1))
        w = max(3, w)
        cells = {(r, w - 1) for r in range(h)} | {(r, c) for r in (0, h - 1) for c in range(w)}
    elif kind == 'outline':
        h, w = max(3, h), max(3, w)
        cells = {(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)}
    elif kind == 'l':
        cells = {(r, 0) for r in range(h)} | {(h - 1, c) for c in range(w)}
    elif kind == 'fork':
        cells = {(r, w // 2) for r in range(h)} | {(0, c) for c in range(w)}
    else:
        cells = {(0, 0)}
        goal = int(rng.integers(2, h * w + 1))
        while len(cells) < goal:
            frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in FOUR
                               if 0 <= r + dr < h and 0 <= c + dc < w} - cells)
            cells.add(_pick(rng, frontier))
    cells = set(_normalized(cells))
    if rng.random() < .5:
        width = max(c for r, c in cells)
        cells = {(r, width - c) for r, c in cells}
    if rng.random() < .5:
        height = max(r for r, c in cells)
        cells = {(height - r, c) for r, c in cells}
    near = {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
    choices = []
    for q in sorted(near):
        if hole and (q in cells or not (0 <= q[0] <= max(r for r, c in cells)
                                      and 0 <= q[1] <= max(c for r, c in cells))):
            continue
        if not hole and q not in cells:
            continue
        for d in FOUR:
            labels = ((q[0] - 2 * d[0], q[1] - 2 * d[1]), (q[0] - d[0], q[1] - d[1]))
            if any(p in cells for p in labels):
                continue
            # Make local reading unique even if source and target are reciprocal.
            points = cells | set(labels) | {q}
            r0, c0 = min(r for r, c in points), min(c for r, c in points)
            tile = np.zeros((max(r for r, c in points) - r0 + 1,
                             max(c for r, c in points) - c0 + 1), dtype=np.int8)
            mask = frozenset((r - r0, c - c0) for r, c in cells)
            for p in mask:
                tile[p] = STENCIL
            sl, tl = ((r - r0, c - c0) for r, c in labels)
            tile[sl], tile[tl] = 1, 2
            opts = _label_options(tile, mask, {1: 2, 2: 1})
            if len(opts) == 1:
                qr, qc = q[0] - r0, q[1] - c0
                offsets = tuple(sorted((r - qr, c - qc) for r, c in mask))
                # Require lateral evidence against automatic centering for the occupied lesson.
                offcenter = min(c for r, c in offsets) + max(c for r, c in offsets) != 0
                if hole or offcenter:
                    choices.append(dict(mask=mask, q=(qr, qc), labels=(sl, tl),
                                        offsets=offsets, shape=tile.shape))
    return _pick(rng, choices) if choices else None


def _piece(rng, height, width):
    h, w = int(rng.integers(1, height + 1)), int(rng.integers(1, width + 1))
    kind = int(rng.integers(4))
    if kind == 0:
        cells = {(r, c) for r in range(h) for c in range(w)}
    elif kind == 1:
        cells = {(r, 0) for r in range(h)} | {(h - 1, c) for c in range(w)}
    elif kind == 2:
        cells = {(r, w // 2) for r in range(h)} | {(h // 2, c) for c in range(w)}
    else:
        cells = {(0, 0)}
        goal = int(rng.integers(1, h * w + 1))
        while len(cells) < goal:
            frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in FOUR
                               if 0 <= r + dr < h and 0 <= c + dc < w} - cells)
            cells.add(_pick(rng, frontier))
    return _normalized(cells)


def _attempt(rng, lesson, count, rotation, novel):
    maxh = 3 if count > 1 else 4
    stencils = []
    for i in range(count):
        hole = (lesson == 'independent' and i == 0) or (lesson != 'split' and rng.random() < .35)
        spec = _stencil(rng, maxh, hole=hole)
        if spec is None:
            return None
        stencils.append(spec)
    if novel and all(_effective(s['offsets']) in novel for s in stencils):
        return None
    widths = [s['shape'][1] for s in stencils]
    gap = int(rng.integers(2, 4))
    low_width = max(16, sum(widths) + gap * (count - 1) + 2)
    if low_width > 30:
        return None
    width = int(rng.integers(low_width, 31))
    header = max(s['shape'][0] for s in stencils) + int(rng.integers(0, 2))
    piece_h = 1 if count == 3 else 2 if count == 2 else 3
    stage_h = int(rng.integers(4, 7)) if count < 3 else int(rng.integers(3, 5))
    stage_start = header + int(rng.integers(1, 3))
    stop = stage_start + stage_h + 1
    floor_cap = 2 if count == 3 else int(rng.integers(2, 4))
    required = [piece_h + max(r for r, c in s['offsets']) - min(r for r, c in s['offsets']) + floor_cap
                for s in stencils]
    separations = [int(rng.integers(1, 3)) for _ in range(count - 1)]
    budget = 30 - stop - sum(required) - sum(separations)
    if budget < 0:
        return None
    extras = [0] * count
    for _ in range(int(rng.integers(budget + 1))):
        extras[int(rng.integers(count))] += 1
    depths = [d + e for d, e in zip(required, extras)]
    height = stop + sum(depths) + sum(separations)
    # Spare rows appear as an optional far margin, not another instruction channel.
    margin = int(rng.integers(min(2, 30 - height) + 1))
    a = np.zeros((height + margin, width), dtype=np.int8)
    palette = list(map(int, rng.permutation(COLORS)))
    routes = palette[:count]
    rails = []
    for source in routes:
        choices = [c for c in COLORS if c != source]
        if count > 1 and rng.random() < .45:
            choices = [c for c in routes if c != source]
        rails.append(_pick(rng, choices))
    cursor = stop
    bands = []
    for i, (source, target, depth) in enumerate(zip(routes, rails, depths)):
        left, right = int(rng.integers(0, 3)), width - 1 - int(rng.integers(0, 3))
        a[cursor:cursor + depth, left:right + 1] = source
        a[cursor:cursor + depth, left] = target
        a[cursor:cursor + depth, right] = target
        terrain = []
        value = int(rng.integers(1, floor_cap + 1))
        for c in range(left + 1, right):
            if rng.random() < .45:
                value = int(rng.integers(1, floor_cap + 1))
            terrain.append(value)
        # Every tray has a genuinely uneven floor, including a one-column tooth.
        terrain[int(rng.integers(len(terrain)))] = 1
        terrain[int(rng.integers(len(terrain)))] = floor_cap
        if len(set(terrain)) == 1:
            terrain[0], terrain[-1] = 1, floor_cap
        floors = {}
        for c, f in zip(range(left + 1, right), terrain):
            floors[c] = cursor + depth - f
            a[floors[c]:cursor + depth, c] = target
        bands.append(dict(top=cursor, bottom=cursor + depth - 1, left=left, right=right,
                          route=source, rail=target, floor=floors))
        cursor += depth + (separations[i] if i < count - 1 else 0)
    order = list(map(int, rng.permutation(count)))
    unused_width = width - sum(widths) - gap * (count - 1)
    x = int(rng.integers(unused_width + 1))
    for i in order:
        s = stencils[i]
        y = int(rng.integers(header - s['shape'][0] + 1))
        for r, c in s['mask']:
            a[y + r, x + c] = STENCIL
        sl, tl = s['labels']
        a[y + sl[0], x + sl[1]] = routes[i]
        a[y + tl[0], x + tl[1]] = rails[i]
        x += s['shape'][1] + gap
    # Component placement is independent of the printer locations and of band order.
    placed = []

    def put(i, mask, row=None, col=None, touching=False):
        mh, mw = 1 + max(r for r, c in mask), 1 + max(c for r, c in mask)
        offsets = stencils[i]['offsets']
        lo = bands[i]['left'] + 1 - min(c for r, c in offsets)
        hi = bands[i]['right'] - 1 - max(c for r, c in offsets) - (mw - 1)
        if hi < lo:
            return False
        positions = [(stage_start + r, c) for r in range(stage_h - mh + 1) for c in range(lo, hi + 1)]
        if row is not None:
            positions = [p for p in positions if p[0] == stage_start + row]
        if col is not None:
            positions = [p for p in positions if p[1] == col]
        for k in rng.permutation(len(positions)):
            top, left = positions[int(k)]
            cells = {(top + r, left + c) for r, c in mask}
            if any(a[p] != 0 for p in cells):
                continue
            if any(a[r + dr, c + dc] == routes[i] for r, c in cells for dr, dc in EIGHT
                   if 0 <= r + dr < stop and 0 <= c + dc < width):
                continue
            if touching and not any(a[r + dr, c + dc] in routes and a[r + dr, c + dc] != routes[i]
                                    for r, c in cells for dr, dc in FOUR
                                    if 0 <= r + dr < stop and 0 <= c + dc < width):
                continue
            for p in cells:
                a[p] = routes[i]
            placed.append((routes[i], frozenset(cells)))
            return True
        return False

    if lesson == 'split':
        # Vary a two-color interlocking motif rather than copying A's nested ring.
        order2 = list(map(int, rng.permutation(2)))
        i, j = order2
        length = int(rng.integers(2, 5))
        mask = tuple([(0, c) for c in range(length)] + [(1, 0)])
        if not put(i, mask, row=int(rng.integers(stage_h - 1))):
            return None
        if not put(j, _piece(rng, 2, 3), touching=True):
            return None
    elif lesson == 'independent':
        if not put(0, ((0, 0), (0, 1), (1, 0)), row=0):
            return None
        if not put(0, ((0, 0), (0, 1)), row=stage_h - 1):
            return None
    for i in range(count):
        if not any(color == routes[i] for color, cells in placed):
            if not put(i, _piece(rng, piece_h, int(rng.integers(2, 5)))):
                return None
    for _ in range(int(rng.integers(1, 5))):
        i = int(rng.integers(count))
        put(i, _piece(rng, piece_h, int(rng.integers(1, 5))))
    try:
        decoded = _parse_canonical(a)
        # Construction ground truth uses sampled components, offsets and floors;
        # decoding must recover all of them without these records.
        _require(set(placed) == set(decoded['components']), "source components unexpectedly merged")
        out = a.copy()
        out[:stop] = 0
        for color, cells in placed:
            i = routes.index(color)
            cloud = _cloud(cells, stencils[i]['offsets'])
            shift = min(bands[i]['floor'][c] - r - 1 for r, c in cloud)
            for r, c in cloud:
                _require(bands[i]['top'] <= r + shift < bands[i]['floor'][c], "construction overflow")
                out[r + shift, c] = rails[i]
        decoded['output'] = _render(decoded)
        _require(np.array_equal(out, decoded['output']), "construction/decoder disagreement")
        if not all(_differs(decoded, v) for v in ('no_stencil', 'flat_floor', 'per_pixel', 'center')):
            return None
        if lesson == 'independent' and not _differs(decoded, 'merge_color'):
            return None
        if lesson == 'independent' and not _differs(decoded, 'solid_box'):
            return None
        pair = {'input': np.rot90(a, rotation).copy(), 'output': np.rot90(out, rotation).copy()}
        _analyze(pair['input'])  # Also exclude accidental rotational parses.
        return pair
    except AssertionError:
        return None


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(int(seed))
    turns = list(map(int, rng.permutation(4)))
    held_out = bool(rng.integers(2))
    puzzle = {'train': [], 'test': []}
    seen, known_stencils = set(), set()
    for split, n in (('train', num_train), ('test', num_test)):
        for index in range(n):
            lesson = ('split' if index == 0 else 'independent' if index == 1 else None) if split == 'train' else None
            rotation = (turns[index % 2] if split == 'train' else turns[2 + index % 2]) if held_out else turns[0]
            pair = None
            # Finite attempts; each retains at most two 30x30 int8 arrays.
            for _ in range(240):
                count = 2 if lesson == 'split' else int(_pick(rng, (1, 1, 2))) if lesson else int(_pick(rng, (1, 2, 2, 3)))
                candidate = _attempt(rng, lesson, count, rotation,
                                     known_stencils if split == 'test' else set())
                if candidate is not None and _key(candidate['input']) not in seen:
                    pair = candidate
                    break
            _require(pair is not None, 'bounded construction exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
            if split == 'train':
                a = _analyze(pair['input'])
                known_stencils.update(_effective(rec['offsets']) for rec in a['records'].values())
    validate_generated_puzzle(puzzle)
    return puzzle


def _contact(a):
    grid = a['grid']
    cells = set().union(*(part for color, part in a['components']))
    return any((r + dr, c + dc) in cells and grid[r, c] != grid[r + dr, c + dc]
               for r, c in cells for dr, dc in FOUR)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, 'illegal counts')
    seen, parsed = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape and _key(inp) not in seen, 'duplicate input or changed dimensions')
            seen.add(_key(inp))
            a = _analyze(inp)
            expected = np.rot90(a['output'], -a['turn'])
            _require(np.array_equal(expected, out), 'incorrect docked stencil output')
            _require(not np.array_equal(inp, out), 'identity example')
            _require(not np.any(a['output'][:a['stop']]) and not np.any(out == 8), 'work field not cleared')
            for b in a['bands']:
                _require(len(set(b['floor'].values())) > 1, 'flat terrain is outside the sampled family')
                for c, first in b['floor'].items():
                    _require(np.all(a['output'][first:b['bottom'] + 1, c] == b['rail']), 'floor changed')
                _require(np.all(a['output'][b['top']:b['bottom'] + 1, b['left']] == b['rail']) and
                         np.all(a['output'][b['top']:b['bottom'] + 1, b['right']] == b['rail']), 'rail changed')
            for color, cells in a['components']:
                b = next(b for b in a['bands'] if b['route'] == color)
                cloud = _cloud(cells, a['records'][color]['offsets'])
                landed = _dock(cloud, b)
                _require(any(r + 1 == b['floor'][c] for r, c in landed), 'no first contact')
                _require(all(a['output'][p] == b['rail'] for p in landed), 'missing footprint')
            parsed[split].append(a)
    train = parsed['train']
    _require(any(_contact(a) for a in train), 'training needs a multicolor splitting lesson')
    _require(any(_differs(a, 'merge_color') for a in train), 'training needs independent same-color bodies')
    for variant in ('no_stencil', 'flat_floor', 'per_pixel', 'center', 'solid_box'):
        _require(any(_differs(a, variant) for a in train), 'undistinguished shortcut: ' + variant)
    _require(any((0, 0) not in rec['offsets'] for a in train for rec in a['records'].values()),
             'training must show an empty registration')
    known = {_effective(rec['offsets']) for a in train for rec in a['records'].values()}
    _require(all(any(_effective(rec['offsets']) not in known for rec in a['records'].values()) for a in parsed['test']),
             'test must read at least one fresh registered footprint')


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded, input-derived diversity and semantic checks over 42 seeds; no I/O."""
    digests, shapes, stencils, pieces, floors = set(), set(), set(), set(), set()
    counts, directions, palette_roles = set(), set(), set()
    hole_splits, support_splits = set(), set()
    overlaps, catches, heldouts = 0, 0, 0
    for seed in range(42):
        p = generate_puzzle(seed)
        digests.add(_fingerprint(p))
        if seed < 2:
            _require(_fingerprint(p) == _fingerprint(generate_puzzle(seed)), 'nondeterminism')
        train_turns, test_turns = set(), set()
        for split in ('train', 'test'):
            for pair in p[split]:
                a = _analyze(pair['input'])
                shapes.add(pair['input'].shape)
                counts.add(len(a['bands']))
                directions.add(a['turn'])
                (train_turns if split == 'train' else test_turns).add(a['turn'])
                for rec in a['records'].values():
                    stencils.add(_effective(rec['offsets']))
                    palette_roles.add((rec['source'], rec['target']))
                    if (0, 0) not in rec['offsets']:
                        hole_splits.add(split)
                for b in a['bands']:
                    floors.add(tuple(b['bottom'] - f + 1 for f in b['floor'].values()))
                for color, cells in a['components']:
                    pieces.add(_normalized(cells))
                    offsets = a['records'][color]['offsets']
                    cloud = _cloud(cells, offsets)
                    overlaps += len(cloud) < len(cells) * len(offsets)
                    b = next(b for b in a['bands'] if b['route'] == color)
                    anchor_columns = {c for r, c in cells}
                    if any(r + 1 == b['floor'][c] and c not in anchor_columns for r, c in _dock(cloud, b)):
                        support_splits.add(split)
                catches += _differs(a, 'flat_floor')
        heldouts += bool(test_turns - train_turns)
    _require(len(digests) == 42 and len(shapes) >= 35, 'insufficient puzzle/canvas diversity')
    _require(counts == {1, 2, 3} and directions == {0, 1, 2, 3}, 'missing tray count or movement orientation')
    _require(len(stencils) >= 50 and len(pieces) >= 15 and len(floors) >= 100, 'narrow structural variation')
    _require(len(palette_roles) >= 35 and overlaps > 20 and catches >= 42 and heldouts > 5,
             'unexercised interaction or role variation')
    _require(hole_splits == support_splits == {'train', 'test'}, 'missing hole or overhang interaction in a split')
    # Exact hand-calculated terrain/contact regression, including union overlap.
    b = dict(top=1, bottom=6, left=0, right=8, floor={c: 6 for c in range(1, 8)})
    b['floor'][4] = 4
    cells = {(0, 2), (0, 3), (1, 2)}
    offsets = ((0, 0), (0, 1), (1, 1))
    cloud = _cloud(cells, offsets)
    _require(cloud == {(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3)}, 'union regression')
    _require(_dock(cloud, b) == {(2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4), (4, 3)},
             'raised tooth must support an overhang before the lower tail bottoms out')
    # Complete hand-calculated input/output, with two disconnected bodies of
    # one color, shared stencil-copy pixels, and an overhang supported by a tooth.
    manual = np.zeros((17, 14), dtype=np.int8)
    manual[0, :4] = (1, 2, 8, 8)
    manual[1, 3] = 8
    manual[5, 5:7] = 1
    manual[6, 5] = 1
    manual[8, 9:11] = 1
    manual[10:17, :] = 1
    manual[10:17, (0, 13)] = 2
    manual[16, :] = 2
    manual[14:17, 7] = 2
    expected = manual.copy()
    expected[:10] = 0
    for p in ((12, 5), (12, 6), (12, 7), (13, 5), (13, 6), (13, 7), (14, 6),
              (14, 9), (14, 10), (14, 11), (15, 10), (15, 11)):
        expected[p] = 2
    for turn in range(4):
        original = np.rot90(manual, turn).copy()
        snapshot = original.copy()
        _require(np.array_equal(solve(original), np.rot90(expected, turn)), 'full manual regression')
        _require(np.array_equal(original, snapshot), 'solver mutated its input')
    for n in range(2, 11):
        for k in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(701 + n * 5 + k, n, k)
            _require(len(p['train']) == n and len(p['test']) == k, 'counts ignored')
    for n, k in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, n, k)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    broken = generate_puzzle(76, 2, 1)
    broken['test'][0]['output'] = broken['test'][0]['output'].copy()
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        return
    raise AssertionError('corrupt output accepted')
