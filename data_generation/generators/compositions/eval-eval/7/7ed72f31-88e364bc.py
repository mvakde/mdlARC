"""Symmetry cargo: virtual completion determines rigid-body docking clearance.

Standard library and NumPy only. The public solver reads a single input; it does
not infer conventions from labels or consult generation state.
"""
from collections import Counter
import json
import numpy as np


DIRECTIONS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
CARDINAL = tuple(d for d in DIRECTIONS if 0 in d)
DIAGONAL = tuple(d for d in DIRECTIONS if 0 not in d)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    groups = []
    steps = DIRECTIONS if diagonal else CARDINAL
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
        groups.append(sorted(cells))
    return groups


def _bbox(cells):
    rows, cols = zip(*cells)
    return min(rows), max(rows), min(cols), max(cols)


def _fingerprint(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= s <= 30 for s in a.shape), 'grid dimensions')
    _require(np.issubdtype(a.dtype, np.integer), 'integer cells required')
    _require(bool(np.all((0 <= a) & (a <= 9))), 'color range')
    return a.astype(np.int8, copy=True)


def _background(grid):
    values, counts = np.unique(grid, return_counts=True)
    _require(np.count_nonzero(counts == counts.max()) == 1, 'background frequency tie')
    return int(values[counts.argmax()])


def _decode_key(grid, bg):
    keys = []
    for component in _components(grid != bg, diagonal=True):
        r0, r1, c0, c1 = _bbox(component)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if h < 4 or w < 4 or len(component) != h * w:
            continue
        sub = grid[r0:r1 + 1, c0:c1 + 1]
        wall = int(sub[0, 0])
        border = np.concatenate((sub[0], sub[-1], sub[:, 0], sub[:, -1]))
        other = set(map(int, np.unique(sub))) - {wall}
        if len(other) != 2 or np.any(border != wall):
            continue
        interpretations = []
        for tip in sorted(other):
            guide = next(iter(other - {tip}))
            tips = set(map(tuple, np.argwhere(sub == tip).tolist()))
            tails = set(map(tuple, np.argwhere(sub == guide).tolist()))
            if len(tips) < 2:
                continue
            for dr, dc in DIRECTIONS:
                used, lengths = set(), []
                for rr, cc in sorted(tips):
                    stroke = []
                    q = rr - dr, cc - dc
                    while q in tails:
                        stroke.append(q)
                        q = q[0] - dr, q[1] - dc
                    if len(stroke) < 2 or used.intersection(stroke):
                        break
                    used.update(stroke)
                    lengths.append(len(stroke))
                if len(lengths) == len(tips) and used == tails:
                    interpretations.append((guide, tip, (dr, dc), tuple(lengths)))
        if len(interpretations) == 1:
            guide, tip, direction, lengths = interpretations[0]
            keys.append(dict(bbox=(r0, r1, c0, c1), wall=wall, guide=guide,
                             tip=tip, direction=direction, lengths=lengths))
    _require(len(keys) == 1, 'exactly one unambiguous arrow key required')
    return keys[0]


def _classify(guide):
    r0, r1, c0, c1 = _bbox(guide)
    if len(guide) == 1:
        return 'P'
    if r0 == r1 and len(guide) == c1 - c0 + 1:
        return 'H'
    if c0 == c1 and len(guide) == r1 - r0 + 1:
        return 'V'
    raise AssertionError('guide must be a straight line or singleton')


def _copy_cells(payload, guide, kind):
    ar, ac = guide[0]
    if kind == 'H':
        return {(2 * ar - r, c): color for (r, c), color in payload.items()}
    if kind == 'V':
        return {(r, 2 * ac - c): color for (r, c), color in payload.items()}
    return {(2 * ar - r, 2 * ac - c): color for (r, c), color in payload.items()}


def _orientation_witness(payload, copied):
    # Opposite-bounding-box translation without changing orientation.
    r0, _, c0, _ = _bbox(payload)
    a0, _, b0, _ = _bbox(copied)
    translated = {(r + a0 - r0, c + b0 - c0): color
                  for (r, c), color in payload.items()}
    return translated != copied


def _read_input(grid):
    bg = _background(grid)
    key = _decode_key(grid, bg)
    wall, guide_color = key['wall'], key['guide']
    key_mask = np.zeros(grid.shape, bool)
    r0, r1, c0, c1 = key['bbox']
    key_mask[r0:r1 + 1, c0:c1 + 1] = True
    wall_mask = (grid == wall) & ~key_mask
    _require(wall_mask.any(), 'no arena wall')
    # Key cells block exterior flood fill but never lie inside a cargo chamber.
    open_mask = ~wall_mask & ~key_mask
    labels = np.full(grid.shape, -1, dtype=int)
    regions = []
    for cells in _components(open_mask):
        edge = any(r in (0, grid.shape[0] - 1) or c in (0, grid.shape[1] - 1)
                   for r, c in cells)
        if not edge:
            index = len(regions)
            mask = np.zeros(grid.shape, bool)
            for q in cells:
                mask[q] = True
                labels[q] = index
            regions.append(mask)
    cargos = []
    foreground = (grid != bg) & ~wall_mask & ~key_mask
    for source in _components(foreground, diagonal=True):
        guide = [q for q in source if grid[q] == guide_color]
        _require(bool(guide), 'cargo lacks guide')
        local = np.zeros(grid.shape, bool)
        for q in guide:
            local[q] = True
        _require(len(_components(local, True)) == 1, 'multiple cargo guides')
        kind = _classify(guide)
        payload = {q: int(grid[q]) for q in source if grid[q] != guide_color}
        _require(bool(payload) and key['tip'] in payload.values(), 'cargo needs tip-colored payload')
        local[:] = False
        for q in payload:
            local[q] = True
        _require(len(_components(local, True)) == 1, 'payload must be eight-connected')
        ar, ac = guide[0]
        a0, a1, b0, b1 = _bbox(payload)
        if kind == 'H':
            _require(a1 < ar or a0 > ar, 'payload crosses horizontal guide')
        elif kind == 'V':
            _require(b1 < ac or b0 > ac, 'payload crosses vertical guide')
        else:
            _require((a1 <= ar or a0 >= ar) and (b1 <= ac or b0 >= ac),
                     'point payload must lie in a closed quadrant')
        _require(min(max(abs(r - rr), abs(c - cc)) for r, c in payload
                     for rr, cc in guide) == 1, 'payload must touch guide')
        copied = _copy_cells(payload, guide, kind)
        _require(set(copied).isdisjoint(source), 'symmetry copy overlaps source')
        body = dict(payload)
        body.update(copied)
        body.update({q: guide_color for q in guide})
        _require(all(0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] for r, c in body),
                 'initial completed cargo leaves grid')
        owners = {int(labels[q]) for q in body}
        _require(len(owners) == 1 and -1 not in owners, 'cargo must fit in one enclosed chamber')
        _require(all(grid[q] == bg for q in copied), 'virtual half overlaps input')
        region = regions[next(iter(owners))]
        cargos.append(dict(kind=kind, guide=guide, payload=payload, copied=copied,
                           body=body, source=source, region=region,
                           orientation=_orientation_witness(payload, copied)))
    _require(bool(cargos), 'no cargo')
    return dict(background=bg, key=key, cargos=cargos, walls=wall_mask, key_mask=key_mask)


def _shifted(mask, dr, dc):
    """Result[r,c] is mask[r+dr,c+dc], with False beyond the boundary."""
    h, w = mask.shape
    result = np.zeros_like(mask)
    r0, r1 = max(0, -dr), min(h, h - dr)
    c0, c1 = max(0, -dc), min(w, w - dc)
    if r0 < r1 and c0 < c1:
        result[r0:r1, c0:c1] = mask[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
    return result


def _fit_map(free, cells):
    result = np.ones(free.shape, bool)
    for dr, dc in cells:
        result &= _shifted(free, dr, dc)
    return result


def _lookup(mask, q):
    r, c = q
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])


def _trace(fit, start, direction, corner=True):
    r, c = start
    dr, dc = direction
    path = [(r, c)]
    for _ in range(31):
        if not _lookup(fit, (r + dr, c + dc)):
            return path
        if corner and dr and dc and (not _lookup(fit, (r + dr, c))
                                     or not _lookup(fit, (r, c + dc))):
            return path
        r, c = r + dr, c + dc
        path.append((r, c))
    raise AssertionError('motion failed to stop within grid')


def _envelope(cells, path, direction):
    poses = set(path)
    dr, dc = direction
    if dr and dc:
        for r, c in path[:-1]:
            poses.add((r + dr, c))
            poses.add((r, c + dc))
    return {(r + a, c + b) for r, c in cells for a, b in poses}


def _box_cells(cells):
    r0, r1, c0, c1 = _bbox(cells)
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _predict(grid, info, mode='full'):
    out = grid.copy()
    for cargo in info['cargos']:
        for q in cargo['source']:
            out[q] = info['background']
    direction = info['key']['direction']
    if mode == 'reverse':
        direction = -direction[0], -direction[1]
    sweeps = set()
    for cargo in info['cargos']:
        body = cargo['body']
        # Use a visible guide cell as the anchor, with relative body offsets.
        # This also permits negative translations within an ordinary fit map.
        anchor = cargo['guide'][0]
        offsets = {(r - anchor[0], c - anchor[1]): color for (r, c), color in body.items()}
        visible = [(r - anchor[0], c - anchor[1]) for r, c in cargo['source']]
        fit = _fit_map(cargo['region'], offsets)
        path = _trace(fit, anchor, direction)
        visible_path = _trace(_fit_map(cargo['region'], visible), anchor, direction)
        no_corner = _trace(fit, anchor, direction, False)
        box_fit = _fit_map(cargo['region'], _box_cells(offsets))
        box_path = _trace(box_fit, anchor, direction)
        distance = len(path) - 1
        cargo.update(distance=distance, visible_distance=len(visible_path) - 1,
                     corner_witness=len(no_corner) > len(path),
                     bbox_witness=_lookup(box_fit, anchor) and len(box_path) < len(path),
                     ghost_witness=len(visible_path) > len(path), path=path)
        envelope = _envelope(offsets, path, direction)
        if mode == 'full':
            _require(sweeps.isdisjoint(envelope), 'cargo sweeps interact')
        sweeps.update(envelope)
        if mode == 'in_place':
            destination = anchor
        elif mode == 'visible':
            destination = visible_path[-1]
        elif mode == 'no_corner':
            destination = no_corner[-1]
        else:
            destination = path[-1]
        for (r, c), color in offsets.items():
            rr, cc = r + destination[0], c + destination[1]
            if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
                out[rr, cc] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover all roles, symmetry mappings and wall clearance from this input."""
    grid = _grid(input_grid)
    return _predict(grid, _read_input(grid))


def _make_key(rng, direction, wall, guide, tip):
    dr, dc = direction
    count = int(rng.choice([2, 2, 3]))
    length = int(rng.integers(2, 5))
    spacing = int(rng.integers(2, 4))
    points = []
    for i in range(count):
        stagger = int(rng.integers(2))
        base = (0, i * spacing) if dr else (i * spacing, 0)
        for j in range(length + 1):
            points.append((base[0] + (j + stagger) * dr,
                           base[1] + (j + stagger) * dc,
                           tip if j == length else guide))
    r0, r1, c0, c1 = _bbox([(r, c) for r, c, _ in points])
    padding = [int(rng.choice([1, 1, 2])) for _ in range(4)]
    key = np.full((r1 - r0 + 1 + padding[0] + padding[1],
                   c1 - c0 + 1 + padding[2] + padding[3]), wall, np.int8)
    for r, c, color in points:
        key[r - r0 + padding[0], c - c0 + padding[2]] = color
    return key


def _silhouette(rng, h, w, witness):
    family = int(rng.integers(5))
    mask = np.zeros((h, w), bool)
    if family == 0:
        mask[:] = True
    elif family == 1:
        mask[0, 0] = True
        target = int(rng.integers(2, h * w + 1))
        for _ in range(target - 1):
            frontier = set()
            for r, c in np.argwhere(mask):
                for dr, dc in DIRECTIONS:
                    a, b = int(r + dr), int(c + dc)
                    if 0 <= a < h and 0 <= b < w and not mask[a, b]:
                        frontier.add((a, b))
            if frontier:
                cells = sorted(frontier)
                mask[cells[int(rng.integers(len(cells)))]] = True
    elif family == 2:
        col = int(rng.integers(w))
        for row in range(h):
            mask[row, col] = True
            col = int(np.clip(col + rng.choice([-1, 1]), 0, w - 1))
    elif family == 3:
        mask[:] = True
        mask[:int(rng.integers(1, h)), :int(rng.integers(1, w))] = False
    else:
        mask[0] = mask[-1] = True
        mask[:, 0] = mask[:, -1] = True
    if rng.random() < .5:
        mask = mask[::-1]
    if rng.random() < .5:
        mask = mask[:, ::-1]
    rows, cols = np.nonzero(mask)
    mask = mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1].copy()
    # Witnesses are checked for the actual operation in _prototype.
    return mask


def _prototype(rng, kind, scale, palette, direction, witness=False, ghost=False):
    bg, wall, guide_color, tip = palette[:4]
    for attempt in range(20):
        h, w = int(rng.integers(2, scale + 1)), int(rng.integers(2, scale + 1))
        mask = _silhouette(rng, h, w, witness)
        if attempt == 19:
            mask = np.array([[1, 1], [1, 0]], bool)
        cells = set(map(tuple, np.argwhere(mask).tolist()))
        if kind == 'P':
            # Force a corner contact; offsets include center-aligned payloads.
            target = min(cells)
            r = c = 0
            cells.add((0, 0))
            while (r, c) != target:
                r += int(np.sign(target[0] - r))
                c += int(np.sign(target[1] - c))
                cells.add((r, c))
            a, b = [(1, 1), (0, 1), (1, 0)][int(rng.integers(3))]
            sr = -direction[0] if ghost and direction[0] else int(rng.choice([-1, 1]))
            sc = -direction[1] if ghost and direction[1] else int(rng.choice([-1, 1]))
            payload_coords = sorted((sr * (r + a), sc * (c + b)) for r, c in cells)
            guide = [(0, 0)]
        else:
            near = [c for r, c in cells if r == 0]
            contact = int(rng.choice(sorted(near)))
            lo = int(rng.integers(contact - 2, contact + 1))
            length = int(rng.integers(2, max(3, mask.shape[1] + 3)))
            hi = max(contact, lo + length - 1)
            guide = [(0, c) for c in range(lo, hi + 1)]
            component = direction[0] if kind == 'H' else direction[1]
            sign = -component if ghost and component else int(rng.choice([-1, 1]))
            payload_coords = [(sign * (r + 1), c) for r, c in sorted(cells)]
            if kind == 'V':
                guide = [(c, r) for r, c in guide]
                payload_coords = [(c, r) for r, c in payload_coords]
        payload = {q: tip for q in payload_coords}
        if len(palette) > 4 and rng.random() < .40:
            secondary = int(rng.choice(palette[4:]))
            for q in payload_coords[1:]:
                if rng.random() < .35:
                    payload[q] = secondary
        copied = _copy_cells(payload, guide, kind)
        if witness and not _orientation_witness(payload, copied):
            continue
        body = dict(payload)
        body.update(copied)
        body.update({q: guide_color for q in guide})
        source = dict(payload)
        source.update({q: guide_color for q in guide})
        return dict(kind=kind, guide=guide, payload=payload, copied=copied,
                    body=body, source=source)
    raise AssertionError('asymmetric cargo construction failed')


def _arena(rng, h, w, complexity, fallback=False):
    footprint = np.ones((h, w), bool)
    style = 0 if fallback else int(rng.integers(4))
    if style == 1:
        for corner in rng.choice(4, int(rng.integers(1, 4)), replace=False):
            a, b = int(rng.integers(1, h // 3)), int(rng.integers(1, w // 3))
            rs = slice(0, a) if corner < 2 else slice(h - a, h)
            cs = slice(0, b) if corner % 2 == 0 else slice(w - b, w)
            footprint[rs, cs] = False
    elif style == 2:
        yy, xx = np.indices((h, w))
        cy, cx = (h - 1) / 2, (w - 1) / 2
        power = float(rng.uniform(1.3, 2.4))
        footprint = abs((yy - cy) / (cy + .5)) ** power + abs((xx - cx) / (cx + .5)) ** power <= 1.25
    elif style == 3:
        left = right = 0
        for row in range(h):
            left = int(np.clip(left + rng.integers(-1, 2), 0, max(1, w // 5)))
            right = int(np.clip(right + rng.integers(-1, 2), 0, max(1, w // 5)))
            footprint[row, :left] = False
            if right:
                footprint[row, w - right:] = False
    interior = footprint.copy()
    for dr, dc in CARDINAL:
        interior &= _shifted(footprint, dr, dc)
    walls = footprint & ~interior
    if not fallback:
        for _ in range(int(rng.integers(complexity + 1))):
            side = int(rng.integers(4))
            if side < 2:
                col = int(rng.integers(3, w - 3))
                length = int(rng.integers(2, max(3, h // 2)))
                rows = slice(0, length) if side == 0 else slice(h - length, h)
                walls[rows, col] |= footprint[rows, col]
            else:
                row = int(rng.integers(3, h - 3))
                length = int(rng.integers(2, max(3, w // 2)))
                cols = slice(0, length) if side == 2 else slice(w - length, w)
                walls[row, cols] |= footprint[row, cols]
        # Interior pegs make exact silhouette clearance useful.
        for _ in range(int(rng.integers(3))):
            q = int(rng.integers(3, h - 3)), int(rng.integers(3, w - 3))
            if interior[q]:
                walls[q] = True
        if rng.random() < .16:
            col = int(rng.integers(5, w - 5))
            walls[:, col] |= footprint[:, col]
    return footprint, walls


def _halo(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _maps(free, proto):
    return (_fit_map(free, proto['body']), _fit_map(free, proto['source']))


def _record(free, proto, start, direction, maps=None):
    full, visible = _maps(free, proto) if maps is None else maps
    if not _lookup(full, start):
        return None
    path = _trace(full, start, direction)
    vpath = _trace(visible, start, direction)
    npath = _trace(full, start, direction, False)
    return dict(start=start, path=path, envelope=_envelope(proto['body'], path, direction),
                distance=len(path) - 1, ghost=len(vpath) > len(path),
                corner=len(npath) > len(path))


def _one_region(free, body, start):
    cells = {(r + start[0], c + start[1]) for r, c in body}
    for component in _components(free):
        if cells <= set(component):
            return True
    return False


def _acceptable(record, proto, occupied, initial_halo, moving, ghost, corner):
    if record is None or (moving and record['distance'] == 0):
        return False
    if ghost and not record['ghost'] or corner and not record['corner']:
        return False
    if not record['envelope'].isdisjoint(occupied):
        return False
    a, b = record['start']
    source = {(r + a, c + b) for r, c in proto['source']}
    return source.isdisjoint(initial_halo)


def _select_cargo(rng, free, proto, direction, occupied, initial_halo,
                  ghost=False, corner=False, moving=True):
    maps = _maps(free, proto)
    anchors = [tuple(map(int, q)) for q in np.argwhere(maps[0])]
    rng.shuffle(anchors)
    for start in anchors:
        record = _record(free, proto, start, direction, maps)
        if _acceptable(record, proto, occupied, initial_halo, moving, ghost, corner):
            if _one_region(free, proto['body'], start):
                return record, None
    if not (ghost or corner):
        return None, None
    # Add an ordinary wall peg to exhibit the required interaction. Never place
    # it on an earlier legal pose, on another sweep, or on the visible-only ray.
    dr, dc = direction
    for start in anchors[:100]:
        base_path = _trace(maps[0], start, direction)
        for n in range(1, min(4, len(base_path) - 1)):
            prefix = base_path[:n + 1]
            prior = _envelope(proto['body'], prefix, direction)
            if not prior.isdisjoint(occupied):
                continue
            a, b = prefix[-1]
            next_pose = {(r + a + dr, c + b + dc) for r, c in proto['body']}
            visible_path = base_path[:n + 2]
            visible_sweep = _envelope(proto['source'], visible_path, direction)
            if corner:
                candidates = ({(r + a + dr, c + b) for r, c in proto['body']}
                              | {(r + a, c + b + dc) for r, c in proto['body']}) - next_pose
            else:
                candidates = next_pose
            candidates -= prior | occupied
            if ghost:
                candidates -= visible_sweep
            choices = sorted(q for q in candidates if _lookup(free, q))
            rng.shuffle(choices)
            for peg in choices[:30]:
                altered = free.copy()
                altered[peg] = False
                record = _record(altered, proto, start, direction)
                if _acceptable(record, proto, occupied, initial_halo, moving, ghost, corner):
                    if _one_region(altered, proto['body'], start):
                        return record, peg
    return None, None


def _layout(rng, arena, key, bg, side):
    ah, aw = arena.shape
    kh, kw = key.shape
    gap = int(rng.integers(1, 3))
    if side in (0, 1):
        if ah + gap + kh > 30:
            gap = 1
        h, w = ah + gap + kh, max(aw, kw)
    else:
        if aw + gap + kw > 30:
            gap = 1
        h, w = max(ah, kh), aw + gap + kw
    if h > 30 or w > 30:
        return None
    h += int(rng.integers(min(3, 31 - h)))
    w += int(rng.integers(min(3, 31 - w)))
    canvas = np.full((h, w), bg, np.int8)
    if side == 0:
        kr, ar = 0, kh + gap
        kc, ac = int(rng.integers(w - kw + 1)), int(rng.integers(w - aw + 1))
    elif side == 1:
        ar, kr = 0, ah + gap
        kc, ac = int(rng.integers(w - kw + 1)), int(rng.integers(w - aw + 1))
    elif side == 2:
        kc, ac = 0, kw + gap
        kr, ar = int(rng.integers(h - kh + 1)), int(rng.integers(h - ah + 1))
    else:
        ac, kc = 0, aw + gap
        kr, ar = int(rng.integers(h - kh + 1)), int(rng.integers(h - ah + 1))
    canvas[ar:ar + ah, ac:ac + aw] = arena
    canvas[kr:kr + kh, kc:kc + kw] = key
    return canvas, (ar, ac)


def _generate_pair(rng, palette, direction, specs, ghost, corner, test=False):
    bg, wall, guide_color, tip = palette[:4]
    key = _make_key(rng, direction, wall, guide_color, tip)
    side = int(rng.integers(4))
    for attempt in range(36):
        fallback = attempt >= 28
        max_h = min(25, 29 - key.shape[0]) if side < 2 else 25
        max_w = min(25, 29 - key.shape[1]) if side >= 2 else 25
        h = max_h if fallback else int(rng.integers(max(12, max_h - 8), max_h + 1))
        w = max_w if fallback else int(rng.integers(max(12, max_w - 8), max_w + 1))
        complexity = 1 if fallback else int(rng.integers(1, 4 + int(test)))
        footprint, walls = _arena(rng, h, w, complexity, fallback)
        free = footprint & ~walls
        placed, occupied, initial_halo = [], set(), set()
        failed = False
        for index, (kind, witness) in enumerate(specs):
            scale = 2 if fallback else int(rng.integers(2, 5 + int(test)))
            if len(specs) == 3:
                scale = min(scale, 3)
            proto = _prototype(rng, kind, scale, palette, direction, witness,
                               ghost=ghost and index == 0)
            record, peg = _select_cargo(rng, free, proto, direction, occupied, initial_halo,
                                        ghost=ghost and index == 0, corner=corner and index == 0,
                                        moving=index == 0 or rng.random() > .10)
            if record is None:
                failed = True
                break
            if peg is not None:
                walls[peg] = True
                free[peg] = False
            occupied.update(record['envelope'])
            a, b = record['start']
            initial_halo.update(_halo({(r + a, c + b) for r, c in proto['source']}))
            placed.append((proto, record))
        if failed:
            continue
        arena = np.full((h, w), bg, np.int8)
        arena[walls] = wall
        for proto, record in placed:
            a, b = record['start']
            for (r, c), color in proto['source'].items():
                arena[r + a, c + b] = color
        layout = _layout(rng, arena, key, bg, side)
        if layout is None:
            continue
        inp, (ar, ac) = layout
        out = inp.copy()
        for proto, record in placed:
            a, b = record['start']
            for r, c in proto['source']:
                out[r + a + ar, c + b + ac] = bg
        for proto, record in placed:
            a, b = record['path'][-1]
            for (r, c), color in proto['body'].items():
                out[r + a + ar, c + b + ac] = color
        # Accept only the actual observable interpretation, not sampler metadata.
        try:
            _require(_background(inp) == bg, 'background must dominate')
            info = _read_input(inp)
            predicted = _predict(inp, info)
            _require(np.array_equal(predicted, out), 'construction/readback mismatch')
            _require(not ghost or any(c['ghost_witness'] for c in info['cargos']), 'ghost witness lost')
            _require(not corner or any(c['corner_witness'] for c in info['cargos']), 'corner witness lost')
            return {'input': inp, 'output': out}
        except AssertionError:
            continue
    raise AssertionError('bounded cargo placement exhausted')


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not lo <= int(value) <= hi:
        raise ValueError(f'{name} must be between {lo} and {hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a non-negative integer')
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.permutation(10)[:int(rng.integers(4, 8))]))
    subsets = [('H',), ('V',), ('P',), ('H', 'V'), ('H', 'P'), ('V', 'P'), ('H', 'V', 'P')]
    kinds = subsets[int(rng.choice(7, p=[.08, .08, .18, .14, .15, .15, .22]))]
    pool = [CARDINAL[int(rng.integers(4))], DIAGONAL[int(rng.integers(4))]]
    if rng.random() < .5:
        remaining = [d for d in DIRECTIONS if d not in pool]
        pool.append(remaining[int(rng.integers(len(remaining)))])
    train_dirs = [pool[i % len(pool)] for i in range(nt)]
    rng.shuffle(train_dirs)
    corner_slot = next(i for i, d in enumerate(train_dirs) if d in DIAGONAL)
    ghost_slot = (corner_slot + 1 + int(rng.integers(nt - 1))) % nt
    plans = [[] for _ in range(nt)]
    slots = list(map(int, rng.permutation(nt)))
    for i, kind in enumerate(rng.permutation(kinds)):
        plans[slots[i % nt]].append((str(kind), True))
    for plan in plans:
        count = max(len(plan), int(rng.choice([1, 1, 2, 2, 3])))
        while len(plan) < count:
            plan.append((str(rng.choice(kinds)), False))
        rng.shuffle(plan)
    novel = [d for d in DIRECTIONS if d not in train_dirs]
    test_pool = novel if rng.random() < .6 else list(set(train_dirs))
    # Sort before random indexing: hash randomization must not affect seeds.
    test_pool = sorted(test_pool)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in [('train', nt), ('test', nq)]:
        for index in range(count):
            if split == 'train':
                direction, specs = train_dirs[index], plans[index]
            else:
                direction = test_pool[int(rng.integers(len(test_pool)))]
                specs = [(str(rng.choice(kinds)), False)
                         for _ in range(int(rng.choice([1, 1, 2, 2, 3])))]
            for _ in range(8):
                pair = _generate_pair(rng, palette, direction, specs,
                                      ghost=split == 'train' and index == ghost_slot,
                                      corner=split == 'train' and index == corner_slot,
                                      test=split == 'test')
                fp = _fingerprint(pair['input'])
                if fp not in seen:
                    break
            _require(fp not in seen, 'duplicate inputs')
            seen.add(fp)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split list required')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'pair counts')
    seen, conventions, train_kinds, test_kinds, witnessed, train_dirs = set(), set(), set(), set(), set(), set()
    ghost = corner = moving = reverse = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'dimensions changed')
            fp = _fingerprint(inp)
            _require(fp not in seen, 'duplicate input')
            seen.add(fp)
            info = _read_input(inp)
            predicted = _predict(inp, info)
            _require(np.array_equal(predicted, out), 'incorrect completed docking')
            key = info['key']
            conventions.add((info['background'], key['wall'], key['guide'], key['tip']))
            fixed = info['walls'] | info['key_mask']
            _require(np.array_equal(inp[fixed], out[fixed]), 'wall/key changed')
            _require(any(c['distance'] > 0 for c in info['cargos']), 'pair needs moving cargo')
            # Count cargo cells independently of the preserved key.
            in_counts = Counter(map(int, inp[(inp != info['background']) & ~fixed]))
            out_counts = Counter(map(int, out[(out != info['background']) & ~fixed]))
            _require(out_counts[key['guide']] == in_counts[key['guide']], 'guide count changed')
            for color, count in in_counts.items():
                if color != key['guide']:
                    _require(out_counts[color] == 2 * count, 'payload color did not double')
            if split == 'train':
                train_dirs.add(key['direction'])
                train_kinds.update(c['kind'] for c in info['cargos'])
                witnessed.update(c['kind'] for c in info['cargos'] if c['orientation'])
                ghost |= (any(c['ghost_witness'] and c['distance'] > 0 for c in info['cargos'])
                          and not np.array_equal(_predict(inp, _read_input(inp), 'visible'), out))
                corner |= any(c['corner_witness'] and c['distance'] > 0 for c in info['cargos'])
                moving |= not np.array_equal(_predict(inp, _read_input(inp), 'in_place'), out)
                reverse |= not np.array_equal(_predict(inp, _read_input(inp), 'reverse'), out)
            else:
                test_kinds.update(c['kind'] for c in info['cargos'])
    _require(len(conventions) == 1, 'puzzle role palette changed')
    _require(test_kinds <= train_kinds <= witnessed, 'missing symmetry/orientation demonstrations')
    _require(len(train_dirs) >= 2 and ghost and corner and moving and reverse,
             'training lacks direction, ghost collision, corner or motion evidence')


def _measure(puzzle):
    rows = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            a = np.asarray(pair['input'])
            info = _read_input(a)
            _predict(a, info)
            wall_cells = list(map(tuple, np.argwhere(info['walls']).tolist()))
            r0, r1, c0, c1 = _bbox(wall_cells)
            sub = info['walls'][r0:r1 + 1, c0:c1 + 1]
            irregular = not (sub[0].all() and sub[-1].all() and sub[:, 0].all() and sub[:, -1].all())
            key = info['key']
            kr0, kr1, kc0, kc1 = key['bbox']
            dy, dx = kr0 + kr1 - r0 - r1, kc0 + kc1 - c0 - c1
            side = ('bottom' if dy > 0 else 'top') if abs(dy) > abs(dx) else ('right' if dx > 0 else 'left')
            silhouettes = []
            for cargo in info['cargos']:
                a0, a1, b0, b1 = _bbox(cargo['payload'])
                silhouettes.append((cargo['kind'], a1 - a0 + 1, b1 - b0 + 1,
                                    tuple(sorted((r - a0, c - b0) for r, c in cargo['payload']))))
            rows.append(dict(split=split, shape=a.shape, direction=key['direction'],
                             kinds=tuple(sorted({c['kind'] for c in info['cargos']})),
                             count=len(info['cargos']), key_shape=(kr1 - kr0 + 1, kc1 - kc0 + 1),
                             key_side=side, lengths=key['lengths'], irregular=irregular,
                             ghost=sum(c['ghost_witness'] for c in info['cargos']),
                             corner=sum(c['corner_witness'] for c in info['cargos']),
                             bbox=sum(c['bbox_witness'] for c in info['cargos']),
                             distances=[c['distance'] for c in info['cargos']],
                             silhouettes=silhouettes, palette=(info['background'], key['wall'], key['guide'], key['tip'])))
    return rows


def self_test():
    """Bounded variation and semantic invariant checks over 42 fixed seeds."""
    # Small hand-constructed fixtures have explicit docking distances. They
    # exercise the input-only solver outside the sampler's normal canvas sizes.
    for direction, arena_shape, center, payload_cells, peg, distance in [
            ((0, 1), (10, 14), (5, 4), [(3, 2), (3, 3), (4, 3)], (7, 10), 3),
            ((0, 1), (10, 14), (5, 4), [(3, 2), (3, 3), (4, 3)], (4, 8), 4),
            ((1, 1), (11, 15), (5, 5), [(3, 3), (3, 4), (4, 4)], (9, 7), 1)]:
        ah, aw = arena_shape
        kh, kw = (5, 5) if direction == (0, 1) else (5, 7)
        fixture = np.zeros((ah, aw + 1 + kw), dtype=np.int8)
        fixture[0, :aw] = fixture[ah - 1, :aw] = 6
        fixture[:ah, 0] = fixture[:ah, aw - 1] = 6
        fixture[peg] = 6
        legend = np.full((kh, kw), 6, np.int8)
        starts = [(1, 1), (3, 1)] if direction == (0, 1) else [(1, 1), (1, 3)]
        for r, c in starts:
            for j in range(3):
                legend[r + j * direction[0], c + j * direction[1]] = 7 if j == 2 else 2
        fixture[:kh, aw + 1:] = legend
        fixture[center] = 2
        for q in payload_cells:
            fixture[q] = 7
        expected = fixture.copy()
        expected[center] = 0
        for q in payload_cells:
            expected[q] = 0
        dr, dc = direction
        expected[center[0] + distance * dr, center[1] + distance * dc] = 2
        # Explicit arithmetic, not the generator's symmetry or tracing helpers.
        for r, c in payload_cells:
            expected[r + distance * dr, c + distance * dc] = 7
            expected[2 * center[0] - r + distance * dr,
                     2 * center[1] - c + distance * dc] = 7
        _require(np.array_equal(solve(fixture), expected), 'explicit docking regression')
    puzzles, rows, signatures = [], [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        rows.extend(_measure(puzzle))
        signatures.add(tuple(_fingerprint(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        # Equivariance tests exercise all four compass quadrants and guide kinds.
        pair = puzzle['test'][0]
        for k in (1, 2, 3):
            _require(np.array_equal(solve(np.rot90(pair['input'], k)), np.rot90(pair['output'], k)),
                     'rotation equivariance')
        _require(np.array_equal(solve(np.fliplr(pair['input'])), np.fliplr(pair['output'])),
                 'reflection equivariance')
    _require(len(signatures) == 42, 'seed diversity')
    _require({r['direction'] for r in rows} == set(DIRECTIONS), 'eight directions')
    _require({r['count'] for r in rows} == {1, 2, 3}, 'cargo counts')
    _require({k for r in rows for k in r['kinds']} == {'H', 'V', 'P'}, 'symmetry kinds')
    _require(len({r['shape'] for r in rows}) >= 30, 'canvas geometry variation')
    _require(len({r['key_shape'] for r in rows}) >= 12, 'key geometry variation')
    _require(len({s for r in rows for s in r['silhouettes']}) >= 50, 'silhouette variation')
    _require({r['key_side'] for r in rows} == {'top', 'bottom', 'left', 'right'}, 'key placements')
    _require(sum(r['irregular'] for r in rows) >= 20, 'irregular outlines')
    _require(sum(r['bbox'] for r in rows) >= 3, 'negative-space clearance evidence')
    _require(any(0 in r['distances'] for r in rows), 'stationary support')
    _require({n for r in rows for n in r['lengths']} >= {2, 3, 4}, 'arrow trail lengths')
    a, b = generate_puzzle(9123, 5, 3), generate_puzzle(9123, 5, 3)
    _require(all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s])
                 for k in ('input', 'output')), 'determinism')
    # Every legal count, with the smallest demonstration sets still checked.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(10000 + 100 * nt + nq, nt, nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, 'count contract')
    for nt, nq in [(1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (2.5, 1), (None, 1), (4, None)]:
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in puzzles[0][s]]
           for s in ('train', 'test')}
    bad['test'][0]['output'] = bad['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt docking label accepted')
    return dict(seeds=42, pairs=len(rows), canvas_shapes=len({r['shape'] for r in rows}),
                key_shapes=len({r['key_shape'] for r in rows}),
                silhouettes=len({s for r in rows for s in r['silhouettes']}),
                ghost_witnesses=sum(r['ghost'] for r in rows),
                corner_witnesses=sum(r['corner'] for r in rows),
                bbox_witnesses=sum(r['bbox'] for r in rows),
                irregular_arenas=sum(r['irregular'] for r in rows))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
