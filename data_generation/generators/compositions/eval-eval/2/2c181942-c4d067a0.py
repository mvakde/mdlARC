"""Compass-map workshop: locate, orient, and populate a lattice of hubs.

All answer conventions are fixed or visible in the input. No parent modules,
fixtures, seeds, training outputs, or sampled metadata enter solve().
"""
from __future__ import annotations

import numpy as np


_N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)),
             "grid cells must be integer colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _components(a, active, diagonal=True):
    """Monochrome components, with explicit eight-connected object semantics."""
    todo = set(map(tuple, np.argwhere(active)))
    found = []
    neighbors = _N8 if diagonal else _N4
    while todo:
        first = min(todo)
        todo.remove(first)
        color = int(a[first])
        cells, stack = [first], [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                p = (r + dr, c + dc)
                if p in todo and int(a[p]) == color:
                    todo.remove(p)
                    stack.append(p)
                    cells.append(p)
        found.append((color, np.array(cells, dtype=int)))
    return found


def _connected(mask, diagonal=True):
    return len(_components(np.ones(mask.shape, dtype=np.int8), mask, diagonal)) == 1


def _mask(*rows):
    return np.array([[x == "#" for x in row] for row in rows], dtype=bool)


def _canonical_options(mask):
    """Return distinct shapes with their centered two-cell edge facing down."""
    result = {}
    for turn in range(4):
        rotated = np.rot90(mask, turn).copy()
        h, w = rotated.shape
        if h < 2 or w < 2 or w % 2:
            continue
        edge = np.zeros(w, dtype=bool)
        edge[w // 2 - 1:w // 2 + 1] = True
        if np.array_equal(rotated[-1], edge):
            result.setdefault(_key(rotated), (rotated, []))[1].append(turn)
    return list(result.values())


def _canonical(mask):
    options = _canonical_options(mask)
    _require(_connected(mask) and len(options) == 1,
             "matching piece has missing or competing attachment edges")
    canonical, turns = options[0]
    _require(2 <= canonical.shape[0] <= 4 and canonical.shape[1] in (2, 4, 6, 8),
             "piece exceeds the family geometry")
    return canonical, turns


def _hub_points(r, c):
    return np.array(((r, c + 1), (r, c + 2), (r + 1, c + 3), (r + 2, c + 3),
                     (r + 3, c + 1), (r + 3, c + 2), (r + 1, c), (r + 2, c)), dtype=int)


def _paint_hub(a, r, c, labels):
    points = _hub_points(r, c)
    a[points[:, 0], points[:, 1]] = np.repeat(labels, 2)


def _find_hubs(a, bg):
    template = np.zeros((4, 4), dtype=bool)
    template[0, 1:3] = template[3, 1:3] = True
    template[1:3, 0] = template[1:3, 3] = True
    hubs = []
    for r in range(a.shape[0] - 3):
        for c in range(a.shape[1] - 3):
            if a[r, c + 1] == bg or a[r, c + 1] != a[r, c + 2]:
                continue
            block = a[r:r + 4, c:c + 4]
            if np.any(block[~template] != bg):
                continue
            points = _hub_points(r, c)
            values = a[points[:, 0], points[:, 1]]
            labels = tuple(map(int, values[::2]))
            if np.array_equal(values[::2], values[1::2]) and bg not in labels and len(set(labels)) == 4:
                hubs.append((r, c, labels))
    return hubs


def _labels_for(cycle, north):
    i = cycle.index(north)
    return cycle[i:] + cycle[:i]


def _dock(mask, side, r, c):
    edge = np.zeros_like(mask)
    edge[-1, mask.shape[1] // 2 - 1:mask.shape[1] // 2 + 1] = True
    oriented = np.rot90(mask, -side)
    anchor = np.argwhere(np.rot90(edge, -side)).min(axis=0)
    targets = ((r - 1, c + 1), (r + 1, c + 4), (r + 4, c + 1), (r + 1, c - 1))
    return np.argwhere(oriented) + np.array(targets[side]) - anchor


def _overlap(rect_a, rect_b):
    return not (rect_a[2] <= rect_b[0] or rect_b[2] <= rect_a[0] or
                rect_a[3] <= rect_b[1] or rect_b[3] <= rect_a[1])


def _matches(plan, sequence, axis):
    h, w = plan.shape
    n = len(sequence)
    return [(r, c) for r in range(h - (n - 1 if axis == 0 else 0))
            for c in range(w - (n - 1 if axis == 1 else 0))
            if tuple(int(plan[r + (axis == 0) * k, c + (axis == 1) * k])
                     for k in range(n)) == sequence]


def _inspect(input_grid):
    a = _grid(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(counts[bg] > a.size // 2, "background must occupy over half the input")
    hubs = _find_hubs(a, bg)
    _require(len(hubs) >= 2, "at least two bare seed hubs are required")
    cycle = hubs[0][2]
    _require(all(labels == _labels_for(cycle, labels[0]) for _, _, labels in hubs),
             "seed hubs disagree on their clockwise color cycle")
    if len({r for r, c, labels in hubs}) == 1:
        axis = 1
    else:
        _require(len({c for r, c, labels in hubs}) == 1, "seed hubs must form one straight run")
        axis = 0
    hubs.sort(key=lambda hub: hub[axis])
    differences = np.diff([hub[axis] for hub in hubs])
    pitch = int(differences[0])
    _require(pitch > 4 and np.all(differences == pitch), "seed hubs must have a common pitch")
    active = a != bg
    for r, c, labels in hubs:
        cells = _hub_points(r, c)
        active[cells[:, 0], cells[:, 1]] = False
    components = _components(a, active)
    swatches = [cells[0] for _, cells in components if len(cells) == 1]
    pieces = [(color, cells) for color, cells in components if len(cells) != 1]
    _require(len(swatches) >= 3, "miniature must have at least three occupied sites")
    points = np.array(swatches)
    mr, mc = map(int, points.min(axis=0))
    end_r, end_c = map(int, points.max(axis=0) + 1)
    _require(np.all((points - (mr, mc)) % 2 == 0), "miniature swatches must have pitch two")
    plan = a[mr:end_r:2, mc:end_c:2].copy()
    _require(max(plan.shape) <= 3 and np.count_nonzero(plan != bg) == len(swatches),
             "malformed miniature extent")
    mini_mask = np.zeros((end_r - mr, end_c - mc), dtype=bool)
    mini_mask[::2, ::2] = True
    _require(np.all(a[mr:end_r, mc:end_c][~mini_mask] == bg), "miniature separators must be empty")
    _require(set(map(int, plan.flat)) <= set(cycle) | {bg}, "map entries must name hub labels")
    sequence = tuple(hub[2][0] for hub in hubs)
    locations = _matches(plan, sequence, axis)
    _require(len(locations) == 1, "north-color sequence must have exactly one map location")
    match = locations[0]
    origin = (hubs[0][0] - match[0] * pitch, hubs[0][1] - match[1] * pitch)
    matching = {}
    unmatched = []
    for color, cells in pieces:
        _require(len(cells) >= 4, "inventory objects must contain at least four cells")
        lo, hi = cells.min(axis=0), cells.max(axis=0) + 1
        mask = np.zeros(tuple(hi - lo), dtype=bool)
        local = cells - lo
        mask[local[:, 0], local[:, 1]] = True
        item = dict(color=color, cells=cells, mask=mask)
        if color in cycle:
            _require(color not in matching, "only one prototype per matching color is allowed")
            canonical, turns = _canonical(mask)
            item.update(canonical=canonical, turns=turns)
            matching[color] = item
        else:
            unmatched.append(item)
    _require(matching, "at least one matching prototype is required")
    # Check object separation without consulting any construction metadata.
    groups = [set(map(tuple, cells)) for _, cells in pieces]
    groups += [set(map(tuple, _hub_points(r, c))) for r, c, _ in hubs]
    groups.append(set(map(tuple, points)))
    for i, group in enumerate(groups):
        halo = {(r + dr, c + dc) for r, c in group for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        _require(all(not (halo & other) for other in groups[i + 1:]), "input objects touch one another")
    for _, cells in pieces:
        _require(not any(mr - 1 <= r <= end_r and mc - 1 <= c <= end_c for r, c in cells),
                 "inventory must be outside the miniature rectangle and its clearance")
    radius = max(item['canonical'].shape[0] for item in matching.values())
    frame = 4 + 2 * radius
    _require(pitch >= frame, "assembled tiles require nonoverlapping bounding frames")
    box = (origin[0] - radius, origin[1] - radius,
           origin[0] + (plan.shape[0] - 1) * pitch + 4 + radius,
           origin[1] + (plan.shape[1] - 1) * pitch + 4 + radius)
    _require(0 <= box[0] < box[2] <= a.shape[0] and 0 <= box[1] < box[3] <= a.shape[1],
             "complete lattice frame must fit the canvas")
    _require(not _overlap(box, (mr, mc, end_r, end_c)), "complete lattice must avoid the miniature")
    seed_sites = {(match[0] + (axis == 0) * k, match[1] + (axis == 1) * k)
                  for k in range(len(hubs))}
    occupied = {(int(r), int(c)) for r, c in np.argwhere(plan != bg)}
    _require(occupied - seed_sites, "at least one unseeded assembly is required")
    expected = np.full_like(a, bg)
    expected[mr:end_r, mc:end_c] = a[mr:end_r, mc:end_c]
    directions, rotations = set(), set()
    for pr, pc in sorted(occupied):
        r, c = origin[0] + pr * pitch, origin[1] + pc * pitch
        labels = _labels_for(cycle, int(plan[pr, pc]))
        hub_cells = _hub_points(r, c)
        _require(np.all(expected[hub_cells[:, 0], hub_cells[:, 1]] == bg), "hub overlap")
        _paint_hub(expected, r, c, labels)
        for color, item in matching.items():
            side = labels.index(color)
            cells = _dock(item['canonical'], side, r, c)
            _require(np.all(expected[cells[:, 0], cells[:, 1]] == bg), "docked pieces overlap")
            expected[cells[:, 0], cells[:, 1]] = color
            directions.add(side)
            if len(item['turns']) == 1:
                rotations.add((item['turns'][0] - side) % 4)
    for r, c, _ in hubs:
        cells = _hub_points(r, c)
        _require(np.array_equal(expected[cells[:, 0], cells[:, 1]], a[cells[:, 0], cells[:, 1]]),
                 "seed hub bars must remain unchanged")
    expected_cells = len(swatches) + len(occupied) * (8 + sum(int(p['canonical'].sum()) for p in matching.values()))
    _require(np.count_nonzero(expected != bg) == expected_cells, "copy count or empty-site invariant failed")
    meta = dict(background=bg, cycle=cycle, plan=plan, miniature=(mr, mc),
                origin=origin, axis=axis, pitch=pitch, radius=radius, match=match,
                sequence=sequence, hubs=hubs, matching=matching, unmatched=unmatched,
                occupied=len(occupied), holes=int(np.count_nonzero(plan == bg)),
                directions=directions, rotations=rotations, box=box)
    return expected, meta


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Compute the entire answer from one input, with no latent puzzle state."""
    return _inspect(input_grid)[0]


def _sample_piece(rng, radius, depth=None, asymmetric=False):
    for _ in range(100):
        h = int(depth if depth is not None else rng.integers(2, radius + 1))
        w = int(rng.choice((2, 4, 6, 8)))
        kind = int(rng.integers(4))
        if kind == 0 and not asymmetric:
            w = 2
        mask = np.zeros((h, w), dtype=bool)
        mid = w // 2
        mask[:, mid - 1:mid + 1] = True
        body = int(rng.integers(1, h))
        if kind == 1:
            mask[:body] = True
        elif kind == 2 and w > 2:
            levels = w // 2 - 1
            if levels >= h:
                continue
            mask[:levels] = False
            for row in range(levels):
                reach = levels - row
                mask[row, mid - 1 - reach] = mask[row, mid + reach] = True
                if rng.random() < .25:
                    mask[row, mid - 1 - reach:mid + reach + 1] = True
        elif w > 2:
            for row in range(body):
                left = int(rng.integers(mid))
                right = int(rng.integers(mid + 1, w + 1))
                mask[row, left:right] = True
            mask[int(rng.integers(body)), :] = True
            for row, col in np.argwhere(mask[:body]):
                if col not in (mid - 1, mid) and rng.random() < .25:
                    mask[row, col] = False
        if not mask[:, 0].any() or not mask[:, -1].any() or not _connected(mask):
            continue
        options = _canonical_options(mask)
        if len(options) != 1:
            continue
        canonical, turns = options[0]
        if canonical.shape[0] != h or canonical.shape[1] != w:
            continue
        if asymmetric and (len(turns) != 1 or np.array_equal(canonical, np.fliplr(canonical))):
            continue
        return canonical
    # A supported connector with a reflection-asymmetric cap and variable stem.
    rows = ['#.##' if asymmetric else '####'] + ['.##.'] * (int(depth or radius) - 1)
    return _mask(*rows)


def _library(rng, labels, bg, radius, witness, forced_match=None, forced_distractor=None, small=False):
    n = int(rng.integers(2 if witness else 1, 5))
    matched = list(map(int, rng.choice(labels, n, replace=False)))
    if forced_match is not None and forced_match not in matched:
        matched[0] = forced_match
    items = []
    occupied = set(map(tuple, _hub_points(0, 0)))
    for i, color in enumerate(matched):
        for _ in range(100):
            mask = (_mask(*(['#.##' if witness == 1 and i == 0 else '####'] + ['.##.'] * (radius - 1)))
                    if small else _sample_piece(rng, radius, radius if i == 0 else None,
                                               asymmetric=witness == 1 and i == 0))
            cells = set(map(tuple, _dock(mask, labels.index(color), 0, 0)))
            if not cells & occupied:
                break
        else:
            mask = _mask(*(['#.##' if witness == 1 and i == 0 else '####'] + ['.##.'] * (radius - 1)))
            cells = set(map(tuple, _dock(mask, labels.index(color), 0, 0)))
        occupied.update(cells)
        items.append(dict(color=color, mask=mask, turn=int(rng.integers(4)), matching=True))
    pool = [c for c in range(10) if c not in labels and c != bg]
    nd = int(rng.integers(1 if witness else 0, 3))
    for i in range(nd):
        color = forced_distractor if i == 0 and forced_distractor is not None else int(rng.choice(pool))
        if small or i == 0 or rng.random() < .5:
            mask = _mask('####', '.##.') if small else _sample_piece(rng, radius)
        else:
            mask = _mask('.##', '###', '##.')
        items.append(dict(color=color, mask=mask, turn=int(rng.integers(4)), matching=False))
    return items


def _geometry_options(axis, mode, split, witness):
    result = []
    for h in range(1, 4):
        for w in range(1, 4):
            shape = (h, w)
            if shape[axis] < 2 or h * w < 3:
                continue
            if witness == 1 and shape != (2, 2):
                continue
            if witness == 2 and sorted(shape) != [2, 3]:
                continue
            if mode == 'fragment' and split == 'train' and shape[1 - axis] < 2:
                continue
            if split == 'test' and ((mode == 'repeat' and h * w < 6) or
                                    (mode == 'fragment' and shape[axis] != 3)):
                continue
            for radius in range(2, 5):
                if witness and radius != witness + 1:
                    continue
                frame = 4 + 2 * radius
                for pitch in range(frame, frame + 4):
                    box = (frame + (h - 1) * pitch, frame + (w - 1) * pitch)
                    mini = (2 * h - 1, 2 * w - 1)
                    for sep in (0, 1):
                        required = [max(box[d], mini[d]) for d in (0, 1)]
                        required[sep] = box[sep] + mini[sep] + 1
                        if max(required) <= 30:
                            result.append((shape, radius, pitch, sep))
    return result


def _seed_choices(plan, bg, axis, mode, split):
    result = []
    extent = plan.shape[axis]
    for length in range(2, extent + 1):
        if mode == 'fragment' and length != (extent if split == 'train' else 2):
            continue
        for r in range(plan.shape[0] - (length - 1 if axis == 0 else 0)):
            for c in range(plan.shape[1] - (length - 1 if axis == 1 else 0)):
                sequence = tuple(int(plan[r + (axis == 0) * k, c + (axis == 1) * k]) for k in range(length))
                if bg in sequence or len(_matches(plan, sequence, axis)) != 1:
                    continue
                if np.count_nonzero(plan != bg) <= length:
                    continue
                if split == 'test' and mode == 'fragment':
                    line = plan[:, c] if axis == 0 else plan[r, :]
                    if bg in line:
                        continue
                result.append(((r, c), length))
    return result


def _plan(rng, shape, bg, labels, axis, mode, split, witness):
    for _ in range(300):
        if witness == 1:
            count = 4
        elif witness == 2:
            count = 5  # Four compass keys plus a repeated address and one hole.
        elif split == 'train' and mode == 'repeat':
            count = int(rng.integers(3, min(5, np.prod(shape)) + 1))
        else:
            low = 6 if split == 'test' and mode == 'repeat' else 4 if np.prod(shape) >= 4 else 3
            count = int(rng.integers(low, np.prod(shape) + 1))
        occupied = rng.choice(np.prod(shape), count, replace=False)
        plan = np.full(shape, bg, dtype=np.int8)
        values = rng.choice(labels, count)
        if witness:
            values[:4] = rng.permutation(labels)
            rng.shuffle(values)
        plan.flat[occupied] = values
        if any(np.all(edge == bg) for edge in (plan[0], plan[-1], plan[:, 0], plan[:, -1])):
            continue
        choices = _seed_choices(plan, bg, axis, mode, split)
        if choices:
            match, length = choices[int(rng.integers(len(choices)))]
            return plan, match, length
    return None


def _layout(rng, shape, radius, pitch, sep, large=False):
    frame = 4 + 2 * radius
    box = tuple(frame + (n - 1) * pitch for n in shape)
    mini = tuple(2 * n - 1 for n in shape)
    gap = int(rng.integers(1, min(4, 30 - box[sep] - mini[sep]) + 1))
    needed = [max(box[d], mini[d]) for d in (0, 1)]
    needed[sep] = box[sep] + mini[sep] + gap
    if large or rng.random() < .15:
        canvas = (30, 30)
    elif rng.random() < .25:
        n = int(rng.integers(max(needed), 31))
        canvas = (n, n)
    else:
        canvas = tuple(int(rng.integers(n, min(30, n + 5) + 1)) for n in needed)
    anchor, corner = [0, 0], [0, 0]
    start = int(rng.integers(canvas[sep] - needed[sep] + 1))
    if rng.integers(2):
        corner[sep] = start
        anchor[sep] = start + box[sep] + gap
    else:
        anchor[sep] = start
        corner[sep] = start + mini[sep] + gap
    other = 1 - sep
    corner[other] = int(rng.integers(canvas[other] - box[other] + 1))
    anchor[other] = int(rng.integers(canvas[other] - mini[other] + 1))
    return canvas, tuple(anchor), tuple(v + radius for v in corner)


def _bare_input(plan, bg, labels, axis, match, length, pitch, canvas, mini, origin):
    a = np.full(canvas, bg, dtype=np.int8)
    mr, mc = mini
    a[mr:mr + 2 * plan.shape[0]:2, mc:mc + 2 * plan.shape[1]:2] = plan
    for k in range(length):
        pr, pc = match[0] + (axis == 0) * k, match[1] + (axis == 1) * k
        _paint_hub(a, origin[0] + pr * pitch, origin[1] + pc * pitch,
                   _labels_for(labels, int(plan[pr, pc])))
    return a


def _block_around(blocked, cells):
    for r, c in cells:
        blocked[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True


def _scatter(rng, bare, bg, mini, plan_shape, items):
    for _ in range(12):
        a = bare.copy()
        blocked = np.zeros(a.shape, dtype=bool)
        _block_around(blocked, np.argwhere(a != bg))
        mr, mc = mini
        blocked[max(0, mr - 1):mr + 2 * plan_shape[0], max(0, mc - 1):mc + 2 * plan_shape[1]] = True
        order = list(rng.permutation(len(items)))
        order.sort(key=lambda i: -items[i]['mask'].size)
        success = True
        for i in order:
            item = items[i]
            mask = np.rot90(item['mask'], item['turn'])
            h, w = mask.shape
            local = np.argwhere(mask)
            tops, lefts = np.indices((a.shape[0] - h + 1, a.shape[1] - w + 1))
            rows = tops.ravel()[:, None] + local[:, 0]
            cols = lefts.ravel()[:, None] + local[:, 1]
            choices = np.flatnonzero(~np.any(blocked[rows, cols], axis=1))
            if not len(choices):
                success = False
                break
            choice = int(rng.choice(choices))
            a[rows[choice], cols[choice]] = item['color']
            _block_around(blocked, np.column_stack((rows[choice], cols[choice])))
        if success and np.count_nonzero(a == bg) > a.size // 2:
            return a
    return None


def _colors(rng, witness, reference):
    palette = list(map(int, rng.permutation(10)))
    if witness == 2:
        old_match = next(iter(reference['matching']))
        old_distractor = reference['unmatched'][0]['color']
        bg = next(c for c in palette if c not in (old_match, old_distractor, reference['background']))
        labels = (old_distractor,) + tuple(c for c in palette if c not in (bg, old_match, old_distractor))[:3]
        labels = tuple(map(int, rng.permutation(labels)))
        return bg, labels, old_distractor, old_match
    return palette[0], tuple(palette[1:5]), palette[1] if witness == 1 else None, None


def _fallback(rng, labels, bg, axis, mode, split, witness, force_match, force_distractor):
    """Constructive 30x30 packing; six small pieces fit a five-cell strip."""
    radius = 3 if witness == 2 else 2
    a, b, c, d = labels
    if witness == 1:
        plan = np.array([[a, b], [c, d]], dtype=np.int8)
    elif split == 'test' and mode == 'repeat':
        plan = np.array([[b, c, d], [a, a, a]], dtype=np.int8)
    elif witness == 2:
        plan = np.array([[a, b, c], [d, a, bg]], dtype=np.int8)
    else:
        plan = np.array([[a, b, c], [d, bg, bg]], dtype=np.int8)
    if axis == 0:
        plan = plan.T.copy()
    pitch = 4 + 2 * radius
    length = plan.shape[axis] if mode == 'fragment' and split == 'train' else 2
    mini = (0, 0)
    origin = (5 + radius, radius) if axis == 1 else (radius, 5 + radius)
    grid = _bare_input(plan, bg, labels, axis, (0, 0), length, pitch, (30, 30), mini, origin)
    items = _library(rng, labels, bg, radius, witness, force_match, force_distractor, small=True)
    for i, item in enumerate(items):
        mask = np.rot90(item['mask'], item['turn'])
        r, c = (26, 5 * i) if axis == 1 else (5 * i, 26)
        cells = np.argwhere(mask) + (r, c)
        grid[cells[:, 0], cells[:, 1]] = item['color']
    return dict(input=grid, output=solve(grid))


def _make_pair(rng, axis, mode, split, witness=0, reference=None):
    bg, labels, force_match, force_distractor = _colors(rng, witness, reference)
    options = _geometry_options(axis, mode, split, witness)
    for attempt in range(64):
        shape, radius, pitch, sep = options[int(rng.integers(len(options)))]
        sampled = _plan(rng, shape, bg, labels, axis, mode, split, witness)
        if sampled is None:
            continue
        plan, match, length = sampled
        items = _library(rng, labels, bg, radius, witness, force_match, force_distractor)
        canvas, mini, origin = _layout(rng, shape, radius, pitch, sep, attempt >= 8)
        bare = _bare_input(plan, bg, labels, axis, match, length, pitch, canvas, mini, origin)
        grid = _scatter(rng, bare, bg, mini, plan.shape, items)
        if grid is not None:
            return dict(input=grid, output=solve(grid))
    return _fallback(rng, labels, bg, axis, mode, split, witness, force_match, force_distractor)


def _count(value, name, lower, upper):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f'{name} must lie in {lower}..{upper}')
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    nt = _count(num_train, 'num_train', 2, 10)
    nq = _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must be greater than num_test')
    rng = np.random.default_rng(seed)
    mode = str(rng.choice(('repeat', 'fragment')))
    axes = ((0,), (1,), (0, 1))[int(rng.integers(3))]
    train_axes = [axes[i % len(axes)] for i in range(nt)]
    rng.shuffle(train_axes)
    result, seen = {'train': [], 'test': []}, set()
    reference = None
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            witness = i + 1 if split == 'train' and i < 2 else 0
            axis = train_axes[i] if split == 'train' else int(rng.choice(axes))
            for _ in range(32):
                pair = _make_pair(rng, axis, mode, split, witness, reference)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('distinct-input construction exhausted')
            seen.add(key)
            result[split].append(pair)
            if witness == 1:
                reference = _inspect(pair['input'])[1]
        rng.shuffle(result[split])
    validate_generated_puzzle(result)
    return result


def validate_generated_puzzle(puzzle):
    """Validate pixel-derived assembly, alignment, teaching, and split invariants."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "split type")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, metas = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape and _key(inp) not in seen, "shape or duplicate input")
            seen.add(_key(inp))
            expected, meta = _inspect(inp)
            _require(np.array_equal(expected, out), "incorrect answer or changed miniature")
            _require(not np.array_equal(inp, out), "identity pair")
            # Omitting A produces bare hubs; omitting B cannot create missing hubs.
            _require(sum(int(p['canonical'].sum()) for p in meta['matching'].values()) > 0, "A has no effect")
            _require(meta['occupied'] > len(meta['hubs']), "B has no effect")
            metas[split].append(meta)
    train, test = metas['train'], metas['test']
    _require({m['axis'] for m in test} <= {m['axis'] for m in train}, "undemonstrated clue axis")
    _require(any(len(set(map(int, m['plan'].flat)) - {m['background']}) == 4 and
                 any(len(p['turns']) == 1 and not np.array_equal(p['canonical'], np.fliplr(p['canonical']))
                     for p in m['matching'].values()) for m in train), "orientation and asymmetric-shape witness")
    _require(set().union(*(m['directions'] for m in train)) == set(range(4)), "docking directions")
    _require(set().union(*(m['rotations'] for m in train)) == set(range(4)), "quarter-turn witnesses")
    _require(any(m['holes'] for m in train), "empty-site witness")
    _require(any(len(set(map(int, m['plan'].flat)) - {m['background']}) < m['occupied'] for m in train),
             "repeated orientation-address witness")
    _require(any(any(len(_canonical_options(p['mask'])) == 1 for p in m['unmatched']) for m in train),
             "legitimate-shaped distractor witness")
    _require(len({m['background'] for m in train}) >= 2 and len({m['radius'] for m in train}) >= 2,
             "background and geometry witnesses")
    _require(any(p['color'] in n['matching'] for m in train for p in m['unmatched'] for n in train if n is not m),
             "locally changing color-role witness")
    repeat_shift = (all(m['occupied'] <= 5 for m in train) and all(m['occupied'] >= 6 for m in test))
    fragment_shift = (all(len(m['sequence']) == m['plan'].shape[m['axis']] for m in train) and
                      all(len(m['sequence']) < m['plan'].shape[m['axis']] for m in test))
    _require(repeat_shift or fragment_shift, "missing intended train/test shift")


def _signature(puzzle):
    return tuple((_key(pair['input']), _key(pair['output']))
                 for split in ('train', 'test') for pair in puzzle[split])


def self_test():
    """Bounded 42-seed sweep, diversity, metamorphic checks, and negative tests."""
    puzzles, records = [], []
    for seed in range(42):
        nt = 2 + seed % 9
        nq = 1 + seed % min(4, nt - 1)
        puzzle = generate_puzzle(seed, nt, nq)
        puzzles.append(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                expected, meta = _inspect(pair['input'])
                assert np.array_equal(expected, pair['output'])
                records.append((split, pair, meta))
    assert len({_signature(p) for p in puzzles}) == 42
    assert _signature(generate_puzzle(9137, 6, 3)) == _signature(generate_puzzle(9137, 6, 3))
    assert {m['axis'] for _, _, m in records} == {0, 1}
    assert {m['background'] for _, _, m in records} == set(range(10))
    assert {m['radius'] for _, _, m in records} >= {2, 3, 4}
    assert len({pair['input'].shape for _, pair, _ in records}) >= 30
    assert len({m['plan'].shape for _, _, m in records}) >= 6
    assert len({m['pitch'] for _, _, m in records}) >= 5
    assert len({m['match'] for _, _, m in records}) >= 6
    assert {len(m['matching']) for _, _, m in records} == {1, 2, 3, 4}
    assert {len(m['unmatched']) for _, _, m in records} == {0, 1, 2}
    assert {len(m['sequence']) for _, _, m in records} == {2, 3}
    masks = [p['canonical'] for _, _, m in records for p in m['matching'].values()]
    assert len({_key(mask) for mask in masks}) >= 35
    assert {mask.shape[1] for mask in masks} == {2, 4, 6, 8}
    assert any(not _connected(mask, False) for mask in masks)
    assert any(len(_canonical_options(mask)[0][1]) > 1 for mask in masks)
    assert any(m['occupied'] >= 7 and m['holes'] >= 1 for _, _, m in records)
    # Colors are roles visible in the input: every injective recoloring commutes.
    sample = puzzles[0]['train'][0]
    mapping = np.array([9, 2, 5, 0, 8, 1, 7, 4, 6, 3], dtype=np.int8)
    assert np.array_equal(solve(mapping[sample['input']]), mapping[sample['output']])
    # Swapping two existing prototype colors changes all their destinations.
    swapped = False
    for _, candidate, meta in records:
        prototypes = list(meta['matching'].values())
        for first in prototypes:
            for second in prototypes:
                if _key(first['canonical']) == _key(second['canonical']):
                    continue
                changed = candidate['input'].copy()
                for item, color in ((first, second['color']), (second, first['color'])):
                    cells = item['cells']
                    changed[cells[:, 0], cells[:, 1]] = color
                try:
                    revised = solve(changed)
                except ValueError:
                    continue  # A swapped wide shoulder can violate tile clearance.
                assert not np.array_equal(revised, candidate['output'])
                swapped = True
                break
            if swapped:
                break
        if swapped:
            break
    assert swapped
    # Change an unseeded map orientation while retaining a unique clue match.
    edited = False
    for _, pair, meta in records:
        plan, bg, axis = meta['plan'], meta['background'], meta['axis']
        seed_sites = {(meta['match'][0] + (axis == 0) * k, meta['match'][1] + (axis == 1) * k)
                      for k in range(len(meta['sequence']))}
        for r, c in np.argwhere(plan != bg):
            if (r, c) in seed_sites:
                continue
            revised = plan.copy()
            revised[r, c] = _labels_for(meta['cycle'], int(plan[r, c]))[1]
            if len(_matches(revised, meta['sequence'], axis)) != 1:
                continue
            changed = pair['input'].copy()
            changed[meta['miniature'][0] + 2 * r, meta['miniature'][1] + 2 * c] = revised[r, c]
            assert not np.array_equal(solve(changed), pair['output'])
            edited = True
            break
        if edited:
            break
    assert edited
    # Deliberately exercise the constructive fallback in every sampling regime.
    for mode in ('repeat', 'fragment'):
        for axis in (0, 1):
            rng = np.random.default_rng(100)
            ref_pair = _fallback(rng, (1, 2, 3, 4), 0, axis, mode, 'train', 1, 1, 5)
            ref = _inspect(ref_pair['input'])[1]
            bg, labels, match_color, distractor = _colors(rng, 2, ref)
            second = _fallback(rng, labels, bg, axis, mode, 'train', 2, match_color, distractor)
            query = _fallback(rng, (1, 2, 3, 4), 0, axis, mode, 'test', 0, None, None)
            validate_generated_puzzle(dict(train=[ref_pair, second], test=[query]))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(5000 + 10 * a + b, a, b)
            assert len(puzzle['train']) == a and len(puzzle['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    damaged = {split: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][split]]
               for split in ('train', 'test')}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    # A source-supported symmetric connector and diagonal wing survive parsing.
    assert len(_canonical(_mask('##', '##'))[1]) == 4
    wing = _mask('#..#', '.##.')
    assert _connected(wing) and not _connected(wing, False)
    assert len(_canonical_options(_mask('.###', '####', '#.##', '.##.'))) > 1
