"""Room-conditioned exchange of doorway-paired motif templates.

Only NumPy and the standard library are used. The reader works entirely from
the visible input; the sampler separately renders construction-derived labels.
"""
from collections import Counter
import numpy as np


_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    parts = []
    steps = _STEPS
    if diagonal:
        steps += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, part = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    part.append(p)
                    stack.append(p)
        parts.append(np.array(part, dtype=int))
    return parts


def _bbox(points):
    p = np.asarray(list(points) if isinstance(points, set) else points)
    return tuple(map(int, p.min(0))) + tuple(map(int, p.max(0) + 1))


def _signature(mapping):
    return tuple(sorted((int(r), int(c), int(v)) for (r, c), v in mapping.items()))


def _rooms(mask):
    padded = np.pad(~mask, 1, constant_values=True)
    return [p - 1 for p in _components(padded)
            if np.all(p > 0) and np.all(p < np.array(padded.shape) - 1)]


def _relations(rooms):
    boxes = [_bbox(p) for p in rooms]
    return [[tuple(-1 if a[k + 2] <= b[k] else
                   1 if b[k + 2] <= a[k] else 0 for k in (0, 1))
             for b in boxes] for a in boxes]


def _door_links(crop, framework, rooms, door):
    labels = np.full(crop.shape, -1, dtype=int)
    for i, room in enumerate(rooms):
        labels[tuple(room.T)] = i
    buddy = {}
    for r, c in np.argwhere(crop == door):
        neighbors = [(int(r + dr), int(c + dc)) for dr, dc in _STEPS]
        _require(all(0 <= y < crop.shape[0] and 0 <= x < crop.shape[1]
                     for y, x in neighbors), 'door on exterior boundary')
        adjacent = [int(labels[p]) for p in neighbors if labels[p] >= 0]
        _require(len(adjacent) == 2 and adjacent[0] != adjacent[1],
                 'door must join two separate chambers')
        _require(sum(bool(framework[p]) for p in neighbors) == 2,
                 'door must lie inside a straight shared wall')
        a, b = adjacent
        _require(a not in buddy and b not in buddy, 'room has multiple doors')
        buddy[a], buddy[b] = b, a
    _require(len(buddy) == len(rooms), 'every room needs exactly one partner')
    return buddy


def _find_frameworks(grid):
    border = np.concatenate((grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    _require(len(np.unique(border)) == 1, 'expected a uniform background border')
    bg = int(border[0])
    palette = set(map(int, np.unique(grid))) - {bg}
    candidates = []
    # A framework color has two connected enclosing outlines. Door colors are
    # sparse internal gaps in these outlines, not a fixed literal color.
    for wall in sorted(palette):
        outlines = _components(grid == wall)
        if len(outlines) != 2 or min(map(len, outlines)) < 12:
            continue
        outer_masks = []
        for outline in outlines:
            r, c, R, C = _bbox(outline)
            mask = np.zeros((R - r, C - c), dtype=bool)
            mask[outline[:, 0] - r, outline[:, 1] - c] = True
            enclosed = _rooms(mask)
            if not enclosed:
                break
            contained = mask.copy()
            for room in enclosed:
                contained[tuple(room.T)] = True
            outer_masks.append(((r, c, R, C), contained))
        if len(outer_masks) != 2:
            continue
        all_contained = np.zeros(grid.shape, dtype=bool)
        for (r, c, R, C), mask in outer_masks:
            all_contained[r:R, c:C] |= mask
        if np.any((grid != bg) & ~all_contained):
            continue
        for door in sorted(palette - {wall}):
            spots = np.argwhere(grid == door)
            if not 4 <= len(spots) <= 8:
                continue
            if any(sum(0 <= r + dr < grid.shape[0] and
                       0 <= c + dc < grid.shape[1] and
                       grid[r + dr, c + dc] == wall for dr, dc in _STEPS) != 2
                   for r, c in spots):
                continue
            try:
                figures = []
                for (r, c, R, C), _ in outer_masks:
                    crop = grid[r:R, c:C].copy()
                    mask = (crop == wall) | (crop == door)
                    rooms = _rooms(mask)
                    _require(len(rooms) in (4, 6, 8), 'unsupported room count')
                    buddy = _door_links(crop, mask, rooms, door)
                    interior = mask.copy()
                    for room in rooms:
                        interior[tuple(room.T)] = True
                    _require(np.all(crop[~interior] == bg), 'material outside framework')
                    figures.append({'crop': crop, 'mask': mask, 'rooms': rooms,
                                    'buddy': buddy, 'box': (r, c, R, C)})
                _require(len(figures[0]['rooms']) == len(figures[1]['rooms']),
                         'figures have different room counts')
                candidates.append((bg, wall, door, figures))
            except ValueError:
                continue
    _require(len(candidates) == 1, 'framework roles must be uniquely visible')
    bg, wall, door, figures = candidates[0]
    donors = [f for f in figures if all(np.all(f['crop'][tuple(p.T)] != bg)
                                      for p in f['rooms'])]
    targets = [f for f in figures if all(np.any(f['crop'][tuple(p.T)] == bg)
                                       for p in f['rooms'])]
    _require(len(donors) == len(targets) == 1, 'expected filled reference and blank workpiece')
    return bg, wall, door, donors[0], targets[0]


def _correspondence(donor, target):
    a, b = _relations(donor['rooms']), _relations(target['rooms'])
    n = len(a)
    matches = []

    def search(mapping, used):
        if len(mapping) == n:
            matches.append(dict(mapping))
            return
        options = {}
        for i in range(n):
            if i in mapping:
                continue
            options[i] = [j for j in range(n) if j not in used and all(
                a[i][k] == b[j][v] and
                (donor['buddy'][i] == k) == (target['buddy'][j] == v)
                for k, v in mapping.items())]
        i = min(options, key=lambda k: (len(options[k]), k))
        for j in options[i]:
            mapping[i] = j
            search(mapping, used | {j})
            del mapping[i]

    search({}, set())
    _require(len(matches) == 1, 'ambiguous orientation-preserving room correspondence')
    return matches[0]


def _read_reference(donor, bg, wall, door):
    payloads, tokens = [], []
    for room in donor['rooms']:
        values = donor['crop'][tuple(room.T)]
        color, count = Counter(map(int, values)).most_common(1)[0]
        _require(count * 2 > len(room), 'reference room fill must be a strict majority')
        payloads.append(color)
        tokens.append({tuple(map(int, p)): int(donor['crop'][tuple(p)])
                       for p in room if donor['crop'][tuple(p)] != color})
    _require(len(set(payloads)) >= 2, 'at least two payload colors required')
    common = set.intersection(*(set(t.values()) for t in tokens))
    _require(len(common) == 1, 'a single shared carrier color must be visible')
    carrier = next(iter(common))
    ink = set.union(*(set(t.values()) for t in tokens))
    _require(not (ink & set(payloads)) and not (ink & {bg, wall, door}),
             'motif ink and payload roles must be distinct')
    _require(len(ink) == 3, 'one carrier and two state inks required')
    templates = []
    for token in tokens:
        points = [p for p, v in token.items() if v == carrier]
        _require(3 <= len(points) <= 6, 'invalid unit scaffold')
        r, c, _, _ = _bbox(points)
        templates.append({(y - r, x - c): v for (y, x), v in token.items()})
    for i, j in donor['buddy'].items():
        ca = {p for p, v in templates[i].items() if v == carrier}
        cb = {p for p, v in templates[j].items() if v == carrier}
        _require(ca == cb, 'paired states need the same registered scaffold')
        _require(payloads[i] != payloads[j], 'paired payloads must differ')
        sa, sb = set(templates[i].values()) - {carrier}, set(templates[j].values()) - {carrier}
        _require(len(sa) == len(sb) == 1 and sa != sb, 'two distinct state inks required')
    return payloads, templates, carrier


def _query_occurrence(target, room_index, templates, carrier, bg):
    room = target['rooms'][room_index]
    token = {tuple(map(int, p)): int(target['crop'][tuple(p)])
             for p in room if target['crop'][tuple(p)] != bg}
    points = [p for p, v in token.items() if v == carrier]
    _require(bool(points), 'missing target scaffold')
    r, c, R, C = _bbox(points)
    base = [p for p, v in templates[0].items() if v == carrier]
    _, _, h, w = _bbox(base)
    _require((R - r) % h == (C - c) % w == 0, 'noninteger scale')
    scale = (R - r) // h
    _require(scale == (C - c) // w and 1 <= scale <= 3, 'nonuniform or unsupported scale')
    matches = []
    for state, template in enumerate(templates):
        enlarged = {(r + y * scale + dy, c + x * scale + dx): v
                    for (y, x), v in template.items()
                    for dy in range(scale) for dx in range(scale)}
        if enlarged == token:
            matches.append(state)
    _require(len(matches) == 1, 'target is not one of its local states')
    allowed = set(map(tuple, room))
    union = set(templates[0]) | set(templates[1])
    _require(all((r + y * scale + dy, c + x * scale + dx) in allowed
                 for y, x in union for dy in range(scale) for dx in range(scale)),
             'opposite state crosses a room boundary')
    return matches[0], (r, c), scale


def _analyze(input_grid):
    grid = np.asarray(input_grid)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape), 'invalid grid shape')
    _require(np.issubdtype(grid.dtype, np.integer) and np.all((grid >= 0) & (grid <= 9)),
             'invalid grid colors')
    bg, wall, door, donor, target = _find_frameworks(grid)
    mapping = _correspondence(donor, target)
    payloads, templates, carrier = _read_reference(donor, bg, wall, door)
    answer = target['crop'].copy()
    occurrences = []
    for i, j in mapping.items():
        buddy = donor['buddy'][i]
        local = [templates[i], templates[buddy]]
        state, anchor, scale = _query_occurrence(target, j, local, carrier, bg)
        source = i if state == 0 else buddy
        opposite = donor['buddy'][source]
        room = target['rooms'][j]
        answer[tuple(room.T)] = payloads[opposite]
        _draw(answer, templates[opposite], anchor, scale)
        occurrences.append({'room': j, 'context': min(i, buddy), 'source': source,
                            'opposite': opposite, 'anchor': anchor, 'scale': scale,
                            'template': templates[source], 'replacement': templates[opposite],
                            'fill': payloads[opposite]})
    return {'expected': answer, 'background': bg, 'wall': wall, 'door': door,
            'carrier': carrier, 'donor': donor, 'target': target, 'mapping': mapping,
            'payloads': payloads, 'templates': templates, 'occurrences': occurrences}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer all local dictionaries, room correspondences, anchors and scales."""
    return _analyze(input_grid)['expected'].copy()


def _draw(grid, template, anchor, scale=1):
    for (r, c), color in template.items():
        y, x = anchor[0] + r * scale, anchor[1] + c * scale
        grid[y:y + scale, x:x + scale] = color


def _pick(rng, items):
    items = sorted(items)
    return items[int(rng.integers(len(items)))]


def _has_hole(mask):
    return bool(_rooms(mask))


def _floorplan(rng, groups, fallback=False):
    if fallback:
        plan = np.empty((2, groups), dtype=int)
        for g in range(groups):
            plan[:, g] = [2 * g, 2 * g + 1]
        return plan
    options = [(2, groups), (groups, 2), (3, 3)]
    if groups == 2:
        options += [(1, 4), (4, 1), (2, 3), (3, 2)]
    h, w = options[int(rng.integers(len(options)))]
    plan = np.full((h, w), -1, dtype=int)
    occupied = set()
    for g in range(groups):
        choices = []
        for r in range(h):
            for c in range(w):
                for dr, dc in ((1, 0), (0, 1)):
                    a, b = (r, c), (r + dr, c + dc)
                    if b[0] >= h or b[1] >= w or a in occupied or b in occupied:
                        continue
                    if occupied and not any((y + dy, x + dx) in occupied
                                            for y, x in (a, b) for dy, dx in _STEPS):
                        continue
                    trial = plan >= 0
                    trial[a] = trial[b] = True
                    if not _has_hole(trial):
                        choices.append((a, b))
        _require(bool(choices), 'domino growth exhausted')
        a, b = choices[int(rng.integers(len(choices)))]
        if rng.integers(2):
            a, b = b, a
        plan[a], plan[b] = 2 * g, 2 * g + 1
        occupied.update((a, b))
    r, c, R, C = _bbox(occupied)
    return plan[r:R, c:C]


def _design(rng, fallback=False):
    if fallback:
        carrier = {(0, 0), (0, 1), (1, 0)}
        inner = {(1, 1)}
        a, b, alternative = {(-1, 0)}, {(0, 2)}, {(2, 0)}
        return carrier, inner | a, inner | b, inner | alternative
    for _ in range(80):
        h, w = int(rng.integers(2, 4)), int(rng.integers(2, 4))
        core = {(int(rng.integers(h)), int(rng.integers(w)))}
        count = int(rng.integers(4, min(7, h * w + 1)))
        while len(core) < count:
            frontier = {(r + dr, c + dc) for r, c in core
                        for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                        if 0 <= r + dr < h and 0 <= c + dc < w} - core
            core.add(_pick(rng, frontier))
        inner = {_pick(rng, core)} if rng.random() < .8 else set()
        carrier = core - inner
        cr, cc, _, _ = _bbox(carrier)
        carrier = {(r - cr, c - cc) for r, c in carrier}
        inner = {(r - cr, c - cc) for r, c in inner}
        occupied = carrier | inner
        frontier = {(r + dr, c + dc) for r, c in occupied
                    for dr in (-1, 0, 1) for dc in (-1, 0, 1)} - occupied
        a = {_pick(rng, frontier)}
        choices = []
        for p in sorted(frontier - a):
            union = occupied | a | {p}
            r, c, R, C = _bbox(union)
            if R - r <= 4 and C - c <= 4:
                choices.append(p)
        if len(choices) >= 2:
            order = rng.permutation(len(choices))
            b, alternative = {choices[int(order[0])]}, {choices[int(order[1])]}
            return carrier, inner | a, inner | b, inner | alternative
    return _design(rng, True)


def _templates(rng, groups, colors, fallback):
    carrier_color, ink0, ink1 = colors
    carrier, a, b, alt = _design(rng, fallback)
    designs = [(carrier, a, b), (carrier, a, alt)]
    for _ in range(groups - 2):
        scaffold, u, v, _ = _design(rng, fallback)
        if not fallback and rng.random() < .3:
            v = u  # The complete exchange also covers pure recoloring.
        designs.append((scaffold, u, v))
    result = []
    for scaffold, u, v in designs:
        states = []
        for support, ink in ((u, ink0), (v, ink1)):
            t = {p: carrier_color for p in scaffold}
            t.update({p: ink for p in support})
            states.append(t)
        result.append(states)
    return result


def _geometry(rng, plan, heights, widths, templates, scales, fallback):
    ys = np.r_[0, np.cumsum(heights + 1)]
    xs = np.r_[0, np.cumsum(widths + 1)]
    labels = np.full((int(ys[-1]) + 1, int(xs[-1]) + 1), -1, dtype=int)
    for r, c in np.argwhere(plan >= 0):
        i = int(plan[r, c])
        region = (slice(ys[r] + 1, ys[r + 1]), slice(xs[c] + 1, xs[c + 1]))
        labels[region] = i
        saved = labels.copy()
        if not fallback:
            # Independent local starts can reverse room discovery order while
            # retaining the same before/after/overlap relation graph. Keep one
            # extreme room per coarse row/column uninset for a tight figure.
            if c != min(np.flatnonzero(plan[r] >= 0)) and rng.random() < .4:
                labels[ys[r] + 1, xs[c] + 1:xs[c + 1]] = -1
            if r != min(np.flatnonzero(plan[:, c] >= 0)) and rng.random() < .3:
                labels[ys[r] + 1:ys[r + 1], xs[c] + 1] = -1
            corners = [(ys[r] + 1, xs[c] + 1), (ys[r] + 1, xs[c + 1] - 1),
                       (ys[r + 1] - 1, xs[c] + 1), (ys[r + 1] - 1, xs[c + 1] - 1)]
            for p in corners:
                if rng.random() < .27:
                    labels[p] = -1
        union = set(templates[i // 2][0]) | set(templates[i // 2][1])
        try:
            _positions(labels == i, union, int(scales[i]))
            _require(np.count_nonzero(labels == i) >
                     2 * max(map(len, templates[i // 2])), 'room too small for visible fill')
        except ValueError:
            labels = saved
    occupied = labels >= 0
    near = np.zeros(labels.shape, dtype=bool)
    padded = np.pad(occupied, 1)
    for dr in range(3):
        for dc in range(3):
            near |= padded[dr:dr + labels.shape[0], dc:dc + labels.shape[1]]
    wall = near & ~occupied
    doors = []
    for g in range(len(templates)):
        points = []
        for r, c in np.argwhere(wall):
            adjacent = [int(labels[r + dr, c + dc]) for dr, dc in _STEPS
                        if 0 <= r + dr < labels.shape[0] and 0 <= c + dc < labels.shape[1]]
            if sorted(v for v in adjacent if v >= 0) == [2 * g, 2 * g + 1] and adjacent.count(-1) == 2:
                points.append((int(r), int(c)))
        _require(bool(points), 'no readable shared doorway')
        doors.append(_pick(rng, points))
    return labels, wall, doors


def _positions(room_mask, support, scale):
    r, c, R, C = _bbox(support)
    h, w = (R - r) * scale, (C - c) * scale
    points = np.argwhere(room_mask)
    _require(len(points) > 0, 'template union cannot fit chamber')
    top, left, bottom, right = _bbox(points)
    local = room_mask[top:bottom, left:right]
    _require(h <= local.shape[0] and w <= local.shape[1],
             'template union cannot fit chamber')
    # All-true rectangles are found in one reduction, restricted to this
    # chamber's bounds. np.argwhere preserves the old row-major candidate
    # order, so this optimization neither changes RNG use nor seeded grids.
    windows = np.lib.stride_tricks.sliding_window_view(local, (h, w))
    fits = np.argwhere(np.all(windows, axis=(-2, -1)))
    positions = [(int(y) + top - r * scale, int(x) + left - c * scale)
                 for y, x in fits]
    _require(bool(positions), 'template union cannot fit chamber')
    return positions


def _render_figure(rng, geometry, templates, fills, states, scales, roles, target=False):
    labels, wall_mask, doors = geometry
    bg, wall, door = roles
    grid = np.full(labels.shape, bg, dtype=np.int8)
    grid[wall_mask] = wall
    for p in doors:
        grid[p] = door
    output = grid.copy()
    for i, state in enumerate(states):
        g, scale = i // 2, int(scales[i])
        union = set(templates[g][0]) | set(templates[g][1])
        anchor = _pick(rng, _positions(labels == i, union, scale))
        if not target:
            grid[labels == i] = fills[g][state]
        _draw(grid, templates[g][state], anchor, scale)
        if target:
            output[labels == i] = fills[g][1 - state]
            _draw(output, templates[g][1 - state], anchor, scale)
    return grid, output


def _embed(rng, donor, target, bg, fallback=False):
    dh, dw = donor.shape
    th, tw = target.shape
    options = []
    for gap in (1, 2, 3):
        if max(dh, th) + 2 <= 30 and dw + tw + gap + 2 <= 30:
            options.append(('side', gap))
        if dh + th + gap + 2 <= 30 and max(dw, tw) + 2 <= 30:
            options.append(('stack', gap))
    _require(bool(options), 'frameworks do not fit a 30 by 30 canvas')
    mode, gap = options[0] if fallback else options[int(rng.integers(len(options)))]
    h = max(dh, th) if mode == 'side' else dh + th + gap
    w = dw + tw + gap if mode == 'side' else max(dw, tw)
    h += int(rng.integers(2, min(30 - h, 6) + 1))
    w += int(rng.integers(2, min(30 - w, 6) + 1))
    grid = np.full((h, w), bg, dtype=np.int8)
    if mode == 'side':
        r = int(rng.integers(1, h - dh))
        c = int(rng.integers(1, w - dw - tw - gap))
        y = int(rng.integers(1, h - th))
        x = c + dw + gap
    else:
        r = int(rng.integers(1, h - dh - th - gap))
        c = int(rng.integers(1, w - dw))
        y = r + dh + gap
        x = int(rng.integers(1, w - tw))
    if rng.integers(2):
        # Exchange the two positions and corresponding rectangle sizes by
        # reflecting the whole placement, before the final orientation choice.
        if mode == 'side':
            c, x = w - c - dw, w - x - tw
        else:
            r, y = h - r - dh, h - y - th
    grid[r:r + dh, c:c + dw] = donor
    grid[y:y + th, x:x + tw] = target
    return grid


def _make_pair(rng, groups, scale_max, assignment):
    for attempt in range(49):
        fallback = attempt == 48
        try:
            palette = list(map(int, rng.permutation(10)))
            bg, wall, door, carrier, ink0, ink1 = palette[:6]
            fill_pool = palette[6:6 + int(rng.integers(2, 5))]
            templates = _templates(rng, groups, (carrier, ink0, ink1), fallback)
            fills = [(fill_pool[1], fill_pool[0]), (fill_pool[0], fill_pool[1])]
            for _ in range(groups - 2):
                order = rng.permutation(len(fill_pool))
                fills.append((fill_pool[int(order[0])], fill_pool[int(order[1])]))
            plan = _floorplan(rng, groups, fallback)
            states = [0, 1] * groups
            query_states = [0, (1 if assignment == 'both' else 0)] * 2
            query_states += list(map(int, rng.integers(0, 2, 2 * (groups - 2))))
            scales = np.ones(2 * groups, dtype=int)
            scales[0] = scale_max
            if not fallback and scale_max == 2 and rng.random() < .3:
                scales[int(rng.integers(1, 2 * groups))] = 2
            dh = np.full(plan.shape[0], 4, dtype=int)
            dw = np.full(plan.shape[1], 4, dtype=int)
            th, tw = dh.copy(), dw.copy()
            for r, c in np.argwhere(plan >= 0):
                i = int(plan[r, c])
                union = set(templates[i // 2][0]) | set(templates[i // 2][1])
                y, x, Y, X = _bbox(union)
                dh[r], dw[c] = max(dh[r], Y - y), max(dw[c], X - x)
                th[r] = max(th[r], (Y - y) * scales[i])
                tw[c] = max(tw[c], (X - x) * scales[i])
            if not fallback:
                dh += rng.integers(0, 3, len(dh))
                dw += rng.integers(0, 3, len(dw))
                th += rng.integers(0, 3, len(th))
                tw += rng.integers(0, 3, len(tw))
            reference_geometry = _geometry(rng, plan, dh, dw, templates,
                                           np.ones_like(scales), fallback)
            target_geometry = _geometry(rng, plan, th, tw, templates, scales, fallback)
            roles = bg, wall, door
            donor, _ = _render_figure(rng, reference_geometry, templates, fills,
                                      states, np.ones_like(scales), roles)
            target, out = _render_figure(rng, target_geometry, templates, fills,
                                         query_states, scales, roles, target=True)
            inp = _embed(rng, donor, target, bg, fallback)
            rotation, reflected = int(rng.integers(4)), bool(rng.integers(2))
            inp, out = np.rot90(inp, rotation), np.rot90(out, rotation)
            if reflected:
                inp, out = np.fliplr(inp), np.fliplr(out)
            pair = {'input': inp.copy(), 'output': out.copy()}
            _require(np.array_equal(solve(pair['input']), pair['output']),
                     'input-derived solution disagrees with construction')
            return pair
        except ValueError:
            if fallback:
                raise
    raise RuntimeError('bounded construction exhausted')


def _grid_key(grid):
    grid = np.asarray(grid)
    return grid.shape, grid.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, name, low, high in ((num_train, 'num_train', 2, 10),
                                   (num_test, 'num_test', 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    curriculum = int(rng.integers(2))  # more local contexts or larger pixel scale
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(count):
            if split == 'train':
                groups = (2, 3)[index] if index < 2 else int(rng.integers(2, 4))
                scale = 2 if index == 0 else int(rng.integers(1, 3))
                assignment = 'both' if index == 0 else 'duplicate' if index == 1 else _pick(rng, ('both', 'duplicate'))
            else:
                groups = 4 if curriculum == 0 else 2
                scale = int(rng.integers(1, 3)) if curriculum == 0 else 3
                assignment = _pick(rng, ('both', 'duplicate'))
            for _ in range(24):
                pair = _make_pair(rng, groups, scale, assignment)
                key = _grid_key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct input construction exhausted')
    return puzzle


def measure_pair(pair):
    inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
    info = _analyze(inp)
    _require(np.array_equal(info['expected'], out), 'incorrect answer')
    target, donor = info['target'], info['donor']
    _require(np.array_equal(out[target['mask']], target['crop'][target['mask']]),
             'framework or doorway changed')
    _require(np.array_equal(out == info['wall'], target['crop'] == info['wall']),
             'wall support changed')
    interior = np.zeros(out.shape, dtype=bool)
    for room in target['rooms']:
        interior[tuple(room.T)] = True
    _require(np.all(out[~interior & ~target['mask']] == info['background']),
             'exterior background changed')
    _require(np.count_nonzero(out[interior] == info['background']) == 0,
             'target has an unfilled room')
    occurrences = info['occurrences']
    alias = any(a['context'] != b['context'] and
                _signature(a['template']) == _signature(b['template']) and
                a['fill'] != b['fill'] and
                _signature(a['replacement']) != _signature(b['replacement'])
                for a in occurrences for b in occurrences)
    # A-only ablation copies the geometrically corresponding room payload.
    a_only_fails = any(o['fill'] != info['payloads'][i]
                       for i, j in info['mapping'].items()
                       for o in occurrences if o['room'] == j)
    # B-only ablation has one global dictionary for a repeated current motif;
    # the alias witness proves no such dictionary gives all answers.
    _require(alias and a_only_fails, 'missing parent-ablation witnesses')
    same_buddy_state = any(a['source'] == b['source'] and
                           target['buddy'][a['room']] == b['room']
                           for a in occurrences for b in occurrences)
    directions, change_types, erased, created = set(), set(), 0, 0
    for o in occurrences:
        a, b = o['template'], o['replacement']
        recolor = any(a[p] != b[p] for p in set(a) & set(b))
        relocate = set(a) != set(b)
        change_types.add('combined' if recolor and relocate else 'relocate' if relocate else 'recolor')
        erased += len(set(a) - set(b)) * o['scale'] ** 2
        created += len(set(b) - set(a)) * o['scale'] ** 2
        directions.add(next(v for v in a.values() if v != info['carrier']))
        r, c = o['anchor']
        for (y, x), color in a.items():
            if color == info['carrier']:
                block = out[r + y * o['scale']:r + (y + 1) * o['scale'],
                            c + x * o['scale']:c + (x + 1) * o['scale']]
                _require(np.all(block == color), 'scaffold moved or scale changed')
        for y, x in set(a) - set(b):
            block = out[r + y * o['scale']:r + (y + 1) * o['scale'],
                        c + x * o['scale']:c + (x + 1) * o['scale']]
            _require(np.all(block == o['fill']), 'old attachment not restored to room fill')
    area_signs = {int(np.sign(len(target['rooms'][j]) - len(donor['rooms'][i])))
                  for i, j in info['mapping'].items()}
    nonrect = sum(len(room) != (_bbox(room)[2] - _bbox(room)[0]) *
                  (_bbox(room)[3] - _bbox(room)[1])
                  for f in (donor, target) for room in f['rooms'])
    wall_structure = (target['mask'].shape, target['mask'].tobytes(),
                      (target['crop'] == info['door']).tobytes())
    graph = _relations(target['rooms'])
    topology = (tuple(tuple(row) for row in graph),
                tuple(sorted(target['buddy'].items())))
    # Measure template geometry without counting palette permutations.
    color_roles = {info['carrier']: 0}
    vocabulary = []
    for t in info['templates']:
        vocabulary.append(tuple((r, c, color_roles.setdefault(v, len(color_roles)))
                                for r, c, v in _signature(t)))
    return {'rooms': len(target['rooms']), 'max_scale': max(o['scale'] for o in occurrences),
            'scales': tuple(sorted(o['scale'] for o in occurrences)),
            'palette_size': len(np.unique(inp)), 'roles': (info['background'], info['wall'], info['door']),
            'alias': alias, 'a_only_fails': a_only_fails, 'same_buddy_state': same_buddy_state,
            'directions': len(directions), 'area_signs': tuple(sorted(area_signs)),
            'nonrect': nonrect, 'erased': erased, 'created': created,
            'changes': tuple(sorted(change_types)), 'structure': wall_structure,
            'scan_order_changed': any(i != j for i, j in info['mapping'].items()),
            'topology': topology, 'vocabulary': tuple(vocabulary),
            'input_shape': inp.shape, 'output_shape': out.shape}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, 'invalid example counts')
    seen, measures = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            for value in pair.values():
                grid = np.asarray(value)
                _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape), 'grid dimensions out of bounds')
                _require(np.issubdtype(grid.dtype, np.integer) and np.all((grid >= 0) & (grid <= 9)), 'invalid grid values')
            key = _grid_key(pair['input'])
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            measures[split].append(measure_pair(pair))
    train = measures['train']
    _require(train[0]['directions'] == 2 and train[0]['max_scale'] == 2,
             'first demonstration must exchange both states and show enlargement')
    _require(train[1]['same_buddy_state'], 'second demonstration must defeat fixed destination copying')
    _require(len({m['rooms'] for m in train}) >= 2, 'training needs different floorplan sizes')
    _require(all(m['alias'] and m['a_only_fails'] for m in train), 'insufficient concept evidence')
    if all(m['rooms'] == 8 for m in measures['test']):
        _require(max(m['rooms'] for m in train) <= 6, 'context shift is absent')
    else:
        _require(all(m['rooms'] == 4 and m['max_scale'] == 3 for m in measures['test']),
                 'scale shift is absent')


def self_test():
    """Bounded semantic checks and pixel-derived variation over 42 seeds."""
    records, shapes, structures, vocabularies, roles, fingerprints = [], set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprint = tuple(_grid_key(p[k]) for split in ('train', 'test')
                            for p in puzzle[split] for k in ('input', 'output'))
        fingerprints.add(fingerprint)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                record = measure_pair(pair)
                records.append(record)
                shapes.add(record['input_shape'])
                structures.add(record['structure'])
                vocabularies.add(record['vocabulary'])
                roles.add(record['roles'])
                # Solver must neither mutate the input nor depend on aliasing.
                grid = np.array(pair['input'], copy=True)
                before = grid.copy()
                _require(np.array_equal(solve(grid), pair['output']) and np.array_equal(grid, before),
                         'solver depends on mutation')
    _require(len(fingerprints) == 42 and len(shapes) >= 30 and len(structures) >= 80,
             'insufficient floorplan or canvas variation')
    _require(len(vocabularies) >= 80 and len(roles) >= 35, 'insufficient local vocabulary or palette variation')
    _require({m['rooms'] for m in records} == {4, 6, 8}, 'missing room-count variation')
    _require({m['max_scale'] for m in records} == {1, 2, 3}, 'missing scale variation')
    _require(sum(m['nonrect'] > 0 for m in records) >= 30, 'missing notched chambers')
    _require(sum(-1 in m['area_signs'] and 1 in m['area_signs'] for m in records) >= 15,
             'missing mixed room shrinkage and growth')
    _require(sum(m['scan_order_changed'] for m in records) >= 10,
             'room correspondence is reducible to scan order')
    _require(len({m['topology'] for m in records}) >= 15, 'insufficient doorway topology variation')
    _require(all(m['erased'] and m['created'] for m in records), 'missing support-changing exchanges')
    # Determinism and every legal explicit count combination.
    a, b = generate_puzzle(901), generate_puzzle(901)
    _require(all(_grid_key(p[k]) == _grid_key(q[k]) for s in a
                 for p, q in zip(a[s], b[s]) for k in ('input', 'output')), 'nondeterministic seed')
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + 10 * nt + ns, nt, ns)
            validate_generated_puzzle(p)
            _require(len(p['train']) == nt and len(p['test']) == ns, 'count request ignored')
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(902, nt, ns)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    # Three independently meaningful corruptions: wall, uniform fill, scaffold.
    sample = generate_puzzle(903)
    info = _analyze(sample['test'][0]['input'])
    wall_point = tuple(np.argwhere(info['target']['crop'] == info['wall'])[0])
    carrier_point = tuple(np.argwhere(info['expected'] == info['carrier'])[0])
    fill_point = tuple(np.argwhere(np.isin(info['expected'], info['payloads']))[0])
    for point in (wall_point, carrier_point, fill_point):
        bad = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in sample[s]]
               for s in sample}
        grid = bad['test'][0]['output']
        grid[point] = (int(grid[point]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            continue
        raise AssertionError('corrupted answer accepted')
    # A broken doorway and an absent current state are invalid visible inputs.
    for color in (info['door'], info['carrier']):
        grid = sample['test'][0]['input'].copy()
        point = tuple(np.argwhere(grid == color)[-1])
        grid[point] = info['background']
        try:
            solve(grid)
        except ValueError:
            continue
        raise AssertionError('damaged input accepted')
    return {'seeds_checked': 42, 'pairs_checked': len(records),
            'input_shapes': len(shapes), 'target_structures': len(structures),
            'local_vocabularies': len(vocabularies), 'role_assignments': len(roles)}
