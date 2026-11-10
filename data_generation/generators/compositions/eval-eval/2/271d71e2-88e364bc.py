"""Fuel-driven couriers: input-decoded gauges and clockwise wall deflection.

Only NumPy and the standard library are used. No parent code is imported.
"""
from __future__ import annotations

import json
import numpy as np

# Clockwise in image coordinates.
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _need(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _need(np.all((0 <= a) & (a <= 9)), "invalid ARC color")
    return a.astype(np.int8, copy=False)


def _signature(a):
    return a.shape, a.tobytes()


def _components(mask, diagonal=False):
    seen = ~mask.copy()
    result = []
    h, w = mask.shape
    neighbors = (tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                       if dy or dx) if diagonal else DIRECTIONS)
    for rr, cc in np.argwhere(mask):
        p = int(rr), int(cc)
        if seen[p]:
            continue
        stack, cells = [p], set()
        seen[p] = True
        while stack:
            r, c = stack.pop()
            cells.add((r, c))
            for dr, dc in neighbors:
                q = r + dr, c + dc
                if 0 <= q[0] < h and 0 <= q[1] < w and not seen[q]:
                    seen[q] = True
                    stack.append(q)
        result.append(cells)
    return result


def _bbox(cells):
    rows, cols = zip(*cells)
    return int(min(rows)), int(max(rows)), int(min(cols)), int(max(cols))


def _order(box, direction):
    t, b, l, r = box
    if direction == (-1, 0):
        return [(y, x) for y in range(t + 1, b) for x in range(l + 1, r)]
    if direction == (1, 0):
        return [(y, x) for y in range(b - 1, t, -1) for x in range(r - 1, l, -1)]
    if direction == (0, -1):
        return [(y, x) for x in range(l + 1, r) for y in range(b - 1, t, -1)]
    if direction == (0, 1):
        return [(y, x) for x in range(r - 1, l, -1) for y in range(t + 1, b)]
    raise AssertionError("noncardinal heading")


def _decode_key(sub, carrier):
    other = set(map(int, np.unique(sub))) - {carrier}
    if len(other) != 2 or not np.any(sub[1:-1, 1:-1] == carrier):
        return []
    answers = []
    for spent in sorted(other):
        fuel = next(iter(other - {spent}))
        tips = set(map(tuple, np.argwhere(sub == spent)))
        trails = set(map(tuple, np.argwhere(sub == fuel)))
        if len(tips) < 2:
            continue
        for dr, dc in DIRECTIONS:
            used, lengths = set(), []
            for r, c in sorted(tips):
                stroke = []
                q = r - dr, c - dc
                while q in trails:
                    stroke.append(q)
                    q = q[0] - dr, q[1] - dc
                if len(stroke) < 2 or used.intersection(stroke):
                    break
                used.update(stroke)
                lengths.append(len(stroke))
            if len(lengths) == len(tips) and used == trails:
                answers.append(dict(spent=spent, fuel=fuel, direction=(dr, dc),
                                    lengths=tuple(lengths), strokes=len(tips)))
    return answers


def _controls(a, bg):
    rectangles = []
    for cells in _components(a != bg):
        t, b, l, r = _bbox(cells)
        h, w = b - t + 1, r - l + 1
        if min(h, w) < 3 or len(cells) != h * w:
            continue
        sub = a[t:b + 1, l:r + 1]
        carrier = int(sub[0, 0])
        border = np.concatenate((sub[0], sub[-1], sub[:, 0], sub[:, -1]))
        if np.all(border == carrier):
            rectangles.append(dict(box=(t, b, l, r), carrier=carrier, sub=sub))
    keys = []
    for rect in rectangles:
        decoded = _decode_key(rect['sub'], rect['carrier'])
        _need(len(decoded) <= 1, "ambiguous direction glyph")
        if decoded:
            keys.append(dict(rect, **decoded[0]))
    _need(keys and len({k['carrier'] for k in keys}) == len(keys), "missing or duplicate keys")
    _need(len({(k['spent'], k['fuel']) for k in keys}) == 1, "inconsistent state alphabet")
    spent, fuel = keys[0]['spent'], keys[0]['fuel']
    _need(len({bg, spent, fuel} | {k['carrier'] for k in keys}) == 3 + len(keys),
          "conflicting carrier roles")
    controls = np.zeros(a.shape, bool)
    for key in keys:
        candidates = []
        for rect in rectangles:
            if rect['carrier'] != key['carrier']:
                continue
            interior = rect['sub'][1:-1, 1:-1]
            if set(map(int, np.unique(interior))) <= {spent, fuel}:
                candidates.append(rect)
        _need(len(candidates) == 1, "each key needs exactly one matching gauge")
        gauge = candidates[0]
        order = _order(gauge['box'], key['direction'])
        values = [int(a[p]) for p in order]
        advanced = values.count(spent)
        _need(values == [spent] * advanced + [fuel] * (len(values) - advanced),
              "gauge is not an initial-heading-relative prefix")
        key.update(gauge=gauge['box'], order=order, advanced=advanced,
                   capacity=len(values), budget=len(values) - advanced)
        for box in (key['box'], gauge['box']):
            t, b, l, r = box
            _need(not np.any(controls[t:b + 1, l:r + 1]), "overlapping controls")
            controls[t:b + 1, l:r + 1] = True
    _need(len(rectangles) == 2 * len(keys), "unexplained rectangular controls")
    return keys, controls, spent, fuel


def _region_explanation(a, bg, keys, controls, marker, gate):
    carriers = {k['carrier'] for k in keys}
    rest = set(map(int, np.unique(a[~controls])))
    _need(rest <= carriers | {bg, marker, gate}, "unexplained outside color")
    _need(np.count_nonzero((a == marker) & ~controls) == len(keys), "wrong courier count")
    _need(np.count_nonzero((a == gate) & ~controls) == len(keys), "wrong gate count")
    free_mask = ((a == bg) | (a == marker)) & ~controls
    by_carrier = {}
    for cells in _components(free_mask):
        starts = [p for p in cells if a[p] == marker]
        if not starts:
            continue
        _need(len(starts) == 1, "multiple couriers in one region")
        _need(not any(y in (0, a.shape[0] - 1) or x in (0, a.shape[1] - 1)
                      for y, x in cells), "courier region is open")
        boundary, gate_cells = set(), set()
        for r, c in cells:
            _need(any((r + dr, c + dc) in cells for dr, dc in DIRECTIONS), "isolated free cell")
            for dr, dc in DIRECTIONS:
                q = r + dr, c + dc
                if q not in cells:
                    boundary.add(int(a[q]))
                    if a[q] == gate:
                        gate_cells.add(q)
        carrier_options = boundary & carriers
        _need(len(carrier_options) == 1 and boundary == carrier_options | {gate},
              "region has ambiguous carrier or lacks a gate")
        carrier = next(iter(carrier_options))
        _need(len(gate_cells) == 1 and carrier not in by_carrier, "ambiguous gate assignment")
        start, gate_cell = starts[0], next(iter(gate_cells))
        _need(all((start[0] + dr, start[1] + dc) in cells for dr, dc in DIRECTIONS),
              'initial courier needs a free cell on all four sides')
        _need(any(int(a[gate_cell[0] + dr, gate_cell[1] + dc]) == carrier
                  and not controls[gate_cell[0] + dr, gate_cell[1] + dc]
                  for dr, dc in DIRECTIONS), 'gate must attach to its carrier wall')
        by_carrier[carrier] = dict(free=cells, start=start, gate_cell=gate_cell)
    _need(set(by_carrier) == carriers, "not every key has one courier region")
    _need(len({v['gate_cell'] for v in by_carrier.values()}) == len(keys), "shared gate")
    systems = [dict(k, **by_carrier[k['carrier']]) for k in keys]
    for s in systems:
        wall = (a == s['carrier']) & ~controls
        wall[s['gate_cell']] = True
        boundary_cells = {(y + dr, x + dc) for y, x in s['free'] for dr, dc in DIRECTIONS
                          if (y + dr, x + dc) not in s['free']}
        _need(all(c.intersection(boundary_cells) for c in _components(wall, diagonal=True)),
              'unassociated wall component')
    return systems


def _read(a):
    a = _grid(a)
    bg = int(a[0, 0])
    _need(np.all(a[0] == bg) and np.all(a[-1] == bg)
          and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg), "missing background border")
    keys, controls, spent, fuel = _controls(a, bg)
    candidates = sorted(set(map(int, np.unique(a[~controls])))
                        - {bg, spent, fuel} - {k['carrier'] for k in keys})
    _need(len(candidates) == 2, "courier and gate roles missing")
    explanations = []
    for marker, gate in (candidates, candidates[::-1]):
        try:
            systems = _region_explanation(a, bg, keys, controls, marker, gate)
            explanations.append(dict(bg=bg, spent=spent, fuel=fuel, marker=marker,
                                     gate=gate, controls=controls, systems=systems))
        except AssertionError:
            pass
    _need(len(explanations) == 1, "input lacks a unique complete role explanation")
    return explanations[0]


def _walk(system, budget=None, *, reverse=False, clockwise=True, first_wall=False):
    """Budget None means walk to the gate; repeated full states imply unreachable."""
    heading = DIRECTIONS.index(system['direction'])
    if reverse:
        heading = (heading + 2) % 4
    p = system['start']
    path, headings, seen = [p], [], set()
    turns = 0
    while budget is None or len(path) - 1 < budget:
        state = p, heading
        if budget is None:
            _need(state not in seen, "unreachable gate")
            seen.add(state)
        dr, dc = DIRECTIONS[heading]
        q = p[0] + dr, p[1] + dc
        if q == system['gate_cell']:
            return dict(path=path, turns=turns, cause='gate', headings=headings)
        if q in system['free']:
            p = q
            path.append(p)
            headings.append(heading)
        else:
            if first_wall:
                return dict(path=path, turns=turns, cause='wall', headings=headings)
            heading = (heading + (1 if clockwise else -1)) % 4
            turns += 1
            # Every valid free cell has a legal cardinal neighbor.
            _need(turns <= 4 * (len(path) + 1), "turning without movement")
    return dict(path=path, turns=turns, cause='fuel', headings=headings)


def _render_answer(a, info):
    out = a.copy()
    for s in info['systems']:
        trip = _walk(s, s['budget'])
        moves = len(trip['path']) - 1
        out[s['start']] = info['bg']
        out[trip['path'][-1]] = info['marker']
        for i, p in enumerate(s['order']):
            out[p] = info['spent'] if i < s['advanced'] + moves else info['fuel']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """All conventions are fixed by the family or visibly encoded in this input."""
    a = _grid(input_grid)
    return _render_answer(a, _read(a))


def _key(rng, direction, carrier, fuel, spent, compact=False):
    dr, dc = direction
    count = 2 if compact else int(rng.choice((2, 2, 3)))
    length = 2 if compact else int(rng.integers(2, 5))
    points = []
    for i in range(count):
        offset = 0 if compact else int(rng.integers(0, 3))
        for j in range(length + 1):
            y, x = ((j + offset) * dr, i * 2) if dr else (i * 2, (j + offset) * dc)
            points.append((y, x, spent if j == length else fuel))
    ymin, ymax = min(p[0] for p in points), max(p[0] for p in points)
    xmin, xmax = min(p[1] for p in points), max(p[1] for p in points)
    pad = [1] * 4 if compact else list(map(int, rng.choice((1, 1, 2), 4)))
    result = np.full((ymax - ymin + 1 + pad[0] + pad[1],
                      xmax - xmin + 1 + pad[2] + pad[3]), carrier, np.int8)
    for y, x, color in points:
        result[y - ymin + pad[0], x - xmin + pad[2]] = color
    return result


def _walls(rng, h, w, complexity, compact=False):
    footprint = np.ones((h, w), bool)
    kind = 0 if compact else int(rng.choice(4, p=(.25, .35, .20, .20)))
    if kind == 1:
        for corner in rng.choice(4, int(rng.integers(1, 4)), replace=False):
            dy = int(rng.integers(1, max(2, h // 3 + 1)))
            dx = int(rng.integers(1, max(2, w // 3 + 1)))
            ys = slice(0, dy) if corner < 2 else slice(h - dy, h)
            xs = slice(0, dx) if corner % 2 == 0 else slice(w - dx, w)
            footprint[ys, xs] = False
    elif kind == 2:
        y, x = np.indices((h, w))
        cy, cx = (h - 1) / 2, (w - 1) / 2
        exponent = float(rng.uniform(1.3, 2.4))
        footprint = (abs((y - cy) / (cy + .4)) ** exponent
                     + abs((x - cx) / (cx + .4)) ** exponent <= 1.22)
    elif kind == 3:
        left = right = 0
        for y in range(h):
            left = int(np.clip(left + rng.integers(-1, 2), 0, max(0, w // 3 - 1)))
            right = int(np.clip(right + rng.integers(-1, 2), 0, max(0, w // 3 - 1)))
            footprint[y, :left] = False
            if right:
                footprint[y, -right:] = False
    padded = np.pad(footprint, 1)
    interior = footprint.copy()
    for dr, dc in DIRECTIONS:
        interior &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    wall = footprint & ~interior
    for _ in range(0 if compact else int(rng.integers(complexity + 1))):
        horizontal = bool(rng.integers(2))
        if horizontal:
            row = int(rng.integers(2, h - 2))
            span = np.flatnonzero(footprint[row])
            if len(span) > 3:
                cut = int(rng.integers(int(span[0]) + 2, int(span[-1])))
                cols = slice(int(span[0]), cut) if rng.random() < .5 else slice(cut, int(span[-1]) + 1)
                wall[row, cols] |= footprint[row, cols]
        else:
            col = int(rng.integers(2, w - 2))
            span = np.flatnonzero(footprint[:, col])
            if len(span) > 3:
                cut = int(rng.integers(int(span[0]) + 2, int(span[-1])))
                rows = slice(int(span[0]), cut) if rng.random() < .5 else slice(cut, int(span[-1]) + 1)
                wall[rows, col] |= footprint[rows, col]
    return wall, footprint


def _route_events(free, start, direction, max_moves=120):
    """First contacts of distinct wall cells, before a receiving gate is chosen."""
    p, heading = start, DIRECTIONS.index(direction)
    seen, walls, events = set(), set(), []
    move_turns, moves, turns = [], 0, 0
    while moves < max_moves:
        state = p, heading
        if state in seen:
            break
        seen.add(state)
        dr, dc = DIRECTIONS[heading]
        q = p[0] + dr, p[1] + dc
        if q in free:
            p = q
            moves += 1
            move_turns.append(turns)
        else:
            if q not in walls:
                events.append((moves, turns, q))
                walls.add(q)
            heading = (heading + 1) % 4
            turns += 1
    return events, move_turns


def _gauge(rng, direction, carrier, fuel, spent, budget, distance, teaching, compact):
    max_side = 5 if compact else 8
    limit = 25 if compact else 56
    choices = []
    for depth in range(1, max_side + 1):
        for cross in range(1, max_side + 1):
            capacity = depth * cross
            if not budget <= capacity <= min(limit, budget + 22):
                continue
            if teaching and (min(depth, cross) < 2 or capacity <= budget):
                continue
            advanced = capacity - budget
            if teaching:
                if not (advanced > 0 and (advanced + distance) // cross > advanced // cross
                        and (advanced + distance) % cross != 0):
                    continue
            choices.append((depth, cross))
    if not choices:
        return None
    # Narrow counters and squares coexist with general aspect ratios.
    selected = choices
    if not teaching and rng.random() < .28:
        thin = [x for x in choices if min(x[:2]) == 1]
        if thin:
            selected = thin
    elif rng.random() < .20:
        squares = [x for x in choices if x[0] == x[1]]
        if squares:
            selected = squares
    depth, cross = selected[int(rng.integers(len(selected)))]
    capacity = depth * cross
    # Capacity minus budget is the initial spent prefix, so fuel is exact.
    # Unused capacity cannot be arbitrary when budget has already been chosen.
    advanced = capacity - budget
    h, w = (depth, cross) if direction[0] else (cross, depth)
    gauge = np.full((h + 2, w + 2), carrier, np.int8)
    for i, p in enumerate(_order((0, h + 1, 0, w + 1), direction)):
        gauge[p] = spent if i < advanced else fuel
    return gauge


def _make_system(rng, roles, carrier, direction, regime, required_turns,
                 teaching=False, count=1, test=False, compact=False):
    for _ in range(80):
        h = 7 if compact else int(rng.integers(7, 18 if count == 1 else 12))
        w = 7 if compact else int(rng.integers(7, 19 if count == 1 else 12))
        wall, footprint = _walls(rng, h, w, 3 if test else 2, compact)
        regions = [c for c in _components(footprint & ~wall) if len(c) >= 8
                   and all(any((y + dr, x + dc) in c for dr, dc in DIRECTIONS) for y, x in c)]
        if not regions:
            continue
        region = regions[int(rng.integers(len(regions)))]
        starts = sorted(p for p in region if all((p[0] + dr, p[1] + dc) in region
                                                for dr, dc in DIRECTIONS))
        for raw in rng.permutation(len(starts))[:min(32, len(starts))]:
            start = starts[int(raw)]
            events, move_turns = _route_events(region, start, direction)
            max_distance = 16 if compact else (29 if count == 1 else 20)
            eligible = []
            for distance, turns, gate in events:
                if not (2 <= distance <= max_distance and turns >= required_turns):
                    continue
                if not (0 <= gate[0] < h and 0 <= gate[1] < w and wall[gate]):
                    continue
                if not any(0 <= gate[0] + dr < h and 0 <= gate[1] + dc < w
                           and wall[gate[0] + dr, gate[1] + dc] for dr, dc in DIRECTIONS):
                    continue
                budgets = [b for b in range(2, distance) if move_turns[b - 1] >= required_turns]
                if regime == 'fuel' and not budgets:
                    continue
                eligible.append((distance, turns, gate, budgets))
            if not eligible:
                continue
            distance, _, gate, budgets = eligible[int(rng.integers(len(eligible)))]
            if regime == 'fuel':
                budget = int(rng.choice(budgets))
            elif regime == 'gate':
                budget = distance + int(rng.integers(1, 6))
            elif regime == 'exact':
                budget = distance
            elif regime == 'zero':
                budget = 0
            else:
                raise AssertionError('unknown resource regime')
            gauge = _gauge(rng, direction, carrier, roles['fuel'], roles['spent'],
                           budget, distance, teaching, compact)
            if gauge is None:
                continue
            model = dict(start=start, free=region, gate_cell=gate, direction=direction)
            trip = _walk(model, budget)
            if required_turns and trip['turns'] < required_turns:
                continue
            if not test and trip['turns'] > 2:
                continue
            if not test and required_turns == 2 and regime == 'fuel':
                end = trip['path'][-1]
                displacement = abs(end[0] - start[0]) + abs(end[1] - start[1])
                if len(trip['path']) - 1 <= displacement:
                    continue
            if regime == 'fuel' and trip['path'][-1] == _walk(model)['path'][-1]:
                continue  # Make the resource ablation visibly change position.
            if teaching:
                alt = _walk(model, budget, clockwise=False)
                rev = _walk(model, budget, reverse=True)
                first = _walk(model, budget, first_wall=True)
                if any(x['path'][-1] == trip['path'][-1] for x in (alt, rev, first)):
                    continue
                box = (0, gauge.shape[0] - 1, 0, gauge.shape[1] - 1)
                moves = len(trip['path']) - 1
                consumed = [p for p in _order(box, direction) if gauge[p] == roles['fuel']][:moves]
                final_heading = DIRECTIONS[(DIRECTIONS.index(direction) + trip['turns']) % 4]
                alternatives = [[p for p in _order(box, d) if gauge[p] == roles['fuel']][:moves]
                                for d in ((-1, 0), final_heading)]
                if any(set(other) == set(consumed) for other in alternatives):
                    continue
            enclosure = np.full((h, w), roles['bg'], np.int8)
            enclosure[wall] = carrier
            enclosure[gate] = roles['gate']
            enclosure[start] = roles['marker']
            key = _key(rng, direction, carrier, roles['fuel'], roles['spent'], compact or count == 2)
            return [enclosure, key, gauge]
    raise AssertionError('bounded system construction failed')


def _positions(occupied, h, w):
    H, W = occupied.shape
    if h > H or w > W:
        return np.empty((0, 2), int)
    p = np.pad(occupied, 1)
    expanded = np.zeros_like(occupied)
    for dy in range(3):
        for dx in range(3):
            expanded |= p[dy:dy + H, dx:dx + W]
    sums = np.pad(expanded.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    windows = sums[h:, w:] - sums[:-h, w:] - sums[h:, :-w] + sums[:-h, :-w]
    return np.argwhere(windows == 0)


def _pack(rng, parts, bg, compact=False):
    if compact:
        # Guaranteed fitting tail: each system occupies one variably ordered row.
        rows = []
        for index in rng.permutation(len(parts)):
            pieces = [parts[int(index)][int(j)] for j in rng.permutation(3)]
            height = max(x.shape[0] for x in pieces)
            width = sum(x.shape[1] for x in pieces) + 2
            if height > 28 or width > 28:
                return None
            row = np.full((height, width), bg, np.int8)
            x = 0
            for piece in pieces:
                y = int(rng.integers(height - piece.shape[0] + 1))
                row[y:y + piece.shape[0], x:x + piece.shape[1]] = piece
                x += piece.shape[1] + 1
            rows.append(row)
        H, W = sum(x.shape[0] for x in rows) + len(rows) - 1, max(x.shape[1] for x in rows)
        if max(H, W) > 28:
            return None
        result = np.full((H, W), bg, np.int8)
        y = 0
        for row in rows:
            x = int(rng.integers(W - row.shape[1] + 1))
            result[y:y + row.shape[0], x:x + row.shape[1]] = row
            y += row.shape[0] + 1
    else:
        pieces = [x for group in parts for x in group]
        order = list(map(int, rng.permutation(len(pieces))))
        order.sort(key=lambda i: -pieces[i].size)
        low = 20 if len(parts) == 1 else 26
        H, W = int(rng.integers(low, 29)), int(rng.integers(low, 29))
        result = np.full((H, W), bg, np.int8)
        occupied = np.zeros((H, W), bool)
        for index in order:
            piece = pieces[index]
            h, w = piece.shape
            positions = _positions(occupied, h, w)
            if not len(positions):
                return None
            y, x = map(int, positions[int(rng.integers(len(positions)))])
            result[y:y + h, x:x + w] = piece
            occupied[y:y + h, x:x + w] = True
        if rng.random() < .85:
            ys, xs = np.nonzero(result != bg)
            t, b = max(0, int(ys.min()) - int(rng.integers(3))), min(H, int(ys.max()) + 1 + int(rng.integers(3)))
            l, r = max(0, int(xs.min()) - int(rng.integers(3))), min(W, int(xs.max()) + 1 + int(rng.integers(3)))
            result = result[t:b, l:r]
    return np.pad(result, 1, constant_values=bg)


def _make_pair(rng, params, direction, regime, *, lesson=False, test=False):
    count = int(rng.choice((1, 2), p=(.72, .28)))
    reserved = {params[k] for k in ('spent', 'fuel', 'marker', 'gate')}
    palette = [c for c in range(10) if c not in reserved]
    if params['shared']:
        bg = params['bg']
        carriers = params['carriers'][:count]
    else:
        colors = list(map(int, rng.permutation(palette)))
        bg, carriers = colors[0], colors[1:1 + count]
    roles = dict(params, bg=bg)
    required = 3 if test else (2 if lesson and regime == 'fuel' else 1 if lesson else 0)
    for attempt in range(25):
        compact = attempt >= 20
        try:
            parts = [_make_system(rng, roles, carriers[0], direction, regime, required,
                                  teaching=lesson and regime == 'gate', count=count,
                                  test=test, compact=compact)]
            if count == 2:
                extra_direction = params['directions'][int(rng.integers(len(params['directions'])))]
                extra_regime = str(rng.choice(('fuel', 'gate', 'exact', 'zero'), p=(.30, .35, .25, .10)))
                parts.append(_make_system(rng, roles, carriers[1], extra_direction, extra_regime, 0,
                                          count=count, test=test, compact=compact))
            a = _pack(rng, parts, bg, compact)
            if a is None:
                continue
            info = _read(a)
            for s in info['systems']:
                _walk(s)  # Reachability is a family invariant.
            return dict(input=a, output=_render_answer(a, info))
        except AssertionError:
            continue
    raise AssertionError('bounded scene construction failed')


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    nt, nq = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    colors = list(map(int, rng.permutation(10)))
    direction_order = [DIRECTIONS[int(i)] for i in rng.permutation(4)]
    params = dict(spent=colors[0], fuel=colors[1], marker=colors[2], gate=colors[3],
                  bg=colors[4], carriers=colors[5:7], shared=bool(rng.integers(2)),
                  directions=direction_order[:int(rng.integers(2, 5))])
    lesson_slots = list(map(int, rng.permutation(nt)))[:2]
    gate_direction = next(d for d in direction_order if d != (-1, 0))
    fuel_direction = next(d for d in direction_order if d != gate_direction)
    params['directions'] = list(dict.fromkeys([gate_direction, fuel_direction] + params['directions']))
    puzzle, seen = {'train': [], 'test': []}, set()
    train_directions = []
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            lesson = split == 'train' and i in lesson_slots
            if lesson:
                regime = 'fuel' if i == lesson_slots[0] else 'gate'
                direction = fuel_direction if regime == 'fuel' else gate_direction
            else:
                pool = params['directions'] if split == 'train' else train_directions
                direction = pool[int(rng.integers(len(pool)))]
                regime = str(rng.choice(('fuel', 'gate', 'exact'), p=(.45, .40, .15)))
            pair = None
            for _ in range(8):
                scene_params = params if split == 'train' else dict(params, directions=train_directions)
                candidate = _make_pair(rng, scene_params, direction, regime, lesson=lesson, test=split == 'test')
                sig = _signature(candidate['input'])
                if sig not in seen:
                    seen.add(sig)
                    pair = candidate
                    break
            _need(pair is not None, 'duplicate input construction')
            puzzle[split].append(pair)
            if split == 'train':
                train_directions = list(dict.fromkeys(train_directions +
                    [s['direction'] for s in _read(pair['input'])['systems']
                     if len(_walk(s, s['budget'])['path']) > 1]))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle schema')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'illegal pair counts')
    seen, alphabets = set(), set()
    fuel_witness = gate_witness = scan_witness = displacement_witness = False
    fixed_up_witness = final_heading_witness = False
    turn_witness = reverse_witness = first_wall_witness = False
    active_headings, test_headings = set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair schema')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _need(a.shape == b.shape, 'changed dimensions')
            sig = _signature(a)
            _need(sig not in seen, 'duplicate input')
            seen.add(sig)
            info = _read(a)
            alphabets.add(tuple(info[k] for k in ('spent', 'fuel', 'marker', 'gate')))
            _need(np.array_equal(_render_answer(a, info), b), 'incorrect output')
            _need(np.count_nonzero(a == info['marker']) == np.count_nonzero(b == info['marker']),
                  'courier conservation failed')
            editable = np.zeros(a.shape, bool)
            test_turns = []
            for s in info['systems']:
                full = _walk(s)
                trip = _walk(s, s['budget'])
                steps, distance = len(trip['path']) - 1, len(full['path']) - 1
                end = trip['path'][-1]
                editable[s['start']] = editable[end] = True
                for p in s['order']:
                    editable[p] = True
                _need(end in s['free'] and steps <= s['budget'], 'illegal journey')
                _need(sum(b[p] == info['spent'] for p in s['order']) - s['advanced'] == steps,
                      'counter delta differs from traveled distance')
                _need(all(p in s['free'] for p in trip['path']), 'wall crossing')
                t, bottom, l, r = s['gauge']
                depth, cross = ((bottom - t - 1, r - l - 1) if s['direction'][0]
                                else (r - l - 1, bottom - t - 1))
                if split == 'train':
                    _need(trip['turns'] <= 2, 'training exceeds the route-composition range')
                    if steps:
                        active_headings.add(s['direction'])
                    fuel_witness |= s['budget'] < distance and trip['turns'] >= 1 and end != full['path'][-1]
                    displacement = abs(end[0] - s['start'][0]) + abs(end[1] - s['start'][1])
                    displacement_witness |= s['budget'] < distance and steps > displacement
                    gate_witness |= s['budget'] > distance and trip['turns'] >= 1
                    scan_witness |= (s['budget'] > distance and min(depth, cross) >= 2
                                     and s['advanced'] > 0
                                     and (s['advanced'] + steps) // cross > s['advanced'] // cross
                                     and (s['advanced'] + steps) % cross != 0)
                    turn_witness |= end != _walk(s, s['budget'], clockwise=False)['path'][-1]
                    reverse_witness |= end != _walk(s, s['budget'], reverse=True)['path'][-1]
                    first_wall_witness |= end != _walk(s, s['budget'], first_wall=True)['path'][-1]
                    consumed = set(s['order'][s['advanced']:s['advanced'] + steps])
                    final_heading = DIRECTIONS[(DIRECTIONS.index(s['direction']) + trip['turns']) % 4]
                    for heading, which in (((-1, 0), 'up'), (final_heading, 'final')):
                        alternative = [p for p in _order(s['gauge'], heading) if a[p] == info['fuel']][:steps]
                        if set(alternative) != consumed:
                            if which == 'up':
                                fixed_up_witness = True
                            else:
                                final_heading_witness = True
                else:
                    test_headings.add(s['direction'])
                    test_turns.append(trip['turns'])
            _need(np.array_equal(a[~editable], b[~editable]), 'fixed structures changed')
            if split == 'test':
                _need(max(test_turns) >= 3, 'test lacks composed-turn shift')
    _need(len(alphabets) == 1, 'puzzle-wide role meanings changed')
    _need(fuel_witness and gate_witness and scan_witness, 'missing resource or scan lesson')
    _need(displacement_witness, 'training does not distinguish path length from displacement')
    _need(fixed_up_witness and final_heading_witness, 'missing initial-orientation gauge evidence')
    _need(turn_witness and reverse_witness and first_wall_witness, 'missing key/wall counterexamples')
    _need(len(active_headings) >= 2 and test_headings <= active_headings, 'unsupported heading shift')


def puzzle_features(puzzle):
    records = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            a = _grid(pair['input'])
            info = _read(a)
            systems = []
            for s in info['systems']:
                trip, full = _walk(s, s['budget']), _walk(s)
                wall_cells = list(map(tuple, np.argwhere((a == s['carrier']) & ~info['controls'])))
                t, b, l, r = _bbox(wall_cells + [s['gate_cell']])
                sub = np.isin(a[t:b + 1, l:r + 1], (s['carrier'], info['gate']))
                irregular = not (sub[0].all() and sub[-1].all() and sub[:, 0].all() and sub[:, -1].all())
                systems.append(dict(direction=s['direction'], capacity=s['capacity'], fuel=s['budget'],
                                    advanced=s['advanced'], distance=len(full['path']) - 1,
                                    moves=len(trip['path']) - 1, turns=trip['turns'],
                                    stop=('fuel' if s['budget'] < len(full['path']) - 1 else
                                          'exact' if s['budget'] == len(full['path']) - 1 else 'gate'),
                                    irregular=irregular, enclosure_shape=(b-t+1, r-l+1),
                                    gauge_shape=(s['gauge'][1]-s['gauge'][0]+1, s['gauge'][3]-s['gauge'][2]+1),
                                    key_shape=s['sub'].shape, stroke_lengths=s['lengths'],
                                    repeated=len(set(trip['path'])) < len(trip['path']),
                                    carrier=s['carrier'], key_box=s['box'], gauge_box=s['gauge']))
            records.append(dict(split=split, shape=a.shape, count=len(systems), bg=info['bg'],
                                alphabet=tuple(info[k] for k in ('spent', 'fuel', 'marker', 'gate')),
                                systems=systems))
    return records


def _puzzle_signature(puzzle):
    return tuple(tuple((_signature(_grid(p['input'])), _signature(_grid(p['output'])))
                       for p in puzzle[s]) for s in ('train', 'test'))


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _need(len({_puzzle_signature(p) for p in puzzles}) == 42, 'seed diversity')
    _need(_puzzle_signature(puzzles[0]) == _puzzle_signature(generate_puzzle(0)), 'nondeterministic seed')
    # Every legal explicit count, including the smallest and largest contracts.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(173 + nt * 7 + nq, nt, nq)
            _need(len(p['train']) == nt and len(p['test']) == nq, 'ignored explicit counts')
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError('illegal count accepted')
    records = [r for p in puzzles for r in puzzle_features(p)]
    objects = [s for r in records for s in r['systems']]
    _need(len({r['shape'] for r in records}) >= 35, 'canvas geometry diversity')
    _need(len({r['alphabet'] for r in records}) >= 30, 'palette diversity')
    _need({r['count'] for r in records} == {1, 2}, 'system count diversity')
    _need(len({s['gauge_shape'] for s in objects}) >= 20, 'gauge aspect diversity')
    _need(len({s['enclosure_shape'] for s in objects}) >= 30, 'enclosure geometry diversity')
    _need(len({s['key_shape'] for s in objects}) >= 15, 'key geometry diversity')
    _need({s['direction'] for s in objects} == set(DIRECTIONS), 'direction diversity')
    _need(any(s['irregular'] for s in objects) and any(not s['irregular'] for s in objects), 'outline diversity')
    _need({s['stop'] for s in objects} == {'fuel', 'gate', 'exact'}, 'stopping diversity')
    _need(any(s['fuel'] == 0 for s in objects), 'exhausted gauge coverage')
    _need(any(s['advanced'] == 0 for s in objects), 'fresh gauge coverage')
    _need(max(s['turns'] for s in objects) >= 4, 'long-route diversity')
    for p in puzzles[:3]:
        # Geometric covariance is a solver invariant. The separate requirement
        # that training distinguish a literal global-up scan is not preserved
        # when a sole partially updated gauge is itself rotated to face up.
        for pairs in p.values():
            for pair in pairs:
                rotated_input, rotated_output = np.rot90(pair['input']), np.rot90(pair['output'])
                _need(np.array_equal(solve(rotated_input), rotated_output), 'rotation covariance failed')
    damaged = {split: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][split]] for split in puzzles[0]}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    return dict(seeds=42, pairs=len(records), systems=len(objects),
                canvas_shapes=len({r['shape'] for r in records}),
                gauge_shapes=len({s['gauge_shape'] for s in objects}),
                enclosure_shapes=len({s['enclosure_shape'] for s in objects}),
                key_shapes=len({s['key_shape'] for s in objects}),
                max_turns=max(s['turns'] for s in objects))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
