"""Ink-budget stencil printers: visible dictionaries, marginal ink costs.

Only standard Python and NumPy are used. No parent code is imported.
"""

import copy
from collections import Counter

import numpy as np


_D4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
_TURNS = (0, 3, 2, 1)  # rotate an upward printer into U, R, D, L


def _need(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _need(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "invalid grid shape")
    _need(np.issubdtype(g.dtype, np.integer), "grid must contain integers")
    _need(np.all((0 <= g) & (g <= 9)), "invalid ARC color")
    return g.astype(np.int8, copy=False)


def _components(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    result = []
    while cells:
        first = min(cells)
        cells.remove(first)
        stack, component = [first], []
        while stack:
            r, c = stack.pop()
            component.append((int(r), int(c)))
            for dr, dc in _D4:
                p = (r + dr, c + dc)
                if p in cells:
                    cells.remove(p)
                    stack.append(p)
        result.append(component)
    return result


def _border(t, b, l, r):
    return ({(t, c) for c in range(l, r + 1)} |
            {(b, c) for c in range(l, r + 1)} |
            {(y, l) for y in range(t, b + 1)} |
            {(y, r) for y in range(t, b + 1)})


def _bounds(cells):
    ys, xs = zip(*cells)
    return min(ys), max(ys), min(xs), max(xs)


def _counter_order(t, b, l, r, direction):
    if direction == 0:
        return [(y, x) for y in range(t + 1, b) for x in range(l + 1, r)]
    if direction == 1:
        return [(y, x) for x in range(r - 1, l, -1) for y in range(t + 1, b)]
    if direction == 2:
        return [(y, x) for y in range(b - 1, t, -1) for x in range(r - 1, l, -1)]
    return [(y, x) for x in range(l + 1, r) for y in range(b - 1, t, -1)]


def _offset_rotate(p, turns):
    r, c = p
    for _ in range(turns % 4):
        r, c = -c, r
    return int(r), int(c)


def _reference_candidates(g, bg):
    """The library is a detached rectangular frame with a blank inner rim."""
    found = []
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for component in _components(g == color):
            t, b, l, r = _bounds(component)
            if b - t < 4 or r - l < 4 or set(component) != _border(t, b, l, r):
                continue
            inner = g[t + 1:b, l + 1:r]
            if not (np.all(inner[0] == bg) and np.all(inner[-1] == bg) and
                    np.all(inner[:, 0] == bg) and np.all(inner[:, -1] == bg)):
                continue
            motifs = _components(inner != bg)
            if len(motifs) < 2:
                continue
            templates, repeats, paints = {}, Counter(), set()
            valid = True
            for cells in motifs:
                values = Counter(int(inner[p]) for p in cells)
                singleton = [c for c, n in values.items() if n == 1]
                if len(cells) < 3 or len(values) != 2 or len(singleton) != 1:
                    valid = False
                    break
                key = singleton[0]
                paint = next(c for c in values if c != key)
                ar, ac = next(p for p in cells if int(inner[p]) == key)
                sig = tuple(sorted((y - ar, x - ac) for y, x in cells if (y, x) != (ar, ac)))
                if key in templates and templates[key] != sig:
                    valid = False
                    break
                templates[key] = sig
                repeats[key] += 1
                paints.add(paint)
            if (valid and len(paints) == 1 and all(n >= 2 for n in repeats.values()) and
                    not (set(templates) & paints) and color not in set(templates) | paints):
                found.append(dict(box=(t, b, l, r), frame=color, templates=templates,
                                  repeats=dict(repeats), paint=next(iter(paints))))
    return found


def _read_battery(g, component, keys):
    """Read one rectangular frame, its straight feed wire, and its keyed head."""
    points = set(component)
    y0, y1, x0, x1 = _bounds(points)
    candidates = []
    for direction, (dy, dx) in enumerate(_D4):
        for wire_length in range(0, 5):
            t, b, l, r = y0, y1, x0, x1
            if direction == 0:
                t += wire_length
            elif direction == 1:
                r -= wire_length
            elif direction == 2:
                b -= wire_length
            else:
                l += wire_length
            if b - t < 2 or r - l < 2:
                continue
            sides = ([(t, x) for x in range(l + 1, r)] if direction == 0 else
                     [(y, r) for y in range(t + 1, b)] if direction == 1 else
                     [(b, x) for x in range(l + 1, r)] if direction == 2 else
                     [(y, l) for y in range(t + 1, b)])
            for side in sides:
                wire = {(side[0] + dy * k, side[1] + dx * k)
                        for k in range(1, wire_length + 1)}
                if points != _border(t, b, l, r) | wire:
                    continue
                head = (side[0] + dy * (wire_length + 1), side[1] + dx * (wire_length + 1))
                if not (0 <= head[0] < g.shape[0] and 0 <= head[1] < g.shape[1]):
                    continue
                key = int(g[head])
                if key not in keys:
                    continue
                order = _counter_order(t, b, l, r, direction)
                values = [int(g[p]) for p in order]
                spent, remaining = values[0], values[-1]
                if spent == remaining or len(set(values)) != 2:
                    continue
                advanced = values.count(spent)
                if values != [spent] * advanced + [remaining] * (len(values) - advanced):
                    continue
                candidates.append(dict(box=(t, b, l, r), direction=direction, head=head,
                                       key=key, order=order, advanced=advanced,
                                       remaining=len(values) - advanced, spent_color=spent,
                                       remaining_color=remaining, wire=wire))
    _need(len(candidates) == 1, "battery lacks a unique nozzle and counter interpretation")
    return candidates[0]


def _interpret(input_grid):
    """Recover every role and route exclusively from visible input cells."""
    g = _grid(input_grid)
    bg = int(g[0, 0])
    _need(np.all(g[0] == bg) and np.all(g[-1] == bg) and
          np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg), "missing blank outer rim")
    refs = _reference_candidates(g, bg)
    _need(len(refs) == 1, "input must have exactly one consistent motif library")
    ref = refs[0]
    t, b, l, r = ref['box']
    reference_border = _border(t, b, l, r)
    batteries = []
    for component in _components(g == ref['frame']):
        if set(component) == reference_border:
            continue
        batteries.append(_read_battery(g, component, set(ref['templates'])))
    _need(1 <= len(batteries) <= 3, "expected one to three printers")
    alphabet = {(v['spent_color'], v['remaining_color']) for v in batteries}
    _need(len(alphabet) == 1, "inconsistent counter alphabet")
    spent, remaining = next(iter(alphabet))
    reserved = {bg, ref['frame'], ref['paint'], spent, remaining} | set(ref['templates'])
    _need(len(reserved) == 5 + len(ref['templates']), "semantic colors must be distinct")
    stops = set(map(int, np.unique(g))) - reserved
    _need(len(stops) == 1, "one distinct terminal-marker color is required")
    stop = next(iter(stops))
    terminals = set(map(tuple, np.argwhere(g == stop)))
    used_terminals, cover, sweeps = set(), np.zeros(g.shape, dtype=np.int8), []
    cover[t:b + 1, l:r + 1] = (g[t:b + 1, l:r + 1] != bg).astype(np.int8)
    for v in batteries:
        dy, dx = _D4[v['direction']]
        hy, hx = v['head']
        candidates = [(int((y - hy) * dy + (x - hx) * dx), (int(y), int(x)))
                      for y, x in terminals if (y - hy) * dx == (x - hx) * dy and
                      (y - hy) * dy + (x - hx) * dx >= 2]
        _need(len(candidates) == 1, "route must have exactly one aligned terminal")
        distance_plus_one, terminal = candidates[0]
        distance = distance_plus_one - 1
        _need(1 <= distance <= 8, "invalid travel distance")
        _need(terminal not in used_terminals, "shared terminal")
        used_terminals.add(terminal)
        stencil = ref['templates'][v['key']]
        lateral = (dx, -dy)
        canonical = [(dr * dy + dc * dx, dr * lateral[0] + dc * lateral[1])
                     for dr, dc in stencil]
        _need(all(cross > 0 for _, cross in canonical), "ink must lie right of the route")
        gap = len(v['wire']) + 1
        _need(gap > -min(s for s, _ in canonical), "stencil could touch the battery")
        stamps = []
        for step in range(1, distance + 1):
            center = (hy + step * dy, hx + step * dx)
            _need(int(g[center]) == bg, "head corridor is not clear")
            stamp = {(center[0] + dr, center[1] + dc) for dr, dc in stencil}
            _need(all(0 < y < g.shape[0] - 1 and 0 < x < g.shape[1] - 1
                      for y, x in stamp), "clipped stencil")
            _need(all(int(g[p]) in (bg, ref['paint']) for p in stamp), "blocked stencil footprint")
            stamps.append(stamp)
        potential = set().union(*stamps)
        old_ink = {p for p in potential if int(g[p]) == ref['paint']}
        vt, vb, vl, vr = v['box']
        structures = {(y, x) for y in range(vt, vb + 1) for x in range(vl, vr + 1)}
        structures |= v['wire'] | {v['head'], terminal}
        _need(not structures & potential, "ink overlaps printer structures")
        route = {(hy + step * dy, hx + step * dx) for step in range(distance + 1)}
        footprint = structures | potential | route
        bounds = _bounds(footprint)
        bt, bb, bl, br = bounds
        _need(not (bt <= b and t <= bb and bl <= r and l <= br), "route crosses library")
        for at, ab, al, ar in sweeps:
            _need(bb < at or ab < bt or br < al or ar < bl, "printer sweep boxes overlap")
        sweeps.append(bounds)
        for p in structures | old_ink:
            cover[p] += 1
        v.update(distance=distance, terminal=terminal, stamps=stamps, old_ink=old_ink,
                 potential=potential, bounds=bounds, stencil=stencil)
    _need(used_terminals == terminals, "unassigned terminal")
    _need(np.max(cover) == 1, "shared input structures")
    _need(np.array_equal(cover == 1, g != bg), "unexplained foreground")
    return g, dict(background=bg, reference=ref, batteries=batteries, stop=stop,
                   spent=spent, remaining=remaining)


def _walk(v, *, ignore_old=False, full_stamp_cost=False, one_cell_cost=False, unlimited=False):
    ink = set() if ignore_old else set(v['old_ink'])
    budget, used, steps, costs = v['remaining'], 0, 0, []
    for stamp in v['stamps']:
        new = stamp - ink
        cost = 1 if one_cell_cost else (len(stamp) if full_stamp_cost else len(new))
        if not unlimited and cost > budget:
            break
        ink |= stamp
        budget -= cost
        used += cost
        steps += 1
        costs.append(cost)
    return dict(ink=ink, used=used, steps=steps, left=budget, costs=costs)


def _pixel_ablation(v):
    """Omit the key-selected motif: print only the nearest right-hand pixel."""
    dy, dx = _D4[v['direction']]
    hy, hx = v['head']
    pixel = dict(v)
    pixel['stamps'] = [{(hy + k * dy + dx, hx + k * dx - dy)}
                       for k in range(1, v['distance'] + 1)]
    return _walk(pixel)


def solve(input_grid: np.ndarray) -> np.ndarray:
    g, info = _interpret(input_grid)
    output = g.copy()
    for v in info['batteries']:
        result = _walk(v)
        output[v['head']] = info['background']
        for p in result['ink']:
            output[p] = info['reference']['paint']
        for p in v['order'][v['advanced']:v['advanced'] + result['used']]:
            output[p] = info['spent']
        dy, dx = _D4[v['direction']]
        output[v['head'][0] + result['steps'] * dy,
               v['head'][1] + result['steps'] * dx] = v['key']
    return output


def _choice(rng, items):
    return items[int(rng.integers(len(items)))]


def _shape(rng, wide=False, small=False, teaching=False):
    """Connected stencil growth in (forward, right) coordinates."""
    low = -1 if small else int(rng.integers(-2, 1))
    high = 1 if small else int(rng.integers(1, 4))
    width = 3 if wide else (2 if teaching else int(rng.integers(1, 3)))
    cells = {(0, 1)}
    if teaching or wide:
        cells.add((0, 2))
    if wide:
        cells.add((0, 3))
    # One axial adjacency in lessons makes translated-footprint reuse visible.
    if teaching:
        cells.add((1, 1))
    limit = min((high - low + 1) * width, 5 if small else 10 if wide else 8)
    n = int(rng.integers(max(2, len(cells)), limit + 1))
    while len(cells) < n:
        frontier = sorted({(s + ds, c + dc) for s, c in cells for ds, dc in _D4
                           if low <= s + ds <= high and 1 <= c + dc <= width} - cells)
        cells.add(_choice(rng, frontier))
    return tuple(sorted(cells))


def _printer(rng, roles, key, shape, direction, regime, prepaint_mode, small=False, teaching=False):
    distance = 3 if small else int(rng.integers(3, 7))
    max_s = max(s for s, _ in shape)
    min_s = min(s for s, _ in shape)
    gap = max(1, 1 - min_s) + (0 if small else int(rng.integers(0, 2)))
    hy = distance + max(1, max_s)
    # Abstract ink coordinates are independent of the counter dimensions.
    stamps = [{(hy - step - s, c) for s, c in shape} for step in range(1, distance + 1)]
    old = set()
    if prepaint_mode == 'first_free':
        old |= stamps[0]
    elif prepaint_mode == 'patches':
        for _ in range(int(rng.integers(1, 4))):
            chosen = _choice(rng, stamps)
            selected = sorted(chosen)
            rng.shuffle(selected)
            old.update(selected[:int(rng.integers(1, len(selected) + 1))])
    union, cumulative, tolls = set(old), [0], []
    for stamp in stamps:
        cost = len(stamp - union)
        tolls.append(cost)
        union |= stamp
        cumulative.append(cumulative[-1] + cost)
    total = cumulative[-1]
    if total == 0:
        return None
    if regime == 'destination':
        fuel = total + int(rng.integers(1, 4))
    elif regime == 'exact':
        fuel = total
    elif regime == 'atomic':
        possible = [k for k in range(1, distance) if cumulative[k] > 0 and tolls[k] >= 2]
        if not possible:
            return None
        k = _choice(rng, possible)
        fuel = cumulative[k] + int(rng.integers(1, tolls[k]))
    elif regime == 'fuel':
        possible = [k for k in range(distance) if cumulative[k + 1] > max(1, cumulative[k])]
        if not possible:
            return None
        k = _choice(rng, possible)
        fuel = int(rng.integers(max(1, cumulative[k]), cumulative[k + 1]))
    else:
        raise AssertionError('unknown stopping regime')
    dimensions = [(h, w) for h in range(1, 7) for w in range(1, 10)
                  if fuel + 1 <= h * w <= 36 and (not teaching or min(h, w) >= 2)]
    if small:
        dimensions = [(h, w) for h, w in dimensions if 2 <= h <= 3 and 2 <= w <= 6]
    if not dimensions:
        return None
    ih, iw = _choice(rng, dimensions)
    fh, fw = ih + 2, iw + 2
    hx = int(rng.integers(1, fw - 1))
    frame_top = hy + gap
    height = frame_top + fh
    width = max(fw, hx + max(c for _, c in shape) + 1)
    bg, frame, spent, remaining, paint, stop = roles
    g = np.full((height, width), bg, dtype=np.int8)
    g[frame_top:frame_top + fh, :fw] = frame
    advanced = ih * iw - fuel
    block = np.full((ih, iw), remaining, dtype=np.int8)
    block.flat[:advanced] = spent
    g[frame_top + 1:frame_top + fh - 1, 1:fw - 1] = block
    g[hy + 1:frame_top, hx] = frame
    g[hy, hx] = key
    g[hy - distance - 1, hx] = stop
    for y, c in old:
        g[y, hx + c] = paint
    out = g.copy()
    budget, paid, steps, ink = fuel, 0, 0, set(old)
    # Labels are constructed before scene placement, without calling solve.
    for stamp in stamps:
        cost = len(stamp - ink)
        if cost > budget:
            break
        ink |= stamp
        budget -= cost
        paid += cost
        steps += 1
    for y, c in ink:
        out[y, hx + c] = paint
    updated = block.copy()
    updated.flat[advanced:advanced + paid] = spent
    out[frame_top + 1:frame_top + fh - 1, 1:fw - 1] = updated
    out[hy, hx] = bg
    out[hy - steps, hx] = key
    turns = _TURNS[direction]
    return np.rot90(g, turns).copy(), np.rot90(out, turns).copy()


def _library_options(rng, roles, templates, small=False):
    """Pack repeated examples into one library, without a fixed panel layout."""
    bg, frame, _, _, paint, _ = roles
    copies = [(key, sig) for key, sig in templates.items() for _ in range(2)]
    rng.shuffle(copies)
    options = []
    for width in range(6, 27):
        y, x, row_height, places = 2, 2, 0, []
        for key, sig in copies:
            t, b, l, r = _bounds([(0, 0)] + list(sig))
            h, w = b - t + 1, r - l + 1
            if w + 4 > width:
                break
            if x + w > width - 2:
                y += row_height + 1
                x, row_height = 2, 0
            places.append((key, y - t, x - l))
            x += w + 1
            row_height = max(row_height, h)
        else:
            height = y + row_height + 2
            if height <= 26:
                panel = np.full((height, width), frame, dtype=np.int8)
                panel[1:-1, 1:-1] = bg
                for key, ar, ac in places:
                    panel[ar, ac] = key
                    for dr, dc in templates[key]:
                        panel[ar + dr, ac + dc] = paint
                options.append(panel)
    options.sort(key=lambda g: g.size)
    return options[:max(2, len(options) // 2)] if small else options


def _pack(rng, shapes):
    """Pack actual reserved sweep rectangles, with one blank cell between them."""
    for _ in range(32):
        occupied = np.zeros((28, 28), dtype=bool)
        places = [None] * len(shapes)
        order = list(range(len(shapes)))
        rng.shuffle(order)
        order.sort(key=lambda i: -shapes[i][0] * shapes[i][1])
        for i in order:
            h, w = shapes[i]
            possible = []
            for y in range(29 - h):
                for x in range(29 - w):
                    if not np.any(occupied[max(0, y - 1):min(28, y + h + 1),
                                           max(0, x - 1):min(28, x + w + 1)]):
                        possible.append((y, x))
            if not possible:
                break
            if rng.random() < .75:
                edge = [p for p in possible if p[0] == 0 or p[1] == 0 or
                        p[0] + h == 28 or p[1] + w == 28]
                if edge:
                    possible = edge
            y, x = _choice(rng, possible)
            occupied[y:y + h, x:x + w] = True
            places[i] = (y, x)
        else:
            return places
    # Deterministic geometric recovery: library on one side of a row/column
    # of printers. Only used when free packing failed; no change to the rule.
    rh, rw = shapes[0]
    printers = shapes[1:]
    if rh + 1 + max(h for h, w in printers) <= 28 and sum(w for h, w in printers) + len(printers) - 1 <= 28:
        places, x = [(0, 0)], 0
        for h, w in printers:
            places.append((rh + 1, x))
            x += w + 1
        return places
    if rw + 1 + max(w for h, w in printers) <= 28 and sum(h for h, w in printers) + len(printers) - 1 <= 28:
        places, y = [(0, 0)], 0
        for h, w in printers:
            places.append((y, rw + 1))
            y += h + 1
        return places
    return None


def _pair(rng, roles, keys, shift, split, direction_pool, lesson=None, small=False,
          forced_direction=None):
    is_test = split == 'test'
    count = int(rng.choice((1, 2, 3), p=(.15, .55, .30) if is_test else (.40, .50, .10)))
    if shift == 'keys':
        ntypes = int(rng.integers(2, 4)) if is_test else 1
        count = max(count, ntypes)
    else:
        ntypes = int(rng.integers(1, min(count, 3) + 1))
    chosen_keys = list(map(int, rng.choice(keys, ntypes, replace=False)))
    common_direction = forced_direction if forced_direction is not None else _choice(rng, direction_pool)
    directions = {key: common_direction if small else _choice(rng, direction_pool) for key in chosen_keys}
    if forced_direction is not None:
        directions[chosen_keys[0]] = forced_direction
    shapes = {key: _shape(rng, wide=is_test and shift == 'width' and i == 0,
                          small=small, teaching=lesson is not None and i == 0)
              for i, key in enumerate(chosen_keys)}
    templates = {key: tuple(sorted(_offset_rotate((-s, c), _TURNS[directions[key]])
                                   for s, c in shape)) for key, shape in shapes.items()}
    target_ids = chosen_keys + [_choice(rng, chosen_keys) for _ in range(count - ntypes)]
    if lesson is None:
        rng.shuffle(target_ids)
    blocks = []
    taught_test_key = False
    for i, key in enumerate(target_ids):
        if i == 0 and lesson == 'destination':
            regime, prepaint = 'destination', 'first_free'
        elif i == 0 and lesson == 'atomic':
            regime, prepaint = 'atomic', _choice(rng, ('none', 'patches'))
        elif i == 0 and lesson == 'orientation':
            regime, prepaint = 'exact', 'none'
        elif is_test and key == chosen_keys[0] and not taught_test_key:
            # The shifted stencil must actually deposit new ink. A test made
            # entirely of one free advance would barely exercise the new width.
            regime, prepaint = _choice(rng, ('destination', 'exact')), _choice(rng, ('none', 'patches', 'first_free'))
            taught_test_key = True
        else:
            regime = _choice(rng, ('destination', 'exact', 'fuel', 'atomic'))
            prepaint = _choice(rng, ('none', 'patches', 'first_free'))
        block = _printer(rng, roles, key, shapes[key], directions[key], regime, prepaint, small,
                         teaching=i == 0 and lesson is not None)
        if block is None:
            return None
        blocks.append(block)
    libraries = _library_options(rng, roles, templates, small)
    if not libraries:
        return None
    rng.shuffle(libraries)
    for ref in libraries[:8]:
        grids = [(ref, ref)] + blocks
        places = _pack(rng, [g.shape for g, _ in grids])
        if places is None:
            continue
        miny, minx = min(y for y, x in places), min(x for y, x in places)
        maxy = max(y + g.shape[0] for (y, x), (g, _) in zip(places, grids))
        maxx = max(x + g.shape[1] for (y, x), (g, _) in zip(places, grids))
        core_h, core_w = maxy - miny, maxx - minx
        top = int(rng.integers(1, min(3, 30 - core_h)))
        left = int(rng.integers(1, min(3, 30 - core_w)))
        bottom = int(rng.integers(1, min(3, 31 - core_h - top)))
        right = int(rng.integers(1, min(3, 31 - core_w - left)))
        inp = np.full((core_h + top + bottom, core_w + left + right), roles[0], dtype=np.int8)
        out = inp.copy()
        for (y, x), (g, o) in zip(places, grids):
            y, x = y - miny + top, x - minx + left
            inp[y:y + g.shape[0], x:x + g.shape[1]] = g
            out[y:y + o.shape[0], x:x + o.shape[1]] = o
        try:
            _, info = _interpret(inp)
            if lesson:
                vs = info['batteries']
                if lesson == 'atomic' and not any(0 < _walk(v)['steps'] < v['distance'] and
                                                  _walk(v)['left'] > 0 for v in vs):
                    continue
                if lesson == 'destination' and not any(_walk(v)['steps'] == v['distance'] and
                                                       _walk(v)['costs'][0] == 0 for v in vs):
                    continue
            if not np.array_equal(solve(inp), out):
                raise AssertionError('constructive label disagrees with input-derived solver')
            return dict(input=inp, output=out)
        except AssertionError:
            continue
    return None


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    return int(num_train), int(num_test)


def _signature(g):
    return g.shape, g.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(10)))
    roles, keys = tuple(palette[:6]), palette[6:9]
    shift = _choice(rng, ('keys', 'width'))
    direction_pool = list(map(int, rng.choice(4, int(rng.integers(2, min(4, ntrain) + 1)), replace=False)))
    lesson_indices = list(map(int, rng.choice(ntrain, len(direction_pool), replace=False)))
    lessons = {lesson_indices[i]: (('destination', 'atomic')[i] if i < 2 else 'orientation', d)
               for i, d in enumerate(direction_pool)}
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', ntrain), ('test', ntest)):
        for index in range(count):
            for attempt in range(80):
                lesson, direction = lessons.get(index, (None, None)) if split == 'train' else (None, None)
                pair = _pair(rng, roles, keys, shift, split, direction_pool, lesson=lesson,
                             small=attempt >= 40, forced_direction=direction)
                if pair is not None and _signature(pair['input']) not in seen:
                    seen.add(_signature(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded scene construction failed')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _need(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, palettes, train_results, train_vehicles, infos = set(), set(), [], [], {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            g, output = _grid(pair['input']), _grid(pair['output'])
            sig = _signature(g)
            _need(sig not in seen, 'duplicate input')
            seen.add(sig)
            _, info = _interpret(g)
            _need(np.array_equal(solve(g), output), 'incorrect output')
            _need(g.shape == output.shape, 'scene dimensions changed')
            ref = info['reference']
            palettes.add((info['background'], ref['frame'], info['spent'], info['remaining'], ref['paint'], info['stop']))
            t, b, l, r = ref['box']
            _need(np.array_equal(output[t:b + 1, l:r + 1], g[t:b + 1, l:r + 1]), 'library changed')
            for v in info['batteries']:
                result = _walk(v)
                _need(0 <= result['used'] <= v['remaining'], 'ink budget exceeded')
                _need(result['used'] == len(result['ink'] - v['old_ink']), 'cost is not marginal ink area')
                _need(v['old_ink'] <= result['ink'], 'old ink erased')
                if result['steps'] < v['distance']:
                    next_cost = len(v['stamps'][result['steps']] - result['ink'])
                    _need(next_cost > result['left'], 'printer stopped before an affordable stamp')
                t, b, l, r = v['box']
                _need(all(int(output[p]) == ref['frame'] for p in _border(t, b, l, r)), 'battery frame moved')
                _need(int(output[v['terminal']]) == info['stop'], 'terminal erased')
                _need(all(int(output[p]) == ref['paint'] for p in result['ink']), 'incomplete ink union')
                _need(sum(int(output[p]) == info['spent'] for p in v['order']) == v['advanced'] + result['used'],
                      'counter does not record ink expenditure')
                if split == 'train':
                    train_results.append(result)
                    train_vehicles.append(v)
            infos[split].append(info)
    _need(len(palettes) == 1, 'puzzle palette conventions changed')
    _need(any(r['steps'] == v['distance'] and r['left'] > 0 for r, v in zip(train_results, train_vehicles)),
          'training lacks destination-limited movement')
    _need(any(0 < r['steps'] < v['distance'] and r['left'] > 0 for r, v in zip(train_results, train_vehicles)),
          'training lacks active atomic ink-limited stopping with residue')
    _need(any(r['costs'] and r['costs'][0] == 0 for r in train_results), 'training lacks free movement on old ink')
    def differs(v, **kwargs):
        correct, shortcut = _walk(v), _walk(v, **kwargs)
        return (correct['steps'], correct['used']) != (shortcut['steps'], shortcut['used'])
    _need(any(differs(v, one_cell_cost=True) for v in train_vehicles),
          'one-cell-per-move shortcut is not contradicted')
    _need(any(_walk(v, unlimited=True)['steps'] != _walk(v)['steps'] for v in train_vehicles),
          'counterless shortcut is not contradicted')
    _need(any(_pixel_ablation(v)['ink'] != _walk(v)['ink'] for v in train_vehicles),
          'omitting the selected stencil does not change the printed answer')
    _need(any(differs(v, full_stamp_cost=True) for v in train_vehicles),
          'full-stamp cost shortcut is not contradicted')
    _need(any(_walk(v, ignore_old=True)['used'] != _walk(v)['used'] or
              _walk(v, ignore_old=True)['steps'] != _walk(v)['steps'] for v in train_vehicles),
          'old-ink reuse is not demonstrated')
    active_directions = {v['direction'] for v in train_vehicles if _walk(v)['steps'] > 0}
    _need(len(active_directions) >= 2, 'training lacks counter-rotation evidence')
    _need(all(v['direction'] in active_directions for i in infos['test'] for v in i['batteries']),
          'test introduces an orientation without an active demonstration')
    _need(all(any(_walk(v)['used'] > 0 and _walk(v)['steps'] > 1 for v in i['batteries'])
              for i in infos['test']), 'test must actively print new ink')
    # The shift is recovered from geometry, without generation metadata.
    train_types = max(len(i['reference']['templates']) for i in infos['train'])
    test_types = min(len(i['reference']['templates']) for i in infos['test'])
    def cross_width(v):
        dy, dx = _D4[v['direction']]
        return max(dr * dx - dc * dy for dr, dc in v['stencil'])
    train_width = max(cross_width(v) for v in train_vehicles)
    test_width = [max(cross_width(v) for v in i['batteries']) for i in infos['test']]
    _need((train_types == 1 and test_types >= 2) or all(w > train_width for w in test_width),
          'missing visible train/test composition or stencil-width shift')


def _puzzle_signature(p):
    return tuple((_signature(np.asarray(pair['input'])), _signature(np.asarray(pair['output'])))
                 for split in ('train', 'test') for pair in p[split])


def self_test():
    """Bounded 42-seed audit, semantic probes, and corrupted-label rejection."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _need(len({_puzzle_signature(p) for p in puzzles}) == 42, 'seed variation collapsed')
    _need(_puzzle_signature(puzzles[0]) == _puzzle_signature(generate_puzzle(0)), 'nondeterminism')
    shapes, silhouettes, directions, regimes, counts, palette_roles = set(), set(), set(), set(), set(), set()
    widths, capacities, prepaint, counter_shapes = set(), set(), set(), set()
    for p in puzzles:
        validate_generated_puzzle(p)
        for split in ('train', 'test'):
            for pair in p[split]:
                g, info = _interpret(pair['input'])
                shapes.add(g.shape)
                counts.add(len(info['batteries']))
                palette_roles.add((info['background'], info['reference']['paint'], info['spent'], info['remaining']))
                for sig in info['reference']['templates'].values():
                    silhouettes.add(sig)
                for v in info['batteries']:
                    result = _walk(v)
                    directions.add(v['direction'])
                    regime = 'destination' if result['steps'] == v['distance'] else 'fuel'
                    regimes.add((v['direction'], regime))
                    capacities.add(len(v['order']))
                    prepaint.add(len(v['old_ink']))
                    t, b, l, r = v['box']
                    counter_shapes.add((b - t - 1, r - l - 1))
                    dy, dx = _D4[v['direction']]
                    widths.add(max(dr * dx - dc * dy for dr, dc in v['stencil']))
    _need(len(shapes) >= 30 and len(silhouettes) >= 70, 'structural diversity collapsed')
    _need(directions == set(range(4)), 'missing a direction')
    _need(regimes == {(d, s) for d in range(4) for s in ('destination', 'fuel')}, 'direction/stopping correlation')
    _need(counts == {1, 2, 3} and widths == {1, 2, 3}, 'count or width diversity collapsed')
    _need(len(capacities) >= 10 and len(counter_shapes) >= 12 and len(prepaint) >= 8,
          'resource geometry collapsed')
    _need(len(palette_roles) >= 30, 'palette variation collapsed')
    # Rotation must rotate the stencil, head route, and counter scan together.
    for pair in puzzles[0]['train']:
        for turns in range(4):
            _need(np.array_equal(solve(np.rot90(pair['input'], turns)), np.rot90(pair['output'], turns)),
                  'rotation covariance failed')
    # Direct marginal-area probe: free first stamp, then a paid stamp, then
    # stop with one ink cell left; no skipping ahead to an already inked island.
    v = dict(remaining=3, old_ink={(0, 0), (0, 1)},
             stamps=[{(0, 0), (0, 1)}, {(0, 1), (0, 2), (1, 2)},
                     {(0, 2), (0, 3), (1, 3)}, {(0, 0), (0, 1)}])
    result = _walk(v)
    _need(result['steps'] == 2 and result['used'] == 2 and result['left'] == 1, 'atomic stopping probe')
    exhausted = dict(remaining=0, old_ink={(0, 0)}, stamps=[{(0, 0)}, {(0, 1)}])
    _need(_walk(exhausted)['steps'] == 1, 'zero-cost motion after exhaustion')
    # A full input-derived probe outside the sampler's teaching dimensions:
    # a thin 1x3 counter, off-center nozzle, and free motion after exhaustion.
    demo = np.zeros((18, 13), dtype=np.int8)
    demo[1:7, 1:11] = 1
    demo[2:6, 2:10] = 0
    for ar, ac in ((4, 3), (4, 7)):
        demo[ar, ac] = 6
        demo[ar, ac + 1] = demo[ar - 1, ac + 1] = 4
    demo[14:17, 2:7] = 1
    demo[15, 3:6] = [2, 3, 3]
    demo[13, 3] = 6
    demo[9, 3] = 5
    demo[9:11, 4] = 4
    expected = demo.copy()
    expected[13, 3] = 0
    expected[10, 3] = 6
    expected[9:13, 4] = 4
    expected[15, 3:6] = 2
    _need(np.array_equal(solve(demo), expected), 'thin counter and exhausted free-travel scene')
    inconsistent = demo.copy()
    inconsistent[3, 8] = 0
    try:
        solve(inconsistent)
    except AssertionError:
        pass
    else:
        raise AssertionError('inconsistent repeated dictionary accepted')
    broken = copy.deepcopy(puzzles[0])
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted a corrupt output')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    return dict(seeds=42, pairs=210, input_shapes=len(shapes), stencil_geometries=len(silhouettes),
                counter_capacities=len(capacities), counter_shapes=len(counter_shapes),
                printer_counts=sorted(counts), directions=sorted(directions),
                palette_assignments=len(palette_roles), prepaint_areas=len(prepaint))


if __name__ == '__main__':
    print(self_test())
