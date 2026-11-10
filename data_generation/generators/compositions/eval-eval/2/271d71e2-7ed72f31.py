"""A shared unary battery funds atomic distance layers of symmetric copies.

Only standard library and NumPy. No parent code, files, or generation state are
used by solve. See task_spec.md for the complete supported input domain.
"""
from __future__ import annotations

import copy
import numpy as np

BACKGROUND, SPENT, LIVE = 0, 1, 3
STEPS = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)


def _need(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'invalid grid dimensions')
    _need(np.issubdtype(a.dtype, np.integer), 'grid must contain integers')
    _need(np.all((a >= 0) & (a <= 9)), 'invalid color')
    return a.astype(int, copy=False)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in STEPS:
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    stack.append(q)
        result.append(sorted(cells))
    return result


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _counter_order(bounds, direction):
    t, b, l, r = bounds
    if direction == 'U':
        return [(y, x) for y in range(t+1, b) for x in range(l+1, r)]
    if direction == 'D':
        return [(y, x) for y in range(b-1, t, -1) for x in range(r-1, l, -1)]
    if direction == 'L':
        return [(y, x) for x in range(l+1, r) for y in range(b-1, t, -1)]
    if direction == 'R':
        return [(y, x) for x in range(r-1, l, -1) for y in range(t+1, b)]
    raise AssertionError('invalid direction')


def _battery_candidates(g):
    candidates = []
    for color in sorted(set(map(int, np.unique(g))) - {BACKGROUND, SPENT, LIVE}):
        for component in _components(g == color):
            t, b, l, r = _bbox(component)
            for direction in ('U', 'D', 'L', 'R'):
                bounds = (t+1, b, l, r) if direction == 'U' else (
                    (t, b-1, l, r) if direction == 'D' else (
                    (t, b, l+1, r) if direction == 'L' else (t, b, l, r-1)))
                y0, y1, x0, x1 = bounds
                if y1-y0 < 3 or x1-x0 < 3:
                    continue
                border = {(y, x) for y in range(y0, y1+1) for x in range(x0, x1+1)
                          if y in (y0, y1) or x in (x0, x1)}
                extra = set(component) - border
                if len(extra) != 1 or set(component) != border | extra:
                    continue
                nose = next(iter(extra))
                eligible = ((direction == 'U' and nose[0] == y0-1 and x0 < nose[1] < x1) or
                            (direction == 'D' and nose[0] == y1+1 and x0 < nose[1] < x1) or
                            (direction == 'L' and nose[1] == x0-1 and y0 < nose[0] < y1) or
                            (direction == 'R' and nose[1] == x1+1 and y0 < nose[0] < y1))
                if not eligible:
                    continue
                order = _counter_order(bounds, direction)
                values = [int(g[q]) for q in order]
                nspent = values.count(SPENT)
                if values != [SPENT]*nspent + [LIVE]*(len(values)-nspent):
                    continue
                candidates.append(dict(frame=color, bounds=bounds, nose=nose,
                                       direction=direction, order=order, spent=nspent,
                                       fuel=len(order)-nspent, border=border))
    return candidates


def _kind(marker):
    t, b, l, r = _bbox(marker)
    if len(marker) == 1:
        return 'P'
    if t == b and len(marker) == r-l+1:
        return 'H'
    if l == r and len(marker) == b-t+1:
        return 'V'
    raise AssertionError('marker is neither a line nor a singleton')


def _mapping(g, marker, payload):
    kind = _kind(marker)
    ar, ac = marker[0]
    t, b, l, r = _bbox(payload)
    if min(max(abs(y-a), abs(x-c)) for y, x in payload for a, c in marker) != 1:
        return None
    if kind == 'H':
        if not (b < ar or t > ar) or r < min(c for a, c in marker) or l > max(c for a, c in marker):
            return None
        dest = [(2*ar-y, x) for y, x in payload]
        layers = [abs(y-ar) for y, x in payload]
    elif kind == 'V':
        if not (r < ac or l > ac) or b < min(a for a, c in marker) or t > max(a for a, c in marker):
            return None
        dest = [(y, 2*ac-x) for y, x in payload]
        layers = [abs(x-ac) for y, x in payload]
    else:
        if not ((b <= ar or t >= ar) and (r <= ac or l >= ac)):
            return None
        dest = [(2*ar-y, 2*ac-x) for y, x in payload]
        layers = [max(abs(y-ar), abs(x-ac)) for y, x in payload]
    if any(not (0 <= y < g.shape[0] and 0 <= x < g.shape[1]) or g[y, x] != BACKGROUND
           for y, x in dest):
        return None
    return dict(kind=kind, marker=marker, source=payload, dest=dest, layers=layers)


def _read(g):
    batteries = _battery_candidates(g)
    _need(len(batteries) == 1, 'input must have one unambiguous nosed battery')
    battery = batteries[0]
    t, b, l, r = battery['bounds']
    masked = g.copy()
    masked[t:b+1, l:r+1] = BACKGROUND
    masked[battery['nose']] = BACKGROUND
    _need(not np.any((masked == LIVE) | (masked == SPENT)), 'counter colors outside battery')
    markers = _components(masked == battery['frame'])
    payloads = _components((masked != BACKGROUND) & (masked != battery['frame']))
    _need(markers and len(markers) == len(payloads), 'unmatched groups')
    _need(all(len({int(g[q]) for q in p}) == 1 for p in payloads), 'payload must be monochrome')
    groups, used, destinations = [], set(), set()
    for marker in markers:
        choices = []
        for index, payload in enumerate(payloads):
            mapped = _mapping(g, marker, payload)
            if mapped is not None:
                choices.append((index, mapped))
        _need(len(choices) == 1, 'ambiguous or missing marker association')
        index, group = choices[0]
        _need(index not in used, 'payload owned twice')
        _need(not destinations.intersection(group['dest']), 'symmetric destinations overlap')
        used.add(index)
        destinations.update(group['dest'])
        group['color'] = int(g[group['source'][0]])
        groups.append(group)
    _need(len(used) == len(payloads), 'unexplained payload')
    return battery, groups


def _costs(groups):
    costs = {}
    for group in groups:
        for d in group['layers']:
            costs[d] = costs.get(d, 0) + 1
    return dict(sorted(costs.items()))


def _progress(groups, fuel):
    used, cutoff = 0, 0
    for distance, cost in _costs(groups).items():
        if used + cost > fuel:
            break
        used += cost
        cutoff = distance
    return cutoff, used


def _apply(g, battery, groups):
    cutoff, used = _progress(groups, battery['fuel'])
    out = g.copy()
    for group in groups:
        for source, dest, distance in zip(group['source'], group['dest'], group['layers']):
            if distance <= cutoff:
                out[dest] = g[source]
    for q in battery['order'][battery['spent']:battery['spent']+used]:
        out[q] = SPENT
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read all geometry and charge from this input alone, without mutation."""
    g = _grid(input_grid)
    battery, groups = _read(g)
    return _apply(g, battery, groups)


def _normalize(group):
    cells = group['marker'] + group['source'] + group['dest']
    t, b, l, r = _bbox(cells)
    result = {k: [(y-t, x-l) for y, x in group[k]] for k in ('marker', 'source', 'dest')}
    result.update(kind=group['kind'], layers=group['layers'][:], shape=(b-t+1, r-l+1))
    return result


def _shape(rng, depth, span, kind, witness=False):
    # A connected scaffold reaches every distance; extra pixels alter layer prices.
    if kind == 'P':
        aligned = bool(rng.random() < .30 and not witness)
        cells = {(d, d) for d in range(1, depth+1)}
        allowed = {(y, x) for y in range(0 if aligned else 1, depth+1)
                   for x in range(0 if aligned else 1, depth+1)} - {(0, 0)}
    else:
        spine = int(rng.integers(span))
        cells = {(d, spine) for d in range(1, depth+1)}
        allowed = {(y, x) for y in range(1, depth+1) for x in range(span)}
    style = int(rng.integers(4))
    if style == 0:
        # Eight-neighbor random growth, with sparse and dense targets.
        target = int(rng.integers(len(cells), len(allowed)+1))
        while len(cells) < target:
            frontier = sorted(q for q in allowed-cells if any(
                max(abs(q[0]-p[0]), abs(q[1]-p[1])) == 1 for p in cells))
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])
    elif style == 1:
        cells |= {q for q in allowed if rng.random() < .55}
        # Keep only the component connected to the scaffold.
        todo, reached = list(cells & ({(1, 1)} if kind == 'P' else {(1, spine)})), set()
        while todo:
            q = todo.pop()
            if q in reached:
                continue
            reached.add(q)
            todo.extend((q[0]+dy, q[1]+dx) for dy, dx in STEPS
                        if (q[0]+dy, q[1]+dx) in cells-reached)
        cells = reached
    elif style == 2:
        cells |= {q for q in allowed if q[0] in (1, depth) or q[1] in
                  ((1, depth) if kind == 'P' else (0, span-1))}
    else:
        cells = allowed - {q for q in allowed if q[0] < depth and
                          q[1] > (depth//2 if kind == 'P' else span//2)}
        # Scaffold prevents disconnected corner cuts.
        cells |= {(d, d) for d in range(1, depth+1)} if kind == 'P' else {(d, spine) for d in range(1, depth+1)}
    if witness:
        # Unequal near/far edges reveal reflection/rotation, not just placement.
        neighbor = 1 if kind == 'P' else (spine+1 if spine+1 < span else spine-1)
        cells.add((1, 2 if kind == 'P' else neighbor))
        cells.add((2, neighbor))
        cells.discard((depth, neighbor))
    # A corner removal can isolate a far perimeter pixel. Retain the component
    # containing the scaffold; the intact scaffold still reaches every depth.
    anchor = (1, 1) if kind == 'P' else (1, spine)
    todo, reached = [anchor], set()
    while todo:
        q = todo.pop()
        if q in reached or q not in cells:
            continue
        reached.add(q)
        todo.extend((q[0]+dy, q[1]+dx) for dy, dx in STEPS
                    if (q[0]+dy, q[1]+dx) in cells-reached)
    return sorted(reached)


def _prototype(rng, kind, depth, witness=False):
    span = int(rng.integers(2, 6))
    source = _shape(rng, depth, span, kind, witness)
    if kind == 'P':
        marker = [(0, 0)]
        sy, sx = map(int, rng.choice((-1, 1), size=2))
        source = [(sy*y, sx*x) for y, x in source]
        dest = [(-y, -x) for y, x in source]
        layers = [max(abs(y), abs(x)) for y, x in source]
    else:
        # The finite axis may be shorter or longer than the shape's span.
        contact = min(x for y, x in source if y == 1)
        first = int(rng.integers(-1, contact+1))
        last = int(rng.integers(max(first+1, contact), max(first+2, span+1)))
        marker = [(0, x) for x in range(first, last+1)]
        sign = int(rng.choice((-1, 1)))
        source = [(sign*y, x) for y, x in source]
        dest = [(-y, x) for y, x in source]
        layers = [abs(y) for y, x in source]
        if kind == 'V':
            marker, source, dest = [[(x, y) for y, x in v] for v in (marker, source, dest)]
    return _normalize(dict(kind=kind, marker=marker, source=source, dest=dest, layers=layers))


def _battery(rng, minimum_capacity):
    # Both counter dimensions can vary; capacities are not a fixed shape code.
    widths = [w for w in range(2, 11) if max(2, (minimum_capacity+w-1)//w) <= 11]
    _need(widths, 'charge panel too large')
    w = int(rng.choice(widths))
    h = max(2, (minimum_capacity+w-1)//w)
    if h < 11 and rng.random() < .35:
        h += 1
    border = [(y, x) for y in range(1, h+3) for x in range(w+2)
              if y in (1, h+2) or x in (0, w+1)]
    nose = (0, int(rng.integers(1, w+1)))
    return dict(shape=(h+3, w+2), bounds=(1, h+2, 0, w+1), nose=nose,
                border=border, order=_counter_order((1, h+2, 0, w+1), 'U'),
                capacity=h*w)


def _pack(rng, shapes, height, width):
    blocked = np.zeros((height, width), bool)
    placements = [None]*len(shapes)
    for index in sorted(range(len(shapes)), key=lambda i: -shapes[i][0]*shapes[i][1]):
        h, w = shapes[index]
        if h > height or w > width:
            return None
        choices = [(y, x) for y in range(height-h+1) for x in range(width-w+1)]
        rng.shuffle(choices)
        for y, x in choices:
            if not blocked[y:y+h, x:x+w].any():
                placements[index] = (y, x)
                blocked[max(0, y-1):min(height, y+h+1), max(0, x-1):min(width, x+w+1)] = True
                break
        else:
            return None
    return placements


def _scene(rng, groups, battery, frame, palette):
    shapes = [battery['shape']] + [g['shape'] for g in groups]
    placements = None
    for attempt in range(24):
        H = int(rng.integers(18, 31)) if attempt < 12 else 30
        W = int(rng.integers(18, 31)) if attempt < 12 else 30
        placements = _pack(rng, shapes, H, W)
        if placements is not None:
            break
    if placements is None:
        return None
    grid = np.zeros((H, W), int)
    y, x = placements[0]
    bat = dict(battery)
    bat['bounds'] = tuple(v + (y if i < 2 else x) for i, v in enumerate(battery['bounds']))
    bat['nose'] = (y+battery['nose'][0], x+battery['nose'][1])
    bat['order'] = [(y+a, x+b) for a, b in battery['order']]
    for a, b in battery['border'] + [battery['nose']]:
        grid[y+a, x+b] = frame
    grid[tuple(np.array(bat['order']).T)] = LIVE
    placed = []
    for group, (y, x) in zip(groups, placements[1:]):
        moved = {k: [(y+a, x+b) for a, b in group[k]] for k in ('marker', 'source', 'dest')}
        moved.update(kind=group['kind'], layers=group['layers'][:], color=int(rng.choice(palette)))
        for q in moved['marker']:
            grid[q] = frame
        for q in moved['source']:
            grid[q] = moved['color']
        placed.append(moved)
    # Trim unused space with unequal outer margins; retain optional edge contact.
    occupied = np.argwhere(grid != BACKGROUND)
    # Include all prospective destinations, even those the budget will not reach.
    extra = np.array([q for g in placed for q in g['dest']])
    bounds = np.vstack((occupied, extra))
    t = max(0, int(bounds[:, 0].min())-int(rng.integers(0, 3)))
    l = max(0, int(bounds[:, 1].min())-int(rng.integers(0, 3)))
    b = min(H, int(bounds[:, 0].max())+1+int(rng.integers(0, 3)))
    r = min(W, int(bounds[:, 1].max())+1+int(rng.integers(0, 3)))
    return grid[t:b, l:r].copy()


def _set_charge(g, fuel):
    battery, groups = _read(g)
    _need(0 <= fuel <= len(battery['order']), 'invalid charge')
    for i, q in enumerate(battery['order']):
        g[q] = SPENT if i < len(battery['order'])-fuel else LIVE
    return g


def _orientation_translation(group):
    t, b, l, r = _bbox(group['source'])
    a, z, c, d = _bbox(group['dest'])
    return [(y+a-t, x+c-l) for y, x in group['source']]


def _alternative(g, mode):
    """Deliberately simpler rules, used solely as falsifiable training checks."""
    battery, groups = _read(g)
    if mode == 'point_manhattan':
        changed = []
        for group in groups:
            group = dict(group)
            if group['kind'] == 'P':
                ar, ac = group['marker'][0]
                group['layers'] = [abs(y-ar)+abs(x-ac) for y, x in group['source']]
            changed.append(group)
        return _apply(g, battery, changed)
    cutoff, used = _progress(groups, battery['fuel'])
    out = g.copy()
    pixels = sorted((d, dest, src) for group in groups
                    for d, dest, src in zip(group['layers'], group['dest'], group['source']))
    if mode == 'pixel_truncate':
        selected = pixels[:battery['fuel']]
        used = len(selected)
    elif mode == 'layer_ticks':
        selected = [p for p in pixels if p[0] <= battery['fuel']]
        used = min(battery['fuel'], max(_costs(groups)))
    elif mode == 'skip_layer':
        selected, used = [], 0
        for d, cost in _costs(groups).items():
            if used+cost <= battery['fuel']:
                selected.extend(p for p in pixels if p[0] == d)
                used += cost
    else:
        selected = None
    if selected is not None:
        for d, dest, src in selected:
            out[dest] = g[src]
        for q in battery['order'][battery['spent']:battery['spent']+used]:
            out[q] = SPENT
        return out
    if mode == 'ignore_A':
        cutoff, used = max(_costs(groups)), 0
    for group in groups:
        translate = (mode == 'ignore_B' or mode == 'point_translation' and group['kind'] == 'P' or
                     mode == 'line_translation' and group['kind'] in ('H', 'V'))
        destinations = _orientation_translation(group) if translate else group['dest']
        for src, dest, layer in zip(group['source'], destinations, group['layers']):
            if layer <= cutoff:
                out[dest] = g[src]
    order = battery['order'] if mode != 'fixed_scan' else sorted(battery['order'])
    pending = [q for q in order if g[q] == LIVE]
    for q in pending[:used]:
        out[q] = SPENT
    return out


def _coupling_witness(g):
    battery, groups = _read(g)
    cutoff, _ = _progress(groups, battery['fuel'])
    for index, removed in enumerate(groups):
        others = groups[:index]+groups[index+1:]
        if not others:
            continue
        newcutoff, _ = _progress(others, battery['fuel'])
        if any(cutoff < d <= newcutoff for group in others for d in group['layers']):
            return True
    return False


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, name, low, high in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a non-negative integer')
    rng = np.random.default_rng(seed)
    frame = int(rng.choice((2, 4, 5, 6, 7, 8, 9)))
    available = sorted(set(range(10)) - {BACKGROUND, SPENT, LIVE, frame})
    palette = list(map(int, rng.choice(available, size=int(rng.integers(1, len(available)+1)), replace=False)))
    train_depth = int(rng.integers(3, 5))
    # Contrastive demonstrations: identical work rotated, different visible charge.
    for attempt in range(100):
        groups = [_prototype(rng, 'P', train_depth, True),
                  _prototype(rng, str(rng.choice(('H', 'V'))), train_depth, True)]
        costs = _costs(groups)
        if costs.get(2, 0) < 2:
            continue
        # Teach that an unaffordable near layer cannot be skipped for a cheaper
        # farther layer, and that both groups draw on the same charge.
        later_price = min(v for d, v in costs.items() if d > 2)
        freeing_price = max(sum(d <= 2 for d in group['layers']) for group in groups)
        lower = max(1, later_price, costs[2]-freeing_price)
        if lower >= costs[2]:
            continue
        total = sum(costs.values())
        surplus = int(rng.integers(1, 6))
        battery = _battery(rng, total+surplus+int(rng.integers(1, 5)))
        base = _scene(rng, groups, battery, frame, palette)
        if base is None:
            continue
        lowfuel = costs[1] + int(rng.integers(lower, costs[2]))
        turn = int(rng.integers(4))
        low = np.rot90(_set_charge(base.copy(), lowfuel), turn).copy()
        full = np.rot90(_set_charge(base.copy(), total+surplus), (turn+int(rng.choice((1, 3)))) % 4).copy()
        lessons = [dict(input=low, output=solve(low)), dict(input=full, output=solve(full))]
        if all(any(not np.array_equal(_alternative(p['input'], mode), p['output']) for p in lessons)
               for mode in ('ignore_A', 'ignore_B', 'fixed_scan', 'pixel_truncate', 'layer_ticks', 'skip_layer',
                            'point_manhattan', 'point_translation', 'line_translation')):
            break
    else:
        raise RuntimeError('could not construct informative foundation')
    rng.shuffle(lessons)
    puzzle = {'train': lessons, 'test': []}
    seen = {_signature(p['input']) for p in lessons}
    for split, count in (('train', num_train-2), ('test', num_test)):
        for index in range(count):
            for attempt in range(100):
                count_groups = int(rng.integers(1, 5))
                maxdepth = train_depth if split == 'train' else train_depth+int(rng.integers(1, 3))
                kinds = list(map(str, rng.choice(('H', 'V', 'P'), size=count_groups)))
                prototypes = [_prototype(rng, k, maxdepth if j == 0 else int(rng.integers(1, maxdepth+1)))
                              for j, k in enumerate(kinds)]
                costs = _costs(prototypes)
                total = sum(costs.values())
                if total > 95:
                    continue
                if split == 'test':
                    # A later partial front at new depth, with some charge possibly left.
                    cutoff = int(rng.integers(2, maxdepth))
                    paid = sum(v for d, v in costs.items() if d <= cutoff)
                    nextcost = next(v for d, v in costs.items() if d > cutoff)
                    fuel = paid + int(rng.integers(nextcost))
                else:
                    fuel = int(rng.integers(total+6))
                capacity_needed = fuel + int(rng.integers(0, 9))
                bat = _battery(rng, max(4, capacity_needed))
                grid = _scene(rng, prototypes, bat, frame, palette)
                if grid is None:
                    continue
                grid = np.rot90(_set_charge(grid, fuel), int(rng.integers(4))).copy()
                signature = _signature(grid)
                if signature in seen:
                    continue
                pair = dict(input=grid, output=solve(grid))
                seen.add(signature)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError('bounded scene construction failed')
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(g):
    return g.shape, tuple(map(int, g.flat))


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle schema')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _need(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4 and
          len(puzzle['train']) > len(puzzle['test']), 'invalid counts')
    seen, roles, training = set(), set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair schema')
            g, out = _grid(pair['input']), _grid(pair['output'])
            _need(g.shape == out.shape, 'shape changed')
            sig = _signature(g)
            _need(sig not in seen, 'duplicate input')
            seen.add(sig)
            bat, groups = _read(g)
            roles.add(bat['frame'])
            costs = _costs(groups)
            _need(set(costs) == set(range(1, max(costs)+1)), 'disconnected distance shells')
            cutoff, used = _progress(groups, bat['fuel'])
            _need(np.array_equal(_apply(g, bat, groups), out), 'incorrect answer')
            # Resource conservation, atomic frontier, maximality, and retained source.
            _need(sum(int(out[q]) == SPENT for q in bat['order']) == bat['spent']+used,
                  'counter charge is not conserved')
            _need(np.count_nonzero((g == BACKGROUND) & (out != BACKGROUND)) == used,
                  'one copied pixel must consume one cell')
            allowed = set(bat['order'])
            for group in groups:
                _need(all(out[q] == g[q] for q in group['marker']+group['source']), 'source or marker changed')
                for src, dest, layer in zip(group['source'], group['dest'], group['layers']):
                    _need(out[dest] == (g[src] if layer <= cutoff else BACKGROUND), 'incomplete or advanced layer')
                    allowed.add(dest)
            _need(all(tuple(q) in allowed for q in np.argwhere(g != out)), 'unrelated cell changed')
            if cutoff < max(costs):
                nextcost = next(v for d, v in costs.items() if d > cutoff)
                _need(bat['fuel']-used < nextcost, 'wave stopped before an affordable layer')
            if split == 'train':
                training.append((g, out, bat, groups, cutoff, used))
    _need(len(roles) == 1, 'structural color changed within puzzle')
    _need(any(0 < cutoff < max(_costs(groups)) and bat['fuel'] > used for g, out, bat, groups, cutoff, used in training),
          'training must show atomic stopping with leftover fuel')
    _need(any(cutoff == max(_costs(groups)) and bat['fuel'] > used for g, out, bat, groups, cutoff, used in training),
          'training must show completion without consuming surplus')
    _need(any(len(groups) >= 2 and used >= 2 for g, out, bat, groups, cutoff, used in training), 'missing shared-cost witness')
    kinds = {group['kind'] for g, out, bat, groups, cutoff, used in training for group in groups}
    _need('P' in kinds and bool(kinds & {'H', 'V'}), 'training must show point and line geometry')
    _need(any(_coupling_witness(g) for g, out, bat, groups, cutoff, used in training),
          'training conceals shared resource coupling')
    for mode in ('ignore_A', 'ignore_B', 'fixed_scan', 'pixel_truncate', 'layer_ticks', 'skip_layer',
                 'point_manhattan', 'point_translation', 'line_translation'):
        _need(any(not np.array_equal(_alternative(g, mode), out) for g, out, bat, groups, cutoff, used in training),
              f'training does not distinguish {mode}')


def self_test():
    """Bounded 42-seed diversity and family-invariant checks; no external solver."""
    shapes, structures, silhouettes, frames, directions, counts, depths, capacities = [set() for _ in range(8)]
    modes, point_aligned, overhang, holes, stopped = set(), 0, 0, 0, 0
    puzzles = []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        for split in ('train', 'test'):
            for pair in p[split]:
                g, out = pair['input'], pair['output']
                original = g.copy()
                _need(np.array_equal(solve(g), out) and np.array_equal(g, original), 'solver mutation or disagreement')
                bat, groups = _read(g)
                frames.add(bat['frame'])
                directions.add(bat['direction'])
                capacities.add(len(bat['order']))
                shapes.add(g.shape)
                counts.add(len(groups))
                remap = {}
                structures.add((g.shape, tuple(remap.setdefault(int(c), len(remap)) for c in g.flat)))
                cut, paid = _progress(groups, bat['fuel'])
                stopped += int(cut < max(_costs(groups)) and bat['fuel'] > paid)
                modes.add(tuple(sorted({x['kind'] for x in groups})))
                for group in groups:
                    t, b, l, r = _bbox(group['source'])
                    local = np.zeros((b-t+1, r-l+1), bool)
                    for y, x in group['source']:
                        local[y-t, x-l] = True
                    silhouettes.add((local.shape, tuple(map(int, local.flat))))
                    depths.add(max(group['layers']))
                    ar, ac = group['marker'][0]
                    point_aligned += int(group['kind'] == 'P' and any(y == ar or x == ac for y, x in group['source']))
                    if group['kind'] in ('H', 'V'):
                        coordinate = 1 if group['kind'] == 'H' else 0
                        lo, hi = min(q[coordinate] for q in group['marker']), max(q[coordinate] for q in group['marker'])
                        overhang += int(any(q[coordinate] < lo or q[coordinate] > hi for q in group['source']))
                    # Hole evidence without SciPy: flood empty padded mask from exterior.
                    padded = np.pad(local, 1)
                    empty = set(map(tuple, np.argwhere(~padded).tolist()))
                    stack, reached = [(0, 0)], set()
                    while stack:
                        q = stack.pop()
                        if q in reached or q not in empty:
                            continue
                        reached.add(q)
                        stack.extend((q[0]+dy, q[1]+dx) for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)))
                    holes += int(len(empty) > len(reached))
                # Rotation covariance is an extra consistency invariant of the rule.
                _need(np.array_equal(solve(np.rot90(g)), np.rot90(out)), 'rule is not rotation covariant')
    _need(len(structures) == 210 and len(shapes) >= 35, 'geometry diversity collapsed')
    _need(len(silhouettes) >= 60 and counts == {1, 2, 3, 4}, 'payload diversity collapsed')
    _need(len(frames) >= 6 and directions == {'U', 'D', 'L', 'R'}, 'role or orientation diversity collapsed')
    _need(len(capacities) >= 20 and max(depths) >= 6, 'counter or distance range collapsed')
    _need(point_aligned > 0 and overhang > 0 and holes > 0 and stopped >= 42, 'compatible geometry modes missing')
    _need(len(modes) >= 6, 'operation mixtures collapsed')
    # Regression for a near-edge witness cut that once isolated a far pixel.
    validate_generated_puzzle(generate_puzzle(195485966))
    repeated = generate_puzzle(17)
    _need(all(np.array_equal(a[k], b[k]) for s in repeated for a, b in zip(repeated[s], puzzles[17][s])
              for k in ('input', 'output')), 'nondeterministic seed')
    # Uniform states remain input-only solvable; state colors need not both appear.
    source = puzzles[0]['train'][0]['input']
    battery, groups = _read(source)
    empty = _set_charge(source.copy(), 0)
    _need(np.array_equal(solve(empty), empty), 'exhausted battery must be stationary')
    fresh = _set_charge(source.copy(), len(battery['order']))
    charged = solve(fresh)
    _need(np.count_nonzero((fresh == BACKGROUND) & (charged != BACKGROUND)) == sum(_costs(groups).values()),
          'fresh battery did not finish affordable work')
    for tr in range(2, 11):
        for te in range(1, min(4, tr-1)+1):
            p = generate_puzzle(1000+tr*10+te, tr, te)
            _need(len(p['train']) == tr and len(p['test']) == te, 'count ignored')
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    corrupt = copy.deepcopy(puzzles[0])
    corrupt['test'][0]['output'][0, 0] = (int(corrupt['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted a wrong label')
    partial = copy.deepcopy(puzzles[0])
    for i, pair in enumerate(partial['train']):
        bat, groups = _read(pair['input'])
        cut, used = _progress(groups, bat['fuel'])
        if cut < max(_costs(groups)) and bat['fuel'] > used:
            pair['output'] = _alternative(pair['input'], 'pixel_truncate')
            break
    try:
        validate_generated_puzzle(partial)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted a partially purchased shell')
    return dict(seeds=42, pairs=210, grid_shapes=len(shapes), color_normalized_structures=len(structures),
                silhouettes=len(silhouettes), frame_colors=len(frames), capacities=len(capacities),
                group_counts=sorted(counts), depths=sorted(depths), point_aligned=point_aligned,
                line_overhang=overhang, holed_payloads=holes, stopped_with_charge=stopped)
