"""Keyed convoy rendezvous. Standard library + NumPy; no parent imports.

The input-only solver reconstructs the manifest, vehicles and counters. It
searches an analytic motion schedule. Construction instead runs a unit clock.
"""
from __future__ import annotations

import itertools
import json

import numpy as np


def _need(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _need(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "bad grid dimensions")
    _need(np.issubdtype(g.dtype, np.integer), "grid must contain integers")
    _need(np.all((0 <= g) & (g <= 9)), "bad grid colors")
    return g.astype(np.int8, copy=True)


def _key(g):
    return g.shape, g.tobytes()


def _trim(g):
    yy, xx = np.nonzero(g)
    _need(len(yy) > 0, "empty token")
    return g[yy.min():yy.max() + 1, xx.min():xx.max() + 1]


def _center(mask, h, w):
    cropped = _trim(mask)
    out = np.zeros((h, w), dtype=bool)
    y, x = (h - cropped.shape[0]) // 2, (w - cropped.shape[1]) // 2
    out[y:y + cropped.shape[0], x:x + cropped.shape[1]] = cropped
    return out


def _order(h, w):
    # Parent A's travel-right raster: nearest column first, top to bottom.
    return [(y, x) for x in range(w - 1, -1, -1) for y in range(h)]


def _components(g, color):
    remaining = set(map(tuple, np.argwhere(g == color)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = [start], []
        while todo:
            y, x = todo.pop()
            cells.append((int(y), int(x)))
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                p = y + dy, x + dx
                if p in remaining:
                    remaining.remove(p)
                    todo.append(p)
        result.append(cells)
    return result


def _frame_boxes(g, color):
    boxes = []
    for cells in _components(g, color):
        yy, xx = zip(*cells)
        t, b, l, r = min(yy), max(yy), min(xx), max(xx)
        _need(4 <= b - t + 1 <= 5 and r - l >= 3, "not a compartment frame")
        block = g[t:b + 1, l:r + 1]
        _need(np.all(block[[0, -1]] == color) and
              np.all(block[:, [0, -1]] == color), "broken outer frame")
        divs = [x for x in range(1, r - l) if np.all(block[:, x] == color)]
        expected = np.zeros(block.shape, bool)
        expected[[0, -1]], expected[:, [0, -1]] = True, True
        expected[:, divs] = True
        _need(np.array_equal(block == color, expected), "broken internal divider")
        edges = [0] + divs + [r - l]
        widths = [v - u - 1 for u, v in zip(edges, edges[1:])]
        _need(all(w >= 1 for w in widths), "empty compartment")
        boxes.append(dict(top=t, left=l, bottom=b, right=r, edges=edges, widths=widths))
    return sorted(boxes, key=lambda v: (v['top'], v['left']))


def _parse_with_frame(g, frame):
    boxes = _frame_boxes(g, frame)
    _need(len(boxes) >= 3, "too few frames")
    racks, vehicles, nose_colors = [], [], set()
    cover = np.zeros(g.shape, bool)
    for box in boxes:
        t, b, l, r = (box[k] for k in ('top', 'bottom', 'left', 'right'))
        nose = g[t:b + 1, r + 1] if r + 1 < g.shape[1] else np.zeros(b - t + 1, int)
        is_car = len(np.unique(nose)) == 1 and int(nose[0]) not in (0, frame)
        if is_car:
            _need(len(box['widths']) == 2, "a vehicle needs two compartments")
            nose_colors.add(int(nose[0]))
            vehicles.append(dict(box))
            cover[t:b + 1, r + 1] = True
        else:
            racks.append(box)
        _need(not np.any(cover[t:b + 1, l:r + 1]), "overlapping objects")
        cover[t:b + 1, l:r + 1] = True
    _need(len(racks) == 1 and len(nose_colors) == 1, "manifest/nose ambiguity")
    rack = racks[0]
    spent = next(iter(nose_colors))
    n = len(rack['widths'])
    _need(2 <= n <= 4 and len(set(rack['widths'])) == 1, "bad manifest slots")
    gw = rack['widths'][0]
    h = rack['bottom'] - rack['top'] - 1
    _need(2 <= gw <= 3 and 2 <= h <= 3, "bad glyph canvas")
    _need(all(v['top'] > rack['bottom'] for v in vehicles), "manifest must precede lanes")
    tops = sorted({v['top'] for v in vehicles})
    _need(len(tops) == n and all(v - u >= h + 3 for u, v in zip(tops, tops[1:])),
          "manifest slot count must equal separated lane count")
    counter_colors = set()
    lookup, trimkeys = {}, set()
    counter_widths = set()
    for v in vehicles:
        t, b, l, r = (v[k] for k in ('top', 'bottom', 'left', 'right'))
        cw, payload_width = v['widths']
        _need(b - t - 1 == h and payload_width == gw and 1 <= cw <= 3,
              "inconsistent compartments")
        counter_widths.add(cw)
        counter = g[t + 1:b, l + 1:l + 1 + cw].copy()
        token = g[t + 1:b, l + cw + 2:r].copy()
        inks = set(map(int, np.unique(token[token != 0])))
        _need(len(inks) == 1 and not (inks & {frame, spent}), "invalid carried token")
        _need(np.array_equal(token != 0, _center(token != 0, h, gw)), "off-center token")
        tk, trimmed = _key(token), _key(_trim(token))
        _need(tk not in lookup and trimmed not in trimkeys, "nonunique exact token")
        lookup[tk] = len(lookup)
        trimkeys.add(trimmed)
        v.update(counter=counter, token=token, lane=tops.index(t), cw=cw,
                 index=lookup[tk], width=r - l + 1)
        counter_colors.update(map(int, np.unique(counter)))
    _need(len(counter_widths) == 1, "counter widths differ within a scene")
    unused_colors = counter_colors - {spent}
    _need(len(unused_colors) <= 1 and not (unused_colors & {0, frame}), "invalid fuel alphabet")
    fuel = next(iter(unused_colors)) if unused_colors else None
    for v in vehicles:
        _need(fuel is None or fuel not in np.unique(v['token']), "fuel is token ink")
        values = [int(v['counter'][p]) for p in _order(h, v['cw'])]
        advanced = values.count(spent)
        _need(values == [spent] * advanced + [fuel] * (len(values) - advanced),
              "counter is not a travel-relative prefix")
        v.update(advanced=advanced, remaining=len(values) - advanced)
        _need(v['right'] + 1 + v['remaining'] < g.shape[1], "full journey leaves canvas")
    for lane in range(n):
        cars = sorted([v for v in vehicles if v['lane'] == lane], key=lambda v: v['left'])
        _need(1 <= len(cars) <= 2, "a lane has one or two vehicles")
        for a, b in zip(cars, cars[1:]):
            _need(a['right'] + 1 + a['remaining'] + 1 < b['left'], "overlapping swept corridors")
    queries, query_slots = [], []
    for slot, (a, b) in enumerate(zip(rack['edges'], rack['edges'][1:])):
        token = g[rack['top'] + 1:rack['bottom'], rack['left'] + a + 1:rack['left'] + b]
        if np.any(token):
            _need(_key(token) in lookup, "manifest glyph lacks an exact vehicle copy")
            queries.append(lookup[_key(token)])
            query_slots.append(slot)
    _need(1 <= len(queries) <= n and len(set(queries)) == len(queries), "invalid manifest selection")
    _need(np.all(g[~cover] == 0), "unexplained foreground or nonzero exterior")
    return dict(grid=g, frame=frame, spent=spent, fuel=fuel, rack=rack, n=n,
                h=h, gw=gw, cw=next(iter(counter_widths)), cars=vehicles,
                queries=queries, query_slots=query_slots)


def _parse(input_grid):
    g = _grid(input_grid)
    candidates = []
    for turn in range(4):
        rotated = np.rot90(g, turn).copy()
        for color in np.unique(rotated):
            if color == 0:
                continue
            try:
                info = _parse_with_frame(rotated, int(color))
            except (AssertionError, IndexError):
                continue
            info['turn'] = turn
            candidates.append(info)
    _need(len(candidates) == 1, "input lacks a unique complete scene explanation")
    return candidates[0]


def _line_complete(cars, selected, n, positions):
    return (len(selected) == n and {cars[i]['lane'] for i in selected} == set(range(n))
            and len({positions[i] for i in selected}) == 1)


def _event(info):
    cars = info['cars']
    horizon = max(v['remaining'] for v in cars)
    for tick in range(horizon + 1):
        noses = [v['right'] + 1 + min(tick, v['remaining']) for v in cars]
        if _line_complete(cars, info['queries'], info['n'], noses):
            return tick, True
    return horizon, False


def _paint_car(g, car, frame, spent, fuel, h, gw, cw):
    t, l = car['top'], car['left']
    width = cw + gw + 3
    g[t:t + h + 2, l:l + width] = frame
    g[t + 1:t + h + 1, l + cw + 2:l + width - 1] = car['token']
    for i, (y, x) in enumerate(_order(h, cw)):
        g[t + 1 + y, l + 1 + x] = spent if i < car['advanced'] else fuel
    g[t:t + h + 2, l + width] = spent


def _output_at(info, advances):
    out = info['grid'].copy()
    for v in info['cars']:
        out[v['top']:v['bottom'] + 1, v['left']:v['right'] + 2] = 0
    for v, movement in zip(info['cars'], advances):
        moved = dict(v, left=v['left'] + movement, advanced=v['advanced'] + movement)
        _paint_car(out, moved, info['frame'], info['spent'], info['fuel'],
                   info['h'], info['gw'], info['cw'])
    return np.rot90(out, -info['turn']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve entirely from input pixels; no training, seed or generated state."""
    info = _parse(input_grid)
    tick, _ = _event(info)
    return _output_at(info, [min(tick, v['remaining']) for v in info['cars']])


def _unit_clock(cars, selected, n, width):
    """Construction-time per-step motion; independent of analytic event search."""
    positions = [v['left'] + width for v in cars]
    remaining = [v['remaining'] for v in cars]
    moves = [0] * len(cars)
    tick = 0
    while True:
        completed = _line_complete(cars, selected, n, positions)
        if completed or not any(remaining):
            return moves, tick, completed
        for i in range(len(cars)):
            if remaining[i]:
                remaining[i] -= 1
                positions[i] += 1
                moves[i] += 1
        tick += 1


def _shape_bank(h, w):
    bank = {}
    # All nonempty masks, including disconnected, solid, hollow and narrow ink.
    for bits in range(1, 1 << (h * w)):
        mask = np.array([(bits >> i) & 1 for i in range(h * w)], bool).reshape(h, w)
        mask = _center(mask, h, w)
        bank.setdefault(_key(mask), mask)
    return list(bank.values())


def _tokens(rng, count, h, w, inks, confusable):
    bank = _shape_bank(h, w)
    candidates = [np.where(mask, ink, 0).astype(np.int8) for mask in bank for ink in inks]
    rng.shuffle(candidates)
    if confusable:
        # Couple each confusable token to the selected car in ITS OWN lane.
        # This lets a one-slot ink-only or shape-only query edit retain full
        # lane coverage while changing the rendezvous on an identical field.
        n, ink_lane, shape_lane = confusable
        mask_indices = rng.choice(len(bank), n, replace=False)
        primary = [np.where(bank[int(i)], int(rng.choice(inks)), 0).astype(np.int8)
                   for i in mask_indices]
        ink_source, shape_source = primary[ink_lane], primary[shape_lane]
        first = np.where(ink_source != 0,
                         next(c for c in inks if c != int(ink_source[ink_source != 0][0])), 0).astype(np.int8)
        second = next(c for c in candidates
                      if int(c[c != 0][0]) == int(shape_source[shape_source != 0][0])
                      and _key(c != 0) != _key(shape_source != 0)
                      and _key(c) != _key(first)
                      and all(_key(c) != _key(v) for v in primary))
        chosen = primary + [first, second]
        while len(chosen) < count:
            chosen.append(next(c for c in candidates if all(_key(c) != _key(v) for v in chosen)))
        selected = chosen
    else:
        selected = candidates[:count]
    _need(len({_key(v) for v in selected}) == count, "token construction alias")
    return selected


def _selected_layout(rng, n, capacity, kind, long_trip):
    offset = 0 if long_trip else int(rng.integers(3))
    if kind == 'unreachable':
        # The leader remains ahead of the follower for the entire schedule.
        gap = int(rng.integers(1, 4))
        behind = int(rng.integers(1, capacity))
        starts = [offset, offset + gap]
        fuels = [behind, int(rng.integers(behind, capacity + 1))]
        while len(starts) < n:
            starts.append(offset + int(rng.integers(gap + 1)))
            fuels.append(int(rng.integers(capacity + 1)))
        return starts, fuels, None
    if kind == 'already':
        starts = [offset] * n
        fuels = list(map(int, rng.integers(0, capacity + 1, size=n)))
        fuels[0], fuels[1] = 0, capacity
        return starts, fuels, 0
    limit = capacity - 1 if long_trip else min(capacity - 1, 2)
    tick = int(rng.integers(max(1, capacity - 3) if long_trip else 1, limit + 1))
    target = offset + tick
    distances = [tick, int(rng.integers(tick))]
    while len(distances) < n:
        distances.append(int(rng.integers(tick + 1)))
    fuels = [int(rng.integers(tick + 1, capacity + 1)), distances[1]]
    for d in distances[2:]:
        fuels.append(int(rng.integers(tick, capacity + 1)) if d == tick else d)
    starts = [target - d for d in distances]
    # Shuffle lane identities so a particular lane never dictates stopping.
    perm = rng.permutation(n)
    return [starts[int(i)] for i in perm], [fuels[int(i)] for i in perm], tick


def _make_pair(rng, parameters, kind, cw, long_trip=False, contrasts=False):
    n, h, gw = (parameters[k] for k in ('n', 'h', 'gw'))
    frame, spent, fuel = (parameters[k] for k in ('frame', 'spent', 'fuel'))
    width, capacity = cw + gw + 3, h * cw
    basekind = 'success' if kind in ('partial', 'same_lane') else kind
    starts, fuels, planned_tick = _selected_layout(rng, n, capacity, basekind, long_trip)
    # Parallel strips need no lattice dividers; vertical spacing can be irregular.
    gaps = [1] * n  # first is between manifest and field
    top_margin, bottom_margin = 0, 0
    spare = 30 - ((n + 1) * (h + 2) + sum(gaps))
    extras = list(map(int, rng.permutation(n + 2)))
    for index in extras:
        if spare and rng.random() < .55:
            if index == n:
                top_margin = 1
            elif index == n + 1:
                bottom_margin = 1
            else:
                gaps[index] += 1
            spare -= 1
    top = top_margin + h + 2 + gaps[0]
    lane_tops = [top]
    for i in range(1, n):
        lane_tops.append(lane_tops[-1] + h + 2 + gaps[i])
    cars = [dict(lane=i, top=lane_tops[i], left=starts[i], remaining=fuels[i],
                 advanced=capacity - fuels[i]) for i in range(n)]
    # Sometimes put the selected cars ahead of their decoys. A common shift
    # preserves the selected motion schedule. Decoys can then sit behind,
    # ahead, or on different sides in different lanes, as free space permits.
    selected_extent = max(v['left'] + v['remaining'] + width + 1 for v in cars)
    max_shift = 30 - selected_extent
    min_shift = width + 2 - min(starts)
    if basekind == 'success' and not long_trip:
        min_shift += planned_tick + 1  # room for an early-stopped moving decoy
    if max_shift >= min_shift and rng.random() < .55:
        shift = int(rng.integers(min_shift, max_shift + 1))
        for car in cars:
            car['left'] += shift
    available = []
    for i, car in enumerate(cars):
        options = []
        ahead = car['left'] + car['remaining'] + width + 2
        if ahead + width < 30:
            options.append((ahead, min(capacity, 29 - ahead - width)))
        behind_capacity = car['left'] - width - 2
        if behind_capacity >= 0:
            limit = min(2, behind_capacity)
            if basekind == 'success' and not long_trip:
                limit = min(limit, max(0, behind_capacity - planned_tick - 1))
            behind = int(rng.integers(limit + 1))
            options.append((behind, min(capacity, behind_capacity - behind)))
        if options:
            available.append((i, options))
    _need(available, "no room for an unselected token")
    rng.shuffle(available)
    if kind == 'same_lane':
        # Any lane with a decoy supports an explicitly wrong-lane selection.
        decoy_count = max(1, int(rng.integers(1, len(available) + 1)))
    elif not long_trip:
        decoy_count = int(rng.integers(2, len(available) + 1))
    else:
        decoy_count = int(rng.integers(1, len(available) + 1))
    surplus_given = False
    for lane, options in available[:decoy_count]:
        surplus_options = [(left, upper) for left, upper in options
                           if basekind == 'success' and upper > planned_tick]
        choices = surplus_options if not surplus_given and surplus_options else options
        left, upper = choices[int(rng.integers(len(choices)))]
        low = planned_tick + 1 if choices is surplus_options else 0
        rem = int(rng.integers(low, upper + 1))
        surplus_given |= basekind == 'success' and rem > planned_tick
        cars.append(dict(lane=lane, top=lane_tops[lane], left=left,
                         remaining=rem, advanced=capacity - rem))
    tokens = _tokens(rng, len(cars), h, gw, parameters['inks'],
                     (n, cars[n]['lane'], cars[n + 1]['lane']) if decoy_count >= 2 else None)
    for car, token in zip(cars, tokens):
        car['token'] = token.copy()
    selected = list(range(n))
    if kind == 'partial':
        selected = list(map(int, rng.choice(n, int(rng.integers(1, n)), replace=False)))
    elif kind == 'same_lane':
        decoy = n
        omitted = next(i for i in range(n) if i != cars[decoy]['lane'])
        selected.remove(omitted)
        selected.append(decoy)
    rng.shuffle(selected)
    slots = sorted(map(int, rng.choice(n, len(selected), replace=False)))
    rack_width = n * (gw + 1) + 1
    required = max(rack_width, max(c['left'] + c['remaining'] + width + 1 for c in cars))
    W = int(rng.integers(required, min(30, required + 3) + 1))
    rack_left = int(rng.integers(W - rack_width + 1))
    H = lane_tops[-1] + h + 2 + bottom_margin
    inp = np.zeros((H, W), np.int8)
    rack = inp[top_margin:top_margin + h + 2, rack_left:rack_left + rack_width]
    rack[[0, -1]], rack[:, ::gw + 1] = frame, frame
    for car in cars:
        _paint_car(inp, car, frame, spent, fuel, h, gw, cw)

    def render(query):
        source = inp.copy()
        for slot, selected_car in zip(slots, query):
            start = rack_left + slot * (gw + 1) + 1
            source[top_margin + 1:top_margin + h + 1, start:start + gw] = cars[selected_car]['token']
        moves, _, _ = _unit_clock(cars, query, n, width)
        out = source.copy()
        for car in cars:
            out[car['top']:car['top'] + h + 2, car['left']:car['left'] + width + 1] = 0
        for car, move in zip(cars, moves):
            moved = dict(car, left=car['left'] + move, advanced=car['advanced'] + move)
            _paint_car(out, moved, frame, spent, fuel, h, gw, cw)
        turn = parameters['orientation']
        return dict(input=np.rot90(source, turn).copy(), output=np.rot90(out, turn).copy())

    pair = render(selected)
    if contrasts:
        _need(kind == 'success' and decoy_count >= 2, 'contrast needs a full success and two decoys')
        alternatives = [render([decoy if i == cars[decoy]['lane'] else i for i in selected])
                        for decoy in (n, n + 1)]
        return pair, alternatives
    return pair


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(name + ' outside its permitted range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    seed = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(np.arange(1, 10))))
    parameters = dict(n=int(rng.integers(2, 5)), h=int(rng.integers(2, 4)),
                      gw=int(rng.integers(2, 4)), frame=palette[0], spent=palette[1],
                      fuel=palette[2], inks=palette[3:3 + int(rng.integers(2, 7))],
                      orientation=int(rng.integers(4)))
    # The first lessons share ALL vehicle pixels. Only one manifest slot
    # changes, retaining n matches and one match per lane. Thus no fixed car
    # rank, query-count, or query-independent geometric shortcut fits training.
    anchor, alternatives = _make_pair(rng, parameters, 'success', 2, contrasts=True)
    rng.shuffle(alternatives)
    training = [anchor, alternatives[0]]
    if nt >= 3:
        training.append(alternatives[1])
    optional = ['partial', 'already', 'same_lane']
    rng.shuffle(optional)
    lessons = optional[:max(0, nt - len(training))]
    while len(training) + len(lessons) < nt:
        lessons.append(str(rng.choice(['success', 'unreachable', 'already', 'partial', 'same_lane'])))
    taught = {'success', 'unreachable'} | set(lessons)
    tests = [str(rng.choice(sorted(taught))) for _ in range(nq)]
    puzzle = dict(train=training, test=[])
    seen = {_key(p['input']) for p in training}
    for split, kinds in (('train', lessons), ('test', tests)):
        for index, kind in enumerate(kinds):
            cw = 3 if split == 'test' else (1 if index == 0 else int(rng.integers(1, 3)))
            for _ in range(32):
                pair = _make_pair(rng, parameters, kind, cw, long_trip=split == 'test')
                signature = _key(pair['input'])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded distinct-input construction failed')
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    tick, success = _event(info)
    cars, selected = info['cars'], info['queries']
    covered = {cars[i]['lane'] for i in selected}
    masks = [_key(v['token'] != 0) for v in cars]
    inks = [int(v['token'][v['token'] != 0][0]) for v in cars]
    return dict(tick=tick, success=success, full=len(selected) == info['n'],
                covered=len(covered) == info['n'], count=len(cars), n=info['n'],
                capacity=info['h'] * info['cw'], counter=(info['h'], info['cw']),
                fresh=any(v['advanced'] == 0 for v in cars),
                exhausted=any(v['remaining'] == 0 for v in cars),
                selected_surplus=success and any(cars[i]['remaining'] > tick for i in selected),
                decoy_surplus=success and any(v['remaining'] > tick for i, v in enumerate(cars) if i not in selected),
                color_relevant=any(masks[i] == masks[j] and inks[i] != inks[j]
                                   for i in selected for j in range(len(cars)) if j not in selected),
                shape_relevant=any(inks[i] == inks[j] and masks[i] != masks[j]
                                   for i in selected for j in range(len(cars)) if j not in selected),
                selected_ranks=tuple(sorted(sum(v['lane'] == cars[i]['lane'] and v['left'] < cars[i]['left']
                                                for v in cars) for i in selected)),
                ordered=selected == sorted(selected), orientation=info['turn'],
                shape=info['grid'].shape, palette=(info['frame'], info['spent'], info['fuel']))


def _lookup_contrasts(infos):
    """Recover manifest-only, full-lane opposite-outcome lessons from pixels."""
    found = set()
    for a, b in itertools.combinations(infos, 2):
        if a['grid'].shape != b['grid'].shape or a['rack'] != b['rack']:
            continue
        if a['turn'] != b['turn'] or a['n'] != b['n']:
            continue
        ea, eb = _event(a), _event(b)
        if ea[1] == eb[1] or min(ea[0], eb[0]) == 0:
            continue
        if any(len(i['queries']) != i['n'] or
               len({i['cars'][q]['lane'] for q in i['queries']}) != i['n'] for i in (a, b)):
            continue
        rack = a['rack']
        field_a, field_b = a['grid'].copy(), b['grid'].copy()
        rs = slice(rack['top'] + 1, rack['bottom'])
        cs = slice(rack['left'] + 1, rack['right'])
        field_a[rs, cs], field_b[rs, cs] = 0, 0
        if not np.array_equal(field_a, field_b):
            continue
        changes = []
        for u, v in zip(rack['edges'], rack['edges'][1:]):
            cols = slice(rack['left'] + u + 1, rack['left'] + v)
            aa, bb = a['grid'][rs, cols], b['grid'][rs, cols]
            if not np.array_equal(aa, bb):
                changes.append((aa, bb))
        if len(changes) == 1:
            aa, bb = changes[0]
            if np.array_equal(aa != 0, bb != 0):
                found.add('ink')
            elif int(aa[aa != 0][0]) == int(bb[bb != 0][0]):
                found.add('shape')
    return found


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "bad puzzle schema")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal pair counts")
    seen, conventions, fuel_colors, token_inks = set(), set(), set(), set()
    records = dict(train=[], test=[])
    train_infos = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "bad pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(inp.shape == out.shape, "dimensions changed")
            _need(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _parse(inp)
            if split == 'train':
                train_infos.append(info)
            conventions.add((info['frame'], info['spent'], info['n'], info['h'], info['gw'], info['turn']))
            if info['fuel'] is not None:
                fuel_colors.add(info['fuel'])
            token_inks.update(int(v['token'][v['token'] != 0][0]) for v in info['cars'])
            cars = info['cars']
            moves, tick, complete = _unit_clock(cars, info['queries'], info['n'], cars[0]['width'])
            analytic_tick, analytic_complete = _event(info)
            _need((tick, complete) == (analytic_tick, analytic_complete), "event timing disagrees")
            _need(np.array_equal(_output_at(info, moves), out), "wrong unit-clock output")
            _need(np.array_equal(solve(inp), out), "input-only solve disagrees")
            canonical_out = np.rot90(out, info['turn'])
            rack = info['rack']
            rs, cs = slice(rack['top'], rack['bottom'] + 1), slice(rack['left'], rack['right'] + 1)
            _need(np.array_equal(info['grid'][rs, cs], canonical_out[rs, cs]), "manifest changed")
            # Recover each moved glyph/counter directly at its computed destination.
            for v, movement in zip(cars, moves):
                t, l, cw = v['top'], v['left'] + movement, v['cw']
                token = canonical_out[t + 1:t + info['h'] + 1,
                                      l + cw + 2:l + cw + 2 + info['gw']]
                _need(np.array_equal(token, v['token']), "carried glyph changed")
                counter = canonical_out[t + 1:t + info['h'] + 1, l + 1:l + cw + 1]
                _need(np.count_nonzero(counter == info['spent']) - v['advanced'] == movement,
                      "one fuel cell must pay for one step")
            records[split].append(_features(info))
    _need(len(conventions) == 1, "puzzle-wide conventions changed")
    _need(len(fuel_colors) == 1 and not (fuel_colors & token_inks), "puzzle-wide unused color changed")
    train, test = records['train'], records['test']
    witnesses = [r for r in train if r['success'] and r['tick'] > 0 and r['selected_surplus']]
    _need(witnesses and any(r['decoy_surplus'] for r in witnesses), "missing early global stop lesson")
    _need(any(not r['success'] and r['full'] and r['covered'] and r['tick'] > 0 for r in train),
          "missing full-manifest unreachable lesson")
    _need(any(r['color_relevant'] and r['shape_relevant'] for r in train), "exact colored lookup untaught")
    contrasts = _lookup_contrasts(train_infos)
    _need(contrasts, "missing manifest-only opposite-outcome training contrast")
    if nt >= 3:
        _need(contrasts == {'ink', 'shape'}, "both exact token cues need isolated contrasts")
    if any(not r['full'] for r in test):
        _need(any(not r['full'] for r in train), "partial manifest not taught")
    if any(r['full'] and not r['covered'] for r in test):
        _need(any(r['full'] and not r['covered'] for r in train), "wrong-lane coverage not taught")
    if any(r['success'] and r['tick'] == 0 for r in test):
        _need(any(r['success'] and r['tick'] == 0 for r in train), "initial alignment not taught")
    _need(min(r['capacity'] for r in test) > max(r['capacity'] for r in train),
          "test must extend the witnessed counter capacity")


def self_test():
    """Bounded semantic, metamorphic, corruption and 42-seed variation checks."""
    def probe(leader_fuel=0, use_decoy=False, exhausted=False):
        # A controlled contrast: changing one counter unit shifts rendezvous
        # from tick 2 to tick 3; changing only query ink selects a distant decoy.
        h, gw, cw, frame, spent, fuel = 2, 2, 2, 5, 8, 9
        glyph = np.array([[1, 0], [1, 1]], np.int8)
        other = np.ones((2, 2), np.int8)
        decoy = np.where(glyph != 0, 2, 0).astype(np.int8)
        cars = [dict(top=6, left=0, token=glyph, advanced=0),
                dict(top=11, left=2, token=other, advanced=4 - leader_fuel),
                dict(top=6, left=16, token=decoy, advanced=1)]
        if exhausted:
            for v in cars:
                v['advanced'] = 4
        g = np.zeros((15, 27), np.int8)
        g[:4, :7] = frame
        g[1:3, 1:3] = decoy if use_decoy else glyph
        g[1:3, 4:6] = other
        for v in cars:
            _paint_car(g, v, frame, spent, fuel, h, gw, cw)
        return g

    first, changed_fuel, changed_query = probe(), probe(1), probe(use_decoy=True)
    _need(_event(_parse(first)) == (2, True), "controlled rendezvous timing")
    _need(_event(_parse(changed_fuel)) == (3, True), "counter-sensitive timing")
    _need(_event(_parse(changed_query)) == (4, False), "exact ink lookup must change the event")
    _need(not np.array_equal(solve(first), solve(changed_fuel)), "A counter contrast unchanged")
    _need(not np.array_equal(solve(first), solve(changed_query)), "B query contrast unchanged")
    empty_fuel = probe(exhausted=True)
    _need(np.array_equal(solve(empty_fuel), empty_fuel), "absent fuel color / exhausted scene")
    remap = np.array([0, 7, 4, 1, 9, 2, 5, 3, 6, 8], np.int8)
    _need(np.array_equal(solve(remap[first]), remap[solve(first)]), "visible palette role equivariance")
    features, signatures, palettes, contrast_types = [], set(), set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        repeated = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        contrast_types.append(_lookup_contrasts([_parse(p['input']) for p in puzzle['train']]))
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair, repeat in zip(puzzle[split], repeated[split]):
                _need(all(np.array_equal(pair[k], repeat[k]) for k in pair), "seed nondeterminism")
                info = _parse(pair['input'])
                feature = _features(info)
                feature['split'] = split
                features.append(feature)
                palettes.add(feature['palette'])
                for turn in range(4):
                    _need(np.array_equal(solve(np.rot90(pair['input'], turn)), np.rot90(pair['output'], turn)),
                          "quarter-turn motion/counter equivariance failed")
                if split == 'test':
                    # Permute whole manifest slots, preserving the displayed
                    # glyphs and their emptiness. The moving field is identical.
                    rack = info['rack']
                    modified = info['grid'].copy()
                    expected = np.rot90(pair['output'], info['turn']).copy()
                    rs = slice(rack['top'] + 1, rack['bottom'])
                    slots = [slice(rack['left'] + a + 1, rack['left'] + b)
                             for a, b in zip(rack['edges'], rack['edges'][1:])]
                    for dest, source in zip(slots, reversed(slots)):
                        modified[rs, dest] = info['grid'][rs, source]
                        expected[rs, dest] = info['grid'][rs, source]
                    _need(np.array_equal(solve(modified), expected), 'manifest-order dependence')
                # Parent ablations on every active success lesson.
                tick, success = _event(info)
                if success and tick > 0 and feature['selected_surplus']:
                    no_matching = _output_at(info, [v['remaining'] for v in info['cars']])
                    _need(not np.array_equal(no_matching, pair['output']), "ignoring B leaves the answer unchanged")
                    _need(any(v['remaining'] < tick for v in info['cars']),
                          "ignoring finite fuel cannot change relative alignment")
    _need(len(signatures) == 42 and len(palettes) >= 20, "cross-seed puzzle/palette diversity collapsed")
    _need(all(v == {'ink', 'shape'} for v in contrast_types), "isolated lookup contrasts collapsed")
    success_ranks = {rank for r in features if r['success'] and r['tick'] > 0 for rank in r['selected_ranks']}
    _need(success_ranks == {0, 1}, "selected vehicle position is a shortcut")
    _need({r['n'] for r in features} == {2, 3, 4}, "lane-count diversity")
    _need({r['orientation'] for r in features} == {0, 1, 2, 3}, "orientation diversity")
    _need({r['counter'] for r in features} == set(itertools.product((2, 3), (1, 2, 3))), "counter geometry diversity")
    _need(len({r['shape'] for r in features}) >= 25, "canvas shape diversity")
    _need(len({r['count'] for r in features}) >= 5, "vehicle-count diversity")
    _need({r['success'] for r in features if r['split'] == 'test'} == {True, False}, "test outcomes collapsed")
    _need(any(r['success'] and r['tick'] >= 5 for r in features if r['split'] == 'test'), "long rendezvous shift absent")
    _need(any(r['fresh'] for r in features) and any(r['exhausted'] for r in features), "counter endpoints absent")
    _need(any(not r['full'] for r in features) and any(r['success'] and r['tick'] == 0 for r in features),
          "partial/already-aligned cases absent")
    baseline = generate_puzzle(0)
    spent_test = {s: [{k: v.copy() for k, v in pair.items()} for pair in baseline[s]] for s in baseline}
    pair = spent_test['test'][0]
    info = _parse(pair['input'])
    canonical = np.rot90(pair['input'], info['turn'])
    for v in info['cars']:
        canonical[v['top'] + 1:v['bottom'], v['left'] + 1:v['left'] + 1 + v['cw']] = info['spent']
    pair['output'] = solve(pair['input'])
    validate_generated_puzzle(spent_test)
    # Semantic corruption and repeated input must be rejected.
    for defect in ('output', 'duplicate', 'counter'):
        broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in baseline[s]] for s in baseline}
        if defect == 'duplicate':
            broken['train'][1] = {k: v.copy() for k, v in broken['train'][0].items()}
        elif defect == 'output':
            g = broken['test'][0]['output']
            g[0, 0] = (int(g[0, 0]) + 1) % 10
        else:
            pair = broken['train'][0]
            info = _parse(pair['input'])
            canonical = np.rot90(pair['input'], info['turn'])
            v = info['cars'][0]
            canonical[v['top'] + 1, v['left'] + 1] = info['frame']
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError('validator accepted ' + defect)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(100 + nt * 10 + nq, nt, nq)
            _need(len(p['train']) == nt and len(p['test']) == nq, "explicit counts ignored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return dict(seeds=42, distinct_puzzles=len(signatures), palettes=len(palettes),
                puzzles_with_both_lookup_contrasts=sum(v == {'ink', 'shape'} for v in contrast_types),
                active_selected_ranks=sorted(success_ranks),
                canonical_shapes=len({r['shape'] for r in features}),
                vehicle_counts=sorted({r['count'] for r in features}),
                lanes=sorted({r['n'] for r in features}), orientations=sorted({r['orientation'] for r in features}),
                counter_shapes=sorted({r['counter'] for r in features}),
                test_outcomes={str(v): sum(r['success'] == v for r in features if r['split'] == 'test') for v in (False, True)},
                max_test_rendezvous=max(r['tick'] for r in features if r['split'] == 'test' and r['success']))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
