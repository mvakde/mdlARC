"""ARC family: rolling stencil printers with fuel charged for new pigment.

No parent code is imported.  Geometry and a visible cartridge supply everything
needed by solve; generation uses a separate, canonical boolean-canvas renderer.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np


BACKGROUND, SPENT, FUEL = 0, 1, 2
_DIRECTIONS = ("R", "U", "L", "D")
_DELTAS = ((0, 1), (-1, 0), (0, -1), (1, 0))


def _need(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _need(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "invalid dimensions")
    _need(np.issubdtype(g.dtype, np.integer), "noninteger grid")
    _need(np.all((g >= 0) & (g <= 9)), "invalid colors")
    return g


def _key(g):
    return g.shape, tuple(map(int, g.flat))


def _rotate_vector(r, c, turn):
    return ((r, c), (-c, r), (-r, -c), (c, -r))[turn]


def _position(r, c, height, width, turn, top, left):
    y, x = ((r, c), (width - 1 - c, r),
            (height - 1 - r, width - 1 - c), (c, height - 1 - r))[turn]
    return top + y, left + x


def _components(mask, diagonal=False):
    todo = set(map(tuple, np.argwhere(mask)))
    result = []
    neighbors = ([(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                  if r or c] if diagonal else [(-1, 0), (1, 0), (0, -1), (0, 1)])
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, cells = [start], []
        while stack:
            p = stack.pop()
            cells.append(p)
            for dr, dc in neighbors:
                q = (p[0] + dr, p[1] + dc)
                if q in todo:
                    todo.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _read_cartridge(g, cells, frame):
    """Recognize a divided rectangular frame, including its orientation."""
    ys, xs = zip(*cells)
    top, bottom, left, right = min(ys), max(ys), min(xs), max(xs)
    block = g[top:bottom + 1, left:right + 1]
    candidates = []
    for turn in range(4):
        a = np.rot90(block, -turn)
        height, width = a.shape
        if height < 7 or width < 4:
            continue
        if not (np.all(a[0] == frame) and np.all(a[-1] == frame)
                and np.all(a[:, 0] == frame) and np.all(a[:, -1] == frame)):
            continue
        dividers = [r for r in range(2, height - 2) if np.all(a[r] == frame)]
        if len(dividers) != 1:
            continue
        divider = dividers[0]
        expected_frame = np.zeros(a.shape, bool)
        expected_frame[0] = expected_frame[-1] = True
        expected_frame[:, 0] = expected_frame[:, -1] = True
        expected_frame[divider] = True
        if not np.array_equal(a == frame, expected_frame):
            continue
        stencil = a[1:divider, 1:-1]
        fuel = a[divider + 1:-1, 1:-1]
        if stencil.shape[0] < 3 or not np.all(stencil[0] == BACKGROUND):
            continue
        inks = set(map(int, np.unique(stencil))) - {BACKGROUND}
        if len(inks) != 1 or not np.all(np.isin(fuel, [SPENT, FUEL])):
            continue
        ink = next(iter(inks))
        if ink in (SPENT, FUEL, frame):
            continue
        mask = stencil == ink
        # One blank guide row, then a tight connected pigment motif.
        motif = mask[1:]
        if not (motif[0].any() and motif[-1].any()
                and motif[:, 0].any() and motif[:, -1].any()
                and len(_components(motif, diagonal=True)) == 1):
            continue
        origin = _position(1, width, height, width, turn, top, left)
        if not (0 <= origin[0] < g.shape[0] and 0 <= origin[1] < g.shape[1]):
            continue
        code = int(g[origin])
        if code in (BACKGROUND, SPENT, FUEL, frame, ink):
            continue
        dy, dx = _DELTAS[turn]
        gate = None
        for distance in range(1, 31):
            p = origin[0] + distance * dy, origin[1] + distance * dx
            if not (0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1]):
                break
            if g[p] != BACKGROUND:
                if g[p] == code:
                    gate = p
                break
        if gate is None:
            continue
        counter = [_position(r, c, height, width, turn, top, left)
                   for c in range(width - 2, 0, -1)
                   for r in range(divider + 1, height - 1)]
        values = [int(g[p]) for p in counter]
        advanced = values.count(SPENT)
        if values != [SPENT] * advanced + [FUEL] * (len(values) - advanced):
            continue
        candidates.append(dict(top=top, bottom=bottom, left=left, right=right,
                               frame=frame, ink=ink, code=code, mask=mask.copy(),
                               turn=turn, origin=origin, gate=gate, distance=distance,
                               counter=counter, advanced=advanced,
                               remaining=len(values) - advanced))
    _need(len(candidates) <= 1, "ambiguous cartridge orientation")
    return candidates[0] if candidates else None


def _pixels(v, step, shape):
    dy, dx = _DELTAS[v['turn']]
    y, x = v['origin'][0] + step * dy, v['origin'][1] + step * dx
    result = set()
    for r, c in np.argwhere(v['mask']):
        dr, dc = _rotate_vector(int(r), int(c), v['turn'])
        q = y + dr, x + dc
        if 0 <= q[0] < shape[0] and 0 <= q[1] < shape[1]:
            result.add(q)
    return result


def _analyze(input_grid):
    g = _grid(input_grid)
    vehicles = []
    for frame in map(int, np.unique(g)):
        if frame in (BACKGROUND, SPENT, FUEL):
            continue
        for cells in _components(g == frame):
            v = _read_cartridge(g, cells, frame)
            if v is not None:
                vehicles.append(v)
    _need(bool(vehicles), "no readable printers")
    cover, reservations = np.zeros(g.shape, int), np.zeros(g.shape, int)
    for v in vehicles:
        t, b, l, r = v['top'], v['bottom'], v['left'], v['right']
        cover[t:b + 1, l:r + 1] += 1
        cover[v['origin']] += 1
        cover[v['gate']] += 1
        # Reserve the full possible journey, irrespective of available fuel.
        possible = {v['origin'], v['gate'], (t, l), (b, r)}
        for k in range(1, v['distance'] + 1):
            possible.update(_pixels(v, k, g.shape))
        ys, xs = zip(*possible)
        bounds = min(ys), max(ys), min(xs), max(xs)
        y0, y1, x0, x1 = bounds
        reservations[y0:y1 + 1, x0:x1 + 1] += 1
        v['bounds'] = bounds
    _need(np.max(cover) == 1, "shared cartridge or marker")
    _need(np.max(reservations) == 1, "overlapping printer corridors")
    _need(not np.any((g != BACKGROUND) & (cover == 0)), "unexplained input material")
    return g, sorted(vehicles, key=lambda v: (v['top'], v['left']))


def _journey(v, shape, charging='new'):
    painted, used, movement, costs = set(), 0, 0, []
    budget = v['remaining']
    for k in range(1, v['distance'] + 1):
        pixels = _pixels(v, k, shape)
        cost = (len(pixels - painted) if charging == 'new' else
                len(pixels) if charging == 'whole' else 1 if charging == 'unit' else 0)
        costs.append(cost)
        if used + cost > budget:
            break
        painted.update(pixels)
        used += cost
        movement = k
    return dict(painted=painted, used=used, movement=movement,
                costs=costs, residual=budget - used,
                stopped='destination' if movement == v['distance'] else 'fuel')


def _predict(g, vehicles, charging='new'):
    out = g.copy()
    for v in vehicles:
        out[v['origin']] = out[v['gate']] = BACKGROUND
        j = _journey(v, g.shape, charging)
        for p in j['painted']:
            out[p] = v['ink']
        for i, p in enumerate(v['counter']):
            out[p] = SPENT if i < v['advanced'] + j['used'] else FUEL
        dy, dx = _DELTAS[v['turn']]
        final = (v['origin'][0] + j['movement'] * dy,
                 v['origin'][1] + j['movement'] * dx)
        out[final] = v['code']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using only the input, including the visible stencil and fuel."""
    g, vehicles = _analyze(input_grid)
    return _predict(g, vehicles)


def _sample_mask(rng):
    for _ in range(100):
        h, w = int(rng.integers(2, 5)), int(rng.integers(2, 6))
        kind = int(rng.integers(5))
        m = np.zeros((h, w), bool)
        if kind == 0:
            m[0] = m[-1] = True
            m[:, 0] = m[:, -1] = True
        elif kind == 1:
            m[:] = True
        elif kind == 2:
            m[-1] = True
            m[:, -1] = True
            m |= rng.random((h, w)) < .35
        elif kind == 3:
            m[0] = m[-1] = True
            m[:, -1] = True
            m |= rng.random((h, w)) < .55
        else:
            m[:] = rng.random((h, w)) < .7
            m[0, -1] = m[-1, -1] = True
        if rng.random() < .5:
            m = m[::-1].copy()
        if (m.sum() >= 5 and m[0].any() and m[-1].any() and m[:, 0].any()
                and m[:, -1].sum() >= 2
                and np.count_nonzero(m[:, :-1] & m[:, 1:]) >= 2
                and len(_components(m, diagonal=True)) == 1):
            return np.vstack((np.zeros((1, w), bool), m))
    # Bounded fallback remains a normal family member.
    return np.array([[0, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1]], bool)


def _schedule(mask, distance, clip):
    """Generation-side boolean canvas, independently of grid parsing."""
    height, width = mask.shape
    canvas = np.zeros((height, distance + width - clip), bool)
    costs, snapshots = [], []
    for step in range(1, distance + 1):
        proposal = canvas.copy()
        visible = min(width, canvas.shape[1] - step)
        if visible > 0:
            proposal[:, step:step + visible] |= mask[:, :visible]
        costs.append(int(proposal.sum() - canvas.sum()))
        canvas = proposal
        snapshots.append(canvas.copy())
    return costs, snapshots


def _make_patch(rng, entry, frame, turn, lesson):
    code, ink, mask = entry
    width, pigment_height = mask.shape[1], mask.shape[0] - 1
    if lesson == 'arrival':
        distance, clip = 2, 1
    elif lesson == 'fuel':
        distance, clip = int(rng.integers(4, 7)), 0
    elif lesson == 'reveal':
        distance, clip = int(rng.integers(1, 3)), 0
    elif lesson == 'long':
        distance = int(rng.integers(7, 11))
        clip = int(rng.integers(0, width)) if rng.random() < .5 else 0
    else:
        distance = int(rng.integers(1, 7))
        clip = int(rng.integers(0, width)) if rng.random() < .25 else 0
    costs, snapshots = _schedule(mask, distance, clip)
    cumulative = np.cumsum(costs)
    if lesson == 'arrival':
        remaining = int(cumulative[-1]) + 1
    elif lesson == 'fuel':
        remaining = int(cumulative[1]) + 1
        _need(costs[2] >= 2, "fuel lesson needs unaffordable next impression")
    elif lesson == 'reveal':
        remaining = int(cumulative[-1]) + int(rng.integers(0, 3))
    elif lesson == 'long':
        if rng.random() < .45:
            remaining = int(cumulative[-1]) + int(rng.integers(0, 4))
        else:
            stop = int(rng.integers(3, min(6, distance - 1) + 1))
            next_cost = costs[stop]
            remaining = int(cumulative[stop - 1]) + int(rng.integers(max(1, next_cost)))
    else:
        mode = int(rng.integers(5))
        if mode == 0:
            remaining = 0
        elif mode == 1:
            remaining = max(0, costs[0] - 1)
        elif mode == 2:
            remaining = int(cumulative[-1])
        else:
            remaining = int(rng.integers(0, int(cumulative[-1]) + 4))
    max_fuel_rows = 26 - pigment_height
    remaining = min(remaining, max_fuel_rows * width)
    fuel_rows = min(max_fuel_rows, max(1, (remaining + width - 1) // width)
                    + int(rng.integers(0, 2)))
    capacity = fuel_rows * width
    advanced = capacity - remaining
    body_height, body_width = pigment_height + fuel_rows + 4, width + 2
    total_width = body_width + distance + width - clip
    _need(body_height <= 30 and total_width <= 30, "oversized patch")
    inp = np.zeros((body_height, total_width), dtype=np.int8)
    inp[:, :body_width] = frame
    divider = mask.shape[0] + 1
    inp[1:divider, 1:body_width - 1] = np.where(mask, ink, BACKGROUND)
    counter = [(r, c) for c in range(body_width - 2, 0, -1)
               for r in range(divider + 1, body_height - 1)]
    for i, p in enumerate(counter):
        inp[p] = SPENT if i < advanced else FUEL
    origin, gate = (1, body_width), (1, body_width + distance)
    inp[origin] = inp[gate] = code
    affordable = np.flatnonzero(cumulative <= remaining)
    movement = int(affordable[-1]) + 1 if len(affordable) else 0
    used = int(cumulative[movement - 1]) if movement else 0
    out = inp.copy()
    out[origin] = out[gate] = BACKGROUND
    if movement:
        printed = snapshots[movement - 1]
        target = out[1:1 + mask.shape[0], body_width:]
        target[printed] = ink
    for i, p in enumerate(counter):
        out[p] = SPENT if i < advanced + used else FUEL
    out[1, body_width + movement] = code
    return np.rot90(inp, turn).copy(), np.rot90(out, turn).copy(), clip > 0


def _free_positions(occupied, shape):
    H, W = occupied.shape
    h, w = shape
    if h > H or w > W:
        return np.empty((0, 2), int)
    p = np.pad(occupied, 1)
    blocked = np.logical_or.reduce([p[r:r + H, c:c + W]
                                    for r in range(3) for c in range(3)])
    s = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = s[h:, w:] - s[:-h, w:] - s[h:, :-w] + s[:-h, :-w]
    return np.argwhere(sums == 0)


def _pair(rng, book, frame, descriptions):
    for attempt in range(80):
        patches = [_make_patch(rng, book[index], frame, turn, lesson)
                   for index, turn, lesson in descriptions]
        if attempt >= 35:
            H = W = 30
        else:
            H, W = int(rng.integers(15, 31)), int(rng.integers(15, 31))
            H = max(H, max(p[0].shape[0] for p in patches))
            W = max(W, max(p[0].shape[1] for p in patches))
        inp, out = np.zeros((H, W), np.int8), np.zeros((H, W), np.int8)
        occupied = np.zeros((H, W), bool)
        okay = True
        for (index, turn, lesson), (a, b, clipped) in zip(descriptions, patches):
            h, w = a.shape
            positions = _free_positions(occupied, a.shape)
            if clipped:
                # Local clipping must coincide with the actual canvas boundary.
                axis, edge = ((1, W - w), (0, 0), (1, 0), (0, H - h))[turn]
                positions = positions[positions[:, axis] == edge]
            elif len(positions) and rng.random() < .25:
                edges = positions[(positions[:, 0] == 0) | (positions[:, 1] == 0)
                                  | (positions[:, 0] == H - h) | (positions[:, 1] == W - w)]
                if len(edges):
                    positions = edges
            if len(positions) == 0:
                okay = False
                break
            y, x = map(int, positions[int(rng.integers(len(positions)))])
            inp[y:y + h, x:x + w] = a
            out[y:y + h, x:x + w] = b
            occupied[y:y + h, x:x + w] = True
        if okay:
            return dict(input=inp, output=out)
    raise RuntimeError("bounded printer packing failed")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    count = int(rng.integers(1, 4))
    palette = list(map(int, rng.permutation(np.arange(3, 10))))
    frame = palette[0]
    book, masks = [], set()
    for i in range(count):
        for _ in range(100):
            mask = _sample_mask(rng)
            if _key(mask) not in masks:
                break
        else:
            raise RuntimeError("could not obtain distinct stencils")
        masks.add(_key(mask))
        book.append((palette[1 + i], palette[1 + count + i], mask))
    # Two shuffled examples carry the essential lessons, even for 2/1 counts.
    order = list(map(int, rng.permutation(nt)))
    turns = list(map(int, rng.permutation(4)))
    train = [[] for _ in range(nt)]
    train[order[0]].append((0, turns[0], 'arrival'))
    train[order[1]].append((0, turns[1], 'fuel'))
    for i in range(1, count):
        train[order[(i - 1) % 2]].append((i, int(rng.integers(4)), 'reveal'))
    for i in range(nt):
        if not train[i]:
            number = int(rng.integers(1, 4))
            train[i] = [(int(rng.integers(count)), int(rng.integers(4)), 'ordinary')
                        for _ in range(number)]
    test = []
    for i in range(nq):
        descriptions = [(int(rng.integers(count)), int(rng.integers(4)), 'long')]
        if rng.random() < .5:
            descriptions.append((int(rng.integers(count)), int(rng.integers(4)), 'ordinary'))
        test.append(descriptions)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, plans in (('train', train), ('test', test)):
        for descriptions in plans:
            for retry in range(30):
                # Optional third printers may be dropped after packing failures;
                # required lessons and dictionary exposures are never dropped.
                try:
                    pair = _pair(rng, book, frame, descriptions)
                except RuntimeError:
                    if all(d[2] == 'ordinary' for d in descriptions) and len(descriptions) > 1:
                        descriptions = descriptions[:-1]
                        continue
                    if descriptions[0][2] == 'long' and len(descriptions) > 1:
                        descriptions = descriptions[:1]
                        continue
                    raise
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("could not construct distinct inputs")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle schema")
    _need(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "invalid splits")
    _need(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
          and len(puzzle['train']) > len(puzzle['test']), "invalid counts")
    seen, dictionary, frames = set(), {}, set()
    demonstrated, tested, lessons, active_turns = set(), set(), set(), set()
    shortcut_failure = {'whole': False, 'unit': False, 'free': False}
    clipping_taught, longer_test = False, False
    max_train_distance = 0
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(inp.shape == out.shape, "dimensions changed")
            _need(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            g, vehicles = _analyze(inp)
            _need(np.array_equal(_predict(g, vehicles), out), "incorrect input-derived output")
            added_pigment = 0
            used_fuel = 0
            for v in vehicles:
                frames.add(v['frame'])
                meaning = v['ink'], _key(v['mask'])
                _need(v['code'] not in dictionary or dictionary[v['code']] == meaning,
                      "inconsistent code-to-stencil dictionary")
                dictionary[v['code']] = meaning
                j = _journey(v, inp.shape)
                # Pigment conservation and exact atomic stopping are measurable.
                _need(j['used'] == len(j['painted']), "fuel is not new pigment area")
                _need(j['used'] <= v['remaining'], "overspent fuel")
                if j['movement'] < v['distance']:
                    next_pixels = _pixels(v, j['movement'] + 1, inp.shape)
                    _need(len(next_pixels - j['painted']) > j['residual'], "premature stop")
                used_fuel += j['used']
                added_pigment += sum(inp[p] == BACKGROUND and out[p] == v['ink']
                                     for p in j['painted'])
                actual_spent = sum(out[p] == SPENT for p in v['counter']) - v['advanced']
                _need(actual_spent == j['used'], "counter charge differs from new pigment")
                counter_cells = set(v['counter'])
                _need(all(out[p] == inp[p] for p in
                          [(y, x) for y in range(v['top'], v['bottom'] + 1)
                           for x in range(v['left'], v['right'] + 1)
                           if (y, x) not in counter_cells]), "cartridge stencil or frame changed")
                dy, dx = _DELTAS[v['turn']]
                final = v['origin'][0] + j['movement'] * dy, v['origin'][1] + j['movement'] * dx
                _need(out[final] == v['code'], "missing final traveler")
                clipped = any(len(_pixels(v, k, inp.shape)) < int(v['mask'].sum())
                              for k in range(1, j['movement'] + 1))
                if split == 'train':
                    max_train_distance = max(max_train_distance, v['distance'])
                    if j['movement']:
                        demonstrated.add(v['code'])
                        active_turns.add(v['turn'])
                    if j['movement'] == v['distance'] and j['residual'] > 0:
                        lessons.add('destination')
                    if 0 < j['movement'] < v['distance'] and j['residual'] > 0:
                        lessons.add('atomic_fuel')
                    if j['movement'] >= 2:
                        whole_cost = sum(len(_pixels(v, k, inp.shape))
                                         for k in range(1, j['movement'] + 1))
                        if whole_cost > j['used']:
                            lessons.add('overlap')
                    clipping_taught |= clipped
                else:
                    tested.add(v['code'])
                    longer_test |= v['distance'] > max_train_distance
            _need(used_fuel == added_pigment, "new visible ink does not equal fuel spent")
            if split == 'train':
                for mode in shortcut_failure:
                    shortcut_failure[mode] |= not np.array_equal(_predict(g, vehicles, mode), out)
    _need(len(frames) == 1, "frame palette must be stable within a puzzle")
    _need(set(dictionary).isdisjoint({m[0] for m in dictionary.values()} | frames), "palette role collision")
    _need(tested <= demonstrated, "test code has no active training demonstration")
    _need({'destination', 'atomic_fuel', 'overlap'} <= lessons, "missing essential training lessons")
    _need(len(active_turns) >= 2, "training must expose rotation")
    _need(clipping_taught, "training must show a clipped impression")
    _need(longer_test, "missing longer-journey test shift")
    _need(all(shortcut_failure.values()), "training admits a parent-omitting shortcut")
    return True


def puzzle_features(puzzle):
    """Observable measurements, reconstructed from grids rather than RNG state."""
    records = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            inp, vehicles = _analyze(pair['input'])
            objects = []
            for v in vehicles:
                j = _journey(v, inp.shape)
                objects.append(dict(direction=_DIRECTIONS[v['turn']], distance=v['distance'],
                                    movement=j['movement'], capacity=len(v['counter']),
                                    advanced=v['advanced'], remaining=v['remaining'], used=j['used'],
                                    residual=j['residual'], stopped=j['stopped'], code=v['code'],
                                    ink=v['ink'], frame=v['frame'], mask_shape=v['mask'].shape,
                                    mask=''.join(str(int(x)) for x in v['mask'].flat),
                                    costs=j['costs'], clipped=any(
                                        len(_pixels(v, k, inp.shape)) < int(v['mask'].sum())
                                        for k in range(1, j['movement'] + 1))))
            records.append(dict(split=split, shape=inp.shape, count=len(vehicles), printers=objects))
    return records


def self_test():
    """Bounded 42-seed diversity, API, metamorphic, and corruption checks."""
    ring = np.array([[0, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1]], bool)
    inp, out, _ = _make_patch(np.random.default_rng(101), (3, 4, ring), 5, 0, 'fuel')
    _, vehicles = _analyze(inp)
    diagnostic = _journey(vehicles[0], inp.shape)
    _need(diagnostic['costs'][:3] == [8, 4, 3] and diagnostic['movement'] == 2
          and diagnostic['used'] == 12 and diagnostic['residual'] == 1,
          "ring overlap and interior-gap price regression")
    _need(np.array_equal(solve(inp), out), "ring label regression")
    solid = np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1]], bool)
    inp, _, _ = _make_patch(np.random.default_rng(102), (3, 4, solid), 5, 0, 'arrival')
    _, vehicles = _analyze(inp)
    v = vehicles[0]
    for i, p in enumerate(v['counter']):
        inp[p] = SPENT if i < len(v['counter']) - 6 else FUEL
    _, vehicles = _analyze(inp)
    diagnostic = _journey(vehicles[0], inp.shape)
    _need(diagnostic['costs'] == [6, 0] and diagnostic['movement'] == 2
          and diagnostic['used'] == 6 and diagnostic['residual'] == 0,
          "zero-cost clipped move after fuel exhaustion")
    result = solve(inp)
    _need(result[vehicles[0]['gate']] == 3
          and all(result[p] == SPENT for p in vehicles[0]['counter']),
          "free-step final point and counter regression")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records = [r for p in puzzles for r in puzzle_features(p)]
    objects = [v for r in records for v in r['printers']]
    _need(len({_key(p['train'][0]['input']) for p in puzzles}) == 42, "seed diversity")
    _need(len({r['shape'] for r in records}) >= 45, "canvas diversity")
    _need({r['count'] for r in records} >= {1, 2, 3}, "printer-count diversity")
    _need(len({(v['mask_shape'], v['mask']) for v in objects}) >= 25, "stencil diversity")
    _need(len({v['capacity'] for v in objects}) >= 15, "counter geometry diversity")
    _need(len({v['frame'] for v in objects}) == 7, "frame palette diversity")
    _need({v['direction'] for v in objects} == set(_DIRECTIONS), "orientation diversity")
    _need(any(v['remaining'] == 0 for v in objects), "no empty fuel cartridge")
    _need(any(0 < v['remaining'] < v['costs'][0] for v in objects), "no blocked first impression")
    _need(any(len(set(v['costs'])) >= 3 for v in objects), "no nonconstant overlap costs")
    _need(all(any(v['clipped'] and v['direction'] == d for v in objects)
              for d in _DIRECTIONS), "clipping at every oriented edge")
    for seed in (0, 17):
        again = generate_puzzle(seed)
        _need(all(np.array_equal(a[k], b[k]) for s in ('train', 'test')
                  for a, b in zip(puzzles[seed][s], again[s]) for k in ('input', 'output')),
              "nondeterministic generation")
    for pair in puzzles[0]['train'] + puzzles[0]['test']:
        original = pair['input'].copy()
        for turn in range(4):
            rotated = np.rot90(original, turn).copy()
            _need(np.array_equal(solve(rotated), np.rot90(pair['output'], turn)), "rotation equivariance")
        _need(np.array_equal(original, pair['input']), "solver changed its input")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(12345, nt, nq)
            _need(len(p['train']) == nt and len(p['test']) == nq, "count contract")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = copy.deepcopy(puzzles[0])
    query = broken['test'][0]
    _, vs = _analyze(query['input'])
    j = _journey(vs[0], query['input'].shape)
    _need(bool(j['painted']), "mutation fixture needs pigment")
    query['output'][next(iter(j['painted']))] = BACKGROUND
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted missing trail ink")
    metrics = dict(seeds=42, pairs=len(records), printers=len(objects),
                   shapes=len({r['shape'] for r in records}),
                   stencils=len({(v['mask_shape'], v['mask']) for v in objects}),
                   capacities=sorted({v['capacity'] for v in objects}),
                   counts=sorted({r['count'] for r in records}),
                   directions=sorted({v['direction'] for v in objects}),
                   clipped_printers=sum(v['clipped'] for v in objects),
                   note='Generator-owned checks; no independent or blind semantic review.')
    return metrics


if __name__ == '__main__':
    Path('scratch').mkdir(exist_ok=True)
    result = self_test()
    Path('scratch/self_test.json').write_text(json.dumps(result, separators=(',', ':')) + '\n')
    print(json.dumps(result, separators=(',', ':')))
