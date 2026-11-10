"""Wired paint reservoirs: conservative harvesting into attached empty area.

Only NumPy and the standard library are used. Generation writes no files.
All semantic inference in solve() comes from the single input array.
"""
from collections import deque
import hashlib
import json

import numpy as np


D4 = ((-1, 0), (0, -1), (0, 1), (1, 0))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _require(value, message):
    if not bool(value):
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "Invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "Grid cells must be integers")
    _require(np.all((0 <= a) & (a <= 9)), "Grid colors must be 0..9")
    return a


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _neighbors(cells, shape, steps=D8):
    h, w = shape
    return {(r + dr, c + dc) for r, c in cells for dr, dc in steps
            if 0 <= r + dr < h and 0 <= c + dc < w} - set(cells)


def _components(cells, steps=D4):
    left = set(cells)
    groups = []
    while left:
        start = min(left)
        left.remove(start)
        queue = [start]
        for r, c in queue:
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in left:
                    left.remove(p)
                    queue.append(p)
        groups.append(set(queue))
    return groups


def _distance(a, background, paints, footprint):
    traversable = (a == background) | np.isin(a, paints)
    distance = np.full(a.shape, -1, dtype=np.int16)
    queue = list(sorted(footprint))
    for p in queue:
        distance[p] = 0
    h, w = a.shape
    for r, c in queue:
        depth = int(distance[r, c]) + 1
        for dr, dc in D4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and traversable[rr, cc] and distance[rr, cc] < 0:
                distance[rr, cc] = depth
                queue.append((rr, cc))
    return distance


def _selected(candidates, distance, capacity):
    ordered = sorted(candidates, key=lambda p: (int(distance[p]), p))
    if 0 < capacity < len(ordered):
        _require(distance[ordered[capacity - 1]] < distance[ordered[capacity]],
                 "Ambiguous tie across capacity frontier")
    return set(ordered[:capacity])


def _analyze(value):
    """Reconstruct all roles, ownership, movement and storage from input only."""
    a = _grid(value)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    _require(sum(counts == counts[bg]) == 1, "Background must be uniquely dominant")
    pumps = []
    for r in range(a.shape[0] - 2):
        for c in range(a.shape[1] - 2):
            wire, paint = int(a[r, c]), int(a[r + 1, c + 1])
            if wire == bg or paint in (bg, wire):
                continue
            patch = a[r:r + 3, c:c + 3]
            if np.count_nonzero(patch == wire) == 8:
                footprint = {(r + i, c + j) for i in range(3) for j in range(3)}
                pumps.append(dict(center=(r + 1, c + 1), footprint=footprint,
                                  paint=paint, wire=wire))
    _require(1 <= len(pumps) <= 2, "Need one or two unique marked pumps")
    paints = [p['paint'] for p in pumps]
    wires = [p['wire'] for p in pumps]
    _require(len(set(paints + wires + [bg])) == 2 * len(pumps) + 1, "Conflicting color roles")
    for pump in pumps:
        cells = set(zip(*np.where(a == pump['wire'])))
        _require(len(_components(cells, D8)) == 1, "Disconnected pump wire")
        pump['wire_cells'] = cells
        pump['regions'] = []
    reservoirs = []
    for interior in _components(zip(*np.where(a == bg))):
        if any(r in (0, a.shape[0] - 1) or c in (0, a.shape[1] - 1) for r, c in interior):
            continue
        edge = _neighbors(interior, a.shape, D4)
        colors = {int(a[p]) for p in edge}
        if len(colors) != 1:
            continue
        outline = next(iter(colors))
        if outline in paints + wires + [bg]:
            continue
        shell = {p for p in _neighbors(interior, a.shape) if a[p] == outline}
        owners = [i for i, pump in enumerate(pumps)
                  if _neighbors(shell, a.shape) & pump['wire_cells']]
        _require(len(owners) <= 1, "Reservoir has conflicting wires")
        owner = owners[0] if owners else None
        reservoirs.append(dict(interior=interior, shell=shell, owner=owner, outline=outline))
        if owner is not None:
            pumps[owner]['regions'].append(interior)
    centers = {p['center'] for p in pumps}
    for pump in pumps:
        slots = set().union(*pump['regions']) if pump['regions'] else set()
        distance = _distance(a, bg, paints, pump['footprint'])
        loose = set(zip(*np.where(a == pump['paint']))) - centers
        reachable = {p for p in loose if distance[p] >= 0}
        selected = _selected(reachable, distance, len(slots))
        pump.update(slots=sorted(slots), distance=distance, loose=loose,
                    reachable=reachable, selected=selected, unreachable=loose - reachable)
    return dict(background=bg, pumps=pumps, reservoirs=reservoirs)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve a family member without seed, demonstrations or construction state."""
    a = _grid(input_grid)
    info = _analyze(a)
    b = a.copy()
    for pump in info['pumps']:
        for p in pump['selected']:
            b[p] = info['background']
        for p in pump['slots'][:len(pump['selected'])]:
            b[p] = pump['paint']
    return b


def _ri(rng, low, high):
    return int(rng.integers(low, high + 1))


def _pick(rng, items):
    return items[_ri(rng, 0, len(items) - 1)]


def _shape(rng, elongated=False, simple=False, force_long=False):
    kind = _pick(rng, ('rect', 'rect', 'bar', 'L', 'steps', 'diamond'))
    if simple:
        kind = 'rect'
    if force_long:
        kind = 'bar'
    h, w = _ri(rng, 1, 3), _ri(rng, 2, 3)
    if kind == 'bar':
        cells = {(0, c) for c in range(_ri(rng, 6 if force_long else 3, 8 if elongated or force_long else 5))}
    elif kind == 'diamond':
        cells = {(1 + dr, 1 + dc) for dr, dc in D4} | {(1, 1)}
    elif kind == 'L':
        h, w = _ri(rng, 2, 3), _ri(rng, 2, 4)
        cells = {(r, c) for r in range(h) for c in range(w) if r == 0 or c == 0}
    elif kind == 'steps':
        cells = {(r, c + r // 2) for r in range(_ri(rng, 2, 3)) for c in range(w)}
    else:
        cells = {(r, c) for r in range(h) for c in range(w)}
    if rng.random() < .5:
        cells = {(c, r) for r, c in cells}
    if rng.random() < .5:
        cells = {(-r, c) for r, c in cells}
    shell = {(r + dr, c + dc) for r, c in cells for dr, dc in D8} - cells
    r0, c0 = min(r for r, c in shell), min(c for r, c in shell)
    move = lambda s: {(r - r0, c - c0) for r, c in s}
    return move(cells), move(shell)


def _route(rng, starts, goals, allowed):
    starts = sorted(starts)
    rng.shuffle(starts)
    queue = deque(starts)
    previous = {p: None for p in starts}
    steps = [D8[int(i)] for i in rng.permutation(8)]
    end = None
    while queue:
        r, c = queue.popleft()
        if (r, c) in goals:
            end = r, c
            break
        for dr, dc in steps:
            q = r + dr, c + dc
            if q in allowed and q not in previous:
                previous[q] = r, c
                queue.append(q)
    if end is None:
        return None
    result = set()
    while end is not None:
        result.add(end)
        end = previous[end]
    return result


def _manhattan(p, footprint):
    return min(abs(p[0] - r) + abs(p[1] - c) for r, c in footprint)


def _choose(rng, cells, count, forced=(), style='dispersed'):
    left = set(cells) - set(forced)
    result = list(dict.fromkeys(forced))
    if len(result) > count or len(left) < count - len(result):
        return None
    anchor = result[0] if result else _pick(rng, sorted(left)) if left else (0, 0)
    horizontal = bool(rng.integers(2))
    while len(result) < count:
        choices = sorted(left)
        if style != 'dispersed':
            def score(p):
                dr, dc = abs(p[0] - anchor[0]), abs(p[1] - anchor[1])
                return dr + dc if style == 'cluster' else (30 * dr + dc if horizontal else dr + 30 * dc)
            best = min(score(p) for p in choices)
            choices = [p for p in choices if score(p) == best]
        p = _pick(rng, choices)
        left.remove(p)
        result.append(p)
    return set(result)


def _place_supply(rng, available, distance, footprint, capacity, count, reversal=False):
    reachable = {p for p in available if distance[p] > 0}
    style = _pick(rng, ('dispersed', 'cluster', 'line'))
    if count <= capacity:
        return _choose(rng, reachable, count, style=style)
    # A strict distance shell separates selected cells from retained excess.
    candidates = sorted(reachable)
    rng.shuffle(candidates)
    for a in candidates:
        near = {p for p in reachable if distance[p] <= distance[a]}
        far = reachable - near
        if len(near) < capacity or len(far) < count - capacity:
            continue
        choices = {p for p in far if _manhattan(p, footprint) < _manhattan(a, footprint)} if reversal else far
        if not choices:
            continue
        b = _pick(rng, sorted(choices))
        chosen = _choose(rng, near, capacity, (a,), style)
        retained = _choose(rng, far, count - capacity, (b,), style)
        return chosen | retained
    return None


def _scene(rng, palette, request, attempt):
    n = request['pumps']
    active_counts = [2 if request['mode'] == 'partial' else
                     _ri(rng, request.get('min_reservoirs', 1), request['max_reservoirs']) for _ in range(n)]
    if n == 2:
        active_counts = [min(2, x) for x in active_counts]
    nloops = sum(active_counts) + _ri(rng, 1, 2)
    large = nloops >= 5 or attempt >= 30
    h, w = (_ri(rng, 25, 30), _ri(rng, 25, 30)) if large else (_ri(rng, 18, 28), _ri(rng, 18, 30))
    shape = h, w
    canvas = {(r, c) for r in range(h) for c in range(w)}
    bg, outline, wall = palette[:3]
    wires = [palette[3 + 2 * i] for i in range(n)]
    paints = [palette[4 + 2 * i] for i in range(n)]
    a = np.full(shape, bg, dtype=np.int8)
    regions, occupied = [], set()
    for index in range(nloops):
        inner, shell = _shape(rng, request['elongated'], simple=attempt >= 45,
                              force_long=request.get('require_long', False) and index == 0)
        hh = max(r for r, c in shell) + 1
        ww = max(c for r, c in shell) + 1
        for trial in range(90):
            rr, cc = _ri(rng, 1, h - hh - 1), _ri(rng, 1, w - ww - 1)
            ins = {(r + rr, c + cc) for r, c in inner}
            edge = {(r + rr, c + cc) for r, c in shell}
            cells = ins | edge
            if not cells & (occupied | _neighbors(occupied, shape)):
                regions.append(dict(interior=ins, shell=edge, owner=None))
                occupied |= cells
                for p in edge:
                    a[p] = outline
                break
        else:
            return None
    footprints, wire_sets = [], []
    source_cells = set()
    for i in range(n):
        for trial in range(100):
            rr, cc = _ri(rng, 0, h - 3), _ri(rng, 0, w - 3)
            footprint = {(rr + r, cc + c) for r in range(3) for c in range(3)}
            if footprint & (occupied | source_cells | _neighbors(occupied | source_cells, shape)):
                continue
            center = rr + 1, cc + 1
            rim = footprint - {center}
            footprints.append(footprint)
            wire_sets.append(rim)
            source_cells |= footprint
            for p in rim:
                a[p] = wires[i]
            a[center] = paints[i]
            break
        else:
            return None
    # A closer unconnected reservoir explicitly defeats proximity matching.
    order = list(range(nloops))
    if n == 1:
        can_detach = [j for j in order if j != 0 or not request.get('require_long', False)]
        detached = min(can_detach, key=lambda j: min(_manhattan(p, footprints[0]) for p in regions[j]['interior']))
        order.remove(detached)
    else:
        rng.shuffle(order)
    if request.get('require_long', False):
        order.remove(0)
        order.insert(0, 0)
    cursor = 0
    for i, count in enumerate(active_counts):
        for j in order[cursor:cursor + count]:
            regions[j]['owner'] = i
        cursor += count
    loop_buffer = occupied | _neighbors(occupied, shape)
    source_buffer = source_cells | _neighbors(source_cells, shape)
    for i in rng.permutation(n):
        i = int(i)
        own = wire_sets[i]
        others = set().union(*(s for j, s in enumerate(wire_sets) if j != i)) if n > 1 else set()
        for j in rng.permutation(nloops):
            region = regions[int(j)]
            if region['owner'] != i:
                continue
            goals = _neighbors(region['shell'], shape) - occupied - source_buffer
            otherloops = occupied - region['shell'] - region['interior']
            goals -= _neighbors(otherloops, shape)
            goals -= others | _neighbors(others, shape)
            blocked = loop_buffer | source_cells | others | _neighbors(others, shape)
            allowed = (canvas - blocked) | own | goals
            if not goals:
                return None
            path = _route(rng, own, goals, allowed)
            if path is None:
                return None
            own |= path
        wire_sets[i] = own
    for i, wire in enumerate(wire_sets):
        for p in wire:
            a[p] = wires[i]
    # Barriers never touch wire/contour structures. They can meet the canvas.
    structure = occupied | source_cells | set().union(*wire_sets)
    protected = structure | _neighbors(structure, shape)
    for _ in range(_ri(rng, 4, 10)):
        rr, cc = _ri(rng, 0, h - 1), _ri(rng, 0, w - 1)
        dr, dc = _pick(rng, ((1, 0), (0, 1)))
        segment = {(rr + dr * k, cc + dc * k) for k in range(_ri(rng, 4, max(h, w)))}
        for p in (segment & canvas) - protected:
            a[p] = wall
    # Closed pockets supply visually explicit unreachable distraction.
    pocket_interiors = set()
    for _ in range(2):
        for trial in range(60):
            hh, ww = _ri(rng, 3, 5), _ri(rng, 3, 5)
            rr, cc = _ri(rng, 0, h - hh), _ri(rng, 0, w - ww)
            box = {(r, c) for r in range(rr, rr + hh) for c in range(cc, cc + ww)}
            if box & protected:
                continue
            inner = {(r, c) for r in range(rr + 1, rr + hh - 1) for c in range(cc + 1, cc + ww - 1)}
            for p in box - inner:
                a[p] = wall
            for p in inner:
                a[p] = bg
            pocket_interiors |= inner
            break
    if not pocket_interiors:
        return None
    centers = {(min(r for r, c in f) + 1, min(c for r, c in f) + 1) for f in footprints}
    intended = []
    for i in range(n):
        slots = sorted(set().union(*(r['interior'] for r in regions if r['owner'] == i)))
        cap = len(slots)
        if cap == 9 and request['mode'] == 'excess':
            return None
        if request['mode'] == 'partial':
            owners = {p: j for j, region in enumerate(regions) for p in region['interior']}
            possible = [k for k in range(1, cap) if len({owners[p] for p in slots[:k]}) >= 2
                        and set(slots[:k]) != set(sorted(slots, key=lambda p: (p[1], p[0]))[:k])]
            if not possible:
                return None
            count = _pick(rng, possible)
        elif request['mode'] == 'excess':
            count = cap + _ri(rng, 2, 7)
        else:
            mode = _pick(rng, ('under', 'under', 'equal', 'excess', 'excess'))
            count = _ri(rng, 1, max(1, cap - 1)) if mode == 'under' else cap if mode == 'equal' else cap + _ri(rng, 1, 7)
        distance = _distance(a, bg, paints, footprints[i])
        available = set(zip(*np.where(a == bg))) - occupied
        supply = _place_supply(rng, available, distance, footprints[i], cap, count,
                               reversal=request['mode'] == 'excess' and i == 0)
        if supply is None:
            return None
        for p in supply:
            a[p] = paints[i]
        inactive = {p for p in available - supply if distance[p] < 0}
        distractions = _choose(rng, inactive, min(len(inactive), _ri(rng, 1, 5)))
        if not distractions:
            return None
        for p in distractions:
            a[p] = paints[i]
        selected = _selected(supply, distance, cap)
        intended.append(dict(slots=slots, selected=selected, paint=paints[i]))
    b = a.copy()
    for pump in intended:
        for p in pump['selected']:
            b[p] = bg
        for p in pump['slots'][:len(pump['selected'])]:
            b[p] = pump['paint']
    try:
        info = _analyze(a)
        if len(info['pumps']) != n or not np.array_equal(solve(a), b):
            return None
        if not _witnesses(info, a)[request['mode']] and request['mode'] != 'random':
            return None
        if n == 2 and request['mode'] == 'excess' and not _witnesses(info, a)['wrong_pump']:
            return None
    except AssertionError:
        return None
    return dict(input=a, output=b)


def _witnesses(info, a):
    pumps = info['pumps']
    detached = [r for r in info['reservoirs'] if r['owner'] is None]
    partial = excess = wrong_pump = False
    for i, pump in enumerate(pumps):
        selected, slots = pump['selected'], pump['slots']
        load, cap = len(selected), len(slots)
        prefix = set(slots[:load])
        touched = sum(bool(prefix & cells) for cells in pump['regions'])
        closer = any(min(_manhattan(p, pump['footprint']) for p in r['interior']) <
                     min(_manhattan(p, pump['footprint']) for p in slots) for r in detached) if slots else False
        partial |= (0 < load < cap and touched >= 2 and closer and
                    prefix != set(sorted(slots, key=lambda p: (p[1], p[0]))[:load]))
        retained = pump['reachable'] - selected
        reversal = bool(selected and retained and
                        max(_manhattan(p, pump['footprint']) for p in selected) >
                        min(_manhattan(p, pump['footprint']) for p in retained))
        excess |= bool(retained and pump['unreachable'] and detached and reversal and cap != 9)
        for j, other in enumerate(pumps):
            if i != j:
                wrong_pump |= any(_manhattan(p, other['footprint']) < _manhattan(p, pump['footprint'])
                                  for p in selected)
    return dict(partial=bool(partial), excess=bool(excess), random=True, wrong_pump=bool(wrong_pump))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not _integer(num_train) or not _integer(num_test):
        raise ValueError("counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("require 2..10 training, 1..4 test, and train > test")
    rng = np.random.default_rng(seed)
    base_palette = [int(x) for x in rng.permutation(10)]
    vary_palette = rng.random() < .65
    allow_two = rng.random() < .65
    shift = _pick(rng, ('branches', 'elongated'))
    requests = [dict(mode='partial', pumps=1, max_reservoirs=2, elongated=False),
                dict(mode='excess', pumps=2 if allow_two else 1, max_reservoirs=2, elongated=False)]
    def random_request(test=False):
        return dict(mode='random', pumps=2 if allow_two and rng.random() < .45 else 1,
                    max_reservoirs=4 if test and shift == 'branches' else 3 if test else 2,
                    elongated=bool(test and shift == 'elongated'))
    requests.extend(random_request() for _ in range(num_train - 2))
    requests = [requests[int(i)] for i in rng.permutation(num_train)]
    puzzle = dict(train=[], test=[])
    seen = set()
    tests = [random_request(True) for _ in range(num_test)]
    # Make the first structural shift observable in every returned puzzle.
    if shift == 'branches':
        tests[0].update(pumps=1, min_reservoirs=3, max_reservoirs=4)
    else:
        tests[0]['require_long'] = True
    for split, reqs in (('train', requests), ('test', tests)):
        for request in reqs:
            for attempt in range(100):
                palette = [int(x) for x in rng.permutation(10)] if vary_palette else base_palette
                pair = _scene(rng, palette, request, attempt)
                if pair is not None and _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Bounded scene construction exhausted for " + request['mode'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "Invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "Invalid pair counts")
    seen, training = set(), []
    max_train_pumps = 0
    taught_color_selection = False
    train_region_count = train_span = 0
    test_shift = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "Invalid pair keys")
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape and _key(a) not in seen, "Duplicate input or changed dimensions")
            seen.add(_key(a))
            info = _analyze(a)
            _require(np.array_equal(solve(a), b), "Incorrect relocation label")
            _require(not np.array_equal(a, b), "Generated examples must show movement")
            permitted = set()
            for pump in info['pumps']:
                selected, slots, paint = pump['selected'], pump['slots'], pump['paint']
                _require(slots and selected, "Generated pumps must have storage and collect paint")
                erased = {p for p in pump['loose'] if b[p] == info['background']}
                filled = {p for p in slots if b[p] == paint}
                _require(erased == selected and filled == set(slots[:len(selected)]), "Wrong selected set or prefix")
                _require(len(erased) == len(filled), "Paint is not conserved")
                _require(np.count_nonzero(a == paint) == np.count_nonzero(b == paint), "Color mass changed")
                _require(all(a[p] == b[p] for p in pump['footprint'] | pump['wire_cells']), "Intake or wire modified")
                permitted |= erased | filled
            _require(set(zip(*np.where(a != b))) == permitted, "Unexpected structural change")
            _require(all(a[p] == b[p] for r in info['reservoirs'] if r['owner'] is None for p in r['interior']),
                     "Detached reservoir changed")
            witnesses = _witnesses(info, a)
            if split == 'train':
                training.append(witnesses)
                max_train_pumps = max(max_train_pumps, len(info['pumps']))
                taught_color_selection |= len(info['pumps']) == 2 and witnesses['wrong_pump']
                train_region_count = max(train_region_count, max(len(p['regions']) for p in info['pumps']))
                train_span = max(train_span, max(max(max(r for r, c in cells) - min(r for r, c in cells),
                                                     max(c for r, c in cells) - min(c for r, c in cells)) + 1
                                                 for p in info['pumps'] for cells in p['regions']))
            else:
                _require(len(info['pumps']) <= max_train_pumps, "Untaught number of paint sources")
                if len(info['pumps']) == 2:
                    _require(taught_color_selection, "Two-pump tests need a color-versus-proximity demonstration")
                test_shift |= (max(len(p['regions']) for p in info['pumps']) > train_region_count or
                               any(max(max(r for r, c in cells) - min(r for r, c in cells),
                                       max(c for r, c in cells) - min(c for r, c in cells)) + 1 > train_span
                                   for p in info['pumps'] for cells in p['regions']))
    _require(any(t['partial'] for t in training), "Missing finite-supply / reading-order / attachment lesson")
    _require(any(t['excess'] for t in training), "Missing capacity / path-distance / unreachable lesson")
    _require(test_shift, "Missing actual structural train/test shift")


def _feature(pair):
    a = np.asarray(pair['input'])
    info = _analyze(a)
    capacities, loads, supplies, unreachable, directions, cavities = [], [], [], [], set(), []
    for pump in info['pumps']:
        capacities.append(len(pump['slots']))
        loads.append(len(pump['selected']))
        supplies.append(len(pump['reachable']))
        unreachable.append(len(pump['unreachable']))
        clean = pump['wire_cells'] - pump['footprint'] - _neighbors(pump['footprint'], a.shape)
        for r, c in clean:
            if (r, c + 1) in clean:
                directions.add('horizontal')
            if (r + 1, c) in clean:
                directions.add('vertical')
            if any((r + 1, c + dc) in clean and (r + 1, c) not in clean and (r, c + dc) not in clean for dc in (-1, 1)):
                directions.add('diagonal')
    for region in info['reservoirs']:
        cells = region['interior']
        r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
        cavities.append(tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells)))
    normalized = {}
    structure = bytes(normalized.setdefault(int(x), len(normalized)) for x in a.ravel())
    return dict(shape=a.shape, background=info['background'], pumps=len(info['pumps']),
                capacities=capacities, loads=loads, supplies=supplies, unreachable=unreachable,
                directions=directions, cavities=cavities, reservoirs=len(info['reservoirs']),
                detached=sum(r['owner'] is None for r in info['reservoirs']),
                structure=(a.shape, structure), witnesses=_witnesses(info, a))


def self_test():
    """Stream 42 seeds; no files, retained runs, logs or execution traces."""
    fingerprints, structures, shapes, cavity_shapes, backgrounds, capacities = set(), set(), set(), set(), set(), set()
    directions, pumps_seen, reservoir_counts, loads = set(), set(), set(), set()
    partial = excess = equal = wrong = 0
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                for role in ('input', 'output'):
                    a = pair[role]
                    digest.update(bytes(a.shape))
                    digest.update(a.tobytes())
                f = _feature(pair)
                structures.add(f['structure'])
                shapes.add(f['shape'])
                backgrounds.add(f['background'])
                capacities.update(f['capacities'])
                directions.update(f['directions'])
                pumps_seen.add(f['pumps'])
                reservoir_counts.add(f['reservoirs'])
                loads.update(f['loads'])
                cavity_shapes.update(f['cavities'])
                partial += f['witnesses']['partial']
                wrong += f['witnesses']['wrong_pump']
                for c, l, supply in zip(f['capacities'], f['loads'], f['supplies']):
                    excess += supply > c
                    equal += supply == c
                    _require(l == min(c, supply), "Load/capacity inconsistency")
        fingerprint = digest.hexdigest()
        fingerprints.add(fingerprint)
        if seed == 0:
            first = puzzle
    _require(len(fingerprints) == 42 and len(structures) >= 190, "Insufficient structural variation")
    _require(len(shapes) >= 45 and len(cavity_shapes) >= 20, "Insufficient shape variation")
    _require(len(backgrounds) >= 8 and len(capacities) >= 15 and len(loads) >= 12, "Insufficient role/load variation")
    _require(pumps_seen == {1, 2} and len(reservoir_counts) >= 5, "Insufficient network variation")
    _require(directions == {'horizontal', 'vertical', 'diagonal'}, "Missing route direction")
    _require(partial >= 42 and excess >= 42 and equal >= 5 and wrong >= 15, "Missing mechanism coverage")
    # Determinism and color-equivariance; mapping is applied before solving.
    again = generate_puzzle(0)
    for split in first:
        for p, q in zip(first[split], again[split]):
            _require(all(np.array_equal(p[k], q[k]) for k in p), "Nondeterministic puzzle")
    permutation = np.array([7, 4, 2, 9, 0, 8, 1, 6, 5, 3], dtype=np.int8)
    for p in first['train'] + first['test']:
        _require(np.array_equal(solve(permutation[p['input']]), permutation[p['output']]), "Literal-color shortcut")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(71203 + 10 * nt + ne, nt, ne)
            _require(len(p['train']) == nt and len(p['test']) == ne, "Count ignored")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError("Illegal counts accepted")
    # No queue-order tie convention is allowed at capacity.
    tied_distance = np.array([[1, 1]], dtype=int)
    try:
        _selected({(0, 0), (0, 1)}, tied_distance, 1)
    except AssertionError:
        pass
    else:
        raise AssertionError("Ambiguous frontier accepted")
    # An independently specified tiny relocation regression: one wired four-slot
    # reservoir, one matching reachable cell, and one inaccessible cell.
    a = np.zeros((12, 17), dtype=np.int8)
    a[1:4, 1:4] = 2
    a[2, 2] = 3
    a[2, 4:8] = 2
    a[1:5, 8:12] = 4
    a[2:4, 9:11] = 0
    a[7:11, 12:16] = 5
    a[8:10, 13:15] = 0
    a[8, 13] = 3
    a[5, 2] = 3
    b = a.copy()
    b[5, 2] = 0
    b[2, 9] = 3
    _require(np.array_equal(solve(a), b), "Hand-built conservative relocation failed")
    # Corrupted labels must fail even when they satisfy shape and palette.
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    broken['test'][0]['output'] = broken['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted output accepted")
    return dict(seeds=42, pairs=210, distinct_puzzles=len(fingerprints), distinct_structures=len(structures),
                grid_shapes=len(shapes), cavity_shapes=len(cavity_shapes), capacities=sorted(capacities),
                loads=sorted(loads), backgrounds=sorted(backgrounds), pump_counts=sorted(pumps_seen),
                reservoir_counts=sorted(reservoir_counts), partial_witnesses=partial,
                excess_pumps=excess, exact_supply_pumps=equal, wrong_pump_witnesses=wrong)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(",", ":")))
