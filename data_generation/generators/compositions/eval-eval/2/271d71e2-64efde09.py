"""Fuel-powered moving stencils, with a pooled, bumper-relative stripe code.

Only NumPy and the standard library are used. The input-only decoder does not
use sampling metadata. Zero is background; 1 is spent fuel and 2 is unspent.
"""
from __future__ import annotations

import math
import numpy as np


TRAVEL = ((-1, 0), (0, -1), (1, 0), (0, 1))


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "noninteger cells")
    _need(bool(np.all((0 <= a) & (a <= 9))), "invalid colors")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(a):
    seen, result = set(), []
    for y, x in np.argwhere(a != 0):
        p = int(y), int(x)
        if p in seen:
            continue
        stack, cells = [p], []
        seen.add(p)
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in TRAVEL:
                q = y + dy, x + dx
                if (0 <= q[0] < a.shape[0] and 0 <= q[1] < a.shape[1]
                        and a[q] != 0 and q not in seen):
                    seen.add(q)
                    stack.append(q)
        result.append(cells)
    return result


def _body_view(block):
    """Find the rotation giving travel up: framed counter, then rear bumper."""
    answers = []
    for k in range(4):
        q = np.rot90(block, -k).copy()
        h, length = q.shape
        if h < 5 or length < 4:
            continue
        body, bumper = q[:-2], q[-2:]
        frame = int(body[0, 0])
        if frame < 3:
            continue
        if not (np.all(body[0] == frame) and np.all(body[-1] == frame)
                and np.all(body[:, 0] == frame) and np.all(body[:, -1] == frame)):
            continue
        counter = body[1:-1, 1:-1]
        if not np.all((counter == 1) | (counter == 2)):
            continue
        spent = int(np.count_nonzero(counter == 1))
        if not np.array_equal(counter.ravel(), [1] * spent + [2] * (counter.size - spent)):
            continue
        colors = set(map(int, bumper.flat))
        if not (2 <= len(colors) <= 3 and colors.isdisjoint({0, 1, 2, frame})):
            continue
        forms = [np.rot90(np.fliplr(bumper) if t >= 4 else bumper, t % 4)
                 for t in range(8)]
        if len({_key(f) for f in forms}) != 8:
            continue
        answers.append(dict(k=k, canonical=q, bumper=bumper, frame=frame,
                            spent=spent, remaining=int(counter.size - spent),
                            depth=counter.shape[0], length=length))
    _need(len(answers) == 1, "vehicle does not have a unique rear bumper/counter explanation")
    return answers[0]


def _cells(bounds):
    t, b, l, r = bounds
    return {(y, x) for y in range(t, b) for x in range(l, r)}


def _rear_cell(v, physical_slot):
    h, length = v['canonical'].shape
    return ((h-1, physical_slot), (length-1-physical_slot, h-1),
            (0, length-1-physical_slot), (physical_slot, 0))[v['k']]


def _rear_ray(v, physical_slot, shape):
    y, x = _rear_cell(v, physical_slot)
    y, x = y + v['top'], x + v['left']
    dy, dx = TRAVEL[v['k']]
    ray = []
    y, x = y - dy, x - dx
    while 0 <= y < shape[0] and 0 <= x < shape[1]:
        ray.append((y, x))
        y, x = y - dy, x - dx
    return tuple(ray)


def _clearance(v, shape):
    t, b, l, r = v['bounds']
    return (t, l, shape[0] - b, shape[1] - r)[v['k']]


def _sweep(v, movement):
    t, b, l, r = v['bounds']
    dy, dx = TRAVEL[v['k']]
    return min(t, t + dy * movement), max(b, b + dy * movement), \
        min(l, l + dx * movement), max(r, r + dx * movement)


def _decode(input_grid):
    a = _grid(input_grid)
    vehicles, clues = [], []
    for component in _components(a):
        if any(a[p] in (1, 2) for p in component):
            ys, xs = zip(*component)
            t, b, l, r = min(ys), max(ys) + 1, min(xs), max(xs) + 1
            _need(len(component) == (b - t) * (r - l), "vehicle is not a filled rectangle")
            v = _body_view(a[t:b, l:r])
            v.update(top=t, left=l, bounds=(t, b, l, r), cells=set(component))
            vehicles.append(v)
        else:
            for y, x in component:
                _need(y in (0, a.shape[0]-1) or x in (0, a.shape[1]-1), "unexplained interior pixels")
                clues.append((y, x, int(a[y, x])))
    _need(2 <= len(vehicles) <= 6 and clues, "need multiple vehicles and boundary clues")
    reference = vehicles[0]['bumper']
    length = reference.shape[1]
    structural = set(map(int, reference.flat)) | {vehicles[0]['frame']}
    occupied = set().union(*(v['cells'] for v in vehicles))
    alignments = {}
    for index, v in enumerate(vehicles):
        _need(v['length'] == length and v['frame'] == vehicles[0]['frame'], "inconsistent vehicle structures")
        matches = [flip for flip in (False, True)
                   if np.array_equal(v['bumper'], np.fliplr(reference) if flip else reference)]
        _need(len(matches) == 1, "bumper copies do not uniquely correspond")
        v['mirror'] = matches[0]
        for slot in range(length):
            physical = length - 1 - slot if v['mirror'] else slot
            ray = _rear_ray(v, physical, a.shape)
            if ray and not (occupied & set(ray)) and all(a[p] == 0 for p in ray[:-1]):
                alignments.setdefault(ray[-1], []).append((index, slot))
    code, allocations = {}, [set() for _ in vehicles]
    for y, x, color in clues:
        _need(color >= 3 and color not in structural, "signal/structure role conflict")
        matches = alignments.get((y, x), [])
        _need(len(matches) == 1, "boundary clue has ambiguous or blocked correspondence")
        index, slot = matches[0]
        _need(slot not in code or code[slot] == color, "conflicting observations")
        code[slot] = color
        allocations[index].add(slot)
    _need(code, "empty stripe code")
    cost = len(code)
    sweeps = set()
    clue_cells = {(y, x) for y, x, _ in clues}
    for v in vehicles:
        distance = _clearance(v, a.shape)
        movement = min(v['remaining'] // cost, distance)
        swept = _cells(_sweep(v, movement))
        _need(not (swept & sweeps), "colliding vehicle motions")
        _need(not (swept & clue_cells), "motion would overwrite a clue")
        sweeps.update(swept)
        v.update(distance=distance, movement=movement, cost=cost)
    return a, vehicles, code, allocations


def _render_answer(a, vehicles, code, *, cost=None, boundary_only=False, ignore_mirror=False, local_codes=None):
    """Also supports explicit ablations used to test the demonstrations."""
    output = a.copy()
    for v in vehicles:
        t, b, l, r = v['bounds']
        output[t:b, l:r] = 0
    for index, v in enumerate(vehicles):
        v_code = code if local_codes is None else local_codes[index]
        charge = len(v_code) if cost is None else cost
        movement = (v['distance'] if boundary_only else
                    min(v['remaining'] // charge, v['distance']) if charge else 0)
        # For ablations, avoid invalid array indexing if a false rule overspends.
        ticks = min(v['remaining'], charge * movement)
        q = v['canonical'].copy()
        interior = q[1:-3, 1:-1].copy()
        interior.flat[:v['spent'] + ticks] = 1
        q[1:-3, 1:-1] = interior
        block = np.rot90(q, v['k'])
        dy, dx = TRAVEL[v['k']]
        t, l = v['top'] + dy * movement, v['left'] + dx * movement
        output[t:t+block.shape[0], l:l+block.shape[1]] = block
        for slot, color in v_code.items():
            physical = v['length'] - 1 - slot if v['mirror'] and not ignore_mirror else slot
            y, x = _rear_cell(v, physical)
            # Rearmost old bumper cell is uncovered at step one; each further
            # step uncovers the next cell along the direction of travel.
            y, x = y + v['top'], x + v['left']
            for step in range(movement):
                output[y + dy * step, x + dx * step] = color
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from the supplied pixels, using fixed family conventions."""
    a, vehicles, code, _ = _decode(input_grid)
    return _render_answer(a, vehicles, code)


def _template(rng, length, colors):
    for _ in range(40):
        style = int(rng.integers(3))
        t = rng.choice(colors, (2, length)).astype(np.int8)
        if style == 1:
            t[0] = colors[0]
            t[1] = colors[1]
            t[1, :int(rng.integers(1, length))] = colors[0]
        elif style == 2:
            t[:] = colors[0]
            t[1, 1:] = colors[1]
            t[0, -1] = colors[-1]
        for i, c in enumerate(colors):
            t[1, i+1] = c
        if len({_key(np.rot90(np.fliplr(t) if k >= 4 else t, k % 4))
                for k in range(8)}) == 8:
            return t
    t = np.full((2, length), colors[0], dtype=np.int8)
    t[1, 1:] = colors[1]
    for i, c in enumerate(colors[2:], 2):
        t[1, i] = c
    return t


def _block(template, frame, depth, remaining, mirror, k):
    length = template.shape[1]
    q = np.full((depth+4, length), frame, dtype=np.int8)
    counter = np.full((depth, length-2), 2, dtype=np.int8)
    counter.flat[:counter.size-remaining] = 1
    q[1:depth+1, 1:-1] = counter
    q[-2:] = np.fliplr(template) if mirror else template
    return np.rot90(q, k).copy()


def _candidate(template, frame, depth, remaining, mirror, k, top, left, code, shape):
    # Sampling knows the proposed structure. Avoid re-decoding every rejected
    # placement; the returned input is always decoded afresh before acceptance.
    canonical = _block(template, frame, depth, remaining, mirror, 0)
    block = np.rot90(canonical, k).copy()
    v = dict(k=k, canonical=canonical, bumper=canonical[-2:].copy(), frame=frame,
             spent=depth*(template.shape[1]-2)-remaining, remaining=remaining,
             depth=depth, length=template.shape[1])
    h, w = block.shape
    v.update(top=top, left=left, bounds=(top, top+h, left, left+w), mirror=mirror)
    v['cells'] = _cells(v['bounds'])
    v['movement'] = min(remaining // len(code), _clearance(v, shape))
    v['swept'] = _cells(_sweep(v, v['movement']))
    v['rays'] = {slot: _rear_ray(v, template.shape[1]-1-slot if mirror else slot, shape)
                 for slot in code}
    v['ray_cells'] = set().union(*(set(r) for r in v['rays'].values()))
    v['block'] = block
    return v


def _compatible(v, previous):
    for old in previous:
        if (v['swept'] & old['swept'] or v['cells'] & old['ray_cells']
                or old['cells'] & v['ray_cells']):
            return False
        if any((y+dy, x+dx) in old['cells'] for y, x in v['cells'] for dy, dx in TRAVEL):
            return False
    return all(len(ray) >= 2 for ray in v['rays'].values())


def _choose_clues(rng, placed, code, shape, mode):
    occupied = set().union(*(v['cells'] for v in placed))
    swept = set().union(*(v['swept'] for v in placed))
    alignments = {}
    for index, v in enumerate(placed):
        for slot in range(v['length']):
            physical = v['length']-1-slot if v['mirror'] else slot
            ray = _rear_ray(v, physical, shape)
            if ray and not (occupied & set(ray)):
                alignments.setdefault(ray[-1], []).append((index, slot))
    slots = list(code)
    rng.shuffle(slots)
    if mode == 'full':
        owner = int(rng.integers(len(placed)))
        owners = [owner] * len(slots)
    elif mode in ('pool', 'test'):
        cut = int(rng.integers(1, len(slots)))
        owners = [0] * cut + [1] * (len(slots)-cut)
    else:
        owners = list(map(int, rng.integers(len(placed), size=len(slots))))
    clues = []
    for slot, index in zip(slots, owners):
        endpoint = placed[index]['rays'][slot][-1]
        if (alignments.get(endpoint) != [(index, slot)] or endpoint in swept
                or any((endpoint[0]+dy, endpoint[1]+dx) in occupied for dy, dx in TRAVEL)):
            return None
        clues.append((endpoint, code[slot]))
    # Optional agreeing duplicate observations, with no new latent slots.
    if mode == 'random' and rng.random() < .35:
        for slot in slots:
            for index, v in enumerate(placed):
                p = v['rays'][slot][-1]
                if (alignments.get(p) == [(index, slot)] and p not in swept
                        and p not in {p0 for p0, _ in clues}
                        and not any((p[0]+dy, p[1]+dx) in occupied for dy, dx in TRAVEL)):
                    clues.append((p, code[slot]))
                    break
    return clues


def _make_pair(rng, lesson, *, _force_fallback=False):
    """Bounded geometric rejection sampling plus a spacious constructive fallback."""
    teaching = lesson in ('pool', 'turn')
    test = lesson == 'test'
    for attempt in ((64,) if _force_fallback else range(65)):
        fallback = attempt == 64
        length = int(rng.integers(5, 9) if teaching else rng.integers(4, 11))
        cost = int(rng.integers(2, min(4, length)+1) if lesson == 'pool'
                   else rng.integers(3, min(4, length)+1) if lesson == 'turn'
                   else rng.integers(2 if test else 1, min(6, length)+1))
        if fallback:
            cost = min(cost, 6)
            length = max(6, cost+2)
        palette = list(map(int, rng.permutation(np.arange(3, 10))))
        structural_count = int(rng.integers(2, 4))
        frame, colors = palette[0], palette[1:1+structural_count]
        available = palette[1+structural_count:]
        template = _template(rng, length, colors)
        slots = sorted(map(int, rng.choice(length, cost, replace=False)))
        if rng.random() < .3:
            start = int(rng.integers(length-cost+1))
            slots = list(range(start, start+cost))
        signal_count = min(cost, len(available))
        if lesson == 'turn':
            signal_count = 2  # Distinct-color count is deliberately not stripe count.
        elif cost > 1 and rng.random() < .4:
            signal_count = int(rng.integers(1, signal_count+1))
        values = available[:signal_count] + list(map(int, rng.choice(available[:signal_count], cost-signal_count)))
        rng.shuffle(values)
        code = dict(zip(slots, values))
        if teaching or test:
            if code == {length-1-s: c for s, c in code.items()}:
                if not fallback:
                    continue
                # The constructive fallback must not depend on a lucky code.
                code[slots[0]] = available[1] if code[slots[0]] == available[0] else available[0]
        count = 3 if test and fallback else 2 if fallback else int(rng.integers(3 if test else 2, 6))
        base = int(rng.integers(4))
        if lesson == 'pool':
            ks, mirrors = [base] * count, [bool(i % 2) for i in range(count)]
        elif lesson == 'turn':
            ks = [base, (base+1) % 4] + list(map(int, rng.integers(4, size=count-2)))
            mirrors = [False] * count
        elif test:
            ks = [base, (base+1) % 4, base] + list(map(int, rng.integers(4, size=count-3)))
            mirrors = [False, True, False] + list(map(bool, rng.integers(2, size=count-3)))
        else:
            ks = list(map(int, rng.integers(4, size=count)))
            mirrors = list(map(bool, rng.integers(2, size=count)))
        if fallback:
            base = 0
            ks = [0, 0] if lesson == 'pool' else [0, 1] + ([0] if test else [])
            mirrors = [False, True] if lesson == 'pool' else [False, test] + ([False] if test else [])
        shape = (26, 26) if fallback else tuple(map(int, rng.integers(18 if teaching or test else 12, 31, size=2)))
        placed = []
        for index in range(count):
            regime = ('fuel', 'edge')[index] if (teaching or test) and index < 2 else str(rng.choice(['free', 'zero', 'fresh', 'exact']))
            if fallback:
                depth = 3
                capacity = depth * (length-2)
                remaining = 2*cost+1 if index == 0 else capacity if index == 1 else cost+1
                y, x = (13, 1) if index == 0 else (1, 17) if lesson == 'pool' else (1, 1) if index == 1 else (13, 17)
                v = _candidate(template, frame, depth, remaining, mirrors[index], ks[index], y, x, code, shape)
                if not _compatible(v, placed):
                    break
                placed.append(v)
                continue
            for _ in range(90):
                depth = int(rng.integers(3 if teaching else 1, 7))
                capacity, width = depth*(length-2), length-2
                if regime == 'fuel':
                    residue = int(rng.integers(1, cost)) if cost > 1 else 0
                    low, high = max(1, math.ceil(width/cost)), (capacity-residue)//cost
                    if low > high:
                        continue
                    remaining = cost*int(rng.integers(low, high+1)) + residue
                elif regime == 'edge':
                    if capacity < 2*cost:
                        continue
                    remaining = int(rng.integers(2*cost, capacity+1))
                elif regime == 'zero':
                    remaining = int(rng.integers(min(cost, capacity+1)))
                elif regime == 'fresh':
                    remaining = capacity
                elif regime == 'exact':
                    if capacity < cost:
                        continue
                    remaining = cost*int(rng.integers(1, capacity//cost+1))
                else:
                    remaining = int(rng.integers(capacity+1))
                h, w = (depth+4, length) if ks[index] % 2 == 0 else (length, depth+4)
                if h > shape[0] or w > shape[1]:
                    continue
                y, x = int(rng.integers(shape[0]-h+1)), int(rng.integers(shape[1]-w+1))
                v = _candidate(template, frame, depth, remaining, mirrors[index], ks[index], y, x, code, shape)
                d, affordable = _clearance(v, shape), remaining//cost
                if regime == 'fuel' and not (d > affordable >= 1):
                    continue
                if regime == 'edge' and not (1 <= d < affordable):
                    continue
                if regime == 'exact' and not (d == affordable):
                    continue
                if _compatible(v, placed):
                    placed.append(v)
                    break
            else:
                break
        if len(placed) != count:
            continue
        mode = 'pool' if lesson == 'pool' else 'full' if lesson == 'turn' else 'test' if test else 'random'
        clues = _choose_clues(rng, placed, code, shape, mode)
        if clues is None:
            continue
        a = np.zeros(shape, dtype=np.int8)
        for v in placed:
            t, b, l, r = v['bounds']
            a[t:b, l:r] = v['block']
        for p, color in clues:
            a[p] = color
        # Rotate the fallback as a scene, without ever reflecting the counter order.
        if fallback:
            a = np.rot90(a, int(rng.integers(4))).copy()
        try:
            decoded = _decode(a)
            out = _render_answer(*decoded[:3])
        except AssertionError:
            continue
        return {'input': a, 'output': out}
    raise RuntimeError('bounded stencil construction failed')


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    lessons = ['pool', 'turn'] + ['random'] * (num_train-2)
    rng.shuffle(lessons)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, schedule in (('train', lessons), ('test', ['test']*num_test)):
        for lesson in schedule:
            for _ in range(12):
                pair = _make_pair(rng, lesson)
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('could not generate distinct inputs')
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    a, vs, code, allocations = _decode(pair['input'])
    return dict(shape=a.shape, length=vs[0]['length'], count=len(vs), cost=len(code),
                signals=len(set(code.values())), depths=tuple(v['depth'] for v in vs),
                directions=tuple(v['k'] for v in vs), mirrors=tuple(v['mirror'] for v in vs),
                pooled=all(len(s) < len(code) for s in allocations),
                unhinted=any(not s for s in allocations),
                allocations=tuple(len(s) for s in allocations),
                residual=any(0 < v['remaining'] % len(code) < len(code) and v['movement'] > 0
                             and v['remaining']//len(code) < v['distance'] for v in vs),
                fuel=any(0 < v['remaining']//len(code) < v['distance'] for v in vs),
                edge=any(0 < v['distance'] < v['remaining']//len(code) for v in vs),
                row_crossing=any((v['spent']+v['movement']*len(code))//(v['length']-2)
                                 > v['spent']//(v['length']-2) for v in vs),
                stationary=any(v['movement'] == 0 for v in vs),
                fresh=any(v['spent'] == 0 for v in vs),
                slots=tuple(sorted(code)), palette=tuple(map(int, np.unique(a))))


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training, ablations = set(), [], set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            a, vs, code, allocations = _decode(pair['input'])
            output = _grid(pair['output'])
            _need(output.shape == a.shape, 'dimensions changed')
            _need(_key(a) not in seen, 'duplicate input')
            seen.add(_key(a))
            expected = _render_answer(a, vs, code)
            _need(np.array_equal(output, expected), 'incorrect stencil motion/trail/counter')
            # Per-vehicle conservation is checked directly against returned pixels.
            for v in vs:
                dy, dx = TRAVEL[v['k']]
                t, l = v['top']+dy*v['movement'], v['left']+dx*v['movement']
                h, w = (v['bounds'][1]-v['top'], v['bounds'][3]-v['left'])
                q = np.rot90(output[t:t+h, l:l+w], -v['k'])
                counter = q[1:-3, 1:-1]
                _need(np.count_nonzero(counter == 1) - v['spent'] == len(code)*v['movement'], 'fuel not conserved')
                _need(np.array_equal(q[-2:], v['bumper']), 'bumper altered')
            trail_colors = set(code.values())
            old_trail = int(np.count_nonzero(np.isin(a, list(trail_colors))))
            new_trail = int(np.count_nonzero(np.isin(output, list(trail_colors))))
            _need(new_trail-old_trail == sum(v['movement']*len(code) for v in vs), 'trail pixels do not equal fuel spent')
            if split == 'train':
                training.append(_features(pair))
                for name, kwargs in (('one_fuel_per_move', {'cost': 1}),
                                     ('ignore_fuel', {'boundary_only': True}),
                                     ('distinct_colors', {'cost': len(set(code.values()))}),
                                     ('ignore_reflection', {'ignore_mirror': True}),
                                     ('no_pooling', {'local_codes': [{s: code[s] for s in observed}
                                                                   for observed in allocations]})):
                    if not np.array_equal(expected, _render_answer(a, vs, code, **kwargs)):
                        ablations.add(name)
    _need(any(f['pooled'] for f in training), 'training lacks distributed pooling')
    _need(any(len(set(f['mirrors'])) > 1 for f in training), 'training lacks reflection')
    _need(any(len({k % 2 for k in f['directions']}) > 1 for f in training), 'training lacks rotation across axes')
    _need(any(f['fuel'] and f['residual'] for f in training), 'training lacks incomplete final fuel batch')
    _need(any(f['edge'] for f in training), 'training lacks boundary stopping')
    _need(any(f['row_crossing'] for f in training), 'training lacks counter row advancement')
    _need(ablations == {'one_fuel_per_move', 'ignore_fuel', 'distinct_colors', 'ignore_reflection', 'no_pooling'},
          'training does not distinguish simpler rules')


def self_test():
    """Bounded 42-seed diversity, semantic invariants, covariance and API probes."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    features = [_features(p) for task in puzzles for s in ('train', 'test') for p in task[s]]
    _need(len({tuple(_key(p['input']) for p in task['train']+task['test']) for task in puzzles}) == 42, 'seed diversity')
    _need(len({f['shape'] for f in features}) >= 35, 'shape diversity')
    _need(len({f['length'] for f in features}) >= 6, 'bumper length diversity')
    _need({f['cost'] for f in features} >= {1, 2, 3, 4, 5, 6}, 'stripe-count diversity')
    _need({f['count'] for f in features} >= {2, 3, 4, 5}, 'vehicle-count diversity')
    _need({k for f in features for k in f['directions']} == set(range(4)), 'direction diversity')
    _need(len({f['slots'] for f in features}) >= 35, 'slot geometry diversity')
    _need(any(f['stationary'] for f in features) and any(f['fresh'] for f in features), 'counter state extremes')
    _need(any(f['signals'] < f['cost'] for f in features), 'repeated stripe colors')
    _need(any(any(b-a == 1 for a, b in zip(f['slots'], f['slots'][1:])) for f in features), 'adjacent stripes')
    _need(any(f['unhinted'] and f['pooled'] for f in features), 'pooled code on unhinted copy')
    _need(any(h > 1.3*w for h, w in (f['shape'] for f in features)) and
          any(w > 1.3*h for h, w in (f['shape'] for f in features)), 'aspect ratio variation')
    _need(all(len({_features(p)['cost'] for p in task['train']+task['test']}) > 1 for task in puzzles), 'within-puzzle costs do not vary')
    repeated = generate_puzzle(0)
    _need(all(np.array_equal(p[k], q[k]) for s in repeated for p, q in zip(puzzles[0][s], repeated[s]) for k in p), 'determinism')
    # Rotation preserves the counter convention. Reflection must preserve the
    # bumper correspondence but is not a valid whole-scene counter isometry.
    for pair in puzzles[0]['train']+puzzles[0]['test']:
        for k in range(4):
            _need(np.array_equal(solve(np.rot90(pair['input'], k)), np.rot90(pair['output'], k)), 'rotation covariance')
        palette = np.arange(10, dtype=np.int8)
        palette[3:] = np.roll(palette[3:], 2)
        _need(np.array_equal(solve(palette[pair['input']]), palette[pair['output']]), 'structural/signal recoloring')
    for seed in range(8):
        for lesson in ('pool', 'turn', 'random', 'test'):
            p = _make_pair(np.random.default_rng(seed), lesson, _force_fallback=True)
            _need(np.array_equal(solve(p['input']), p['output']), 'constructive fallback')
    # Compatible inputs beyond the sampler's size/code range, and a wake longer
    # than its source object, must follow exactly the same input-only rule.
    for length, depth, remaining, top, slots in (
            (12, 12, 79, 9, tuple(range(8))),
            (10, 2, 16, 20, (3,))):
        rng = np.random.default_rng(9000+length)
        template = _template(rng, length, (4, 5))
        code = {s: 6+s % 2 for s in slots}
        shape = (30, 30)
        placed = [
            _candidate(template, 3, depth, remaining, False, 0, top, 1, code, shape),
            _candidate(template, 3, 1, length-2, True, 0, 5, 16, code, shape)]
        _need(_compatible(placed[1], placed[:1]), 'large-case geometry')
        clues = _choose_clues(rng, placed, code, shape, 'full')
        _need(clues is not None, 'large-case clue attribution')
        a = np.zeros(shape, dtype=np.int8)
        for v in placed:
            t, b, l, r = v['bounds']
            a[t:b, l:r] = v['block']
        for p, color in clues:
            a[p] = color
        output = solve(a)
        _, decoded, recovered, _ = _decode(a)
        _need(len(recovered) == len(slots), 'large code lost slots')
        _need(np.count_nonzero(np.isin(output, (6, 7))) - len(clues)
              == sum(v['movement']*len(slots) for v in decoded), 'large-case conservation')
        if len(slots) == 1:
            _need(max(v['movement'] for v in decoded) > depth+4, 'long wake case did not exceed object length')
    # Every legal explicit count combination is small enough for this bounded test.
    for tr in range(2, 11):
        for te in range(1, min(4, tr-1)+1):
            p = generate_puzzle(1000+tr*10+te, tr, te)
            _need(len(p['train']) == tr and len(p['test']) == te, 'count ignored')
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = {s: [{k: a.copy() for k, a in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('output corruption accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds and all legal count combinations)')
