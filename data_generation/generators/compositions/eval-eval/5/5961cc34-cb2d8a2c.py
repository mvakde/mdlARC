"""Counted edge relays: reachability depends on a unary, offset output port.

All decoding is input-only. Construction plans cells; the solver independently
segments the rendered scene and projects the nearest hit along each ray.
"""
from __future__ import annotations

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # N, E, S, W


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "bad grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(bool(np.all((a >= 0) & (a <= 9))), "bad color")
    return a.astype(np.int8, copy=True)


def _step(cell, direction, distance=1):
    dr, dc = DIRS[direction]
    return cell[0] + distance * dr, cell[1] + distance * dc


def _inside(cell, shape):
    return 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]


def _line(a, b):
    _require(a[0] == b[0] or a[1] == b[1], "nonorthogonal segment")
    if a[0] == b[0]:
        return {(a[0], c) for c in range(min(a[1], b[1]), max(a[1], b[1]) + 1)}
    return {(r, a[1]) for r in range(min(a[0], b[0]), max(a[0], b[0]) + 1)}


def _components(cells):
    todo, result = set(cells), []
    while todo:
        start = min(todo)
        todo.remove(start)
        group, stack = {start}, [start]
        while stack:
            cell = stack.pop()
            for d in range(4):
                q = _step(cell, d)
                if q in todo:
                    todo.remove(q)
                    group.add(q)
                    stack.append(q)
        result.append(group)
    return result


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _separate(cells, occupied):
    return not any((r + dr, c + dc) in occupied for r, c in cells
                   for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def _elbow(obj, distance=None):
    """A physical lead from the unmarked rail end to the virtual output port."""
    d = obj['n'] + 1 if distance is None else distance
    end = obj['end']
    normal = obj['direction']
    knee = _step(end, normal, d)
    axis = (0, 1) if normal % 2 == 0 else (1, 0)
    exit_cell = (knee[0] + obj['sign'] * d * axis[0],
                 knee[1] + obj['sign'] * d * axis[1])
    return _line(end, knee) | _line(knee, exit_cell), exit_cell


def _decode(value):
    a = _grid(value)
    h, w = a.shape
    bg = int(a[0, 0])
    _require(all(int(a[p]) == bg for p in ((0, w - 1), (h - 1, 0), (h - 1, w - 1))),
             "corners must be background")
    _require(np.count_nonzero(a == bg) > a.size // 2, "background must be a majority")
    groups = _components(map(tuple, np.argwhere(a != bg).tolist()))
    boundary = [g for g in groups if any(r in (0, h - 1) or c in (0, w - 1) for r, c in g)]
    _require(len(boundary) == 1, "one boundary launcher required")
    launcher = boundary[0]
    starts = [p for p in launcher if p[0] in (0, h - 1) or p[1] in (0, w - 1)]
    _require(len(starts) == 1, "launcher touches one boundary cell")
    start = starts[0]
    heading = 2 if start[0] == 0 else 0 if start[0] == h - 1 else 1 if start[1] == 0 else 3
    ordered = [_step(start, heading, k) for k in range(len(launcher))]
    _require(set(ordered) == launcher and 3 <= len(launcher) <= 5, "straight launcher of length 3..5")
    signal, marker = int(a[start]), int(a[ordered[-1]])
    _require(signal != marker and all(int(a[p]) == signal for p in ordered[:-1]), "tail and cap")
    bodies, rails, objects = set(), set(), []
    for group in groups:
        if group == launcher:
            continue
        r0, r1, c0, c1 = _bbox(group)
        _require(0 < r0 <= r1 < h - 1 and 0 < c0 <= c1 < w - 1, "interior relay")
        colors, counts = np.unique([a[p] for p in group], return_counts=True)
        _require(len(colors) == 3 and marker in colors and signal not in colors, "three-color relay")
        _require(np.count_nonzero(counts == counts.max()) == 1, "unique carrier majority")
        body = int(colors[counts.argmax()])
        _require(body != marker, "carrier must dominate")
        rail = next(int(c) for c in colors if c not in (body, marker))
        rail_cells = {p for p in group if int(a[p]) in (rail, marker)}
        mark_cells = {p for p in rail_cells if int(a[p]) == marker}
        n = len(mark_cells)
        _require(1 <= n <= 4, "one to four unary marks")
        _require(2 * n <= len(rail_cells) <= 2 * n + 3, "rail length")
        sides = [d for d, match in enumerate((all(r == r0 for r, c in rail_cells),
                 all(c == c1 for r, c in rail_cells), all(r == r1 for r, c in rail_cells),
                 all(c == c0 for r, c in rail_cells))) if match]
        _require(len(sides) == 1, "rail must identify a unique extreme face")
        normal = sides[0]
        span = sorted(c if normal % 2 == 0 else r for r, c in rail_cells)
        _require(span == list(range(span[0], span[-1] + 1)), "continuous rail")
        _require((c0 < span[0] <= span[-1] < c1) if normal % 2 == 0
                 else (r0 < span[0] <= span[-1] < r1), "rail inset from both corners")
        marks = sorted(c if normal % 2 == 0 else r for r, c in mark_cells)
        low = marks == [span[0] + 2 * j for j in range(n)]
        high = marks == sorted(span[-1] - 2 * j for j in range(n))
        _require(low != high, "asymmetric alternating endpoint code")
        end_value, sign = (span[-1], 1) if low else (span[0], -1)
        end = next(p for p in rail_cells if (p[1] if normal % 2 == 0 else p[0]) == end_value)
        carrier = group - rail_cells
        _require(len(carrier) > len(rail_cells) and len(_components(carrier)) == 1,
                 "connected carrier larger than rail")
        _require(all(_step(p, (normal + 2) % 4) in carrier for p in rail_cells), "rail backed by body")
        obj = dict(cells=group, rail=rail_cells, marks=mark_cells, n=n,
                   direction=normal, sign=sign, end=end, body=body, rail_color=rail,
                   bbox=(r0, r1, c0, c1))
        lead, _ = _elbow(obj)
        _require(all(_inside(p, a.shape) for p in lead), "elbow out of bounds")
        objects.append(obj)
        bodies.add(body)
        rails.add(rail)
    _require(1 <= len(objects) <= 9, "one to nine relays")
    _require(len(bodies) == len(rails) == 1, "shared carrier and rail roles within a scene")
    body, rail = next(iter(bodies)), next(iter(rails))
    _require(len({bg, signal, marker, body, rail}) == 5, "five distinct visible roles")
    for i, obj in enumerate(objects):
        _require(_separate(obj['cells'], launcher | set().union(*(o['cells'] for o in objects[:i]))),
                 "objects need an eight-neighbor gap")
    return a, (bg, signal, body, rail, marker), launcher, ordered[-1], heading, objects


def _project_trace(shape, launcher, cap, heading, objects, *, direct=False, fixed_distance=None):
    """Input solver: nearest positive projection, independent of placement order."""
    trace, reached, gaps, transitions = set(launcher), [], [], []
    origin = cap
    for _ in range(len(objects) + 1):
        dr, dc = DIRS[heading]
        boundary_distance = (origin[0] + 1 if heading == 0 else shape[1] - origin[1]
                             if heading == 1 else shape[0] - origin[0] if heading == 2 else origin[1] + 1)
        hits = [(boundary_distance, -1)]
        for i, obj in enumerate(objects):
            for r, c in obj['cells']:
                distance = (r - origin[0]) * dr + (c - origin[1]) * dc
                transverse = (r - origin[0]) * dc - (c - origin[1]) * dr
                if distance > 0 and transverse == 0:
                    hits.append((distance, i))
        distance, index = min(hits)
        trace.update(_step(origin, heading, k) for k in range(1, distance))
        if index == -1:
            return trace, dict(reached=tuple(reached), gaps=tuple(gaps), transitions=tuple(transitions))
        _require(index not in reached, "relay cycle or reactivation")
        obj = objects[index]
        _require(obj['direction'] != (heading + 2) % 4, "direct reversal excluded")
        reached.append(index)
        gaps.append(distance - 1)
        transitions.append((heading, obj['direction'], obj['n'], obj['sign']))
        trace.update(obj['cells'])
        if direct:
            origin = obj['end']
        else:
            lead, origin = _elbow(obj, fixed_distance)
            _require(all(_inside(p, shape) for p in lead), "active elbow out of bounds")
            others = set().union(*(o['cells'] for j, o in enumerate(objects) if j != index))
            _require(not lead & others, "elbows cannot cross another relay")
            trace.update(lead)
        heading = obj['direction']
    raise ValueError("unterminated route")


def _analyze(input_grid):
    a, palette, launcher, cap, heading, objects = _decode(input_grid)
    trace, facts = _project_trace(a.shape, launcher, cap, heading, objects)
    out = np.full_like(a, palette[0])
    for p in trace:
        out[p] = palette[1]
    facts.update(shape=a.shape, palette=palette, initial=heading, objects=objects,
                 active=len(facts['reached']), decoys=len(objects) - len(facts['reached']), trace=trace)
    return out, facts


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from visible launcher, footprints, and alternating rails."""
    return _analyze(input_grid)[0]


def _shape(rng, n, outgoing, sign, small=False):
    # Build with its rail on the top, then rotate the entire footprint.
    length = 2 * n + int(rng.integers(0, 4))
    h = 3 if small else int(rng.integers(3, 8))
    w = length + 2 + int(rng.integers(0, 4))
    start = int(rng.integers(1, w - length))
    cells = {(r, c) for r in range(h) for c in range(w)}
    kind = int(rng.integers(4))
    if kind == 1:
        cells -= {(h - 1, 0), (h - 1, w - 1)}
    elif kind == 2:
        depth = int(rng.integers(1, h - 1))
        cells -= {(r, 0) for r in range(h - depth, h)}
    elif kind == 3:
        at = int(rng.integers(1, w - 1))
        cells -= {(h - 1, at)}
    rail = {(0, c) for c in range(start, start + length)}
    marks = {(0, start + 2 * j) for j in range(n)}
    # N -> E -> S -> W by clockwise rotation.
    for _ in range(outgoing):
        cells = {(c, h - 1 - r) for r, c in cells}
        rail = {(c, h - 1 - r) for r, c in rail}
        marks = {(c, h - 1 - r) for r, c in marks}
        h, w = w, h
    # Rotation can reverse the natural positive coordinate of the rail.
    values = sorted(c if outgoing % 2 == 0 else r for r, c in rail)
    mark_values = {values[0] + 2 * j for j in range(n)} if sign == 1 else {values[-1] - 2 * j for j in range(n)}
    marks = {p for p in rail if (p[1] if outgoing % 2 == 0 else p[0]) in mark_values}
    end_value = values[-1] if sign == 1 else values[0]
    end = next(p for p in rail if (p[1] if outgoing % 2 == 0 else p[0]) == end_value)
    return dict(cells=cells, rail=rail, marks=marks, n=n, direction=outgoing, sign=sign, end=end)


def _translate(obj, r0, c0):
    result = dict(obj)
    for name in ('cells', 'rail', 'marks'):
        result[name] = {(r + r0, c + c0) for r, c in obj[name]}
    result['end'] = (obj['end'][0] + r0, obj['end'][1] + c0)
    return result


def _constructor_trace(shape, launcher, cap, heading, objects):
    """Cell stepping used only to label constructed scenes and recheck order."""
    owner = {p: i for i, obj in enumerate(objects) for p in obj['cells']}
    trace, order = set(launcher), []
    point = _step(cap, heading)
    for _ in range(shape[0] * shape[1] * (len(objects) + 2)):
        if not _inside(point, shape):
            return trace, order
        if point not in owner:
            trace.add(point)
            point = _step(point, heading)
            continue
        i = owner[point]
        if i in order:
            return None
        obj = objects[i]
        if obj['direction'] == (heading + 2) % 4:
            return None
        lead, end = _elbow(obj)
        if any(not _inside(p, shape) or (p in owner and owner[p] != i) for p in lead):
            return None
        order.append(i)
        trace.update(obj['cells'] | lead)
        heading, point = obj['direction'], _step(end, obj['direction'])
    return None


def _try_layout(rng, ns, initial, signs, turn_index, decoy_count):
    k = len(ns)
    lower = 14 if k <= 1 else 20 if k == 2 else 27
    h, w = map(int, rng.integers(lower, 31, size=2))
    if k <= 1 and rng.random() < .3:
        h, w = int(rng.integers(12, 19)), int(rng.integers(24, 31))
    line = int(rng.integers(2, (w if initial % 2 == 0 else h) - 2))
    start = (0, line) if initial == 2 else (h - 1, line) if initial == 0 else (line, 0) if initial == 1 else (line, w - 1)
    tail_length = int(rng.integers(2, 5))
    launcher = {_step(start, initial, j) for j in range(tail_length + 1)}
    cap = _step(start, initial, tail_length)
    budget = [420]

    def grow(objects, occupied, trace, origin, incoming):
        index = len(objects)
        if index == k:
            simulation = _constructor_trace((h, w), launcher, cap, initial, objects)
            if simulation is not None and simulation[1] == list(range(k)):
                return objects, simulation[0]
            return None
        for _ in range(65):
            if budget[0] <= 0:
                return None
            budget[0] -= 1
            turn = int(rng.choice((-1, 1) if index == turn_index else (-1, 0, 1)))
            outgoing = (incoming + turn) % 4
            sign = signs.get(index, int(rng.choice((-1, 1))))
            obj = _shape(rng, ns[index], outgoing, sign, small=k >= 3)
            r0, r1, c0, c1 = _bbox(obj['cells'])
            entry_side = (incoming + 2) % 4
            entry = [p for p in sorted(obj['cells']) if (p[0] == r0 if entry_side == 0 else
                     p[1] == c1 if entry_side == 1 else p[0] == r1 if entry_side == 2 else p[1] == c0)]
            anchor = entry[int(rng.integers(len(entry)))]
            options = []
            for distance in range(2, 26):
                target = _step(origin, incoming, distance)
                placed = _translate(obj, target[0] - anchor[0], target[1] - anchor[1])
                b0, b1, b2, b3 = _bbox(placed['cells'])
                if not (0 < b0 <= b1 < h - 1 and 0 < b2 <= b3 < w - 1):
                    continue
                if not _separate(placed['cells'], occupied) or placed['cells'] & trace:
                    continue
                lead, exit_cell = _elbow(placed)
                if not all(_inside(p, (h, w)) for p in lead) or lead & occupied:
                    continue
                approach = {_step(origin, incoming, j) for j in range(1, distance)}
                if approach & occupied:
                    continue
                options.append((placed, lead, exit_cell, approach))
            if not options:
                continue
            placed, lead, exit_cell, approach = options[int(rng.integers(len(options)))]
            result = grow(objects + [placed], occupied | placed['cells'],
                          trace | approach | lead | placed['cells'], exit_cell, outgoing)
            if result is not None:
                return result
        return None

    built = grow([], launcher, launcher, cap, initial)
    if built is None:
        return None
    objects, trace = built
    occupied = launcher | set().union(*(o['cells'] for o in objects))
    for _ in range(decoy_count):
        for attempt in range(240):
            obj = _shape(rng, int(rng.choice(ns or [1, 2])), int(rng.integers(4)), int(rng.choice((-1, 1))), small=attempt > 100)
            b0, b1, b2, b3 = _bbox(obj['cells'])
            oh, ow = b1 + 1, b3 + 1
            if oh + 2 > h or ow + 2 > w:
                continue
            obj = _translate(obj, int(rng.integers(1, h - oh)), int(rng.integers(1, w - ow)))
            lead, _ = _elbow(obj)
            if (obj['cells'] & trace or not _separate(obj['cells'], occupied)
                    or not all(_inside(p, (h, w)) for p in lead)):
                continue
            objects.append(obj)
            occupied |= obj['cells']
            break
        else:
            return None
    if len(occupied) >= h * w // 2:
        return None
    return (h, w), launcher, cap, initial, objects


def _render(layout, palette):
    shape, launcher, cap, heading, objects = layout
    bg, signal, body, rail, marker = palette
    a = np.full(shape, bg, dtype=np.int8)
    for obj in objects:
        for p in obj['cells']:
            a[p] = marker if p in obj['marks'] else rail if p in obj['rail'] else body
    for p in launcher:
        a[p] = marker if p == cap else signal
    simulation = _constructor_trace(shape, launcher, cap, heading, objects)
    _require(simulation is not None, "invalid constructed scene")
    b = np.full(shape, bg, dtype=np.int8)
    for p in simulation[0]:
        b[p] = signal
    return {'input': a, 'output': b}


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _count(value, name, lower, upper):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lower <= value <= upper):
        raise ValueError(f'{name} must be an integer from {lower} to {upper}')
    return int(value)


def _ablation_differences(a):
    a, palette, launcher, cap, heading, objects = _decode(a)
    good, facts = _project_trace(a.shape, launcher, cap, heading, objects)
    results = []
    for options in (dict(direct=True), dict(fixed_distance=2)):
        try:
            other, altered = _project_trace(a.shape, launcher, cap, heading, objects, **options)
            results.append((other != good, altered['reached'] != facts['reached']))
        except ValueError:
            # A failed counterfactual is not counted as a differing solved answer.
            results.append((False, False))
    return results


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    _require(num_train > num_test, 'num_train must exceed num_test')
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)) and seed >= 0,
             'seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    maximum = int(rng.choice((2, 3, 4), p=(.4, .4, .2)))
    base_palette = tuple(map(int, rng.choice(10, 5, replace=False)))
    vary_palette = bool(rng.random() < .6)
    shift = int(rng.integers(2))  # held-out entry heading or longer activation chain
    directions = list(map(int, rng.permutation(4)))
    train_dirs = directions[:3] if shift == 0 else directions
    test_dirs = directions[3:] if shift == 0 else directions
    # Random slots, rather than a fixed count order, demonstrate the unary law.
    demo_sizes = ([int(rng.integers(2, 4)), int(rng.integers(1, 4))] if shift == 0
                  else [2, int(rng.integers(1, 3))])
    while sum(demo_sizes) < maximum:
        demo_sizes[1] += 1
    demo_ns = [[int(rng.integers(1, maximum + 1)) for _ in range(k)] for k in demo_sizes]
    slots = [(i, j) for i, ns in enumerate(demo_ns) for j in range(len(ns))]
    slots = [slots[int(i)] for i in rng.permutation(len(slots))]
    for n, slot in zip(range(1, maximum + 1), slots):
        demo_ns[slot[0]][slot[1]] = n
    sign = int(rng.choice((-1, 1)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            demo = split == 'train' and index < 2
            if demo:
                ns = demo_ns[index]
                signs = {int(rng.integers(len(ns))): sign if index == 0 else -sign}
                turn_index = int(rng.integers(len(ns))) if index == 0 else None
                decoys = 1 if index == 0 else int(rng.integers(0, 3))
            else:
                if split == 'test' and shift == 1:
                    k = int(rng.choice((3, 4), p=(.8, .2)))
                else:
                    k = int(rng.choice((0, 1, 2), p=(.15, .3, .55)))
                cap_n = min(maximum, 2) if k == 4 else maximum
                ns = list(map(int, rng.integers(1, cap_n + 1, size=k)))
                signs, turn_index = {}, None
                decoys = int(rng.integers(0 if k else 1, 4))
            heading = int(rng.choice(train_dirs if split == 'train' else test_dirs))
            palette = (tuple(map(int, rng.choice(10, 5, replace=False))) if vary_palette else base_palette)
            for _ in range(90):
                layout = _try_layout(rng, ns, heading, signs, turn_index, decoys)
                if layout is None:
                    continue
                pair = _render(layout, palette)
                if _key(pair['input']) in seen:
                    continue
                if demo and index == 0:
                    # Require a visible change of selected relays, not only of line position.
                    if not _ablation_differences(pair['input'])[0][1]:
                        continue
                if demo and max(ns) > 1 and not _ablation_differences(pair['input'])[1][0]:
                    continue
                break
            else:
                raise RuntimeError('bounded placement exhausted for the retained specification')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    for split in puzzle:
        puzzle[split] = [puzzle[split][int(i)] for i in rng.permutation(len(puzzle[split]))]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'pair lists')
    _count(len(puzzle['train']), 'num_train', 2, 10)
    _count(len(puzzle['test']), 'num_test', 1, 4)
    _require(len(puzzle['train']) > len(puzzle['test']), 'counts')
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape and _key(a) not in seen, 'shape or duplicate')
            seen.add(_key(a))
            predicted, facts = _analyze(a)
            _require(np.array_equal(predicted, b), 'incorrect output')
            bg, signal, _, _, marker = facts['palette']
            _require(set(map(int, np.unique(b))) == {bg, signal}, 'output palette')
            _require(np.all(b[a == signal] == signal) and np.all(b[a == marker] != marker), 'launcher and marks')
            for i, obj in enumerate(facts['objects']):
                color = signal if i in facts['reached'] else bg
                _require(all(int(b[p]) == color for p in obj['cells']), 'exact activation footprint / erasure')
            records[split].append(facts)
    train, test = records['train'], records['test']
    counts = {t[2] for f in train for t in f['transitions']}
    _require(1 in counts and len(counts) >= 2, 'demonstrate counting')
    _require({t[3] for f in train for t in f['transitions']} == {-1, 1}, 'demonstrate both rail ends')
    _require(any(f['active'] >= 2 for f in train) and any(f['decoys'] for f in train), 'chain and decoy demonstrations')
    _require(any(any(p not in f['trace'] for p in o['cells'])
                 for f in train for i, o in enumerate(f['objects']) if i not in f['reached']),
             'omitting reachability must change an answer')
    _require(any(t[0] != t[1] for f in train for t in f['transitions']), 'demonstrate face direction')
    _require({t[2] for f in test for t in f['transitions']} <= counts, 'test counts must be demonstrated')
    ablations = [_ablation_differences(p['input']) for p in puzzle['train']]
    _require(any(x[0][1] for x in ablations), 'endpoint offset must change training reachability')
    _require(any(x[1][0] for x in ablations), 'fixed two-cell clearance must fail')
    # Generated shifts are visible in the returned scenes, not hidden flags.
    _require(bool({f['initial'] for f in test} - {f['initial'] for f in train})
             or max(f['active'] for f in test) > max(f['active'] for f in train), 'missing train/test shift')


def _puzzle_key(puzzle):
    return tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in puzzle[s])


def self_test() -> dict:
    """Bounded 42-seed diversity audit, mutations, and hand-labeled regression."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, 'seed diversity')
    _require(_puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0)), 'determinism')
    shapes, layouts, palettes, initials, counts, active, decoys, faces = set(), set(), set(), set(), set(), set(), set(), set()
    for puzzle in puzzles:
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                out, f = _analyze(pair['input'])
                _require(np.array_equal(out, pair['output']), 'solver agreement')
                shapes.add(f['shape'])
                palettes.add(f['palette'])
                initials.add((split, f['initial']))
                active.add((split, f['active']))
                decoys.add(f['decoys'])
                counts.update(t[2] for t in f['transitions'])
                faces.update((t[1] - t[0]) % 4 for t in f['transitions'])
                normalized = np.empty_like(pair['input'])
                for i, color in enumerate(f['palette']):
                    normalized[pair['input'] == color] = i
                layouts.add(_key(normalized))
    _require(len(shapes) >= 35 and len(layouts) >= 180 and len(palettes) >= 35, 'structural diversity')
    _require(initials == {(s, d) for s in ('train', 'test') for d in range(4)}, 'heading diversity')
    _require(counts == {1, 2, 3, 4} and faces == {0, 1, 3}, 'unary counts and turns')
    _require({0, 1, 2, 3} <= decoys and ('test', 4) in active, 'length and decoy diversity')
    # Literal fixture: down entry; two-marker east face throws downward.
    a = np.zeros((15, 17), dtype=np.int8)
    a[0:2, 5] = 2
    a[2, 5] = 4
    a[5:11, 3:7] = 1
    a[6:10, 6] = (4, 3, 4, 3)
    expected = np.zeros_like(a)
    expected[0:5, 5] = 2
    expected[5:11, 3:7] = 2
    expected[9, 6:10] = 2
    expected[9:13, 9] = 2
    expected[12, 9:17] = 2
    _require(np.array_equal(solve(a), expected), 'literal elbow regression')
    # Literal two-hit fork: direct east emission misses the second relay;
    # the counted exit at row 12 reaches it, then its north face throws left.
    fork = np.zeros((18, 20), dtype=np.int8)
    fork[0:2, 5] = 2
    fork[2, 5] = 4
    fork[5:11, 3:7] = 1
    fork[10, 3] = 0  # Preserve a notch, rather than filling the bounding box.
    fork[6:10, 6] = (4, 3, 4, 3)
    fork[11:14, 12:16] = 1
    fork[11, 13:15] = (3, 4)
    fork[3:7, 15:18] = 1  # Unreachable decoy.
    fork[4:6, 17] = (4, 3)
    fork_expected = np.zeros_like(fork)
    fork_expected[0:5, 5] = 2
    fork_expected[5:11, 3:7] = 2
    fork_expected[10, 3] = 0
    fork_expected[9, 6:10] = 2
    fork_expected[9:13, 9] = 2
    fork_expected[12, 9:12] = 2
    fork_expected[11:14, 12:16] = 2
    fork_expected[9:12, 13] = 2
    fork_expected[9, 11:14] = 2
    fork_expected[0:10, 11] = 2
    _require(np.array_equal(solve(fork), fork_expected), 'literal selective chain regression')
    _require(_ablation_differences(fork)[0][1], 'B omission must change reached objects')
    invalid = a.copy()
    invalid[6:10, 6] = (3, 4, 4, 3)
    try:
        solve(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError('malformed endpoint pattern accepted')
    # Recoloring, rotations, and reflections change no convention.
    permutation = np.array([7, 0, 8, 5, 9, 1, 2, 3, 4, 6], dtype=np.int8)
    for source, target in ((a, expected), (fork, fork_expected)):
        for rotation in range(4):
            for mirror in (False, True):
                inp, out = np.rot90(source, rotation), np.rot90(target, rotation)
                if mirror:
                    inp, out = inp[:, ::-1], out[:, ::-1]
                _require(np.array_equal(solve(permutation[inp]), permutation[out]), 'dihedral and palette invariance')
    for nt in range(2, 11):
        for ne in range(1, min(nt - 1, 4) + 1):
            validate_generated_puzzle(generate_puzzle(171, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    pair = damaged['test'][0]
    f = _analyze(pair['input'])[1]
    point = next(p for p in sorted(f['trace']) if pair['input'][p] == f['palette'][0])
    pair['output'][point] = f['palette'][0]
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError('incomplete route accepted')
    return dict(seeds=42, distinct_puzzles=42, shapes=len(shapes), normalized_inputs=len(layouts),
                palettes=len(palettes), active_counts=sorted(active), marker_counts=sorted(counts))


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
