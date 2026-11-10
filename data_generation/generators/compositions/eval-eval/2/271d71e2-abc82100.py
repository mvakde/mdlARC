"""Ink-budget stencil walkers: registered sweeps paid for by unary ink counters.

Only NumPy and the standard library are used. Structural colors are fixed;
all unused/spent ink roles are supplied by the local stencil labels.
"""
from __future__ import annotations

import json
import numpy as np

BG, GOAL, FRAME, MASK = 0, 7, 8, 9
INK = frozenset(range(1, 7))
DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int64, copy=True)


def _components(a, color, diagonal=False):
    cells = set(map(tuple, np.argwhere(a == color).tolist()))
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    result = []
    while cells:
        start = min(cells)
        cells.remove(start)
        component, pending = {start}, [start]
        while pending:
            r, c = pending.pop()
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in cells:
                    cells.remove(q)
                    component.add(q)
                    pending.append(q)
        result.append(component)
    return result


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _counter_order(bounds, direction):
    """Row-major when the launch direction is rotated to point upward."""
    t, b, l, r = bounds
    if direction == (-1, 0):
        return [(y, x) for y in range(t + 1, b) for x in range(l + 1, r)]
    if direction == (1, 0):
        return [(y, x) for y in range(b - 1, t, -1) for x in range(r - 1, l, -1)]
    if direction == (0, -1):
        return [(y, x) for x in range(l + 1, r) for y in range(b - 1, t, -1)]
    return [(y, x) for x in range(r - 1, l, -1) for y in range(t + 1, b)]


def _frames(a):
    records = []
    for cells in _components(a, FRAME):
        ys, xs = zip(*cells)
        t, b, l, r = min(ys), max(ys), min(xs), max(xs)
        if (b - t < 2 or r - l < 2
                or cells != {(y, x) for y in range(t, b + 1)
                             for x in range(l, r + 1)
                             if y in (t, b) or x in (l, r)}):
            raise ValueError("frame must be a complete one-cell rectangular border")
        region = {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)}
        records.append(dict(bounds=(t, b, l, r), region=region))
    if not records:
        raise ValueError("missing cartridge")
    return records


def _instructions(a, excluded):
    records = []
    for cells in _components(a, MASK, True):
        near = {(r + dr, c + dc) for r, c in cells
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        candidates = []
        for q in sorted(near):
            if not _inside(q, a.shape) or int(a[q]) not in (BG, MASK):
                continue
            for dr, dc in DIRS:
                sl, tl = (q[0] - 2 * dr, q[1] - 2 * dc), (q[0] - dr, q[1] - dc)
                if not _inside(sl, a.shape) or sl in excluded or tl in excluded:
                    continue
                source, target = int(a[sl]), int(a[tl])
                if source in INK and target in INK and source != target:
                    candidates.append(dict(source=source, target=target, q=q,
                                           labels={sl, tl}, cells=cells,
                                           label_direction=(dr, dc),
                                           offsets=tuple(sorted((r - q[0], c - q[1])
                                                                for r, c in cells))))
        if len(candidates) != 1:
            raise ValueError("stencil must have one unique source-target registration")
        records.append(candidates[0])
    if not records or len({x['source'] for x in records}) != len(records):
        raise ValueError("one stencil instruction per locally active source is required")
    labels = [p for rec in records for p in rec['labels']]
    if len(labels) != len(set(labels)):
        raise ValueError("instructions cannot share labels")
    return records


def _decode(a):
    frames = _frames(a)
    cartridge_cells = set().union(*(f['region'] for f in frames))
    instructions = _instructions(a, cartridge_cells)
    by_source = {rec['source']: rec for rec in instructions}
    instruction_cells = set().union(*(rec['cells'] | rec['labels'] for rec in instructions))
    cover = cartridge_cells | instruction_cells
    used_sources = set()
    goal_cells = set()
    for f in frames:
        t, b, l, r = f['bounds']
        sides = [((-1, 0), [(t - 1, c) for c in range(l + 1, r)]),
                 ((1, 0), [(b + 1, c) for c in range(l + 1, r)]),
                 ((0, -1), [(y, l - 1) for y in range(t + 1, b)]),
                 ((0, 1), [(y, r + 1) for y in range(t + 1, b)])]
        readings = []
        for direction, positions in sides:
            dr, dc = direction
            for launch in positions:
                if not _inside(launch, a.shape) or launch in cover:
                    continue
                source = int(a[launch])
                if source not in by_source:
                    continue
                for distance in range(1, 31):
                    q = launch[0] + distance * dr, launch[1] + distance * dc
                    if not _inside(q, a.shape):
                        break
                    if int(a[q]) == GOAL:
                        readings.append((direction, launch, q, distance, by_source[source]))
                        break
                    if int(a[q]) != BG:
                        break
        if len(readings) != 1:
            raise ValueError("cartridge needs one launch and a clear ray to one goal")
        direction, launch, goal, distance, ins = readings[0]
        if goal in goal_cells or launch in cover:
            raise ValueError("routes cannot share input markers")
        order = _counter_order(f['bounds'], direction)
        values = [int(a[p]) for p in order]
        spent = values.count(ins['target'])
        if values != [ins['target']] * spent + [ins['source']] * (len(values) - spent):
            raise ValueError("counter must be a direction-relative spent prefix")
        f.update(direction=direction, launch=launch, goal=goal, distance=distance,
                 instruction=ins, order=order, initial_spent=spent,
                 budget=len(order) - spent)
        cover |= {launch, goal}
        goal_cells.add(goal)
        used_sources.add(ins['source'])
    if used_sources != set(by_source):
        raise ValueError("every instruction needs a cartridge")
    foreground = set(map(tuple, np.argwhere(a != BG).tolist()))
    if cover != foreground:
        raise ValueError("unexplained input cells")
    # Paint cannot destroy a cartridge. Every hypothetical route position,
    # even one beyond exhaustion, must satisfy this geometric family constraint.
    for f in frames:
        for footprint in _footprints(f, a.shape):
            if footprint & cartridge_cells:
                raise ValueError("stamp corridor intersects a cartridge")
    return instructions, frames


def _footprints(f, shape, offsets=None):
    offsets = f['instruction']['offsets'] if offsets is None else offsets
    ar, ac = f['launch']
    dr, dc = f['direction']
    return [{(ar + k * dr + r, ac + k * dc + c) for r, c in offsets
             if _inside((ar + k * dr + r, ac + k * dc + c), shape)}
            for k in range(1, f['distance'] + 1)]


def _walk(f, shape, policy="union"):
    seen, steps, costs = set(), 0, []
    budget = f['budget']
    for footprint in _footprints(f, shape):
        cost = (len(footprint - seen) if policy == 'union'
                else len(footprint) if policy == 'area' else 1)
        if cost > budget:
            return dict(paint=seen, steps=steps, costs=costs, spent=sum(costs),
                        remaining=budget, blocked_cost=cost)
        budget -= cost
        costs.append(cost)
        steps += 1
        seen |= footprint
    return dict(paint=seen, steps=steps, costs=costs, spent=sum(costs),
                remaining=budget, blocked_cost=None)


def _shows_clipping(f, shape):
    steps = _walk(f, shape)['steps']
    size = len(f['instruction']['offsets'])
    return any(0 < len(fp) < size for fp in _footprints(f, shape)[:steps])


def _render(a, instructions, frames, plans):
    out = a.copy()
    for rec in instructions:
        for p in rec['cells'] | rec['labels']:
            out[p] = BG
    for f in frames:
        out[f['launch']] = out[f['goal']] = BG
    painted = {}
    for f, plan in zip(frames, plans):
        rec = f['instruction']
        for i, p in enumerate(f['order']):
            out[p] = rec['target'] if i < f['initial_spent'] + plan['spent'] else rec['source']
        for p in plan['paint']:
            if p in painted and painted[p] != rec['target']:
                raise ValueError("different inks cannot overlap")
            painted[p] = rec['target']
    for p, color in painted.items():
        out[p] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode all conventions from this input, then perform one simultaneous pass."""
    a = _grid(input_grid)
    instructions, frames = _decode(a)
    return _render(a, instructions, frames, [_walk(f, a.shape) for f in frames])


def _rotate(p, direction):
    r, c = p
    if direction == (0, 1):
        return r, c
    if direction == (1, 0):
        return c, -r
    if direction == (0, -1):
        return -r, -c
    return -c, r


def _geometry_key(offsets):
    variants = []
    for d in DIRS:
        for sign in (-1, 1):
            cells = [_rotate((sign * r, c), d) for r, c in offsets]
            r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
            variants.append(tuple(sorted((r - r0, c - c0) for r, c in cells)))
    return min(variants)


def _choice(rng, items):
    return items[int(rng.integers(len(items)))]


def _shape(rng, lesson=None):
    """Procedural right-going geometry; the registration can be a cell or a hole."""
    family = ('solid' if lesson == 'fuel' else 'c' if lesson == 'goal' else
              _choice(rng, ('solid', 'outline', 'c', 'branch', 'growth', 'growth',
                            'chevron', 'longline', 'crossline', 'single')))
    h, w = int(rng.integers(2, 6)), int(rng.integers(2, 6))
    if lesson == 'fuel':
        # At least three longitudinal columns separate the visible label origin
        # from even a center rule rounded to the upper/left central cell.
        h, w = int(rng.integers(2, 4)), int(rng.integers(3, 5))
    if family == 'single':
        cells = {(0, 0)}
    elif family == 'longline':
        cells = {(0, c) for c in range(int(rng.integers(2, 7)))}
    elif family == 'crossline':
        cells = {(r, 0) for r in range(int(rng.integers(2, 7)))}
    elif family in ('solid', 'outline', 'c'):
        h = max(3, h) if family == 'c' else h
        cells = {(r, c) for r in range(h) for c in range(w)
                 if family == 'solid' or r in (0, h - 1) or c == w - 1
                 or family == 'outline' and c == 0}
    elif family == 'branch':
        cells = {(r, 0) for r in range(h)} | {(h // 2, c) for c in range(w)}
        cells |= {(r, w - 1) for r in range(h // 2, h)}
    elif family == 'chevron':
        radius = int(rng.integers(1, 4))
        cells = {(r, radius - abs(r)) for r in range(-radius, radius + 1)}
    else:
        cells = {(int(rng.integers(h)), 0)}
        for _ in range(int(rng.integers(4, h * w + 1)) * 3):
            r, c = _choice(rng, sorted(cells))
            dr, dc = _choice(rng, DIRS)
            if 0 <= r + dr < h and 0 <= c + dc < w:
                cells.add((r + dr, c + dc))
    if family == 'c':
        # An open left side admits labels pointing into a true bounding-box hole.
        q = (1, 1) if w >= 3 else (1, 0)
    else:
        left = min(c for r, c in cells)
        q = _choice(rng, sorted(p for p in cells if p[1] == left))
        if lesson is None and rng.random() < .18:
            q = q[0], q[1] - 1  # adjacent exterior registration hole
    offsets = tuple(sorted((r - q[0], c - q[1]) for r, c in cells))
    if rng.random() < .5:
        offsets = tuple(sorted((-r, c) for r, c in offsets))
    return offsets


def _make_module(rng, source, target, direction, lesson=None, novel=None):
    for _ in range(100):
        offsets = _shape(rng, lesson)
        if novel is None or _geometry_key(offsets) not in novel:
            break
    else:
        return None
    if lesson:
        ih, iw = int(rng.integers(3, 7)), int(rng.integers(3, 7))
        distance = int(rng.integers(4, 9) if lesson == 'fuel' else rng.integers(2, 5))
    else:
        ih, iw = int(rng.integers(1, 7)), int(rng.integers(1, 7))
        distance = int(rng.integers(1, 11))
    launch = int(rng.integers(1, ih + 1)), iw + 2
    goal = launch[0], launch[1] + distance
    frame = {(r, c) for r in range(ih + 2) for c in range(iw + 2)}
    reserve = frame | {(launch[0], launch[1] + k) for k in range(distance + 1)}
    reserve |= {(launch[0] + r, launch[1] + k + c)
                for k in range(1, distance + 1) for r, c in offsets}
    transformed = {_rotate(p, direction) for p in reserve}
    y0, x0 = min(r for r, c in transformed), min(c for r, c in transformed)

    def relocate(p):
        r, c = _rotate(p, direction)
        return r - y0, c - x0

    drawing = {}
    for p in frame:
        r, c = p
        drawing[relocate(p)] = FRAME if r in (0, ih + 1) or c in (0, iw + 1) else source
    drawing[relocate(launch)], drawing[relocate(goal)] = source, GOAL
    region = {relocate(p) for p in frame}
    ys, xs = zip(*region)
    return dict(drawing=drawing, reserve={relocate(p) for p in reserve}, region=region,
                bounds=(min(ys), max(ys), min(xs), max(xs)), direction=direction,
                launch=relocate(launch), goal=relocate(goal), distance=distance,
                instruction=dict(source=source, target=target,
                                 offsets=tuple(sorted(_rotate(p, direction) for p in offsets))),
                lesson=lesson, capacity=ih * iw)


def _positions(occupied, footprint):
    """Fit actual swept cells, not a bank of layout slots."""
    h, w = max(r for r, c in footprint) + 1, max(c for r, c in footprint) + 1
    H, W = occupied.shape
    if h > H or w > W:
        return []
    padded = np.pad(occupied, 1)
    blocked = np.zeros_like(occupied)
    for dr in range(3):
        for dc in range(3):
            blocked |= padded[dr:dr + H, dc:dc + W]
    good = np.ones((H - h + 1, W - w + 1), dtype=bool)
    for r, c in footprint:
        good &= ~blocked[r:r + good.shape[0], c:c + good.shape[1]]
    return np.argwhere(good).tolist()


def _shift_module(m, y, x):
    m = dict(m)
    shift = lambda p: (p[0] + y, p[1] + x)
    for key in ('reserve', 'region'):
        m[key] = set(map(shift, m[key]))
    m['drawing'] = {shift(p): c for p, c in m['drawing'].items()}
    m['launch'], m['goal'] = shift(m['launch']), shift(m['goal'])
    t, b, l, r = m['bounds']
    m['bounds'] = t + y, b + y, l + x, r + x
    return m


def _place_instruction(rng, grid, occupied, modules, record):
    offsets = record['offsets']
    options = []
    for dr, dc in DIRS:
        shape = set(offsets) | {(0, 0), (-dr, -dc), (-2 * dr, -2 * dc)}
        r0, c0 = min(r for r, c in shape), min(c for r, c in shape)
        shifted = {(r - r0, c - c0) for r, c in shape}
        for y, x in _positions(occupied, shifted):
            options.append((y - r0, x - c0, dr, dc))
    if not options:
        return None
    for index in rng.permutation(len(options))[:120]:
        qr, qc, dr, dc = options[int(index)]
        cells = {(qr + r, qc + c) for r, c in offsets}
        sl, tl = (qr - 2 * dr, qc - 2 * dc), (qr - dr, qc - dc)
        trial = grid.copy()
        for p in cells:
            trial[p] = MASK
        trial[sl], trial[tl] = record['source'], record['target']
        excluded = set().union(*(m['region'] for m in modules))
        try:
            parsed = _instructions(trial, excluded)
        except ValueError:
            continue
        matching = [r for r in parsed if r['source'] == record['source']]
        if len(matching) != 1 or matching[0]['offsets'] != offsets:
            continue
        record = matching[0]
        grid[:] = trial
        for p in cells | {sl, tl, (qr, qc)}:
            occupied[p] = True
        return record
    return None


def _set_budget(rng, m, shape, active=False):
    footprints = _footprints(m, shape)
    union, costs, totals = set(), [], []
    for fp in footprints:
        costs.append(len(fp - union))
        union |= fp
        totals.append(len(union))
    cap = m['capacity']
    if m['lesson'] == 'fuel':
        options = []
        for k in range(2, m['distance']):
            for residue in range(1, costs[k]):
                budget = totals[k - 1] + residue
                if (budget <= cap and min(budget, m['distance']) > k
                        and totals[k - 1] > k
                        and budget < sum(len(f) for f in footprints[:k])):
                    options.append(budget)
        if not options:
            return False
        budget = _choice(rng, options)
    elif m['lesson'] == 'goal':
        if not totals[-1] < cap or totals[-1] <= m['distance']:
            return False
        budget = int(rng.integers(totals[-1] + 1, cap + 1))
    else:
        mode = _choice(rng, ('random', 'random', 'fresh', 'zero', 'exact'))
        budget = (cap if mode == 'fresh' else 0 if mode == 'zero'
                  else _choice(rng, [v for v in totals if v <= cap] or [0])
                  if mode == 'exact' else int(rng.integers(cap + 1)))
        if active and (not costs[0] or budget < costs[0]):
            if not 0 < costs[0] <= cap:
                return False
            budget = int(rng.integers(costs[0], cap + 1))
    m['budget'], m['initial_spent'] = budget, cap - budget
    m['order'] = _counter_order(m['bounds'], m['direction'])
    return True


def _make_pair(rng, lesson, direction, repeated_source, targets, novel=None, allow_clipping=True):
    for attempt in range(90):
        count = _choice(rng, (1, 1, 2, 2, 3))
        palette = list(map(int, rng.permutation(sorted(INK - {repeated_source}))))
        sources = [repeated_source] + palette[:count - 1]
        mappings = [(s, targets[0] if i == 0 else _choice(rng, sorted(INK - {s})))
                    for i, s in enumerate(sources)]
        modules = []
        for i, (s, t) in enumerate(mappings):
            m = _make_module(rng, s, t, direction if i == 0 else _choice(rng, DIRS),
                             lesson if i == 0 else None, novel if i == 0 else None)
            if m is None:
                break
            modules.append(m)
        if len(modules) != count:
            continue
        if attempt >= 15:
            H = W = 30
        else:
            minimum = 12 if count == 1 else 19
            H, W = int(rng.integers(minimum, 31)), int(rng.integers(minimum, 31))
        grid = np.zeros((H, W), dtype=np.int64)
        occupied = np.zeros((H, W), dtype=bool)
        placed = []
        for i, m in enumerate(modules):
            choices = _positions(occupied, m['reserve'])
            if not choices:
                break
            # Free placement, with occasional shared rows or columns.
            if placed and rng.random() < .35:
                aligned = [(y, x) for y, x in choices
                           if y == placed[0]['bounds'][0] or x == placed[0]['bounds'][2]]
                if aligned:
                    choices = aligned
            y, x = _choice(rng, choices)
            m = _shift_module(m, y, x)
            for p, color in m['drawing'].items():
                grid[p] = color
            for p in m['reserve']:
                occupied[p] = True
            placed.append(m)
        if len(placed) != count:
            continue
        records = []
        for m in placed:
            rec = _place_instruction(rng, grid, occupied, placed, m['instruction'])
            if rec is None:
                break
            m['instruction'] = rec
            records.append(rec)
        if len(records) != count:
            continue
        # Crop around actual input controls, rather than around future ink. This
        # can clip the brush while keeping both the launch and destination visible.
        ys, xs = np.nonzero(grid if allow_clipping else occupied)
        if rng.random() < .80:
            t = int(ys.min()) - int(rng.integers(min(3, int(ys.min())) + 1))
            l = int(xs.min()) - int(rng.integers(min(3, int(xs.min())) + 1))
            b = int(ys.max()) + 1 + int(rng.integers(min(3, H - 1 - int(ys.max())) + 1))
            r = int(xs.max()) + 1 + int(rng.integers(min(3, W - 1 - int(xs.max())) + 1))
            grid = grid[t:b, l:r].copy()
            placed = [_shift_module(m, -t, -l) for m in placed]
            # Stencil offsets are translation-invariant; absolute controls shift.
            records = [{**rec, 'q': (rec['q'][0] - t, rec['q'][1] - l),
                        'cells': {(y - t, x - l) for y, x in rec['cells']},
                        'labels': {(y - t, x - l) for y, x in rec['labels']}}
                       for rec in records]
            for m, rec in zip(placed, records):
                m['instruction'] = rec
        if not all(_set_budget(rng, m, grid.shape, active=(i == 0 and novel is not None))
                   for i, m in enumerate(placed)):
            continue
        for m in placed:
            for i, p in enumerate(m['order']):
                grid[p] = m['instruction']['target'] if i < m['initial_spent'] else m['instruction']['source']
        plans = [_walk(m, grid.shape) for m in placed]
        output = _render(grid, records, placed, plans)
        try:
            decoded_records, decoded_frames = _decode(grid)
            predicted = _render(grid, decoded_records, decoded_frames,
                                [_walk(f, grid.shape) for f in decoded_frames])
        except ValueError:
            continue
        _need(np.array_equal(output, predicted), "sampled plans and input decoding disagree")
        return {'input': grid, 'output': output}
    raise RuntimeError("bounded geometric construction exhausted")


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    colors = list(map(int, rng.permutation(sorted(INK))))
    shared_source, target0, target1 = colors[:3]
    first_direction = _choice(rng, DIRS)
    second_direction = _choice(rng, [d for d in DIRS if d[0] * first_direction[0] + d[1] * first_direction[1] == 0])
    lessons = [('fuel', first_direction, target0), ('goal', second_direction, target1)]
    while len(lessons) < num_train:
        lessons.append((None, _choice(rng, DIRS), _choice(rng, sorted(INK - {shared_source}))))
    rng.shuffle(lessons)
    puzzle, seen, training_geometry = {'train': [], 'test': []}, set(), set()
    training_clipping = False
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            lesson, d, target = (lessons[i] if split == 'train' else
                                 (None, _choice(rng, DIRS), _choice(rng, sorted(INK - {shared_source}))))
            for _ in range(12):
                pair = _make_pair(rng, lesson, d, shared_source, (target,),
                                  novel=training_geometry if split == 'test' else None,
                                  allow_clipping=(split == 'train' or training_clipping))
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError("failed to produce distinct inputs")
            seen.add(key)
            puzzle[split].append(pair)
            if split == 'train':
                records, frames = _decode(pair['input'])
                training_geometry.update(_geometry_key(rec['offsets']) for rec in records)
                training_clipping |= any(_shows_clipping(f, pair['input'].shape) for f in frames)
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(a):
    return a.shape, tuple(map(int, a.flat))


def _puzzle_key(p):
    return tuple((_key(np.asarray(pair['input'])), _key(np.asarray(pair['output'])))
                 for split in ('train', 'test') for pair in p[split])


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "split types")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, evidence, train_geometry, target_changes, test_geometry = set(), set(), set(), {}, []
    training_clipping = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair keys")
            a, out = _grid(pair['input']), _grid(pair['output'])
            _need(a.shape == out.shape, "changed dimensions")
            key = _key(a)
            _need(key not in seen, "duplicate input")
            seen.add(key)
            records, frames = _decode(a)
            plans = [_walk(f, a.shape) for f in frames]
            clipping = any(_shows_clipping(f, a.shape) for f in frames)
            if split == 'train':
                training_clipping |= clipping
            else:
                _need(not clipping or training_clipping, "test clipping needs active training evidence")
            _need(np.array_equal(_render(a, records, frames, plans), out), "incorrect output")
            _need(not np.any(out == MASK) and not np.any(out == GOAL), "unerased instruction/route controls")
            _need(np.array_equal(a == FRAME, out == FRAME), "cartridges must stay fixed")
            for f, plan in zip(frames, plans):
                _need(plan['spent'] == len(plan['paint']), "ink cost must equal union area")
                _need(f['budget'] - plan['spent'] == plan['remaining'] >= 0, "ink accounting")
                _need(plan['steps'] <= f['distance'], "passed destination")
                _need(plan['blocked_cost'] is None or plan['blocked_cost'] > plan['remaining'], "premature stop")
                target = f['instruction']['target']
                _need(sum(out[p] == target for p in f['order']) == f['initial_spent'] + plan['spent'], "wrong counter update")
                if split == 'train':
                    if 0 < plan['steps'] < f['distance'] and plan['remaining'] > 0:
                        evidence.add('residual_stop')
                    if plan['steps'] == f['distance'] and plan['remaining'] > 0 and plan['spent'] > plan['steps']:
                        evidence.add('goal')
                    if plan['steps'] > _walk(f, a.shape, 'area')['steps']:
                        evidence.add('overlap')
                    if plan['steps'] != min(f['budget'], f['distance']):
                        evidence.add('not_unit_fuel')
                    t, b, l, r = f['bounds']
                    if b - t > 2 and r - l > 2 and 0 < f['initial_spent'] + plan['spent'] < len(f['order']):
                        evidence.add('raster')
            geometry = {_geometry_key(rec['offsets']) for rec in records}
            if split == 'train':
                train_geometry |= geometry
                if any((0, 0) not in rec['offsets'] for rec in records):
                    evidence.add('hole')
                if any((0, 0) in rec['offsets'] and
                       (sum(r for r, c in rec['offsets']) or sum(c for r, c in rec['offsets']))
                       for rec in records):
                    evidence.add('offcenter')
                for rec in records:
                    target_changes.setdefault(rec['source'], set()).add(rec['target'])
            else:
                test_geometry.append(geometry)
    _need({'residual_stop', 'goal', 'overlap', 'not_unit_fuel', 'raster', 'hole', 'offcenter'} <= evidence,
          "demonstrations do not expose the interaction")
    _need(any(len(v) > 1 for v in target_changes.values()), "training does not teach local ink mappings")
    _need(all(g - train_geometry for g in test_geometry), "test must contain fresh stencil geometry")


def self_test():
    """Bounded 42-seed coverage, count validation and semantic regression probes."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _need(len({_puzzle_key(p) for p in puzzles}) == 42, "seed diversity")
    _need(_puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0)), "determinism")
    shapes, geometries, capacities, distances, directions, labels, counts, palettes = [set() for _ in range(8)]
    clipped = fresh = zero = residual = 0
    for p in puzzles:
        for split in ('train', 'test'):
            for pair in p[split]:
                a = pair['input']
                original = a.copy()
                _need(np.array_equal(solve(a), pair['output']), "input-only solve")
                _need(np.array_equal(a, original), "solve mutated input")
                records, frames = _decode(a)
                shapes.add(a.shape)
                counts.add(len(frames))
                palettes.add(tuple(sorted((r['source'], r['target']) for r in records)))
                for rec in records:
                    geometries.add(_geometry_key(rec['offsets']))
                    labels.add(rec['label_direction'])
                for f in frames:
                    capacities.add(len(f['order']))
                    distances.add(f['distance'])
                    directions.add(f['direction'])
                    footprints = _footprints(f, a.shape)
                    clipped += any(len(fp) < len(f['instruction']['offsets']) for fp in footprints)
                    fresh += f['initial_spent'] == 0
                    zero += f['budget'] == 0
                    plan = _walk(f, a.shape)
                    residual += plan['blocked_cost'] is not None and plan['remaining'] > 0
    _need(len(shapes) >= 35 and len(geometries) >= 25, "geometric variation")
    _need(len(capacities) >= 12 and len(distances) >= 8, "counter/route variation")
    _need(directions == set(DIRS) and labels == set(DIRS), "orientation variation")
    _need({1, 2, 3} <= counts and len(palettes) >= 25, "count/local palette variation")
    _need(clipped and fresh and zero and residual >= 42, "boundary/counter regimes")
    for ntr in range(2, 11):
        for nte in range(1, min(4, ntr - 1) + 1):
            p = generate_puzzle(314159, ntr, nte)
            _need(len(p['train']) == ntr and len(p['test']) == nte, "ignored explicit counts")
    for ntr, nte in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                     (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, ntr, nte)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Manual cost regressions, independent of the random sampler:
    manual_input = np.array([[int(c) for c in row] for row in (
        '129990000000', '009990000000', '000000000000', '888880000000',
        '811181000007', '811180000000', '811180000000', '888880000000')])
    manual_output = np.array([[int(c) for c in row] for row in (
        '000000000000', '000000000000', '000000000000', '888880000000',
        '822280222200', '822280222200', '812280000000', '888880000000')])
    _need(np.array_equal(solve(manual_input), manual_output), "complete input-only overlap/counter regression")
    # Two reservoirs pay independently even when their printed unions overlap.
    # Probe both a shared instruction and two sources sharing a target. Each
    # seven-pixel stamp costs seven, although the merged output has 13 pixels.
    for second_source in (1, 3):
        a = np.zeros((16, 16), dtype=int)
        expected = np.zeros_like(a)
        for top, source in ((2, 1), (8, second_source)):
            a[top:top + 5, :5] = FRAME
            a[top + 1:top + 4, 1:4] = source
            a[top + 1, 5], a[top + 1, 8] = source, GOAL
            expected[top:top + 5, :5] = FRAME
            expected[top + 1:top + 4, 1:4] = 2
            expected[top + 2:top + 4, 1] = source
        a[:7, 9] = MASK
        a[0, 7:9] = (1, 2)
        if second_source == 3:
            a[8:15, 14] = MASK
            a[8, 12:14] = (3, 2)
        expected[3:16, 6] = 2
        _need(np.array_equal(solve(a), expected), "independent reservoirs/compatible overlap regression")
    # a 2x3 brush costs 6,2,2,...; nine ink buys two steps with one left.
    f = dict(launch=(3, 2), direction=(0, 1), distance=6, budget=9,
             instruction=dict(offsets=tuple((r, c) for r in (0, 1) for c in range(3))))
    plan = _walk(f, (12, 12))
    _need((plan['steps'], plan['spent'], plan['remaining'], plan['blocked_cost']) == (2, 8, 1, 2), "atomic overlap regression")
    _need(_walk(f, (12, 12), 'area')['steps'] == 1, "area shortcut witness")
    _need(_walk(f, (12, 12), 'unit')['steps'] == 6, "unit-fuel shortcut witness")
    f['budget'] = 10
    other = {**f, 'instruction': dict(offsets=tuple((r, c) for r in range(3) for c in range(2)))}
    _need(_walk(f, (12, 12))['steps'] == 3 and _walk(other, (12, 12))['steps'] == 2,
          "equal-area brushes must retain different geometric frontier costs")
    # At the right edge, a clipped 2x3 brush costs 4 then 0. Exhausted ink
    # can pay a zero-cost final move; the distance cap still stops the walk.
    f.update(launch=(3, 8), distance=2, budget=4)
    _need(_walk(f, (12, 11))['costs'] == [4, 0], "clipping/zero-cost regression")
    # A hole registration is a translation origin, not an extra painted cell.
    f.update(launch=(3, 2), distance=1, budget=2,
             instruction=dict(offsets=((-1, 0), (1, 0))))
    _need(_walk(f, (10, 10))['paint'] == {(2, 3), (4, 3)}, "hole registration regression")
    broken = {s: [{k: g.copy() for k, g in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("wrong output accepted")
    return dict(seeds=42, distinct_puzzles=42, shapes=len(shapes),
                stencil_geometries=len(geometries), capacities=len(capacities),
                route_distances=len(distances), cartridge_counts=sorted(counts),
                clipped_routes=clipped, fresh_counters=fresh, empty_counters=zero,
                residual_stops=residual)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
