"""The nth collision-free registered stencil landing beyond a hollow tip.

Only NumPy and the standard library are used. Background 0 and stencil 8 are
fixed visible roles; every other color is local. No parent code is imported.
"""
from __future__ import annotations

import json
import numpy as np


BG, NEUTRAL = 0, 8
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
COLORS = (1, 2, 3, 4, 5, 6, 7, 9)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _choose(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]


def _add(p, q):
    return p[0] + q[0], p[1] + q[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _halo(cells, radius=1):
    return {(r + dr, c + dc) for r, c in cells
            for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1)}


def _components(cells):
    todo = set(cells)
    components = []
    while todo:
        start = min(todo)
        todo.remove(start)
        component, stack = {start}, [start]
        while stack:
            p = stack.pop()
            for q in _halo({p}) - {p}:
                if q in todo:
                    todo.remove(q)
                    component.add(q)
                    stack.append(q)
        components.append(component)
    return components


def _array(grid):
    value = np.asarray(grid)
    if (value.ndim != 2 or not all(1 <= n <= 30 for n in value.shape)
            or not np.issubdtype(value.dtype, np.integer)
            or not np.all((0 <= value) & (value <= 9))):
        raise ValueError("expected an integer ARC grid with axes 1..30")
    return value.astype(np.int8, copy=True)


def _parse_instructions(grid):
    masks = _components(map(tuple, np.argwhere(grid == NEUTRAL)))
    if not masks:
        raise ValueError("no stencil instruction")
    records = []
    for mask in masks:
        options = []
        for q in sorted(_halo(mask)):
            if not _inside(q, grid.shape) or grid[q] not in (BG, NEUTRAL):
                continue
            for dr, dc in DIRECTIONS:
                target_pos = (q[0] - dr, q[1] - dc)
                source_pos = (q[0] - 2 * dr, q[1] - 2 * dc)
                if not _inside(source_pos, grid.shape) or not _inside(target_pos, grid.shape):
                    continue
                source, target = int(grid[source_pos]), int(grid[target_pos])
                if source in (BG, NEUTRAL) or target in (BG, NEUTRAL, source):
                    continue
                options.append(dict(mask=set(mask), registration=q,
                    labels={source_pos, target_pos}, source=source, target=target,
                    direction=(dr, dc),
                    offsets=tuple(sorted((r - q[0], c - q[1]) for r, c in mask))))
        if len(options) != 1:
            raise ValueError("stencil needs one unique label/registration reading")
        records.append(options[0])
    labels = [p for m in records for p in m['labels']]
    if len(set(labels)) != len(labels) or len({m['source'] for m in records}) != len(records):
        raise ValueError("shared labels or repeated source instruction")
    return records


def _frame_read(cells, direction):
    """Recognize monotone thin sides from cross-sections, not a template bank."""
    dr, dc = direction
    # t grows inward; u is the transverse coordinate.
    coordinates = {p: (-p[0] * dr - p[1] * dc, -p[0] * dc + p[1] * dr)
                   for p in cells}
    levels = [t for t, u in coordinates.values()]
    t0, t1 = min(levels), max(levels)
    if t1 - t0 < 3:
        return None
    rows = [sorted(u for t, u in coordinates.values() if t == level)
            for level in range(t0, t1 + 1)]
    if any(not row for row in rows) or len(rows[0]) != 1 or len(rows[-1]) != 2:
        return None
    axis = rows[0][0]
    left = right = axis
    cavity = set()
    split = False
    for level, row in zip(range(t0 + 1, t1 + 1), rows[1:]):
        if len(row) == 1 and not split and row[0] == axis:
            continue
        if len(row) != 2:
            return None
        lo, hi = row
        if lo > left or hi < right or left - lo > 1 or hi - right > 1:
            return None
        left, right, split = lo, hi, True
        for u in range(lo + 1, hi):
            cavity.add((-level * dr - u * dc, -level * dc + u * dr))
    if not cavity or not left < axis < right:
        return None
    tip = next(p for p, local in coordinates.items() if local == (t0, axis))
    return tip, cavity


def _parse_launchers(grid, instructions):
    mappings = {m['source']: m for m in instructions}
    excluded = set().union(*(m['mask'] | m['labels'] for m in instructions))
    candidates = []
    for color in sorted(set(int(c) for c in np.unique(grid)) - {BG, NEUTRAL}):
        for cells in _components({tuple(p) for p in np.argwhere(grid == color)} - excluded):
            if len(cells) < 7:
                continue
            readings = []
            for direction in DIRECTIONS:
                geometry = _frame_read(cells, direction)
                if geometry is None:
                    continue
                tip, cavity = geometry
                if any(not _inside(p, grid.shape) for p in cavity) or cavity & excluded:
                    continue
                values = {int(grid[p]) for p in cavity} - {BG}
                if len(values) != 1:
                    continue
                source = next(iter(values))
                if source not in mappings or source == color:
                    continue
                payload = {p for p in cavity if grid[p] == source}
                if not payload or len(payload) == len(cavity):
                    continue
                readings.append(dict(frame=set(cells), frame_color=color,
                    cavity=cavity, tip=tip, direction=direction, source=source,
                    quantity=len(payload), payload=payload, stencil=mappings[source]))
            if len(readings) > 1:
                raise ValueError("ambiguous frame direction")
            candidates.extend(readings)
    # A payload pattern is owned by its enclosing frame even if it resembles one.
    launchers = [x for x in candidates if not any(
        x is not y and x['frame'] <= y['cavity'] for y in candidates)]
    if not launchers:
        raise ValueError("no hollow launcher")
    owned = set()
    for item in launchers:
        region = item['frame'] | item['cavity']
        if region & owned:
            raise ValueError("overlapping launcher ownership")
        owned |= region
    if {x['source'] for x in launchers} != set(mappings):
        raise ValueError("every instruction must address a launcher")
    return sorted(launchers, key=lambda x: x['tip'])


def _decode(grid):
    instructions = _parse_instructions(grid)
    launchers = _parse_launchers(grid, instructions)
    clear = grid.copy()
    erased = set().union(*(m['mask'] | m['labels'] for m in instructions),
                         *(x['payload'] for x in launchers))
    for p in erased:
        clear[p] = BG
    return clear, instructions, launchers, erased


def _admissible(clear, launcher, offsets=None):
    offsets = launcher['stencil']['offsets'] if offsets is None else offsets
    dr, dc = launcher['direction']
    r, c = launcher['tip']
    landings = []
    distance = 0
    while True:
        distance += 1
        r, c = r + dr, c + dc
        if not _inside((r, c), clear.shape):
            break
        footprint = {(r + dy, c + dx) for dy, dx in offsets}
        if all(_inside(p, clear.shape) and clear[p] == BG for p in footprint):
            landings.append(((r, c), footprint, distance))
    return landings


def _paint(clear, launchers, variant=None):
    output = clear.copy()
    paint, choices = {}, []
    for item in launchers:
        offsets = item['stencil']['offsets']
        quantity = item['quantity']
        if variant == 'first':
            quantity = 1
        elif variant == 'components':
            quantity = len(_components(item['payload']))
        elif variant == 'point':
            offsets = ((0, 0),)
        elif variant == 'box':
            rs, cs = zip(*offsets)
            offsets = tuple((r, c) for r in range(min(rs), max(rs) + 1)
                            for c in range(min(cs), max(cs) + 1))
        if variant == 'distance':
            d = item['direction']
            q = _add(item['tip'], (quantity * d[0], quantity * d[1]))
            footprint = {_add(q, o) for o in offsets if _inside(_add(q, o), clear.shape)}
            choice = q, footprint, quantity
        else:
            options = _admissible(clear, item, offsets)
            if len(options) < quantity:
                raise ValueError("not enough admissible landings")
            choice = options[quantity - 1]
        q, footprint, distance = choice
        # For the box shortcut, test fitting on the box but still copy the mask.
        if variant == 'box':
            footprint = {_add(q, o) for o in item['stencil']['offsets']}
        for p in footprint:
            color = item['stencil']['target']
            if p in paint and paint[p] != color:
                raise ValueError("incompatible simultaneous paint")
            paint[p] = color
        choices.append(choice)
    for p, color in paint.items():
        output[p] = color
    return output, choices


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer only from this input; never infer roles from a seed."""
    grid = _array(input_grid)
    clear, instructions, launchers, erased = _decode(grid)
    return _paint(clear, launchers)[0]


def _profile(rng, teaching=False):
    a = int(rng.integers(1 if not teaching else 2, 4))
    b = int(rng.integers(1 if not teaching else 2, 4))
    if rng.random() < .3:
        b = a
    stem, tail = int(rng.integers(2)), int(rng.integers(3, 6))
    frame = {(t, 0) for t in range(stem + 1)}
    cavity = set()
    t, lo, hi = stem, 0, 0
    while lo < a or hi < b:
        dl = lo < a and (hi == b or rng.random() < .8)
        dh = hi < b and (lo == a or rng.random() < .8)
        if not dl and not dh:
            dl, dh = lo < a, lo == a
        lo, hi, t = lo + int(dl), hi + int(dh), t + 1
        frame |= {(t, -lo), (t, hi)}
        cavity |= {(t, u) for u in range(-lo + 1, hi)}
    for _ in range(tail):
        t += 1
        frame |= {(t, -a), (t, b)}
        cavity |= {(t, u) for u in range(-a + 1, b)}
    return frame, cavity


def _payload(rng, cells, n, style):
    ordered = sorted(cells)
    if style == 'random':
        return {ordered[int(i)] for i in rng.choice(len(ordered), n, replace=False)}
    selected = {_choose(rng, ordered)}
    while len(selected) < n:
        remaining = sorted(cells - selected)
        if style == 'compact':
            choices = [p for p in remaining if any(abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1
                                                    for q in selected)]
        else:
            distances = [min(max(abs(p[0] - q[0]), abs(p[1] - q[1])) for q in selected)
                         for p in remaining]
            choices = [p for p, d in zip(remaining, distances) if d == max(distances)]
        selected.add(_choose(rng, choices or remaining))
    return selected


def _turn(offsets, direction):
    # Canonical stencil labels point right; rotate the complete instruction.
    dr, dc = direction
    return tuple(sorted((r * dc + c * dr, -r * dr + c * dc) for r, c in offsets))


def _mask(rng, kind=None):
    kind = kind or _choose(rng, ('growth', 'growth', 'branch', 'line', 'hole', 'hole', 'single'))
    if kind == 'single':
        cells = {(0, 0)}
    elif kind == 'line':
        cells = {(0, c) for c in range(int(rng.integers(2, 6)))}
    elif kind == 'branch':
        a, b = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        cells = {(r, 0) for r in range(-a, b + 1)}
        arm = int(rng.integers(-a, b + 1))
        cells |= {(arm, c) for c in range(1, int(rng.integers(2, 5)))}
    elif kind == 'hole':
        a, b, w = int(rng.integers(1, 3)), int(rng.integers(1, 3)), int(rng.integers(1, 4))
        # The next-cell point must be in the one-cell neighborhood of the
        # component. A wider cavity with all three reaches >=2 would violate
        # that association convention regardless of where it is placed.
        if min(a, b, w) > 1:
            a = 1
        cells = {(r, w) for r in range(-a, b + 1)}
        cells |= {(r, c) for r in (-a, b) for c in range(-1, w + 1)}
    else:
        h, w = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        cells = {(0, 0)}
        desired = int(rng.integers(3, min(16, (2 * h + 1) * w) + 1))
        for _ in range(150):
            if len(cells) >= desired:
                break
            r, c = _choose(rng, sorted(cells))
            dy, dx = _choose(rng, DIRECTIONS)
            p = r + dy, c + dx
            if -h <= p[0] <= h and 0 <= p[1] < w:
                cells.add(p)
    _require(len(_components(cells)) == 1, 'mask construction disconnected')
    return tuple(sorted(cells))


def _geometry_key(offsets):
    variants = []
    for direction in DIRECTIONS:
        for reflection in (1, -1):
            turned = _turn(tuple((reflection * r, c) for r, c in offsets), direction)
            r0, c0 = min(r for r, c in turned), min(c for r, c in turned)
            variants.append(tuple(sorted((r - r0, c - c0) for r, c in turned)))
    return min(variants)


def _positions(relative, shape, rng):
    rs, cs = zip(*relative)
    values = [(r, c) for r in range(-min(rs), shape[0] - max(rs))
              for c in range(-min(cs), shape[1] - max(cs))]
    return [values[int(i)] for i in rng.permutation(len(values))]


def _enough(clear, launchers, extra=True):
    try:
        if any(len(_admissible(clear, x)) < x['quantity'] + int(extra) for x in launchers):
            return False
        _paint(clear, launchers)
        return True
    except ValueError:
        return False


def _make_pair(rng, plans, frame_color, mask_specs, teaching=None):
    for attempt in range(100):
        minimum = 12 if len(plans) == 1 else 20
        shape = (int(rng.integers(minimum, 31)), int(rng.integers(minimum, 31)))
        if attempt >= 50:
            shape = (30, 30)
        inp = np.zeros(shape, dtype=np.int8)
        launchers, protected, obstacle_protected = [], set(), set()
        for plan in plans:
            frame, cavity = plan['profile']
            dr, dc = plan['direction']
            transform = lambda p: (-p[0] * dr - p[1] * dc, -p[0] * dc + p[1] * dr)
            local_frame = {transform(p) for p in frame}
            local_cavity = {transform(p) for p in cavity}
            relative = local_frame | local_cavity
            payload = _payload(rng, cavity, plan['quantity'], plan['style'])
            placed = False
            for tip in _positions(relative, shape, rng):
                region = {_add(tip, p) for p in relative}
                room = (tip[0] if dr < 0 else shape[0] - 1 - tip[0] if dr > 0
                        else tip[1] if dc < 0 else shape[1] - 1 - tip[1])
                if region & protected or room < plan['quantity'] + 5:
                    continue
                record = dict(frame={_add(tip, p) for p in local_frame},
                    cavity={_add(tip, p) for p in local_cavity},
                    payload={_add(tip, transform(p)) for p in payload},
                    tip=tip, direction=plan['direction'], source=plan['source'],
                    frame_color=frame_color, quantity=plan['quantity'])
                for p in record['frame']:
                    inp[p] = frame_color
                for p in record['payload']:
                    inp[p] = plan['source']
                launchers.append(record)
                protected |= _halo(region, 2)
                # Marks can approach a frame without changing its geometry.
                # Only its actual owned region is forbidden to obstacles.
                obstacle_protected |= region
                placed = True
                break
            if not placed:
                break
        if len(launchers) != len(plans):
            continue
        instructions = []
        for source, target, offsets, direction in mask_specs:
            dr, dc = direction
            relative = set(offsets) | {(0, 0), (-dr, -dc), (-2 * dr, -2 * dc)}
            placed = False
            for q in _positions(relative, shape, rng):
                mask = {_add(q, p) for p in offsets}
                labels = {(q[0] - dr, q[1] - dc), (q[0] - 2 * dr, q[1] - 2 * dc)}
                cells = mask | labels | {q}
                if cells & protected:
                    continue
                trial = inp.copy()
                for p in mask:
                    trial[p] = NEUTRAL
                trial[q[0] - dr, q[1] - dc] = target
                trial[q[0] - 2 * dr, q[1] - 2 * dc] = source
                try:
                    parsed = _parse_instructions(trial)
                except ValueError:
                    continue
                if len(parsed) != len(instructions) + 1:
                    continue
                rec = next(m for m in parsed if m['source'] == source)
                if rec['registration'] != q or rec['offsets'] != offsets:
                    continue
                instructions.append(rec)
                inp = trial
                protected |= _halo(cells, 3)
                obstacle_protected |= _halo(cells, 3)
                placed = True
                break
            if not placed:
                break
        if len(instructions) != len(mask_specs):
            continue
        by_source = {m['source']: m for m in instructions}
        for item in launchers:
            item['stencil'] = by_source[item['source']]
        erased = set().union(*(m['mask'] | m['labels'] for m in instructions),
                             *(x['payload'] for x in launchers))
        clear = inp.copy()
        for p in erased:
            clear[p] = BG
        if not _enough(clear, launchers):
            continue
        obstacle_colors = [c for c in COLORS if c != frame_color]
        hole_point = None
        # An obstacle actually rejects an early candidate in the first lesson.
        if teaching == 'occupied':
            primary = launchers[0]
            d = primary['direction']
            positions = []
            for step in range(1, primary['quantity'] + 1):
                q = _add(primary['tip'], (step * d[0], step * d[1]))
                positions += [_add(q, o) for o in primary['stencil']['offsets']]
            success = False
            for p in sorted(set(positions)):
                if not _inside(p, shape) or p in obstacle_protected or clear[p] != BG:
                    continue
                trial = clear.copy()
                trial[p] = primary['source']
                if _enough(trial, launchers):
                    clear, inp[p], success = trial, trial[p], True
                    break
            if not success:
                continue
        # Put a persistent obstacle exactly at the selected empty registration.
        if teaching == 'hole':
            primary = launchers[0]
            success = False
            for q, footprint, distance in _admissible(clear, primary):
                if q in obstacle_protected or clear[q] != BG:
                    continue
                trial = clear.copy()
                trial[q] = primary['source']
                if (_enough(trial, launchers)
                        and _admissible(trial, primary)[primary['quantity'] - 1][0] == q):
                    clear, inp[q], hole_point, success = trial, trial[q], q, True
                    break
            if not success:
                continue
        for _ in range(int(rng.integers(0, 8))):
            for _ in range(12):
                p = (int(rng.integers(shape[0])), int(rng.integers(shape[1])))
                step = _choose(rng, DIRECTIONS)
                cells = {_add(p, (k * step[0], k * step[1]))
                         for k in range(int(rng.integers(1, 4)))}
                if any(not _inside(q, shape) or q in obstacle_protected or clear[q] != BG for q in cells):
                    continue
                trial = clear.copy()
                color = _choose(rng, obstacle_colors)
                for q in cells:
                    trial[q] = color
                if not _enough(trial, launchers):
                    continue
                if hole_point is not None and _admissible(trial, launchers[0])[launchers[0]['quantity'] - 1][0] != hole_point:
                    continue
                for q in cells:
                    inp[q] = color
                clear = trial
                break
        try:
            expected, choices = _paint(clear, launchers)
            observed_clear, parsed, observed, removed = _decode(inp)
            if not np.array_equal(clear, observed_clear) or len(observed) != len(launchers):
                continue
            if not np.array_equal(expected, solve(inp)):
                continue
            if teaching == 'occupied' and not any(s[2] > x['quantity'] for x, s in zip(launchers, choices)):
                continue
            if teaching == 'hole' and not any(_hole_obstacle(clear, x, choice)
                                              for x, choice in zip(launchers, choices)):
                continue
            # Keep a clearly dominant field even in compact inputs.
            if np.count_nonzero(inp == BG) <= inp.size / 2:
                continue
            return {'input': inp, 'output': expected}
        except ValueError:
            continue
    raise RuntimeError('bounded scene construction exhausted')


def _hole_obstacle(clear, item, choice):
    offsets = item['stencil']['offsets']
    rs, cs = zip(*offsets)
    q = choice[0]
    return ((0, 0) not in offsets and min(rs) <= 0 <= max(rs)
            and min(cs) <= 0 <= max(cs) and clear[q] != BG)


def _validate_count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(name + ' out of range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _validate_count(num_train, 'num_train', 2, 10)
    num_test = _validate_count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = [int(c) for c in rng.permutation(COLORS)]
    frame_color, source = palette[:2]
    targets = palette[2:4]
    witness_profile = _profile(rng, teaching=True)
    witness_direction = _choose(rng, DIRECTIONS)
    low, high = int(rng.integers(2, 4)), int(rng.integers(4, 6))
    puzzle, seen, training_geometries = {'train': [], 'test': []}, set(), set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            lesson = index if split == 'train' and index < 2 else None
            for retry in range(30):
                if lesson is not None:
                    plans = [dict(profile=witness_profile, direction=witness_direction,
                                  quantity=n, source=source,
                                  style='sparse' if lesson == 1 and i == 0 else 'compact' if i == 0 else 'random')
                             for i, n in enumerate((low, high))]
                    local_frame = frame_color
                    kind = 'branch' if lesson == 0 else 'hole'
                    offsets = _mask(rng, kind)
                    # Align the occupied lesson's incoming axis with the ray:
                    # its forward branch gives a clear early blocker without
                    # reaching back into its frame. The hole
                    # lesson and ordinary scenes keep independent directions.
                    mask_direction = witness_direction if lesson == 0 else _choose(rng, DIRECTIONS)
                    mask_specs = [(source, targets[lesson], _turn(offsets, mask_direction), mask_direction)]
                    teaching = 'occupied' if lesson == 0 else 'hole'
                else:
                    local_palette = [int(c) for c in rng.permutation(COLORS)]
                    local_frame = local_palette[0]
                    number = int(_choose(rng, (1, 1, 2, 2, 3)))
                    instruction_count = 1 if number == 1 else int(_choose(rng, (1, 1, 2)))
                    sources = local_palette[1:instruction_count + 1]
                    mapping = [(s, _choose(rng, [c for c in COLORS if c not in (s, local_frame)]))
                               for s in sources]
                    mask_specs = []
                    for s, target in mapping:
                        offsets = _mask(rng)
                        if split == 'test' and index == 0 and s == sources[0]:
                            for _ in range(100):
                                if _geometry_key(offsets) not in training_geometries and len(offsets) >= 2:
                                    break
                                offsets = _mask(rng)
                            else:
                                continue
                        direction = _choose(rng, DIRECTIONS)
                        mask_specs.append((s, target, _turn(offsets, direction), direction))
                    if len(mask_specs) != instruction_count:
                        continue
                    plans = []
                    common = _choose(rng, DIRECTIONS)
                    for i in range(number):
                        profile = _profile(rng)
                        upper = min(6, len(profile[1]) - 1)
                        lower = 2 if split == 'test' and index == 0 and i == 0 else 1
                        if upper < lower:
                            break
                        n = int(rng.integers(lower, upper + 1))
                        plans.append(dict(profile=profile, direction=common if rng.random() < .55 else _choose(rng, DIRECTIONS),
                            quantity=n, source=sources[i % len(sources)], style=_choose(rng, ('compact', 'sparse', 'random'))))
                    if len(plans) != number:
                        continue
                    teaching = None
                try:
                    pair = _make_pair(rng, plans, local_frame, mask_specs, teaching)
                except RuntimeError:
                    continue
                key = _grid_key(pair['input'])
                if key in seen:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                if split == 'train':
                    training_geometries |= {_geometry_key(m['offsets']) for m in _parse_instructions(pair['input'])}
                break
            else:
                raise RuntimeError('could not construct a distinct example')
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _profile_key(item):
    dr, dc = item['direction']
    tr, tc = item['tip']
    return tuple(sorted((-(r - tr) * dr - (c - tc) * dc,
                         -(r - tr) * dc + (c - tc) * dr) for r, c in item['frame']))


def _alternative_differs(clear, launchers, out, variant):
    try:
        other = _paint(clear, launchers, variant)[0]
    except ValueError:
        return True  # This competing rule cannot even produce an admissible answer.
    return not np.array_equal(other, out)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    n, m = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, 'invalid counts')
    seen, full_counts, matched, components_by_count = set(), set(), {}, {}
    evidence = {k: False for k in ('skip', 'hole', 'offcenter', 'source_obstacle',
                                  'first', 'point', 'components', 'box', 'distance')}
    train_geometries, test_geometries = set(), set()
    test_stopping = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            inp, out = _array(pair['input']), _array(pair['output'])
            _require(inp.shape == out.shape, 'shape changed')
            key = _grid_key(inp)
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            clear, instructions, launchers, erased = _decode(inp)
            expected, choices = _paint(clear, launchers)
            _require(np.array_equal(expected, out), 'incorrect fitted stencil output')
            _require(not np.array_equal(inp, out), 'unchanged pair')
            _require(np.count_nonzero(inp == BG) > inp.size / 2, 'background is not dominant')
            _require(1 <= len(instructions) <= 2 and 1 <= len(launchers) <= 3, 'unsupported scene counts')
            # Every original solid cell survives; no painted footprint is clipped.
            solids = clear != BG
            _require(np.array_equal(out[solids], clear[solids]), 'overwritten solid cell')
            _require(all(out[p] == x['frame_color'] for x in launchers for p in x['frame']), 'changed frame')
            geometries = {_geometry_key(x['offsets']) for x in instructions}
            (train_geometries if split == 'train' else test_geometries).update(geometries)
            for item, choice in zip(launchers, choices):
                options = _admissible(clear, item)
                _require(choice == options[item['quantity'] - 1], 'wrong ordinal landing')
                _require(all(clear[p] == BG for p in choice[1]), 'invalid fit')
                if split == 'train':
                    full_counts.add(item['quantity'])
                    signature = (_profile_key(item), item['direction'], item['frame_color'], item['source'])
                    matched.setdefault(signature, set()).add(item['quantity'])
                    components_by_count.setdefault(item['quantity'], set()).add(len(_components(item['payload'])))
                    evidence['skip'] |= choice[2] > item['quantity']
                    evidence['hole'] |= _hole_obstacle(clear, item, choice)
                    offsets = item['stencil']['offsets']
                    evidence['offcenter'] |= ((0, 0) in offsets and
                        (sum(r for r, c in offsets) != 0 or sum(c for r, c in offsets) != 0))
                else:
                    test_stopping |= len(options) > item['quantity'] >= 2
            if split == 'train':
                evidence['source_obstacle'] |= any(np.any(clear == x['source']) for x in launchers)
                for variant in ('first', 'point', 'components', 'box', 'distance'):
                    evidence[variant] |= _alternative_differs(clear, launchers, out, variant)
    _require(len(full_counts) >= 2 and any(len(v) >= 2 for v in matched.values()), 'count/geometry teaching insufficient')
    _require(any(len(v) >= 2 for v in components_by_count.values()), 'no rearranged equal-count witness')
    _require(all(evidence.values()), 'missing family evidence: ' + str(evidence))
    _require(test_geometries - train_geometries, 'test must read a new mask geometry')
    _require(test_stopping, 'tests hide ordinal-dependent stopping')


def _fingerprint(puzzle):
    return tuple((_grid_key(p['input']), _grid_key(p['output']))
                 for s in ('train', 'test') for p in puzzle[s])


def _manual_regressions():
    """Small hand-labelled grids, independent of the random scene planner."""
    def start(shape):
        return np.zeros(shape, dtype=np.int8), np.zeros(shape, dtype=np.int8)

    def frame(inp, expected, tip, direction, depth, n):
        dr, dc = direction
        inp[tip] = expected[tip] = 9
        for t in range(1, depth + 1):
            for u in (-1, 1):
                p = (tip[0] - t * dr - u * dc, tip[1] - t * dc + u * dr)
                inp[p] = expected[p] = 9
            if t > depth - n:
                inp[tip[0] - t * dr, tip[1] - t * dc] = 2

    def instruction(inp, q, offsets):
        for p in offsets:
            inp[_add(q, p)] = NEUTRAL
        inp[q[0], q[1] - 2:q[1]] = (2, 3)

    def stamp(expected, q, offsets):
        for p in offsets:
            expected[_add(q, p)] = 3

    fixtures = []
    inp, expected = start((11, 20))
    frame(inp, expected, (5, 5), (0, 1), 4, 2)
    offsets = ((0, 0), (1, 0), (1, 1))
    instruction(inp, (1, 11), offsets)
    inp[6, 7] = expected[6, 7] = 7
    # Trials at columns 6 and 7 hit the obstacle. Columns 8,9 are valid.
    stamp(expected, (5, 9), offsets)
    fixtures.append((inp, expected, 'two rejected trials'))

    inp, expected = start((11, 20))
    frame(inp, expected, (5, 5), (0, 1), 4, 2)
    offsets = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, -1), (1, 0), (1, 1))
    instruction(inp, (1, 12), offsets)
    inp[5, 8] = expected[5, 8] = 7
    # Column 6 fits, column 7 hits the mark, column 8 fits around it.
    stamp(expected, (5, 8), offsets)
    fixtures.append((inp, expected, 'solid registration hole'))

    inp, expected = start((13, 23))
    frame(inp, expected, (7, 4), (0, 1), 3, 2)
    offsets = tuple(sorted({(r, 0) for r in range(4)} | {(3, c) for c in range(-6, 1)}))
    instruction(inp, (1, 11), offsets)
    # Column 5 has an out-of-bounds tail; 6 and 7 are the first full fits.
    stamp(expected, (7, 7), offsets)
    fixtures.append((inp, expected, 'reject rather than clip'))

    inp, expected = start((14, 20))
    frame(inp, expected, (8, 3), (0, 1), 3, 2)
    frame(inp, expected, (5, 5), (1, 0), 5, 3)
    offsets = ((0, 0), (0, 1))
    instruction(inp, (2, 12), offsets)
    # Both select the same two cells. Painting the right-facing one first
    # must not make the downward one skip its third originally empty trial.
    stamp(expected, (8, 5), offsets)
    fixtures.append((inp, expected, 'simultaneous compatible overlap'))
    for inp, expected, name in fixtures:
        for k in range(4):
            _require(np.array_equal(solve(np.rot90(inp, k)), np.rot90(expected, k)), name)
    return len(fixtures)


def self_test():
    """Bounded 42-seed variation audit, legal counts, and semantic regressions."""
    manual_cases = _manual_regressions()
    signatures, dimensions, mask_shapes, frame_shapes = set(), set(), set(), set()
    directions, quantities, counts, palette_roles = set(), set(), set(), set()
    holes = skips = overlaps = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_fingerprint(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                grid = pair['input']
                dimensions.add(grid.shape)
                clear, instructions, launchers, erased = _decode(grid)
                out, choices = _paint(clear, launchers)
                _require(np.array_equal(solve(grid.copy()), pair['output']), 'solver mismatch')
                mask_shapes.update(_geometry_key(x['offsets']) for x in instructions)
                counts.add((len(instructions), len(launchers)))
                footprints = set()
                for item, choice in zip(launchers, choices):
                    frame_shapes.add(_profile_key(item))
                    directions.add(item['direction'])
                    quantities.add(item['quantity'])
                    palette_roles.add((item['frame_color'], item['source'], item['stencil']['target']))
                    holes += _hole_obstacle(clear, item, choice)
                    skips += choice[2] > item['quantity']
                    overlaps += bool(footprints & choice[1])
                    footprints |= choice[1]
        if seed < 3:
            _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), 'nondeterminism')
    _require(len(signatures) == 42, 'seed diversity failure')
    _require(len(dimensions) >= 50 and max(max(s) for s in dimensions) == 30, 'insufficient dimension variation')
    _require(len(mask_shapes) >= 30 and len(frame_shapes) >= 30, 'insufficient geometric variation')
    _require(directions == set(DIRECTIONS) and quantities == set(range(1, 7)), 'direction/count variation failure')
    _require({(1, 1), (1, 2), (2, 2), (1, 3)} <= counts, 'scene count variation failure')
    _require(len(palette_roles) >= 50 and holes >= 42 and skips >= 42, 'role/fit variation failure')
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            generate_puzzle(173 + a * 10 + b, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Input-only geometric equivariance: rotate everything, then recolor only
    # the variable roles (0 and 8 remain the explicit fixed conventions).
    pair = generate_puzzle(909, 2, 1)['train'][1]
    for rotation in range(4):
        x, y = np.rot90(pair['input'], rotation), np.rot90(pair['output'], rotation)
        _require(np.array_equal(solve(x), y), 'rotation regression')
        mapping = np.arange(10, dtype=np.int8)
        for old, new in zip(COLORS, reversed(COLORS)):
            mapping[old] = new
        _require(np.array_equal(solve(mapping[x]), mapping[y]), 'local role recoloring regression')
    # Solver must not modify input; validation must reject a corrupted answer.
    puzzle = generate_puzzle(901, 2, 1)
    saved = puzzle['test'][0]['input'].copy()
    solve(puzzle['test'][0]['input'])
    _require(np.array_equal(saved, puzzle['test'][0]['input']), 'input mutated')
    puzzle['test'][0]['output'] = saved.copy()
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt answer accepted')
    return dict(seeds=42, distinct_puzzles=len(signatures), dimensions=len(dimensions),
                mask_geometries=len(mask_shapes), frame_profiles=len(frame_shapes),
                directions=len(directions), quantities=sorted(quantities),
                scene_counts=sorted(counts), local_role_triples=len(palette_roles),
                hole_obstacle_landings=int(holes), skipped_position_landings=int(skips),
                compatible_stamp_overlaps=int(overlaps),
                hand_labelled_regressions=manual_cases,
                semantic_verification='generator-owned checks only')


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
