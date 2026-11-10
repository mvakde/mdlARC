"""Tip-led travel with waypoint-selected docking of an embedded handle.

Only standard-library modules and NumPy are used. Construction stores a short
route temporarily; the solver and validator reconstruct it from the input.
"""
from __future__ import annotations

import json
import numpy as np


# Clockwise directions in row/column coordinates: up, right, down, left.
_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "colors outside 0..9")
    return a


def _rot(p, turns):
    r, c = p
    for _ in range(turns % 4):
        r, c = c, -r
    return r, c


def _direction(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    _require((dr == 0) != (dc == 0), "leg must be nonzero and orthogonal")
    return (1 if dc > 0 else 3) if dr == 0 else (2 if dr > 0 else 0)


def _distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _components(points):
    remaining, result = set(points), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = r + dr, c + dc
                    if q in remaining:
                        remaining.remove(q)
                        component.add(q)
                        stack.append(q)
        result.append(component)
    return result


def _tip_geometry(points):
    """Read a unique projecting tip and reflection axis from the actual mask."""
    candidates = []
    for p in sorted(points):
        for d in range(4):
            offsets = {_rot((r - p[0], c - p[1]), 1 - d) for r, c in points}
            if any(c > 0 for r, c in offsets):
                continue
            if {x for x in offsets if x[1] == 0} != {(0, 0)}:
                continue
            if not any(r != 0 for r, c in offsets):
                continue
            if {(-r, c) for r, c in offsets} == offsets:
                candidates.append((p, d))
    _require(len(candidates) == 1, "arrow has no unique geometric tip and direction")
    return candidates[0]


def _footprint(colored, tip, direction):
    return {(tip[0] + r, tip[1] + c): color
            for p, color in colored.items() for r, c in [_rot(p, direction - 1)]}


def _inside(points, shape):
    return all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in points)


def _halo(points):
    return {(r + dr, c + dc) for r, c in points
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _render(shape, background, anchors, colored, tip, direction):
    out = np.full(shape, background, dtype=np.int8)
    for p, color in anchors.items():
        out[p] = color
    cells = _footprint(colored, tip, direction)
    _require(_inside(cells, shape), "arrow placement falls outside canvas")
    _require(not set(cells) & set(anchors), "arrow covers a fixed anchor")
    for p, color in cells.items():
        out[p] = color
    return out


def _decode(input_grid):
    grid = _grid(input_grid)
    shape = tuple(grid.shape)
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[np.argmax(counts)])
    _require(counts.max() > grid.size / 2, "background must occupy a strict majority")
    comps = _components(map(tuple, np.argwhere(grid != background)))
    objects = [c for c in comps if len(c) > 1]
    _require(len(objects) == 1, "exactly one separated arrow is required")
    arrow = objects[0]
    tip, initial = _tip_geometry(arrow)
    colored = {_rot((r - tip[0], c - tip[1]), 1 - initial): int(grid[r, c])
               for r, c in arrow}
    anchors, dots = {}, {}
    for comp in comps:
        if len(comp) != 1:
            continue
        p = next(iter(comp))
        if p[0] in (0, shape[0] - 1) or p[1] in (0, shape[1] - 1):
            _require(p[0] not in (0, shape[0] - 1) or p[1] not in (0, shape[1] - 1),
                     "corner anchors are ambiguous")
            anchors[p] = int(grid[p])
        else:
            dots[p] = int(grid[p])
    _require(anchors and dots, "anchors and at least one waypoint are required")
    systems = {}
    for color in sorted(set(anchors.values())):
        points = [p for p, v in anchors.items() if v == color]
        horizontal = len({r for r, c in points}) == 1 and all(c in (0, shape[1] - 1) for r, c in points)
        vertical = len({c for r, c in points}) == 1 and all(r in (0, shape[0] - 1) for r, c in points)
        _require(horizontal != vertical, "anchors do not specify one guide")
        axis = 0 if horizontal else 1  # coordinate changed by docking
        systems[color] = (axis, points[0][axis])
    _require(1 <= len(systems) <= 3, "one to three guide identities are required")
    _require(set(dots.values()) <= set(systems), "waypoint lacks a matching guide")
    registrations = {}
    for color in systems:
        matches = [p for p, v in colored.items() if v == color]
        _require(len(matches) == 1, "each guide needs one unique embedded handle")
        _require(matches[0] != (0, 0), "the travel tip and docking handles are distinct")
        registrations[color] = matches[0]
    payload = set(colored.values()) - set(systems)
    _require(len(payload) == 1 and colored[0, 0] in payload, "arrow needs one non-marker body color")
    _require(sum(v in payload for v in colored.values()) >= 2, "body color must repeat")

    start, remaining, actions = tip, dict(dots), []
    direction = initial
    while remaining:
        rays = {}
        for q in remaining:
            if q[0] == tip[0] or q[1] == tip[1]:
                _require(q != tip, "unvisited waypoint already occupies the current tip")
                d = _direction(tip, q)
                rays.setdefault(d, []).append(q)
        _require(len(rays) == 1, "route is disconnected or has competing occupied rays")
        d, choices = next(iter(rays.items()))
        q = min(choices, key=lambda p: _distance(tip, p))
        if not actions:
            _require(d == initial, "first leg must follow the visible arrow")
        color = remaining.pop(q)
        arrived = _footprint(colored, q, d)
        _require(_inside(arrived, shape), "arrival arrow must fit completely")
        offset = _rot(registrations[color], d - 1)
        reg = q[0] + offset[0], q[1] + offset[1]
        axis, guide = systems[color]
        delta = [0, 0]
        delta[axis] = guide - reg[axis]
        docked = q[0] + delta[0], q[1] + delta[1]
        placed = _footprint(colored, docked, d)
        _require(_inside(placed, shape), "docked arrow must fit completely")
        _require(not (set(arrived) | set(placed)) & _halo(anchors), "arrow must remain separate from anchors")
        _require(not set(remaining) & (_halo(arrived) | _halo(placed)), "future waypoints must remain visible and separate")
        actions.append(dict(source=tip, waypoint=q, color=color, direction=d, choices=len(choices),
                            registration=reg, delta=tuple(delta), docked=docked))
        tip, direction = docked, d
    info = dict(shape=shape, background=background, anchors=anchors, systems=systems,
                colored=colored, registrations=registrations, dots=dots,
                start=start, initial=initial, actions=actions, tip=tip, final=direction)
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Predict from this grid alone, without a seed or generation records."""
    a = _decode(input_grid)
    return _render(a['shape'], a['background'], a['anchors'], a['colored'], a['tip'], a['final'])


def _distinguishes_farthest(info, correct):
    """Counterfactual ablation: keep docking but choose farthest on each ray.

    Read the decoded input geometry and dots, never the constructed route.
    Deliberately do not use future-footprint separation to reject this shortcut:
    require a different terminal grid, impossible continuation, or lost bounds.
    """
    tip, facing = info['start'], info['initial']
    remaining = dict(info['dots'])
    while remaining:
        rays = {}
        for q in remaining:
            if q == tip:
                return True
            if q[0] == tip[0] or q[1] == tip[1]:
                rays.setdefault(_direction(tip, q), []).append(q)
        if len(rays) != 1:
            return True
        direction, choices = next(iter(rays.items()))
        q = max(choices, key=lambda p: _distance(tip, p))
        if len(remaining) == len(info['dots']) and direction != facing:
            return True
        color = remaining.pop(q)
        if not _inside(_footprint(info['colored'], q, direction), info['shape']):
            return True
        offset = _rot(info['registrations'][color], direction - 1)
        axis, guide = info['systems'][color]
        p = list(q)
        p[axis] = guide - offset[axis]
        tip, facing = tuple(p), direction
        if not _inside(_footprint(info['colored'], tip, facing), info['shape']):
            return True
    try:
        result = _render(info['shape'], info['background'], info['anchors'],
                         info['colored'], tip, facing)
    except AssertionError:
        return True
    return not np.array_equal(result, correct)


def _sample_mask(rng):
    # Independent head width, shaft length, rear bands, and hollow/solid bands.
    span = int(rng.integers(1, 3))
    shaft = int(rng.integers(1, 3))
    depth = int(rng.integers(2, 5))
    cells = {(0, -i) for i in range(shaft)}
    for j in range(depth):
        width = min(span, j + 1)
        if j >= 2 and rng.random() < .35:
            width = int(rng.integers(1, span + 1))
        rows = range(-width, width + 1) if rng.random() < .65 else (-width, 0, width)
        cells.update((r, -shaft - j) for r in rows)
    _require(_tip_geometry(cells) == ((0, 0), 1), "constructed mask is ambiguous")
    return tuple(sorted(cells))


def _anchors_for(shape, color, axis, guide, ends):
    if axis == 0:
        return {(guide, 0 if end == 0 else shape[1] - 1): color for end in ends}
    return {(0 if end == 0 else shape[0] - 1, guide): color for end in ends}


def _route(rng, shape, colored, registrations, systems, anchors, length, teaching,
           require_nearest=False, allow_nearest=True):
    """Grow visits using the post-docking tip, checking all earlier decisions."""
    rotations = [{_rot(p, d - 1): v for p, v in colored.items()} for d in range(4)]
    bounds = [(min(r for r, c in v), max(r for r, c in v),
               min(c for r, c in v), max(c for r, c in v)) for v in rotations]
    handles = {(color, d): _rot(p, d - 1) for color, p in registrations.items() for d in range(4)}
    fixed_halo = _halo(anchors)
    first_color = next(iter(systems))
    for _ in range(64):
        axis = systems[first_color][0]
        # Teaching 1: slide across the travel axis. Teaching 2: slide along it.
        movement_axis = 1 - axis if teaching == 1 else axis if teaching == 2 else int(rng.integers(2))
        initial = int(rng.choice((0, 2) if movement_axis == 0 else (1, 3)))
        lowr, highr, lowc, highc = bounds[initial]
        start = (int(rng.integers(max(1, -lowr), min(shape[0] - 1, shape[0] - highr))),
                 int(rng.integers(max(1, -lowc), min(shape[1] - 1, shape[1] - highc))))
        footprint = set(_footprint(colored, start, initial))
        if footprint & fixed_halo:
            continue
        blocked = _halo(footprint) | fixed_halo
        actions, tip = [], start
        for index in range(length):
            candidates = []
            dirs = [initial] if index == 0 else list(range(4))
            if teaching == 2 and index == length - 1:
                dirs = [d for d in dirs if d != initial]
            for d in dirs:
                dr, dc = _DIRS[d]
                lr, hr, lc, hc = bounds[d]
                for distance in range(2, max(shape)):
                    q = tip[0] + dr * distance, tip[1] + dc * distance
                    if not (1 <= q[0] < shape[0] - 1 and 1 <= q[1] < shape[1] - 1) or q in blocked:
                        continue
                    if not (0 <= q[0] + lr and q[0] + hr < shape[0]
                            and 0 <= q[1] + lc and q[1] + hc < shape[1]):
                        continue
                    if any((q[0] == a['source'][0] or q[1] == a['source'][1]) and
                           (_direction(a['source'], q) != a['direction'] or
                            _distance(a['source'], q) <= _distance(a['source'], a['waypoint']))
                           for a in actions):
                        continue
                    if not allow_nearest and any(q[0] == a['source'][0] or q[1] == a['source'][1]
                                                 for a in actions):
                        continue
                    if teaching == 2 and index == 1:
                        prev = actions[-1]['waypoint']
                        if q[0] == prev[0] or q[1] == prev[1]:
                            continue  # omission of the first docking breaks the next leg
                    colors = [first_color] if index == 0 and teaching else list(systems)
                    for color in colors:
                        off = handles[color, d]
                        axis, guide = systems[color]
                        delta = [0, 0]
                        delta[axis] = guide - q[axis] - off[axis]
                        if (index == 0 or index == length - 1) and delta == [0, 0]:
                            continue
                        p = q[0] + delta[0], q[1] + delta[1]
                        if not (0 <= p[0] + lr and p[0] + hr < shape[0]
                                and 0 <= p[1] + lc and p[1] + hc < shape[1]):
                            continue
                        if p == tip:
                            continue
                        arrival = {(q[0] + r, q[1] + c) for r, c in rotations[d]}
                        docked = {(p[0] + r, p[1] + c) for r, c in rotations[d]}
                        if (arrival | docked) & fixed_halo:
                            continue
                        action = dict(source=tip, waypoint=q, color=color, direction=d,
                                      delta=tuple(delta), docked=p)
                        # Encourage turns and multiple selected handles, without a priority rule.
                        weight = 1.0
                        if actions and d != actions[-1]['direction']:
                            weight *= 1.5
                        if color not in {a['color'] for a in actions}:
                            weight *= 1.4
                        candidates.append((action, arrival | docked, weight))
            if not candidates:
                break
            weights = np.array([x[2] for x in candidates], dtype=float)
            action, occupied, _ = candidates[int(rng.choice(len(candidates), p=weights / weights.sum()))]
            actions.append(action)
            blocked |= _halo(occupied) | _halo({action['waypoint']})
            tip = action['docked']
        if len(actions) == length:
            # A farther future dot on an earlier occupied ray makes nearest
            # selection observable. Do not reserve this choice for a test.
            nearest_witness = any(q['waypoint'][0] == a['source'][0] or
                                  q['waypoint'][1] == a['source'][1]
                                  for i, a in enumerate(actions) for q in actions[i + 1:])
            if require_nearest and not nearest_witness:
                continue
            return start, initial, actions
    return None


def _make_pair(rng, n, length, teaching, palette, avoid_mask=None, large=False,
               require_nearest=False, allow_nearest=True):
    for attempt in range(80):
        mask = _sample_mask(rng)
        if mask == avoid_mask:
            continue
        # Small, elongated, square, and large canvases; no destination template.
        minimum = 20 if large else 14
        shape = (int(rng.integers(minimum, 31)), int(rng.integers(minimum, 31)))
        bg, body = palette[:2]
        markers = palette[2:2 + n]
        eligible = [p for p in mask if p[0] != 0 and p[1] != 0]
        handles = [eligible[int(i)] for i in rng.choice(len(eligible), n, replace=False)]
        registrations = dict(zip(markers, handles))
        colored = {p: body for p in mask}
        for color, p in registrations.items():
            colored[p] = color
        systems, anchors = {}, {}
        for j, color in enumerate(markers):
            axis = int(rng.integers(2))
            guide = int(rng.integers(2, shape[axis] - 2))
            if any(v == (axis, guide) for v in systems.values()):
                # Distinct identities may share a geometric line in the rule,
                # but construction uses disjoint guides for clearer evidence.
                guide = int(rng.choice([g for g in range(2, shape[axis] - 2)
                                        if (axis, g) not in systems.values()]))
            ends = (0, 1) if rng.random() < .35 else (int(rng.integers(2)),)
            points = _anchors_for(shape, color, axis, guide, ends)
            if set(points) & _halo(anchors):
                break
            systems[color] = axis, guide
            anchors.update(points)
        if len(systems) != n:
            continue
        route = _route(rng, shape, colored, registrations, systems, anchors, length, teaching,
                       require_nearest=require_nearest, allow_nearest=allow_nearest)
        if route is None:
            continue
        start, initial, actions = route
        inp = _render(shape, bg, anchors, colored, start, initial)
        for action in actions:
            _require(inp[action['waypoint']] == bg, "construction overlaps a waypoint")
            inp[action['waypoint']] = action['color']
        out = _render(shape, bg, anchors, colored, actions[-1]['docked'], actions[-1]['direction'])
        info = _decode(inp)
        _require([a['waypoint'] for a in info['actions']] == [a['waypoint'] for a in actions],
                 "constructed route differs from input-visible decisions")
        if require_nearest and not _distinguishes_farthest(info, out):
            continue
        return {'input': inp, 'output': out}, mask
    raise RuntimeError("bounded route construction exhausted")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a = _count(num_train, 'num_train', 2, 10)
    b = _count(num_test, 'num_test', 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    maximum_systems = int(rng.integers(1, 4))
    stable = bool(rng.random() < .25)
    shared = tuple(map(int, rng.permutation(10)))
    train_lengths = [1, 2]
    if a >= 3:
        train_lengths.append(int(rng.integers(3, 5)))
    train_lengths += [int(rng.integers(2, 5)) for _ in range(a - 3)]
    test_lengths = [int(rng.integers(max(train_lengths) + 1, max(train_lengths) + 3)) for _ in range(b)]
    puzzle, seen, first_mask = {'train': [], 'test': []}, set(), None
    for split, lengths in (('train', train_lengths), ('test', test_lengths)):
        for index, length in enumerate(lengths):
            teaching = index + 1 if split == 'train' and index < 2 else 0
            n = maximum_systems if teaching else int(rng.integers(1, maximum_systems + 1))
            for _ in range(24):
                palette = shared if stable else tuple(map(int, rng.permutation(10)))
                pair, mask = _make_pair(rng, n, length, teaching, palette,
                                        avoid_mask=first_mask if teaching == 2 else None,
                                        large=split == 'test',
                                        require_nearest=split == 'train' and index == 2,
                                        allow_nearest=a >= 3)
                key = _signature(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not construct distinct inputs")
            if teaching == 1:
                first_mask = mask
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return tuple(a.shape), a.tobytes()


def _analyze(pair):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    _require(inp.shape == out.shape, "canvas dimensions must be retained")
    info = _decode(inp)
    predicted = _render(info['shape'], info['background'], info['anchors'], info['colored'], info['tip'], info['final'])
    _require(np.array_equal(predicted, out), "output is not the final docked arrow and anchors")
    # Check rigid color preservation with NumPy's bitmap rotation as well as
    # the coordinate renderer. This is an invariant, not a second solver.
    initial_cells = _footprint(info['colored'], info['start'], info['initial'])
    r0, r1 = min(r for r, c in initial_cells), max(r for r, c in initial_cells)
    c0, c1 = min(c for r, c in initial_cells), max(c for r, c in initial_cells)
    original_mask = np.full((r1 - r0 + 1, c1 - c0 + 1), info['background'], dtype=inp.dtype)
    for r, c in initial_cells:
        original_mask[r - r0, c - c0] = inp[r, c]
    terminal = out.copy()
    for p in info['anchors']:
        terminal[p] = info['background']
    positions = np.argwhere(terminal != info['background'])
    low, high = positions.min(axis=0), positions.max(axis=0)
    terminal_mask = terminal[low[0]:high[0] + 1, low[1]:high[1] + 1]
    _require(np.array_equal(np.rot90(original_mask, info['initial'] - info['final']), terminal_mask),
             "terminal bitmap is not the exact rotated colored input mask")
    for action in info['actions']:
        axis, guide = info['systems'][action['color']]
        offset = _rot(info['registrations'][action['color']], action['direction'] - 1)
        _require(action['docked'][axis] + offset[axis] == guide, "selected handle is off its guide")
        _require(action['delta'][1 - axis] == 0, "docking moved along the guide")
        _require(action['docked'][1 - axis] == action['waypoint'][1 - axis], "other coordinate changed")
    # Concrete counterfactual: correct route/last facing, but omit registration.
    no_b = _render(info['shape'], info['background'], info['anchors'], info['colored'],
                   info['actions'][-1]['waypoint'], info['final'])
    info['b_witness'] = not np.array_equal(no_b, out)
    # Concrete counterfactual: project the unmoved, unrotated original carrier
    # once to the final active guide, as if this were an ordinary B scene.
    last = info['actions'][-1]
    axis, guide = info['systems'][last['color']]
    off = _rot(info['registrations'][last['color']], info['initial'] - 1)
    p = list(info['start'])
    p[axis] = guide - off[axis]
    cells = _footprint(info['colored'], tuple(p), info['initial'])
    no_a = np.full(info['shape'], info['background'], dtype=np.int8)
    for q, color in info['anchors'].items():
        no_a[q] = color
    for q, color in cells.items():
        if _inside([q], info['shape']):
            no_a[q] = color
    info['a_witness'] = not np.array_equal(no_a, out)
    info['feedback'] = any(a['waypoint'][0] != prev['waypoint'][0] and
                           a['waypoint'][1] != prev['waypoint'][1]
                           for prev, a in zip(info['actions'], info['actions'][1:]))
    info['nearest_witness'] = (any(a['choices'] > 1 for a in info['actions']) and
                               _distinguishes_farthest(info, out))
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal example counts")
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "wrong pair keys")
            info = _analyze(pair)
            key = _signature(pair['input'])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            analyses[split].append(info)
    train = analyses['train']
    _require(any(x['a_witness'] for x in train), "training does not expose travel")
    _require(any(x['b_witness'] for x in train), "training does not expose docking")
    _require(any(x['feedback'] for x in train), "training does not expose the shifted next-leg origin")
    _require(any(x['initial'] != x['final'] for x in train), "training does not expose rotation")
    _require(any(len(x['actions']) == 1 for x in train), "a single-event demonstration is required")
    single_teachers = [x for x in train if len(x['actions']) == 1]
    _require(any(x['actions'][0]['delta'] != (0, 0) and
                 x['systems'][x['actions'][0]['color']][0] != x['initial'] % 2 and
                 _rot(x['registrations'][x['actions'][0]['color']], x['initial'] - 1)
                 [x['systems'][x['actions'][0]['color']][0]] != 0
                 for x in single_teachers),
             "single-event training must expose eccentric docking across travel")
    if a >= 3:
        _require(any(x['nearest_witness'] for x in train),
                 "training with at least three examples must expose nearest selection")
    if any(a['choices'] > 1 for x in analyses['test'] for a in x['actions']):
        _require(any(x['nearest_witness'] for x in train),
                 "nearest selection in a test needs a training witness")
    masks = {tuple(sorted(x['colored'])) for x in train}
    _require(len(masks) >= 2, "training needs different visible arrow masks")
    return None


def _fingerprint(puzzle):
    return tuple(tuple((_signature(p['input']), _signature(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def _semantic_regressions():
    """Hand-calculated family members outside the sampler's teaching choices."""
    # A three-cell chevron touching the left edge is a mover, not an anchor.
    inp = np.zeros((12, 16), dtype=np.int8)
    inp[4, 1], inp[3, 0], inp[5, 0] = 1, 2, 1
    inp[4, 10], inp[7, 15] = 2, 2
    expected = np.zeros_like(inp)
    expected[8, 10], expected[7, 9], expected[9, 9], expected[7, 15] = 1, 2, 1, 2
    original = inp.copy()
    _require(np.array_equal(solve(inp), expected), "edge-touching chevron regression")
    _require(np.array_equal(inp, original), "solve modified its input")

    # The handle may lie on the shaft; its transverse offset can be zero.
    shaft = np.zeros((12, 16), dtype=np.int8)
    for r, c, color in ((4, 3, 1), (4, 2, 2), (3, 1, 1), (4, 1, 1), (5, 1, 1),
                         (4, 12, 2), (0, 7, 2)):
        shaft[r, c] = color
    expected = np.zeros_like(shaft)
    for r, c, color in ((4, 8, 1), (4, 7, 2), (3, 6, 1), (4, 6, 1), (5, 6, 1), (0, 7, 2)):
        expected[r, c] = color
    _require(np.array_equal(solve(shaft.tolist()), expected), "shaft-handle regression")

    # A final zero slide is legal even though generated teaching endpoints move.
    aligned = np.zeros((12, 16), dtype=np.int8)
    aligned[4, 3], aligned[3, 2], aligned[5, 2] = 1, 2, 1
    aligned[4, 10], aligned[3, 0] = 2, 2
    expected = np.zeros_like(aligned)
    expected[4, 10], expected[3, 9], expected[5, 9], expected[3, 0] = 1, 2, 1, 2
    _require(np.array_equal(solve(aligned), expected), "zero-slide regression")

    # Reject an extra occupied ray and inconsistent same-colored edge guides.
    branching = aligned.copy()
    branching[8, 3] = 2
    inconsistent = aligned.copy()
    inconsistent[8, 15] = 2
    for bad in (branching, inconsistent):
        try:
            solve(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("ambiguous route or guide accepted")


def self_test():
    """Bounded diversity, input-only semantics, and negative/metamorphic checks."""
    fingerprints, dimensions, masks, routes, system_counts, backgrounds = set(), set(), set(), set(), set(), set()
    initial_dirs, final_dirs, anchor_sides, slide_signs, lengths, axes = set(), set(), set(), set(), set(), set()
    aligned = mixed_turns = bilateral = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        fingerprints.add(_fingerprint(p))
        for split in ('train', 'test'):
            for pair in p[split]:
                x = _analyze(pair)
                _require(np.array_equal(solve(pair['input']), pair['output']), "input-only solver mismatch")
                dimensions.add(x['shape'])
                masks.add(tuple(sorted(x['colored'])))
                system_counts.add(len(x['systems']))
                backgrounds.add(x['background'])
                initial_dirs.add(x['initial'])
                final_dirs.add(x['final'])
                lengths.add(len(x['actions']))
                axes.update(v[0] for v in x['systems'].values())
                route = tuple((a['waypoint'][0] - x['start'][0], a['waypoint'][1] - x['start'][1],
                               a['docked'][0] - x['start'][0], a['docked'][1] - x['start'][1], a['direction'])
                              for a in x['actions'])
                routes.add(route)
                for r, c in x['anchors']:
                    anchor_sides.update(side for side, hit in (('top', r == 0), ('bottom', r == x['shape'][0] - 1),
                                                              ('left', c == 0), ('right', c == x['shape'][1] - 1)) if hit)
                bilateral += len(x['anchors']) > len(x['systems'])
                ds = [a['direction'] for a in x['actions']]
                turns = {(q - d) % 4 for d, q in zip(ds, ds[1:]) if d != q}
                mixed_turns += 1 in turns and 3 in turns
                for a in x['actions']:
                    axis = x['systems'][a['color']][0]
                    slide_signs.add((axis, int(np.sign(a['delta'][axis]))))
                    aligned += a['delta'] == (0, 0)
        _require(min(len(_decode(q['input'])['actions']) for q in p['test']) >
                 max(len(_decode(q['input'])['actions']) for q in p['train']), "missing compositional test shift")
    _require(len(fingerprints) == 42, "puzzles repeat")
    _require(len(dimensions) >= 70 and len(masks) >= 20 and len(routes) >= 150, "insufficient structural variation")
    _require(system_counts == {1, 2, 3} and backgrounds == set(range(10)), "role/count variation missing")
    _require(initial_dirs == final_dirs == set(range(4)), "orientation coverage missing")
    _require(anchor_sides == {'top', 'bottom', 'left', 'right'} and axes == {0, 1}, "guide coverage missing")
    _require({(a, s) for a in (0, 1) for s in (-1, 1)} <= slide_signs, "slide direction coverage missing")
    _require({1, 2, 3, 4, 5} <= lengths and aligned > 0 and bilateral > 0 and mixed_turns > 0,
             "route/boundary variation missing")
    _semantic_regressions()
    p = generate_puzzle(101, 2, 1)
    _require(_fingerprint(p) == _fingerprint(generate_puzzle(101, 2, 1)), "not deterministic")
    _require(all(event['choices'] == 1 for pair in p['train'] + p['test']
                 for event in _decode(pair['input'])['actions']),
             "two-demonstration tests must not introduce nearest choices")
    # Global rotations/reflections and independent color permutations preserve the rule.
    rng = np.random.default_rng(987)
    for turns in (0, 1, 2, 3):
        q = {'train': [], 'test': []}
        for split in q:
            for pair in p[split]:
                palette = rng.permutation(10)
                q[split].append({k: palette[np.rot90(np.fliplr(v), turns)].astype(np.int8) for k, v in pair.items()})
        validate_generated_puzzle(q)
        for pair in q['train'] + q['test']:
            _require(np.array_equal(solve(pair['input']), pair['output']), "metamorphic prediction failed")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            q = generate_puzzle(300 + 10 * a + b, a, b)
            _require(len(q['train']) == a and len(q['test']) == b, "explicit counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")


if __name__ == '__main__':
    self_test()
    print(json.dumps({'self_test': 'passed', 'diversity_seeds': 42}, separators=(',', ':')))
