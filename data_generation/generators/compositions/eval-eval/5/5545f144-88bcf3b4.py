"""Ribbon-limited couriers: rigid heads, unary ribbon, and buffered contours.

Only NumPy and the standard library are used. All solving conventions are
geometric and input-visible; construction metadata is never read by solve.
"""
from __future__ import annotations

import numpy as np

_D4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
_D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _add(p, d, n=1):
    return p[0] + d[0] * n, p[1] + d[1] * n


def _right(d):
    return d[1], -d[0]


def _dot(p, d):
    return p[0] * d[0] + p[1] * d[1]


def _local(p, origin, f, side):
    delta = p[0] - origin[0], p[1] - origin[1]
    return _dot(delta, f), _dot(delta, side)


def _world(p, origin, f, side):
    return _add(_add(origin, f, p[0]), side, p[1])


def _direction(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    _require((dr == 0) != (dc == 0), "Nonorthogonal or repeated route references")
    return (0, 1 if dc > 0 else -1) if dr == 0 else (1 if dr > 0 else -1, 0)


def _components(cells, diagonal=True):
    unseen, result = set(cells), []
    neighbors = _D8 if diagonal else _D4
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        component, stack = {first}, [first]
        while stack:
            p = stack.pop()
            for d in neighbors:
                q = _add(p, d)
                if q in unseen:
                    unseen.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _mask_pose(cells):
    """Read the symmetric, uniquely projecting tip from actual visible cells."""
    matches = []
    for tip in sorted(cells):
        for f in _D4:
            mask = {_local(p, tip, f, _right(f)) for p in cells}
            radius = max(max(abs(u), abs(v)) for u, v in mask)
            if (1 <= radius <= 2 and max(u for u, v in mask) == 0
                    and {(u, v) for u, v in mask if u == 0} == {(0, 0)}
                    and any(v != 0 for u, v in mask)
                    and {(u, -v) for u, v in mask} == mask):
                matches.append((tip, f, tuple(sorted(mask)), radius))
    _require(len(matches) == 1, "Mover must have a unique symmetric pointed pose")
    return matches[0]


def _place_mask(mask, tip, f):
    return {_world(p, tip, f, _right(f)) for p in mask}


def _ordered_references(tip, waypoints, initial):
    remaining, points = set(waypoints), [tip]
    while remaining:
        rays = {}
        for p in sorted(remaining):
            if p[0] == tip[0] or p[1] == tip[1]:
                d = _direction(tip, p)
                rays.setdefault(d, []).append(p)
        _require(len(rays) == 1, "Checkpoints must have exactly one occupied ray")
        d, candidates = next(iter(rays.items()))
        if len(points) == 1:
            _require(d == initial, "The initial checkpoint must be ahead of the tip")
        tip = min(candidates, key=lambda p: abs(p[0] - tip[0]) + abs(p[1] - tip[1]))
        points.append(tip)
        remaining.remove(tip)
    return points


def _unimodal(profile):
    peak = max(profile)
    first, last = profile.index(peak), len(profile) - 1 - profile[::-1].index(peak)
    return (all(a <= b for a, b in zip(profile[:first + 1], profile[1:first + 1]))
            and all(v == peak for v in profile[first:last + 1])
            and all(a >= b for a, b in zip(profile[last:], profile[last + 1:])))


def _contour_profile(obstacle, distance, radius, construction=False):
    """The chosen-side skyline of a square-buffered obstacle.

    The solver uses an analytic maximum over actual cells. Construction can
    instead dilate the bitmap explicitly, providing a second implementation
    of this particular geometric primitive.
    """
    if not obstacle:
        return [0] * (distance + 1)
    if construction:
        forbidden = {(u + du, v + dv) for u, v in obstacle
                     for du in range(-radius, radius + 1)
                     for dv in range(-radius, radius + 1)}
        profile = [max([0] + [v + 1 for t, v in forbidden if t == u])
                   for u in range(distance + 1)]
    else:
        profile = [max([0] + [v + radius + 1 for t, v in obstacle
                             if abs(t - u) <= radius])
                   for u in range(distance + 1)]
    _require(profile[0] == profile[-1] == 0, "Blocker halo reaches a checkpoint")
    _require(_unimodal(profile), "Only a single exposed hill is supported per leg")
    return profile


def _trace_profile(profile):
    """Orthogonal exterior boundary: rise before entering, fall after leaving."""
    path = [(0, 0)]
    for u, (a, b) in enumerate(zip(profile, profile[1:])):
        high = max(a, b)
        path.extend((u, v) for v in range(a + 1, high + 1))
        path.append((u + 1, high))
        path.extend((u + 1, v) for v in range(high - 1, b - 1, -1))
    return path


def _journey(points, obstacles, radius, hand, construction=False):
    groups = _components(obstacles)
    assigned = [[] for _ in range(len(points) - 1)]
    for group in groups:
        matches = []
        for i, (a, b) in enumerate(zip(points, points[1:])):
            f = _direction(a, b)
            side = tuple(hand * x for x in _right(f))
            local = {_local(p, a, f, side) for p in group}
            distance = abs(b[0] - a[0]) + abs(b[1] - a[1])
            if (min(u for u, v in local) > radius
                    and max(u for u, v in local) < distance - radius
                    and min(v for u, v in local) <= 0 <= max(v for u, v in local)):
                matches.append((i, local))
        _require(len(matches) == 1, "Each blocker must cross exactly one leg")
        i, local = matches[0]
        assigned[i].append(local)
    path, arrivals, profiles = [points[0]], [0], []
    for i, (a, b) in enumerate(zip(points, points[1:])):
        _require(len(assigned[i]) <= 1, "At most one blocker on a leg")
        f = _direction(a, b)
        side = tuple(hand * x for x in _right(f))
        distance = abs(b[0] - a[0]) + abs(b[1] - a[1])
        obstacle = assigned[i][0] if assigned[i] else set()
        profile = _contour_profile(obstacle, distance, radius, construction)
        local_path = _trace_profile(profile)
        path.extend(_world(p, a, f, side) for p in local_path[1:])
        arrivals.append(len(path) - 1)
        profiles.append(profile)
    _require(len(path) == len(set(path)), "The ribbon itinerary may not cross itself")
    return path, arrivals, profiles


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape), "Grid dimensions must be 1..30")
    _require(np.issubdtype(grid.dtype, np.integer), "Grid cells must be integers")
    _require(np.all((0 <= grid) & (grid <= 9)), "Grid colors must be 0..9")
    return grid.astype(np.int8, copy=True)


def _decode(input_grid):
    inp = _grid(input_grid)
    colors, counts = np.unique(inp, return_counts=True)
    _require(len(colors) == 5 and int(max(counts)) > inp.size // 2,
             "Exactly five distinct geometric roles and a majority background are required")
    bg = int(colors[int(np.argmax(counts))])
    cells = {int(c): {tuple(map(int, p)) for p in np.argwhere(inp == c)}
             for c in colors if c != bg}
    candidates = []
    for ink, colored in cells.items():
        components = _components(colored)
        movers = [c for c in components if len(c) > 1]
        dots = [next(iter(c)) for c in components if len(c) == 1]
        if len(movers) != 1 or not 2 <= len(dots) <= 6:
            continue
        try:
            tip, initial, mask, radius = _mask_pose(movers[0])
            points = _ordered_references(tip, dots, initial)
        except ValueError:
            continue
        for support_color, support in cells.items():
            if support_color == ink or not 3 <= len(support) <= 7:
                continue
            endpoint = max(support, key=lambda p: _dot(p, initial))
            if support != {_add(endpoint, initial, -k) for k in range(len(support))}:
                continue
            for hand in (-1, 1):
                side = tuple(hand * x for x in _right(initial))
                anchor = _add(endpoint, side)
                gap, lateral = _local(tip, anchor, initial, side)
                if lateral != 0 or not radius + 2 <= gap <= 7:
                    continue
                for ribbon_color, ribbon in cells.items():
                    if ribbon_color in (ink, support_color) or anchor not in ribbon:
                        continue
                    slack = {_local(p, endpoint, initial, side) for p in ribbon}
                    if (len(_components(ribbon)) != 1 or
                            any(u > 0 or v < 1 for u, v in slack)):
                        continue
                    obstacle_color = next(c for c in cells if c not in (ink, support_color, ribbon_color))
                    obstacles = cells[obstacle_color]
                    try:
                        journey, arrivals, profiles = _journey(points, obstacles, radius, hand)
                        lead = [_add(anchor, initial, k) for k in range(gap + 1)]
                        path = lead + journey[1:]
                        _require(len(path) == len(set(path)), "Lead intersects the journey")
                        length = len(ribbon)
                        _require(gap + 2 <= length <= len(path), "Ribbon must move the head and fit the itinerary")
                        _require(not set(path) & (support | obstacles), "Ribbon hits fixed material")
                        _require(all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) > radius
                                     for p in journey for q in obstacles),
                                 "Another leg's obstacle halo interferes with the journey")
                        # Every prospective pose is in bounds, separated from fixed material.
                        # A radius-sized square around each tip is the movement safety halo.
                        for i, p in enumerate(journey):
                            f = initial if i == 0 else _direction(journey[i - 1], p)
                            posed = _place_mask(mask, p, f)
                            _require(not posed & (support | obstacles), "Mover hits fixed material")
                            _require(all(0 <= r < inp.shape[0] and 0 <= c < inp.shape[1]
                                         for r, c in posed), "Full itinerary mover pose is out of bounds")
                        _require(all(0 <= r < inp.shape[0] and 0 <= c < inp.shape[1]
                                     for r, c in path), "Full ribbon itinerary is out of bounds")
                    except ValueError:
                        continue
                    candidates.append(dict(inp=inp, bg=bg, ink=ink, support_color=support_color,
                                           ribbon_color=ribbon_color, obstacle_color=obstacle_color,
                                           support=support, ribbon=ribbon, obstacles=obstacles,
                                           tip=tip, initial=initial, mask=mask, radius=radius,
                                           endpoint=endpoint, anchor=anchor, hand=hand, gap=gap,
                                           points=points, journey=journey, arrivals=arrivals,
                                           profiles=profiles, path=path, length=length))
    _require(len(candidates) == 1, "Input must admit one geometric role assignment")
    return candidates[0]


def _render_answer(info, path=None, length=None, final_direction=None):
    path = info["path"] if path is None else path
    length = info["length"] if length is None else length
    used = path[:length]
    _require(len(used) >= 2, "A ribbon needs an endpoint")
    f = _direction(used[-2], used[-1]) if final_direction is None else final_direction
    out = np.full(info["inp"].shape, info["bg"], dtype=np.int8)
    for p in info["support"]:
        out[p] = info["support_color"]
    for p in info["obstacles"]:
        out[p] = info["obstacle_color"]
    for p in used:
        if 0 <= p[0] < out.shape[0] and 0 <= p[1] < out.shape[1]:
            out[p] = info["ribbon_color"]
    for p in _place_mask(info["mask"], used[-1], f):
        if 0 <= p[0] < out.shape[0] and 0 <= p[1] < out.shape[1]:
            out[p] = info["ink"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive roles, itinerary, budget, and exact terminal head from the input."""
    return _render_answer(_decode(input_grid))


def _arrow(span, extension, solid, tails):
    cells = {(-k, 0) for k in range(extension)}
    for depth in range(1, span + 1):
        vs = range(-depth, depth + 1) if solid else (-depth, depth)
        cells.update((-extension + 1 - depth, v) for v in vs)
    cells.update((-extension + 1 - span - k, v)
                 for k in range(1, tails + 1) for v in (-span, span))
    return tuple(sorted(cells))


_MASKS = tuple(dict.fromkeys(_arrow(span, extension, solid, tails)
                            for span in (1, 2) for extension in (1, 2)
                            for solid in (False, True) for tails in (0, 1)
                            if span + extension - 1 + tails <= 2))


def _radius(mask):
    return max(max(abs(u), abs(v)) for u, v in mask)


def _grow_references(rng, first_distance, count):
    points = [(0, 0), (0, first_distance)]
    for _ in range(count - 1):
        options = []
        last_direction = _direction(points[-2], points[-1])
        for d in _D4:
            if d == tuple(-x for x in last_direction):
                continue
            for distance in range(4, 12):
                q = _add(points[-1], d, distance)
                if not (-19 <= q[0] <= 19 and 1 <= q[1] <= 20):
                    continue
                trial = points + [q]
                if (max(r for r, c in trial) - min(r for r, c in trial) > 22
                        or max(c for r, c in trial) - min(c for r, c in trial) > 20):
                    continue
                if any(max(abs(q[0] - p[0]), abs(q[1] - p[1])) <= 2 for p in points):
                    continue
                try:
                    if _ordered_references((0, 0), trial[1:], (0, 1)) != trial:
                        continue
                except ValueError:
                    continue
                options.append((q, 1.5 if d != last_direction else 0.6))
        _require(bool(options), "Route growth exhausted")
        weights = np.array([w for q, w in options], dtype=float)
        choice = int(rng.choice(len(options), p=weights / weights.sum()))
        points.append(options[choice][0])
    return points


def _raw_obstacle(rng, width, sloped=False, extended=False):
    peak = int(rng.integers(1 if sloped else 0, 3))
    if sloped or (width >= 3 and rng.random() < .6):
        center = int(rng.integers(width))
        plateau = int(rng.integers(1, 3 if extended else 2))
        caps = [max(0, peak - max(center - i, i - center - plateau + 1, 0))
                for i in range(width)]
        if sloped and len(set(caps)) == 1:
            caps[0] = max(0, peak - 1)
    else:
        caps = [peak] * width
    bottoms = [int(rng.integers(0, 3)) for _ in range(width)]
    # Actual silhouettes vary on both sides; the outer side determines routing.
    return {(u, v) for u in range(width) for v in range(-bottoms[u], caps[u] + 1)}


def _slack_cells(rng, length, gap, diagonal):
    width = int(rng.integers(4, 8))
    depth = max(3, (length - 1 + width - 2) // (width - 1) + 1)
    anchor = (0, -gap)
    cells = {anchor}
    if diagonal:
        cells.add((1, -gap - 1))
        allowed = lambda p: 1 <= p[0] <= depth and -gap - width + 1 <= p[1] <= -gap - 1
    else:
        allowed = lambda p: 0 <= p[0] <= depth and -gap - width + 1 <= p[1] <= -gap
    style = int(rng.integers(4))
    dirs = _D4 if style == 0 and not diagonal else _D8
    while len(cells) < length:
        frontier = sorted({_add(p, d) for p in cells for d in dirs
                           if allowed(_add(p, d))} - cells)
        _require(bool(frontier), "Slack region is full")
        weights = []
        for p in frontier:
            neighbors = sum(_add(p, d) in cells for d in _D8)
            weights.append((neighbors + 1) ** 2 if style == 0 else
                           1 / (neighbors + 1) ** 2 if style == 1 else
                           3 if style == 2 and p[0] <= 2 else 1)
        w = np.asarray(weights, dtype=float)
        cells.add(frontier[int(rng.choice(len(frontier), p=w / w.sum()))])
    return cells


def _bounds(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _make_pair(rng, lesson, regime, test=False, excluded_mask=None, forced_mask=None):
    for attempt in range(350):
        try:
            radius = (_radius(forced_mask) if forced_mask is not None else
                      1 if lesson == "contour" else 2 if lesson == "reach" else int(rng.integers(1, 3)))
            pool = [m for m in _MASKS if _radius(m) == radius and m != excluded_mask]
            mask = forced_mask if forced_mask is not None else pool[int(rng.integers(len(pool)))]
            width = (int(rng.integers(6, 8)) if test and regime == 1 else
                     int(rng.integers(3, 6)) if lesson == "contour" else int(rng.integers(1, 6)))
            first_distance = width + 2 * radius + 4 + int(rng.integers(0, 3))
            count = int(rng.integers(4, 6)) if test and regime == 0 else int(rng.integers(2, 4))
            points = _grow_references(rng, first_distance, count)
            if lesson == "reach" and _direction(points[1], points[2]) == (0, 1):
                continue
            obstacles = set()
            for i, (a, b) in enumerate(zip(points, points[1:])):
                d = abs(b[0] - a[0]) + abs(b[1] - a[1])
                if i and (rng.random() > .32 or d < 2 * radius + 6):
                    continue
                w = width if i == 0 else int(rng.integers(1, min(4, d - 2 * radius - 3)))
                near = int(rng.integers(radius + 2, d - radius - w))
                raw = _raw_obstacle(rng, w, sloped=lesson == "contour" and i == 0,
                                    extended=test and regime == 1)
                f = _direction(a, b)
                obstacles.update(_world((u + near, v), a, f, _right(f)) for u, v in raw)
            journey, arrivals, profiles = _journey(points, obstacles, radius, 1, construction=True)
            if any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= radius
                   for p in journey for q in obstacles):
                continue
            gap = radius + 2 + int(rng.integers(0, 2))
            lead = [(0, c) for c in range(-gap, 1)]
            path = lead + journey[1:]
            if len(path) != len(set(path)):
                continue
            if lesson == "contour":
                stops = [i for i in range(2, arrivals[1])
                         if _direction(journey[i - 1], journey[i])[0] != 0]
                if not stops:
                    continue
                stop = int(rng.choice(stops))
                length = gap + 1 + stop
            elif lesson == "reach" or test:
                # Expose checkpoint traversal, then spend the remaining ribbon
                # on a later leg. Longer tests recombine that learned operation.
                lo, hi = arrivals[1] + 2, arrivals[-1]
                if lo >= hi:
                    continue
                stop = int(rng.integers(lo, hi))
                length = gap + 1 + stop
            else:
                length = len(path) if rng.random() < .20 else int(rng.integers(gap + 4, len(path)))
            if length > 55:
                continue
            slack = _slack_cells(rng, length, gap, diagonal=lesson == "reach" or rng.random() < .45)
            support_length = int(rng.integers(3, 7))
            support = {(-1, -gap - k) for k in range(support_length)}
            mover = _place_mask(mask, (0, 0), (0, 1))
            roles = [support, slack, obstacles, mover | set(points[1:])]
            if any(a & b for i, a in enumerate(roles) for b in roles[i + 1:]):
                continue
            if len(_components(roles[-1])) != count + 1:
                continue
            if any(set(path) & role for role in (support, obstacles)):
                continue
            envelope = set(path) | support | slack | obstacles
            for i, p in enumerate(journey):
                f = (0, 1) if i == 0 else _direction(journey[i - 1], p)
                posed = _place_mask(mask, p, f)
                if posed & (support | obstacles):
                    raise ValueError("Collision in a constructed pose")
                envelope.update(posed)
            r0, r1, c0, c1 = _bounds(envelope)
            h, w = r1 - r0 + 1, c1 - c0 + 1
            if max(h, w) > 30:
                continue
            padding = [0, 0, 0, 0]
            for dim, size in enumerate((h, w)):
                extra = int(rng.integers(0, min(5, 30 - size) + 1))
                before = int(rng.integers(extra + 1))
                padding[2 * dim:2 * dim + 2] = [before, extra - before]
            h += padding[0] + padding[1]
            w += padding[2] + padding[3]
            colors = tuple(map(int, rng.choice(10, 5, replace=False)))
            bg, support_color, ribbon_color, obstacle_color, ink = colors
            inp = np.full((h, w), bg, dtype=np.int8)
            out = inp.copy()
            dr, dc = padding[0] - r0, padding[2] - c0
            shift = lambda p: (p[0] + dr, p[1] + dc)
            for role, color in zip(roles, colors[1:]):
                for p in role:
                    inp[shift(p)] = color
            for p in support:
                out[shift(p)] = support_color
            for p in obstacles:
                out[shift(p)] = obstacle_color
            for p in path[:length]:
                out[shift(p)] = ribbon_color
            final_direction = _direction(path[length - 2], path[length - 1])
            for p in _place_mask(mask, path[length - 1], final_direction):
                out[shift(p)] = ink
            # Reflections reverse the input-visible winding hand. No hidden
            # clockwise convention or fixed final direction survives these.
            if rng.random() < .5:
                inp, out = np.flipud(inp), np.flipud(out)
            turns = int(rng.integers(4))
            inp, out = np.rot90(inp, turns).copy(), np.rot90(out, turns).copy()
            info = _decode(inp)
            if not np.array_equal(_render_answer(info), out):
                raise ValueError("Constructed and decoded paths disagree")
            if lesson == "contour" and _direction(info["path"][length - 2], info["path"][length - 1]) == info["initial"]:
                continue
            if lesson == "contour" and np.array_equal(_render_answer(info, _bbox_itinerary(info)), out):
                continue
            if lesson == "contour" and np.array_equal(_render_answer(info, _unbuffered_itinerary(info)), out):
                continue
            return {"input": inp, "output": out}
        except ValueError:
            continue
    raise RuntimeError("Bounded scene construction exhausted")


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 2, 10, "num_train"), _count(num_test, 1, 4, "num_test")
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    held_pool = [m for m in _MASKS if _radius(m) == 2]
    held_mask = held_pool[int(rng.integers(len(held_pool)))] if regime == 2 else None
    puzzle, seen = {"train": [], "test": []}, set()
    for split, number in (("train", nt), ("test", nq)):
        for index in range(number):
            lesson = ("contour" if index == 0 else "reach" if index == 1 else None) if split == "train" else None
            for _ in range(30):
                pair = _make_pair(rng, lesson, regime, test=split == "test",
                                  excluded_mask=held_mask if split == "train" else None,
                                  forced_mask=held_mask if split == "test" else None)
                key = _key(pair["input"])
                if key in seen:
                    continue
                info = _decode(pair["input"])
                if split == "train" and index == 1:
                    if info["length"] == _decode(puzzle["train"][0]["input"])["length"]:
                        continue
                seen.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("Distinct scene construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _straight_itinerary(info):
    path = [_add(info["anchor"], info["initial"], k) for k in range(info["gap"] + 1)]
    for a, b in zip(info["points"], info["points"][1:]):
        d = _direction(a, b)
        distance = abs(b[0] - a[0]) + abs(b[1] - a[1])
        path.extend(_add(a, d, k) for k in range(1, distance + 1))
    return path


def _bbox_itinerary(info):
    boxes = set()
    for component in _components(info["obstacles"]):
        r0, r1, c0, c1 = _bounds(component)
        boxes.update((r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))
    journey, _, _ = _journey(info["points"], boxes, info["radius"], info["hand"])
    return [_add(info["anchor"], info["initial"], k) for k in range(info["gap"] + 1)] + journey[1:]


def _unbuffered_itinerary(info):
    journey, _, _ = _journey(info["points"], info["obstacles"], 0, info["hand"])
    return [_add(info["anchor"], info["initial"], k) for k in range(info["gap"] + 1)] + journey[1:]


def validate_generated_puzzle(puzzle) -> None:
    """Input-derived semantic checks, including omission counterexamples."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _count(nt, 2, 10, "num_train")
    _count(nq, 1, 4, "num_test")
    _require(nt > nq, "Too many tests")
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Bad pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "Canvas must be preserved")
            key = _key(inp)
            _require(key not in seen, "Duplicate input")
            seen.add(key)
            info = _decode(inp)
            expected = _render_answer(info)
            _require(np.array_equal(expected, out), "Incorrect ribbon or terminal rigid mover")
            _require(not np.array_equal(inp, out), "Identity pair")
            _require(np.count_nonzero(out == info["ink"]) == len(info["mask"]), "Wrong mask area")
            _require(all(out[p] == info["support_color"] for p in info["support"]), "Support changed")
            _require(all(out[p] == info["obstacle_color"] for p in info["obstacles"]), "Obstacle changed")
            _require(out[info["anchor"]] == info["ribbon_color"], "Fixed attachment lost")
            built, _, _ = _journey(info["points"], info["obstacles"], info["radius"], info["hand"], True)
            _require(built == info["journey"], "Analytic and bitmap halos disagree")
            if split == "train":
                training.append((info, out))
    _require({a["radius"] for a, out in training} == {1, 2}, "Teach both halo radii")
    _require(len({a["mask"] for a, out in training}) >= 2, "Teach exact mask preservation")
    _require(len({a["length"] for a, out in training}) >= 2, "Teach a variable ribbon budget")
    _require(any(len(_components(a["ribbon"], diagonal=False)) > 1 for a, out in training), "Teach diagonal counting")
    _require(any(_direction(a["path"][a["length"] - 2], a["path"][a["length"] - 1]) != a["initial"]
                 for a, out in training), "Teach rotation")
    _require(any(a["length"] - 1 - a["gap"] > a["arrivals"][1] for a, out in training), "Teach a later checkpoint leg")
    _require(any(a["length"] < len(a["path"]) for a, out in training), "Teach budget exhaustion before the final checkpoint")
    _require(any(not np.array_equal(_render_answer(a, _straight_itinerary(a), min(a["length"], len(_straight_itinerary(a)))), out)
                 for a, out in training), "No obstacle-omission counterexample")
    _require(any(not np.array_equal(_render_answer(a, length=len(a["path"])), out)
                 for a, out in training), "No length-omission counterexample")
    _require(any(not np.array_equal(_render_answer(a, final_direction=a["initial"]), out)
                 for a, out in training), "No rotation-omission counterexample")
    _require(any(not np.array_equal(_render_answer(a, _bbox_itinerary(a)), out)
                 for a, out in training), "No actual-silhouette counterexample")
    _require(any(not np.array_equal(_render_answer(a, _unbuffered_itinerary(a)), out)
                 for a, out in training), "No mover-radius counterexample")


def measure_pair(pair):
    a = _decode(pair["input"])
    index = a["length"] - 1 - a["gap"]
    final = _direction(a["path"][a["length"] - 2], a["path"][a["length"] - 1])
    normalized_points = tuple(_local(p, a["tip"], a["initial"], tuple(a["hand"] * x for x in _right(a["initial"])))
                              for p in a["points"])
    normalized_obstacles = tuple(sorted(_local(p, a["tip"], a["initial"], tuple(a["hand"] * x for x in _right(a["initial"])))
                                        for p in a["obstacles"]))
    return dict(shape=tuple(a["inp"].shape), radius=a["radius"], mask=a["mask"], length=a["length"],
                checkpoints=len(a["points"]) - 1, blockers=len(_components(a["obstacles"])),
                initial=a["initial"], final=final, hand=a["hand"], gap=a["gap"],
                budget_exhausted=a["length"] < len(a["path"]),
                visited=sum(i <= index for i in a["arrivals"][1:]),
                diagonal_slack=len(_components(a["ribbon"], diagonal=False)) > 1,
                density=len(a["ribbon"]) / ((max(r for r, c in a["ribbon"]) - min(r for r, c in a["ribbon"]) + 1)
                                            * (max(c for r, c in a["ribbon"]) - min(c for r, c in a["ribbon"]) + 1)),
                palette=(a["bg"], a["support_color"], a["ribbon_color"], a["obstacle_color"], a["ink"]),
                structure=(a["mask"], normalized_points, normalized_obstacles, a["length"], a["gap"]))


def self_test():
    """Bounded 42-seed sweep plus meaningful local and corruption checks."""
    # A rising/falling profile is traced cell-by-cell, not as its bounding box.
    assert _trace_profile([0, 2, 3, 2, 0]) == [
        (0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (2, 3),
        (3, 3), (3, 2), (4, 2), (4, 1), (4, 0)]
    # Exercise the solver grammar beyond generation's mask bank, gap range,
    # and support lengths. The expected tip/pose is read from this input alone.
    new_mask = ((-2, -2), (-2, -1), (-2, 1), (-2, 2), (-1, 0), (0, 0))
    assert new_mask not in _MASKS
    extended = np.zeros((20, 30), dtype=np.int8)
    tip, initial = (10, 14), (0, 1)
    for p in _place_mask(new_mask, tip, initial) | {(10, 26), (2, 26)}:
        extended[p] = 4
    extended[9, 1:8] = 1
    extended[10:14, 4:8] = 2
    for u, cap in ((5, 1), (6, 2), (7, 1)):
        for v in range(-1, cap + 1):
            extended[_world((u, v), tip, initial, _right(initial))] = 3
    extended_info = _decode(extended)
    assert extended_info["mask"] == new_mask and extended_info["gap"] == 7
    assert len(extended_info["support"]) == 7
    assert np.count_nonzero(solve(extended) == 4) == len(new_mask)
    records, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p[k]) for s in ("train", "test") for p in puzzle[s] for k in ("input", "output")))
        records.extend((s, measure_pair(p)) for s in ("train", "test") for p in puzzle[s])
        for s in ("train", "test"):
            for pair in puzzle[s]:
                inp = pair["input"].copy()
                assert np.array_equal(solve(inp), pair["output"])
                assert np.array_equal(inp, pair["input"]), "solve modified its argument"
    assert len(signatures) == 42
    all_rows = [m for s, m in records]
    assert len({m["structure"] for m in all_rows}) >= 180
    assert len({m["shape"] for m in all_rows}) >= 35
    assert len({m["mask"] for m in all_rows}) >= 6
    assert len({m["length"] for m in all_rows}) >= 15
    assert {m["radius"] for m in all_rows} == {1, 2}
    assert {m["hand"] for m in all_rows} == {-1, 1}
    assert {m["initial"] for m in all_rows} == set(_D4)
    assert {m["final"] for m in all_rows} == set(_D4)
    assert {m["blockers"] for m in all_rows} >= {1, 2}
    assert len({m["checkpoints"] for m in all_rows}) >= 3
    assert any(m["visited"] >= 2 for m in all_rows)
    assert any(m["density"] < .5 for m in all_rows)
    assert any(not m["diagonal_slack"] for m in all_rows)
    assert all({m["palette"][i] for m in all_rows} == set(range(10)) for i in range(5))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + 10 * nt + nq, nt, nq)
            assert (len(p["train"]), len(p["test"])) == (nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.2), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")
    a, b = generate_puzzle(123, 2, 1), generate_puzzle(123, 2, 1)
    assert all(np.array_equal(p[k], q[k]) for s in a for p, q in zip(a[s], b[s]) for k in p)
    info = _decode(a["test"][0]["input"])
    fixed = next(iter(info["obstacles"]))
    a["test"][0]["output"][fixed] = info["bg"]
    try:
        validate_generated_puzzle(a)
    except ValueError:
        pass
    else:
        raise AssertionError("Changed fixed material accepted")
    try:
        _ordered_references((0, 0), {(0, 5), (5, 0)}, (0, 1))
    except ValueError:
        pass
    else:
        raise AssertionError("Competing checkpoint rays accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds and all legal explicit counts)")
