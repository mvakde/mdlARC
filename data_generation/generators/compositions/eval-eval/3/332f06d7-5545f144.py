"""Aperture routing: an exact rotating footprint selects a waypoint route.

Only NumPy and the standard library are used. Generation has no file I/O;
solve reads a single bitmap and does not call the sampler.
"""
from __future__ import annotations

import hashlib
import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(np.all((0 <= a) & (a <= 9)), "invalid grid colors")
    return a


def _rotate(p, turns):
    r, c = p
    for _ in range(turns % 4):
        r, c = c, -r
    return int(r), int(c)


def _oriented(mask, direction):
    return tuple(_rotate(p, direction - 1) for p in mask)


def _footprint(mask, tip, direction):
    return {(tip[0] + r, tip[1] + c) for r, c in _oriented(mask, direction)}


def _direction(a, b):
    r, c = b[0] - a[0], b[1] - a[1]
    _require((r == 0) != (c == 0), "leg is not nonzero and orthogonal")
    return (1 if c > 0 else 3) if r == 0 else (2 if r > 0 else 0)


def _line(a, b):
    dr, dc = _DIRS[_direction(a, b)]
    return [(a[0] + t * dr, a[1] + t * dc)
            for t in range(abs(b[0] - a[0]) + abs(b[1] - a[1]) + 1)]


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        group = {remaining.pop()}
        queue = list(group)
        for r, c in queue:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = r + dr, c + dc
                    if p in remaining:
                        remaining.remove(p)
                        group.add(p)
                        queue.append(p)
        result.append(group)
    return result


def _identify(cells):
    """Infer the sole projecting end of a bilaterally symmetric arrow."""
    matches = []
    for tip in sorted(cells):
        for facing in range(4):
            mask = {_rotate((r - tip[0], c - tip[1]), 1 - facing) for r, c in cells}
            if (max(c for r, c in mask) == 0
                    and {p for p in mask if p[1] == 0} == {(0, 0)}
                    and max(abs(r) for r, c in mask) > 0
                    and {(-r, c) for r, c in mask} == mask):
                matches.append((tuple(sorted(mask)), tip, facing))
    _require(len(matches) == 1, "arrow needs one geometrically unique tip")
    return matches[0]


def _decode(input_grid):
    g = _grid(input_grid)
    frame = np.concatenate((g[0], g[-1], g[:, 0], g[:, -1]))
    _require(len(np.unique(frame)) == 1, "wall frame is not uniform")
    wall = int(frame[0])
    colors, counts = np.unique(g[g != wall], return_counts=True)
    _require(len(colors) == 2 and counts[0] != counts[1], "need floor and ink")
    floor = int(colors[np.argmax(counts)])
    ink = int(colors[np.argmin(counts)])
    groups = _components(map(tuple, np.argwhere(g == ink)))
    movers = [v for v in groups if len(v) > 1]
    _require(len(movers) == 1, "one separated multi-cell arrow is required")
    mask, tip, facing = _identify(movers[0])
    waypoints = {next(iter(v)) for v in groups if len(v) == 1}
    _require(1 <= len(waypoints) <= 16, "need 1..16 separated waypoints")
    return dict(grid=g, wall=wall, floor=floor, ink=ink, mask=mask,
                tip=tip, facing=facing, waypoints=waypoints)


def _rays(tip, remaining):
    nearest = {}
    for q in sorted(remaining):
        if q[0] == tip[0] or q[1] == tip[1]:
            d = _direction(tip, q)
            distance = abs(q[0] - tip[0]) + abs(q[1] - tip[1])
            if d not in nearest or distance < nearest[d][0]:
                nearest[d] = distance, q
    return [(d, nearest[d][1]) for d in sorted(nearest)]


def _leg_fits(allowed, mask, a, b, orientation=None):
    """All discrete placements, including a tip-pivot before translation."""
    d = _direction(a, b)
    offsets = np.asarray(_oriented(mask, d if orientation is None else orientation), dtype=int)
    steps = np.asarray(_line(a, b), dtype=int)
    rr = steps[:, 0, None] + offsets[None, :, 0]
    cc = steps[:, 1, None] + offsets[None, :, 1]
    if np.any(rr < 0) or np.any(cc < 0) or np.any(rr >= allowed.shape[0]) or np.any(cc >= allowed.shape[1]):
        return False
    return bool(allowed[rr, cc].all())


def _placement_fits(allowed, mask, tip, direction):
    cells = _footprint(mask, tip, direction)
    return all(0 <= r < allowed.shape[0] and 0 <= c < allowed.shape[1] and allowed[r, c]
               for r, c in cells)


def _run(info, *, mask=None, point_only=False, forward_shortcut=False, frozen=False):
    g = info["grid"]
    allowed = g != info["wall"]
    mask = info["mask"] if mask is None else mask
    tip, facing = info["tip"], info["facing"]
    remaining = set(info["waypoints"])
    points, headings, decisions = [tip], [facing], []
    while remaining:
        rays = _rays(tip, remaining)
        statuses = []
        for d, q in rays:
            fit = _leg_fits(allowed, ((0, 0),) if point_only else mask, tip, q,
                            info["facing"] if frozen else None)
            statuses.append((d, q, fit))
        viable = [(d, q) for d, q, fit in statuses if fit]
        decisions.append(statuses)
        if not viable:
            break
        if forward_shortcut:
            # A deliberately wrong alternative, used only in diagnostics.
            viable.sort(key=lambda v: (v[0] != facing,
                                       abs(v[1][0]-tip[0])+abs(v[1][1]-tip[1]), v[0]))
        else:
            _require(len(viable) == 1, "two traversable waypoint rays")
        direction, tip = viable[0]
        facing = info["facing"] if frozen else direction
        remaining.remove(tip)
        points.append(tip)
        headings.append(facing)
    out = g.copy()
    out[out == info["ink"]] = info["floor"]
    for q in remaining:
        out[q] = info["ink"]
    for p in _footprint(mask, tip, facing):
        _require(0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1], "terminal mask out of bounds")
        out[p] = info["ink"]
    return out, dict(points=points, headings=headings, remaining=remaining, decisions=decisions)


def _solve_info(input_grid):
    info = _decode(input_grid)
    out, route = _run(info)
    return out, dict(info, **route)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the final arrow, unvisited waypoints, and unchanged architecture."""
    return _solve_info(input_grid)[0]


def _arrow(span, extension, solid, rear):
    # Head + projecting nose + optional paired rear rails, never a shape ID.
    cells = {(0, -i) for i in range(extension)}
    for depth in range(1, span + 1):
        rows = range(-depth, depth + 1) if solid else (-depth, depth)
        cells.update((r, 1 - extension - depth) for r in rows)
    for j in range(1, rear + 1):
        cells.update((r, 1 - extension - span - j) for r in (-span, span))
    return tuple(sorted(cells))


_MASKS = tuple(dict.fromkeys(_arrow(s, e, solid, rear)
                            for s in (1, 2) for e in (1, 2)
                            for solid in (False, True) for rear in (0, 1)))


def _sweep(mask, a, b):
    d = _direction(a, b)
    return set().union(*(_footprint(mask, p, d) for p in _line(a, b)))


def _inside(cells, side=30):
    return all(1 <= r < side - 1 and 1 <= c < side - 1 for r, c in cells)


def _near(a, b, radius=1):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= radius


def _sample_route(rng, mask, steps, need_bypass):
    span = max(abs(r) for r, c in mask)
    margin = max(span, -min(c for r, c in mask)) + 1
    for _ in range(64):
        points = [(int(rng.integers(margin, 30-margin)),
                   int(rng.integers(margin, 30-margin)))]
        lines, ds = [], []
        for _ in range(steps):
            choices = []
            p = points[-1]
            for d, (dr, dc) in enumerate(_DIRS):
                if ds and d == (ds[-1] + 2) % 4:
                    continue
                for length in range(max(4, span + 3), 13):
                    q = p[0] + dr * length, p[1] + dc * length
                    if not _inside(_sweep(mask, p, q)):
                        continue
                    if any(_near(q, old, span + 1) for old in points[:-1]):
                        continue
                    new_line = _line(p, q)[1:]
                    old_line = [v for leg in lines[:-1] for v in leg]
                    if any(_near(x, y, 2 * span + 1) for x in new_line for y in old_line):
                        continue
                    choices.append((d, q))
            if not choices:
                break
            # A sampled walk, not a bank of named route templates.
            weights = np.array([0.65 if ds and d == ds[-1] else 1.0 for d, q in choices])
            d, q = choices[int(rng.choice(len(choices), p=weights/weights.sum()))]
            lines.append(_line(p, q))
            ds.append(d)
            points.append(q)
        if len(ds) != steps:
            continue
        if need_bypass and not any(a != b for a, b in zip(ds, ds[1:])):
            continue
        # The mover and all future stations remain visually separable.
        if any(_near(q, cell) for i, p in enumerate(points)
               for q in points[i+1:]
               for cell in _footprint(mask, p, ds[max(0, i-1)])):
            continue
        return points, ds
    return None


def _paint(floor, cells):
    for p in cells:
        floor[p] = True


def _main_floor(rng, mask, points, *, notched):
    floor = np.zeros((30, 30), dtype=bool)
    for j, (a, b) in enumerate(zip(points, points[1:])):
        cells = _sweep(mask, a, b)
        _paint(floor, cells)
        # Optional asymmetric expansions vary widths independently of the mask.
        if not (notched and j == len(points)-2) and rng.random() < 0.70:
            d = _direction(a, b)
            vr, vc = _DIRS[(d+1) % 4]
            for sign in rng.choice([-1, 1], int(rng.integers(1, 3)), replace=False):
                extra = {(r+int(sign)*vr, c+int(sign)*vc) for r, c in cells}
                if _inside(extra):
                    _paint(floor, extra)
        if j and rng.random() < 0.35:
            radius = max(abs(r) for r, c in mask) + int(rng.integers(0, 2))
            room = {(a[0]+r, a[1]+c) for r in range(-radius, radius+1)
                    for c in range(-radius, radius+1)}
            if _inside(room):
                _paint(floor, room)
    return floor


def _render(floor, mask, points, directions, decoys, palette):
    wall, ground, ink = palette
    g = np.full(floor.shape, wall, dtype=np.int8)
    g[floor] = ground
    for p in points[1:] + list(decoys):
        _require(floor[p], "waypoint not on floor")
        g[p] = ink
    for p in _footprint(mask, points[0], directions[0]):
        _require(floor[p], "initial arrow not on floor")
        g[p] = ink
    return g


def _add_decoy(rng, floor, mask, points, ds, decoys, palette, forced=None):
    depth = -min(c for r, c in mask)
    span = max(abs(r) for r, c in mask)
    options = [(i, d) for i in range(len(points)) for d in range(4)]
    if forced is not None:
        options = [forced]
    else:
        rng.shuffle(options)
    for index, d in options:
        p = points[index]
        dr, dc = _DIRS[d]
        distances = list(range(depth+5, 14))
        rng.shuffle(distances)
        for length in distances:
            q = p[0]+dr*length, p[1]+dc*length
            arrival = _footprint(mask, q, d)
            if not _inside(arrival | set(_line(p, q))):
                continue
            if any(_near(q, old, 2) for old in points + list(decoys)):
                continue
            if any(_near(q, cell) for i, a in enumerate(points)
                   for cell in _footprint(mask, a, ds[max(0, i-1)])):
                continue
            candidate = floor.copy()
            approach = int(rng.integers(1, 4))
            lead = p[0]+dr*approach, p[1]+dc*approach
            added = _sweep(mask, p, lead) | arrival | set(_line(p, q))
            if not _inside(added):
                continue
            _paint(candidate, added)
            # Some rejected spurs have a wider, but still insufficient throat.
            if span == 2 and rng.random() < 0.5:
                vr, vc = _DIRS[(d+1) % 4]
                extra = {(r+vr, c+vc) for r, c in _line(p, q)}
                if _inside(extra):
                    _paint(candidate, extra)
            if _leg_fits(candidate, mask, p, q):
                continue
            inp = _render(candidate, mask, points, ds, decoys + [q], palette)
            try:
                _, info = _solve_info(inp)
            except AssertionError:
                continue
            if info["points"] != points or info["remaining"] != set(decoys + [q]):
                continue
            if not any(old_d == d and old_q == q and not ok
                       for old_d, old_q, ok in info["decisions"][index]):
                continue
            # Every rejected ray is a real tip corridor, not a marker behind a wall.
            if not _leg_fits(candidate, ((0, 0),), p, q):
                continue
            return candidate, decoys + [q]
    return None


def _bbox_mask(mask):
    return tuple((r, c) for r in range(min(r for r, c in mask), max(r for r, c in mask)+1)
                 for c in range(min(c for r, c in mask), max(c for r, c in mask)+1))


def _features(input_grid):
    out, info = _solve_info(input_grid)
    allowed = info["grid"] != info["wall"]
    branch = bypass = endpoint_trap = terminal_stop = 0
    for p, facing, choices in zip(info["points"], info["headings"], info["decisions"]):
        feasible = [(d, q) for d, q, ok in choices if ok]
        traps = []
        for d, q, ok in choices:
            if (not ok and _leg_fits(allowed, ((0, 0),), p, q)
                    and _placement_fits(allowed, info["mask"], p, d)
                    and _placement_fits(allowed, info["mask"], q, d)):
                endpoint_trap += 1
                traps.append(d)
        if feasible and traps:
            branch += 1
        if feasible and feasible[0][0] != facing and facing in traps:
            bypass += 1
        if not feasible and traps:
            for d in traps:
                dr, dc = _DIRS[d]
                if _leg_fits(allowed, info["mask"], p, (p[0]+dr, p[1]+dc)):
                    terminal_stop += 1
    bbox_legs = sum(not _leg_fits(allowed, _bbox_mask(info["mask"]), a, b)
                    for a, b in zip(info["points"], info["points"][1:]))
    return out, info, dict(branches=branch, bypasses=bypass, endpoint_traps=endpoint_trap,
                          bbox_legs=bbox_legs, terminal_stops=terminal_stop,
                          visits=len(info["points"])-1,
                          turns=sum(a != b for a, b in zip(info["headings"], info["headings"][1:])))


def _make_pair(rng, mask, steps, palette, *, bypass, notched, terminal, seen):
    for _ in range(120):
        sampled = _sample_route(rng, mask, steps, bypass)
        if sampled is None:
            continue
        points, ds = sampled
        floor = _main_floor(rng, mask, points, notched=notched)
        decoys = []
        forced = None
        if bypass:
            candidates = [(i, ds[i-1]) for i in range(1, len(ds)) if ds[i] != ds[i-1]]
            rng.shuffle(candidates)
            result = None
            for forced in candidates:
                result = _add_decoy(rng, floor, mask, points, ds, decoys, palette, forced)
                if result is not None:
                    break
        else:
            result = _add_decoy(rng, floor, mask, points, ds, decoys, palette)
        if result is None:
            continue
        floor, decoys = result
        if terminal:
            choices = [(len(points)-1, d) for d in (ds[-1], (ds[-1]+1) % 4, (ds[-1]+3) % 4)]
            result = None
            for forced in choices:
                result = _add_decoy(rng, floor, mask, points, ds, decoys, palette, forced)
                if result is not None:
                    break
            if result is None:
                continue
            floor, decoys = result
        for _ in range(int(rng.integers(0, 4-len(decoys)))):
            result = _add_decoy(rng, floor, mask, points, ds, decoys, palette)
            if result is not None:
                floor, decoys = result
        inp = _render(floor, mask, points, ds, decoys, palette)
        try:
            _, info, metrics = _features(inp)
        except AssertionError:
            continue
        if (not metrics["branches"] or not metrics["endpoint_traps"]
                or (bypass and not metrics["bypasses"])
                or (notched and not metrics["bbox_legs"])
                or (terminal and not metrics["terminal_stops"])):
            continue
        # Crop unused wall and then pad asymmetrically. The full wall frame remains.
        occupied = np.argwhere(floor)
        lo, hi = occupied.min(axis=0), occupied.max(axis=0)+1
        pads = []
        for low, high in zip(lo, hi):
            spare = 30 - int(high-low)
            total = int(rng.integers(2, min(7, spare)+1))
            before = int(rng.integers(1, total))
            pads.append((before, total-before))
        inp = np.pad(inp[lo[0]:hi[0], lo[1]:hi[1]], pads, constant_values=palette[0])
        turns = int(rng.integers(4))
        inp = np.rot90(inp, turns)
        reflect = bool(rng.integers(2))
        if reflect:
            inp = np.fliplr(inp)
        inp = np.array(inp, dtype=np.int8, copy=True)
        signature = (inp.shape, inp.tobytes())
        if signature in seen:
            continue
        # Construction label uses the sampled path, not the input decoder.
        raw_output = _render(floor, mask, [points[-1]], [ds[-1]], decoys, palette)
        output = np.pad(raw_output[lo[0]:hi[0], lo[1]:hi[1]], pads, constant_values=palette[0])
        output = np.rot90(output, turns)
        if reflect:
            output = np.fliplr(output)
        predicted = solve(inp)
        _require(np.array_equal(output, predicted), "constructed route disagrees with bitmap")
        seen.add(signature)
        return {"input": inp, "output": np.array(output, dtype=np.int8, copy=True)}
    raise RuntimeError("bounded aperture-route construction exhausted")


def _counts(num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"), (num_test, 1, 4, "num_test")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(name + " has an illegal count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 3, replace=False)))
    variable_palette = rng.random() < 0.65
    # Tests extend the station chain; new masks are also possible, always visible.
    first = int(rng.integers(len(_MASKS)))
    second = int(rng.choice([j for j in range(len(_MASKS)) if j != first]))
    specs = [(first, 2, True, False, True), (second, 3, False, True, False)]
    for _ in range(num_train-2):
        specs.append((int(rng.integers(len(_MASKS))), int(rng.integers(2, 4)), False, bool(rng.integers(2)), False))
    rng.shuffle(specs)
    test_specs = [(int(rng.integers(len(_MASKS))), int(rng.integers(4, 6)), False, bool(rng.integers(2)), False)
                  for _ in range(num_test)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, rows in (("train", specs), ("test", test_specs)):
        for index, (shape, steps, bypass, notched, terminal) in enumerate(rows):
            if variable_palette:
                palette = tuple(map(int, rng.choice(10, 3, replace=False)))
            puzzle[split].append(_make_pair(rng, _MASKS[shape], steps, palette,
                                            bypass=bypass, notched=notched, terminal=terminal, seen=seen))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, rows = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = inp.shape, inp.tobytes()
            _require(key not in seen, "duplicate input")
            seen.add(key)
            expected, info, metrics = _features(inp)
            _require(np.array_equal(out, expected), "incorrect routing, mask, or scenery")
            _require(inp.shape == out.shape and not np.array_equal(inp, out), "unchanged pair")
            _require(np.array_equal(inp == info["wall"], out == info["wall"]), "wall changed")
            _require(metrics["branches"] >= 1 and metrics["endpoint_traps"] >= 1, "no genuine collision fork")
            _require(info["remaining"], "rejected waypoints missing")
            # At every reached station all future waypoints are visibly separate.
            remaining = set(info["waypoints"])
            for j, (p, d) in enumerate(zip(info["points"], info["headings"])):
                if j:
                    remaining.remove(p)
                _require(all(not _near(q, cell) for q in remaining
                             for cell in _footprint(info["mask"], p, d)), "waypoint merges with arrow")
            rows[split].append((info, metrics))
    train = rows["train"]
    _require(len({v[0]["mask"] for v in train}) >= 2, "need two demonstrated masks")
    _require(any(v[1]["bypasses"] for v in train), "need a blocked-forward turn")
    _require(any(v[1]["bbox_legs"] for v in train), "need exact-mask versus box contrast")
    _require(any(v[1]["terminal_stops"] for v in train), "need station halt before partially traversable spur")
    _require(all(v[1]["visits"] > max(t[1]["visits"] for t in train) for v in rows["test"]),
             "test must extend the demonstrated chain")


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _route_signature(points):
    origin = points[0]
    return min(tuple(_rotate((r-origin[0], sign*(c-origin[1])), k) for r, c in points)
               for sign in (-1, 1) for k in range(4))


def _controlled_checks():
    # Same ink bitmap and same two branches; swap which throat admits wings.
    # These coordinates and expected destinations are hand specified.
    mask = ((-1, -1), (0, -1), (0, 0), (1, -1))
    p0, fork, north, east = (14, 4), (14, 12), (6, 12), (14, 22)
    fixtures = []
    for destination, rejected, facing in ((north, east, 0), (east, north, 1)):
        floor = np.zeros((30, 30), dtype=bool)
        # Shared arrival from the west.
        floor[13:16, 3:12] = True
        floor[14, 3:13] = True
        if facing == 0:
            floor[6:16, 11:14] = True
            floor[14, 12:23] = True
            floor[13:16, 11:14] = True
            floor[13:16, 21] = True
        else:
            floor[13:16, 11:22] = True
            floor[14, 11:23] = True
            floor[6:15, 12] = True
            floor[13, 11:14] = True
            floor[7, 11:14] = True
        inp = np.full((30, 30), 9, dtype=np.int8)
        inp[floor] = 0
        for q in (fork, north, east):
            inp[q] = 2
        inp[13:16, 3] = 2
        inp[14, 4] = 2
        expected = inp.copy()
        expected[expected == 2] = 0
        expected[rejected] = 2
        for q in _footprint(mask, destination, facing):
            expected[q] = 2
        actual, info = _solve_info(inp)
        _require(info["points"] == [p0, fork, destination], "controlled gate choice failed")
        _require(np.array_equal(actual, expected), "controlled output failed")
        _require(not _leg_fits(floor, mask, fork, rejected), "rejected throat is passable")
        _require(_leg_fits(floor, ((0, 0),), fork, rejected), "tip cannot enter rejected throat")
        _require(_placement_fits(floor, mask, rejected, _direction(fork, rejected)), "rejected endpoint does not fit")
        fixtures.append((inp, actual))
    _require(np.array_equal(fixtures[0][0] == 2, fixtures[1][0] == 2), "ink contrast changed")
    _require(not np.array_equal(fixtures[0][1], fixtures[1][1]), "walls do not change answer")
    # With the arrow held east-facing the north branch is physically blocked:
    # correct quarter-turns, not merely a changing tip coordinate, are required.
    inp, actual = fixtures[0]
    wrong = inp.copy()
    wrong[wrong == 2] = 0
    wrong[east] = 2
    for q in _footprint(mask, north, 1):
        wrong[q] = 2
    _require(not np.array_equal(wrong, actual), "rotation ablation did not change answer")


def self_test():
    """Bounded, streaming, no stored puzzles or execution traces."""
    keys, shapes, masks, routes, role_colors = set(), set(), set(), set(), [set(), set(), set()]
    visits, decoys, turns = set(), set(), set()
    _controlled_checks()
    for mask in _MASKS:
        _require(_identify(_footprint(mask, (10, 10), 1))[0] == mask, "ambiguous sampled arrow")
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        keys.add(_fingerprint(puzzle))
        point_diff = frozen_diff = False
        for split in ("train", "test"):
            for pair in puzzle[split]:
                _, info, features = _features(pair["input"])
                shapes.add(pair["input"].shape)
                masks.add(info["mask"])
                routes.add(_route_signature(info["points"]))
                for bucket, role in zip(role_colors, ("wall", "floor", "ink")):
                    bucket.add(info[role])
                visits.add(features["visits"])
                decoys.add(len(info["remaining"]))
                turns.add(features["turns"])
                if split == "train":
                    try:
                        wrong, _ = _run(info, point_only=True, forward_shortcut=True)
                        point_diff |= not np.array_equal(wrong, pair["output"])
                    except AssertionError:
                        point_diff = True
                    try:
                        wrong, _ = _run(info, frozen=True)
                        frozen_diff |= not np.array_equal(wrong, pair["output"])
                    except AssertionError:
                        frozen_diff = True
        _require(point_diff and frozen_diff, "training fails a parent-concept ablation")
    _require(len(keys) == 42 and len(shapes) >= 30 and len(routes) >= 100, "insufficient structural diversity")
    _require(len(masks) >= 12 and all(v == set(range(10)) for v in role_colors), "mask/palette collapse")
    _require(len(visits) >= 4 and len(decoys) >= 2 and len(turns) >= 3, "route/count collapse")
    _require(_fingerprint(generate_puzzle(531, 2, 1)) == _fingerprint(generate_puzzle(531, 2, 1)), "nondeterminism")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    bad = generate_puzzle(0, 2, 1)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
