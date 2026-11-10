"""A tip-led courier selectively repairs a two-fold mirror field at its stops.

Only standard-library modules and NumPy are used. No files, seeds, demonstration
labels, or construction metadata enter solve(). Fixed roles: 0 = gap, 9 = route
overlay, 1..8 = field payload. See task_spec.md for the intended family.
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np

GAP, INK = 0, 9
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _rotate(point, turns):
    r, c = point
    for _ in range(turns % 4):
        r, c = c, -r
    return int(r), int(c)


def _footprint(mask, tip, facing):
    return {(tip[0] + r, tip[1] + c)
            for r, c in (_rotate(p, facing - 1) for p in mask)}


def _bounded(cells, shape):
    return all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in cells)


def _direction(a, b):
    r, c = b[0] - a[0], b[1] - a[1]
    if (r == 0) == (c == 0):
        raise ValueError("Consecutive tips must be distinct and axis-aligned")
    return (1 if c > 0 else 3) if r == 0 else (2 if r > 0 else 0)


def _components(cells):
    pending, components = set(cells), []
    while pending:
        first = min(pending)
        pending.remove(first)
        stack, component = [first], {first}
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in pending:
                        pending.remove(q)
                        stack.append(q)
                        component.add(q)
        components.append(component)
    return components


def _pointed_mask(cells):
    """Infer the actual mask from its unique projecting tip and lateral symmetry."""
    candidates = []
    for tip in sorted(cells):
        for facing in range(4):
            local = {_rotate((r - tip[0], c - tip[1]), 1 - facing)
                     for r, c in cells}
            if (max(c for r, c in local) == 0
                    and {p for p in local if p[1] == 0} == {(0, 0)}
                    and max(abs(r) for r, c in local) > 0
                    and {(-r, c) for r, c in local} == local):
                candidates.append((tuple(sorted(local)), tip, facing))
    if len(candidates) != 1:
        raise ValueError("Courier must have one geometrically identifiable tip")
    return candidates[0]


def _read_route(grid):
    components = _components({tuple(map(int, p)) for p in np.argwhere(grid == INK)})
    movers = [c for c in components if len(c) > 1]
    if len(movers) != 1:
        raise ValueError("Need one separated courier and singleton waypoints")
    mask, tip, initial = _pointed_mask(movers[0])
    remaining = {next(iter(c)) for c in components if len(c) == 1}
    if len(remaining) < 2:
        raise ValueError("Need at least two waypoints")
    points, directions = [tip], []
    while remaining:
        rays = {}
        for q in sorted(remaining):
            if tip[0] == q[0] or tip[1] == q[1]:
                rays.setdefault(_direction(tip, q), []).append(q)
        if len(rays) != 1:
            raise ValueError("There must be exactly one occupied ray at every stop")
        facing, choices = next(iter(rays.items()))
        q = min(choices, key=lambda p: abs(p[0] - tip[0]) + abs(p[1] - tip[1]))
        points.append(q)
        directions.append(facing)
        remaining.remove(q)
        tip = q
    if directions[0] != initial:
        raise ValueError("The first leg must follow the courier's tip")
    poses = [_footprint(mask, points[0], initial)]
    poses.extend(_footprint(mask, p, d) for p, d in zip(points[1:], directions))
    for i, cells in enumerate(poses):
        if not _bounded(cells, grid.shape):
            raise ValueError("An arrival footprint leaves the canvas")
        if any(max(abs(r - q[0]), abs(c - q[1])) <= 1
               for q in points[i + 1:] for r, c in cells):
            raise ValueError("Future waypoints must remain separated from the courier")
    coverage = set().union(*poses[1:])
    return dict(mask=mask, points=points, directions=directions,
                initial=initial, poses=poses, coverage=coverage)


@lru_cache(maxsize=256)
def _orbits(shape, axes):
    h, w = shape
    a, b = axes
    seen, orbits = set(), []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            rows = {r, a - r} if 0 <= a - r < h else {r}
            cols = {c, b - c} if 0 <= b - c < w else {c}
            orbit = tuple(sorted((y, x) for y in rows for x in cols))
            seen.update(orbit)
            orbits.append(orbit)
    return tuple(orbits)


def _axis_candidates(grid, dimension):
    a = grid if dimension == 0 else grid.T
    result = []
    for fold in range(1, 2 * a.shape[0] - 2):
        left = [r for r in range(a.shape[0]) if r < fold - r < a.shape[0]]
        right = [fold - r for r in left]
        x, y = a[left], a[right]
        visible = (x != GAP) & (x != INK) & (y != GAP) & (y != INK)
        if visible.any() and np.all(x[visible] == y[visible]):
            result.append(fold)
    return result


def _recover_field(grid):
    candidates = [_axis_candidates(grid, d) for d in (0, 1)]
    if any(len(v) != 1 for v in candidates):
        raise ValueError("Surviving payload must identify one fold in each dimension")
    axes = tuple(v[0] for v in candidates)
    field = grid.copy()
    for orbit in _orbits(grid.shape, axes):
        values = {int(grid[p]) for p in orbit if grid[p] not in (GAP, INK)}
        if len(values) != 1:
            raise ValueError("Every orbit needs unanimous surviving payload")
        value = next(iter(values))
        for p in orbit:
            field[p] = value
    return field, axes


def _analyze(grid):
    route = _read_route(grid)
    field, axes = _recover_field(grid)
    route.update(field=field, axes=axes)
    return route


def _as_grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30 and colors 0..9")
    return grid.astype(np.int8, copy=True)


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _as_grid(input_grid)
    info = _analyze(grid)
    result = grid.copy()
    result[grid == INK] = info["field"][grid == INK]
    for p in info["coverage"]:
        if grid[p] == GAP:
            result[p] = info["field"][p]
    return result


def _arrow(span, extension, solid, tail):
    cells = {(0, -i) for i in range(extension)}
    for depth in range(1, span + 1):
        rows = range(-depth, depth + 1) if solid else (-depth, depth)
        cells.update((r, 1 - extension - depth) for r in rows)
    for t in range(1, tail + 1):
        cells.update((r, 1 - extension - span - t) for r in (-span, span))
    return tuple(sorted(cells))


MASKS = tuple(dict.fromkeys(_arrow(s, e, full, tail)
                           for s in (1, 2) for e in (1, 2)
                           for full in (False, True) for tail in (0, 1)))


def _grow_route(rng, shape, mask, steps, mode):
    """Sample a geometric walk; prohibit a future point making any earlier junction."""
    for _ in range(70):
        points = [(int(rng.integers(shape[0])), int(rng.integers(shape[1])))]
        directions, poses = [], []
        while len(directions) < steps:
            options = []
            origin = points[-1]
            for facing, (dr, dc) in enumerate(DIRS):
                if mode == "straight" and directions and facing != directions[0]:
                    continue
                initial = _footprint(mask, points[0], facing) if not poses else poses[0]
                if not _bounded(initial, shape):
                    continue
                previous_poses = poses if poses else [initial]
                for distance in range(2, max(shape)):
                    q = (origin[0] + dr * distance, origin[1] + dc * distance)
                    arrival = _footprint(mask, q, facing)
                    if not _bounded(arrival, shape) or q in points:
                        continue
                    if any(max(abs(q[0] - r), abs(q[1] - c)) <= 1
                           for cells in previous_poses for r, c in cells):
                        continue
                    valid = True
                    for past, nxt in zip(points[:-1], points[1:]):
                        if q[0] == past[0] or q[1] == past[1]:
                            if (_direction(past, q) != _direction(past, nxt)
                                    or abs(q[0] - past[0]) + abs(q[1] - past[1])
                                    <= abs(nxt[0] - past[0]) + abs(nxt[1] - past[1])):
                                valid = False
                                break
                    if valid:
                        options.append((q, facing, initial, arrival))
            if not options:
                break
            # Directions are sampled before distances so long rays do not dominate.
            ds = sorted({o[1] for o in options})
            weights = np.array([1.2 if directions and d == directions[-1] else 1.0 for d in ds])
            d = int(rng.choice(ds, p=weights / weights.sum()))
            options = [o for o in options if o[1] == d]
            q, facing, initial, arrival = options[int(rng.integers(len(options)))]
            if not poses:
                poses.append(initial)
            points.append(q)
            directions.append(facing)
            poses.append(arrival)
        if len(directions) == steps and (mode != "turn" or directions[0] != directions[-1]):
            return dict(mask=mask, points=points, directions=directions,
                        initial=directions[0], poses=poses,
                        coverage=set().union(*poses[1:]))
    return None


def _swept_cells(route):
    cells = set()
    for a, b, d in zip(route["points"], route["points"][1:], route["directions"]):
        dr, dc = DIRS[d]
        distance = abs(a[0] - b[0]) + abs(a[1] - b[1])
        for k in range(1, distance):
            cells.update(_footprint(route["mask"], (a[0] + k * dr, a[1] + k * dc), d))
    return cells


def _field(rng, shape, axes, colors, style):
    """Reflect a varied fragment; intended speckles remain payload."""
    result = np.empty(shape, dtype=np.int8)
    bh, bw = int(rng.integers(1, 5)), int(rng.integers(1, 5))
    phase, k = int(rng.integers(len(colors))), len(colors)
    coarse = rng.integers(k, size=shape)
    center = (int(rng.integers(shape[0])), int(rng.integers(shape[1])))
    for orbit in _orbits(shape, axes):
        r, c = orbit[0]
        if style == 0:
            value = int(rng.integers(k))
        elif style == 1:
            value = int(coarse[r // bh, c // bw])
        elif style == 2:
            value = (r // bh + c // bw + phase) % k
        elif style == 3:
            value = ((r - c) // bw + phase) % k
        else:
            value = (max(abs(r - center[0]), abs(c - center[1])) // bh + phase) % k
        if style and rng.random() < 0.18:
            value = int(rng.integers(k))
        for p in orbit:
            result[p] = colors[value]
    return result


def _raw_gaps(rng, shape):
    gaps = set()
    family = int(rng.integers(5))
    if family in (0, 4):
        rate = float(rng.uniform(0.015, 0.065))
        gaps.update(tuple(map(int, p)) for p in np.argwhere(rng.random(shape) < rate))
    for _ in range(int(rng.integers(1, 5)) if family else 0):
        h, w = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        if family == 2:
            h, w = 1, int(rng.integers(3, 8))
        if family == 3:
            h, w = int(rng.integers(3, 8)), 1
        r, c = int(rng.integers(shape[0] - h + 1)), int(rng.integers(shape[1] - w + 1))
        for y in range(h):
            for x in range(w):
                if family != 4 or x <= (y + 1) * w // h:
                    gaps.add((r + y, c + x))
    return gaps


def _make_pair(rng, steps, mode, mask, hard, require_rotation):
    for attempt in range(100):
        low = 20 if steps >= 5 else 14
        shape = tuple(int(v) for v in rng.integers(low, 31, size=2))
        route = _grow_route(rng, shape, mask, steps, mode)
        if route is None:
            continue
        coverage = route["coverage"]
        ink = route["poses"][0] | set(route["points"][1:])
        a = shape[0] - 1 + int(rng.integers(-4, 5))
        b = shape[1] - 1 + int(rng.integers(-4, 5))
        axes = (a, b)
        orbits = _orbits(shape, axes)
        lookup = {p: o for o in orbits for p in o}
        if any(set(o) <= ink for o in orbits):
            continue
        eligible = coverage - ink
        early = eligible - route["poses"][-1]
        sweep_only = {p for p in _swept_cells(route) - coverage - ink if _bounded([p], shape)}
        fixed = set().union(*(_footprint(mask, p, route["initial"]) for p in route["points"][1:]))
        rotation_difference = (coverage ^ fixed) - ink
        rotation_difference = {p for p in rotation_difference if _bounded([p], shape)}
        if not early or not sweep_only or (require_rotation and not rotation_difference):
            continue
        # Mandatory gaps are selected before optional damage. Every orbit must
        # retain a payload witness; protected gaps cannot be silently revealed.
        protected, protected_visible = set(), set()
        if hard:
            options = [p for p in sorted(eligible) if len(lookup[p]) == 4
                       and not set(lookup[p]) & ink]
            if not options:
                continue
            p = options[int(rng.integers(len(options)))]
            protected.update((p, (a - p[0], p[1]), (p[0], b - p[1])))
            protected_visible.add((a - p[0], b - p[1]))
        choices = [early, sweep_only]
        if require_rotation:
            choices.append(rotation_difference)
        possible = True
        for pool in choices:
            options = [p for p in sorted(pool) if p not in protected_visible
                       and set(lookup[p]) - ink - protected - {p}]
            if not options:
                possible = False
                break
            protected.add(options[int(rng.integers(len(options)))])
        if not possible:
            continue
        # Include several stencil gaps spread across stops, plus independent
        # off-route gaps. The optional topology is not tied to the split.
        gaps = _raw_gaps(rng, shape) - ink - protected_visible
        candidates = sorted(eligible - protected_visible)
        rng.shuffle(candidates)
        gaps.update(candidates[:min(len(candidates), max(3, steps))])
        gaps.update(protected)
        valid = True
        for orbit in orbits:
            if not set(orbit) - ink - gaps:
                options = sorted(set(orbit) - ink - protected)
                if not options:
                    valid = False
                    break
                gaps.discard(options[int(rng.integers(len(options)))])
        if not valid:
            continue
        colors = tuple(int(v) for v in rng.choice(np.arange(1, 9), int(rng.integers(2, 9)), replace=False))
        style = int(rng.integers(5))
        for texture_attempt in range(12):
            field = _field(rng, shape, axes, colors, style if texture_attempt < 6 else 0)
            inp = field.copy()
            for p in gaps:
                inp[p] = GAP
            for p in ink:
                inp[p] = INK
            if any(_axis_candidates(inp, d) != [axes[d]] for d in (0, 1)):
                continue
            selected = gaps & coverage
            if len({int(field[p]) for p in selected}) < 2:
                continue
            out = inp.copy()
            for p in ink | selected:
                out[p] = field[p]
            return {"input": inp, "output": out}
    raise RuntimeError("Bounded courier construction exhausted")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _fingerprint(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ns = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    masks = rng.permutation(len(MASKS))
    # Shuffled teaching order removes a fixed spatial/example-index convention.
    profiles = [(4, "turn", True, True), (2, "straight", False, False)]
    for _ in range(nt - 2):
        profiles.append((int(rng.integers(2, 5)), "any", bool(rng.random() < 0.45), False))
    rng.shuffle(profiles)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ns)):
        for i in range(count):
            steps, mode, hard, rotation = profiles[i] if split == "train" else (
                int(rng.integers(5, 9)), "turn", True, True)
            mask = MASKS[int(masks[i])] if split == "train" else MASKS[int(rng.integers(len(MASKS)))]
            for _ in range(12):
                pair = _make_pair(rng, steps, mode, mask, hard, rotation)
                key = _fingerprint(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("Distinct input construction exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(grid, info):
    gaps = {tuple(map(int, p)) for p in np.argwhere(grid == GAP)}
    ink = {tuple(map(int, p)) for p in np.argwhere(grid == INK)}
    selected = gaps & info["coverage"]
    a, b = info["axes"]
    diagonal = 0
    for r, c in selected:
        h, v, d = (a - r, c), (r, b - c), (a - r, b - c)
        diagonal += int(h != (r, c) and v != (r, c)
                        and _bounded([h, v, d], grid.shape)
                        and grid[h] in (GAP, INK) and grid[v] in (GAP, INK)
                        and grid[d] not in (GAP, INK))
    fixed = set().union(*(_footprint(info["mask"], p, info["initial"])
                         for p in info["points"][1:]))
    return dict(steps=len(info["directions"]), mask=info["mask"], axes=info["axes"],
                initial=info["initial"], final=info["directions"][-1],
                distances=tuple(abs(x[0]-y[0])+abs(x[1]-y[1])
                                for x,y in zip(info["points"],info["points"][1:])),
                turns=tuple((y-x)%4 for x,y in zip(info["directions"],info["directions"][1:]) if x!=y),
                selected=len(selected), untouched=len(gaps - selected), diagonal=diagonal,
                early=bool(selected - info["poses"][-1]),
                sweep_gap=bool(gaps & (_swept_cells(info) - info["coverage"]) - ink),
                rotation=bool(gaps & (fixed ^ info["coverage"]) - ink))


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _as_grid(pair["input"]), _as_grid(pair["output"])
            assert inp.shape == out.shape
            key = _fingerprint(inp)
            assert key not in seen, "Duplicate input"
            seen.add(key)
            info = _analyze(inp)
            f = _features(inp, info)
            features[split].append(f)
            # Validate selective orbit repair directly, including unchanged
            # payload, all overlay removal, and every gap outside the stencil.
            for p in np.ndindex(inp.shape):
                change = inp[p] == INK or (inp[p] == GAP and p in info["coverage"])
                expected = info["field"][p] if change else inp[p]
                assert out[p] == expected, "Wrong repair, footprint, or preserved pixel"
            assert not np.any(out == INK)
            assert f["selected"] >= 2 and f["untouched"] >= 1
            assert f["early"] and f["sweep_gap"], "Need earlier repairs and unbrushed transit gaps"
            assert len({int(info["field"][p]) for p in info["coverage"] if inp[p] == GAP}) >= 2
    train, test = features["train"], features["test"]
    assert len({f["mask"] for f in train}) >= 2
    assert any(f["steps"] == 2 and not f["turns"] for f in train)
    assert any(f["turns"] and f["rotation"] and f["diagonal"] for f in train)
    assert max(f["steps"] for f in train) <= 4
    assert all(f["steps"] >= 5 and f["rotation"] and f["diagonal"] for f in test)


def self_test() -> None:
    """Bounded 42-seed semantic/diversity sweep, determinism, and negative checks."""
    shapes, axes, masks, directions, palettes, routes = set(), set(), set(), set(), set(), set()
    parity, offsets, turn_senses, lengths, counts, keys = set(), set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = tuple(_fingerprint(p["input"]) for p in puzzle["train"] + puzzle["test"])
        keys.add(signature)
        repeat = generate_puzzle(seed)
        assert all(np.array_equal(p[k], q[k]) for s in puzzle
                   for p,q in zip(puzzle[s], repeat[s]) for k in ("input", "output"))
        for pair in puzzle["train"] + puzzle["test"]:
            inp, out = pair["input"], pair["output"]
            info = _analyze(inp)
            assert np.array_equal(solve(inp), out)
            f = _features(inp, info)
            # Concrete parent/interaction ablations: full B completion omits
            # the transported selector; terminal-only and continuous brushing
            # omit distinct aspects of A's discrete journey.
            assert not np.array_equal(info["field"], out)
            terminal_only = inp.copy()
            terminal_only[inp == INK] = info["field"][inp == INK]
            for p in info["poses"][-1]:
                if inp[p] == GAP:
                    terminal_only[p] = info["field"][p]
            assert not np.array_equal(terminal_only, out)
            swept = out.copy()
            for p in _swept_cells(info):
                if _bounded([p], inp.shape) and inp[p] == GAP:
                    swept[p] = info["field"][p]
            assert not np.array_equal(swept, out)
            if f["rotation"]:
                fixed = inp.copy()
                fixed[inp == INK] = info["field"][inp == INK]
                for tip in info["points"][1:]:
                    for p in _footprint(info["mask"], tip, info["initial"]):
                        if _bounded([p], inp.shape) and inp[p] == GAP:
                            fixed[p] = info["field"][p]
                assert not np.array_equal(fixed, out)
            shapes.add(inp.shape)
            axes.add(info["axes"])
            masks.add(info["mask"])
            directions.update((f["initial"], f["final"]))
            palettes.add(tuple(int(v) for v in np.unique(info["field"])))
            parity.add(tuple(v % 2 for v in info["axes"]))
            offsets.add(tuple(int(np.sign(a - n + 1)) for a,n in zip(info["axes"], inp.shape)))
            turn_senses.update(f["turns"])
            lengths.update(f["distances"])
            counts.add(f["steps"])
            origin = info["points"][0]
            routes.add(tuple((r-origin[0],c-origin[1]) for r,c in info["points"]))
    assert len(keys) == 42 and len(shapes) >= 60 and len(masks) >= 12
    assert directions == {0,1,2,3} and parity == {(0,0),(0,1),(1,0),(1,1)}
    assert len(offsets) >= 7 and len(palettes) >= 35 and len(routes) >= 150
    assert {1,3} <= turn_senses and len(lengths) >= 10 and counts == set(range(2,9))
    for nt in range(2,11):
        for ns in range(1,min(4,nt-1)+1):
            p = generate_puzzle(101,nt,ns)
            assert len(p["train"]) == nt and len(p["test"]) == ns
    for nt,ns in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1),(4,None)):
        try:
            generate_puzzle(0,nt,ns)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    sample = generate_puzzle(0)
    bad = {s:[{k:v.copy() for k,v in p.items()} for p in sample[s]] for s in sample}
    bad["test"][0]["output"][0,0] = (int(bad["test"][0]["output"][0,0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Corrupted output accepted")
    # No hidden tie break: an additional ray from the start must be rejected.
    board = np.ones((12,12),dtype=np.int8)
    mask = _arrow(1,1,True,0)
    for p in _footprint(mask,(5,5),1) | {(5,8),(8,5)}:
        board[p] = INK
    try:
        _read_route(board)
    except ValueError:
        pass
    else:
        raise AssertionError("Branching route accepted")
    # A missing entire reflection orbit is never guessed from neighboring
    # texture. Such an input is outside the specified recoverable family.
    inp = sample["train"][0]["input"].copy()
    info = _analyze(inp)
    p = next(p for p in sorted(info["coverage"]) if inp[p] == GAP)
    orbit = next(o for o in _orbits(inp.shape,info["axes"]) if p in o)
    for q in orbit:
        if inp[q] != INK:
            inp[q] = GAP
    try:
        solve(inp)
    except ValueError:
        pass
    else:
        raise AssertionError("Witness-free orbit accepted")
    # Change only field payload in one orbit while keeping all route geometry
    # and gap locations fixed. Mirror information, rather than a route/color
    # constant, must change the selected repair's value.
    original = sample["train"][0]["input"]
    info = _analyze(original)
    p = next(p for p in sorted(info["coverage"]) if original[p] == GAP)
    orbit = next(o for o in _orbits(original.shape,info["axes"]) if p in o)
    changed_value = False
    for color in range(1,9):
        if color == info["field"][p]:
            continue
        variant = original.copy()
        for q in orbit:
            if variant[q] not in (GAP,INK):
                variant[q] = color
        try:
            after = _analyze(variant)
        except ValueError:
            continue
        predicted = solve(variant)
        assert after["points"] == info["points"] and after["coverage"] == info["coverage"]
        assert predicted[p] == color and predicted[p] != solve(original)[p]
        assert np.array_equal(predicted == GAP, solve(original) == GAP)
        changed_value = True
        break
    assert changed_value, "Could not demonstrate geometry-preserving mirror value dependence"
