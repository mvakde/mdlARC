"""Periodic harbor lights: repair a rhythm in the mover's travelling frame.

Only standard-library modules and NumPy are used. The solver reads the input;
generation metadata never enters decoding or prediction.
"""
from __future__ import annotations

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # up, right, down, left


def _rotate(point, turns):
    r, c = point
    for _ in range(turns % 4):
        r, c = c, -r
    return r, c


def _footprint(mask, tip, direction):
    return {(tip[0] + r, tip[1] + c)
            for r, c in (_rotate(p, direction - 1) for p in mask)}


def _inside(cells, shape):
    return all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in cells)


def _light_slots(tip, direction):
    dr, dc = DIRS[direction]
    # Counterclockwise from travel is port (left); clockwise is starboard.
    return (tip[0] - dc, tip[1] + dr), (tip[0] + dc, tip[1] - dr)


def _components(cells):
    unseen = set(cells)
    result = []
    while unseen:
        start = unseen.pop()
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = r + dr, c + dc
                    if q in unseen:
                        unseen.remove(q)
                        component.add(q)
                        stack.append(q)
        result.append(component)
    return result


def _read_arrow(cells):
    """Infer the actual offsets from symmetry and a unique projecting tip."""
    possibilities = []
    for tip in sorted(cells):
        for direction in range(4):
            mask = {_rotate((r - tip[0], c - tip[1]), 1 - direction)
                    for r, c in cells}
            if (all(c <= 0 for r, c in mask)
                    and {p for p in mask if p[1] == 0} == {(0, 0)}
                    and any(r != 0 for r, c in mask)
                    and {(-r, c) for r, c in mask} == mask):
                possibilities.append((tuple(sorted(mask)), tip, direction))
    assert len(possibilities) == 1, "arrow must have a unique geometric tip"
    return possibilities[0]


def _ray(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    if dr == 0 and dc != 0:
        return (1 if dc > 0 else 3), abs(dc)
    if dc == 0 and dr != 0:
        return (2 if dr > 0 else 0), abs(dr)
    return None


def _order_stations(start, stations):
    remaining = set(stations)
    points, directions = [start], []
    while remaining:
        candidates = [(q, _ray(points[-1], q)) for q in remaining]
        candidates = [(q, ray) for q, ray in candidates if ray is not None]
        assert len({ray[0] for q, ray in candidates}) == 1, \
            "exactly one orthogonal ray must contain remaining stations"
        q, (direction, _) = min(candidates, key=lambda item: item[1][1])
        remaining.remove(q)
        points.append(q)
        directions.append(direction)
    return points, directions


def _check_grid(value):
    grid = np.asarray(value)
    assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))
    return grid.astype(np.int8, copy=False)


def _decode_input(value):
    grid = _check_grid(value)
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 3, "background, mover/station ink, and light ink required"
    bg = int(colors[np.argmax(counts)])
    assert np.max(counts) > grid.size // 2
    alternatives = []
    for ink in (int(c) for c in colors if c != bg):
        light = next(int(c) for c in colors if c not in (bg, ink))
        try:
            components = _components(set(zip(*np.where(grid == ink))))
            arrows = [c for c in components if len(c) > 1]
            assert len(arrows) == 1
            mask, start, initial = _read_arrow(arrows[0])
            stations = {next(iter(c)) for c in components if len(c) == 1}
            assert len(stations) >= 8
            points, directions = _order_stations(start, stations)
            assert directions[0] == initial
            observed = np.zeros((2, len(stations)), dtype=np.int8)
            slots = set()
            for i, (tip, d) in enumerate(zip(points[1:], directions)):
                pair = _light_slots(tip, d)
                assert _inside(pair, grid.shape)
                assert not slots.intersection(pair), "signal slots must be disjoint"
                slots.update(pair)
                assert _inside(_footprint(mask, tip, d), grid.shape)
                for side, q in enumerate(pair):
                    assert int(grid[q]) in (bg, light)
                    observed[side, i] = int(grid[q] == light)
            actual_lights = set(zip(*np.where(grid == light)))
            assert actual_lights <= slots, "every light belongs to a station"
            alternatives.append(dict(grid=grid, bg=bg, ink=ink, light=light,
                                     mask=mask, start=start, initial=initial,
                                     points=points, directions=directions,
                                     observed=observed))
        except AssertionError:
            continue
    assert len(alternatives) == 1, "color roles and route must be geometrically unique"
    return alternatives[0]


def _recover_lights(observed):
    """Unique nearest periodic repair, considering every four-copy period.

    Each template bit needs a strict majority. Distinct periods can represent
    the same repaired word; a tie between different repaired words is invalid.
    """
    n = observed.shape[1]
    candidates = []
    for period in range(1, n // 4 + 1):
        template = np.zeros((2, period), dtype=np.int8)
        supported = True
        for phase in range(period):
            group = observed[:, phase::period]
            ones = np.sum(group, axis=1)
            if np.any(2 * ones == group.shape[1]):
                supported = False
                break
            template[:, phase] = (2 * ones > group.shape[1]).astype(np.int8)
        if supported:
            repaired = template[:, np.arange(n) % period]
            cost = int(np.count_nonzero(repaired != observed))
            if cost <= (2 * n) // 5:
                candidates.append((cost, period, repaired))
    assert candidates, "no sparsely corrupted, majority-supported rhythm"
    cost = min(item[0] for item in candidates)
    best = [item for item in candidates if item[0] == cost]
    repaired = best[0][2]
    assert all(np.array_equal(item[2], repaired) for item in best), \
        "different periodic repairs tie for minimum error"
    period = min(item[1] for item in best)
    assert np.any(np.all(repaired.T == (1, 0), axis=1)), "no open berth"
    assert np.any(np.any(repaired.T != (1, 0), axis=1)), "rhythm must contain closed stations"
    return repaired.copy(), period, cost


def _last_open(bits):
    indices = np.flatnonzero((bits[0] == 1) & (bits[1] == 0))
    return int(indices[-1]) if len(indices) else None


def _terminal(a, index, rotate=True):
    result = np.full(a["grid"].shape, a["bg"], dtype=np.int8)
    direction = a["directions"][index] if rotate else a["initial"]
    cells = _footprint(a["mask"], a["points"][index + 1], direction)
    assert _inside(cells, result.shape)
    for q in cells:
        result[q] = a["ink"]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Repair route-ordered lights and dock the input arrow at the last 10."""
    a = _decode_input(input_grid)
    repaired, _, _ = _recover_lights(a["observed"])
    return _terminal(a, _last_open(repaired))


def _make_masks():
    masks = set()
    for span in (1, 2):
        for extension in (1, 2):
            for solid in (False, True):
                cells = {(0, -i) for i in range(extension)}
                for depth in range(1, span + 1):
                    rows = range(-depth, depth + 1) if solid else (-depth, depth)
                    cells.update((r, -(extension - 1 + depth)) for r in rows)
                masks.add(tuple(sorted(cells)))
    return tuple(sorted(masks))


MASKS = _make_masks()


def _grow_route(rng, mask, n, shape):
    """Sample distances and turns; adding a point must preserve every old ray."""
    for attempt in range(48):
        start = (int(rng.integers(3, shape[0] - 3)),
                 int(rng.integers(3, shape[1] - 3)))
        points, directions, patches = [start], [], set()
        initial_cells = None
        straight_weight = float(rng.uniform(0.35, 1.8))
        while len(directions) < n:
            options = []
            for d, (dr, dc) in enumerate(DIRS):
                for distance in range(2, min(11, max(shape))):
                    q = (points[-1][0] + dr * distance,
                         points[-1][1] + dc * distance)
                    lamps = set(_light_slots(q, d))
                    patch = lamps | {q}
                    if not _inside(patch | _footprint(mask, q, d), shape):
                        continue
                    first = initial_cells if directions else _footprint(mask, start, d)
                    if not _inside(first, shape) or patch.intersection(first):
                        continue
                    if any(max(abs(q[0] - r), abs(q[1] - c)) <= 1 for r, c in first):
                        continue
                    if any(max(abs(q[0] - p[0]), abs(q[1] - p[1])) <= 1 for p in points):
                        continue
                    if patches.intersection(patch):
                        continue
                    valid = True
                    for p, nxt, old in zip(points[:-1], points[1:], directions):
                        ray = _ray(p, q)
                        if ray is not None and (ray[0] != old or ray[1] <= _ray(p, nxt)[1]):
                            valid = False
                            break
                    if valid:
                        weight = (straight_weight if directions and d == directions[-1] else 1.0)
                        options.append((d, q, patch, weight / (distance - 1) ** 1.4))
            if not options:
                break
            weights = np.array([o[3] for o in options])
            d, q, patch, _ = options[int(rng.choice(len(options), p=weights / weights.sum()))]
            if initial_cells is None:
                initial_cells = _footprint(mask, start, d)
            directions.append(d)
            points.append(q)
            patches.update(patch)
        if len(directions) == n:
            return points, directions
    return None


def _fallback_route(mask, n, teach_target=None):
    # A finite construction for the rare exhausted random search. An alternating
    # staircase with two-cell legs fits 20 stations plus all supported arrows.
    points, directions = [(3, 3)], []
    ds = [1 if i % 2 == 0 else 2 for i in range(n)]
    if teach_target is not None and ds[teach_target] == 1:
        ds[1], ds[teach_target] = ds[teach_target], ds[1]
    for d in ds:
        dr, dc = DIRS[d]
        points.append((points[-1][0] + 2 * dr, points[-1][1] + 2 * dc))
        directions.append(d)
    if teach_target is not None:
        points = [(29 - r, 29 - c) for r, c in points]
        directions = [(d + 2) % 4 for d in directions]
    return points, directions, (30, 30)


def _sample_lights(rng, period, n, closed_tail, missing_target, teaching=None):
    for attempt in range(192):
        if teaching == 0:
            unit = [(1, 0), (0, 0)]
            rng.shuffle(unit)
        elif teaching == 1:
            unit = [(1, 0), (0, 1), (1, 1)]
            rng.shuffle(unit)
        elif attempt < 160:
            unit = [(1, 0), (0, 1)]
            unit += [tuple(int(x) for x in rng.integers(0, 2, 2)) for _ in range(period - 2)]
            rng.shuffle(unit)
        else:
            unit = [(1, 0), (0, 1), (0, 0), (1, 1)][:period]
        template = np.array(unit, dtype=np.int8).T
        shifts = [s for s in range(period)
                  if ((tuple(template[:, (n - 1 + s) % period]) != (1, 0)) == closed_tail
                      and (teaching != 0 or tuple(template[:, (n - 1 + s) % period]) == (0, 0))
                      and (teaching != 1 or tuple(template[:, (n - 1 + s) % period]) == (1, 1)))]
        shift = int(rng.choice(shifts))
        clean = template[:, (np.arange(n) + shift) % period]
        target = _last_open(clean)
        noisy, used = clean.copy(), set()

        def flip(side, index):
            key = side, index % period
            assert key not in used
            used.add(key)
            noisy[side, index] ^= 1

        if closed_tail:
            for side, expected in enumerate((1, 0)):
                if noisy[side, -1] != expected:
                    flip(side, n - 1)  # a false open berth at the end
        else:
            flip(0, target)  # erase the genuine final port light
        if closed_tail and missing_target:
            flip(0, target)
        # Damage the first copy of the chosen phase as well. Copying the first
        # repeat then loses the actual last open phase. Groups remain distinct.
        flip(1 if missing_target or not closed_tail else 0, target % period)
        if attempt < 160:
            spare = [(side, phase) for side in range(2) for phase in range(period)
                     if (side, phase) not in used]
            rng.shuffle(spare)
            extra = int(rng.integers(0, min(len(spare), (2 * n) // 5 - len(used), 2) + 1))
            for side, phase in spare[:extra]:
                indices = np.arange(phase, n, period)
                flip(side, int(rng.choice(indices)))
        try:
            repaired, recovered_period, _ = _recover_lights(noisy)
        except AssertionError:
            continue
        if recovered_period != period or not np.array_equal(repaired, clean):
            continue
        first_copy = noisy[:, np.arange(n) % period]
        if _last_open(noisy) != target and _last_open(first_copy) != target:
            return noisy, clean, target
    raise RuntimeError("bounded light-word construction exhausted")


def _draw_pair(shape, palette, mask, points, directions, noisy, target):
    bg, ink, light = palette
    inp = np.full(shape, bg, dtype=np.int8)
    for q in _footprint(mask, points[0], directions[0]):
        inp[q] = ink
    for i, (tip, d) in enumerate(zip(points[1:], directions)):
        assert inp[tip] == bg
        inp[tip] = ink
        for side, q in enumerate(_light_slots(tip, d)):
            assert inp[q] == bg
            if noisy[side, i]:
                inp[q] = light
    out = np.full(shape, bg, dtype=np.int8)
    for q in _footprint(mask, points[target + 1], directions[target]):
        out[q] = ink
    return {"input": inp, "output": out}


def _make_pair(rng, palette, mask, period, n, closed_tail, missing_target, teach, teaching=None):
    noisy, clean, target = _sample_lights(rng, period, n, closed_tail, missing_target, teaching)
    for attempt in range(40):
        # Geometry is independently sampled, with enough room for the rhythm.
        lo = 18 if n <= 10 else 22 if n <= 14 else 25
        shape = tuple(int(x) for x in rng.integers(lo, 31, 2))
        route = _grow_route(rng, mask, n, shape) if attempt < 39 else None
        if route is None:
            points, directions, shape = _fallback_route(mask, n, target if teach else None)
        else:
            points, directions = route
        if teach and directions[target] == directions[0]:
            continue
        # Quarter-turn whole boards, including their margins. This does not
        # reflect arrows or exchange port and starboard.
        pair = _draw_pair(shape, palette, mask, points, directions, noisy, target)
        rotations = []
        for turns in rng.permutation(4):
            rotated = {k: np.rot90(v, int(turns)).copy() for k, v in pair.items()}
            try:
                a = _decode_input(rotated["input"])
                repaired, p, _ = _recover_lights(a["observed"])
                assert p == period and np.array_equal(repaired, clean)
                assert np.array_equal(_terminal(a, target), rotated["output"])
                if teach:
                    assert a["initial"] != a["directions"][target]
                    last_raster = max((a["points"][i + 1], i) for i in range(n)
                                      if tuple(clean[:, i]) == (1, 0))[1]
                    assert last_raster != target
            except AssertionError:
                continue
            rotations.append(rotated)
        if rotations:
            return rotations[int(rng.integers(len(rotations)))]
    raise RuntimeError("bounded geometric construction exhausted")


def _key(grid):
    grid = np.asarray(grid)
    return grid.shape, tuple(int(x) for x in grid.flat)


def _validate_count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train = _validate_count(num_train, "num_train", 2, 10)
    num_test = _validate_count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    shared_palette = bool(rng.random() < 0.2)
    palette = tuple(int(x) for x in rng.choice(10, 3, replace=False))
    first_mask = None
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            period = 2 if split == "train" and i == 0 else 3 if split == "train" and i == 1 \
                else int(rng.integers(2, 5))
            n = int(rng.integers(4 * period, 5 * period)) if split == "train" \
                else int(rng.integers(max(16, 4 * period), 21))
            pool = [m for m in MASKS if not (split == "train" and i == 1 and m == first_mask)]
            mask = pool[int(rng.integers(len(pool)))]
            if split == "train" and i == 0:
                first_mask = mask
            if not shared_palette:
                palette = tuple(int(x) for x in rng.choice(10, 3, replace=False))
            closed = True if split == "train" and i < 2 else bool(rng.random() < 0.65)
            missing = bool(split == "train" and i == 1 or rng.random() < 0.35)
            # The period-2 shortest case has a three-error budget.
            if period == 2 and n < 10 and closed:
                missing = False
            for attempt in range(24):
                pair = _make_pair(rng, palette, mask, period, n, closed, missing,
                                  teach=(split == "train" and i == 0),
                                  teaching=i if split == "train" and i < 2 else None)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyse_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = _check_grid(pair["input"]), _check_grid(pair["output"])
    assert inp.shape == out.shape
    a = _decode_input(inp)
    clean, period, errors = _recover_lights(a["observed"])
    target = _last_open(clean)
    assert np.array_equal(out, _terminal(a, target)), "wrong dock or non-rigid mover"
    assert 1 <= errors <= (2 * clean.shape[1]) // 5
    assert set(np.unique(out)) == {a["bg"], a["ink"]}
    assert np.count_nonzero(out == a["ink"]) == len(a["mask"])
    assert _last_open(a["observed"]) != target, "raw lights must be a real counterexample"
    assert _last_open(a["observed"][:, np.arange(clean.shape[1]) % period]) != target
    # Construction uses at most one flipped bit in each phase/side group.
    for phase in range(period):
        assert np.all(np.sum(clean[:, phase::period] != a["observed"][:, phase::period], axis=1) <= 1)
    a.update(clean=clean, period=period, errors=errors, target=target)
    return a


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            a = _analyse_pair(pair)
            key = _key(pair["input"])
            assert key not in seen
            seen.add(key)
            analyses[split].append(a)
    train = analyses["train"]
    assert len({a["mask"] for a in train}) >= 2
    assert {2, 3} <= {a["period"] for a in train}
    assert len({len(a["directions"]) for a in train}) >= 2
    assert all(train[i]["target"] < len(train[i]["directions"]) - 1 for i in (0, 1))
    assert tuple(train[0]["clean"][:, -1]) == (0, 0)
    assert tuple(train[1]["clean"][:, -1]) == (1, 1)
    # Teach the conjunction rather than just one side's state, even at 2/1.
    signals = {tuple(v) for a in train[:2] for v in a["clean"].T}
    assert signals == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert int(np.flatnonzero(train[0]["clean"][1] == 0)[-1]) != train[0]["target"]
    assert int(np.flatnonzero(train[1]["clean"][0] == 1)[-1]) != train[1]["target"]
    assert train[0]["initial"] != train[0]["directions"][train[0]["target"]]
    raster_last = max((train[0]["points"][i + 1], i)
                      for i in range(len(train[0]["directions"]))
                      if tuple(train[0]["clean"][:, i]) == (1, 0))[1]
    assert raster_last != train[0]["target"]
    assert train[1]["observed"][0, train[1]["target"]] == 0
    assert all(len(a["directions"]) >= 16 for a in analyses["test"])


def self_test():
    """Bounded 42-seed sweep, geometric invariants, ablations, and negatives."""
    rows, puzzles = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyse_pair(pair)
                a["split"] = split
                rows.append(a)
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
        first = _analyse_pair(puzzle["train"][0])
        translated_cells = _footprint(first["mask"], first["points"][first["target"] + 1], first["initial"])
        output_cells = set(zip(*np.where(puzzle["train"][0]["output"] == first["ink"])))
        assert translated_cells != output_cells  # out-of-bounds cells also fail the ablation
    assert len(puzzles) == 42
    for split in ("train", "test"):
        group = [a for a in rows if a["split"] == split]
        route_shapes = set()
        for a in group:
            offsets = [(r - a["start"][0], c - a["start"][1]) for r, c in a["points"]]
            route_shapes.add(min(tuple(_rotate((r, sign * c), turns) for r, c in offsets)
                                 for sign in (-1, 1) for turns in range(4)))
        assert len(route_shapes) >= (100 if split == "train" else 30)
        assert {a["initial"] for a in group} == set(range(4))
        assert {a["directions"][a["target"]] for a in group} == set(range(4))
        assert {a["period"] for a in group} == {2, 3, 4}
        assert len({a["mask"] for a in group}) >= 6
        assert len({a["grid"].shape for a in group}) >= 12
        assert len({len(a["directions"]) for a in group}) >= (8 if split == "train" else 4)
        for role in ("bg", "ink", "light"):
            assert len({a[role] for a in group}) >= 8
    assert len({a["errors"] for a in rows}) >= 4
    assert any(a["target"] == len(a["directions"]) - 1 for a in rows)
    assert any(len(a["directions"]) % a["period"] for a in rows)
    assert any(not len(a["directions"]) % a["period"] for a in rows)
    assert any(1 in {(b - d) % 4 for d, b in zip(a["directions"], a["directions"][1:])}
               and 3 in {(b - d) % 4 for d, b in zip(a["directions"], a["directions"][1:])} for a in rows)
    assert any(d == b for a in rows for d, b in zip(a["directions"], a["directions"][1:]))
    # Source-free fallback check; no file reads or parent imports.
    for mask in MASKS:
        points, directions, shape = _fallback_route(mask, 20)
        noisy, clean, target = _sample_lights(np.random.default_rng(90), 4, 20, True, True)
        pair = _draw_pair(shape, (0, 1, 2), mask, points, directions, noisy, target)
        assert np.array_equal(solve(pair["input"]), pair["output"])
    for a, b in ((2, 1), (10, 1), (10, 4), (5, 4)):
        p = generate_puzzle(100 + a + b, a, b)
        assert (len(p["train"]), len(p["test"])) == (a, b)
    a, b = generate_puzzle(900, 3, 2), generate_puzzle(900, 3, 2)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    try:
        _order_stations((3, 3), {(3, 6), (6, 3)})
    except AssertionError:
        pass
    else:
        raise AssertionError("competing rays accepted")
    ambiguous = np.array([[int(c) for c in "100100101010"], [0] * 12], dtype=np.int8)
    try:
        _recover_lights(ambiguous)
    except AssertionError:
        pass
    else:
        raise AssertionError("different equally close rhythmic repairs accepted")
    broken = generate_puzzle(31, 2, 1)
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
