"""A damaged recursive arrow travels between tile-scale beacons and repairs itself.

Only the input bitmap is used by solve. Generation renders its own sampled
geometry; validation decodes the returned bitmap rather than retaining metadata.
"""
import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _rot(point, turns):
    r, c = point
    for _ in range(turns % 4):
        r, c = c, -r
    return r, c


def _offsets(mask, direction):
    return tuple(_rot(p, direction - 1) for p in mask)


def _footprint(offsets, tip):
    return {(tip[0] + r, tip[1] + c) for r, c in offsets}


def _fits(points, shape):
    return all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in points)


def _direction(a, b):
    r, c = b[0] - a[0], b[1] - a[1]
    assert (r == 0) != (c == 0), "distinct orthogonal references required"
    return (1 if c > 0 else 3) if r == 0 else (2 if r > 0 else 0)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer), "integer grids required"
    assert np.all((0 <= a) & (a <= 9))
    return a


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _normalized(a):
    roles = {}
    return a.shape, tuple(roles.setdefault(int(c), len(roles)) for c in a.flat)


def _runs(indices):
    groups = []
    for i in map(int, indices):
        if groups and i == groups[-1][1]:
            groups[-1] = (groups[-1][0], i + 1)
        else:
            groups.append((i, i + 1))
    return groups


def _detect_lattice(grid):
    candidates = []
    for color in map(int, np.unique(grid)):
        groups = [_runs(np.flatnonzero(np.all(grid == color, axis=1 - axis)))
                  for axis in (0, 1)]
        if any(len(gs) < 2 for gs in groups):
            continue
        thicknesses = {b - a for gs in groups for a, b in gs}
        if len(thicknesses) != 1 or next(iter(thicknesses)) not in (1, 2):
            continue
        t = next(iter(thicknesses))
        intervals, borders = [], []
        for axis, gs in enumerate(groups):
            framed = gs[0][0] == 0 and gs[-1][1] == grid.shape[axis]
            if framed:
                spans = [(b, c) for (_, b), (c, _) in zip(gs, gs[1:])]
            else:
                spans = [(0, gs[0][0])]
                spans += [(b, c) for (_, b), (c, _) in zip(gs, gs[1:])]
                spans += [(gs[-1][1], grid.shape[axis])]
            intervals.append(spans)
            borders.append(framed)
        sizes = {b - a for spans in intervals for a, b in spans}
        if len(sizes) != 1 or next(iter(sizes)) not in (3, 4) or borders[0] != borders[1]:
            continue
        if min(map(len, intervals)) < 5:
            continue
        side = next(iter(sizes))
        if any(color in grid[y:y + side, x:x + side]
               for y, _ in intervals[0] for x, _ in intervals[1]):
            continue
        candidates.append(((side, t, borders[0], len(intervals[0]), len(intervals[1])),
                           color, intervals))
    assert len(candidates) == 1, "one regular square-tile lattice is required"
    return candidates[0]


def _order_route(tip, waypoints):
    remaining, route, directions = set(waypoints), [tip], []
    while remaining:
        rays = {}
        for q in remaining:
            if q[0] == tip[0] or q[1] == tip[1]:
                rays.setdefault(_direction(tip, q), []).append(q)
        assert len(rays) == 1, "zero or competing occupied rays"
        direction, candidates = next(iter(rays.items()))
        nxt = min(candidates, key=lambda q: abs(q[0] - tip[0]) + abs(q[1] - tip[1]))
        remaining.remove(nxt)
        route.append(nxt)
        directions.append(direction)
        tip = nxt
    return route, directions


def _infer(input_grid):
    grid = _grid(input_grid)
    geometry, divider, intervals = _detect_lattice(grid)
    side, thickness, border, rows, cols = geometry
    records = {}
    for r, (y, _) in enumerate(intervals[0]):
        for c, (x, _) in enumerate(intervals[1]):
            tile = grid[y:y + side, x:x + side]
            records.setdefault(_key(tile), [tile.copy(), []])[1].append((r, c))
    assert len(records) == 3, "ordinary, glyph and marked-tip tiles required"
    ordered = sorted(records.values(), key=lambda item: len(item[1]), reverse=True)
    ordinary, ordinary_positions = ordered[0]
    glyph, glyph_positions = ordered[1]
    marked, marked_positions = ordered[2]
    assert len(ordinary_positions) > len(glyph_positions) + 1
    assert len(glyph_positions) >= 3 and len(marked_positions) == 1
    ordinary_colors = set(map(int, np.unique(ordinary)))
    assert len(ordinary_colors) == 2
    ink_candidates = set(map(int, np.unique(glyph))) - ordinary_colors
    assert len(ink_candidates) == 1
    ink = ink_candidates.pop()
    mask_bitmap = glyph == ink
    assert 3 <= int(mask_bitmap.sum()) < side * side
    assert len(np.unique(glyph[~mask_bitmap])) == 1
    background = int(glyph[~mask_bitmap][0])
    assert background in ordinary_colors
    differences = np.argwhere(marked != glyph)
    assert len(differences) == 1, "marker replaces exactly the miniature tip pixel"
    fine_tip = tuple(map(int, differences[0]))
    anchor_color = int(marked[fine_tip])
    assert glyph[fine_tip] == ink
    assert anchor_color not in ordinary_colors | {ink, divider}
    offsets = tuple((int(r) - fine_tip[0], int(c) - fine_tip[1])
                    for r, c in np.argwhere(mask_bitmap))
    facings = []
    for d in range(4):
        canonical = {_rot(p, 1 - d) for p in offsets}
        if (all(c <= 0 for r, c in canonical)
                and {p for p in canonical if p[1] == 0} == {(0, 0)}
                and min(r for r, c in canonical) < 0 < max(r for r, c in canonical)):
            facings.append(d)
    assert len(facings) == 1, "marked miniature tip must have one orthogonal facing"
    initial = facings[0]
    canonical_mask = tuple(sorted(_rot(p, 1 - initial) for p in offsets))
    tip = marked_positions[0]
    initial_shape = _footprint(offsets, tip)
    assert _fits(initial_shape, (rows, cols))
    supplied = set(glyph_positions)
    body = supplied & initial_shape
    assert 1 <= len(body) <= len(offsets) - 2, "strict damaged body required"
    waypoints = supplied - initial_shape
    assert 2 <= len(waypoints) <= 9
    route, directions = _order_route(tip, waypoints)
    assert directions[0] == initial, "initial leg must follow the small arrow"
    for i, point in enumerate(route):
        d = initial if i == 0 else directions[i - 1]
        cells = _footprint(tuple(_rot(p, d - initial) for p in offsets), point)
        assert _fits(cells, (rows, cols)), "all arrival footprints must fit"
        assert not cells & set(route[i + 1:]), "future beacons must stay uncovered"
    turns = (directions[-1] - initial) % 4
    final_offsets = tuple(_rot(p, turns) for p in offsets)
    final_shape = _footprint(final_offsets, route[-1])
    final_glyph = np.rot90(glyph, -turns).copy()
    expected = grid.copy()
    for r, (y, _) in enumerate(intervals[0]):
        for c, (x, _) in enumerate(intervals[1]):
            expected[y:y + side, x:x + side] = final_glyph if (r, c) in final_shape else ordinary
    info = dict(geometry=geometry, divider=divider, ordinary=ordinary, glyph=glyph,
                marked=marked, ink=ink, background=background, anchor_color=anchor_color,
                tip=tip, fine_tip=fine_tip, initial=initial, final=directions[-1],
                mask=canonical_mask, offsets=offsets, body=body, initial_shape=initial_shape,
                waypoints=waypoints, route=route, directions=directions,
                final_shape=final_shape, final_glyph=final_glyph, intervals=intervals,
                fragment=tuple(sorted(_rot((r - tip[0], c - tip[1]), 1 - initial)
                                      for r, c in body)))
    return expected, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode palette, damaged stencil, route and synchronized rotations from input."""
    return _infer(input_grid)[0]


def _paint(geometry, divider, ordinary, glyph, positions, marked=None):
    side, t, border, rows, cols = geometry
    margin = t if border else 0
    shape = (rows * side + (rows - 1) * t + 2 * margin,
             cols * side + (cols - 1) * t + 2 * margin)
    grid = np.full(shape, divider, dtype=np.int8)
    for r in range(rows):
        for c in range(cols):
            y, x = margin + r * (side + t), margin + c * (side + t)
            tile = glyph if (r, c) in positions else ordinary
            if marked is not None and (r, c) == marked[0]:
                tile = marked[1]
            grid[y:y + side, x:x + side] = tile
    return grid


def _geometry(rng, preference=None):
    if preference is None:
        side = int(rng.choice((3, 4), p=(.6, .4)))
        t = int(rng.choice((1, 2), p=(.8, .2)))
        border = bool(rng.integers(2)) if (side, t) != (4, 2) else False
    else:
        side, t, border = preference
    largest = (30 - (2 * t if border else 0) + t) // (side + t)
    rows = largest if rng.random() < .5 else int(rng.integers(5, largest + 1))
    cols = largest if rng.random() < .5 else int(rng.integers(5, largest + 1))
    return side, t, border, rows, cols


def _sample_mask(rng, side, minimum=3, teaching=False, excluded=()):
    for _ in range(100):
        depth = int(rng.integers(1, min(side - 1, 2 if teaching else 3) + 1))
        upper, lower = 1, 1
        if side == 4 and rng.random() < .4:
            if rng.integers(2):
                upper = 2
            else:
                lower = 2
        cells = {(0, 0), (-1, -1), (1, -1)}
        if teaching or rng.random() < .65:
            cells.add((0, -1))
        density = float(rng.uniform(.2, .85))
        for c in range(-depth, 0):
            for r in range(-upper, lower + 1):
                if abs(r) == 2 and c == -1:
                    continue
                if rng.random() < density:
                    cells.add((r, c))
        mask = tuple(sorted(cells))
        if len(mask) >= minimum and mask not in excluded:
            return mask
    # Useful asymmetric five-cell arrow, with an axial body tile as a witness.
    fallback = tuple(sorted(((0, 0), (-1, -1), (0, -1), (1, -1), (1, -2))))
    if fallback in excluded:
        fallback = tuple(sorted(((-r, c) for r, c in fallback)))
    return fallback


def _sample_route(rng, shape, mask, initial, count, profile):
    initial_offsets = _offsets(mask, initial)
    tips = [(r, c) for r in range(shape[0]) for c in range(shape[1])
            if _fits(_footprint(initial_offsets, (r, c)), shape)]
    for _ in range(36):
        route = [tips[int(rng.integers(len(tips)))]]
        ds, footprints = [], [_footprint(initial_offsets, route[0])]
        for step in range(count):
            options = {}
            for d, (dr, dc) in enumerate(_DIRS):
                if step == 0 and d != initial:
                    continue
                if profile == "bend" and step == 1 and (d - initial) % 4 not in (1, 3):
                    continue
                if profile == "straight_bend":
                    if step == 1 and d != initial:
                        continue
                    if step == 2 and (d - initial) % 4 not in (1, 3):
                        continue
                for distance in range(1, max(shape)):
                    q = (route[-1][0] + dr * distance, route[-1][1] + dc * distance)
                    if q in route or any(q in cells for cells in footprints):
                        continue
                    cells = _footprint(_offsets(mask, d), q)
                    if not _fits(cells, shape):
                        continue
                    valid = True
                    for p, nxt in zip(route[:-1], route[1:]):
                        if p[0] == q[0] or p[1] == q[1]:
                            if (_direction(p, q) != _direction(p, nxt)
                                    or abs(q[0] - p[0]) + abs(q[1] - p[1])
                                    <= abs(nxt[0] - p[0]) + abs(nxt[1] - p[1])):
                                valid = False
                                break
                    if valid:
                        options.setdefault(d, []).append((q, cells))
            if not options:
                break
            d = int(rng.choice(sorted(options)))
            q, cells = options[d][int(rng.integers(len(options[d])))]
            route.append(q)
            ds.append(d)
            footprints.append(cells)
        if len(ds) == count:
            return route, ds
    return None


def _glyph(rng, side, palette, mask, initial):
    _, bg, _, ink, marker = palette
    offsets = _offsets(mask, initial)
    r0, r1 = min(r for r, c in offsets), max(r for r, c in offsets)
    c0, c1 = min(c for r, c in offsets), max(c for r, c in offsets)
    fine_tip = (int(rng.integers(-r0, side - r1)), int(rng.integers(-c0, side - c1)))
    glyph = np.full((side, side), bg, dtype=np.int8)
    for r, c in _footprint(offsets, fine_tip):
        glyph[r, c] = ink
    marked = glyph.copy()
    marked[fine_tip] = marker
    return glyph, marked


def _ordinary(rng, side, palette, final_glyph, forbidden=None):
    _, bg, texture, ink, _ = palette
    inactive = np.argwhere(final_glyph != ink)
    for _ in range(100):
        labels = rng.random((side, side)) < float(rng.uniform(.2, .7))
        labels[tuple(inactive[int(rng.integers(len(inactive)))])] = True
        if labels.all():
            labels.flat[int(rng.integers(side * side))] = False
        ordinary = np.where(labels, texture, bg).astype(np.int8)
        if (_normalized(ordinary) != forbidden
                and np.any(ordinary[final_glyph != ink] != bg)):
            return ordinary
    for r, c in inactive:
        ordinary = np.full((side, side), bg, dtype=np.int8)
        ordinary[r, c] = texture
        if _normalized(ordinary) != forbidden:
            return ordinary
    raise AssertionError("no ordinary-pattern witness")


def _make_pair(rng, palette, geometry_preference, split, index, previous):
    teach0 = split == "train" and index == 0
    teach1 = split == "train" and index == 1
    profile = "bend" if teach0 else "straight_bend" if teach1 else "free"
    count = 2 if teach0 else 3 if teach1 else int(rng.choice(
        (2, 3, 4, 5, 6, 7), p=(.25, .2, .2, .15, .12, .08)))
    if split == "test":
        count = int(rng.integers(4, 8))
    for attempt in range(120):
        if not (teach0 or teach1) and attempt >= 10 and attempt % 10 == 0:
            count = max(4 if split == "test" else 2, count - 1)
        geometry = _geometry(rng, geometry_preference)
        side, _, _, rows, cols = geometry
        excluded = {previous[0]["mask"]} if teach1 else set()
        mask = _sample_mask(rng, side, 5 if teach0 else 4 if teach1 else 3,
                            teach0 or teach1, excluded)
        if teach1 and rows == cols == 5:
            # Three beacons with an initial straight continuation need a shallow
            # rear on a five-cell lattice; train[0] has at least five ink cells.
            mask = tuple(sorted(((0, 0), (-1, -1), (0, -1), (1, -1))))
        initial = int(rng.integers(4))
        # Some tests reuse a demonstrated blueprint but receive a sparser body.
        if split == "test" and previous and rng.random() < .35:
            old = previous[int(rng.integers(len(previous)))]
            candidate = old["mask"]
            if (max(r for r, c in candidate) - min(r for r, c in candidate) < side
                    and -min(c for r, c in candidate) < side):
                mask = candidate
        if attempt == 119:
            # A fitting five-cell subwindow supplies a constructive fallback.
            # It follows the same route and stencil rules as every sampled case.
            mask = tuple(sorted(((0, 0), (-1, -1), (0, -1), (1, -1))
                                + (((1, -2),) if teach0 else ())))
            base = ([(1, 2), (1, 3), (3, 3)] if teach0 else
                    [(1, 1), (1, 2), (1, 3), (3, 3)] if teach1 else
                    [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3)] if split == "test" else
                    [(1, 1), (1, 3), (3, 3)])
            k = int(rng.integers(4))
            transformed = []
            for r, c in base:
                for _ in range(k):
                    r, c = c, 4 - r
                transformed.append((r, c))
            dy, dx = int(rng.integers(rows - 4)), int(rng.integers(cols - 4))
            route = [(r + dy, c + dx) for r, c in transformed]
            initial = (1 + k) % 4
            sampled = route, [_direction(a, b) for a, b in zip(route, route[1:])]
        else:
            sampled = _sample_route(rng, (rows, cols), mask, initial, count, profile)
        if sampled is None:
            # The split shift is sparse-body reconstruction, not a compulsory
            # maximal route; a four-beacon test still exceeds the first two demos.
            if attempt > 60 and split == "test":
                count = 4
            continue
        route, directions = sampled
        body_options = [p for p in mask if p != (0, 0)]
        body_count = (1 if split == "test" else
                      int(rng.integers(2 if teach0 or teach1 else 1, len(mask) - 1)))
        fragment = []
        if teach0:
            fragment = [(0, -1)]
        pool = [p for p in body_options if p not in fragment]
        chosen = rng.choice(len(pool), body_count - len(fragment), replace=False)
        fragment += [pool[int(i)] for i in chosen]
        body = _footprint(tuple(_rot(p, initial - 1) for p in fragment), route[0])
        glyph, marked = _glyph(rng, side, palette, mask, initial)
        turns = (directions[-1] - initial) % 4
        final_glyph = np.rot90(glyph, -turns).copy()
        if (teach0 or teach1) and np.array_equal(glyph, final_glyph):
            continue
        ordinary = _ordinary(rng, side, palette, final_glyph,
                             _normalized(previous[0]["ordinary"]) if teach1 else None)
        if rows * cols - len(body) - len(directions) - 1 <= len(body) + len(directions) + 1:
            continue
        inp = _paint(geometry, palette[0], ordinary, glyph, body | set(route[1:]),
                     (route[0], marked))
        terminal = _footprint(_offsets(mask, directions[-1]), route[-1])
        out = _paint(geometry, palette[0], ordinary, final_glyph, terminal)
        return {"input": inp, "output": out}
    raise AssertionError("bounded route sampler exhausted")


def _validate_counts(num_train, num_test):
    for value, name, lo, hi in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _validate_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    stable_palette = bool(rng.random() < .2)
    palette = tuple(map(int, rng.choice(10, 5, replace=False)))
    geometry_preference = _geometry(rng)[:3] if rng.random() < .25 else None
    puzzle, used, training_info = {"train": [], "test": []}, set(), []
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(30):
                if not stable_palette:
                    palette = tuple(map(int, rng.choice(10, 5, replace=False)))
                pair = _make_pair(rng, palette, geometry_preference, split, index, training_info)
                if _key(pair["input"]) not in used:
                    break
            else:
                raise AssertionError("distinct-input sampler exhausted")
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
            if split == "train":
                expected, info = _infer(pair["input"])
                assert np.array_equal(expected, pair["output"])
                training_info.append(info)
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _validate_counts(len(puzzle["train"]), len(puzzle["test"]))
    infos, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            expected, info = _infer(inp)
            assert np.array_equal(expected, out), "wrong repaired terminal recursive arrow"
            assert not np.array_equal(inp, out)
            assert info["anchor_color"] not in out
            side, t, border, rows, cols = info["geometry"]
            # Every active coarse cell holds the entire rotated miniature.
            assert int(np.count_nonzero(out == info["ink"])) == len(info["mask"]) ** 2
            assert np.any((info["ordinary"] != info["final_glyph"])
                          & (info["final_glyph"] != info["ink"])), "ink-only copying must differ"
            assert all(np.array_equal(inp[i], out[i]) for i in range(inp.shape[0])
                       if np.all(inp[i] == info["divider"]))
            assert all(np.array_equal(inp[:, i], out[:, i]) for i in range(inp.shape[1])
                       if np.all(inp[:, i] == info["divider"]))
            infos[split].append(info)
    first, second = infos["train"][:2]
    assert first["mask"] != second["mask"]
    assert _normalized(first["ordinary"]) != _normalized(second["ordinary"])
    assert first["initial"] != first["final"]
    assert all(not np.array_equal(f["glyph"], f["final_glyph"]) for f in (first, second))
    assert len(first["directions"]) == 2 and len(second["directions"]) == 3
    assert second["directions"][0] == second["directions"][1] != second["directions"][2]
    assert (0, -1) in first["fragment"], "rear body tile must disambiguate route membership"
    assert all(len(f["body"]) >= 2 for f in (first, second))
    assert all(len(f["body"]) == 1 and len(f["waypoints"]) >= 4 for f in infos["test"])
    return infos


def validate_generated_puzzle(puzzle):
    """Check bitmap-derived semantics, teaching witnesses and sparse-body test shift."""
    try:
        _validate(puzzle)
    except (AssertionError, ValueError):
        raise
    except Exception as exc:
        raise AssertionError(f"invalid puzzle: {exc}") from exc


def _route_signature(route):
    variants = []
    for sign in (-1, 1):
        for k in range(4):
            variants.append(tuple(_rot((r - route[0][0], sign * (c - route[0][1])), k)
                                  for r, c in route))
    return min(variants)


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test():
    """Bounded 42-seed sweep, counterfactual checks, count checks and negative tests."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records = [f for p in puzzles for fs in _validate(p).values() for f in fs]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len({f["mask"] for f in records}) >= 20
    assert len({_route_signature(f["route"]) for f in records}) >= 40
    assert len({f["geometry"] for f in records}) >= 12
    assert len({f["fragment"] for f in records}) >= 15
    assert len({_normalized(f["ordinary"]) for f in records}) >= 40
    conditional = {}
    for f in records:
        conditional.setdefault(f["mask"], set()).add(f["fragment"])
    assert sum(len(fragments) > 1 for fragments in conditional.values()) >= 5
    assert any(len({f["mask"] for f in records if len(f["mask"]) == area}) > 1
               for area in {len(f["mask"]) for f in records})
    assert {f["geometry"][0] for f in records} == {3, 4}
    assert {f["geometry"][1] for f in records} == {1, 2}
    assert {f["geometry"][2] for f in records} == {False, True}
    for role in ("divider", "ink", "anchor_color", "background"):
        assert {f[role] for f in records} == set(range(10))
    for split in ("train", "test"):
        fs = [f for p in puzzles for f in _validate(p)[split]]
        assert {f["initial"] for f in fs} == set(range(4))
        assert {f["final"] for f in fs} == set(range(4))
        assert any(f["initial"] == f["final"] for f in fs)
    mixed = [f for f in records if {1, 3} <= {
        (b - a) % 4 for a, b in zip(f["directions"], f["directions"][1:])}]
    assert len(mixed) >= 3
    assert any(len(set(abs(b[0] - a[0]) + abs(b[1] - a[1])
                       for a, b in zip(f["route"], f["route"][1:]))) > 1 for f in records)
    assert any(f["geometry"][3] != f["geometry"][4] for f in records)
    assert any(29 in p["input"].shape for task in puzzles for s in task for p in task[s])
    for p in puzzles:
        original = p["train"][0]["input"].copy()
        assert np.array_equal(solve(original), p["train"][0]["output"])
        assert np.array_equal(original, p["train"][0]["input"]), "solve mutated its input"
        f = _infer(p["train"][0]["input"])[1]
        # Omitting B's body/stencil classification creates a backward/forward
        # junction even though the intended waypoint route has one ray.
        try:
            _order_route(f["tip"], f["waypoints"] | f["body"])
        except AssertionError:
            pass
        else:
            raise AssertionError("body-as-waypoint ablation unexpectedly solved")
        # Omitting A's motion leaves the completed stencil at the old tip.
        no_motion = _paint(f["geometry"], f["divider"], f["ordinary"], f["glyph"],
                           f["initial_shape"])
        assert not np.array_equal(no_motion, p["train"][0]["output"])
        # Omitting microscopic rotation while rotating the macro shape differs.
        macro_only = _paint(f["geometry"], f["divider"], f["ordinary"], f["glyph"],
                            f["final_shape"])
        assert not np.array_equal(macro_only, p["train"][0]["output"])
    assert _puzzle_key(generate_puzzle(506, 7, 3)) == _puzzle_key(generate_puzzle(506, 7, 3))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(700 + a * 10 + b, a, b)
            assert (len(p["train"]), len(p["test"])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Validators must reject errors involving cargo backgrounds and marker pixels.
    try:
        _order_route((2, 2), {(2, 3), (3, 2)})
    except AssertionError:
        pass
    else:
        raise AssertionError("competing waypoint rays accepted")
    p = puzzles[0]
    f = _infer(p["test"][0]["input"])[1]
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    r, c = next(iter(sorted(f["final_shape"])))
    y, x = f["intervals"][0][r][0], f["intervals"][1][c][0]
    dy, dx = np.argwhere((f["ordinary"] != f["final_glyph"])
                        & (f["final_glyph"] != f["ink"]))[0]
    broken["test"][0]["output"][y + dy, x + dx] = f["ordinary"][dy, dx]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("ink-only copy error accepted")
    damaged = p["test"][0]["input"].copy()
    r, c = f["tip"]
    y, x = f["intervals"][0][r][0], f["intervals"][1][c][0]
    damaged[y:y + f["geometry"][0], x:x + f["geometry"][0]] = f["glyph"]
    try:
        solve(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing anchor accepted")
    return dict(seeds=42, pairs=len(records), masks=len({f["mask"] for f in records}),
                routes=len({_route_signature(f["route"]) for f in records}),
                geometries=len({f["geometry"] for f in records}),
                fragments=len({f["fragment"] for f in records}), mixed_turns=len(mixed),
                masks_with_multiple_fragments=sum(len(fs) > 1 for fs in conditional.values()))
