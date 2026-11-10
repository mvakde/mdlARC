"""Cargo on invisible pinwheel rails.

The cross supplies an ordered four-phase winding. Forks select which pieces of
that winding become swept trails. All conventions are visible or family-fixed;
solve never reads examples, a seed, files, or generation metadata.
"""
from __future__ import annotations

import copy
from collections import Counter
import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _add(point, direction, distance=1):
    return (point[0] + distance * direction[0],
            point[1] + distance * direction[1])


def _inside(point, shape):
    return 0 <= point[0] < shape[0] and 0 <= point[1] < shape[1]


def _paint(grid, cells, color):
    for p in cells:
        if _inside(p, grid.shape):
            grid[p] = color


def _halo(cells):
    return {(r + dy, c + dx) for r, c in cells
            for dy, dx in ((0, 0),) + _DIRS}


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = {start}, [start]
        while stack:
            point = stack.pop()
            for d in _DIRS:
                q = _add(point, d)
                if q in remaining:
                    remaining.remove(q)
                    cells.add(q)
                    stack.append(q)
        result.append(cells)
    return result


def _rectangle(cells):
    rows, cols = zip(*cells)
    h, w = max(rows) - min(rows) + 1, max(cols) - min(cols) + 1
    return len(cells) == h * w, h, w


def _check_grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any(grid < 0) or np.any(grid > 9)):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return grid.astype(np.int8, copy=False)


def _cross_cells(center, length):
    return {_add(center, d, k) for d in _DIRS for k in range(1, length + 1)}


def _find_cross(grid, background):
    matches = []
    for color in np.unique(grid):
        if int(color) == background:
            continue
        cells = set(map(tuple, np.argwhere(grid == color)))
        if len(cells) % 4:
            continue
        length = len(cells) // 4
        if not 1 <= length <= 6:
            continue
        rows, cols = zip(*cells)
        center = ((min(rows) + max(rows)) // 2, (min(cols) + max(cols)) // 2)
        if cells == _cross_cells(center, length) and int(grid[center]) == background:
            matches.append((int(color), length, center, cells))
    if len(matches) != 1:
        raise ValueError("expected one uniquely colored equal-spoke hollow cross")
    return matches[0]


def _rail_paths(length, shape, center):
    """Ordered infinite-plane rails, including excursions outside the canvas.

    Points are never dropped merely because they are outside. A four-stage
    cycle moves the enclosing winding outward by eight cells. This bound goes
    two full cycles beyond the canvas's largest center-relative coordinate.
    Direction and (stage, subsegment) annotations serve generation/auditing;
    solving uses the points and their order.
    """
    radius = max(2, length)
    distance = max(center[0], center[1], shape[0] - 1 - center[0],
                   shape[1] - 1 - center[1])
    stages = 4 * (distance // 8 + 2)
    paths = []
    for outward in _DIRS:
        points = [_add(center, outward, k) for k in range(1, radius + 1)]
        directions = [outward] * radius
        labels = [(-1, 0)] * radius
        point = points[-1]
        direction = (outward[1], -outward[0])
        for stage in range(stages):
            right = (direction[1], -direction[0])
            for part, (d, steps) in enumerate(((direction, 2 + 4 * stage),
                                               (right, radius - 2),
                                               (direction, radius))):
                for _ in range(steps):
                    point = _add(point, d)
                    points.append(point)
                    directions.append(d)
                    labels.append((stage, part))
            direction = right
        paths.append((points, directions, labels))
    return paths


def _fork_cells(dock, outward, gap, lengths):
    """The dock's first cell lies immediately beyond the central tip."""
    left, middle, right = lengths
    perp = (outward[1], -outward[0])
    base = _add(dock, outward, -middle - 1)
    cells = {_add(base, perp, q) for q in range(-gap, gap + 1)}
    for q, n in ((-gap, left), (0, middle), (gap, right)):
        cells |= {_add(_add(base, perp, q), outward, k) for k in range(1, n + 1)}
    return cells


def _parse_fork(cells, color):
    parses = []
    for outward in _DIRS:
        perp = (outward[1], -outward[0])
        k0 = min(r * outward[0] + c * outward[1] for r, c in cells)
        q0 = min(r * perp[0] + c * perp[1] for r, c in cells)
        local = {(r * outward[0] + c * outward[1] - k0,
                  r * perp[0] + c * perp[1] - q0) for r, c in cells}
        width = max(q for _, q in local) + 1
        if width < 5 or width % 2 == 0:
            continue
        gap = width // 2
        lengths = tuple(sum((k, q) in local for k in range(1, 31))
                        for q in (0, gap, width - 1))
        if min(lengths) < 1:
            continue
        expected = {(0, q) for q in range(width)}
        for q, n in zip((0, gap, width - 1), lengths):
            expected |= {(k, q) for k in range(1, n + 1)}
        if local != expected:
            continue
        tip_k, tip_q = k0 + lengths[1], q0 + gap
        tip = (tip_k * outward[0] + tip_q * perp[0],
               tip_k * outward[1] + tip_q * perp[1])
        parses.append(dict(cells=cells, color=color, outward=outward,
                           tip=tip, dock=_add(tip, outward), gap=gap,
                           lengths=lengths))
    if len(parses) != 1:
        raise ValueError("fork needs one full base and three positive thin prongs")
    return parses[0]


def _analyze(input_grid):
    grid = _check_grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) < 4 or counts.max() <= grid.size // 2:
        raise ValueError("background must be a unique absolute majority")
    background = int(colors[np.argmax(counts)])
    rail_color, length, center, seed = _find_cross(grid, background)
    forks, tokens = [], []
    for color in colors:
        color = int(color)
        if color in (background, rail_color):
            continue
        for cells in _components(grid == color):
            rectangular, h, w = _rectangle(cells)
            if rectangular:
                tokens.append(dict(cells=cells, color=color, h=h, w=w))
            else:
                forks.append(_parse_fork(cells, color))
    if not forks or not tokens:
        raise ValueError("expected receivers and cargo")
    fork_colors = {f["color"] for f in forks}
    token_colors = {t["color"] for t in tokens}
    if not fork_colors.isdisjoint(token_colors):
        raise ValueError("receiver and cargo palettes must be disjoint")
    paths = _rail_paths(length, grid.shape, center)
    locations = {}
    for phase, (points, _, _) in enumerate(paths):
        for index, p in enumerate(points):
            if _inside(p, grid.shape):
                if p in locations:
                    raise ValueError("rails unexpectedly intersect")
                locations[p] = phase, index
    structures = {p: fi for fi, f in enumerate(forks) for p in f["cells"]}
    occupied_tokens = {p: ti for ti, t in enumerate(tokens) for p in t["cells"]}
    matches, rail_tokens, assigned = [], [], set()
    for ti, token in enumerate(tokens):
        m = len(token["cells"])
        if min(token["h"], token["w"]) != 1 or m > 3:
            continue
        places = [locations.get(p) for p in token["cells"]]
        if any(p is None for p in places) or len({p[0] for p in places}) != 1:
            continue
        phase = places[0][0]
        indices = sorted(p[1] for p in places)
        start = indices[0]
        points, directions, labels = paths[phase]
        if indices != list(range(start, start + m)) or start < max(2, length):
            continue
        if any(directions[k] != directions[start + 1]
               for k in range(start + 1, start + m)):
            continue
        rail_tokens.append(dict(ti=ti, phase=phase, start=start, size=m))
        # The train's leading cell walks backward. Its first structural hit
        # must be a central tip facing along the forward rail, never a guard.
        hit = None
        for index in range(start - 1, max(2, length) - 2, -1):
            p = points[index]
            if p in occupied_tokens and occupied_tokens[p] != ti:
                break
            if p in seed:
                break
            if p in structures:
                fi = structures[p]
                f = forks[fi]
                dock = index + 1
                final_points = points[dock:dock + m]
                if (p == f["tip"] and len(final_points) == m
                        and final_points[0] == f["dock"]
                        and all(final_points[k] == _add(f["dock"], f["outward"], k)
                                for k in range(m))
                        and all(_inside(q, grid.shape) and q not in structures
                                and (q not in occupied_tokens or occupied_tokens[q] == ti)
                                for q in final_points)):
                    hit = dict(ti=ti, fi=fi, phase=phase, dock=dock, start=start,
                               size=m, final=set(final_points),
                               route=set(points[dock:start + m]),
                               bends=sum(directions[k] != directions[k - 1]
                                         for k in range(dock + 1, start + 1)),
                               clipped=any(not _inside(q, grid.shape)
                                           for q in points[dock:start + m]),
                               stage=labels[start][0])
                break
        if hit is not None:
            if hit["fi"] in assigned:
                raise ValueError("more than one cargo for a receiver")
            assigned.add(hit["fi"])
            matches.append(hit)
    for i, a in enumerate(matches):
        for b in matches[i + 1:]:
            if a["route"] & b["route"]:
                raise ValueError("successful routes must be disjoint")
    return dict(grid=grid, background=background, rail_color=rail_color,
                length=length, center=center, seed=seed, forks=forks,
                tokens=tokens, paths=paths, matches=matches, rail_tokens=rail_tokens)


def _answer(info):
    output = info["grid"].copy()
    for token in info["tokens"]:
        _paint(output, token["cells"], info["background"])
    for match in info["matches"]:
        _paint(output, match["route"] - match["final"], info["rail_color"])
        _paint(output, match["final"], info["tokens"][match["ti"]]["color"])
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Resolve all rail membership, movement, stopping and clipping from input."""
    return _answer(_analyze(input_grid))


def _options(length, shape, center, size, min_stage, max_stage, min_bends, geometries):
    """Finite placement search; options contain geometry, never puzzle answers."""
    paths = _rail_paths(length, shape, center)
    seed = _cross_cells(center, length)
    all_options = []
    for phase, (points, directions, labels) in enumerate(paths):
        for dock in range(length + 2, len(points) - size):
            stage, part = labels[dock]
            if stage > 2:
                break
            if stage < 1 or part != 0:
                continue
            d = directions[dock]
            # Keep the whole final rectangle straight, before the next turn.
            if any(directions[k] != d for k in range(dock - 1, dock + size + 1)):
                continue
            final = set(points[dock:dock + size])
            if not all(_inside(p, shape) for p in final):
                continue
            for gap, lengths in geometries:
                cells = _fork_cells(points[dock], d, gap, lengths)
                if cells & seed or not all(_inside(p, shape) for p in cells):
                    continue
                for start in range(dock + size + 3, len(points) - size):
                    token_stage = labels[start][0]
                    if token_stage > max_stage:
                        break
                    if token_stage < min_stage:
                        continue
                    if any(directions[k] != directions[start]
                           for k in range(start, start + size + 1)):
                        continue
                    original = set(points[start:start + size])
                    if not all(_inside(p, shape) for p in original):
                        continue
                    segment = points[dock:start + size]
                    route = set(segment)
                    bends = sum(directions[k] != directions[k - 1]
                                for k in range(dock + 1, start + 1))
                    if bends < min_bends or route & cells:
                        continue
                    all_options.append(dict(phase=phase, gate=cells,
                                            halo=_halo(cells),
                                            original=original, final=final,
                                            route=route, dock=dock, start=start,
                                            gap=gap, lengths=lengths, bends=bends,
                                            clipped=any(not _inside(p, shape) for p in segment)))
    return all_options


def _compatible(a, b):
    return (a["phase"] != b["phase"]
            and not (a["halo"] & b["gate"]
                     or a["gate"] & b["route"] or b["gate"] & a["route"]
                     or a["route"] & b["route"]))


def _select(rng, options, count, require_clipped=False):
    if not options:
        return None
    for _ in range(60):
        chosen = []
        # Repeated cargo positions can give thousands of options for the
        # same fork. A fresh bounded subset avoids quadratic search cost.
        available = ([options[int(i)] for i in rng.choice(len(options), 384, replace=False)]
                     if len(options) > 384 else options)
        for index in range(count):
            if not available:
                break
            choices = ([o for o in options if o["clipped"]]
                       if index == 0 and require_clipped else available)
            if not choices:
                break
            option = choices[int(rng.integers(len(choices)))]
            chosen.append(option)
            available = [b for b in available if _compatible(option, b)]
        if len(chosen) == count:
            return chosen
    return None


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _make_pair(rng, length, active, inactive, token_size, journey, palette, seen):
    background, rail_color, fork_color, *cargo_colors = palette
    teaching = journey in ("teaching", "clipped_teaching")
    require_clipped = journey == "clipped_teaching"
    min_stage, max_stage = (3, 3) if teaching else (4, 5)
    for attempt in range(18):
        # Sample positive lengths independently, rather than choose a fixed
        # collection of receiver silhouettes. Include one standard and one
        # long-center geometry to make the packing search reliable.
        side = int(rng.integers(2, 5))
        middle = int(rng.integers(3, 5))
        geometries = {(int(rng.integers(2, 5)), (side, 1, side)),
                      (int(rng.integers(2, 5)),
                       (int(rng.integers(1, middle)), middle,
                        int(rng.integers(1, middle))))}
        while len(geometries) < 10:
            geometries.add((int(rng.integers(2, 5)),
                            tuple(map(int, rng.integers(1, 5, size=3)))))
        geometries = sorted(geometries)
        if attempt == 17:
            shape, center = (30, 30), (14, 14)
        else:
            low = 25 if teaching else 22
            shape = tuple(map(int, rng.integers(low, 31, size=2)))
            center = tuple((n - 1) // 2 + int(rng.integers(-3, 4)) for n in shape)
        if require_clipped:
            # Stage three has an outer coordinate -(R+6), then jogs back
            # to -8. A margin from 8 through R+5 exposes exit and re-entry.
            # The cross, receiving dock and original cargo remain visible.
            axis = int(rng.integers(2))
            margin = int(rng.integers(8, length + 6))
            center = list(center)
            center[axis] = margin if rng.integers(2) else shape[axis] - 1 - margin
            center = tuple(center)
        options = []
        for size in ((1, 2, 3) if token_size == 0 else (token_size,)):
            options.extend(_options(length, shape, center, size, min_stage,
                                    max_stage, 2, geometries))
        chosen = _select(rng, options, active, require_clipped)
        if chosen is None:
            continue
        grid = np.full(shape, background, dtype=np.int8)
        output = grid.copy()
        seed = _cross_cells(center, length)
        _paint(grid, seed, rail_color)
        _paint(output, seed, rail_color)
        for index, option in enumerate(chosen):
            color = cargo_colors[index % len(cargo_colors)]
            _paint(grid, option["gate"], fork_color)
            _paint(output, option["gate"], fork_color)
            _paint(grid, option["original"], color)
            _paint(output, option["route"] - option["final"], rail_color)
            _paint(output, option["final"], color)
        # An unused receiver shares the hub but must not obstruct successful
        # movement. It is not a separate puzzle region or a forced corner mark.
        if inactive:
            choices = [o for o in options if all(_compatible(o, p) for p in chosen)]
            if not choices:
                continue
            extra = choices[int(rng.integers(len(choices)))]
            _paint(grid, extra["gate"], fork_color)
            _paint(output, extra["gate"], fork_color)
        try:
            info = _analyze(grid)
        except ValueError:
            continue
        if len(info["matches"]) != active:
            continue
        routes = set().union(*(o["route"] for o in chosen))
        protected = _halo(set(map(tuple, np.argwhere(grid != background)))) | routes
        rails = {p for points, _, _ in info["paths"] for p in points if _inside(p, shape)}
        # First distractor is thin and on a rail but has no accepting receiver.
        # This is the explicit counterexample to 'all on-rail cargo moves'.
        candidates = sorted(rails - protected)
        rng.shuffle(candidates)
        placed = False
        for p in candidates:
            color = cargo_colors[-1]
            trial = grid.copy()
            trial[p] = color
            try:
                trial_info = _analyze(trial)
            except ValueError:
                continue
            if (len(trial_info["matches"]) == active
                    and any(p in trial_info["tokens"][t["ti"]]["cells"]
                            for t in trial_info["rail_tokens"])):
                grid = trial
                protected |= _halo({p})
                placed = True
                break
        if not placed:
            continue
        # A 2x2 square intersects a rail but cannot be a one-cell-wide train.
        positions = []
        for r in range(shape[0] - 1):
            for c in range(shape[1] - 1):
                cells = {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}
                if not cells & protected and cells & rails:
                    positions.append(cells)
        if not positions:
            continue
        cells = positions[int(rng.integers(len(positions)))]
        _paint(grid, cells, cargo_colors[-1])
        protected |= _halo(cells)
        # Off-rail distractors vary in number, orientation and length.
        for _ in range(int(rng.integers(1, 4))):
            positions = sorted({(r, c) for r in range(shape[0]) for c in range(shape[1])}
                               - protected - rails)
            if not positions:
                break
            p = positions[int(rng.integers(len(positions)))]
            cells = {p}
            d = _DIRS[int(rng.integers(4))]
            for k in range(1, int(rng.integers(1, 4))):
                q = _add(p, d, k)
                if _inside(q, shape) and q not in protected and q not in rails:
                    cells.add(q)
                else:
                    break
            _paint(grid, cells, cargo_colors[int(rng.integers(len(cargo_colors)))])
            protected |= _halo(cells)
        try:
            final_info = _analyze(grid)
        except ValueError:
            continue
        if (len(final_info["matches"]) != active or _key(grid) in seen
                or not np.array_equal(_answer(final_info), output)):
            continue
        seen.add(_key(grid))
        return {"input": grid, "output": output}
    raise RuntimeError("finite geometric packing search exhausted")


def _legal_count(value, lo, hi):
    return (isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_)) and lo <= int(value) <= hi)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (not _legal_count(num_train, 2, 10) or not _legal_count(num_test, 1, 4)
            or num_train <= num_test):
        raise ValueError("need 2..10 train, 1..4 test, and num_train > num_test")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    # Palette identities stay consistent within one puzzle. The cross visibly
    # supplies the trail color, so this is not a latent convention of solve.
    colors = list(map(int, rng.permutation(10)))
    palette = colors[:int(rng.integers(5, 7))]
    regime = str(rng.choice(("more_cargo", "longer_journey", "next_radius")))
    base_length = int(rng.choice((2, 3, 4), p=(.5, .3, .2)))
    puzzle, seen = {"train": [], "test": []}, set()
    for i in range(int(num_train)):
        length = base_length if i == 0 else base_length + 1 if i == 1 else int(
            rng.integers(base_length, base_length + 2))
        active = 1 if i == 0 else 2 if i == 1 else int(rng.integers(1, 3))
        # Radius two's tightly interleaved rails support opposite receivers,
        # not three simultaneously; do not force impossible layouts.
        inactive = i == 0 or (length >= 3 and active == 1 and bool(rng.integers(2)))
        token_size = 1 if i == 0 else 2 if i == 1 else int(rng.integers(0, 4))
        journey = "clipped_teaching" if i == 1 else "teaching"
        puzzle["train"].append(_make_pair(rng, length, active, inactive,
                                         token_size, journey, palette, seen))
    for i in range(int(num_test)):
        length = base_length + 2 if regime == "next_radius" else base_length + 1
        active = 3 if regime == "more_cargo" else 2
        journey = "long" if regime == "longer_journey" else "teaching"
        token_size = int(rng.integers(0, 4))
        puzzle["test"].append(_make_pair(rng, length, active, False,
                                        token_size, journey, palette, seen))
    return puzzle


def _without_receivers(info):
    """Counterfactual: every thin rail token returns to just beyond its seed."""
    output = info["grid"].copy()
    for t in info["tokens"]:
        _paint(output, t["cells"], info["background"])
    structures = set().union(*(f["cells"] for f in info["forks"]))
    for rt in info["rail_tokens"]:
        points = info["paths"][rt["phase"]][0]
        dock = max(2, info["length"])
        final = set(points[dock:dock + rt["size"]])
        route = set(points[dock:rt["start"] + rt["size"]]) - structures - info["seed"]
        _paint(output, route - final, info["rail_color"])
        _paint(output, final - structures, info["tokens"][rt["ti"]]["color"])
    return output


def _without_winding(info):
    """Counterfactual: A's original straight, aligned insertion only."""
    output = info["grid"].copy()
    for t in info["tokens"]:
        _paint(output, t["cells"], info["background"])
    for f in info["forks"]:
        d = f["outward"]
        perp = (d[1], -d[0])
        candidates = []
        for t in info["tokens"]:
            coords = [((p[0] - f["dock"][0]) * d[0] + (p[1] - f["dock"][1]) * d[1],
                       (p[0] - f["dock"][0]) * perp[0] + (p[1] - f["dock"][1]) * perp[1])
                      for p in t["cells"]]
            if any(q != 0 for _, q in coords) or min(k for k, _ in coords) <= 0:
                continue
            ks = sorted(k for k, _ in coords)
            if ks != list(range(ks[0], ks[-1] + 1)):
                continue
            route = {_add(f["dock"], d, k) for k in range(ks[-1] + 1)}
            if any(int(info["grid"][p]) != info["background"] and p not in t["cells"]
                   for p in route):
                continue
            final = {_add(f["dock"], d, k) for k in range(len(t["cells"]))}
            candidates.append((t, route, final))
        if len(candidates) == 1:
            token, route, final = candidates[0]
            _paint(output, route - final, info["rail_color"])
            _paint(output, final, token["color"])
    return output


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert _legal_count(len(puzzle["train"]), 2, 10)
    assert _legal_count(len(puzzle["test"]), 1, 4)
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, palettes = set(), set()
    evidence = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _check_grid(pair["input"]), _check_grid(pair["output"])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            info = _analyze(a)
            assert np.array_equal(_answer(info), b)
            assert not np.array_equal(a, b)
            palettes.add((info["background"], info["rail_color"],
                          tuple(sorted({f["color"] for f in info["forks"]}))))
            assert b[info["center"]] == info["background"]
            assert all(b[p] == info["rail_color"] for p in info["seed"])
            assert all(b[p] == f["color"] for f in info["forks"] for p in f["cells"])
            assert len(info["matches"]) >= 1
            assert all(m["bends"] >= 2 and m["start"] - m["dock"] > m["size"]
                       for m in info["matches"])
            used = {m["ti"] for m in info["matches"]}
            assert len(info["tokens"]) - len(used) >= 3
            assert any(rt["ti"] not in used for rt in info["rail_tokens"])
            assert any(t["h"] == t["w"] == 2 for t in info["tokens"])
            for m in info["matches"]:
                t = info["tokens"][m["ti"]]
                assert len(m["final"]) == len(t["cells"])
                assert all(b[p] == t["color"] for p in m["final"])
                assert all(b[p] == info["rail_color"] for p in m["route"] - m["final"]
                           if _inside(p, b.shape))
            # All off-route cells are unchanged except erased distractors.
            affected = set().union(*(t["cells"] for t in info["tokens"]))
            affected |= set().union(*(m["route"] for m in info["matches"]))
            assert all(a[r, c] == b[r, c] for r in range(a.shape[0])
                       for c in range(a.shape[1]) if (r, c) not in affected)
            info["no_A_diff"] = not np.array_equal(_without_receivers(info), b)
            info["no_B_diff"] = not np.array_equal(_without_winding(info), b)
            evidence[split].append(info)
    assert len(palettes) == 1
    training = evidence["train"]
    train_lengths = {i["length"] for i in training}
    assert len(train_lengths) == 2 and max(train_lengths) == min(train_lengths) + 1
    assert {1, 2} <= {m["size"] for i in training for m in i["matches"]}
    assert any(len(i["forks"]) > len(i["matches"]) for i in training)
    assert any(m["clipped"] for i in training for m in i["matches"])
    assert all(i["no_A_diff"] and i["no_B_diff"] for i in training)
    # Each puzzle exhibits at least one of the documented structural shifts.
    tests = evidence["test"]
    assert (any(i["length"] > max(t["length"] for t in training) for i in tests)
            or any(len(i["matches"]) > max(len(t["matches"]) for t in training) for i in tests)
            or any(m["stage"] > max(n["stage"] for t in training for n in t["matches"])
                   for i in tests for m in i["matches"]))


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test():
    """Bounded semantic checks and actual-grid variation over 42 fixed seeds."""
    stats = Counter()
    shapes, centers, lengths, gaps, geometries, phases, sizes, palettes = (set() for _ in range(8))
    layouts, signatures = set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                shapes.add(info["grid"].shape)
                centers.add((info["grid"].shape, info["center"]))
                lengths.add(info["length"])
                palettes.add((info["background"], info["rail_color"]))
                layouts.add((info["length"], info["grid"].shape, info["center"],
                             tuple(sorted(tuple(sorted(f["cells"])) for f in info["forks"]))))
                stats["pairs"] += 1
                stats["clipped_routes"] += sum(m["clipped"] for m in info["matches"])
                stats["three_cargo_pairs"] += len(info["matches"]) == 3
                for f in info["forks"]:
                    gaps.add(f["gap"])
                    geometries.add((f["gap"], f["lengths"]))
                    stats["long_centers"] += f["lengths"][1] > max(f["lengths"][0], f["lengths"][2])
                    stats["unequal_guards"] += f["lengths"][0] != f["lengths"][2]
                for m in info["matches"]:
                    phases.add(m["phase"])
                    sizes.add(m["size"])
    assert len(signatures) == 42 and len(layouts) >= 180
    assert len(shapes) >= 25 and len(centers) >= 100 and len(palettes) >= 25
    assert lengths == {2, 3, 4, 5, 6} and phases == set(range(4)) and sizes == {1, 2, 3}
    assert gaps == {2, 3, 4} and len(geometries) >= 80
    assert stats["three_cargo_pairs"] and stats["clipped_routes"]
    assert stats["long_centers"] and stats["unequal_guards"]
    # Canvas cropping is not a stopping event, and later re-entry survives.
    for length in range(1, 7):
        small = _rail_paths(length, (30, 27), (length, 26 - length))
        big = _rail_paths(length, (90, 90), (45, 45))
        offset = (length - 45, 26 - length - 45)
        for a, b in zip(small, big):
            expected = {_add(p, offset) for p in b[0]
                        if _inside(_add(p, offset), (30, 27))}
            assert {p for p in a[0] if _inside(p, (30, 27))} == expected
    assert _puzzle_key(generate_puzzle(200, 2, 1)) == _puzzle_key(generate_puzzle(200, 2, 1))
    for a, b in ((2, 1), (5, 4), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(999 + a + b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # The parser's answer respects rotations, including rectangles at bends.
    puzzle = generate_puzzle(27, 2, 1)
    for pair in puzzle["train"] + puzzle["test"]:
        for k in (1, 2, 3):
            assert np.array_equal(solve(np.rot90(pair["input"], k)),
                                  np.rot90(pair["output"], k))
        permutation = np.array([8, 3, 1, 7, 0, 9, 5, 2, 6, 4], dtype=np.int8)
        assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    # Input-only counterfactuals are semantic corruption checks, not a second
    # independent oracle. Validator must reject these well-shaped wrong labels.
    for mode in ("pixel", "no_A", "no_B"):
        bad = copy.deepcopy(puzzle)
        pair = bad["test"][0]
        info = _analyze(pair["input"])
        if mode == "pixel":
            p = next(iter(info["matches"][0]["final"]))
            pair["output"][p] = info["rail_color"]
        elif mode == "no_A":
            pair["output"] = _without_receivers(info)
        else:
            pair["output"] = _without_winding(info)
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("semantic corruption accepted")
    return dict(stats, distinct_puzzles=len(signatures), grid_shapes=len(shapes),
                geometric_layouts=len(layouts), ordered_palettes=len(palettes))


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
