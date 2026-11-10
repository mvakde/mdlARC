"""Endpoint rays pick up colors from reconstructed, occluded rectangular loops.

The input reader, scene painter, and last-crossing check are separate. No parent
module, saved example, seed, or generation metadata is consulted by solve().
"""
from __future__ import annotations

import numpy as np


DIRECTIONS = ((-1, -1), (-1, 1), (1, 1), (1, -1))


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _strict(p, box):
    t, b, l, r = box
    return t < p[0] < b and l < p[1] < r


def _area(box):
    t, b, l, r = box
    return {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)}


def _walk(box):
    """Clockwise perimeter, starting at top-left; each corner counted once."""
    t, b, l, r = box
    return ([(t, x) for x in range(l, r + 1)]
            + [(y, r) for y in range(t + 1, b + 1)]
            + [(b, x) for x in range(r - 1, l - 1, -1)]
            + [(y, l) for y in range(b - 1, t, -1)])


def _bbox(points):
    ys, xs = zip(*points)
    return min(ys), max(ys), min(xs), max(xs)


def _halo(points):
    return {q for y, x in points for q in
            ((y, x), (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))}


def _components(points):
    remaining, result = set(points), []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        group, stack = {p}, [p]
        while stack:
            y, x = stack.pop()
            for q in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        result.append(group)
    return result


def _diamond(center, radius):
    y, x = center
    return {(y + a, x + b) for a in range(-radius, radius + 1)
            for b in range(-radius, radius + 1) if abs(a) + abs(b) == radius}


def _cap(center, radius, direction):
    y, x = center
    dy, dx = direction
    return [(y + dy * (k + 1), x + dx * (radius - k + 1))
            for k in range(radius + 1)]


def _line(start, direction, shape):
    dy, dx = direction
    y, x = start[0] + dy, start[1] + dx
    points = []
    while _inside((y, x), shape):
        points.append((y, x))
        y, x = y + dy, x + dx
    return points


def _rays(emitters, shape, lanes="ends"):
    rays = []
    for emitter in emitters:
        for direction, color in sorted(emitter["caps"].items()):
            cap = _cap(emitter["center"], emitter["radius"], direction)
            starts = (cap[0], cap[-1]) if lanes == "ends" else cap[1:-1]
            for start in starts:
                rays.append({"path": _line(start, direction, shape), "color": color,
                             "direction": direction, "radius": emitter["radius"]})
    return rays


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected an integer ARC grid with sizes 1..30 and colors 0..9")
    return grid.astype(np.int8, copy=False)


def _background(shape, colors):
    return np.asarray(colors, dtype=np.int8)[np.indices(shape).sum(axis=0) % 2]


def _frames(grid, background):
    frames = []
    h, w = grid.shape
    for color in np.unique(grid):
        color = int(color)
        if color in background:
            continue
        for group in _components(map(tuple, np.argwhere(grid == color))):
            t, b, l, r = _bbox(group)
            if min(b - t + 1, r - l + 1) < 2 or len(group) != (b - t + 1) * (r - l + 1):
                continue
            if t <= 0 or l <= 0 or b >= h - 1 or r >= w - 1:
                continue
            box = (t - 1, b + 1, l - 1, r + 1)
            border = {int(grid[p]) for p in _walk(box)}
            if len(border) == 1 and not border.intersection((*background, color)):
                frames.append({"box": box, "fill": color, "frame": border.pop(),
                               "size": len(group)})
    frames.sort(key=lambda f: (-f["size"], f["box"]))
    assert frames and (len(frames) == 1 or frames[0]["size"] > frames[1]["size"]), "largest screen tie"
    assert len(frames) <= 3
    return frames


def _periodic(grid, box, visible):
    walk = _walk(box)
    predictions, periods = set(), []
    for period in range(1, 5):
        word, counts = {}, [0] * period
        for i, p in enumerate(walk):
            if p not in visible:
                continue
            color, residue = int(grid[p]), i % period
            if residue in word and word[residue] != color:
                break
            word[residue] = color
            counts[residue] += 1
        else:
            if min(counts) >= 2:
                predictions.add(tuple(word[i % period] for i in range(len(walk))))
                periods.append(period)
    if len(predictions) != 1:
        return None
    return dict(zip(walk, next(iter(predictions)))), min(periods)


def _read_input(value):
    """Geometric parsing and uniquely witnessed completion, using input alone."""
    grid = _grid(value)
    h, w = grid.shape
    assert min(h, w) >= 9
    background = (int(grid[0, 0]), int(grid[0, 1]))
    assert background[0] != background[1]
    base = _background(grid.shape, background)
    assert (np.array_equal(grid[0], base[0]) and np.array_equal(grid[-1], base[-1])
            and np.array_equal(grid[:, 0], base[:, 0]) and np.array_equal(grid[:, -1], base[:, -1]))
    frames = _frames(grid, background)
    masks = set()
    for frame in frames:
        cells = _area(frame["box"])
        assert not masks.intersection(cells)
        masks.update(cells)
    foreground = set(map(tuple, np.argwhere(grid != base))) - masks
    emitters = []
    for radius in range(2, 5):
        sh = (h - 2 * radius, w - 2 * radius)
        offsets = sorted(_diamond((0, 0), radius))
        a, b = offsets[0]
        colors = grid[radius + a:radius + a + sh[0], radius + b:radius + b + sh[1]]
        match = (colors != background[0]) & (colors != background[1])
        for a, b in offsets[1:]:
            match &= grid[radius + a:radius + a + sh[0], radius + b:radius + b + sh[1]] == colors
        for a, b in np.argwhere(match):
            center = (int(a) + radius, int(b) + radius)
            cells = _diamond(center, radius)
            if cells <= foreground:
                emitters.append({"center": center, "radius": radius,
                                 "struct": int(colors[a, b]), "caps": {}})
    assert 1 <= len(emitters) <= 3
    objects = set()
    for emitter in emitters:
        cells = _diamond(emitter["center"], emitter["radius"])
        assert not cells.intersection(objects)
        objects.update(cells)
    for emitter in emitters:
        for direction in DIRECTIONS:
            cap = _cap(emitter["center"], emitter["radius"], direction)
            if not all(_inside(p, grid.shape) for p in cap):
                continue
            if not set(cap) <= foreground:
                continue
            colors = {int(grid[p]) for p in cap}
            if len(colors) == 1 and not colors.intersection((*background, emitter["struct"])):
                assert not objects.intersection(cap), "cap ownership conflict"
                emitter["caps"][direction] = colors.pop()
                objects.update(cap)
        assert emitter["caps"]
    emitters.sort(key=lambda e: e["center"])
    parts = _components(foreground - objects)
    candidates = []
    main_mask = _area(frames[0]["box"])
    for i, first in enumerate(parts):
        for j in range(i, len(parts)):
            visible = first | parts[j]
            box = _bbox(visible)
            t, b, l, r = box
            if min(b - t, r - l) < 3:
                continue
            full = set(_walk(box))
            if full - main_mask != visible or full.intersection(objects):
                continue
            completed = _periodic(grid, box, visible)
            if completed is None:
                continue
            colors, period = completed
            candidates.append({"parts": frozenset((i, j)), "box": box,
                               "colors": colors, "period": period, "visible": visible})
    covers = []

    def search(remaining, chosen, occupied):
        if len(covers) > 1:
            return
        if not remaining:
            covers.append(chosen)
            return
        for c in candidates:
            full = set(c["colors"])
            if min(remaining) in c["parts"] and c["parts"] <= remaining and not full.intersection(_halo(occupied)):
                search(remaining - c["parts"], chosen + [c], occupied | full)

    search(set(range(len(parts))), [], set())
    assert len(covers) == 1 and 1 <= len(covers[0]) <= 3, "ambiguous loop geometry or color phase"
    loops = sorted(covers[0], key=lambda c: c["box"])
    dye = {p: color for loop in loops for p, color in loop["colors"].items()}
    gate_colors = set(dye.values())
    screens = {f[k] for f in frames for k in ("fill", "frame")}
    structs = {e["struct"] for e in emitters}
    payloads = {c for e in emitters for c in e["caps"].values()}
    assert not structs.intersection(set(background) | screens | payloads)
    assert not payloads.intersection(set(background) | screens)
    role_colors = set(background) | screens | structs | payloads
    assert not gate_colors.intersection(role_colors)
    assert all(not set(loop["colors"]).intersection(_area(f["box"])) for loop in loops for f in frames[1:])
    rays, occupied = _rays(emitters, grid.shape), set()
    other_masks = masks - main_mask
    for ray in rays:
        path = set(ray["path"])
        assert not path.intersection(occupied | objects | other_masks), "ray collision"
        occupied.update(path)
    return {"grid": grid, "background": background, "frames": frames, "main": frames[0],
            "emitters": emitters, "objects": objects, "loops": loops, "dye": dye, "rays": rays}


def _trace_output(scene, *, use_dye=True, lanes="ends", solid=False, first_only=False):
    t, b, l, r = scene["main"]["box"]
    output = np.full((b - t - 1, r - l - 1), scene["main"]["fill"], dtype=np.int8)
    dye = scene["dye"] if use_dye else {}
    if solid:
        dye = {p: int(scene["grid"][min(loop["visible"])])
               for loop in scene["loops"] for p in loop["colors"]}
    rays = scene["rays"] if lanes == "ends" else _rays(scene["emitters"], scene["grid"].shape, lanes)
    for ray in rays:
        color, picked = ray["color"], False
        for p in ray["path"]:
            if p in dye and (not first_only or not picked):
                color, picked = dye[p], True
            if _strict(p, (t, b, l, r)):
                output[p[0] - t - 1, p[1] - l - 1] = color
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the largest screen's interior, containing only dyed endpoint rays."""
    return _trace_output(_read_input(input_grid))


def _last_crossing_output(scene):
    """Check pixels by closed-form lane membership and latest upstream crossing.

    This shares parsing with solve, but never walks or recolors a ray in order.
    """
    t, b, l, r = scene["main"]["box"]
    output = np.full((b - t - 1, r - l - 1), scene["main"]["fill"], dtype=np.int8)
    for emitter in scene["emitters"]:
        cy, cx = emitter["center"]
        radius = emitter["radius"]
        for (dy, dx), initial in emitter["caps"].items():
            for sign in (-1, 1):
                events = []
                for (y, x), color in scene["dye"].items():
                    u, v = (y - cy) * dy, (x - cx) * dx
                    if u - v == sign * radius and min(u, v) >= 2:
                        events.append((min(u, v), color))
                for y in range(t + 1, b):
                    for x in range(l + 1, r):
                        u, v = (y - cy) * dy, (x - cx) * dx
                        if u - v != sign * radius or min(u, v) < 2:
                            continue
                        prior = [(k, c) for k, c in events if k <= min(u, v)]
                        color = max(prior)[1] if prior else initial
                        output[y - t - 1, x - l - 1] = color
    return output


def _sample_box(rng, shape, main):
    h, w = shape
    t, b, l, r = main
    kind = int(rng.integers(4))
    if kind == 0 and t > 2 and b < h - 3 and r - l >= 5:
        ys = (int(rng.integers(1, t)), int(rng.integers(b + 1, h - 1)))
        xs = sorted(map(int, rng.choice(np.arange(l + 1, r), 2, replace=False)))
    elif kind == 1 and l > 2 and r < w - 3 and b - t >= 5:
        xs = (int(rng.integers(1, l)), int(rng.integers(r + 1, w - 1)))
        ys = sorted(map(int, rng.choice(np.arange(t + 1, b), 2, replace=False)))
    else:
        ys = sorted(map(int, rng.choice(np.arange(1, h - 1), 2, replace=False)))
        xs = sorted(map(int, rng.choice(np.arange(1, w - 1), 2, replace=False)))
    return ys[0], ys[1], xs[0], xs[1]


def _place_emitters(rng, shape, main, radii, directions, struct, caps_bank, minimum_run=3):
    mask = _area(main)
    banks = {}
    for radius, direction in sorted(set(zip(radii, directions))):
        options = []
        for y in range(radius + 2, shape[0] - radius - 2):
            for x in range(radius + 2, shape[1] - radius - 2):
                center = (y, x)
                cap = _cap(center, radius, direction)
                cells = _diamond(center, radius) | set(cap)
                if cells.intersection(mask):
                    continue
                primary = [_line(p, direction, shape) for p in (cap[0], cap[-1])]
                if min(sum(_strict(p, main) for p in path) for path in primary) < minimum_run:
                    continue
                paths = {p for path in primary for p in path}
                options.append((center, cells, paths))
        rng.shuffle(options)
        banks[(radius, direction)] = options
    budget = [2200]

    def search(index, chosen, objects, paths):
        if index == len(radii):
            return chosen, objects, paths
        radius, direction = radii[index], directions[index]
        for center, cells, new_paths in banks[(radius, direction)]:
            budget[0] -= 1
            if budget[0] < 0:
                return None
            if any(sum(abs(a - b) for a, b in zip(center, old["center"]))
                   < radius + old["radius"] + 3 for old in chosen):
                continue
            if cells.intersection(objects | paths) or new_paths.intersection(objects | paths):
                continue
            emitter = {"center": center, "radius": radius, "struct": struct,
                       "caps": {direction: int(rng.choice(caps_bank))}}
            result = search(index + 1, chosen + [emitter], objects | cells, paths | new_paths)
            if result is not None:
                return result
        return None

    result = search(0, [], set(), set())
    if result is None:
        return None
    emitters, objects, paths = result
    # Add extra caps only after the productive sources have all been placed.
    # Backtracking above avoids a greedy first source blocking a valid scene.
    for emitter in emitters:
        for extra in rng.permutation(4):
            d = DIRECTIONS[int(extra)]
            if d in emitter["caps"] or rng.random() >= .40:
                continue
            cap = _cap(emitter["center"], emitter["radius"], d)
            if not all(_inside(p, shape) for p in cap):
                continue
            new_paths = {p for endpoint in (cap[0], cap[-1]) for p in _line(endpoint, d, shape)}
            if (set(cap).intersection(mask | objects | paths)
                    or new_paths.intersection(objects | paths)
                    or any(_strict(p, main) for p in new_paths)):
                continue
            emitter["caps"][d] = int(rng.choice(caps_bank))
            objects.update(cap)
            paths.update(new_paths)
    return emitters, objects, _rays(emitters, shape)


def _sample_loops(rng, shape, main, objects, rays, count, patterned, gate_bank, max_period, teaching=False):
    mask = _area(main)
    loops, occupied = [], set()
    useful = [ray for ray in rays if any(_strict(p, main) for p in ray["path"])]
    for index in range(count):
        for _ in range(850):
            box = _sample_box(rng, shape, main)
            t, b, l, r = box
            if min(b - t, r - l) < 3:
                continue
            walk = _walk(box)
            full = set(walk)
            visible = full - mask
            if not visible or _bbox(visible) != box or len(visible) < 10:
                continue
            if full.intersection(objects | _halo(occupied)):
                continue
            # Relevant pickups happen under the screen, not on its frame or
            # before entry. This makes the hidden pattern causally necessary.
            hits, valid = [], True
            for ray in useful:
                inside_indices = [k for k, p in enumerate(ray["path"]) if _strict(p, main)]
                relevant = [(k, p) for k, p in enumerate(ray["path"]) if p in full and k <= inside_indices[-1]]
                if any(not _strict(p, main) for _, p in relevant):
                    valid = False
                    break
                hits.extend(relevant)
            if not valid or not hits:
                continue
            if teaching:
                combined = occupied | full
                good = True
                for ray in useful:
                    inside = [p for p in ray["path"] if _strict(p, main)]
                    encounter = [k for k, p in enumerate(inside) if p in combined]
                    if not encounter or encounter[0] < 2 or len(inside) - encounter[0] < 3:
                        good = False
                        break
                if not good:
                    continue
                if index == 1 and not any(set(ray["path"]).intersection(full)
                                          and set(ray["path"]).intersection(occupied) for ray in useful):
                    continue
            period = max_period if patterned and index == 0 else int(rng.choice([1] + list(range(1, max_period + 1))))
            if not patterned:
                period = 1
            if period == 1:
                word = [int(rng.choice(gate_bank))]
            else:
                bank = list(map(int, rng.permutation(gate_bank)))
                ncolors = int(rng.integers(2, min(period, len(bank)) + 1))
                word = bank[:ncolors] + [int(rng.choice(bank[:ncolors])) for _ in range(period - ncolors)]
                rng.shuffle(word)
            colors = {p: word[k % period] for k, p in enumerate(walk)}
            # The visible fragment must independently support its exact phase.
            fits = []
            for p in range(1, 5):
                residues = {}
                counts = [0] * p
                for k, cell in enumerate(walk):
                    if cell not in visible:
                        continue
                    j = k % p
                    if j in residues and residues[j] != colors[cell]:
                        break
                    residues[j] = colors[cell]
                    counts[j] += 1
                else:
                    if min(counts) >= 2:
                        fits.append(tuple(residues[k % p] for k in range(len(walk))))
            if len(set(fits)) != 1:
                continue
            loops.append({"box": box, "colors": colors, "visible": visible})
            occupied.update(full)
            break
        else:
            return None
    return loops


def _paint_pair(shape, background, main, emitters, loops, extras, fill, frame):
    grid = _background(shape, background)
    for loop in loops:
        for p, color in loop["colors"].items():
            grid[p] = color
    for emitter in emitters:
        for p in _diamond(emitter["center"], emitter["radius"]):
            grid[p] = emitter["struct"]
        for d, color in emitter["caps"].items():
            for p in _cap(emitter["center"], emitter["radius"], d):
                grid[p] = color
    # Ground truth comes from the sampled full loops, before input occlusion.
    t, b, l, r = main
    output = np.full((b - t - 1, r - l - 1), fill, dtype=np.int8)
    dye = {p: c for loop in loops for p, c in loop["colors"].items()}
    for ray in _rays(emitters, shape):
        color = ray["color"]
        for p in ray["path"]:
            color = dye.get(p, color)
            if _strict(p, main):
                output[p[0] - t - 1, p[1] - l - 1] = color
    for box in [main] + extras:
        y0, y1, x0, x1 = box
        grid[y0 + 1:y1, x0 + 1:x1] = fill
        for p in _walk(box):
            grid[p] = frame
    return {"input": grid, "output": output}


def _features(scene):
    main = scene["main"]["box"]
    events, witnesses, productive_directions, interior_cells = [], {}, set(), set()
    full_crossing, persistence = False, 0
    loop_at = {p: i for i, loop in enumerate(scene["loops"]) for p in loop["colors"]}
    for ray in scene["rays"]:
        inside = [p for p in ray["path"] if _strict(p, main)]
        interior_cells.update(inside)
        if not inside:
            continue
        productive_directions.add(ray["direction"])
        hits = [(k, p, loop_at[p], scene["dye"][p]) for k, p in enumerate(inside) if p in scene["dye"]]
        events.append(hits)
        persisted = sum(p not in scene["dye"] for p in inside[hits[0][0] + 1:]) if hits else 0
        persistence += persisted
        for i, loop in enumerate(scene["loops"]):
            indices = [event[0] for event in hits if event[2] == i]
            if (len(indices) >= 2 and len(inside) - indices[-1] >= 3
                    and any(_strict(p, loop["box"]) for p in inside[indices[0] + 1:indices[-1]])):
                full_crossing = True
        if hits and hits[0][0] >= 2 and len(inside) - hits[0][0] >= 3 and persisted >= 2:
            key = (ray["direction"], ray["radius"])
            witnesses[key] = witnesses.get(key, 0) + 1
    nested = any(a["box"][0] < b["box"][0] < b["box"][1] < a["box"][1]
                 and a["box"][2] < b["box"][2] < b["box"][3] < a["box"][3]
                 for a in scene["loops"] for b in scene["loops"])
    return {"events": events, "teaching": {key for key, n in witnesses.items() if n >= 2}, "directions": productive_directions,
            "ray_cells": interior_cells, "nested": nested,
            "multi_loop": any(len({e[2] for e in hits}) >= 2 for hits in events),
            "full_crossing": full_crossing, "persistence": persistence,
            "hidden_hits": sum(len(e) for e in events),
            "max_hits": max(map(len, events), default=0),
            "periods": tuple(loop["period"] for loop in scene["loops"])}


def _sample_pair(rng, max_radius, max_period, direction, kind):
    # kind 0 and 1 are teaching witnesses; kind 2 is additional training;
    # kind 3 tests simultaneous emitters and additional hidden color encounters.
    count = 1 if kind < 2 else int(rng.integers(1 if kind == 2 else 2, 4))
    loop_count = 1 if kind == 0 else 2 if kind == 1 else int(rng.integers(1 if kind == 2 else 2, 4))
    patterned = kind != 0
    extras_count = 0 if kind == 0 else 1 if kind == 1 else int(rng.integers(1 if kind == 3 else 0, 3))
    colors = list(map(int, rng.permutation(10)))
    background, fill, frame, struct = tuple(colors[:2]), colors[2], colors[3], colors[4]
    caps_bank, gate_bank = colors[5:7], colors[7:]
    for _ in range(240):
        low = 28 if count >= 2 or max_radius == 3 else 23
        shape = tuple(map(int, rng.integers(low, 31, size=2)))
        oh, ow = map(int, rng.integers(12 if kind == 1 else 10 if count >= 2 else 8,
                                     17 if kind == 1 or count >= 2 else 15, size=2))
        radii = [max_radius] if kind < 2 else list(map(int, rng.integers(2, max_radius + 1, size=count)))
        if count == 3:
            radii = [2, 2, 2]
        ds = ([direction] * count if kind < 2 or rng.random() < .65 else
              [direction if rng.random() < .5 else (-direction[0], -direction[1]) for _ in range(count)])
        origin = []
        for axis, n in enumerate((oh, ow)):
            low_pos, high_pos = 3, shape[axis] - n - 4
            mid = (low_pos + high_pos) // 2
            if len(set(ds)) == 1:
                if ds[0][axis] > 0:
                    low_pos = mid
                else:
                    high_pos = mid
            else:
                low_pos, high_pos = max(low_pos, mid - 1), min(high_pos, mid + 2)
            origin.append(int(rng.integers(low_pos, high_pos + 1)))
        t, l = origin
        main = (t, t + oh + 1, l, l + ow + 1)
        placed = _place_emitters(rng, shape, main, radii, ds, struct, caps_bank, 6 if kind < 2 else 3)
        if placed is None:
            continue
        emitters, objects, rays = placed
        loops = _sample_loops(rng, shape, main, objects, rays, loop_count, patterned, gate_bank, max_period, kind < 2)
        if loops is None:
            continue
        blocked = (_area(main) | objects | {p for ray in rays for p in ray["path"]}
                   | {p for loop in loops for p in loop["colors"]})
        extras = []
        for _ in range(extras_count):
            candidates = [(y, y + hh + 1, x, x + ww + 1)
                          for hh, ww in ((2, 2), (2, 3), (3, 2), (3, 3))
                          for y in range(1, shape[0] - hh - 2)
                          for x in range(1, shape[1] - ww - 2)]
            rng.shuffle(candidates)
            for box in candidates:
                cells = _area(box)
                if not cells.intersection(_halo(blocked)):
                    extras.append(box)
                    blocked.update(cells)
                    break
            else:
                break
        if len(extras) != extras_count:
            continue
        pair = _paint_pair(shape, background, main, emitters, loops, extras, fill, frame)
        try:
            scene = _read_input(pair["input"])
            out = _trace_output(scene)
            assert np.array_equal(pair["output"], out)
            assert len(scene["emitters"]) == count and len(scene["loops"]) == loop_count
            f = _features(scene)
            assert f["hidden_hits"] >= 2
            assert any(_strict(p, main) and p not in f["ray_cells"] for p in scene["dye"])
            assert not np.array_equal(out, _trace_output(scene, use_dye=False))
            assert not np.array_equal(out, _trace_output(scene, lanes="middle"))
            if kind < 2:
                assert (direction, max_radius) in f["teaching"]
                assert all(any(_strict(p, main) for p in ray["path"]) is False
                           or any(p in scene["dye"] for p in ray["path"] if _strict(p, main))
                           for ray in scene["rays"])
            if kind == 1:
                assert f["multi_loop"] and f["full_crossing"] and max(f["periods"]) >= max_period
                assert not np.array_equal(out, _trace_output(scene, solid=True))
                assert not np.array_equal(out, _trace_output(scene, first_only=True))
        except AssertionError:
            continue
        return pair
    raise RuntimeError("bounded scene search exhausted")


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(num_train, num_test):
    a, b = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require 2..10 training, 1..4 test examples and num_train > num_test")
    return a, b


def _key(grid):
    grid = _grid(grid)
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _counts(num_train, num_test)
    seed = _integer(seed, "seed")
    rng = np.random.default_rng(seed)
    direction = DIRECTIONS[int(rng.integers(4))]
    max_radius = int(rng.integers(2, 4))
    max_period = int(rng.integers(2, 5))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", a), ("test", b)):
        for index in range(count):
            kind = min(index, 2) if split == "train" else 3
            d = (-direction[0], -direction[1]) if kind == 1 else direction
            for _ in range(12):
                pair = _sample_pair(rng, max_radius, max_period, d, kind)
                key = _key(pair["input"])
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise RuntimeError("bounded unique-input search exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, teaching, tests, train_scenes = set(), set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            scene = _read_input(inp)
            assert np.array_equal(out, _trace_output(scene)), "wrong ray color or crop"
            assert np.array_equal(out, _last_crossing_output(scene)), "last-crossing check disagrees"
            assert inp.shape != out.shape
            f = _features(scene)
            assert f["hidden_hits"] >= 2
            assert not np.array_equal(out, _trace_output(scene, use_dye=False)), "B has no causal effect"
            assert not np.array_equal(out, _trace_output(scene, lanes="middle")), "A has no causal effect"
            # Untouched loop cells disappear; the opaque fill remains everywhere
            # except on the two endpoint lanes of each productive cap.
            t, b, l, r = scene["main"]["box"]
            painted = {(y - t - 1, x - l - 1) for y, x in f["ray_cells"]}
            assert set(map(tuple, np.argwhere(out != scene["main"]["fill"]))) == painted
            assert any(_strict(p, scene["main"]["box"]) and p not in f["ray_cells"] for p in scene["dye"])
            if split == "train":
                teaching.update(f["teaching"])
                train_scenes.append(scene)
            else:
                tests.append(scene)
    assert len({d for d, _ in teaching}) >= 2
    train_period = max(loop["period"] for s in train_scenes for loop in s["loops"])
    for scene in tests:
        assert len(scene["emitters"]) >= 2 and len(scene["frames"]) >= 2
        assert max(loop["period"] for loop in scene["loops"]) <= train_period
        for ray in scene["rays"]:
            if any(_strict(p, scene["main"]["box"]) for p in ray["path"]):
                assert any(d == ray["direction"] and radius >= ray["radius"] for d, radius in teaching)
    assert any(len(s["frames"]) >= 2 and _features(s)["multi_loop"] and _features(s)["full_crossing"]
               and not np.array_equal(_trace_output(s), _trace_output(s, solid=True))
               and not np.array_equal(_trace_output(s), _trace_output(s, first_only=True)) for s in train_scenes)


def self_test():
    """Bounded structural, causal, phase, interface, and 42-seed diversity checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, layouts, directions, periods, radii, counts = set(), set(), set(), set(), set(), set()
    nested, rich, masks = 0, 0, set()
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                s = _read_input(pair["input"])
                f = _features(s)
                shapes.add((pair["input"].shape, pair["output"].shape))
                layouts.add((s["grid"].shape, s["main"]["box"],
                             tuple((e["center"], e["radius"], tuple(e["caps"])) for e in s["emitters"]),
                             tuple(l["box"] for l in s["loops"])))
                directions.update(f["directions"])
                periods.update(f["periods"])
                radii.update(e["radius"] for e in s["emitters"])
                counts.add((len(s["emitters"]), len(s["loops"]), len(s["frames"])))
                nested += f["nested"]
                rich += f["max_hits"] >= 3
                masks.update(sum(1 << DIRECTIONS.index(d) for d in e["caps"]) for e in s["emitters"])
    assert len(shapes) >= 80 and len(layouts) >= 180
    assert directions == set(DIRECTIONS) and periods == {1, 2, 3, 4} and radii == {2, 3}
    assert {a for a, _, _ in counts} == {1, 2, 3}
    assert {b for _, b, _ in counts} == {1, 2, 3}
    assert {c for _, _, c in counts} == {1, 2, 3}
    assert nested >= 5 and rich >= 20 and len(masks) >= 10
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(421 + a * 10 + b, a, b)
            assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
    repeat = generate_puzzle(0)
    assert all(_key(x[k]) == _key(y[k]) for s in repeat for x, y in zip(repeat[s], puzzles[0][s]) for k in x)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Mutation checks include both a wrong color and a missing ray cell.
    for delta in (0, 1):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
        scene = _read_input(bad["test"][0]["input"])
        out = bad["test"][0]["output"]
        p = tuple(np.argwhere(out != scene["main"]["fill"])[0])
        out[p] = scene["main"]["fill"] if delta == 0 else (int(out[p]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("damaged ray accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed")
