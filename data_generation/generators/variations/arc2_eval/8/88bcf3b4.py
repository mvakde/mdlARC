"""Anchored strands wrapped around a discrete convex obstacle silhouette.

The digital grammar is deliberately limited to the source's parallel and
45-degree tangents. Colors are local roles, never hidden puzzle parameters.
"""
import hashlib
import numpy as np

_DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _validate_optional_count(value, name, low, high):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(seed, num_train, num_test):
    _validate_optional_count(num_train, "num_train", 2, 10)
    _validate_optional_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid_key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _fingerprint(grid):
    return hashlib.sha256(bytes(grid.shape) + grid.astype(np.int8).tobytes()).hexdigest()


# Exact source input fingerprints; square sizes and source palettes remain legal.
_SOURCE_INPUTS = {'f233e0e1e2f57ddb1f364afd9d08fdf384ae016f0979f0a98b1cb9dc53cf31c6', '35bdabde41a0ddd7ff6c5d7f67f3e66da177fd94718b4aa32376cefacdc2cdca', '2189360f98a7e4b629d85652dab124cb496901d6c9035a61770dabe111c0f264', 'c15c07fae9f3088bd3b01b66ae68988d3dcde0b29441ab726fcd9e78ab296e95', '87528e79c30faaa1a2f3ba3aed51179dcaa3d31a9cc1fd80d93b8f435a5894a3', '0738c2f66fbd7db824ab725552866672fcd1376c08ab828540cded7df9d6f2ef', 'bec12db8ec3200d259beabffff6f60f422efaeaedb7a2ef1b7b83eb718555bda'}


def _mode_color(grid):
    return int(np.bincount(grid.ravel().astype(int), minlength=10).argmax())


def _components(grid, background):
    unseen = {tuple(map(int, p)) for p in np.argwhere(grid != background)}
    result = []
    while unseen:
        start = min(unseen)
        color = int(grid[start])
        cells, todo = {start}, [start]
        unseen.remove(start)
        while todo:
            r, c = todo.pop()
            for dr, dc in _DIR8:
                q = r + dr, c + dc
                if q in unseen and int(grid[q]) == color:
                    unseen.remove(q)
                    cells.add(q)
                    todo.append(q)
        result.append((color, cells))
    return result


def _to_global(cell, endpoint, forward, lateral):
    u, v = cell
    return (endpoint[0] + u * forward[0] + v * lateral[0],
            endpoint[1] + u * forward[1] + v * lateral[1])


def _to_local(cell, endpoint, forward, lateral):
    r, c = cell[0] - endpoint[0], cell[1] - endpoint[1]
    return r * forward[0] + c * forward[1], r * lateral[0] + c * lateral[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _four_connected(cells):
    reached = {min(cells)}
    todo = list(reached)
    while todo:
        u, v = todo.pop()
        for du, dv in _DIR4:
            q = u + du, v + dv
            if q in cells and q not in reached:
                reached.add(q)
                todo.append(q)
    return len(reached) == len(cells)


def _prediction_path(obstacle, length):
    """Input-only analytic reconstruction, separate from construction's sweeps.

    Each obstacle cell requires one lateral cell of clearance. Its influence
    falls by one per axial step. The upper envelope gives the supported convex
    contour; an axis-parallel incoming tangent is held at the attachment side.
    """
    peak = max(v for u, v in obstacle)
    tangent = max(v - u for u, v in obstacle)
    assert min(u for u, v in obstacle) >= 1 and peak >= 0
    assert tangent == 0 or peak == 0, "unsupported fractional incoming tangent"
    first_peak = min(u for u, v in obstacle if v == peak)
    far = max(u for u, v in obstacle)
    n = max(length, far + 4)
    route = [(u, 1 if peak == 0 and u <= first_peak else
              max(v + 1 - abs(u - t) for t, v in obstacle)) for u in range(n)]
    slopes = [b[1] - a[1] for a, b in zip(route, route[1:])]
    assert route[0] == (0, 1)
    assert all(s in (-1, 0, 1) for s in slopes)
    assert slopes == sorted(slopes, reverse=True), "nonconvex exposed contour"
    assert not set(route) & obstacle
    return route[:length]


def _construct_path(obstacle, length):
    """Raster clearance profile with forward/backward Lipschitz sweeps."""
    far = max(u for u, v in obstacle)
    n = max(length, far + 3)
    profile = [-1000] * n
    for u, v in obstacle:
        profile[u] = max(profile[u], v + 1)
    for u in range(1, n):
        profile[u] = max(profile[u], profile[u - 1] - 1)
    for u in range(n - 2, -1, -1):
        profile[u] = max(profile[u], profile[u + 1] - 1)
    if max(v for u, v in obstacle) == 0:
        first = min(u for u, v in obstacle if v == 0)
        profile[:first + 1] = [1] * (first + 1)
    return list(enumerate(profile[:length]))


def _infer_assemblies(inp):
    """Unique exact cover of input components into support/strand/obstacle triples.

    Neither an output nor generation metadata participates in role assignment.
    In particular, separate components can reuse a literal color in any role.
    """
    background = _mode_color(inp)
    components = _components(inp, background)
    candidates = []
    for i, (sc, support) in enumerate(components):
        if len(support) < 3:
            continue
        if len({r for r, c in support}) == 1:
            positive = (0, 1)
        elif len({c for r, c in support}) == 1:
            positive = (1, 0)
        else:
            continue
        endpoints = ((max(support), positive),
                     (min(support), (-positive[0], -positive[1])))
        for endpoint, forward in endpoints:
            for lateral in ((forward[1], -forward[0]), (-forward[1], forward[0])):
                attachment = _to_global((0, 1), endpoint, forward, lateral)
                for j, (tc, strand) in enumerate(components):
                    if tc == sc or attachment not in strand or len(strand) < 2:
                        continue
                    slack = {_to_local(p, endpoint, forward, lateral) for p in strand}
                    if any(u > 0 or v < 1 for u, v in slack):
                        continue
                    for k, (oc, obstacle) in enumerate(components):
                        if oc in (sc, tc):
                            continue
                        local = {_to_local(p, endpoint, forward, lateral) for p in obstacle}
                        try:
                            path = _prediction_path(local, len(strand))
                        except AssertionError:
                            continue
                        candidates.append({
                            "ids": {i, j, k}, "support": support, "strand": strand,
                            "obstacle": obstacle, "local_obstacle": local, "slack": slack,
                            "endpoint": endpoint, "f": forward, "l": lateral,
                            "colors": (sc, tc, oc), "length": len(strand),
                            "path": [_to_global(p, endpoint, forward, lateral) for p in path],
                        })
    covers = []

    def search(used, chosen):
        if len(covers) > 1:
            return
        if len(used) == len(components):
            covers.append(chosen)
            return
        first = min(set(range(len(components))) - used)
        for candidate in candidates:
            if first in candidate["ids"] and not candidate["ids"] & used:
                search(used | candidate["ids"], chosen + [candidate])

    search(set(), [])
    assert len(covers) == 1 and covers[0], "ambiguous input assembly assignment"
    assemblies = covers[0]
    # A second plausible nearby obstacle is rejected even if exact cover could
    # resolve the assignment globally. Cross-scene far-away matches are ignored.
    for a in assemblies:
        own = max(abs(u) + abs(v) for u, v in a["local_obstacle"])
        alternatives = [b for b in candidates if b["support"] == a["support"]
                        and b["strand"] == a["strand"] and b["obstacle"] != a["obstacle"]
                        and max(abs(u) + abs(v) for u, v in b["local_obstacle"]) <= own + 2]
        assert not alternatives, "competing local obstacles"
    return background, assemblies


def _predict(inp):
    background, assemblies = _infer_assemblies(inp)
    out = inp.copy()
    for a in assemblies:
        for p in a["strand"]:
            out[p] = background
    claimed = set()
    for a in assemblies:
        visible = {p for p in a["path"] if _inside(p, inp.shape)}
        assert len(visible) >= 1
        for p in visible:
            assert out[p] == background and p not in claimed, "route collision"
            out[p] = a["colors"][1]
        claimed.update(visible)
        a["visible"] = visible
    assert _mode_color(out) == background
    return out, assemblies


def solve(train_pairs, test_input):
    """Recover roles from the query after checking the demonstrations' rule."""
    assert len(train_pairs) >= 2
    for pair in train_pairs:
        expected, _ = _predict(pair["input"])
        assert np.array_equal(expected, pair["output"])
    return _predict(test_input)[0]


def _obstacle(rng, intensity, approach, tiny=False, compact=False):
    kind = int(rng.integers(0, 6)) if not tiny else int(rng.integers(0, 3))
    scale = 2 + (min(intensity, 1) if compact else intensity)
    if kind == 0:  # point
        raw = {(0, 0)}
    elif kind == 1:  # thin bar, including lateral bars
        h = 2 if tiny else int(rng.integers(2, scale + 2))
        raw = {(u, 0) for u in range(h)} if rng.random() < .65 else {(0, v) for v in range(h)}
    elif kind == 2:  # filled rectangle
        h, w = map(int, rng.integers(1, scale + 1, size=2))
        raw = {(u, v) for u in range(h) for v in range(w)}
    elif kind == 3:  # thin diagonal cap or filled diamond
        radius = int(rng.integers(1, 2 + (intensity == 2)))
        if rng.random() < .5:
            raw = {(u, radius - abs(u - radius)) for u in range(2 * radius + 1)}
        else:
            raw = {(u, v) for u in range(2 * radius + 1)
                   for v in range(-radius, radius + 1) if abs(u - radius) + abs(v) <= radius}
    elif kind == 4:  # stepped filled wedge, preserving the exposed steps
        ramp, plateau = int(rng.integers(1, scale + 1)), int(rng.integers(1, 4))
        raw = {(u, v) for u in range(plateau + ramp)
               for v in range(ramp - max(0, u - plateau + 1) + 1)}
    else:  # thin zigzag band with a plateau then descending steps
        plateau, ramp = int(rng.integers(1, 4)), int(rng.integers(1, scale + 1))
        thickness = int(rng.integers(1, 3))
        raw = {(u, ramp - max(0, u - plateau + 1) - t)
               for u in range(plateau + ramp) for t in range(thickness)}
        if rng.random() < .5:
            # A transposed two-cell band includes the source's four-cell
            # magenta zigzag, with its 1/2/1 axial slice widths.
            raw = {(v, u) for u, v in raw}
            start = min(u for u, v in raw)
            raw = {(u - start, v) for u, v in raw}
    if kind in (4, 5) and rng.random() < .5:
        end = max(u for u, v in raw)
        raw = {(end - u, v) for u, v in raw}
    near = int(rng.integers(1, (2 if tiny else 3 if compact else 5 + intensity) + 1))
    if approach == "parallel":
        offset = -max(v for u, v in raw)
    else:
        offset = near - max(v - u for u, v in raw)
    obstacle = {(u + near, v + offset) for u, v in raw}
    # Disallow profiles which would require a new turn convention.
    _prediction_path(obstacle, 2)
    return obstacle


def _slack(rng, length, depth, width, style):
    """Eight-connected length encodings with independent density and reach."""
    cells = {(0, 1)}
    if style in (1, 2, 3):
        # Monotone filament with long horizontal runs, diagonal links or bends.
        u, v = 0, 1
        while len(cells) < length:
            choices = [(u - 1, v + 1), (u, v + 1), (u - 1, v)]
            weights = ([5, 1, 1] if style == 1 else [2, 5, 1] if style == 3 else [2, 2, 4])
            possible = [(p, w) for p, w in zip(choices, weights)
                        if p[0] >= -depth and p[1] <= width]
            if not possible:
                break
            probs = np.array([w for p, w in possible], dtype=float)
            u, v = possible[int(rng.choice(len(possible), p=probs / probs.sum()))][0]
            cells.add((u, v))
    if style == 5:
        # Rectangular count blocks, with a partial final row when necessary.
        w = int(rng.integers(2, min(width, length) + 1))
        for k in range(length):
            p = -(k // w), 1 + k % w
            if p[0] >= -depth:
                cells.add(p)
            if len(cells) == length:
                break
    while len(cells) < length:
        dirs = _DIR4 if style in (0, 5) else _DIR8
        frontier = sorted({(u + du, v + dv) for u, v in cells for du, dv in dirs
                           if -depth <= u + du <= 0 and 1 <= v + dv <= width} - cells)
        assert frontier
        scores = []
        for p in frontier:
            neighbors = sum((p[0] + du, p[1] + dv) in cells for du, dv in _DIR8)
            scores.append((neighbors + 1) ** 2 if style == 0 else
                          1 / (neighbors + 1) ** 2 if style == 4 else 1)
        weights = np.array(scores, dtype=float)
        cells.add(frontier[int(rng.choice(len(frontier), p=weights / weights.sum()))])
    assert len(cells) == length
    return cells


def _geometry(rng, intensity, approach=None, lesson=None, compact=False, tiny=False):
    approach = approach or ("parallel" if rng.random() < .3 else "diagonal")
    obstacle = _obstacle(rng, 0 if tiny else intensity, approach, tiny, compact)
    far, near = max(u for u, v in obstacle), min(u for u, v in obstacle)
    support_length = int(rng.integers(3, (5 if tiny else 7 if compact else 11)))
    if lesson:
        length = far + int(rng.integers(3, 6))
    else:
        stop = int(rng.integers(0, 4))
        if stop == 0 and near >= 3:
            length = int(rng.integers(2, near + 1))
        elif stop == 1:
            length = int(rng.integers(max(2, near + 1), far + 2))
        elif stop == 2:
            length = far + int(rng.integers(2, 4))
        else:
            length = far + int(rng.integers(4, 9 if not compact else 6))
    length = min(length, 26 - support_length)
    if tiny:
        length = min(length, 6)
    depth = int(rng.integers(1 if tiny else 2, (3 if tiny else 5 if compact else 9) + 1))
    width = int(rng.integers(2, (4 if tiny else 5 if compact else 10) + 1))
    # Sometimes leave only the contour's one-cell clearance outside the input
    # bounding box. Cropping that real canvas edge then truncates the contour.
    if approach == "diagonal" and rng.random() < .3:
        width = max(2, max(v for u, v in obstacle))
    depth = max(depth, (length + width - 1) // width - 1)
    if lesson == "visible":
        # The unshortened length demonstration must distinguish eight- from
        # four-connectivity, including when only two training pairs are asked.
        width = max(3, width)
        depth = max(depth, (length + width - 3) // (width - 1))
        rest = _slack(rng, length - 1, depth - 1, width - 1, int(rng.integers(0, 6)))
        slack = {(0, 1)} | {(u - 1, v + 1) for u, v in rest}
    else:
        slack = _slack(rng, length, depth, width, int(rng.integers(0, 6)))
    path = _construct_path(obstacle, length)
    assert path == _prediction_path(obstacle, length)
    return {"support": {(-k, 0) for k in range(support_length)},
            "strand": slack, "obstacle": obstacle, "path": path}


def _bbox(cells):
    return min(r for r, c in cells), max(r for r, c in cells), min(c for r, c in cells), max(c for r, c in cells)


def _make_pair(rng, intensity, count, axes, background, lesson=None):
    """Compose independently oriented patches, then choose the actual canvas.

    Patches reserve the full conceptual path. Only the final outer canvas can
    clip it; internal packing boundaries never truncate a strand.
    """
    small_scene = count == 1 and lesson is None and rng.random() < .25
    for attempt in range(240):
        tiny = small_scene or attempt >= 100
        patches = []
        for j in range(count):
            approach = ("diagonal" if lesson == "visible" else "parallel") if j == 0 and lesson else None
            geometry = _geometry(rng, intensity, approach, lesson if j == 0 else None,
                                 compact=count >= 3, tiny=tiny)
            axis = int(rng.choice(axes))
            sign, side = (int(rng.choice((-1, 1))) for _ in range(2))
            f, l = ((sign, 0), (0, side)) if axis == 0 else ((0, sign), (side, 0))
            patch = {role: [_to_global(p, (0, 0), f, l) for p in cells]
                     for role, cells in geometry.items()}
            bounds = _bbox([p for cells in patch.values() for p in cells])
            patch = {role: [(r - bounds[0], c - bounds[2]) for r, c in cells]
                     for role, cells in patch.items()}
            patches.append((patch, bounds[1] - bounds[0] + 1, bounds[3] - bounds[2] + 1))
        rows = int(rng.choice([1, 2] if count == 2 else [1, 2, count])) if count > 1 else 1
        cols = (count + rows - 1) // rows
        rh = [max((p[1] for j, p in enumerate(patches) if j // cols == r), default=0) for r in range(rows)]
        cw = [max((p[2] for j, p in enumerate(patches) if j % cols == c), default=0) for c in range(cols)]
        gap = int(rng.integers(2, 5))
        height, width = sum(rh) + gap * (rows - 1), sum(cw) + gap * (cols - 1)
        if max(height, width) > (9 if small_scene else 30):
            continue
        placed = []
        for j, (patch, h, w) in enumerate(patches):
            row, col = j // cols, j % cols
            r0 = sum(rh[:row]) + row * gap + int(rng.integers(0, rh[row] - h + 1))
            c0 = sum(cw[:col]) + col * gap + int(rng.integers(0, cw[col] - w + 1))
            placed.append({role: [(r + r0, c + c0) for r, c in cells] for role, cells in patch.items()})
        input_cells = [p for a in placed for role in ("support", "strand", "obstacle") for p in a[role]]
        all_path = [p for a in placed for p in a["path"]]
        ir0, ir1, ic0, ic1 = _bbox(input_cells)
        bounds = [0, height - 1, 0, width - 1]
        want_clip = lesson == "clipped" or (not lesson and rng.random() < .38)
        clipped_edge = None
        if want_clip:
            options = [edge for edge, limit in enumerate((ir0, ir1, ic0, ic1))
                       if any((p[edge // 2] < limit if edge % 2 == 0 else p[edge // 2] > limit) for p in all_path)]
            if not options:
                continue
            clipped_edge = int(rng.choice(options))
            bounds[clipped_edge] = (ir0, ir1, ic0, ic1)[clipped_edge]
        height, width = bounds[1] - bounds[0] + 1, bounds[3] - bounds[2] + 1
        margins = [0, 0, 0, 0]
        for axis, size in enumerate((height, width)):
            budget = min((9 if small_scene else 30) - size,
                         int(rng.integers(0, 3 if small_scene else 3 + 5 * intensity)))
            available = [2 * axis + q for q in (0, 1) if 2 * axis + q != clipped_edge]
            for _ in range(budget):
                margins[int(rng.choice(available))] += 1
        height += margins[0] + margins[1]
        width += margins[2] + margins[3]
        if rng.random() < .38:
            axis = 0 if height < width else 1
            difference = abs(height - width)
            options = [2 * axis + q for q in (0, 1) if 2 * axis + q != clipped_edge]
            margins[int(rng.choice(options))] += difference
            height = width = max(height, width)
        dr, dc = margins[0] - bounds[0], margins[2] - bounds[2]
        placed = [{role: [(r + dr, c + dc) for r, c in cells] for role, cells in a.items()} for a in placed]
        palette = [c for c in range(10) if c != background]
        rng.shuffle(palette)
        disjoint = count <= 3 and rng.random() < .5
        colors = [tuple(palette[3 * j:3 * j + 3]) if disjoint else
                  tuple(map(int, rng.choice(palette, 3, replace=False))) for j in range(count)]
        inp = np.full((height, width), background, dtype=np.int8)
        for a, palette3 in zip(placed, colors):
            for role, color in zip(("support", "strand", "obstacle"), palette3):
                for p in a[role]:
                    assert _inside(p, inp.shape) and inp[p] == background
                    inp[p] = color
        out = inp.copy()
        for a in placed:
            for p in a["strand"]:
                out[p] = background
        for a, palette3 in zip(placed, colors):
            for p in a["path"]:
                if _inside(p, out.shape):
                    assert out[p] == background
                    out[p] = palette3[1]
        if np.array_equal(inp, out) or _fingerprint(inp) in _SOURCE_INPUTS:
            continue
        try:
            predicted, inferred = _predict(inp)
            assert len(inferred) == count and np.array_equal(predicted, out)
            if lesson == "visible":
                assert any(_has_departure(a, inp.shape) and len(a["visible"]) == a["length"] for a in inferred)
            if want_clip:
                assert any(len(a["visible"]) < a["length"] for a in inferred)
        except AssertionError:
            continue
        return {"input": inp, "output": out}
    # Small independent fallback patches fit a 2x2 layout; continue with a
    # deterministic derived stream, a simpler scene, and the same lesson/axes.
    # This is bounded separately so legal requests never use unbounded retries.
    return _fallback_pair(rng, axes, background, lesson, count)


def _fallback_pair(rng, axes, background, lesson, count):
    # A guaranteed simple geometric construction. Fallback colors are all
    # distinct for triples when possible, with generous spatial separation.
    axis = int(axes[0])
    h, w = (20, 20) if count > 1 else (10, 10)
    inp = np.full((h, w), background, dtype=np.int8)
    out = inp.copy()
    palette = [c for c in range(10) if c != background]
    rng.shuffle(palette)
    for j in range(count):
        r0, c0 = (j // 2) * 10, (j % 2) * 10
        forward, lateral = ((-1, 0), (0, -1)) if axis == 0 else ((0, -1), (-1, 0))
        end = (r0 + 5, c0 + 5)
        obs = {(2, 0)} if lesson == "clipped" and j == 0 else {(2, 2)}
        length = 9 if lesson == "clipped" and j == 0 else 6
        slack = {(0, 1)} | {(-1 - k // 3, 2 + k % 3) for k in range(length - 1)}
        role_cells = ({(-k, 0) for k in range(4)}, slack, obs)
        colors = [palette[(3 * j + k) % 9] for k in range(3)]
        for cells, color in zip(role_cells, colors):
            for p in cells:
                inp[_to_global(p, end, forward, lateral)] = color
    predicted, _ = _predict(inp)
    return {"input": inp, "output": predicted}


def _has_departure(a, shape):
    far = max(u for u, v in a["local_obstacle"])
    return any(_to_local(p, a["endpoint"], a["f"], a["l"])[0] > far + 1
               for p in a["path"] if _inside(p, shape))


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    nt, nq = _resolve_counts(seed, num_train, num_test)
    # These govern curriculum and variation intensity, not output semantics.
    intensity, regime = int(rng.integers(0, 3)), int(rng.integers(0, 7))
    backgrounds = list(map(int, rng.permutation(10)))
    demonstration_slots = list(map(int, rng.choice(nt, 2, replace=False)))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, number in (("train", nt), ("test", nq)):
        for index in range(number):
            if regime in (0, 1):
                axes = [regime]
            elif regime in (2, 3):
                axes = [(regime - 2) if split == "train" else (3 - regime)]
            else:
                axes = [0, 1]
            if regime == 5:
                count = 1 if split == "train" else int(rng.integers(2, 5))
            else:
                limit = 2 + intensity
                count = int(rng.choice(np.arange(1, limit + 1), p=np.array([4] + [1] * (limit - 1)) / (limit + 3)))
            lesson = None
            if split == "train" and index in demonstration_slots:
                lesson = "visible" if index == demonstration_slots[0] else "clipped"
            bg = backgrounds[(index + (nt if split == "test" else 0)) % 10]
            for retry in range(20):
                pair = _make_pair(rng, intensity, count, axes, bg, lesson)
                key = _grid_key(pair["input"])
                different_size = (split != "train" or index != 1 or
                                  pair["input"].shape != puzzle["train"][0]["input"].shape)
                if key not in seen and different_size:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("unique pair construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs, training, backgrounds = set(), set(), [], set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for g in pair.values():
                assert isinstance(g, np.ndarray) and g.ndim == 2
                assert np.issubdtype(g.dtype, np.integer)
                assert all(1 <= d <= 30 for d in g.shape)
                assert np.all((0 <= g) & (g <= 9))
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            ik, pk = _grid_key(inp), (_grid_key(inp), _grid_key(out))
            assert ik not in seen_inputs and pk not in seen_pairs
            assert _fingerprint(inp) not in _SOURCE_INPUTS
            seen_inputs.add(ik)
            seen_pairs.add(pk)
            predicted, assemblies = _predict(inp)
            assert np.array_equal(predicted, out), "input-only reconstruction mismatch"
            assert 1 <= len(assemblies) <= 4
            occupied = set()
            for a in assemblies:
                assert 3 <= len(a["support"]) <= 10 and 2 <= a["length"] <= 25
                for role, color in (("support", a["colors"][0]), ("obstacle", a["colors"][2])):
                    assert all(out[p] == color for p in a[role])
                assert _to_global((0, 1), a["endpoint"], a["f"], a["l"]) in a["visible"]
                footprint = a["support"] | a["strand"] | a["obstacle"] | a["visible"]
                assert not footprint & occupied, "assemblies contact one another"
                occupied.update((r + dr, c + dc) for r, c in footprint
                                for dr in (-1, 0, 1) for dc in (-1, 0, 1))
            if split == "train":
                training.extend((a, inp.shape) for a in assemblies)
                backgrounds.add(_mode_color(inp))
    assert len(backgrounds) >= 2
    assert any(len(a["visible"]) < a["length"] for a, shape in training), "missing clipping demonstration"
    assert any(_has_departure(a, shape) and len(a["visible"]) == a["length"] for a, shape in training)
    assert any(not _four_connected(a["slack"]) and len(a["visible"]) == a["length"]
               for a, shape in training), "missing diagonal length-count demonstration"
    assert {max(v for u, v in a["local_obstacle"]) == 0 for a, shape in training} == {False, True}


def measure_pair(pair):
    """Measurements reconstructed from returned grids, for audit and self-tests."""
    inp = pair["input"]
    _, assemblies = _predict(inp)
    rows = []
    for a in assemblies:
        obstacle = a["local_obstacle"]
        near, far, lo, hi = _bbox(obstacle)
        rectangular = len(obstacle) == (far - near + 1) * (hi - lo + 1)
        lost = [p for p in a["path"] if not _inside(p, inp.shape)]
        edges = {name for p in lost for name, cond in
                 (("top", p[0] < 0), ("bottom", p[0] >= inp.shape[0]),
                  ("left", p[1] < 0), ("right", p[1] >= inp.shape[1])) if cond}
        stage = "approach" if a["length"] <= near else "contour" if a["length"] <= far + 1 else "turn" if a["length"] <= far + 3 else "outgoing"
        sr0, sr1, sc0, sc1 = _bbox(a["slack"])
        branches = sum(sum((u + du, v + dv) in a["slack"] for du, dv in _DIR8) >= 3
                       for u, v in a["slack"])
        rows.append({"axis": "vertical" if a["f"][0] else "horizontal", "forward": a["f"], "side": a["l"],
                     "support_length": len(a["support"]), "length": a["length"], "stop": stage,
                     "near": near, "far": far, "lateral_min": lo, "lateral_max": hi,
                     "obstacle_height": far - near + 1, "obstacle_width": hi - lo + 1,
                     "obstacle": "point" if len(obstacle) == 1 else "bar" if rectangular and (near == far or lo == hi) else "rectangle" if rectangular else "stepped",
                     "approach": "parallel" if hi == 0 else "diagonal", "needs_diagonal": not _four_connected(a["slack"]),
                     "slack_span": sc1 - sc0 + 1, "slack_density": round(a["length"] / ((sr1 - sr0 + 1) * (sc1 - sc0 + 1)), 3),
                     "branched": branches > 0,
                     "clipped": bool(lost), "clip_edges": sorted(edges),
                     "contour_clipped": any(_to_local(p, a["endpoint"], a["f"], a["l"])[0] <= far for p in lost),
                     "colors": a["colors"], "edge_support": any(r in (0, inp.shape[0] - 1) or c in (0, inp.shape[1] - 1) for r, c in a["support"]),
                     "silhouette": tuple(sorted((u - near, v - lo) for u, v in obstacle))})
    roles = {}
    for a in rows:
        for role, color in enumerate(a["colors"]):
            roles.setdefault(color, set()).add(role)
    return {"shape": inp.shape, "background": _mode_color(inp), "color_count": len(np.unique(inp)), "count": len(rows),
            "role_reuse": len(roles) < 3 * len(rows),
            "cross_role_reuse": any(len(v) > 1 for v in roles.values()), "assemblies": rows}


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(p[k], q[k])
               for p, q in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    # Source-derived normalized route regressions, including vertical approaches,
    # the three diagonal green cells, stepped obstacles, and early departure.
    cases = [({(2, 0)}, 6, [(0, 1), (1, 1), (2, 1), (3, 0), (4, -1), (5, -2)]),
             ({(u, 3) for u in range(3, 6)}, 9, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 4), (5, 4), (6, 3), (7, 2), (8, 1)]),
             ({(1, 1), (2, 2), (3, 1)}, 6, [(0, 1), (1, 2), (2, 3), (3, 2), (4, 1), (5, 0)]),
             ({(u, v) for u in (2, 3) for v in (0, 1, 2)}, 6, [(0, 1), (1, 2), (2, 3), (3, 3), (4, 2), (5, 1)]),
             ({(3, 3), (4, 3), (4, 2), (5, 2)}, 7, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 4), (5, 3), (6, 2)])]
    for obs, n, expected in cases:
        assert _construct_path(obs, n) == expected == _prediction_path(obs, n)
    counts, tests, sizes, structures, backgrounds = set(), set(), set(), set(), set()
    all_rows, pairs, observed_regimes, varying_counts = [], [], set(), 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        counts.add(len(p["train"]))
        tests.add(len(p["test"]))
        puzzle_metrics = [measure_pair(pair) for split in ("train", "test") for pair in p[split]]
        assert len({m["background"] for m in puzzle_metrics}) >= 3
        assert len({tuple(m["shape"]) for m in puzzle_metrics}) >= 2
        varying_counts += len({m["count"] for m in puzzle_metrics}) > 1
        observed_regimes.add(tuple(tuple(sorted({a["axis"] for pair in p[split]
                                                for a in measure_pair(pair)["assemblies"]}))
                                  for split in ("train", "test")))
        for split in ("train", "test"):
            for pair in p[split]:
                m = measure_pair(pair)
                pairs.append((split, m))
                sizes.add(m["shape"])
                backgrounds.add(m["background"])
                all_rows.extend(m["assemblies"])
                structures.add(tuple((a["forward"], a["side"], a["support_length"], a["length"], a["silhouette"]) for a in m["assemblies"]))
        for pair in p["test"]:
            train = [{k: g.copy() for k, g in v.items()} for v in p["train"]]
            assert np.array_equal(solve(train, pair["input"].copy()), pair["output"])
    assert len(counts) >= 4 and len(tests) >= 3 and len(backgrounds) == 10
    assert len(sizes) > 60 and len(structures) > 150
    for split in ("train", "test"):
        ms = [m for s, m in pairs if s == split]
        rows = [a for m in ms for a in m["assemblies"]]
        assert {a["axis"] for a in rows} == {"horizontal", "vertical"}
        assert {m["count"] for m in ms} >= {1, 2, 3}
        assert any(m["shape"][0] == m["shape"][1] for m in ms)
        assert sum(max(m["shape"]) <= 9 for m in ms) >= 3
        assert {a["obstacle"] for a in rows} == {"point", "bar", "rectangle", "stepped"}
        assert {a["stop"] for a in rows} == {"approach", "contour", "turn", "outgoing"}
        assert {e for a in rows for e in a["clip_edges"]} == {"top", "bottom", "left", "right"}
        assert any(a["contour_clipped"] for a in rows)
        assert any(a["needs_diagonal"] for a in rows) and any(not a["needs_diagonal"] for a in rows)
        assert any(m["count"] > 1 and m["role_reuse"] for m in ms)
        assert any(m["count"] > 1 and not m["role_reuse"] for m in ms)
        assert any(m["cross_role_reuse"] for m in ms)
        assert any(m["count"] > 1 and any(a["clipped"] for a in m["assemblies"]) for m in ms)
        assert {a["axis"] for a in rows if a["clipped"]} == {"vertical", "horizontal"}
        assert any(a["edge_support"] for a in rows)
        assert any(a["branched"] for a in rows) and any(not a["branched"] for a in rows)
        assert min(a["slack_density"] for a in rows) < .3 and max(a["slack_density"] for a in rows) == 1
        assert any(m["count"] > 1 and any(a["obstacle"] == "stepped" for a in m["assemblies"])
                   and any(a["needs_diagonal"] for a in m["assemblies"])
                   and any(a["clipped"] for a in m["assemblies"]) for m in ms)
    assert min(a["length"] for a in all_rows) <= 3 and max(a["length"] for a in all_rows) >= 15
    assert max(a["support_length"] for a in all_rows) >= 9
    assert max(a["slack_span"] for a in all_rows) >= 7
    assert len({a["silhouette"] for a in all_rows}) > 25
    assert varying_counts >= 25
    assert (("vertical",), ("horizontal",)) in observed_regimes
    assert (("horizontal",), ("vertical",)) in observed_regimes
    assert (("vertical",), ("vertical",)) in observed_regimes
    assert (("horizontal",), ("horizontal",)) in observed_regimes
    assert (("horizontal", "vertical"), ("horizontal", "vertical")) in observed_regimes
    offsets = {}
    for a in all_rows:
        offsets.setdefault(a["silhouette"], set()).add((a["near"], a["lateral_min"]))
    assert sum(len(v) >= 4 for v in offsets.values()) >= 5
    assert len({a["obstacle_height"] for a in all_rows}) >= 5
    assert len({a["obstacle_width"] for a in all_rows}) >= 4
    assert any(a["length"] == 5 for a in all_rows) and any(a["support_length"] == 7 for a in all_rows)
    for nt, nq in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None), (6, 3)):
        p = generate_puzzle(991, nt, nq)
        validate_generated_puzzle(p)
        assert nt is None or len(p["train"]) == nt
        assert nq is None or len(p["test"]) == nq
    assert _puzzles_equal(generate_puzzle(71, 4, 2), generate_puzzle(71, 4, 2))
    assert not _puzzles_equal(generate_puzzle(71, 4, 2), generate_puzzle(72, 4, 2))
    for nt, nq in ((1, 1), (11, 1), (5, 0), (6, 5), (3, 3), (2, 4), (True, 1), (3.2, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    damaged = generate_puzzle(812, 2, 1)
    q = damaged["test"][0]
    _, assemblies = _predict(q["input"])
    p = next(iter(assemblies[0]["support"]))
    q["output"][p] = _mode_color(q["input"])
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        return
    raise AssertionError("validator accepted damaged support")
