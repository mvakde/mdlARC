"""Shadow-clearance routing: an ordered path and its translated footprint.

Only standard-library modules and NumPy are used. No parent generator, file,
seed, or generated example is consulted by the input-only solver.
"""
from __future__ import annotations

import numpy as np


OFFSETS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
WALL_SHAPES = ((1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid cells must be integers")
    _require(np.all((a >= 0) & (a <= 9)), "invalid color")
    return a.astype(np.int8, copy=True)


def _sign(x):
    return int(x > 0) - int(x < 0)


def _line(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    n = max(abs(dr), abs(dc))
    if n == 0 or not (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
        return None
    return tuple((a[0] + i * _sign(dr), a[1] + i * _sign(dc)) for i in range(n + 1))


def _translate(pixels, offset):
    return {(r + offset[0], c + offset[1]) for r, c in pixels}


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _intersects(a, b, c, d):
    if (max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0])
            or max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1])):
        return False
    return _cross(a, b, c) * _cross(a, b, d) <= 0 and _cross(c, d, a) * _cross(c, d, b) <= 0


def _conflict(a, b, c, d):
    common = {a, b} & {c, d}
    if common:
        # For allowed lattice directions, overlapping incident segments also
        # overlap in raster cells. A shared endpoint alone is permitted.
        return (set(_line(a, b)) & set(_line(c, d))) != common
    return _intersects(a, b, c, d)  # includes diagonal half-cell crossings


def _components(pixels):
    pending = set(pixels)
    result = []
    while pending:
        start = min(pending)
        pending.remove(start)
        stack, component = [start], {start}
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in pending:
                    pending.remove(p)
                    stack.append(p)
                    component.add(p)
        result.append(component)
    return result


def _decode(grid):
    bg = int(grid[0, 0])
    _require(all(np.all(e == bg) for e in (grid[0], grid[-1], grid[:, 0], grid[:, -1])),
             "input needs a uniform background frame")
    occupied_rows = np.flatnonzero(np.any(grid != bg, axis=1))
    _require(len(occupied_rows) > 0, "empty input")
    key_y = int(occupied_rows[-1])
    xs = np.flatnonzero(grid[key_y] != bg)
    _require(3 <= len(xs) <= 10 and np.all(np.diff(xs) == 2), "invalid detached key")
    key = tuple(int(grid[key_y, x]) for x in xs)
    _require(len(set(key)) >= 2, "key must use at least two labels")
    _require(np.all(grid[key_y-1] == bg), "key must have a blank row above it")
    scene = grid[:key_y-1]
    colors = set(int(c) for c in np.unique(scene)) - {bg} - set(key)
    _require(len(colors) == 2, "need one marker color and one obstacle color")
    singletons = [c for c in colors if np.count_nonzero(scene == c) == 1]
    _require(len(singletons) == 1, "the companion marker must be a singleton color")
    marker_color = singletons[0]
    wall_color = next(c for c in colors if c != marker_color)
    marker = tuple(int(x) for x in np.argwhere(scene == marker_color)[0])
    walls = {tuple(map(int, p)) for p in np.argwhere(scene == wall_color)}
    wall_components = _components(walls)
    for component in wall_components:
        rr, cc = zip(*component)
        h, w = max(rr)-min(rr)+1, max(cc)-min(cc)+1
        _require(1 <= h <= 3 and 1 <= w <= 3 and 2 <= len(component) == h*w,
                 "obstacles must be filled rectangles of at least two cells")
    points = sorted(tuple(map(int, p)) for p in np.argwhere(np.isin(scene, key)))
    _require(4 <= len(points) <= 16 and len(key) <= len(points), "invalid waypoint count")
    _require(all(max(abs(a[0]-b[0]), abs(a[1]-b[1])) >= 3
                 for i, a in enumerate(points) for b in points[:i]), "waypoints too close")
    _require(all(max(abs(a[0]-b[0]), abs(a[1]-b[1])) >= 2
                 for a in points + [marker] for b in walls), "obstacle touches a point")
    labels = tuple(int(grid[p]) for p in points)
    _require(set(labels) == set(key), "labels must occur in both key and scene")
    anchors = [i for i, p in enumerate(points)
               if max(abs(p[0]-marker[0]), abs(p[1]-marker[1])) == 1]
    _require(len(anchors) == 1, "marker must adjoin exactly one waypoint")
    start = anchors[0]
    _require(labels[start] == key[0], "marker anchor must match first key entry")
    offset = (marker[0]-points[start][0], marker[1]-points[start][1])
    return dict(background=bg, key=key, key_y=key_y, points=points, labels=labels,
                marker=marker, marker_color=marker_color, wall_color=wall_color,
                walls=walls, wall_components=wall_components, start=start,
                offset=offset, shape=grid.shape)


def _edges(info, shadow=True):
    points, walls = info["points"], info["walls"]
    point_set = set(points)
    result = {}
    for i, a in enumerate(points):
        for j in range(i+1, len(points)):
            b = points[j]
            line = _line(a, b)
            if line is None or set(line[1:-1]) & (point_set | walls | {info["marker"]}):
                continue
            copy = _translate(line, info["offset"])
            if not all(0 < r < info["key_y"]-1 and 0 < c < info["shape"][1]-1 for r, c in copy):
                continue
            if shadow and copy & walls:
                continue
            result[i, j] = line
    return result


def _routes(info, edges, limit=2, simple=True):
    adjacency = [[] for _ in info["points"]]
    for i, j in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)
    answers = []
    points, labels, key = info["points"], info["labels"], info["key"]

    def visit(path):
        if len(answers) >= limit:
            return
        if len(path) == len(key):
            answers.append(tuple(path))
            return
        a = points[path[-1]]
        for j in adjacency[path[-1]]:
            if j in path or labels[j] != key[len(path)]:
                continue
            b = points[j]
            if simple and any(_conflict(a, b, points[u], points[v]) for u, v in zip(path, path[1:])):
                continue
            visit(path+[j])

    visit([info["start"]])
    return answers


def _footprint(points, route):
    return set().union(*(set(_line(points[i], points[j])) for i, j in zip(route, route[1:])))


def _greedy_source_route(info, nearest):
    """Two explicit shadow-blind alternatives used only to check demonstrations."""
    edges = _edges(info, shadow=False)
    points, labels = info["points"], info["labels"]
    path = [info["start"]]
    for color in info["key"][1:]:
        a = points[path[-1]]
        options = [j for j, b in enumerate(points) if j not in path and labels[j] == color
                   and tuple(sorted((path[-1], j))) in edges
                   and not any(_conflict(a, b, points[u], points[v]) for u, v in zip(path, path[1:]))]
        if not options:
            break
        rank = (lambda j: ((a[0]-points[j][0])**2 + (a[1]-points[j][1])**2, points[j])) if nearest else (lambda j: points[j])
        path.append(min(options, key=rank))
    return tuple(path)


def _paint(grid, info, route):
    out = grid.copy()
    source = _footprint(info["points"], route)
    copy = _translate(source, info["offset"])
    for p in copy - source:
        if grid[p] == info["background"]:
            out[p] = info["marker_color"]
    for i, j in zip(route, route[1:]):
        for p in _line(info["points"][i], info["points"][j]):
            if grid[p] == info["background"]:
                out[p] = info["labels"][i]
    return out


def _solve_info(input_grid):
    grid = _grid(input_grid)
    info = _decode(grid)
    edges = _edges(info)
    routes = _routes(info, edges)
    _require(len(routes) == 1, "key and dual clearance must select exactly one simple route")
    info["route"], info["edges"] = routes[0], edges
    return _paint(grid, info, routes[0]), info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer roles, displacement and the unique route entirely from the input."""
    return _solve_info(input_grid)[0]


def _axis(rng, n, maximum, uniform=None):
    if uniform is not None:
        return [i*uniform for i in range(n)]
    gaps = [3] * (n-1)
    spare = maximum - sum(gaps)
    for i in rng.permutation(n-1):
        extra = int(rng.integers(min(4, spare)+1))
        gaps[int(i)] += extra
        spare -= extra
    return [0] + list(np.cumsum(gaps).astype(int))


def _random_route(rng, points, edges, length, mixed):
    adjacency = [[] for _ in points]
    for i, j in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)
    budget = [1800]

    def visit(path, axes):
        budget[0] -= 1
        if budget[0] < 0:
            return None
        if len(path) == length:
            if mixed and not {"d", "o"} <= axes:
                return None
            return path
        a = points[path[-1]]
        for raw_j in rng.permutation(adjacency[path[-1]]):
            j = int(raw_j)
            if j in path:
                continue
            b = points[j]
            if any(_conflict(a, b, points[u], points[v]) for u, v in zip(path, path[1:])):
                continue
            axis = "d" if a[0] != b[0] and a[1] != b[1] else "o"
            result = visit(path+[j], axes | {axis})
            if result is not None:
                return result
        return None

    for start in rng.permutation(len(points)):
        result = visit([int(start)], set())
        if result is not None:
            return tuple(result)
        if budget[0] < 0:
            break
    return None


def _wall_options(rng, info, forbidden, hit=None):
    """Rectangles clear of the intended strokes; hit optionally blocks a shadow."""
    h, w = info["shape"]
    if hit is None:
        origins = [(r, c) for r in range(1, info["key_y"]-1) for c in range(1, w-1)]
    else:
        origins = sorted({(r-dr, c-dc) for r, c in hit for dr in range(3) for dc in range(3)})
    rng.shuffle(origins)
    shapes = [WALL_SHAPES[int(i)] for i in rng.permutation(len(WALL_SHAPES))]
    halo = {(r+dr, c+dc) for r, c in info["walls"] | set(info["points"]) | {info["marker"]}
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
    for r, c in origins:
        for hh, ww in shapes:
            if not (r >= 1 and c >= 1 and r+hh <= info["key_y"]-1 and c+ww <= w-1):
                continue
            pixels = {(r+dr, c+dc) for dr in range(hh) for dc in range(ww)}
            if pixels & (forbidden | halo) or (hit is not None and not pixels & hit):
                continue
            return pixels
    return None


def _construct(rng, offsets, length, witness=False, mixed=False, crossing_evidence=True):
    """Propose geometry and labels, then block unwanted translated branches."""
    for _ in range(150):
        nr, nc = int(rng.integers(2, 6)), int(rng.integers(2, 7))
        minimum = max(4, length)
        n = int(rng.integers(minimum, min(16, length+6, nr*nc)+1)) if nr*nc >= minimum else 0
        if n == 0:
            continue
        uniform = int(rng.integers(3, min(5, 20//(nr-1), 25//(nc-1))+1)) if mixed or rng.random() < .5 else None
        ys, xs = _axis(rng, nr, 20, uniform), _axis(rng, nc, 25, uniform)
        top = int(rng.integers(2, min(5, 23-max(ys))+1))
        left = int(rng.integers(2, min(6, 27-max(xs))+1))
        universe = [(int(r)+top, int(c)+left) for r in ys for c in xs]
        points = sorted(universe[int(i)] for i in rng.choice(len(universe), n, replace=False))
        height_min = max(r for r, c in points)+5
        width_min = max(max(c for r, c in points)+3, 2*length+1)
        h, w = int(rng.integers(height_min, 31)), int(rng.integers(width_min, 31))
        dummy = dict(points=points, walls=set(), marker=(-10, -10), offset=(0, 0),
                     shape=(h, w), key_y=h-2)
        raw_edges = _edges(dummy)
        route = _random_route(rng, points, raw_edges, length, mixed)
        if route is None:
            continue
        source = _footprint(points, route)
        feasible = [d for d in offsets if (points[route[0]][0]+d[0], points[route[0]][1]+d[1]) not in source]
        if not feasible:
            continue
        offset = feasible[int(rng.integers(len(feasible)))]
        marker = (points[route[0]][0]+offset[0], points[route[0]][1]+offset[1])
        palette = [int(c) for c in rng.permutation(10)]
        bg, wall_color, marker_color = palette[:3]
        k = int(rng.integers(2, min(7, length, 4 if witness else 7)+1))
        pool = palette[3:3+k]
        for _ in range(35):
            key = list(rng.permutation(pool + [int(rng.choice(pool)) for _ in range(length-k)]))
            labels = [int(rng.choice(pool)) for _ in points]
            for i, color in zip(route, key):
                labels[i] = int(color)
            info = dict(dummy, marker=marker, offset=offset, labels=tuple(labels), key=tuple(key),
                        start=route[0], background=bg, marker_color=marker_color, wall_color=wall_color)
            original_routes = _routes(info, _edges(info, shadow=False), limit=3)
            if witness and len(original_routes) < 2:
                continue
            protected = source | _translate(source, offset)
            walls = set()
            for _ in range(5):
                twin_routes = _routes(info, _edges(info), limit=40)
                wrong = [p for p in twin_routes if p != route]
                if not wrong:
                    break
                rng.shuffle(wrong)
                addition = None
                for path in wrong:
                    wrong_source = _footprint(points, path)
                    hit = _translate(wrong_source, offset) - protected - wrong_source
                    addition = _wall_options(rng, info, protected | wrong_source, hit)
                    if addition is not None:
                        break
                if addition is None:
                    break
                walls.update(addition)
                info["walls"] = walls
            twin_routes = _routes(info, _edges(info), limit=2)
            if twin_routes != [route]:
                continue
            if witness and len(_routes(info, _edges(info, shadow=False), limit=2)) < 2:
                continue
            # Inert obstacles make clearance a visible convention even in easy
            # examples. They are not extra source constructions.
            for _ in range(int(rng.integers(1, 4)) if not walls else int(rng.integers(3))):
                addition = _wall_options(rng, info, protected)
                if addition is not None:
                    walls.update(addition)
                    info["walls"] = walls
            if not walls or _routes(info, _edges(info)) != [route]:
                continue
            grid = np.full((h, w), bg, dtype=np.int8)
            for p, color in zip(points, labels):
                grid[p] = color
            grid[marker] = marker_color
            for p in walls:
                grid[p] = wall_color
            key_x = int(rng.integers(1, w-2*length+1))
            grid[h-2, key_x:key_x+2*length:2] = key
            output = _paint(grid, info, route)
            # Require a key omission to add a visible, dual-clear edge.
            solved, decoded = _solve_info(grid)
            metrics = _features(grid, decoded)
            if (not np.array_equal(solved, output) or metrics["shadow_visible"] < 2
                    or (witness and metrics["key_omission_cells"] == 0)
                    or (witness and not metrics["source_only_ambiguous"])
                    or (witness and not (metrics["nearest_fails"] and metrics["row_major_fails"]))
                    or (not crossing_evidence and metrics["crossing_rejection_needed"])
                    or (mixed and (metrics["new_overlap"] == 0 or metrics["copy_endpoint_cells"] == 0))):
                continue
            return dict(input=grid, output=output)
    return None


def _fallback(rng, offset, witness, mixed):
    """A reflected/scaled fork with a shadow-only blocker after bounded search."""
    d = int(rng.integers(6, 9))
    # The wrong second waypoint is both closer and earlier in row-major order.
    # Choose the side so the given marker never lies on the selected diagonal
    # or the rejected branch's original horizontal segment.
    side = -_sign(offset[1]) if offset[0] in (0, 1) and offset[1] else -1
    points = [(0, 0), (0, side*d), (d, side*d), (d, 0), (0, side*2*d)]
    route = (0, 2, 3)
    labels = [0, 1, 1, 2, 0]
    if offset[0]:
        wall = {(offset[0], side*3), (offset[0], side*4)}
    else:
        wall = {(1, side*(d-3)), (2, side*(d-3))}
    # Witness key 0,1,2 has routes 0->1->3 and 0->2->3 before shadow clearance.
    marker = offset
    all_pixels = set(points) | wall | _translate(_footprint(points, route), offset) | {marker}
    dr, dc = 2-min(r for r, c in all_pixels), 2-min(c for r, c in all_pixels)
    points = [(r+dr, c+dc) for r, c in points]
    wall = {(r+dr, c+dc) for r, c in wall}
    marker = marker[0]+dr, marker[1]+dc
    max_r, max_c = max(r+dr for r, c in all_pixels), max(c+dc for r, c in all_pixels)
    h, w = max_r+5, max(max_c+3, 9)
    palette = [int(c) for c in rng.permutation(10)]
    bg, wc, mc = palette[:3]
    labels = tuple(palette[3+i] for i in labels)
    key = tuple(labels[i] for i in route)
    grid = np.full((h, w), bg, dtype=np.int8)
    for p, c in zip(points, labels):
        grid[p] = c
    for p in wall:
        grid[p] = wc
    grid[marker] = mc
    grid[h-2, 1:7:2] = key
    info = dict(points=points, labels=labels, background=bg, offset=offset,
                marker_color=mc)
    output = _paint(grid, info, route)
    return dict(input=grid, output=output)


def _fingerprint(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    held = OFFSETS[int(rng.integers(8))]
    training_offsets = tuple(d for d in OFFSETS if d != held)
    axial = [d for d in training_offsets if 0 in d]
    diagonal = [d for d in training_offsets if 0 not in d]
    introductory_offsets = [axial[int(rng.integers(len(axial)))],
                            diagonal[int(rng.integers(len(diagonal)))]]
    rng.shuffle(introductory_offsets)
    demos = [int(i) for i in rng.choice(int(num_train), 2, replace=False)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            witness = split == "test" or index == demos[0]
            mixed = split == "train" and index == demos[1]
            length = (int(rng.integers(5, 10)) if split == "test" else
                      int(rng.integers(4, 7)) if witness or mixed else int(rng.integers(3, 8)))
            offsets = ((introductory_offsets[index],) if index < 2 else training_offsets) if split == "train" else (held,)
            crossing_evidence = split == "train" or any(
                _features(p["input"], _solve_info(p["input"])[1])["crossing_rejection_needed"]
                for p in puzzle["train"])
            pair = _construct(rng, offsets, length, witness=witness, mixed=mixed,
                              crossing_evidence=crossing_evidence)
            if pair is None:
                pair = _fallback(rng, offsets[int(rng.integers(len(offsets)))], witness, mixed)
            # A color bijection preserves the visible rule and provides a finite
            # uniqueness fallback even if two structural proposals coincide.
            if _fingerprint(pair["input"]) in seen:
                variants = ({k: ((mult*v.astype(int)+shift) % 10).astype(np.int8) for k, v in pair.items()}
                            for mult in (1, 3, 7, 9) for shift in range(10))
                pair = next((p for p in variants if _fingerprint(p["input"]) not in seen), None)
                if pair is None:
                    raise RuntimeError("distinct input construction exhausted")
            seen.add(_fingerprint(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(grid, info):
    route, points = info["route"], info["points"]
    source = _footprint(points, route)
    copy = _translate(source, info["offset"])
    interior = set().union(*(set(_line(points[i], points[j])[1:-1]) for i, j in zip(route, route[1:])))
    interior_copy = _translate(interior, info["offset"])
    chosen = {tuple(sorted((i, j))) for i, j in zip(route, route[1:])}
    extra = set().union(*(set(line) for edge, line in info["edges"].items() if edge not in chosen)) if set(info["edges"]) - chosen else set()
    axes, lengths = [], []
    for i, j in zip(route, route[1:]):
        a, b = points[i], points[j]
        axes.append("diagonal" if a[0] != b[0] and a[1] != b[1] else "horizontal" if a[0] == b[0] else "vertical")
        lengths.append(max(abs(a[0]-b[0]), abs(a[1]-b[1])))
    return dict(shape=grid.shape, waypoints=len(points), route_length=len(route),
                unused=len(points)-len(route), labels=len(set(info["key"])),
                repeated=len(set(info["key"])) < len(info["key"]),
                offset=info["offset"], axes=tuple(sorted(set(axes))), lengths=tuple(lengths),
                walls=len(info["wall_components"]),
                wall_shapes=tuple(sorted((max(r for r, c in s)-min(r for r, c in s)+1,
                                          max(c for r, c in s)-min(c for r, c in s)+1)
                                         for s in info["wall_components"])),
                shadow_visible=sum(grid[p] == info["background"] for p in copy-source),
                copy_endpoint_cells=sum(grid[p] == info["background"] for p in copy-source-interior_copy),
                new_overlap=sum(grid[p] == info["background"] for p in copy & source),
                source_only_ambiguous=len(_routes(info, _edges(info, shadow=False), limit=2)) >= 2,
                nearest_fails=_greedy_source_route(info, nearest=True) != route,
                row_major_fails=_greedy_source_route(info, nearest=False) != route,
                crossing_rejection_needed=len(_routes(info, info["edges"], limit=2, simple=False)) >= 2,
                key_omission_cells=sum(grid[p] == info["background"] for p in extra-source),
                background=info["background"], wall_color=info["wall_color"],
                marker_color=info["marker_color"],
                layout=tuple((r-min(x[0] for x in points), c-min(x[1] for x in points)) for r, c in points))


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "invalid pair counts")
    seen, features = set(), {"train": [], "test": []}
    for split in puzzle:
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "canvas changed")
            fp = _fingerprint(inp)
            _require(fp not in seen, "duplicate input")
            seen.add(fp)
            expected, info = _solve_info(inp)
            _require(np.array_equal(expected, out), "incorrect route or shadow")
            _require(np.array_equal(inp[inp != info["background"]], out[inp != info["background"]]), "existing pixel changed")
            m = _features(inp, info)
            _require(m["shadow_visible"] >= 2, "companion translation has no visible evidence")
            _require(np.any(inp != out), "identity pair")
            features[split].append(m)
    train, test = features["train"], features["test"]
    _require(any(m["source_only_ambiguous"] and m["key_omission_cells"] > 0
                 and m["nearest_fails"] and m["row_major_fails"] for m in train),
             "training must distinguish both parent omissions")
    _require(any("diagonal" in m["axes"] and len(m["axes"]) >= 2 and m["new_overlap"] > 0 for m in train),
             "training must show mixed directions and source-over-shadow priority")
    _require(any(m["copy_endpoint_cells"] > 0 for m in train), "translated vertices must be demonstrated")
    _require(any(0 in m["offset"] for m in train) and any(0 not in m["offset"] for m in train),
             "training must show axial and diagonal displacement markers")
    _require(all(m["source_only_ambiguous"] and m["key_omission_cells"] > 0 for m in test),
             "queries must require both parent mechanisms")
    _require(all(m["nearest_fails"] and m["row_major_fails"] for m in test), "query fork shortcuts must fail")
    if any(m["crossing_rejection_needed"] for m in test):
        _require(any(m["crossing_rejection_needed"] for m in train),
                 "query crossing rejection must have a training counterexample")
    _require({m["offset"] for m in test}.isdisjoint({m["offset"] for m in train}),
             "test displacement must be held out")


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in a)


def self_test() -> None:
    """Bounded 42-seed coverage, count, determinism and semantic checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    metrics = []
    fingerprints = set()
    for p in puzzles:
        validate_generated_puzzle(p)
        fingerprints.add(tuple(_fingerprint(pair["input"]) for s in p for pair in p[s]))
        for split in p:
            for pair in p[split]:
                _, info = _solve_info(pair["input"])
                metrics.append(_features(pair["input"], info))
    assert len(fingerprints) == 42
    assert len({m["shape"] for m in metrics}) >= 35
    assert len({m["layout"] for m in metrics}) >= 80
    assert {m["offset"] for m in metrics} == set(OFFSETS)
    assert {m["background"] for m in metrics} == set(range(10))
    assert {m["wall_color"] for m in metrics} == set(range(10))
    assert {m["marker_color"] for m in metrics} == set(range(10))
    assert len({m["waypoints"] for m in metrics}) >= 6
    assert len({m["route_length"] for m in metrics}) >= 5
    assert len({m["walls"] for m in metrics}) >= 3
    assert len({s for m in metrics for s in m["wall_shapes"]}) >= 5
    assert any(m["repeated"] for m in metrics) and any(not m["repeated"] for m in metrics)
    assert any(len(set(m["lengths"])) >= 3 for m in metrics)
    assert _equal(puzzles[0], generate_puzzle(0))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(42+nt*5+nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Literal palette roles may exchange freely: every color bijection preserves
    # solving, and input/output lists work just like NumPy arrays.
    pair = puzzles[0]["train"][0]
    permutation = np.random.default_rng(123).permutation(10)
    assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
    for offset in OFFSETS:
        p = _fallback(np.random.default_rng(17), offset, True, True)
        expected, info = _solve_info(p["input"])
        assert np.array_equal(expected, p["output"])
        m = _features(p["input"], info)
        assert m["source_only_ambiguous"] and m["key_omission_cells"] > 0
        assert m["nearest_fails"] and m["row_major_fails"]
        assert m["new_overlap"] > 0 and m["copy_endpoint_cells"] > 0
    # A shadow-blind complete route must actually give a different answer, not
    # merely a second enumeration of the same segments.
    for p in puzzles:
        for pair in p["train"] + p["test"]:
            _, info = _solve_info(pair["input"])
            if not _features(pair["input"], info)["source_only_ambiguous"]:
                continue
            alternatives = _routes(info, _edges(info, shadow=False), limit=2)
            wrong = next(route for route in alternatives if route != info["route"])
            footprint = _footprint(info["points"], wrong)
            assert not footprint & info["walls"]
            assert _translate(footprint, info["offset"]) & info["walls"]
            assert not np.array_equal(_paint(pair["input"], info, wrong), pair["output"])
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    changed = tuple(np.argwhere(bad["test"][0]["input"] != bad["test"][0]["output"])[0])
    bad["test"][0]["output"][changed] = bad["test"][0]["input"][changed]
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("missing completion accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds and every legal count combination)")
