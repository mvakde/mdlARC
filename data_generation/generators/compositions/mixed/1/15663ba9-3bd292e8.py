"""Divider phase controls region fills and the labels of contour bends.

Only NumPy and the standard library are used. Generation, validation and tests
write no files, keep no cross-call state, and retain no execution traces.
"""
from __future__ import annotations

from collections import deque
import hashlib

import numpy as np


FILL = (3, 5)
BEND = (4, 2)  # index = (0 for convex, 1 for concave) XOR region phase
SOURCE_COLORS = (0, 1, 6, 7, 8, 9)
DIRS = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((a >= 0) & (a <= 9)), "grid colors must be 0..9")
    return a.astype(np.int8, copy=False)


def _neighbors(cell):
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in DIRS]


def _components(mask):
    labels = np.full(mask.shape, -1, dtype=np.int16)
    cells = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if labels[r, c] >= 0:
            continue
        index = len(cells)
        component = set()
        stack = [(r, c)]
        labels[r, c] = index
        while stack:
            current = stack.pop()
            component.add(current)
            for nr, nc in _neighbors(current):
                if (0 <= nr < h and 0 <= nc < w and mask[nr, nc]
                        and labels[nr, nc] < 0):
                    labels[nr, nc] = index
                    stack.append((nr, nc))
        cells.append(component)
    return labels, cells


def _boundary(cell, shape):
    return cell[0] in (0, shape[0] - 1) or cell[1] in (0, shape[1] - 1)


def _ordered(cells, closed):
    adjacent = {p: [q for q in _neighbors(p) if q in cells] for p in cells}
    ends = sorted(p for p, ns in adjacent.items() if len(ns) == 1)
    _require(all(len(ns) == 2 or (not closed and len(ns) == 1)
                 for ns in adjacent.values()), "path branches or touches itself")
    _require((closed and not ends and len(cells) >= 8)
             or (not closed and len(ends) == 2), "path has invalid endpoints")
    start = min(cells) if closed else ends[0]
    path, seen, previous, current = [], set(), None, start
    while current not in seen:
        seen.add(current)
        path.append(current)
        choices = [p for p in adjacent[current] if p != previous]
        if not choices:
            break
        previous, current = current, choices[0]
    _require(len(path) == len(cells), "path is not a single component")
    if closed:
        _require(current == start, "contour must close")
    else:
        _require(path[-1] == ends[1], "divider ordering failed")
    return path


def _inside(path, row, col):
    # Ray casting at noninteger coordinates, independent of turn orientation.
    crossings = 0
    for (r1, c1), (r2, c2) in zip(path, path[1:] + path[:1]):
        if c1 == c2 and min(r1, r2) < row < max(r1, r2) and c1 > col:
            crossings += 1
    return crossings % 2 == 1


def _loop(cells):
    path = _ordered(cells, True)
    area = sum(c * path[(i + 1) % len(path)][0]
               - r * path[(i + 1) % len(path)][1]
               for i, (r, c) in enumerate(path))
    _require(area != 0, "contour has no area")
    classes = {}
    for i, (r, c) in enumerate(path):
        pr, pc = path[i - 1]
        nr, nc = path[(i + 1) % len(path)]
        turn = (c - pc) * (nr - r) - (r - pr) * (nc - c)
        classes[(r, c)] = 0 if turn == 0 else (1 if turn * area > 0 else 2)
    convex = sum(k == 1 for k in classes.values())
    concave = sum(k == 2 for k in classes.values())
    _require(convex - concave == 4, "invalid contour turning number")
    r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
    r1, c1 = max(r for r, _ in cells), max(c for _, c in cells)
    # Padded flood fill also works for complete contours on the canvas edge.
    blocked = np.zeros((r1 - r0 + 3, c1 - c0 + 3), dtype=bool)
    for r, c in cells:
        blocked[r - r0 + 1, c - c0 + 1] = True
    labels, _ = _components(~blocked)
    interior = (~blocked) & (labels != labels[0, 0])
    _require(interior.any(), "contour must enclose at least one grid cell")
    inside_cells = {(int(r) + r0 - 1, int(c) + c0 - 1)
                    for r, c in np.argwhere(interior)}
    return {"cells": cells, "path": path, "classes": classes,
            "convex": convex, "concave": concave,
            "interior": inside_cells, "bbox": (r0, c0, r1, c1)}


def _wall_analysis(wall):
    wall_labels, paths = _components(wall)
    _require(paths and not wall[-1, 0], "dividers or open lower-left anchor missing")
    orders = []
    for cells in paths:
        path = _ordered(cells, False)
        _require(_boundary(path[0], wall.shape) and _boundary(path[-1], wall.shape),
                 "divider endpoints must be on the canvas boundary")
        orders.append(path)
        for r, c in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < wall.shape[0] and 0 <= nc < wall.shape[1]:
                        _require(not wall[nr, nc] or wall_labels[nr, nc] == wall_labels[r, c],
                                 "distinct dividers touch")
    regions, region_cells = _components(~wall)
    _require(len(region_cells) == len(paths) + 1,
             "each divider must add exactly one region")
    graph = [set() for _ in region_cells]
    edges = []
    for cells in paths:
        sides = set()
        for cell in cells:
            for r, c in _neighbors(cell):
                if 0 <= r < wall.shape[0] and 0 <= c < wall.shape[1] and regions[r, c] >= 0:
                    sides.add(int(regions[r, c]))
        _require(len(sides) == 2, "a divider must separate two nonempty regions")
        a, b = sorted(sides)
        _require(b not in graph[a], "duplicate region edge")
        graph[a].add(b)
        graph[b].add(a)
        edges.append((a, b))
    root = int(regions[-1, 0])
    depth = [-1] * len(graph)
    depth[root] = 0
    todo = deque([root])
    while todo:
        a = todo.popleft()
        for b in sorted(graph[a]):
            if depth[b] < 0:
                depth[b] = depth[a] + 1
                todo.append(b)
            else:
                _require((depth[b] - depth[a]) % 2 == 1, "region graph has inconsistent parity")
    _require(min(depth) >= 0 and len(edges) == len(graph) - 1,
             "region graph must be a connected tree")
    turns = []
    for path in orders:
        directions = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
        turns.append(sum(a != b for a, b in zip(directions, directions[1:])))
    return {"regions": regions, "region_cells": region_cells, "graph": graph,
            "depth": depth, "root": root, "paths": orders, "turns": turns}


def _analyze(grid):
    field = int(grid[-1, 0])
    _require(set(map(int, np.unique(grid))).issubset(SOURCE_COLORS),
             "input uses a reserved output color")
    labels, objects = _components(grid != field)
    _require(objects, "input contains no objects")
    wall = np.zeros(grid.shape, dtype=bool)
    loops = []
    for cells in objects:
        colors = {int(grid[p]) for p in cells}
        _require(len(colors) == 1, "each object must be monochrome")
        degrees = [sum(q in cells for q in _neighbors(p)) for p in cells]
        if 1 in degrees:
            path = _ordered(cells, False)
            _require(_boundary(path[0], grid.shape) and _boundary(path[-1], grid.shape),
                     "open paths must connect boundaries")
            for p in cells:
                wall[p] = True
        else:
            info = _loop(cells)
            info["color"] = colors.pop()
            loops.append(info)
        # Separation is by occupied cells, not by bounding boxes.
        for r, c in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                        _require(labels[nr, nc] < 0 or labels[nr, nc] == labels[r, c],
                                 "distinct objects touch")
    _require(loops, "input contains no contours")
    info = _wall_analysis(wall)
    for loop in loops:
        hosts = {int(info["regions"][p]) for p in loop["cells"] | loop["interior"]}
        _require(len(hosts) == 1 and min(hosts) >= 0, "contour crosses a divider")
        loop["region"] = hosts.pop()
        loop["phase"] = info["depth"][loop["region"]] % 2
    info.update({"wall": wall, "loops": loops, "field": field})
    return info


def _paint(grid, info, *, fixed_bends=False, no_concave=False):
    out = grid.copy()
    for region, depth in enumerate(info["depth"]):
        out[(grid == info["field"]) & (info["regions"] == region)] = FILL[depth % 2]
    for loop in info["loops"]:
        phase = 0 if fixed_bends else loop["phase"]
        for cell, kind in loop["classes"].items():
            if kind and not (no_concave and kind == 2):
                out[cell] = BEND[(kind - 1) ^ phase]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read all geometry from the input; four output colors are family constants."""
    grid = _grid(input_grid)
    return _paint(grid, _analyze(grid))


def _raster(vertices, closed=False):
    cells = []
    end_points = vertices[1:] + vertices[:1] if closed else vertices[1:]
    for a, b in zip(vertices, end_points):
        dr, dc = b[0] - a[0], b[1] - a[1]
        _require((dr == 0) != (dc == 0), "nonorthogonal or empty edge")
        n = abs(dr) + abs(dc)
        sr, sc = dr // n, dc // n
        cells.extend((a[0] + k * sr, a[1] + k * sc) for k in range(n))
    if not closed:
        cells.append(vertices[-1])
    _require(len(cells) == len(set(cells)), "self-intersecting raster path")
    return set(cells)


def _jog(vertices, rng, closed):
    index = int(rng.integers(len(vertices) if closed else len(vertices) - 1))
    a, b = vertices[index], vertices[(index + 1) % len(vertices)]
    length = abs(b[0] - a[0]) + abs(b[1] - a[1])
    if length < 4:
        return None
    u = int(rng.integers(1, length - 1))
    v = int(rng.integers(u + 1, length))
    depth = int(rng.integers(1, 4)) * int(rng.choice((-1, 1)))
    dr, dc = (b[0] - a[0]) // length, (b[1] - a[1]) // length
    p, q = (a[0] + u * dr, a[1] + u * dc), (a[0] + v * dr, a[1] + v * dc)
    middle = [(p[0] + depth * dc, p[1] - depth * dr),
              (q[0] + depth * dc, q[1] - depth * dr)]
    return vertices[:index + 1] + [p] + middle + [q] + vertices[index + 1:]


def _cut_corner(vertices, rng, closed):
    options = range(len(vertices)) if closed else range(1, len(vertices) - 1)
    options = list(options)
    if not options:
        return None
    i = options[int(rng.integers(len(options)))]
    a, p, b = vertices[i - 1], vertices[i], vertices[(i + 1) % len(vertices)]
    la, lb = abs(a[0] - p[0]) + abs(a[1] - p[1]), abs(b[0] - p[0]) + abs(b[1] - p[1])
    if min(la, lb) < 3:
        return None
    x, y = int(rng.integers(1, min(4, la))), int(rng.integers(1, min(4, lb)))
    q = (p[0] + x * (a[0] - p[0]) // la, p[1] + x * (a[1] - p[1]) // la)
    s = (p[0] + y * (b[0] - p[0]) // lb, p[1] + y * (b[1] - p[1]) // lb)
    t = (q[0] + s[0] - p[0], q[1] + s[1] - p[1])
    return vertices[:i] + [q, t, s] + vertices[i + 1:]


def _shape_from_vertices(vertices):
    cells = _raster(vertices, True)
    info = _loop(cells)
    r0, c0, r1, c1 = info["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    footprint = mask.copy()
    for r, c in cells:
        mask[r - r0, c - c0] = True
    for r, c in cells | info["interior"]:
        footprint[r - r0, c - c0] = True
    return mask, footprint, info["concave"]


def _sample_shape(rng, target=1, compact=False):
    high = 7 if compact else 12
    h, w = int(rng.integers(5, high)), int(rng.integers(5, high))
    if target == 0:
        h, w = int(rng.integers(3, high)), int(rng.integers(3, high))
        vertices = [(0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, 0)]
    else:
        a, b = int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2))
        vertices = [(0, 0), (0, w - 1), (h - 1, w - 1),
                    (h - 1, b), (a, b), (a, 0)]
    best = _shape_from_vertices(vertices)
    for _ in range(32):
        if best[2] >= target:
            break
        candidate = (_cut_corner(vertices, rng, True) if rng.random() < .55
                     else _jog(vertices, rng, True))
        if candidate is None:
            continue
        try:
            trial = _shape_from_vertices(candidate)
        except ValueError:
            continue
        if max(trial[0].shape) <= 13 and best[2] < trial[2] <= target + 1:
            vertices, best = candidate, trial
    mask, footprint, _ = best
    k, mirror = int(rng.integers(4)), bool(rng.integers(2))
    mask, footprint = np.rot90(mask, k), np.rot90(footprint, k)
    if mirror:
        mask, footprint = np.fliplr(mask), np.fliplr(footprint)
    return np.ascontiguousarray(mask), np.ascontiguousarray(footprint)


def _flip_path(path, h, w, corner):
    return [(h - 1 - r if corner in (2, 3) else r,
             w - 1 - c if corner in (1, 2) else c) for r, c in path]


def _skeleton(rng, mode):
    if mode == "anchor":
        h, w = int(rng.integers(26, 31)), int(rng.integers(26, 31))
        paths = []
        for corner in (0, 1, 2):
            r, c = int(rng.integers(8, 11)), int(rng.integers(8, 11))
            paths.append(_flip_path([(r, 0), (r, c), (0, c)], h, w, corner))
        return h, w, paths
    if mode in ("bands", "depth"):
        n = 2 if mode == "depth" else int(rng.integers(1, 4))
        h = int(rng.integers(23 if n == 2 else 29 if n == 3 else 15, 31))
        w = int(rng.integers(16, 31))
        base = (h - n) // (n + 1)
        sizes = [base] * (n + 1)
        for i in rng.permutation(n + 1)[:h - n - sum(sizes)]:
            sizes[int(i)] += 1
        rows, pos = [], -1
        for size in sizes[:-1]:
            pos += size + 1
            rows.append(pos)
        paths = [[(r, 0), (r, w - 1)] for r in rows]
        if bool(rng.integers(2)):
            paths = [[(c, r) for r, c in p] for p in paths]
            h, w = w, h
        return h, w, paths
    h, w = int(rng.integers(23, 31)), int(rng.integers(23, 31))
    if mode == "nested":
        n = int(rng.integers(2, 4))
        reaches = [6 + 8 * i for i in range(n)]
        corner = int(rng.integers(4))
        paths = [_flip_path([(r, 0), (r, r), (0, r)], h, w, corner) for r in reaches]
    elif mode == "u":
        a = int(rng.integers(1, 4))
        b = int(rng.integers(h - 5, h - 1))
        depth = int(rng.integers(8, w - 8))
        paths = [[(a, 0), (a, depth), (b, depth), (b, 0)]]
        if bool(rng.integers(2)):
            paths = [[(c, r) for r, c in p] for p in paths]
            h, w = w, h
        if bool(rng.integers(2)):
            paths = [[(r, w - 1 - c) for r, c in p] for p in paths]
    elif mode == "mixed":
        middle = h // 2
        paths = [[(middle, 0), (middle, w - 1)]]
        corners = [int(x) for x in rng.choice(4, int(rng.integers(1, 4)), replace=False)]
        for corner in corners:
            r, c = int(rng.integers(7, 10)), int(rng.integers(7, 10))
            paths.append(_flip_path([(r, 0), (r, c), (0, c)], h, w, corner))
    else:  # Corner cuts can induce a genuinely branching region graph.
        corners = [int(x) for x in rng.choice(4, int(rng.integers(2, 5)), replace=False)]
        paths = []
        for corner in corners:
            r, c = int(rng.integers(7, 10)), int(rng.integers(7, 10))
            paths.append(_flip_path([(r, 0), (r, c), (0, c)], h, w, corner))
    return h, w, paths


def _wall_mask(paths, h, w):
    wall = np.zeros((h, w), dtype=bool)
    occupied = set()
    for vertices in paths:
        cells = _raster(vertices)
        _require(not (cells & occupied), "overlapping divider paths")
        occupied.update(cells)
        for r, c in cells:
            _require(0 <= r < h and 0 <= c < w, "divider leaves canvas")
            wall[r, c] = True
    return wall


def _make_walls(rng, mode, richer):
    h, w, paths = _skeleton(rng, mode)
    wall = _wall_mask(paths, h, w)
    info = _wall_analysis(wall)
    target_edits = int(rng.integers(1 if richer else 0, 5))
    if mode in ("anchor", "depth"):
        target_edits = 0
    edits = 0
    for _ in range(24):
        if edits >= target_edits:
            break
        index = int(rng.integers(len(paths)))
        candidate = (_cut_corner(paths[index], rng, False) if rng.random() < .5
                     else _jog(paths[index], rng, False))
        if candidate is None:
            continue
        changed = paths.copy()
        changed[index] = candidate
        try:
            trial = _wall_mask(changed, h, w)
            trial_info = _wall_analysis(trial)
        except ValueError:
            continue
        paths, wall, info = changed, trial, trial_info
        edits += 1
    if richer:
        _require(max(info["turns"]) >= 2, "test must recombine nontrivial divider geometry")
    return wall, info


def _halo(mask):
    padded = np.pad(mask, 1)
    return np.lib.stride_tricks.sliding_window_view(padded, (3, 3)).any(axis=(-1, -2))


def _place(grid, info, blocked, shape, region, color, rng):
    mask, footprint = shape
    sh, sw = mask.shape
    if sh > grid.shape[0] or sw > grid.shape[1]:
        return False
    allowed = (info["regions"] == region) & ~blocked
    windows = np.lib.stride_tricks.sliding_window_view(allowed, (sh, sw))
    feasible = np.argwhere(windows[..., footprint].all(axis=-1))
    if not len(feasible):
        return False
    r, c = map(int, feasible[int(rng.integers(len(feasible)))])
    grid[r:r + sh, c:c + sw][mask] = color
    occupied = np.zeros(grid.shape, dtype=bool)
    occupied[r:r + sh, c:c + sw] = footprint
    blocked |= _halo(occupied)
    return True


def _add_loop(grid, info, blocked, regions, color, rng, target=1):
    for attempt in range(24):
        shape = _sample_shape(rng, target, compact=attempt >= 8)
        for region in rng.permutation(regions):
            if _place(grid, info, blocked, shape, int(region), color, rng):
                return True
    return False


def _make_pair(rng, mode, palette, test=False):
    for _ in range(36):
        try:
            wall, info = _make_walls(rng, mode, test)
        except ValueError:
            continue
        field = int(rng.choice(palette[:3]))
        divider = int(rng.choice([c for c in palette[:3] if c != field]))
        color = int(palette[5] if test else rng.choice(palette[3:5]))
        grid = np.full(wall.shape, field, dtype=np.int8)
        grid[wall] = divider
        blocked = _halo(wall)
        # Keep the anchor and the phase-demonstration corner cells visible.
        for p in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
            blocked[p] = True
        even = [i for i, d in enumerate(info["depth"]) if d % 2 == 0]
        odd = [i for i, d in enumerate(info["depth"]) if d % 2]
        if mode == "anchor":
            # Identical inscriptions in both phases directly expose the coupling.
            shape = _sample_shape(rng, 1, compact=True)
            success = _place(grid, info, blocked, shape, info["root"], color, rng)
            success &= any(_place(grid, info, blocked, shape, int(r), color, rng)
                           for r in rng.permutation(odd))
            if not success:
                continue
        elif mode == "depth":
            depth_two = [i for i, d in enumerate(info["depth"]) if d == 2]
            if not _add_loop(grid, info, blocked, depth_two, color, rng):
                continue
            if not _add_loop(grid, info, blocked, odd, color, rng):
                continue
        else:
            # Every test contains a held-out-color irregular contour at odd phase.
            if not _add_loop(grid, info, blocked, odd, color, rng,
                             target=int(rng.integers(2, 7)) if test else 1):
                continue
            if rng.random() < .85:
                _add_loop(grid, info, blocked, even, int(rng.choice(palette[3:6] if test else palette[3:5])),
                          rng, int(rng.integers(0, 5)))
        for _ in range(int(rng.integers(0, 5))):
            _add_loop(grid, info, blocked, list(range(len(info["depth"]))),
                      int(rng.choice(palette[3:6] if test else palette[3:5])), rng,
                      int(rng.integers(0, 5)))
        analysis = _analyze(grid)
        return {"input": grid, "output": _paint(grid, analysis)}
    raise RuntimeError("bounded construction attempts exhausted")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and int(seed) >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = [int(x) for x in rng.permutation(SOURCE_COLORS)]
    modes = ("bands", "corners", "mixed", "nested", "u")
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            mode = ("anchor" if index == 0 else "depth") if split == "train" and index < 2 else str(rng.choice(modes))
            for _ in range(4):
                pair = _make_pair(rng, mode, palette, test=split == "test")
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct a unique input")
    validate_generated_puzzle(puzzle)
    return puzzle


def _relative(loop):
    r0, c0, _, _ = loop["bbox"]
    return frozenset((r - r0, c - c0) for r, c in loop["cells"])


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _count(a, "num_train", 2, 10)
    _count(b, "num_test", 1, 4)
    _require(a > b, "training must outnumber tests")
    seen, train_infos, test_infos = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            grid, out = _grid(pair["input"]), _grid(pair["output"])
            _require(grid.shape == out.shape, "dimensions changed")
            key = (grid.shape, grid.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _analyze(grid)
            _require(np.array_equal(out, _paint(grid, info)), "output disagrees with the rule")
            _require(np.array_equal(out[info["wall"]], grid[info["wall"]]), "divider changed")
            for i, depth in enumerate(info["depth"]):
                field_cells = (info["regions"] == i) & (grid == info["field"])
                _require(field_cells.any() and np.all(out[field_cells] == FILL[depth % 2]),
                         "region field not filled uniformly")
            for loop in info["loops"]:
                for cell, kind in loop["classes"].items():
                    if not kind:
                        _require(out[cell] == grid[cell], "straight contour cell changed")
                    else:
                        r, c = cell
                        quadrants = sum(_inside(loop["path"], r + dr, c + dc)
                                        for dr in (-.25, .25) for dc in (-.25, .25))
                        _require(quadrants == (1 if kind == 1 else 3), "bend interior angle mismatch")
                for cell in loop["interior"]:
                    if grid[cell] == info["field"]:
                        _require(out[cell] == FILL[loop["phase"]], "loop incorrectly toggles fill")
            (train_infos if split == "train" else test_infos).append(info)
    first = train_infos[0]
    first_grid = _grid(puzzle["train"][0]["input"])
    for cell in ((0, 0), (0, first_grid.shape[1] - 1), (first_grid.shape[0] - 1, first_grid.shape[1] - 1)):
        _require(first_grid[cell] == first["field"] and first["depth"][int(first["regions"][cell])] % 2 == 1,
                 "training does not identify the lower-left anchor")
    _require(any(x["phase"] != y["phase"] and x["concave"] and x["color"] == y["color"]
                 and _relative(x) == _relative(y)
                 for x in first["loops"] for y in first["loops"]),
             "training lacks matching irregular contours in opposite phases")
    _require(any(info["depth"][loop["region"]] >= 2 and loop["phase"] == 0 and loop["concave"]
                 for info in train_infos for loop in info["loops"]), "missing even nonanchor bend witness")
    for pair, info in zip(puzzle["train"], train_infos):
        _require(not np.array_equal(_paint(_grid(pair["input"]), info, fixed_bends=True), pair["output"]),
                 "B does not affect bend labels in this demonstration")
        _require(not np.array_equal(_paint(_grid(pair["input"]), info, no_concave=True), pair["output"]),
                 "A's concave classification has no witness")
    train_colors = {loop["color"] for info in train_infos for loop in info["loops"]}
    for info in test_infos:
        _require(max(info["turns"]) >= 2, "test lacks divider recombination")
        _require(any(loop["color"] not in train_colors and loop["phase"] == 1 and loop["concave"]
                     for loop in info["loops"]), "missing held-out-color odd-phase irregular contour")


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(split.encode())
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = _grid(pair[role])
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded 42-seed audit; only hashes and small aggregate sets are retained."""
    shapes, structures, palettes, counts, turns, depths, degrees, concaves, patterns = (set() for _ in range(9))
    signatures = set()
    pairs = objects = 0
    for seed in range(42):
        a = 2 + seed % 9
        b = 1 + seed % min(4, a - 1)
        puzzle = generate_puzzle(seed, a, b)
        signatures.add(_fingerprint(puzzle))
        for pair in puzzle["train"] + puzzle["test"]:
            grid = pair["input"]
            info = _analyze(grid)
            pairs += 1
            shapes.add(grid.shape)
            counts.add((len(info["paths"]), len(info["loops"])))
            turns.update(info["turns"])
            depths.update(info["depth"])
            degrees.update(map(len, info["graph"]))
            palettes.add((info["field"], tuple(sorted(set(map(int, np.unique(grid)))))))
            # Color-normalized whole-input hashes measure structure, not palettes.
            normalized = np.zeros_like(grid)
            normalized[info["wall"]] = 1
            for loop in info["loops"]:
                for p in loop["cells"]:
                    normalized[p] = 2
                objects += 1
                concaves.add(loop["concave"])
                pattern = [loop["classes"][p] for p in loop["path"] if loop["classes"][p]]
                variants = [tuple(v[i:] + v[:i]) for v in (pattern, pattern[::-1]) for i in range(len(v))]
                patterns.add(min(variants))
            structures.add((grid.shape, hashlib.sha256(normalized.tobytes()).hexdigest()))
    _require(len(signatures) == 42 and len(shapes) >= 50 and len(structures) >= 150,
             "insufficient seed or geometry variation")
    _require(len(palettes) >= 20 and len(counts) >= 12 and max(turns) >= 5,
             "insufficient palette, object-count or divider-turn variation")
    _require(max(depths) >= 3 and max(degrees) >= 3 and max(concaves) >= 4 and len(patterns) >= 8,
             "missing graph or contour complexity")
    _require(_fingerprint(generate_puzzle(777, 5, 2)) == _fingerprint(generate_puzzle(777, 5, 2)),
             "whole-puzzle determinism failed")
    # All legal count combinations, streamed one puzzle at a time.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            generate_puzzle(12345, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": pairs, "contours": objects,
            "distinct_puzzles": len(signatures), "shapes": len(shapes),
            "structures": len(structures), "input_palettes": len(palettes),
            "divider_contour_count_combinations": sorted(counts),
            "divider_turn_counts": sorted(turns), "crossing_depths": sorted(depths),
            "region_degrees": sorted(degrees), "concave_counts": sorted(concaves),
            "contour_turn_patterns": len(patterns)}
