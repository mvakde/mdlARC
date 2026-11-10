"""Contour clocks: a local component census becomes a cyclic band-width code.

Only NumPy and the standard library are used. The solver reads input pixels;
the generator constructs outputs from its sampled census, then checks those
outputs by reconstructing the census and geometry from the returned inputs.
"""
from collections import deque
import json

import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
MARKER = 9


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    return int(value)


def _counts(num_train, num_test):
    a, b = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("Require 2..10 training, 1..4 test, and num_train > num_test")
    return a, b


def _grid(value):
    a = np.asarray(value)
    _check(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _check(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _check(np.all((0 <= a) & (a <= 9)), "color range")
    return a.astype(np.int8, copy=False)


def _components(mask, neighbors=D4):
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for rr, cc in np.argwhere(mask):
        r, c = int(rr), int(cc)
        if seen[r, c]:
            continue
        seen[r, c] = True
        queue, cells = deque([(r, c)]), []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        result.append(tuple(cells))
    return result


def _mask(shape, cells):
    a = np.zeros(shape, dtype=bool)
    for r, c in cells:
        a[r, c] = True
    return a


def _depth_bfs(region):
    h, w = region.shape
    depth = np.full(region.shape, -1, dtype=np.int16)
    q = deque()
    for rr, cc in np.argwhere(region):
        r, c = int(rr), int(cc)
        if any(not (0 <= r + dr < h and 0 <= c + dc < w and region[r + dr, c + dc])
               for dr, dc in D8):
            depth[r, c] = 0
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in D8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and region[nr, nc] and depth[nr, nc] < 0:
                depth[nr, nc] = depth[r, c] + 1
                q.append((nr, nc))
    return depth


def _depth_erosion(region, neighbors=D8):
    """Separate vectorized erosion implementation for validation."""
    h, w = region.shape
    current = region.copy()
    depth = np.full(region.shape, -1, dtype=np.int16)
    level = 0
    while current.any():
        padded = np.pad(current, 1)
        interior = current.copy()
        for dr, dc in neighbors:
            interior &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        depth[current & ~interior] = level
        current = interior
        level += 1
    return depth


def _dilate(mask):
    h, w = mask.shape
    padded = np.pad(mask, 1)
    a = mask.copy()
    for dr, dc in D8:
        a |= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return a


def _roles(inp):
    frequencies = np.bincount(inp.ravel(), minlength=10)
    bg = int(np.argmax(frequencies))
    _check(np.count_nonzero(frequencies == frequencies[bg]) == 1 and bg != MARKER,
           "background must be the unique non-marker mode")
    candidates = []
    for color in range(9):
        if color == bg:
            continue
        m = inp == color
        edges = sum(bool(x.any()) for x in (m[0], m[-1], m[:, 0], m[:, -1]))
        if edges >= 2 and len(_components(m)) == 1:
            candidates.append(color)
    _check(len(candidates) == 1, "one connected wall color must meet two canvas sides")
    wall = candidates[0]
    m = inp == wall
    _check(not np.any(m[:-1, :-1] & m[1:, :-1] & m[:-1, 1:] & m[1:, 1:]),
           "walls must be one cell thick")
    return bg, wall


def _inspect(inp, depth_fn=_depth_bfs):
    bg, wall = _roles(inp)
    clocks = []
    all_objects = []
    for cells in _components(inp != wall):
        region = _mask(inp.shape, cells)
        depth = depth_fn(region)
        delimiters = [tuple(map(int, p)) for p in np.argwhere(region & (inp == MARKER))]
        key_cells, colors, direction = (), (), None
        if delimiters:
            _check(len(delimiters) == 2, "exactly two delimiters per keyed chamber")
            start, end = sorted(delimiters, key=lambda p: int(depth[p]))
            _check(depth[start] == 0, "key must start on contour zero")
            length = int(depth[end]) - 1
            _check(1 <= length <= 3, "key length must be 1..3")
            delta = (end[0] - start[0], end[1] - start[1])
            _check(abs(delta[0]) == abs(delta[1]) == length + 1, "key must be diagonal")
            dr, dc = (1 if x > 0 else -1 for x in delta)
            direction = (dr, dc)
            key_cells = tuple((start[0] + d * dr, start[1] + d * dc)
                              for d in range(length + 2))
            _check(all(region[p] and depth[p] == d for d, p in enumerate(key_cells)),
                   "key diagonal must advance one contour per position")
            colors = tuple(int(inp[p]) for p in key_cells[1:-1])
            _check(len(set(colors)) == length and not set(colors) & {bg, wall, MARKER},
                   "categories must be distinct and separate from roles")
        key = _mask(inp.shape, key_cells)
        object_mask = region & (inp != bg) & ~key
        _check(not np.any(object_mask & _dilate(key)), "object touches a key")
        objects = []
        labels = np.full(inp.shape, -1, dtype=np.int16)
        for color in range(9):
            if color == bg:
                continue
            for points in _components(object_mask & (inp == color)):
                _check(all(depth[p] >= 1 for p in points), "objects must clear walls and exterior")
                label = len(objects)
                for p in points:
                    labels[p] = label
                objects.append({"color": color, "cells": points, "area": len(points)})
        for a, b in ((labels[:-1, :], labels[1:, :]), (labels[:, :-1], labels[:, 1:])):
            _check(not np.any((a >= 0) & (b >= 0) & (a != b)), "different objects edge-touch")
        counts = tuple(sum(o["color"] == color for o in objects) for color in colors)
        areas = tuple(sum(o["area"] for o in objects if o["color"] == color) for color in colors)
        period = tuple(color for color, count in zip(colors, counts) for _ in range(count))
        rows, cols = np.nonzero(region)
        concave = int(region.sum()) != (int(rows.max() - rows.min()) + 1) * (int(cols.max() - cols.min()) + 1)
        clocks.append(dict(region=region, depth=depth, key_cells=key_cells, colors=colors,
                           direction=direction, objects=objects, counts=counts, areas=areas,
                           period=period, concave=concave))
        all_objects.extend(objects)
    return dict(background=bg, wall=wall, wall_mask=inp == wall,
                clocks=clocks, objects=all_objects, shape=inp.shape)


def _render(info):
    out = np.full(info["shape"], info["background"], dtype=np.int8)
    out[info["wall_mask"]] = info["wall"]
    for clock in info["clocks"]:
        if clock["period"]:
            region, depth = clock["region"], clock["depth"]
            period = np.array(clock["period"], dtype=np.int8)
            out[region] = period[depth[region] % len(period)]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the complete result from this input alone, without modifying it."""
    return _render(_inspect(_grid(input_grid)))


def _paths(region, depth, dr):
    h, w = region.shape
    paths = []
    def inside(r, c):
        return 0 <= r < h and 0 <= c < w and region[r, c]
    for rr, cc in np.argwhere(region & (depth == 0)):
        r, c = int(rr), int(cc)
        for dc in (-1, 1):
            if inside(r - dr, c) or inside(r, c - dc):
                continue
            if not (inside(r + dr, c) and inside(r, c + dc)):
                continue
            n = 0
            while inside(r + n * dr, c + n * dc) and depth[r + n * dr, c + n * dc] == n:
                n += 1
            if n >= 3:
                paths.append((r, c, dr, dc, n))
    return paths


def _partition(rng, shape, target):
    h, w = shape
    walls = np.zeros(shape, dtype=bool)
    rooms = [(0, h - 1, 0, w - 1)]
    while len(rooms) < target:
        choices = [(i, axis) for i, (a, b, c, d) in enumerate(rooms)
                   for axis, extent in ((0, b - a + 1), (1, d - c + 1)) if extent >= 11]
        if not choices:
            break
        weights = np.array([(rooms[i][1] - rooms[i][0] + 1) *
                            (rooms[i][3] - rooms[i][2] + 1) for i, _ in choices], dtype=float)
        i, axis = choices[int(rng.choice(len(choices), p=weights / weights.sum()))]
        a, b, c, d = rooms.pop(i)
        if axis == 0:
            cut = int(rng.integers(a + 5, b - 4))
            walls[cut, max(0, c - 1):min(w, d + 2)] = True
            rooms.extend([(a, cut - 1, c, d), (cut + 1, b, c, d)])
        else:
            cut = int(rng.integers(c + 5, d - 4))
            walls[max(0, a - 1):min(h, b + 2), cut] = True
            rooms.extend([(a, b, c, cut - 1), (a, b, cut + 1, d)])
    return walls, rooms


def _box(rng, walls, rooms):
    choices = []
    for a, b, c, d in rooms:
        for transpose in (False, True):
            aa, bb, cc, dd = (c, d, a, b) if transpose else (a, b, c, d)
            work = walls.T if transpose else walls
            if bb - aa < 10 or dd - cc < 9:
                continue
            for side in (-1, 1):
                column = cc - 1 if side == 1 else dd + 1
                if 0 <= column < work.shape[1] and np.all(work[aa:bb + 1, column]):
                    choices.append((transpose, aa, bb, cc, dd, side, column))
    if not choices:
        return
    transpose, a, b, c, d, side, wall = choices[int(rng.integers(len(choices)))]
    work = walls.T if transpose else walls
    top = int(rng.integers(a + 2, b - 7))
    bottom = int(rng.integers(top + 6, b - 1))
    far = wall + side * int(rng.integers(6, min(9, d - c - 1) + 1))
    lo, hi = sorted((wall, far))
    work[top, lo:hi + 1] = work[bottom, lo:hi + 1] = True
    work[top:bottom + 1, far] = True


def _steps(rng, walls, budget):
    for _ in range(budget):
        for _ in range(35):
            work = walls.T if rng.integers(2) else walls
            points = np.argwhere(work)
            r, c = map(int, points[int(rng.integers(len(points)))])
            end = r + int(rng.integers(4, 10))
            shift = int(rng.choice((-3, -2, 2, 3)))
            lo, hi = sorted((c, c + shift))
            if r < 1 or end + 1 >= work.shape[0] or lo < 1 or hi + 1 >= work.shape[1]:
                continue
            block = work[r - 1:end + 2, lo - 1:hi + 2]
            expected = np.zeros(block.shape, dtype=bool)
            expected[:, c - lo + 1] = True
            if not np.array_equal(block, expected):
                continue
            work[r + 1:end, c] = False
            work[r, lo:hi + 1] = work[end, lo:hi + 1] = True
            work[r:end + 1, c + shift] = True
            break


def _geometry(rng, teaching, test):
    if teaching:
        h, w = int(rng.integers(24, 31)), int(rng.integers(28, 31))
        cut = int(rng.integers(17, w - 7))
        walls = np.zeros((h, w), dtype=bool)
        walls[:, cut] = True
        if teaching == 1 or rng.random() < .65:
            # Leave both upper diagonals long enough for the five-position
            # teaching keys before the wall begins its indentation.
            top = int(rng.integers(9, 12))
            bottom = int(rng.integers(max(top + 4, h - 8), h - 3))
            shift = int(rng.integers(2, 4))
            walls[top + 1:bottom, cut] = False
            walls[top, cut - shift:cut + 1] = walls[bottom, cut - shift:cut + 1] = True
            walls[top:bottom + 1, cut - shift] = True
        if teaching == 2:
            # A third chamber teaches the absence of a local legend even with
            # the minimum two demonstrations, and changes the room count.
            row = h // 2
            junction = int(np.flatnonzero(walls[row])[0])
            walls[row, junction:] = True
    else:
        low = 19 if test else 14
        h, w = int(rng.integers(low, 31)), int(rng.integers(low, 31))
        target = int(rng.integers(2, 7 if test else 6))
        walls, rooms = _partition(rng, (h, w), target)
        if rng.random() < (.55 if test else .35):
            _box(rng, walls, rooms)
        _steps(rng, walls, int(rng.integers(0, 5 if test else 4)))
        if rng.integers(2):
            walls = walls.T.copy()
    if rng.integers(2):
        walls = walls[:, ::-1].copy()
    return walls


def _shape(rng, area=None):
    if area is None:
        area = int(rng.choice((1, 2, 3, 4, 5, 6, 7, 8, 10),
                              p=(.15, .18, .18, .16, .10, .09, .06, .05, .03)))
    kind = int(rng.integers(4))
    if kind == 0 and area <= 6:
        cells = {(0, i) for i in range(area)}
    elif kind == 1:
        width = min(area, int(rng.integers(2, 5)))
        cells = {(i // width, i % width) for i in range(area)}
    elif kind == 2 and area <= 7:
        arm = int(rng.integers(1, area + 1))
        cells = {(0, i) for i in range(arm)} | {(i, 0) for i in range(area - arm + 1)}
    else:
        cells = {(0, 0)}
        while len(cells) < area:
            choices = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in D4}
                             - cells)
            # Keep arbitrary growth compact enough for the smallest chambers.
            choices = [(r, c) for r, c in choices if
                       max(r, max(x[0] for x in cells)) - min(r, min(x[0] for x in cells)) < 4
                       and max(c, max(x[1] for x in cells)) - min(c, min(x[1] for x in cells)) < 4]
            cells.add(choices[int(rng.integers(len(choices)))])
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    if rng.integers(2):
        cells = {(r, -c) for r, c in cells}
    a, b = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r - a, c - b) for r, c in cells))


def _pack(rng, inp, allowed, items):
    occupied = ~allowed.copy()
    items = sorted(items, key=lambda x: (-len(x[1]), float(rng.random())))
    h, w = inp.shape
    for color, cells in items:
        sh, sw = max(r for r, c in cells) + 1, max(c for r, c in cells) + 1
        if sh > h or sw > w:
            return False
        candidates = np.ones((h - sh + 1, w - sw + 1), dtype=bool)
        for dr, dc in cells:
            candidates &= ~occupied[dr:dr + h - sh + 1, dc:dc + w - sw + 1]
        positions = np.argwhere(candidates)
        if not len(positions):
            return False
        r0, c0 = map(int, positions[int(rng.integers(len(positions)))])
        placed = np.zeros(inp.shape, dtype=bool)
        for dr, dc in cells:
            inp[r0 + dr, c0 + dc] = color
            placed[r0 + dr, c0 + dc] = True
        occupied |= _dilate(placed)
    return True


def _make_pair(rng, bg, wall, anchors, novel, teaching, test, extrapolate):
    walls = _geometry(rng, teaching, test)
    shape = walls.shape
    regions = [_mask(shape, cells) for cells in _components(~walls)]
    depths = [_depth_bfs(r) for r in regions]
    dr = -1 if test else 1
    options = [_paths(r, d, dr) for r, d in zip(regions, depths)]
    capacities = [max((p[4] for p in paths), default=0) for paths in options]
    main = max(range(len(regions)), key=lambda i: (capacities[i], int(regions[i].sum())))
    if capacities[main] < (5 if teaching else 4 if test else 3):
        return None
    inp = np.full(shape, bg, dtype=np.int8)
    out = inp.copy()
    inp[walls] = out[walls] = wall
    x, y, z = anchors
    secondary = max((i for i in range(len(regions)) if i != main), key=lambda i: capacities[i])
    for i, (region, depth, paths, capacity) in enumerate(zip(regions, depths, options, capacities)):
        max_key = min(3, capacity - 2)
        key_colors, census, forced, diagonal = [], [], {}, False
        if teaching and i == main:
            key_colors = [x, y, z] if teaching == 1 else [z, y, x]
            census = [1, 2, 0] if teaching == 1 else [1, 1, 2]
            forced = {x: [6], y: [2, 2]} if teaching == 1 else {z: [1], y: [4], x: [3, 3]}
            diagonal = teaching == 2
        elif teaching and i == secondary:
            if max_key < 2:
                return None
            key_colors = [x, y] if teaching == 1 else [y, z]
            census = [2, 1] if teaching == 1 else [0, 0]
            forced = {x: [1, 1], y: [3]} if teaching == 1 else {}
        elif teaching:
            # The remaining teaching chamber has no category legend.
            pass
        elif max_key >= 1 and (i == main or rng.random() < .80):
            length = int(rng.integers(2 if test and i == main else 1, max_key + 1))
            pool = [c for c in range(9) if c not in (bg, wall) and (test or c != novel)]
            if test and i == main:
                pool.remove(novel)
                key_colors = [novel] + [int(c) for c in rng.choice(pool, length - 1, replace=False)]
            else:
                key_colors = [int(c) for c in rng.choice(pool, length, replace=False)]
            census = [int(rng.integers(0, 4 if test else 3)) for _ in key_colors]
            if rng.random() < .17:
                census = [0] * length
            if test and i == main:
                census[0] = int(rng.integers(3, 5)) if extrapolate else int(rng.integers(1, 3))
                census[1] = int(rng.integers(1, 3))
            while sum(census) > 7:
                census[int(np.argmax(census))] -= 1
        key_cells = ()
        if key_colors:
            valid = [p for p in paths if p[4] >= len(key_colors) + 2]
            if teaching and i == main:
                # Both upper corner directions are taught before upward tests.
                preferred = [p for p in valid if p[3] == (1 if teaching == 1 else -1)]
                valid = preferred
            if not valid:
                return None
            r, c, dr, dc, _ = valid[int(rng.integers(len(valid)))]
            key_cells = tuple((r + d * dr, c + d * dc) for d in range(len(key_colors) + 2))
            for p, color in zip(key_cells, [MARKER] + key_colors + [MARKER]):
                inp[p] = color
        period = tuple(color for color, count in zip(key_colors, census) for _ in range(count))
        if period:
            out[region] = np.asarray(period, dtype=np.int8)[depth[region] % len(period)]
        items = []
        for color, count in zip(key_colors, census):
            if diagonal and color == x:
                # Two horizontal triominoes meet diagonally, never by an edge.
                group = ((0, 0), (0, 1), (0, 2), (1, 3), (1, 4), (1, 5))
                if rng.integers(2):
                    group = tuple((c, r) for r, c in group)
                items.append((color, group))
            else:
                areas = forced.get(color, [None] * count)
                items.extend((color, _shape(rng, area)) for area in areas)
        distractor_pool = [c for c in range(9) if c not in (bg, wall) and c not in key_colors]
        if teaching and (i == main or (teaching == 2 and i == secondary)):
            items.append((novel, _shape(rng, int(rng.integers(1, 5)))))
        elif teaching and not key_colors:
            items.append((x, _shape(rng, 2)))
        elif distractor_pool:
            for _ in range(int(rng.integers(0, 3))):
                items.append((int(rng.choice(distractor_pool)), _shape(rng)))
        allowed = region & (depth >= 1) & ~_dilate(_mask(shape, key_cells))
        if not _pack(rng, inp, allowed, items):
            return None
    return {"input": inp, "output": out}


def _different_depths(clock):
    if not clock["concave"] or len(set(clock["period"])) < 2:
        return False
    region = clock["region"]
    rows, cols = np.nonzero(region)
    box = np.minimum.reduce((rows - rows.min(), rows.max() - rows,
                             cols - cols.min(), cols.max() - cols))
    period = np.array(clock["period"], dtype=np.int8)
    actual = period[clock["depth"][region] % len(period)]
    return bool(np.any(actual != period[box % len(period)]) and
                np.any(actual != period[_depth_erosion(region, D4)[region] % len(period)]))


def _teaching_evidence(details):
    clocks = [c for d in details for c in d["clocks"]]
    census = [(color, count, area) for c in clocks
              for color, count, area in zip(c["colors"], c["counts"], c["areas"])]
    collision = any(color == other and count != n and area == a and count > 0 and n > 0
                    for color, count, area in census for other, n, a in census)
    independent = any(count == n == 2 and area != a for _, count, area in census for _, n, a in census)
    diagonal = any(len(_components(c["region"] & (d_input == color) &
                      ~_mask(d_input.shape, c["key_cells"]), D8)) < count
                   for d, d_input in details_with_inputs(details)
                   for c in d["clocks"] for color, count in zip(c["colors"], c["counts"]))
    return collision, independent, diagonal


def details_with_inputs(details):
    # Validator attaches only the input it is currently inspecting; this is not
    # solver state and is never read by solve().
    return ((d, d["input"]) for d in details)


def _local_witness(detail):
    global_counts = {color: sum(o["color"] == color for o in detail["objects"]) for color in range(9)}
    for c in detail["clocks"]:
        if not c["period"]:
            continue
        global_period = tuple(color for color in c["colors"] for _ in range(global_counts[color]))
        if global_period:
            p, g = np.array(c["period"]), np.array(global_period)
            depths = c["depth"][c["region"]]
            if np.any(p[depths % len(p)] != g[depths % len(g)]):
                return True
    return False


def _uniform_witness(clock):
    if not clock["period"]:
        return False
    period, key = np.array(clock["period"]), np.array(clock["colors"])
    depths = clock["depth"][clock["region"]]
    return bool(np.any(period[depths % len(period)] != key[depths % len(key)]))


def _input_key(grid):
    return tuple(grid.shape), grid.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    nt, ne = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    colors = [int(x) for x in rng.permutation(9)]
    anchors, novel = colors[:3], colors[3]
    extrapolate = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    previous_bg = None
    for index in range(nt + ne):
        test = index >= nt
        teaching = index + 1 if index < 2 else 0
        forbidden = set(anchors + [novel]) if teaching else {novel} if test else set()
        bg_pool = [c for c in range(9) if c not in forbidden and c != previous_bg]
        bg = int(rng.choice(bg_pool))
        wall = int(rng.choice([c for c in range(9) if c not in forbidden and c != bg]))
        for _attempt in range(80):
            pair = _make_pair(rng, bg, wall, anchors, novel, teaching, test, extrapolate)
            if pair is None or _input_key(pair["input"]) in seen:
                continue
            try:
                detail = _inspect(pair["input"])
                _check(np.array_equal(_render(detail), pair["output"]), "sampled census changed in packing")
                if teaching == 1:
                    _check(any(_different_depths(c) for c in detail["clocks"]), "contour witness")
                    _check(_local_witness(detail), "local counting witness")
            except AssertionError:
                continue
            break
        else:
            raise RuntimeError("bounded scene construction exhausted")
        seen.add(_input_key(pair["input"]))
        puzzle["test" if test else "train"].append(pair)
        previous_bg = bg
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _check(all(isinstance(puzzle[s], list) for s in puzzle), "split schema")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, details = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _check(inp.shape == out.shape, "dimensions changed")
            key = _input_key(inp)
            _check(key not in seen, "duplicate inputs")
            seen.add(key)
            d = _inspect(inp, _depth_erosion)
            _check(np.array_equal(out, _render(d)), "wrong weighted contours or erasure")
            _check(np.array_equal(out == d["wall"], d["wall_mask"]), "wall cells changed")
            _check(not np.any(out == MARKER), "key delimiter survived")
            _check(not np.array_equal(inp, out), "identity example")
            _check(all(1 <= o["area"] <= 10 for o in d["objects"]), "object area range")
            d["input"] = inp
            details[split].append(d)
    train, test = details["train"], details["test"]
    collision, independent, diagonal = _teaching_evidence(train)
    _check(collision and independent, "area-independent counting evidence missing")
    _check(diagonal, "four-connected counting evidence missing")
    clocks = [c for d in train for c in d["clocks"]]
    _check(any(_different_depths(c) for c in clocks), "both contour alternatives must fail")
    _check(any(_uniform_witness(c) for c in clocks), "count-free cycling must fail")
    _check(any(_local_witness(d) for d in train), "global counting must fail")
    _check(any(0 in c["counts"] and c["period"] for c in clocks), "skipped category missing")
    _check(any(c["colors"] and not c["period"] and c["objects"] for c in clocks), "zero clock missing")
    _check(any(not c["colors"] and c["objects"] for c in clocks), "unkeyed chamber evidence missing")
    _check(any(len(c["period"]) >= 3 and c["depth"].max() >= len(c["period"]) for c in clocks),
           "visible repeated weighted cycle missing")
    _check({(1, -1), (1, 1)} <= {c["direction"] for c in clocks}, "training key directions")
    _check(len({d["background"] for d in train}) >= 2, "backgrounds must vary")
    train_colors = {color for c in clocks for color in c["colors"]}
    unmatched = {o["color"] for c in clocks for o in c["objects"] if o["color"] not in c["colors"]}
    for d in test:
        _check(any(c["direction"] is not None and c["direction"][0] == -1 for c in d["clocks"]),
               "test must have an upward key")
        new = {color for c in d["clocks"] for color, count in zip(c["colors"], c["counts"])
               if count > 0 and color not in train_colors}
        _check(bool(new & unmatched), "test needs a former distractor as a new category")


def _signature(puzzle):
    return tuple((split, _input_key(p["input"]), _input_key(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def self_test():
    """Bounded 42-seed diversity/semantic checks; returns compact measurements."""
    puzzles, shapes, backgrounds, walls, geometries = set(), set(), set(), set(), set()
    regions, active, lengths, periods, counts, masks, directions = (set() for _ in range(7))
    concave = enclosed = diagonal = zeros = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                original = inp.copy()
                _check(np.array_equal(solve(inp), pair["output"]), "solver mismatch")
                _check(np.array_equal(inp, original), "solver mutated input")
                d = _inspect(inp)
                shapes.add(inp.shape)
                backgrounds.add(d["background"])
                walls.add(d["wall"])
                geometries.add(_input_key(d["wall_mask"].astype(np.int8)))
                regions.add(len(d["clocks"]))
                active.add(sum(bool(c["colors"]) for c in d["clocks"]))
                for c in d["clocks"]:
                    _check(np.array_equal(c["depth"], _depth_erosion(c["region"])), "depth implementations disagree")
                    lengths.add(len(c["colors"]))
                    periods.add(len(c["period"]))
                    counts.update(c["counts"])
                    if c["direction"]:
                        directions.add(c["direction"])
                    concave += c["concave"]
                    zeros += bool(c["colors"] and not c["period"])
                    m = c["region"]
                    enclosed += not (m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())
                    for o in c["objects"]:
                        a = min(r for r, col in o["cells"])
                        b = min(col for r, col in o["cells"])
                        masks.add(tuple(sorted((r - a, col - b) for r, col in o["cells"])))
                diagonal += _teaching_evidence([dict(d, input=inp)])[2]
    _check(len(puzzles) == 42, "repeated puzzles")
    _check(len(shapes) >= 60 and (30, 30) in shapes, "canvas variation")
    _check(backgrounds == walls == set(range(9)), "local palette roles")
    _check(len(geometries) >= 150 and len(regions) >= 4 and len(active) >= 3, "structural variation")
    _check({1, 2, 3} <= lengths and {0, 1, 2, 3, 4} <= counts and len(periods) >= 7,
           "key/count/period variation")
    _check(len(masks) >= 70 and max(map(len, masks)) == 10, "object vocabulary")
    _check(directions == {(r, c) for r in (-1, 1) for c in (-1, 1)}, "key orientations")
    _check(concave >= 100 and enclosed >= 3 and zeros >= 42 and diagonal >= 42, "semantic variations")
    _check(_signature(generate_puzzle(170105076, 2, 1)) == _signature(generate_puzzle(170105076, 2, 1)),
           "determinism")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(393217, a, b)
            _check((len(p["train"]), len(p["test"])) == (a, b), "explicit counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0, 2, 1)
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    d = _inspect(broken["test"][0]["input"])
    position = tuple(np.argwhere(d["wall_mask"])[0])
    broken["test"][0]["output"][position] = d["background"]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a changed wall")
    return dict(seeds=42, distinct_puzzles=len(puzzles), canvas_shapes=len(shapes),
                wall_geometries=len(geometries), chamber_counts=sorted(regions),
                active_clock_counts=sorted(active), key_lengths=sorted(lengths),
                object_counts=sorted(counts), period_lengths=sorted(periods),
                object_masks=len(masks), concave_chambers=concave,
                enclosed_chambers=enclosed, zero_clocks=zeros, diagonal_pairs=diagonal)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
