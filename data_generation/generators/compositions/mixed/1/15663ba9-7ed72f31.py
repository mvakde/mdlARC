"""Marker completion exposes convex/concave bends of simple orthogonal loops.

Fixed semantic colors: marker=8, convex=4, concave=2. Background and
monochrome contour colors are geometric roles. No parent imports or data reads.
"""
from __future__ import annotations

import hashlib

import numpy as np


MARKER, CONVEX, CONCAVE = 8, 4, 2
ROLE_COLORS = (0, 1, 3, 5, 6, 7, 9)
HOST_SEEDS = (0, 701132253, 949731080, 839208398, 780792752, 277402250,
              778614063, 302828527, 963060207, 56441211, 316516774,
              226304337, 75131437, 335913123, 149832770, 8932103,
              739596031, 553356895, 977133663, 922230657, 113172783,
              530156096, 826463117, 492558504, 520039369, 44274901,
              902665676, 811351077, 361867180, 679447223, 307445471,
              680492580, 239407990, 66990285, 893698531, 40621195,
              137757748, 847194795, 74872454, 841675081, 146612311, 173796198)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "Invalid dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "Grid must contain integers")
    _need(bool(np.all((a >= 0) & (a <= 9))), "Invalid color")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _adj(p):
    r, c = p
    return ((r-1, c), (r+1, c), (r, c-1), (r, c+1))


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, found = [start], {start}
        while stack:
            for q in _adj(stack.pop()):
                if q in remaining:
                    remaining.remove(q)
                    found.add(q)
                    stack.append(q)
        result.append(found)
    return result


def _bbox(cells):
    rs, cs = zip(*cells)
    return min(rs), min(cs), max(rs), max(cs)


def _interior(cells):
    """Padded flood fill, including when a complete contour touches a grid edge."""
    r0, c0, r1, c1 = _bbox(cells)
    h, w = r1-r0+3, c1-c0+3
    blocked = {(r-r0+1, c-c0+1) for r, c in cells}
    outside, stack = {(0, 0)}, [(0, 0)]
    while stack:
        for r, c in _adj(stack.pop()):
            if (0 <= r < h and 0 <= c < w and (r, c) not in blocked
                    and (r, c) not in outside):
                outside.add((r, c))
                stack.append((r, c))
    return {(r+r0-1, c+c0-1) for r in range(h) for c in range(w)
            if (r, c) not in blocked and (r, c) not in outside}


def _cycle(cells):
    _need(len(cells) >= 8, "Contour too small")
    neighbors = {p: sorted(q for q in _adj(p) if q in cells) for p in cells}
    _need(all(len(n) == 2 for n in neighbors.values()), "Contour branches or ends")
    start = min(cells)
    path, seen = [start], {start}
    previous, current = start, neighbors[start][0]
    while current != start:
        _need(current not in seen, "Contour repeats")
        path.append(current)
        seen.add(current)
        nxt = [p for p in neighbors[current] if p != previous][0]
        previous, current = current, nxt
    _need(len(path) == len(cells), "Multiple cycles")
    # Columns are x, rows are y; signed turns are compared to signed area,
    # so the answer is independent of traversal direction and orientation.
    area = sum(c*path[(i+1) % len(path)][0]
               - path[(i+1) % len(path)][1]*r for i, (r, c) in enumerate(path))
    _need(area != 0, "Zero-area loop")
    classes = {}
    for i, (r, c) in enumerate(path):
        pr, pc = path[i-1]
        nr, nc = path[(i+1) % len(path)]
        cross = (c-pc)*(nr-r) - (r-pr)*(nc-c)
        classes[(r, c)] = 0 if not cross else (1 if cross*area > 0 else 2)
    convex = sum(k == 1 for k in classes.values())
    concave = sum(k == 2 for k in classes.values())
    _need(convex-concave == 4, "Invalid orthogonal turning number")
    return {"cells": cells, "path": path, "classes": classes,
            "convex": convex, "concave": concave}


def _endpoints(source):
    degrees = {p: sum(q in source for q in _adj(p)) for p in source}
    ends = {p for p, d in degrees.items() if d == 1}
    _need(len(ends) == 2 and all(d in (1, 2) for d in degrees.values()),
          "Payload must be a simple open path")
    _need(len(_components(source)) == 1, "Disconnected payload")
    return ends


def _marker_kind(marker):
    if len(marker) == 1:
        return "P"
    r0, c0, r1, c1 = _bbox(marker)
    if r0 == r1 and len(marker) == c1-c0+1:
        return "H"
    if c0 == c1 and len(marker) == r1-r0+1:
        return "V"
    raise ValueError("Marker must be a straight segment or singleton")


def _image(cells, marker, kind):
    r, c = min(marker)
    if kind == "H":
        return {(2*r-a, b) for a, b in cells}
    if kind == "V":
        return {(a, 2*c-b) for a, b in cells}
    return {(2*r-a, 2*c-b) for a, b in cells}


def _one_side(source, ends, axis, coordinate):
    return (all(p[coordinate] == axis for p in ends)
            and (all(p[coordinate] < axis for p in source-ends)
                 or all(p[coordinate] > axis for p in source-ends)))


def _candidate(source, marker, kind, grid, background, occupied):
    ends = _endpoints(source)
    ar, ac = min(marker)
    if kind in ("H", "V"):
        _need(_one_side(source, ends, ar if kind == "H" else ac,
                        0 if kind == "H" else 1), "Wrong reflection seam")
    else:
        p, q = sorted(ends)
        _need((p[0]+q[0], p[1]+q[1]) == (2*ar, 2*ac), "Wrong center")
        _need(_one_side(source, ends, ar, 0) or _one_side(source, ends, ac, 1),
              "Half-turn arc must occupy one half-plane")
    transformed = _image(source, marker, kind)
    _need(source & transformed == ends, "Only the two seams may overlap")
    new = transformed-source
    h, w = grid.shape
    _need(all(0 <= r < h and 0 <= c < w and grid[r, c] == background
              for r, c in new), "Copy is clipped or collides")
    info = _cycle(source | transformed)
    inside = _interior(info["cells"])
    _need(bool(inside) and marker <= inside, "Marker must be inside the loop")
    region = inside | info["cells"]
    _need(occupied & region == source | marker, "Nested or conflicting group")
    info.update(source=source, marker=marker, kind=kind, added=new,
                region=region, color=int(grid[min(source)]))
    return info


def _analyze(input_grid):
    grid = _grid(input_grid)
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[counts.argmax()])
    _need(int(counts.max()) > grid.size//2, "Background must be a strict majority")
    _need(background in ROLE_COLORS, "Reserved semantic color used as background")
    _need(not bool(np.any((grid == CONVEX) | (grid == CONCAVE))),
          "Bend labels are reserved for output")
    marker_cells = {tuple(map(int, p)) for p in np.argwhere(grid == MARKER)}
    sources = _components(tuple(map(int, p)) for p in np.argwhere(
        (grid != background) & (grid != MARKER)))
    markers = _components(marker_cells)
    _need(bool(markers) and len(markers) == len(sources), "Unpaired markers/arcs")
    for source in sources:
        _need(len({int(grid[p]) for p in source}) == 1, "Arc is not monochrome")
        _endpoints(source)
    occupied = {tuple(map(int, p)) for p in np.argwhere(grid != background)}
    groups, used, reserved = [], set(), set()
    for marker in markers:
        kind = _marker_kind(marker)
        candidates = []
        for i, source in enumerate(sources):
            try:
                info = _candidate(source, marker, kind, grid, background, occupied)
            except ValueError:
                continue
            candidates.append((i, info))
        _need(len(candidates) == 1, "Missing or ambiguous marker ownership")
        i, info = candidates[0]
        _need(i not in used, "Two markers own one arc")
        _need(not bool(info["region"] & reserved), "Complete groups touch or overlap")
        used.add(i)
        for r, c in info["region"]:
            reserved.update((r+dr, c+dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))
        groups.append(info)
    _need(len(used) == len(sources), "Unowned arc")
    return background, groups


def _paint(grid, groups, classify=True):
    output = grid.copy()
    for g in groups:
        for p, kind in g["classes"].items():
            output[p] = (CONVEX if kind == 1 else CONCAVE if kind == 2
                         else g["color"]) if classify else g["color"]
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from this array alone; no training pairs, seed or sampler state."""
    grid = _grid(input_grid)
    _, groups = _analyze(grid)
    return _paint(grid, groups)


def _raster(vertices):
    path = [vertices[0]]
    for a, b in zip(vertices, vertices[1:]):
        dr, dc = b[0]-a[0], b[1]-a[1]
        _need((dr == 0) != (dc == 0), "Nonorthogonal or zero-length edge")
        length = abs(dr)+abs(dc)
        dr, dc = dr//length, dc//length
        path.extend((a[0]+i*dr, a[1]+i*dc) for i in range(1, length+1))
    _need(len(set(path)) == len(path), "Arc crosses itself")
    return set(path)


def _simplify(vertices):
    result = []
    for p in vertices:
        if result and p == result[-1]:
            continue
        while len(result) >= 2:
            a, b = result[-2:]
            if ((a[0] == b[0] == p[0] and (b[1]-a[1])*(p[1]-b[1]) > 0)
                    or (a[1] == b[1] == p[1] and (b[0]-a[0])*(p[0]-b[0]) > 0)):
                result.pop()
            else:
                break
        result.append(p)
    return result


def _edit(vertices, rng):
    """Local corner cuts and inward/outward edge detours, with no seam edits."""
    if rng.random() < .45:
        i = int(rng.integers(1, len(vertices)-1))
        a, b, c = vertices[i-1:i+2]
        la, lb = abs(a[0]-b[0])+abs(a[1]-b[1]), abs(c[0]-b[0])+abs(c[1]-b[1])
        if min(la, lb) < 2:
            return None
        u, v = int(rng.integers(1, min(la, 4))), int(rng.integers(1, min(lb, 4)))
        p = (b[0]+u*(a[0]-b[0])//la, b[1]+u*(a[1]-b[1])//la)
        q = (b[0]+v*(c[0]-b[0])//lb, b[1]+v*(c[1]-b[1])//lb)
        middle = (p[0]+q[0]-b[0], p[1]+q[1]-b[1])
        return _simplify(vertices[:i]+[p, middle, q]+vertices[i+1:])
    i = int(rng.integers(len(vertices)-1))
    a, b = vertices[i:i+2]
    length = abs(a[0]-b[0])+abs(a[1]-b[1])
    if length < 4:
        return None
    dr, dc = (b[0]-a[0])//length, (b[1]-a[1])//length
    # A two-step mouth avoids a spurious four-neighbor chord across a
    # one-step detour. The admissibility check protects the marker corridor,
    # so even a seam-adjacent long side can acquire shoulders away from it.
    u = int(rng.integers(1, length-2))
    v = int(rng.integers(u+2, length))
    depth = int(rng.choice((-2, -1, 1, 2)))
    pr, pc = dc*depth, -dr*depth
    additions = [(a[0]+u*dr, a[1]+u*dc),
                 (a[0]+u*dr+pr, a[1]+u*dc+pc),
                 (a[0]+v*dr+pr, a[1]+v*dc+pc),
                 (a[0]+v*dr, a[1]+v*dc)]
    return _simplify(vertices[:i+1]+additions+vertices[i+1:])


def _prototype(rng, kind, cap, rich=False, witness=False):
    for _ in range(16):
        # Start with a variable stepped cap, not a library of fixed polygons.
        span = int(rng.integers(8 if rich else 6, cap))
        if kind == "P" and span % 2:
            span -= 1
        max_sections = min(5 if rich else 3, span//2)
        sections = int(rng.integers(4 if rich else 2 if witness else 1,
                                    max_sections+1))
        lengths = [2]*sections
        for _ in range(span-2*sections):
            lengths[int(rng.integers(sections))] += 1
        depths = []
        for _ in range(sections):
            choices = [d for d in range(2, (cap-1)//2+1)
                       if not depths or d != depths[-1]]
            depths.append(int(rng.choice(choices)))
        if kind == "P" and witness and depths[0] == depths[-1]:
            choices = [d for d in range(2, (cap-1)//2+1) if d != depths[0]]
            depths[-1] = int(rng.choice(choices))
        vertices, x = [(0, 0), (-depths[0], 0)], 0
        for i, length in enumerate(lengths):
            x += length
            vertices.append((-depths[i], x))
            if i+1 < sections:
                vertices.append((-depths[i+1], x))
        vertices.append((0, span))
        vertices = _simplify(vertices)
        # Work horizontally; transpose/rotate only after constructing the loop.
        marker = ({(0, span//2)} if kind == "P" else
                  {(0, c) for c in range(2, span-1)})
        transform_kind = "P" if kind == "P" else "H"

        def admissible(vs):
            source = _raster(vs)
            ends = _endpoints(source)
            _need(ends == {(0, 0), (0, span)}, "Changed seam")
            _need(all(r < 0 for r, c in source-ends), "Arc crosses seam")
            other = _image(source, marker, transform_kind)
            _need(source & other == ends, "Unexpected overlap")
            loop = _cycle(source | other)
            r0, c0, r1, c1 = _bbox(loop["cells"])
            _need(r1-r0+1 <= cap and c1-c0+1 <= cap, "Shape exceeds cap")
            _need(all(r <= -2 for r, c in source if c not in (0, span)),
                  "Contour approaches internal marker corridor")
            return source, loop

        try:
            source, loop = admissible(vertices)
        except ValueError:
            continue
        edits = int(rng.integers(1 if rich else 0, 5 if rich else 3))
        accepted = 0
        for _ in range(32 if rich else 18):
            if accepted >= edits:
                break
            candidate = _edit(vertices, rng)
            if candidate is None:
                continue
            try:
                new_source, new_loop = admissible(candidate)
            except ValueError:
                continue
            vertices, source, loop = candidate, new_source, new_loop
            accepted += 1
        if witness and not loop["concave"]:
            continue
        if kind == "P" and witness and _image(source, marker, "H") == _image(source, marker, "P"):
            continue
        if kind != "P":
            start = int(rng.integers(2, span-2))
            stop = int(rng.integers(start+2, span))
            marker = {(0, c) for c in range(start, stop)}
        inside = _interior(loop["cells"])
        if not marker <= inside:
            continue
        turns = int(rng.integers(4)) if kind == "P" else (0 if kind == "H" else 1)
        flip_r, flip_c = int(rng.choice((-1, 1))), int(rng.choice((-1, 1)))

        def move(cells):
            transformed = set()
            for r, c in cells:
                for _ in range(turns):
                    r, c = -c, r
                transformed.add((flip_r*r, flip_c*c))
            return transformed

        source, marker = move(source), move(marker)
        cells, region = move(loop["cells"]), move(loop["cells"] | inside)
        r0, c0, _, _ = _bbox(region)
        shift = lambda s: {(r-r0, c-c0) for r, c in s}
        return {"source": shift(source), "marker": shift(marker),
                "cells": shift(cells), "region": shift(region), "kind": kind}
    raise RuntimeError("Could not construct a valid contour prototype")


def _pack(rng, prototypes, h, w, deterministic=False):
    for _ in range(1 if deterministic else 4):
        forbidden, placed = set(), []
        order = sorted(range(len(prototypes)), key=lambda i: -len(prototypes[i]["region"]))
        for i in order:
            g = prototypes[i]
            _, _, bottom, right = _bbox(g["region"])
            candidates = [(r, c) for r in range(h-bottom) for c in range(w-right)]
            if not deterministic:
                rng.shuffle(candidates)
                if rng.random() < .30:
                    candidates.sort(key=lambda p: min(p[0], p[1], h-bottom-1-p[0], w-right-1-p[1]))
            for row, col in candidates:
                region = {(r+row, c+col) for r, c in g["region"]}
                if region & forbidden:
                    continue
                moved = {k: ({(r+row, c+col) for r, c in v}
                             if isinstance(v, set) else v) for k, v in g.items()}
                moved["index"] = i
                placed.append(moved)
                for r, c in region:
                    forbidden.update((r+dr, c+dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))
                break
            else:
                break
        if len(placed) == len(prototypes):
            return placed
    return None


def _pair(rng, specs, parameters, test=False):
    count = len(specs)
    for attempt in range(19):
        compact = attempt == 18
        cap = 11 if compact or attempt >= 7 else (23 if count == 1 else 15 if count == 2 else 13)
        prototypes = [_prototype(rng, kind, cap, rich=test and (i == 0 or rng.random() < .6),
                                 witness=witness) for i, (kind, witness) in enumerate(specs)]
        max_h = max(_bbox(g["region"])[2]+1 for g in prototypes)
        max_w = max(_bbox(g["region"])[3]+1 for g in prototypes)
        if compact or attempt >= 7:
            h = w = 30
        else:
            low = 7 if count == 1 else 16 if count == 2 else 21
            h = int(rng.integers(max(low, max_h), 31))
            w = int(rng.integers(max(low, max_w), 31))
        groups = _pack(rng, prototypes, h, w, deterministic=compact)
        if groups is None:
            continue
        bg = int(rng.choice(parameters["backgrounds"]))
        grid = np.full((h, w), bg, dtype=np.int8)
        colors = [int(rng.choice(parameters["colors"])) for _ in specs]
        if test:
            colors[0] = parameters["novel"]
        for g in groups:
            g["color"] = colors[g["index"]]
            for p in g["marker"]:
                grid[p] = MARKER
            for p in g["source"]:
                grid[p] = g["color"]
            g.update(_cycle(g["cells"]))
        if np.count_nonzero(grid == bg) <= grid.size//2:
            continue
        # Expected labels come from the deliberately constructed complete paths;
        # solve separately rediscovers ownership and completion from the input.
        output = _paint(grid, groups)
        return {"input": grid, "output": output}
    raise RuntimeError("Bounded contour packing failed")


def _counts(num_train, num_test):
    for name, n, low, high in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        _need(not isinstance(n, (bool, np.bool_)) and isinstance(n, (int, np.integer))
              and low <= n <= high, f"Illegal {name}")
    _need(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    _need(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
          and seed >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    colors = [int(c) for c in rng.permutation(ROLE_COLORS)]
    parameters = {"backgrounds": colors[:1 if rng.random() < .3 else 2],
                  "colors": colors[2:5], "novel": colors[5]}
    # Minimum two demonstrations still cover all operations, both labels,
    # and an asymmetric point witness. Their presentation order is randomized.
    first_line = str(rng.choice(("H", "V")))
    other_line = "V" if first_line == "H" else "H"
    plans = [[(first_line, True)], [(other_line, True), ("P", True)]]
    for _ in range(num_train-2):
        plans.append([(str(rng.choice(("H", "V", "P"))), bool(rng.random() < .4))
                      for _ in range(int(rng.integers(1, 4)))])
    rng.shuffle(plans)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            specs = plans[i] if split == "train" else [
                (str(rng.choice(("H", "V", "P"))), True)
                for _ in range(int(rng.integers(1, 4)))]
            for _ in range(12):
                pair = _pair(rng, specs, parameters, test=split == "test")
                key = _key(pair["input"])
                if key not in seen:
                    # Do not retain a private ownership interpretation that the
                    # observable input cannot reproduce uniquely.
                    _analyze(pair["input"])
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Unique input retry limit exceeded")
    validate_generated_puzzle(puzzle)
    return puzzle


def _translation_image(g):
    a, b, _, _ = _bbox(g["source"])
    c, d, _, _ = _bbox(_image(g["source"], g["marker"], g["kind"]))
    return {(r+c-a, col+d-b) for r, col in g["source"]}


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in ("train", "test")), "Splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, trained, concave_witnesses, point_witness = set(), set(), set(), False
    train_colors, test_colors = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _need(inp.shape == out.shape, "Output shape changed")
            key = _key(inp)
            _need(key not in seen, "Repeated input")
            seen.add(key)
            bg, groups = _analyze(inp)
            _need(np.array_equal(out, _paint(inp, groups)), "Wrong completion or bend labels")
            _need(np.array_equal(out[inp == MARKER], inp[inp == MARKER]), "Marker changed")
            _need(not np.array_equal(_paint(inp, groups, classify=False), out), "A is redundant")
            without_b = inp.copy()
            for g in groups:
                for p in g["source"]:
                    k = g["classes"][p]
                    if k:
                        without_b[p] = CONVEX if k == 1 else CONCAVE
                _need(any(g["classes"][p] == 1 for p in g["added"]), "No newly exposed convex bend")
                # Symmetry pairs every non-seam bend; odd concave counts would
                # violate the construction's exact reflection/half-turn closure.
                _need(g["concave"] % 2 == 0, "Symmetry lost")
                for p in g["cells"]:
                    if not g["classes"][p]:
                        _need(out[p] == g["color"], "Straight contour cell recolored")
                if split == "train":
                    trained.add(g["kind"])
                    train_colors.add(g["color"])
                    if g["concave"]:
                        concave_witnesses.add(g["kind"])
                    if g["kind"] == "P":
                        ar, ac = min(g["marker"])
                        h_image = {(2*ar-r, c) for r, c in g["source"]}
                        v_image = {(r, 2*ac-c) for r, c in g["source"]}
                        actual = _image(g["source"], g["marker"], "P")
                        point_witness |= (actual != h_image and actual != v_image
                                          and actual != _translation_image(g))
                else:
                    _need(g["kind"] in trained, "Untrained test operation")
                    test_colors.add(g["color"])
            _need(not np.array_equal(without_b, out), "B is redundant")
            occupied = set().union(*(g["cells"] | g["marker"] for g in groups))
            _need(all(out[r, c] == bg for r in range(inp.shape[0]) for c in range(inp.shape[1])
                      if (r, c) not in occupied), "Unrelated background changed")
    _need(trained == {"H", "V", "P"}, "Training lacks an operation")
    _need(concave_witnesses == trained and point_witness, "Training conceals a concept")
    _need(bool(test_colors-train_colors), "Test lacks the intended palette shift")


def _quadrants(path, row, col):
    """Independent local-angle check via point-in-polygon, without signed turns."""
    total = 0
    for dr in (-.25, .25):
        for dc in (-.25, .25):
            y, x = row+dr, col+dc
            crossings = 0
            for (a, b), (c, d) in zip(path, path[1:]+path[:1]):
                if b == d and min(a, c) < y < max(a, c) and b > x:
                    crossings += 1
            total += crossings % 2
    return total


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for p in puzzle[split]:
            for name in ("input", "output"):
                a = np.asarray(p[name], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 seeds, one puzzle live at a time; summary only, no disk writes."""
    shapes, silhouettes, counts, concaves, patterns, backgrounds = (set() for _ in range(6))
    signatures, sides = set(), {"H": set(), "V": set(), "P": set()}
    groups_checked = bends_checked = adjacent_bends = edge_groups = nonmonotone_arcs = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                bg, groups = _analyze(inp)
                shapes.add(inp.shape)
                counts.add(len(groups))
                backgrounds.add(bg)
                _need(np.array_equal(solve(inp.copy()), pair["output"]), "Solver mismatch")
                for g in groups:
                    groups_checked += 1
                    concaves.add(g["concave"])
                    r0, c0, r1, c1 = _bbox(g["cells"])
                    silhouettes.add(tuple(sorted((r-r0, c-c0) for r, c in g["cells"])))
                    path, classes = g["path"], g["classes"]
                    turns = [classes[p] for p in path if classes[p]]
                    reverse = turns[::-1]
                    patterns.add(min(tuple(seq[i:]+seq[:i])
                                     for seq in (turns, reverse) for i in range(len(seq))))
                    adjacent_bends += any(classes[p] and classes[path[(i+1) % len(path)]]
                                          for i, p in enumerate(path))
                    edge_groups += r0 == 0 or c0 == 0 or r1 == inp.shape[0]-1 or c1 == inp.shape[1]-1
                    e, f = sorted(_endpoints(g["source"]))
                    axis = 1 if e[0] == f[0] else 0
                    axial_edges = sum(p < q and p[axis] != q[axis]
                                      for p in g["source"] for q in _adj(p)
                                      if q in g["source"])
                    nonmonotone_arcs += axial_edges > abs(e[axis]-f[axis])
                    ar, ac = min(g["marker"])
                    mean_r = sum(r-ar for r, c in g["source"])
                    mean_c = sum(c-ac for r, c in g["source"])
                    sides[g["kind"]].add((int(np.sign(mean_r)), int(np.sign(mean_c)))
                                         if g["kind"] == "P" else int(np.sign(
                                             mean_r if g["kind"] == "H" else mean_c)))
                    for (r, c), k in classes.items():
                        if k:
                            _need(_quadrants(path, r, c) == (1 if k == 1 else 3),
                                  "Signed-turn and interior-angle checks disagree")
                            bends_checked += 1
                    # Check the actual symmetry independently with cropped NumPy
                    # flips, allowing the two self/shared seam endpoints.
                    a, b, c, d = _bbox(g["source"])
                    mask = np.zeros((c-a+1, d-b+1), dtype=bool)
                    for r, col in g["source"]:
                        mask[r-a, col-b] = True
                    if g["kind"] == "H":
                        tile, top, left = np.flipud(mask), 2*ar-c, b
                    elif g["kind"] == "V":
                        tile, top, left = np.fliplr(mask), a, 2*ac-d
                    else:
                        tile, top, left = np.rot90(mask, 2), 2*ar-c, 2*ac-d
                    copied = {(top+int(r), left+int(col)) for r, col in np.argwhere(tile)}
                    _need(g["cells"] == g["source"] | copied, "Independent symmetry check failed")
    _need(len(signatures) == 42 and len(shapes) >= 50 and len(silhouettes) >= 100,
          "Insufficient structural variation")
    _need(counts == {1, 2, 3} and len(concaves) >= 5 and len(patterns) >= 8,
          "Insufficient count or bend-pattern variation")
    _need(backgrounds == set(ROLE_COLORS) and adjacent_bends and edge_groups,
          "Missing palette, compact-corner or boundary coverage")
    _need(nonmonotone_arcs > 0, "Mutations never extend the initial monotone-cap vocabulary")
    _need(sides["H"] == {-1, 1} and sides["V"] == {-1, 1}, "Missing reflection side")
    _need(any(r > 0 for r, c in sides["P"]) and any(r < 0 for r, c in sides["P"])
          and any(c > 0 for r, c in sides["P"]) and any(c < 0 for r, c in sides["P"]),
          "Missing point orientations")
    _need(_digest(generate_puzzle(73, 2, 1)) == _digest(generate_puzzle(73, 2, 1)),
          "Nondeterministic seed")
    _need(_digest(generate_puzzle(74)) != _digest(generate_puzzle(75)), "Identical seeds")
    # Input-only family members outside the sampler's roomy marker corridor:
    # a 3x3 point loop and a 3x5 reflection loop with an edge-touching top.
    for point in (True, False):
        inp = np.zeros((5, 7), dtype=np.int8)
        top, seam, bottom = (1, 2, 3) if point else (0, 1, 2)
        left, right, color = 1, 3 if point else 5, 6 if point else 7
        for col in range(left, right+1):
            inp[top, col] = color
        inp[seam, left] = inp[seam, right] = color
        marker_cols = (2,) if point else (2, 3)
        for col in marker_cols:
            inp[seam, col] = MARKER
        expected = inp.copy()
        expected[bottom, left:right+1] = color
        for r in (top, bottom):
            expected[r, left] = expected[r, right] = CONVEX
        _need(np.array_equal(solve(inp), expected), "Unsampled small-loop member failed")
        malformed = inp.copy()
        malformed[top, left-1] = color
        try:
            solve(malformed)
        except ValueError:
            pass
        else:
            raise AssertionError("Branched contour accepted")
    # Every explicit count, invalid types, and input-only operation independence.
    for nt in range(2, 11):
        for ns in range(1, min(4, nt-1)+1):
            p = generate_puzzle(91+nt*4+ns, nt, ns)
            _need((len(p["train"]), len(p["test"])) == (nt, ns), "Wrong count")
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    p = generate_puzzle(0)
    # Reject a missing B addition, an A-label swap, and a changed marker.
    for corruption in ("missing_copy", "swap_labels", "marker"):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        pair = bad["train"][0]
        bg, groups = _analyze(pair["input"])
        out = pair["output"]
        if corruption == "missing_copy":
            out[min(groups[0]["added"])] = bg
        elif corruption == "swap_labels":
            mask_a, mask_b = out == CONVEX, out == CONCAVE
            out[mask_a], out[mask_b] = CONCAVE, CONVEX
        else:
            out[min(groups[0]["marker"])] = bg
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Corrupt label accepted")
    return {"seeds": 42, "pairs": 210, "groups": groups_checked,
            "bends_checked": bends_checked, "grid_shapes": len(shapes),
            "contour_silhouettes": len(silhouettes), "turn_patterns": len(patterns),
            "group_counts": sorted(counts), "concave_counts": sorted(concaves),
            "backgrounds": sorted(backgrounds), "adjacent_bend_groups": adjacent_bends,
            "edge_groups": edge_groups, "nonmonotone_arcs": nonmonotone_arcs,
            "distinct_puzzles": len(signatures)}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
