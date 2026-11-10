"""Ranked cast shadows: sparse geometry, unit translation, occlusion, unary rank.

All semantic decoding is input-only. The sampler uses its own proposed vertex
orders to rasterize labels; validation compares those labels to a fresh decode.
"""

import numpy as np


DIRECTIONS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _sign(x):
    return int(x > 0) - int(x < 0)


def _aligned(a, b):
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dr, dc) > 0 and (dr == 0 or dc == 0 or dr == dc)


def _line(a, b):
    _need(_aligned(a, b), "link is not octilinear")
    dr, dc = b[0] - a[0], b[1] - a[1]
    return [(a[0] + j * _sign(dr), a[1] + j * _sign(dc))
            for j in range(max(abs(dr), abs(dc)) + 1)]


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _intersection(a, b, c, d):
    if (max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0])
            or max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1])):
        return False
    return _cross(a, b, c) * _cross(a, b, d) <= 0 and _cross(c, d, a) * _cross(c, d, b) <= 0


def _topology(points):
    """Prefer a unique simple full cycle; otherwise require a unique full path.

    Reversal and cyclic starting point are not different completions. The search
    compares edge sets, and excludes continuous crossings between disjoint edges.
    """
    n = len(points)
    _need(3 <= n <= 12 and len(set(points)) == n, "invalid source landmarks")
    point_set = set(points)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)
             if _aligned(points[i], points[j])
             and not point_set.intersection(_line(points[i], points[j])[1:-1])]
    adjacent = [[] for _ in points]
    for e, (i, j) in enumerate(edges):
        adjacent[i].append((j, e))
        adjacent[j].append((i, e))
    _need(all(adjacent), "disconnected source landmarks")
    conflicts = [set() for _ in edges]
    for e, (i, j) in enumerate(edges):
        for f, (k, l) in enumerate(edges[:e]):
            if not {i, j}.intersection((k, l)) and _intersection(points[i], points[j], points[k], points[l]):
                conflicts[e].add(f)
                conflicts[f].add(e)

    def search(closed):
        solutions = {}
        visits = 0

        def visit(order, used, chosen):
            nonlocal visits
            visits += 1
            _need(visits <= 30000, "ambiguous/expensive landmark set")
            if len(solutions) > 1:
                return
            if len(order) == n:
                if closed:
                    end = [e for v, e in adjacent[order[-1]] if v == order[0]]
                    if not end or conflicts[end[0]].intersection(chosen):
                        return
                    chosen = chosen + (end[0],)
                solutions[tuple(sorted(chosen))] = order
                return
            for v, e in adjacent[order[-1]]:
                if v not in used and not conflicts[e].intersection(chosen):
                    visit(order + (v,), used | {v}, chosen + (e,))

        for start in (range(1) if closed else range(n)):
            visit((start,), {start}, ())
            if len(solutions) > 1:
                break
        return solutions

    cycles = search(True)
    _need(len(cycles) <= 1, "competing closed contours")
    solutions = cycles if cycles else search(False)
    _need(len(solutions) == 1, "no unique complete contour/path")
    chosen, order = next(iter(solutions.items()))
    selected_edges = [edges[e] for e in chosen]
    pixels = set().union(*(set(_line(points[i], points[j])) for i, j in selected_edges))
    _need(len(pixels) > len(points), "source is already complete")
    long_pixels = set().union(*(set(_line(points[i], points[j])) for i, j in selected_edges
                                if len(_line(points[i], points[j])) >= 3))
    _need(set(points) <= long_pixels, "unit-only appendage is outside the sampled family")
    turns = {_sign(_cross(points[order[i - 1]], points[order[i]], points[order[(i + 1) % n]]))
             for i in range(n)} if cycles else set()
    return {"points": points, "edges": selected_edges, "order": order,
            "pixels": pixels, "closed": bool(cycles), "chords": len(edges) - len(chosen),
            "concave": {-1, 1} <= turns}


def _components(pixels, diagonal=True):
    remaining = set(pixels)
    result = []
    directions = DIRECTIONS if diagonal else ((1, 0), (-1, 0), (0, 1), (0, -1))
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        group, todo = {first}, [first]
        while todo:
            r, c = todo.pop()
            for dr, dc in directions:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    todo.append(q)
        result.append(group)
    return sorted(result, key=lambda g: (-len(g), min(g)))


def _unary(points):
    """A key is a gapless straight run of one to six neighboring cells."""
    if not 1 <= len(points) <= 6:
        return False
    if len(points) == 1:
        return True
    ps = set(points)
    return any({(p[0] + j * dr, p[1] + j * dc) for j in range(len(ps))} == ps
               for p in ps for dr, dc in DIRECTIONS)


def _grid(value):
    grid = np.asarray(value)
    _need(grid.ndim == 2 and all(1 <= d <= 30 for d in grid.shape), "invalid dimensions")
    _need(np.issubdtype(grid.dtype, np.integer) and np.all((grid >= 0) & (grid <= 9)), "invalid colors")
    return grid


def _decode(value):
    grid = _grid(value)
    frequencies = np.bincount(grid.ravel().astype(int), minlength=10)
    background = int(np.argmax(frequencies))
    _need(frequencies[background] > grid.size // 2, "background must occupy more than half the input")
    groups = {int(c): [tuple(map(int, p)) for p in np.argwhere(grid == c)]
              for c in np.unique(grid) if c != background}
    sources, keys = [], []
    for color, points in groups.items():
        if _unary(points):
            keys.append((color, set(points)))
        else:
            info = _topology(points)
            info["color"] = color
            sources.append(info)
    _need(1 <= len(sources) <= 4 and len(keys) == len(sources), "each source needs its own key")
    all_sources = set()
    for s in sources:
        _need(not all_sources.intersection(s["pixels"]), "source drawings overlap")
        all_sources.update(s["pixels"])
    _need(all(grid[p] in (background, s["color"]) for s in sources for p in s["pixels"]), "key/landmark on another source")
    used = set()
    for color, key in keys:
        matches = []
        for i, s in enumerate(sources):
            for v in s["points"]:
                for dr, dc in DIRECTIONS:
                    if key == {(v[0] + j * dr, v[1] + j * dc) for j in range(1, len(key) + 1)}:
                        matches.append((i, v, (dr, dc)))
        _need(len(matches) == 1, "key has no unique source anchor and direction")
        i, anchor, offset = matches[0]
        first = (anchor[0] + offset[0], anchor[1] + offset[1])
        _need(sum(max(abs(first[0] - v[0]), abs(first[1] - v[1])) == 1
                  for s in sources for v in s["points"]) == 1,
              "first key cell touches more than one source landmark")
        _need(i not in used, "two keys address the same drawing")
        used.add(i)
        s = sources[i]
        dr, dc = offset
        translated = {(r + dr, c + dc) for r, c in s["pixels"]}
        _need(all(0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] for r, c in translated), "translated stroke would be clipped")
        _need(not translated.intersection(all_sources - s["pixels"]), "projection hits another drawing")
        fragments = _components(translated - s["pixels"])
        sizes = [len(g) for g in fragments]
        _need(len(fragments) >= 2 and len(set(sizes)) == len(sizes), "shadow needs unequal competitors")
        _need(1 <= len(key) <= len(fragments), "requested rank does not exist")
        s.update(key_color=color, key=key, anchor=anchor, offset=offset,
                 translated=translated, fragments=fragments, rank=len(key),
                 selected=fragments[len(key) - 1])
    for i, s in enumerate(sources):
        for t in sources[:i]:
            _need(not s["translated"].intersection(t["translated"]), "shadows overlap between drawings")
        _need(all(grid[p] in (background, s["key_color"], s["color"]) for p in s["translated"]), "projection crosses another key")
    return grid, background, sources


def _answer(grid, sources, background):
    out = grid.copy()
    for s in sources:
        for p in s["pixels"]:
            if grid[p] == background:
                out[p] = s["color"]
        for p in s["selected"]:
            if grid[p] == background:
                out[p] = s["key_color"]
    return np.ascontiguousarray(out)


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid, background, sources = _decode(input_grid)
    return _answer(grid, sources, background)


def _straight(pixels):
    points = sorted(pixels)
    if len(points) <= 2:
        return True
    a, b = points[0], points[-1]
    return _aligned(a, b) and set(_line(a, b)) == set(points)


def _naive_pixels(source):
    points = source["points"]
    return set().union(*(set(_line(a, b)) for i, a in enumerate(points)
                         for b in points[i + 1:] if _aligned(a, b)))


def _spatial_shortcuts(pairs):
    """Common position-based ordinal models that survive every demonstration."""
    remaining = {"row", "reverse_row", "column", "reverse_column"}
    for pair in pairs:
        _, _, sources = _decode(pair["input"])
        for s in sources:
            for rule in tuple(remaining):
                key = (lambda g: min(g)) if "row" in rule else (lambda g: min((c, r) for r, c in g))
                ordered = sorted(s["fragments"], key=key, reverse=rule.startswith("reverse"))
                if ordered[s["rank"] - 1] != s["selected"]:
                    remaining.remove(rule)
    return remaining


def measure_pair(pair):
    grid, bg, sources = _decode(pair["input"])
    return {"shape": list(grid.shape), "background": bg, "sources": len(sources),
            "vertices": [len(s["points"]) for s in sources],
            "closed": sum(s["closed"] for s in sources),
            "open": sum(not s["closed"] for s in sources),
            "concave": sum(s["concave"] for s in sources),
            "chords": sum(s["chords"] for s in sources),
            "unit_links": sum(len(_line(s["points"][i], s["points"][j])) == 2 for s in sources for i, j in s["edges"]),
            "ranks": [s["rank"] for s in sources],
            "populations": [len(s["fragments"]) for s in sources],
            "fragment_sizes": [[len(g) for g in s["fragments"]] for s in sources],
            "offsets": [list(s["offset"]) for s in sources],
            "bent_selections": sum(not _straight(s["selected"]) for s in sources),
            "diagonal_connectivity": sum(len(_components(s["selected"], False)) > 1 for s in sources),
            "key_elsewhere": sum(not s["selected"].intersection(s["key"]) for s in sources),
            "intermediate": sum(1 < s["rank"] < len(s["fragments"]) for s in sources),
            "last_ranked": sum(s["rank"] == len(s["fragments"]) for s in sources),
            "occluded_gap_cells": sum(len((s["pixels"] & s["translated"]) - set(s["points"])) for s in sources),
            "boundary": any(r in (0, grid.shape[0] - 1) or c in (0, grid.shape[1] - 1)
                            for s in sources for r, c in s["pixels"] | s["translated"] | s["key"]),
            "geometry": repr(sorted((s["closed"], tuple(sorted(_normalize(s["points"])))) for s in sources))}


def _normalize(points):
    r0, c0 = min(r for r, c in points), min(c for r, c in points)
    return [(r - r0, c - c0) for r, c in points]


def _transform(points, code):
    result = []
    for r, c in points:
        if code // 4:
            c = -c
        for _ in range(code % 4):
            r, c = c, -r
        result.append((r, c))
    return result


def _propose(rng, kind, size):
    if kind == "walk":
        points = [(0, 0)]
        for _ in range(int(rng.integers(3, 8))):
            dr, dc = DIRECTIONS[int(rng.integers(8))]
            length = int(rng.integers(1, size + 1))
            points.append((points[-1][0] + length * dr, points[-1][1] + length * dc))
        return points, False
    h, w = int(rng.integers(4, size + 3)), int(rng.integers(4, size + 4))
    if kind == "notch":
        a = int(rng.integers(1, w - 2))
        b = int(rng.integers(a + 1, w))
        depth = int(rng.integers(1, h))
        return [(0, 0), (0, w), (h, w), (h, b), (h - depth, b),
                (h - depth, a), (h, a), (h, 0)], True
    mode = int(rng.integers(4))
    if mode == 0:
        w = h
    if mode == 1:
        # A diagonal diamond need not have four equal side lengths.
        return [(0, h), (w, h + w), (h + w, w), (h, 0)], True
    if mode == 2:
        cuts = [int(rng.integers(0, min(h, w) // 2)) for _ in range(4)]
        a, b, c, d = cuts
        raw = [(0, a), (0, w - b), (b, w), (h - c, w),
               (h, w - c), (h, d), (h - d, 0), (a, 0)]
        return [p for j, p in enumerate(raw) if p != raw[j - 1]], True
    return [(0, 0), (0, w), (h, w), (h, 0)], True


def _options(points, closed, rank, offsets, minimum, witness, elsewhere):
    if len(points) != len(set(points)) or not 3 <= len(points) <= 12 or _unary(points):
        return []
    info = _topology(points)
    intended = {(j, j + 1) for j in range(len(points) - 1)}
    if closed:
        intended.add((0, len(points) - 1))
    if {tuple(sorted(e)) for e in info["edges"]} != intended:
        return []
    if witness and (not info["chords"] or not closed or len(_naive_pixels(info) - info["pixels"]) < 2):
        return []
    source = info["pixels"]
    result = []
    for dr, dc in offsets:
        translated = {(r + dr, c + dc) for r, c in source}
        fragments = _components(translated - source)
        sizes = [len(g) for g in fragments]
        if len(fragments) < max(minimum, rank) or len(set(sizes)) != len(sizes):
            continue
        chosen = fragments[rank - 1]
        if len(chosen) < 3 or (witness and (_straight(chosen) or not (source & translated) - set(points))):
            continue
        for v in points:
            key = {(v[0] + j * dr, v[1] + j * dc) for j in range(1, rank + 1)}
            if key.intersection(source) or len(chosen - key) < 2:
                continue
            # The first key cell must visibly address exactly one landmark.
            first = (v[0] + dr, v[1] + dc)
            if sum(max(abs(first[0] - q[0]), abs(first[1] - q[1])) == 1 for q in points) != 1:
                continue
            matches = sum(key == {(q[0] + j * a, q[1] + j * b) for j in range(1, rank + 1)}
                          for q in points for a, b in DIRECTIONS)
            if matches != 1:
                continue
            if elsewhere and key.intersection(chosen):
                continue
            if elsewhere and len(_components(chosen, False)) == 1:
                continue
            if len((translated - source) - chosen - key) < 2:
                continue
            footprint = source | translated | key
            if max(r for r, c in footprint) - min(r for r, c in footprint) > 24:
                continue
            if max(c for r, c in footprint) - min(c for r, c in footprint) > 24:
                continue
            result.append(dict(info, key=key, offset=(dr, dc), rank=rank,
                               translated=translated, fragments=fragments, selected=chosen))
    return result


def _make_object(rng, rank, offsets, size, kind=None, minimum=2,
                 witness=False, elsewhere=False):
    selected_kind = kind or str(rng.choice(["notch", "walk"] if max(rank, minimum) >= 3
                                         else ["convex", "notch", "walk"]))
    for _ in range(450):
        points, closed = _propose(rng, selected_kind, size)
        points = _normalize(_transform(points, int(rng.integers(8))))
        try:
            options = _options(points, closed, rank, offsets, minimum, witness, elsewhere)
        except AssertionError:
            continue
        if options:
            return options[int(rng.integers(len(options)))]
    # Search a small parameterized rescue set, still checking visible topology.
    # Normal cases do not use these; they ensure rare layout requests terminate.
    fallback = [([(0, 0), (0, 6), (6, 6), (6, 0)], True),
                ([(0, 6), (6, 12), (12, 6), (6, 0)], True),
                ([(2, 0), (0, 2), (5, 2), (8, 5), (6, 7), (9, 10)], False),
                ([(4, 9), (0, 5), (6, 5), (6, 0), (9, 0), (7, 2), (7, 4), (7, 11)], False),
                ([(0, 0), (0, 5), (4, 5), (4, 4), (2, 4), (2, 1), (4, 1), (4, 0)], True),
                ([(7, 0), (7, 9), (0, 9), (0, 8), (3, 8), (3, 4), (0, 4), (0, 0)], True),
                ([(9, 0), (0, 0), (0, 7), (2, 7), (2, 4), (5, 4), (5, 7), (9, 7)], True)]
    for points, closed in fallback:
        if kind == "walk" and closed or kind in ("notch", "convex") and not closed:
            continue
        for code in rng.permutation(8):
            transformed = _normalize(_transform(points, int(code)))
            try:
                options = _options(transformed, closed, rank, offsets, minimum, witness, elsewhere)
            except AssertionError:
                continue
            if options and (kind != "notch" or options[0]["concave"]) and (kind != "convex" or not options[0]["concave"]):
                return options[int(rng.integers(len(options)))]
    raise AssertionError("no feasible ranked shadow for requested geometry")


def _footprint(obj):
    return obj["pixels"] | obj["translated"] | obj["key"]


def _shift(obj, dr, dc):
    result = dict(obj)
    for k in ("pixels", "translated", "key", "selected"):
        result[k] = {(r + dr, c + dc) for r, c in obj[k]}
    result["points"] = [(r + dr, c + dc) for r, c in obj["points"]]
    result["fragments"] = [{(r + dr, c + dc) for r, c in g} for g in obj["fragments"]]
    return result


def _pack(rng, objects):
    occupied, placed = set(), []
    # Stroke packing permits a small drawing inside a larger contour's cavity.
    for obj in sorted(objects, key=lambda s: -len(_footprint(s))):
        pixels = _footprint(obj)
        r0, c0 = min(r for r, c in pixels), min(c for r, c in pixels)
        h = max(r for r, c in pixels) - r0 + 1
        w = max(c for r, c in pixels) - c0 + 1
        options = [(r - r0, c - c0) for r in range(31 - h) for c in range(31 - w)]
        rng.shuffle(options)
        for dr, dc in options:
            moved = {(r + dr, c + dc) for r, c in pixels}
            if not moved.intersection(occupied):
                placed.append(_shift(obj, dr, dc))
                occupied.update((r + a, c + b) for r, c in moved for a in (-1, 0, 1) for b in (-1, 0, 1))
                break
        else:
            return None
    return placed


def _draw(rng, objects, background, anchor_color):
    occupied = set().union(*(_footprint(s) for s in objects))
    r0, c0 = min(r for r, c in occupied), min(c for r, c in occupied)
    h, w = max(r for r, c in occupied) - r0 + 1, max(c for r, c in occupied) - c0 + 1
    extras = [int(rng.integers(0, min(6, 30 - d) + 1)) for d in (h, w)]
    before = [int(rng.integers(x + 1)) for x in extras]
    objects = [_shift(s, before[0] - r0, before[1] - c0) for s in objects]
    h, w = h + extras[0], w + extras[1]
    inp = np.full((h, w), background, dtype=np.int8)
    palette = [int(c) for c in rng.permutation(10) if c not in (background, anchor_color)]
    for s in objects:
        s["color"] = anchor_color if s.get("anchor") else palette.pop()
        s["key_color"] = palette.pop()
        for p in s["points"]:
            inp[p] = s["color"]
        for p in s["key"]:
            inp[p] = s["key_color"]
    while np.count_nonzero(inp == background) <= inp.size // 2:
        axis = int(inp.shape[1] == 30 or (inp.shape[0] < 30 and inp.shape[0] <= inp.shape[1]))
        _need(min(inp.shape) < 30, "cannot pad background")
        inp = np.pad(inp, ((0, 1), (0, 0)) if axis else ((0, 0), (0, 1)), constant_values=background)
    # Label rasterization follows proposed shapes rather than invoking solve.
    out = _answer(inp, objects, background)
    return {"input": inp, "output": out}


def _build_pair(rng, background, anchor_color, offsets, ranks, anchor_rank=None,
                witness=False, open_demo=False, concave_demo=False, minimum=2,
                avoid_concavity=False):
    for _ in range(20):
        objects = []
        for i, rank in enumerate(ranks):
            kind = "convex" if witness and i == 0 else "walk" if open_demo and i == 0 else "notch" if concave_demo and i == 0 else None
            if kind is None and avoid_concavity:
                kind = str(rng.choice(["convex", "walk"]))
            obj = _make_object(rng, rank, offsets, 5 if len(ranks) >= 3 else 7,
                               kind, minimum if i == 0 else max(2, rank),
                               witness and i == 0, open_demo and i == 0)
            obj["anchor"] = anchor_rank is not None and i == 0
            objects.append(obj)
        placed = _pack(rng, objects)
        if placed is None:
            continue
        pair = _draw(rng, placed, background, anchor_color)
        if np.array_equal(solve(pair["input"]), pair["output"]):
            return pair
        raise AssertionError("proposal and input-only decoder disagree")
    # Fewer simultaneous objects is a sampling repair, never a changed rule.
    return _build_pair(rng, background, anchor_color, offsets, ranks[:1], anchor_rank,
                       witness, open_demo, concave_demo, minimum, avoid_concavity)


def _count(name, value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _signature(grid):
    return grid.shape, np.ascontiguousarray(grid, dtype=np.int8).tobytes()


def _repair_training(rng, puzzle, demos, anchor_color, train_offsets, shift):
    if not _spatial_shortcuts(puzzle["train"]):
        return
    index = demos[1]
    old = puzzle["train"][index]
    _, bg, sources = _decode(old["input"])
    ranks = [2] + [s["rank"] for s in sources if s["color"] != anchor_color]
    _, _, other_sources = _decode(puzzle["train"][demos[0]]["input"])
    other_offset = next(s["offset"] for s in other_sources if s["color"] == anchor_color)
    offsets = tuple(d for d in train_offsets if d != other_offset)
    seen = {_signature(p["input"]) for j, p in enumerate(puzzle["train"]) if j != index}
    for _ in range(40):
        candidate = _build_pair(rng, bg, anchor_color, offsets, ranks, 2,
                                open_demo=True, minimum=3,
                                avoid_concavity=shift == "concavity")
        if _signature(candidate["input"]) in seen:
            continue
        puzzle["train"][index] = candidate
        if not _spatial_shortcuts(puzzle["train"]):
            return
    raise AssertionError("position-shortcut contrast repair exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt, nq = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    anchor_color = int(rng.integers(10))
    shift = str(rng.choice(["direction", "ordinal", "concavity"]))
    held = DIRECTIONS[int(rng.integers(8))]
    train_offsets = tuple(d for d in DIRECTIONS if shift != "direction" or d != held)
    demos = [int(i) for i in rng.choice(nt, 2, replace=False)]
    counts = [int(rng.integers(1, 4)) for _ in range(nt + nq)]
    counts[demos[1]] = int(rng.choice([c for c in (1, 2, 3) if c != counts[demos[0]]]))
    puzzle, seen, familiar = {"train": [], "test": []}, set(), set()
    first_demo_offset = None
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            demo = demos.index(index) + 1 if split == "train" and index in demos else 0
            rank = demo if demo else (3 if split == "test" and shift == "ordinal" else int(rng.integers(1, 3)))
            source_count = counts[index if split == "train" else nt + index]
            ranks = [rank] + [int(rng.integers(1, 3)) for _ in range(source_count - 1)]
            offsets = train_offsets if split == "train" else (held,) if shift == "direction" else tuple(sorted(familiar))
            if demo and first_demo_offset is not None:
                offsets = tuple(d for d in offsets if d != first_demo_offset)
            minimum = 3 if demo == 2 else int(rng.integers(3, 5)) if split == "test" and shift == "ordinal" else 2
            bg = int(rng.choice([c for c in range(10) if c != anchor_color]))
            for _ in range(20):
                pair = _build_pair(rng, bg, anchor_color, offsets, ranks,
                                   demo or None, demo == 1, demo == 2,
                                   split == "test" and shift == "concavity", minimum,
                                   split == "train" and shift == "concavity")
                if _signature(pair["input"]) not in seen:
                    break
            else:
                raise AssertionError("duplicate-input retry exhausted")
            seen.add(_signature(pair["input"]))
            puzzle[split].append(pair)
            if split == "train":
                _, _, sources = _decode(pair["input"])
                familiar.update(s["offset"] for s in sources)
                if demo:
                    first_demo_offset = next(s["offset"] for s in sources if s["color"] == anchor_color)
        if split == "train":
            # Rebuild a teaching contrast before selecting familiar query offsets.
            _repair_training(rng, puzzle, demos, anchor_color, train_offsets, shift)
            familiar = {s["offset"] for p in puzzle["train"] for s in _decode(p["input"])[2]}
            seen = {_signature(p["input"]) for p in puzzle["train"]}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _need(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
          and len(puzzle["train"]) > len(puzzle["test"]), "invalid pair counts")
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _need(inp.shape == out.shape, "canvas size changed")
            signature = _signature(inp)
            _need(signature not in seen, "duplicate input")
            seen.add(signature)
            _, bg, sources = _decode(inp)
            _need(np.array_equal(out, _answer(inp, sources, bg)), "incorrect source completion or ranked shadow")
            _need(np.array_equal(out[inp != bg], inp[inp != bg]), "existing pixels changed")
            for s in sources:
                _need(any(inp[p] == bg for p in s["pixels"]), "source has no missing stroke")
                _need(sum(inp[p] == bg for p in s["selected"]) >= 2, "selection is not visible")
                _need(sum(inp[p] == bg for g in s["fragments"] if g != s["selected"] for p in g) >= 2,
                      "ignoring rank would give the same answer")
            (training if split == "train" else testing).extend(sources)
    # These are teaching obligations, separate from per-input solve semantics.
    ranks = {s["rank"] for s in training}
    _need({1, 2} <= ranks, "training must establish ordinal progression")
    _need(any(1 < s["rank"] < len(s["fragments"]) for s in training), "no intermediate rank demonstration")
    _need(any(s["chords"] and s["closed"] for s in training), "contour chord rejection not taught")
    _need(any(_naive_pixels(s) - s["pixels"] - s["key"] for s in training), "chord rejection has no visible consequence")
    _need(any(not s["closed"] for s in training), "open-path completion not taught")
    _need(any(not _straight(s["selected"]) for s in training), "ranking fragments versus individual edges not taught")
    _need(any(len(_components(s["selected"], False)) > 1 for s in training), "diagonal fragment connectivity not taught")
    _need(any(not s["key"].intersection(s["selected"]) for s in training), "nearest-marker shortcut not distinguished")
    histories = {}
    for s in training:
        histories.setdefault(s["color"], set()).add(s["rank"])
    _need(any({1, 2} <= rs for rs in histories.values()), "same source identity must change requested rank")
    _need(not _spatial_shortcuts(puzzle["train"]), "position-based ranking shortcut survives training")
    for s in testing:
        _need(s["rank"] in ranks or set(range(1, s["rank"])) <= ranks, "query ordinal lacks training progression")
    offsets = {s["offset"] for s in training}
    _need(any(s["offset"] not in offsets or s["rank"] not in ranks
              or (s["concave"] and not any(t["concave"] for t in training)) for s in testing),
          "no meaningful direction, ordinal, or concavity query shift")


def _puzzle_signature(puzzle):
    return tuple(tuple((_signature(p["input"]), _signature(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test():
    """A bounded 42-seed semantic/variation suite, plus metamorphic probes."""
    # Hand-rasterized square: exposed component sizes are 9 and 5. The two
    # instructions keep the same unit displacement but select opposite Ls.
    for rank in (1, 2):
        inp = np.zeros((8, 8), dtype=np.int8)
        for p in ((1, 1), (1, 5), (5, 1), (5, 5)):
            inp[p] = 2
        for j in range(rank):
            inp[6 + j, 6 + j] = 3
        out = inp.copy()
        out[1, 1:6] = out[5, 1:6] = 2
        out[1:6, 1] = out[1:6, 5] = 2
        if rank == 1:
            out[6, 2:7] = out[2:7, 6] = 3
        else:
            out[2, 2:5] = out[2:5, 2] = 3
        _need(np.array_equal(solve(inp), out), "hand-rasterized ranked occlusion regression")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    rows = [measure_pair(p) for puzzle in puzzles for split in ("train", "test") for p in puzzle[split]]
    _need(len({_puzzle_signature(p) for p in puzzles}) == 42, "seed diversity")
    _need({x["background"] for x in rows} == set(range(10)), "background role variation")
    _need({tuple(d) for x in rows for d in x["offsets"]} == set(DIRECTIONS), "all unit displacements")
    _need({1, 2, 3} <= {x["sources"] for x in rows}, "source-count variation")
    _need({1, 2, 3} <= {r for x in rows for r in x["ranks"]}, "ordinal variation")
    _need({2, 3, 4} <= {n for x in rows for n in x["populations"]}, "fragment populations independent of rank")
    _need(len({tuple(x["shape"]) for x in rows}) >= 40, "canvas variation")
    _need(len({x["geometry"] for x in rows}) >= 100, "geometry variation")
    for metric in ("concave", "open", "closed", "chords", "unit_links", "bent_selections",
                   "diagonal_connectivity", "key_elsewhere", "intermediate", "last_ranked", "occluded_gap_cells", "boundary"):
        _need(sum(x[metric] for x in rows) > 0, f"missing variation: {metric}")
    _need(all(len({measure_pair(p)["geometry"] for p in puzzle["train"]}) >= 2 for puzzle in puzzles), "within-puzzle structure variation")
    _need(_puzzle_signature(puzzles[0]) == _puzzle_signature(generate_puzzle(0)), "determinism")
    for nt, nq in ((2, 1), (10, 4), (7, 3)):
        validate_generated_puzzle(generate_puzzle(113 + nt, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (True, 1), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Equivariance checks use only transformed inputs and their solved labels.
    palette = np.array([7, 3, 8, 1, 9, 0, 5, 2, 6, 4], dtype=np.int8)
    for pair in puzzles[0]["train"]:
        for k in range(4):
            for reflected in (False, True):
                a, b = np.rot90(pair["input"], k), np.rot90(pair["output"], k)
                if reflected:
                    a, b = np.fliplr(a), np.fliplr(b)
                _need(np.array_equal(solve(palette[a]), palette[b]), "rotation/reflection/palette invariance")
    bad = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    pair = bad["test"][0]
    p = tuple(np.argwhere(pair["input"] != pair["output"])[0])
    pair["output"][p] = pair["input"][p]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        return
    raise AssertionError("incorrect label accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds)")
