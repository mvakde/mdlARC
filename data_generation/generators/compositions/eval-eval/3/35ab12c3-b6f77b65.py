"""Hollow castings: sparse plans specify independently falling material frames.

Only standard-library modules and NumPy are used. No parent code is imported.
"""
from __future__ import annotations

import itertools
import json
import numpy as np


OFFSETS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _line(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    n = max(abs(dr), abs(dc))
    assert n and (dr == 0 or dc == 0 or abs(dr) == abs(dc))
    sr, sc = int(np.sign(dr)), int(np.sign(dc))
    return frozenset((a[0] + k * sr, a[1] + k * sc) for k in range(n + 1))


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _intersect(a, b, c, d):
    if any(max(a[k], b[k]) < min(c[k], d[k]) or
           max(c[k], d[k]) < min(a[k], b[k]) for k in (0, 1)):
        return False
    return _cross(a, b, c) * _cross(a, b, d) <= 0 and _cross(c, d, a) * _cross(c, d, b) <= 0


def _topology(points):
    """Unique noncrossing spanning cycle, otherwise unique spanning path.

    Edge sets, rather than traversal direction/start, define distinct answers.
    An eligible edge cannot jump over another landmark.
    """
    assert 2 <= len(points) <= 10 and len(set(points)) == len(points)
    edges, rasters = [], []
    landmarks = set(points)
    for i, j in itertools.combinations(range(len(points)), 2):
        a, b = points[i], points[j]
        dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
        if dr == 0 or dc == 0 or dr == dc:
            cells = _line(a, b)
            if not (cells - {a, b}) & landmarks:
                edges.append((i, j))
                rasters.append(cells)
    adj = [[] for _ in points]
    for k, (i, j) in enumerate(edges):
        adj[i].append((j, k))
        adj[j].append((i, k))
    conflicts = [set() for _ in edges]
    for k, (a, b) in enumerate(edges):
        for l, (c, d) in enumerate(edges[:k]):
            if not {a, b} & {c, d} and _intersect(points[a], points[b], points[c], points[d]):
                conflicts[k].add(l)
                conflicts[l].add(k)

    def search(closed):
        answers = {}

        def walk(order, selected):
            if len(answers) > 1:
                return
            if len(order) == len(points):
                if closed:
                    closing = [e for v, e in adj[order[-1]] if v == order[0]]
                    if not closing or conflicts[closing[0]] & set(selected):
                        return
                    selected = selected + [closing[0]]
                answers[tuple(sorted(selected))] = order
                return
            for v, edge in adj[order[-1]]:
                if v not in order and not conflicts[edge] & set(selected):
                    walk(order + [v], selected + [edge])

        for start in ([0] if closed else range(len(points))):
            walk([start], [])
            if len(answers) > 1:
                break
        return answers

    cycles = search(True) if len(points) >= 3 else {}
    assert len(cycles) <= 1, "competing cycles"
    answers = cycles or search(False)
    assert len(answers) == 1, "landmarks do not determine one outline"
    chosen, order = next(iter(answers.items()))
    cells = frozenset().union(*(rasters[k] for k in chosen))
    turns = [_cross(points[order[i-1]], points[order[i]], points[order[(i+1) % len(order)]])
             for i in range(len(order))] if cycles else []
    return {"cells": cells, "closed": bool(cycles), "edges": [edges[k] for k in chosen],
            "order": order, "chords": len(edges) - len(chosen),
            "all_cells": frozenset().union(*rasters),
            "concave": any(t < 0 for t in turns) and any(t > 0 for t in turns)}


def _shift(cells, dr=0, dc=0):
    return frozenset((r + dr, c + dc) for r, c in cells)


def _settle(shapes, height):
    """Greatest mutually movable set descends one row per synchronous tick."""
    current = list(shapes)
    for _ in range(height * max(1, len(current)) + 1):
        targets = [_shift(s, 1) for s in current]
        moving = {i for i, s in enumerate(targets) if all(r < height for r, c in s)}
        while True:
            blocked = {i for i in moving if any(targets[i] & current[j]
                       for j in range(len(current)) if j not in moving)}
            if not blocked:
                break
            moving -= blocked
        if not moving:
            return current
        current = [targets[i] if i in moving else s for i, s in enumerate(current)]
    raise AssertionError("gravity did not converge")


def _settle_exhaustive(shapes, height):
    """Small exhaustive motion oracle; no support-elimination algorithm."""
    current = list(shapes)
    for _ in range(height * max(1, len(current)) + 1):
        targets, feasible = [_shift(s, 1) for s in current], []
        for mask in range(1, 1 << len(current)):
            moving = {i for i in range(len(current)) if mask & (1 << i)}
            if all(all(r < height for r, c in targets[i]) and
                   all(not targets[i] & current[j] for j in range(len(current)) if j not in moving)
                   for i in moving):
                feasible.append(moving)
        if not feasible:
            return current
        moving = set().union(*feasible)
        assert moving in feasible
        current = [targets[i] if i in moving else s for i, s in enumerate(current)]
    raise AssertionError("exhaustive gravity did not converge")


def _render(shape, background, band, objects, cells):
    result = np.full(shape, background, dtype=np.int8)
    result[0] = band
    occupied = set()
    for obj, footprint in zip(objects, cells):
        assert not occupied & footprint, "material castings overlap"
        occupied.update(footprint)
        for p in footprint:
            assert 2 <= p[0] < shape[0] and 0 <= p[1] < shape[1]
            result[p] = obj["material"]
    return result


def _decode(grid):
    assert grid.ndim == 2 and grid.shape[0] >= 4
    bg = int(grid[1, 0])
    assert np.all(grid[1] == bg), "missing blank control separator"
    commands = {int(c) for c in grid[0] if c != bg}
    assert 1 <= len(commands) <= 3
    assert sum(int(c) != bg for c in grid[0]) == len(commands)
    groups = {int(c): [tuple(map(int, p)) for p in np.argwhere(grid[2:] == c) + (2, 0)]
              for c in np.unique(grid[2:]) if c != bg}
    sources = {c: ps for c, ps in groups.items() if len(ps) >= 2}
    markers = {c: ps[0] for c, ps in groups.items() if len(ps) == 1}
    assert 1 <= len(sources) <= 4 and len(markers) == len(sources)
    attached = {}
    for material, marker in markers.items():
        anchors = [(c, p) for c, ps in sources.items() for p in ps
                   if max(abs(p[0]-marker[0]), abs(p[1]-marker[1])) == 1]
        assert len(anchors) == 1, "material marker needs exactly one adjacent landmark"
        source, anchor = anchors[0]
        assert source not in attached
        attached[source] = (material, (marker[0]-anchor[0], marker[1]-anchor[1]))
    objects, occupied = [], set()
    for source, points in sources.items():
        info = _topology(points)
        material, offset = attached[source]
        physical = _shift(info["cells"], *offset)
        assert all(2 <= r < grid.shape[0] and 0 <= c < grid.shape[1] for r, c in physical)
        assert not physical & occupied
        occupied.update(physical)
        objects.append({**info, "cells": physical, "source": source, "material": material,
                        "offset": offset, "points": points})
    return bg, commands, objects


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode only this input, discard plans, delete material identities, settle."""
    grid = np.asarray(input_grid)
    bg, commands, objects = _decode(grid)
    survivors = [o for o in objects if o["material"] not in commands]
    settled = _settle([o["cells"] for o in survivors], grid.shape[0])
    return _render(grid.shape, bg, grid[0], survivors, settled)


def _object(points, closed):
    points = [tuple(map(int, p)) for p in points]
    cells = frozenset().union(*(_line(a, b) for a, b in zip(points, points[1:])))
    if closed:
        cells |= _line(points[-1], points[0])
    inferred = _topology(points)
    assert inferred["cells"] == cells and inferred["closed"] == closed
    return {"points": points, "cells": cells, "closed": closed}


def _place(obj, dr, dc):
    return {**obj, "points": [(r+dr, c+dc) for r, c in obj["points"]],
            "cells": _shift(obj["cells"], dr, dc)}


def _frame(rng, height, width, kind=None):
    kind = int(rng.integers(3)) if kind is None else kind
    if kind == 0:
        points = [(0, 0), (0, width), (height, width), (height, 0)]
    elif kind == 1:
        a, b = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        points = [(0, a), (0, width-b), (b, width), (height, width), (height, 0), (a, 0)]
    else:
        a, b, d = width//3, 2*width//3, int(rng.integers(1, 3))
        points = [(0, 0), (0, a), (d, a), (d, b), (0, b),
                  (0, width), (height, width), (height, 0)]
    return _object(points, True)


def _cavity(rng, teaching=False, four=False):
    fh = int(rng.integers(8, 13))
    fw = fh if teaching else int(rng.integers(8, 15))
    h = int(rng.integers(fh+6, min(30, fh+16)+1))
    w = int(rng.integers(fw+6, min(30, fw+15)+1))
    left = int(rng.integers(2, w-fw-2))
    bottom, top = h-1, h-1-fh
    frame = _place(_frame(rng, fh, fw, 0 if teaching else None), top, left)
    n = int(rng.integers(2, 5))
    row = bottom-n-1
    start = left+2 if four else left + int(rng.integers(2, max(3, fw//2)))
    end = left+fw-2 if four else left + int(rng.integers(max(start-left+2, fw//2+1), fw-1))
    x = start if four else int(rng.integers(start, end+1))
    post = _object([(row+1, x), (bottom-1, x)], False)
    cargo = _object([(row, start), (row, end)], False)
    objects = [frame, post, cargo]
    if four:
        # A lower passenger rests on the frame, beside the removable post.
        # Removing frame and post releases both passengers: one falls one row,
        # the other falls farther and lands on the first. This is one network.
        objects.append(_object([(bottom-1, start+2), (bottom-1, end)], False))
    return (h, w), objects, [1], [0]


def _bridge(rng, release=False, four=True, large=False):
    leg = int(rng.integers(3, 10 if large else 8))
    fh, fw = int(rng.integers(5, 11)), int(rng.integers(7, 15))
    h = int(rng.integers(fh+leg+5, min(30, fh+leg+11)+1))
    w = int(rng.integers(fw+6, min(30, fw+15)+1))
    left = int(rng.integers(2, w-fw-2))
    bottom = h-leg-1
    frame = _place(_frame(rng, fh, fw), bottom-fh, left)
    xs = (left+int(rng.integers(1, 3)), left+fw-int(rng.integers(1, 3)))
    posts = [_object([(bottom+1, x), (h-1, x)], False) for x in xs]
    objects = posts + [frame]
    if four:
        if leg >= 4 and rng.random() < .4:
            # A shorter third post is a landing, not an initial support.
            short = int(rng.integers(2, leg-1))
            x = left+fw//2
            objects.append(_object([(h-short, x), (h-1, x)], False))
        else:
            objects.append(_object([(bottom-1, left+2), (bottom-1, left+fw-2)], False))
    return (h, w), objects, [0, 1] if release else [int(rng.integers(2))], []


def _random_shape(rng):
    for _ in range(80):
        mode = int(rng.integers(5))
        try:
            if mode == 0:
                dr, dc = OFFSETS[int(rng.integers(8))]
                n = int(rng.integers(2, 9))
                points, closed = [(0, 0), (dr*n, dc*n)], False
            elif mode == 1:
                a, b = int(rng.integers(2, 5)), int(rng.integers(2, 5))
                points, closed = [(0, a), (a, 2*a), (a+b, 2*a-b), (b, a-b)], True
            elif mode == 2:
                a, b = int(rng.integers(2, 5)), int(rng.integers(2, 5))
                points, closed = [(0, 0), (0, a), (b, a), (b, a+2)], False
            elif mode == 3:
                frame = _frame(rng, int(rng.integers(4, 9)), int(rng.integers(5, 10)))
                points, closed = frame["points"], True
            else:
                n = int(rng.integers(3, 7))
                points, closed = [(0, 0)], False
                for _ in range(n-1):
                    dr, dc = OFFSETS[int(rng.integers(8))]
                    length = int(rng.integers(1, 5))
                    points.append((points[-1][0]+dr*length, points[-1][1]+dc*length))
            # Rotate/reflection changes the whole plan, not the direction of gravity.
            code = int(rng.integers(8))
            transformed = []
            for r, c in points:
                if code & 4:
                    c = -c
                for _ in range(code % 4):
                    r, c = c, -r
                transformed.append((r, c))
            mr, mc = min(r for r, c in transformed), min(c for r, c in transformed)
            transformed = [(r-mr, c-mc) for r, c in transformed]
            if max(r for r, c in transformed) > 11 or max(c for r, c in transformed) > 14:
                continue
            return _object(transformed, closed)
        except AssertionError:
            continue
    return _object([(0, 0), (0, 4)], False)


def _pile(rng, multi=False):
    h, w = int(rng.integers(17, 31)), int(rng.integers(14, 31))
    objects = []
    n = int(rng.integers(2, 5))
    for _ in range(150):
        if len(objects) == n:
            break
        obj = _random_shape(rng)
        width = max(c for r, c in obj["cells"])+1
        left = int(rng.integers(1, w-width))
        item = _place(obj, 3, left)
        occupied = set().union(*(o["cells"] for o in objects))
        if item["cells"] & occupied:
            continue
        while all(r+1 < h for r, c in item["cells"]) and not _shift(item["cells"], 1) & occupied:
            item = _place(item, 1, 0)
        objects.append(item)
    assert len(objects) == n
    count = min(n-1, int(rng.integers(2, 4)) if multi else int(rng.integers(1, 4)))
    selected = [int(x) for x in rng.choice(n, size=count, replace=False)]
    absent = [int(rng.integers(n))] if count < 3 and rng.random() < .35 else []
    return (h, w), objects, selected, absent


def _encode(rng, scene, background=None):
    shape, objects, selected, absent_sources = scene
    h, w = shape
    physical = [o["cells"] for o in objects]
    assert all(not a & b for a, b in itertools.combinations(physical, 2))
    assert _settle(physical, h) == physical
    bg = int(rng.integers(10)) if background is None else background
    palette = [int(c) for c in rng.permutation(10) if c != bg]
    n = len(objects)
    source_colors, material_colors = palette[:n], palette[n:2*n]
    choices = []
    for obj in objects:
        candidates = []
        for dr, dc in OFFSETS:
            points = [(r-dr, c-dc) for r, c in obj["points"]]
            if not all(2 <= r < h and 0 <= c < w for r, c in points):
                continue
            for marker in obj["points"]:
                if marker in points or sum(max(abs(r-marker[0]), abs(c-marker[1])) == 1 for r, c in points) != 1:
                    continue
                candidates.append((points, marker, (dr, dc)))
        rng.shuffle(candidates)
        choices.append(candidates)
    assignment, visits = [], [0]

    def assign(i):
        visits[0] += 1
        if visits[0] > 8000:
            return False
        if i == n:
            return any(a[2][1] and j not in selected for j, a in enumerate(assignment))
        for points, marker, offset in choices[i]:
            if any((set(points) | {marker}) & (set(ps) | {m}) or
                   any(max(abs(marker[0]-r), abs(marker[1]-c)) == 1 for r, c in ps) or
                   any(max(abs(m[0]-r), abs(m[1]-c)) == 1 for r, c in points)
                   for ps, m, d in assignment):
                continue
            assignment.append((points, marker, offset))
            if assign(i+1):
                return True
            assignment.pop()
        return False

    assert assign(0), "no unambiguous material-marker layout"
    inp = np.full(shape, bg, dtype=np.int8)
    for i, (points, marker, offset) in enumerate(assignment):
        for p in points:
            inp[p] = source_colors[i]
        inp[marker] = material_colors[i]
    commands = [material_colors[i] for i in selected] + [source_colors[i] for i in absent_sources]
    assert 1 <= len(commands) <= 3 and len(commands) == len(set(commands))
    columns = rng.choice(w, size=len(commands), replace=False)
    for col, command in zip(columns, commands):
        inp[0, col] = command
    live = [{**obj, "material": material_colors[i]} for i, obj in enumerate(objects) if i not in selected]
    out = _render(shape, bg, inp[0], live, _settle([o["cells"] for o in live], h))
    _, _, decoded = _decode(inp)
    assert {(o["material"], o["cells"]) for o in decoded} == set(zip(material_colors, physical))
    assert np.array_equal(solve(inp), out)
    return {"input": inp, "output": out}


def _build(rng, kind, background=None):
    for _ in range(100):
        try:
            if kind == "cavity_demo":
                scene = _cavity(rng, teaching=True)
            elif kind == "bridge_demo":
                scene = _bridge(rng)
            elif kind == "joint_query":
                scene = _bridge(rng, release=True, large=True)
            elif kind == "cavity_query":
                shape, objects, _, _ = _cavity(rng, four=True)
                scene = shape, objects, [0, 1], []
            else:
                scene = _pile(rng, multi=kind == "pile_query")
            return _encode(rng, scene, background)
        except AssertionError:
            continue
    raise AssertionError(f"bounded construction exhausted: {kind}")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    bg = int(rng.integers(10))
    train = [_build(rng, "cavity_demo", bg),
             _build(rng, "bridge_demo", (bg+int(rng.integers(1, 10))) % 10)]
    seen = {_key(p["input"]) for p in train}
    for _ in range(nt-2):
        for _ in range(30):
            kind = str(rng.choice(["pile", "cavity_query", "bridge_demo", "joint_query"]))
            pair = _build(rng, kind)
            if _key(pair["input"]) not in seen:
                seen.add(_key(pair["input"]))
                train.append(pair)
                break
        else:
            raise AssertionError("distinct demonstrations exhausted")
    tests = []
    # The first query combines familiar support facts through either joint
    # release or unequal cascades in a single connected hollow construction.
    first_kind = str(rng.choice(["joint_query", "cavity_query"], p=[.65, .35]))
    kinds = [first_kind] + [str(rng.choice(["joint_query", "cavity_query", "pile_query"])) for _ in range(nq-1)]
    for kind in kinds:
        for _ in range(30):
            pair = _build(rng, kind)
            if _key(pair["input"]) not in seen:
                seen.add(_key(pair["input"]))
                tests.append(pair)
                break
        else:
            raise AssertionError("distinct queries exhausted")
    rng.shuffle(train)
    rng.shuffle(tests)
    puzzle = {"train": train, "test": tests}
    validate_generated_puzzle(puzzle)
    return puzzle


def _inside(p, polygon):
    # Boundary is excluded by callers using disjoint physical footprints.
    r, c = p
    inside = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if (a[0] > r) != (b[0] > r):
            crossing = a[1] + (r-a[0]) * (b[1]-a[1]) / (b[0]-a[0])
            if c < crossing:
                inside = not inside
    return inside


def _measure(pair):
    inp, out = np.asarray(pair["input"]), np.asarray(pair["output"])
    bg, commands, objects = _decode(inp)
    live = [o for o in objects if o["material"] not in commands]
    settled = _settle([o["cells"] for o in live], inp.shape[0])
    distances = [min(r for r, c in b)-min(r for r, c in a["cells"]) for a, b in zip(live, settled)]
    supports = [(i, j) for i, a in enumerate(objects) for j, b in enumerate(objects)
                if i != j and _shift(a["cells"], 1) & b["cells"]]
    final_by_color = {o["material"]: s for o, s in zip(live, settled)}
    redundant = any(sum(a == i for a, b in supports) >= 2 and
                    any(a == i and objects[b]["material"] in commands for a, b in supports) and
                    final_by_color.get(o["material"]) == o["cells"] for i, o in enumerate(objects))
    removed = [o for o in objects if o["material"] in commands]
    joint = len(removed) >= 2 and bool(any(distances)) and all(
        _settle([o["cells"] for o in objects if o["material"] != rem["material"]], inp.shape[0]) ==
        [o["cells"] for o in objects if o["material"] != rem["material"]] for rem in removed)
    nested = 0
    for outer in objects:
        if outer["closed"]:
            polygon = [(outer["points"][k][0]+outer["offset"][0],
                        outer["points"][k][1]+outer["offset"][1]) for k in outer["order"]]
            nested += sum(inner is not outer and all(_inside(p, polygon) for p in inner["cells"])
                          for inner in objects)
    no_gravity = _render(inp.shape, bg, inp[0], live, [o["cells"] for o in live])
    vertices_only = [_shift(o["points"], *o["offset"]) for o in live]
    sparse_answer = _render(inp.shape, bg, inp[0], live, _settle(vertices_only, inp.shape[0]))
    no_deletion = _render(inp.shape, bg, inp[0], objects, [o["cells"] for o in objects])
    source_deleted = [o for o in objects if o["source"] not in commands]
    wrong_key = _render(inp.shape, bg, inp[0], source_deleted,
                        _settle([o["cells"] for o in source_deleted], inp.shape[0]))
    rejected = 0
    for obj, distance in zip(live, distances):
        all_links = _shift(obj["all_cells"], obj["offset"][0]+distance, obj["offset"][1])
        rejected += sum(int(out[p]) != obj["material"] for p in all_links)
    return {"shape": tuple(inp.shape), "background": bg, "pieces": len(objects),
            "vertices": [len(o["points"]) for o in objects], "commands": len(commands),
            "offsets": [o["offset"] for o in objects], "closed": sum(o["closed"] for o in objects),
            "concave": sum(o["concave"] for o in objects), "chords": sum(o["chords"] for o in live),
            "rejected_chord_cells": rejected,
            "nested": nested, "support_edges": len(supports), "redundant_harmless": redundant,
            "joint_release": joint, "distances": distances, "removed": len(removed),
            "absent_source_key": any(o["source"] in commands and o["material"] not in commands for o in objects),
            "completion_needed": not np.array_equal(sparse_answer, out),
            "gravity_needed": not np.array_equal(no_gravity, out),
            "deletion_needed": not np.array_equal(no_deletion, out),
            "material_key_needed": not np.array_equal(wrong_key, out),
            "translation_needed": any(o["offset"][1] for o in live),
            "diagonal": any(a[0] != b[0] and a[1] != b[1] for o in objects
                            for i, j in o["edges"] for a, b in [(o["points"][i], o["points"][j])])}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = (np.asarray(pair[k]) for k in ("input", "output"))
            for grid in (inp, out):
                assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer) and np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape and np.array_equal(inp[:2], out[:2])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            bg, commands, objects = _decode(inp)
            physical = [o["cells"] for o in objects]
            assert _settle(physical, inp.shape[0]) == physical, "uncommanded stack is not stable"
            assert np.array_equal(solve(inp), out)
            live = [o for o in objects if o["material"] not in commands]
            settled = _settle([o["cells"] for o in live], inp.shape[0])
            assert _settle(settled, inp.shape[0]) == settled
            for obj, cells in zip(live, settled):
                d = min(r for r, c in cells)-min(r for r, c in obj["cells"])
                assert d >= 0 and cells == _shift(obj["cells"], d)
                assert np.count_nonzero(out[2:] == obj["material"]) == len(obj["cells"])
                visible_cols = set(np.argwhere(out[2:] == obj["material"])[:, 1].tolist())
                assert visible_cols == {c for r, c in obj["cells"]}
            for obj in objects:
                assert not np.any(out[2:] == obj["source"]), "unconsumed blueprint"
                if obj["material"] in commands:
                    assert not np.any(out[2:] == obj["material"])
            features[split].append(_measure(pair))
    train = features["train"]
    assert any(f["gravity_needed"] and f["completion_needed"] and f["nested"] and f["rejected_chord_cells"] for f in train)
    assert any(f["redundant_harmless"] for f in train)
    assert any(f["absent_source_key"] and f["material_key_needed"] for f in train)
    assert all(any(f[name] for f in train) for name in ("completion_needed", "deletion_needed", "gravity_needed", "translation_needed"))
    assert len({f["background"] for f in train}) >= 2
    assert any(f["removed"] >= 2 and f["gravity_needed"] and
               (f["joint_release"] or len(set(f["distances"])-{0}) >= 2) for f in features["test"])


def _fingerprint(puzzle):
    return tuple((_key(p["input"]), _key(p["output"])) for s in ("train", "test") for p in puzzle[s])


def _mechanics_regressions():
    # Analytic outcomes: a complete empty square catches an interior beam.
    frame = _object([(3, 1), (3, 9), (11, 9), (11, 1)], True)
    post = _line((8, 3), (10, 3))
    beam = _line((7, 3), (7, 7))
    assert _topology(frame["points"])["chords"] == 2
    assert _settle([frame["cells"], post, beam], 12) == [frame["cells"], post, beam]
    assert _settle([frame["cells"], beam], 12) == [frame["cells"], _shift(beam, 3)]
    # Remove enclosure and post: a lower passenger drops one row and catches
    # the upper one after a three-row drop, rather than a common translation.
    lower = _line((10, 6), (10, 7))
    assert _settle([frame["cells"], post, beam, lower], 12) == [frame["cells"], post, beam, lower]
    assert _settle([beam, lower], 12) == [_shift(beam, 3), _shift(lower, 1)]
    # Either tall leg suffices. Both removals release the frame onto a shorter
    # remaining post, two rows below, rather than onto the floor four rows below.
    bridge = _object([(3, 1), (3, 9), (7, 9), (7, 1)], True)["cells"]
    legs = [_line((8, c), (11, c)) for c in (2, 8)]
    short = _line((10, 5), (11, 5))
    for leg in legs:
        assert _settle([bridge, leg, short], 12) == [bridge, leg, short]
    assert _settle([bridge, short], 12) == [_shift(bridge, 2), short]


def self_test():
    """Bounded, input-reconstructed invariants and diversity on 42 fixed seeds."""
    _mechanics_regressions()
    records, signatures, shapes = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_fingerprint(puzzle))
        for pair in puzzle["train"] + puzzle["test"]:
            records.append(_measure(pair))
            shapes.add(pair["input"].shape)
            _, commands, objects = _decode(pair["input"])
            live = [o["cells"] for o in objects if o["material"] not in commands]
            assert _settle(live, pair["input"].shape[0]) == _settle_exhaustive(live, pair["input"].shape[0])
    assert len(signatures) == 42 and len(shapes) >= 45
    assert {f["background"] for f in records} == set(range(10))
    assert {f["pieces"] for f in records} >= {2, 3, 4}
    assert {d for f in records for d in f["offsets"]} == set(OFFSETS)
    assert len({n for f in records for n in f["vertices"]}) >= 5
    assert len({d for f in records for d in f["distances"] if d}) >= 5
    assert all(any(f[name] for f in records) for name in ("concave", "diagonal", "joint_release", "nested"))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            puzzle = generate_puzzle(7000+nt*10+nq, nt, nq)
            assert len(puzzle["train"]) == nt and len(puzzle["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    assert _fingerprint(generate_puzzle(12345)) == _fingerprint(generate_puzzle(12345))
    # Validate that the validator notices an incorrect material cell.
    puzzle = generate_puzzle(0)
    out = puzzle["test"][0]["output"]
    p = tuple(np.argwhere(out[2:] != out[1, 0])[0] + (2, 0))
    out[p] = int(out[1, 0])
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "pairs": len(records), "shapes": len(shapes),
            "piece_counts": sorted({f["pieces"] for f in records}),
            "offsets": sorted({d for f in records for d in f["offsets"]}),
            "joint_releases": sum(f["joint_release"] for f in records),
            "nested_pairs": sum(bool(f["nested"]) for f in records)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
