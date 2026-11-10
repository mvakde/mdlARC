"""Beam-locked docking. All inference uses the supplied input pixels only.

No files, examples, RNG state, or generation metadata are read by solve().
Generation and self_test keep their bounded diagnostics in memory.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


DIAGONALS = ((-1, -1), (-1, 1), (1, 1), (1, -1))
ORTHOGONAL = ((-1, 0), (0, 1), (1, 0), (0, -1))
NEIGHBORS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _signature(cells):
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r-r0, c-c0, int(v)) for (r, c), v in cells.items()))


def _transform(cells, t):
    result = {}
    for (r, c), v in cells.items():
        if t >= 4:
            c = -c
        for _ in range(t % 4):
            r, c = c, -r
        result[(r, c)] = int(v)
    r0, c0 = min(r for r, _ in result), min(c for _, c in result)
    return {(r-r0, c-c0): v for (r, c), v in result.items()}


def _translate(cells, delta):
    return {(r+delta[0], c+delta[1]): v for (r, c), v in cells.items()}


def _halo(points):
    return {(r+dr, c+dc) for r, c in points for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _components(points):
    remaining = set(points)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, component = [start], {start}
        while todo:
            r, c = todo.pop()
            for dr, dc in NEIGHBORS:
                p = r+dr, c+dc
                if p in remaining:
                    remaining.remove(p)
                    component.add(p)
                    todo.append(p)
        result.append(component)
    return result


def _background(shape, colors):
    return np.asarray(colors, dtype=np.int8)[np.indices(shape).sum(axis=0) % 2]


def _infer_background(grid):
    parity = np.indices(grid.shape).sum(axis=0) % 2
    colors = []
    for k in (0, 1):
        counts = np.bincount(grid[parity == k].astype(int), minlength=10)
        color = int(counts.argmax())
        assert counts[color] * 2 > int((parity == k).sum()), "unclear checkerboard"
        colors.append(color)
    assert colors[0] != colors[1]
    return tuple(colors)


def _outline(center, radius):
    r, c = center
    return {(r+dr, c+dc) for dr in range(-radius, radius+1)
            for dc in range(-radius, radius+1) if abs(dr)+abs(dc) == radius}


def _cap(center, radius, direction):
    r, c = center
    vr, vc = direction
    return [(r+vr*(k+1), c+vc*(radius-k+1)) for k in range(radius+1)]


def _emitter_cells(e):
    result = {p: e["color"] for p in _outline(e["center"], e["radius"])}
    for d, color in e["caps"].items():
        for p in _cap(e["center"], e["radius"], d):
            assert p not in result
            result[p] = color
    return result


def _rays(emitters, shape, whole_cap=False):
    result = []
    for index, e in enumerate(emitters):
        for d, color in sorted(e["caps"].items()):
            cap = _cap(e["center"], e["radius"], d)
            starts = cap if whole_cap else (cap[0], cap[-1])
            for endpoint in starts:
                p = endpoint[0]+d[0], endpoint[1]+d[1]
                path = []
                while _inside(p, shape):
                    path.append(p)
                    p = p[0]+d[0], p[1]+d[1]
                result.append({"emitter": index, "direction": d, "color": color,
                               "path": path, "points": set(path)})
    return result


def _terminals(cells, body, registration):
    core = {p for p, v in cells.items() if v in (body, registration)}
    assert core and any(v == body for v in cells.values())
    r0, r1 = min(r for r, _ in core), max(r for r, _ in core)
    c0, c1 = min(c for _, c in core), max(c for _, c in core)
    corners = {(r0-1, c0-1): (-1, -1), (r0-1, c1+1): (-1, 1),
               (r1+1, c1+1): (1, 1), (r1+1, c0-1): (1, -1)}
    labels = {p: (v, corners[p]) for p, v in cells.items() if p not in core and p in corners}
    assert len(labels) == len(cells)-len(core) == 2, "two exterior corner terminals required"
    assert all((r, c) in core for r in (r0, r1) for c in (c0, c1))
    assert len(_components(cells)) == 1
    return labels


def _trace(emitters, placements, shape, body, registration, require_all=True, whole_cap=False):
    """Trace against the final scene, never against the scrambled sources."""
    fixed, occupied, terminals, gaps = {}, {}, {}, set()
    for e in emitters:
        cells = _emitter_cells(e)
        assert not fixed.keys() & cells.keys()
        fixed.update(cells)
    for cells in placements:
        assert not cells.keys() & (fixed.keys() | occupied.keys())
        occupied.update(cells)
        terminals.update(_terminals(cells, body, registration))
        core = {p for p, v in cells.items() if v in (body, registration)}
        gaps.update({(r, c) for r in range(min(r for r, _ in core), max(r for r, _ in core)+1)
                     for c in range(min(c for _, c in core), max(c for _, c in core)+1)} - core)
    claims, hits, records = {}, set(), []
    for ray in _rays(emitters, shape, whole_cap):
        path, hit = [], None
        d, color = ray["direction"], ray["color"]
        for p in ray["path"]:
            assert p not in fixed, "beam hits another emitter"
            if p in occupied:
                assert terminals.get(p) == (color, (-d[0], -d[1])), "unmatched beam contact"
                assert p not in hits, "two beams share a terminal"
                hits.add(p)
                hit = p
                break
            assert p not in claims, "beam paths intersect"
            assert p not in gaps, "layout sends a beam through a receiver gap"
            claims[p] = color
            path.append(p)
        records.append({"emitter": ray["emitter"], "direction": d,
                        "color": color, "path": path, "hit": hit})
    if require_all:
        assert hits == terminals.keys(), "unlit terminal"
    return claims, hits, records


def _analyze(grid):
    grid = np.asarray(grid)
    assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
    bg = _infer_background(grid)
    base = _background(grid.shape, bg)
    components = _components(map(tuple, np.argwhere(grid != base)))
    by_cell = {p: i for i, comp in enumerate(components) for p in comp}
    emitter_matches = {}
    h, w = grid.shape
    for radius in (2, 3, 4):
        if min(h, w) < 2*radius+1:
            continue
        sh = h-2*radius, w-2*radius
        offsets = sorted(_outline((0, 0), radius))
        dr, dc = offsets[0]
        colors = grid[radius+dr:radius+dr+sh[0], radius+dc:radius+dc+sh[1]]
        mask = (colors != bg[0]) & (colors != bg[1])
        for dr, dc in offsets[1:]:
            mask &= grid[radius+dr:radius+dr+sh[0], radius+dc:radius+dc+sh[1]] == colors
        for rr, cc in np.argwhere(mask):
            center = int(rr)+radius, int(cc)+radius
            e = {"center": center, "radius": radius, "color": int(colors[rr, cc]), "caps": {}}
            for d in DIAGONALS:
                cap = _cap(center, radius, d)
                if not all(_inside(p, grid.shape) for p in cap):
                    continue
                values = {int(grid[p]) for p in cap}
                if len(values) == 1 and not values & {*bg, e["color"]}:
                    e["caps"][d] = next(iter(values))
            if not e["caps"]:
                continue
            footprint = _emitter_cells(e)
            index = by_cell[min(footprint)]
            if footprint.keys() == components[index] and all(int(grid[p]) == v for p, v in footprint.items()):
                assert index not in emitter_matches, "ambiguous emitter"
                emitter_matches[index] = e
    assert 2 <= len(emitter_matches) <= 4, "expect two to four complete capped diamonds"
    emitters = sorted(emitter_matches.values(), key=lambda e: e["center"])
    sources, targets, target_colors = [], [], set()
    for i, component in enumerate(components):
        if i in emitter_matches:
            continue
        cells = {p: int(grid[p]) for p in component}
        if len(set(cells.values())) == 1:
            targets.append(set(cells))
            target_colors.update(cells.values())
        else:
            sources.append(cells)
    assert 1 <= len(sources) == len(targets) <= 2
    assert len(target_colors) == 1
    registration = next(iter(target_colors))
    cap_colors = {c for e in emitters for c in e["caps"].values()}
    structs = {e["color"] for e in emitters}
    assert len(structs) == 1 and not cap_colors & {*bg, registration, *structs}
    body_colors = {v for s in sources for v in s.values()} - cap_colors - {registration}
    assert len(body_colors) == 1
    body = next(iter(body_colors))
    assert body not in {*bg, registration, *structs, *cap_colors}
    for s in sources:
        assert registration in s.values()
        assert set(v for v, _ in _terminals(s, body, registration).values()) <= cap_colors
        assert len(_components(p for p, v in s.items() if v == registration)) == 1
    fixed = {p: v for e in emitters for p, v in _emitter_cells(e).items()}
    return {"background": bg, "base": base, "emitters": emitters, "fixed": fixed,
            "sources": sources, "targets": targets, "body": body, "registration": registration,
            "rays": _rays(emitters, grid.shape), "shape": grid.shape}


def _candidates(source, info, beams=True):
    candidates, seen = [], set()
    reg, body = info["registration"], info["body"]
    for t in range(8):
        cells = _transform(source, t)
        marks = {p for p, v in cells.items() if v == reg}
        for index, target in enumerate(info["targets"]):
            if len(marks) != len(target):
                continue
            a, b = min(marks), min(target)
            delta = b[0]-a[0], b[1]-a[1]
            if {(r+delta[0], c+delta[1]) for r, c in marks} != target:
                continue
            placed = _translate(cells, delta)
            if any(not _inside(p, info["shape"]) for p in placed) or placed.keys() & info["fixed"].keys():
                continue
            terminals = _terminals(placed, body, reg)
            if beams and any(not any(p in ray["points"] and color == ray["color"] and
                                     direction == tuple(-v for v in ray["direction"])
                                     for ray in info["rays"])
                             for p, (color, direction) in terminals.items()):
                continue
            signature = index, tuple(sorted(placed.items()))
            if signature not in seen:
                seen.add(signature)
                candidates.append((index, placed))
    return candidates


def _solutions(info, beams=True, limit=2):
    choices = [_candidates(s, info, beams) for s in info["sources"]]
    order = sorted(range(len(choices)), key=lambda i: len(choices[i]))
    answers = {}

    def search(depth, used, occupied, placements):
        if len(answers) >= limit:
            return
        if depth == len(order):
            try:
                claims, _, records = _trace(info["emitters"], placements, info["shape"],
                                             info["body"], info["registration"]) if beams else ({}, set(), [])
            except AssertionError:
                return
            signature = tuple(sorted({**occupied, **claims}.items()))
            answers[signature] = (list(placements), claims, records)
            return
        for index, cells in choices[order[depth]]:
            if index in used or _halo(cells) & occupied.keys():
                continue
            search(depth+1, used | {index}, {**occupied, **cells}, placements+[cells])

    search(0, set(), {}, [])
    return list(answers.values())


def _render_final(info, solution):
    output = info["base"].copy()
    placements, claims, _ = solution
    for cells in [info["fixed"], *placements, claims]:
        for p, color in cells.items():
            output[p] = color
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover objects, enumerate rigid docking poses, and resolve beam contacts."""
    info = _analyze(np.asarray(input_grid))
    answers = _solutions(info)
    assert len(answers) == 1, "input must have exactly one resulting output"
    return _render_final(info, answers[0])


def _path(rng, a, b):
    r, c = a
    points = {a}
    while (r, c) != b:
        if r != b[0] and (c == b[1] or rng.random() < .5):
            r += 1 if b[0] > r else -1
        else:
            c += 1 if b[1] > c else -1
        points.add((r, c))
    return points


def _tile(rng, palette, directions, teaching, small):
    body, reg = palette[2:4]
    h, w = (int(rng.integers(2, 5 if small else 6)) for _ in range(2))
    if teaching:
        # Cardinal domino: its symmetries leave beam orientation necessary,
        # while its axis resolves the diagonal reflection of opposite terminals.
        marks = {(0, 0), (0, 1)}
        if rng.random() < .5:
            marks = {(0, 0), (1, 0)}
    else:
        marks = {(0, 0)}
        target_size = int(rng.integers(1, min(5, h*w-3)+1))
        for _ in range(30):
            if len(marks) >= target_size:
                break
            p = sorted(marks)[int(rng.integers(len(marks)))]
            dr, dc = NEIGHBORS[int(rng.integers(len(NEIGHBORS)))]
            trial = marks | {(p[0]+dr, p[1]+dc)}
            r0, c0 = min(r for r, _ in trial), min(c for _, c in trial)
            trial = {(r-r0, c-c0) for r, c in trial}
            if max(r for r, _ in trial) < h and max(c for _, c in trial) < w:
                marks = trial
    mh, mw = 1+max(r for r, _ in marks), 1+max(c for _, c in marks)
    offset = int(rng.integers(h-mh+1)), int(rng.integers(w-mw+1))
    marks = {(r+offset[0], c+offset[1]) for r, c in marks}
    endpoints = [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1), *sorted(marks)]
    rng.shuffle(endpoints)
    core = {endpoints[0]}
    for p in endpoints[1:]:
        anchor = sorted(core)[int(rng.integers(len(core)))]
        core.update(_path(rng, anchor, p))
    density = float(rng.uniform(.45, .95))
    for _ in range(h*w):
        if len(core) >= density*h*w:
            break
        frontier = sorted({(r+dr, c+dc) for r, c in core for dr, dc in ORTHOGONAL
                           if 0 <= r+dr < h and 0 <= c+dc < w} - core)
        if not frontier:
            break
        core.add(frontier[int(rng.integers(len(frontier)))])
    cells = {p: reg if p in marks else body for p in core}
    assert sum(v == body for v in cells.values()) >= 2
    colors = list(map(int, rng.choice(palette[5:], 2, replace=False)))
    for d, color in zip(directions, colors):
        cells[(-1 if d[0] < 0 else h, -1 if d[1] < 0 else w)] = color
    return _transform(cells, 0)


def _build_emitters(rng, finals, shape, palette, radius_max, teaching, extra):
    terminals = {p: item for cells in finals for p, item in _terminals(cells, palette[2], palette[3]).items()}
    pending = list(terminals)
    rng.shuffle(pending)
    occupied = set().union(*(set(cells) for cells in finals))
    emitters, fixed, served = [], set(), set()
    for p in pending:
        if p in served:
            continue
        color, q = terminals[p]
        d = -q[0], -q[1]
        found = False
        options = [(radius, k, endpoint) for radius in ([radius_max] if teaching else range(2, radius_max+1))
                   for k in range(3 if teaching else 2, 9) for endpoint in (0, 1)]
        rng.shuffle(options)
        for radius, k, endpoint in options:
            a, b = (1, radius+1) if endpoint == 0 else (radius+1, 1)
            center = p[0]-d[0]*(k+a), p[1]-d[1]*(k+b)
            e = {"center": center, "radius": radius, "color": palette[4], "caps": {d: color}}
            cells = _emitter_cells(e)
            if any(not _inside(point, shape) for point in cells) or _halo(cells) & (occupied | fixed):
                continue
            try:
                _, hits, _ = _trace(emitters+[e], finals, shape, palette[2], palette[3], require_all=False)
            except AssertionError:
                continue
            if p not in hits:
                continue
            emitters.append(e)
            fixed.update(cells)
            served = hits
            found = True
            break
        if not found:
            return None
    if not 2 <= len(emitters) <= 4:
        return None
    additions = 0
    if extra:
        options = [(i, d) for i, e in enumerate(emitters) for d in DIAGONALS if d not in e["caps"]]
        rng.shuffle(options)
        target = int(rng.integers(1, 4))
        for i, d in options:
            old = emitters[i]
            e = {**old, "caps": {**old["caps"], d: int(rng.choice(palette[5:]))}}
            cells = _emitter_cells(e)
            other = occupied | set().union(*(set(_emitter_cells(x)) for j, x in enumerate(emitters) if j != i))
            if any(not _inside(p, shape) for p in cells) or _halo(cells) & other:
                continue
            trial = emitters[:i]+[e]+emitters[i+1:]
            try:
                _trace(trial, finals, shape, palette[2], palette[3])
            except AssertionError:
                continue
            emitters = trial
            additions += 1
            if additions >= target:
                break
        if not additions:
            return None
    return emitters


def _registration_matters(info, solution):
    """Exhibit a different beam-valid placement if stencil alignment is omitted."""
    placements = solution[0]
    for i, cells in enumerate(placements):
        directions = [d for _, d in _terminals(cells, info["body"], info["registration"]).values()]
        if directions[0] != tuple(-x for x in directions[1]):
            continue
        for sign in (-1, 1):
            delta = tuple(sign*x for x in directions[0])
            moved = _translate(cells, delta)
            if any(not _inside(p, info["shape"]) for p in moved):
                continue
            others = [p for j, p in enumerate(placements) if j != i]
            if any(_halo(moved) & other.keys() for other in others):
                continue
            try:
                claims, _, records = _trace(info["emitters"], others+[moved], info["shape"], info["body"], info["registration"])
            except AssertionError:
                continue
            alternative = _render_final(info, (others+[moved], claims, records))
            if not np.array_equal(alternative, _render_final(info, solution)):
                return True
    return False


def _hollow_witnesses(info, solution):
    output = _render_final(info, solution)
    directions = set()
    for e in info["emitters"]:
        for d in e["caps"]:
            probes = [(p[0]+d[0]*k, p[1]+d[1]*k)
                      for p in _cap(e["center"], e["radius"], d)[1:-1] for k in (1, 2)]
            related = [r for r in solution[2] if r["direction"] == d and
                       info["emitters"][r["emitter"]] is e]
            if related and min(len(r["path"]) for r in related) >= 2 and all(
                    _inside(p, info["shape"]) and output[p] == info["base"][p] for p in probes):
                directions.add((d, e["radius"]))
    return directions


def _make_pair(rng, n, radius_max, directions=None, teaching=None, extra=False):
    palette = list(map(int, rng.permutation(10)[:9]))
    low = max(20 if n == 1 else 26, 2*radius_max+20 if teaching else 20)
    shape = tuple(int(rng.integers(min(low, 30), 31)) for _ in range(2))
    finals, occupied = [], set()
    sharing = n == 2 and rng.random() < .35
    shared_terminal = None
    for i in range(n):
        ds = directions if i == 0 and directions is not None else tuple(
            DIAGONALS[j] for j in map(int, rng.choice(4, 2, replace=False)))
        if sharing and i == 1:
            _, (_, q) = shared_terminal
            ds = (q, [d for d in ds if d != q][0])
        cells = _tile(rng, palette, ds, bool(teaching), n > 1)
        h, w = 1+max(r for r, _ in cells), 1+max(c for _, c in cells)
        if sharing and i == 1:
            anchor, (color, q) = shared_terminal
            local = _terminals(cells, palette[2], palette[3])
            pad = next(p for p, (_, d) in local.items() if d == q)
            for p, (_, d) in local.items():
                cells[p] = color if d == q else int(rng.choice([c for c in palette[5:] if c != color]))
            # Put the second receiver's matching pad on the other endpoint
            # lane. Axial staggering keeps the irregular bodies separate.
            origins = []
            for radius in range(2, radius_max+1):
                for sign in (-1, 1):
                    for shift in range(-9, 10):
                        p = (anchor[0]+sign*q[0]*radius+shift*q[0],
                             anchor[1]-sign*q[1]*radius+shift*q[1])
                        origin = p[0]-pad[0], p[1]-pad[1]
                        if 3 <= origin[0] < shape[0]-h-2 and 3 <= origin[1] < shape[1]-w-2:
                            origins.append(origin)
        else:
            origins = [(r, c) for r in range(3, shape[0]-h-2) for c in range(3, shape[1]-w-2)]
        rng.shuffle(origins)
        for origin in origins:
            placed = _translate(cells, origin)
            if not _halo(placed) & occupied:
                finals.append(placed)
                occupied.update(placed)
                if sharing and i == 0:
                    shared_terminal = list(_terminals(placed, palette[2], palette[3]).items())[int(rng.integers(2))]
                break
        else:
            return None
    emitters = _build_emitters(rng, finals, shape, palette, radius_max, bool(teaching), extra)
    if emitters is None:
        return None
    if sharing and len(emitters) != 3:
        return None
    fixed = {p: v for e in emitters for p, v in _emitter_cells(e).items()}
    targets = {p: v for cells in finals for p, v in cells.items() if v == palette[3]}
    blocked = _halo(fixed.keys() | targets.keys())
    sources = []
    for i, final in enumerate(finals):
        ts = list(map(int, rng.permutation(8)))
        if teaching and i == 0:
            ts = [t for t in ts if (0 < t < 4 if teaching == "rotation" else t >= 4)]
        found = False
        for t in ts:
            cells = _transform(final, t)
            if teaching == "reflection" and i == 0 and any(
                    _signature(_transform(cells, j)) == _signature(final) for j in range(4)):
                continue
            h, w = 1+max(r for r, _ in cells), 1+max(c for _, c in cells)
            origins = [(r, c) for r in range(shape[0]-h+1) for c in range(shape[1]-w+1)]
            rng.shuffle(origins)
            for origin in origins:
                placed = _translate(cells, origin)
                if placed.keys() & blocked or placed == final:
                    continue
                sources.append(placed)
                blocked.update(_halo(placed))
                found = True
                break
            if found:
                break
        if not found:
            return None
    inp = _background(shape, palette[:2])
    for cells in [fixed, targets, *sources]:
        for p, v in cells.items():
            inp[p] = v
    try:
        info = _analyze(inp)
        answers = _solutions(info)
        if len(answers) != 1:
            return None
        claims, _, records = _trace(emitters, finals, shape, palette[2], palette[3])
        intended = _render_final(info, (finals, claims, records))
        out = _render_final(info, answers[0])
        if not np.array_equal(out, intended):
            return None
        if teaching:
            required = {(tuple(-v for v in d), radius_max) for d in directions}
            if not required <= _hollow_witnesses(info, answers[0]):
                return None
            if len(_solutions(info, beams=False)) < 2 or not _registration_matters(info, answers[0]):
                return None
        return {"input": inp, "output": out}
    except AssertionError:
        return None


def _count(value, low, high, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    radius_max = int(rng.integers(2, 5))
    regime = str(rng.choice(["receivers", "caps"]))
    diagonal = int(rng.integers(2))
    teaching_slots = list(map(int, rng.choice(num_train, 2, replace=False)))
    teaching = {teaching_slots[0]: ("rotation", (DIAGONALS[diagonal], DIAGONALS[diagonal+2])),
                teaching_slots[1]: ("reflection", (DIAGONALS[1-diagonal], DIAGONALS[3-diagonal]))}
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            lesson, directions = teaching.get(index, (None, None)) if split == "train" else (None, None)
            n = (1 if split == "train" else 2) if regime == "receivers" else (
                1 if lesson else int(rng.integers(1, 3)))
            extra = split == "test" if regime == "caps" else (not lesson and rng.random() < .4)
            # Bounded free construction. Retry policy never changes requested
            # receiver count, teaching requirements, or test-shift condition.
            for _ in range(240):
                pair = _make_pair(rng, n, radius_max, directions, lesson, extra)
                if pair is not None and _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded scene construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect(pair):
    inp, out = (np.asarray(pair[k]) for k in ("input", "output"))
    info = _analyze(inp)
    answers = _solutions(info)
    assert len(answers) == 1, "ambiguous docking"
    solution = answers[0]
    assert np.array_equal(out, _render_final(info, solution)), "incorrect docking or beam pixels"
    assert inp.shape == out.shape and not np.array_equal(inp, out)
    assert all(int(out[p]) == v for p, v in info["fixed"].items())
    assert sum(r["hit"] is not None for r in solution[2]) == 2*len(info["sources"])
    assert all(any(r["hit"] is not None and r["emitter"] == i for r in solution[2])
               for i in range(len(info["emitters"]))), "disconnected decorative emitter"
    for source in info["sources"]:
        assert any(_signature(_transform(source, t)) == _signature(final)
                   for final in solution[0] for t in range(8)), "piece not preserved rigidly"
    moves = []
    for source in info["sources"]:
        final = next(p for p in solution[0] if any(_signature(_transform(source, t)) == _signature(p) for t in range(8)))
        ts = [t for t in range(8) if _signature(_transform(source, t)) == _signature(final)]
        moves.append("translation" if 0 in ts else "rotation" if any(t < 4 for t in ts) else "reflection")
    signatures = []
    for final in solution[0]:
        marks = {p: 1 for p, v in final.items() if v == info["registration"]}
        signatures.append(min(_signature(_transform(marks, t)) for t in range(8)))
    old = set().union(*(set(s) for s in info["sources"]))
    return info, solution, {"receivers": len(solution[0]), "emitters": len(info["emitters"]),
        "radii": [e["radius"] for e in info["emitters"]],
        "cap_counts": [len(e["caps"]) for e in info["emitters"]],
        "cap_masks": [sum(1 << DIAGONALS.index(d) for d in e["caps"]) for e in info["emitters"]],
        "lengths": [len(r["path"]) for r in solution[2]], "moves": moves,
        "shape": tuple(inp.shape), "patterns": signatures,
        "piece_shapes": [(1+max(r for r, _, _ in _signature(p)), 1+max(c for _, c, _ in _signature(p))) for p in solution[0]],
        "adjacent": [len({d[0] for _, d in _terminals(p, info["body"], info["registration"]).values()}) == 1 or
                     len({d[1] for _, d in _terminals(p, info["body"], info["registration"]).values()}) == 1 for p in solution[0]],
        "old_ray_overlap": len(old & solution[1].keys()), "background": info["background"],
        "shared_emitter": any(sum(r["hit"] is not None and r["emitter"] == j for r in solution[2]) >= 2
                              for j in range(len(info["emitters"]))),
        "hollow": _hollow_witnesses(info, solution),
        "beam_needed": len(_solutions(info, beams=False)) > 1,
        "registration_needed": _registration_matters(info, solution)}


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, evidence, train_stats, test_stats = set(), set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for value in pair.values():
                a = np.asarray(value)
                assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
                assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
            key = _key(pair["input"])
            assert key not in seen
            seen.add(key)
            info, solution, stats = _inspect(pair)
            (train_stats if split == "train" else test_stats).append(stats)
            if split == "train":
                evidence.update(stats["hollow"])
    assert {d for d, _ in evidence} == set(DIAGONALS), "teach hollow rays in all four directions"
    largest_test = max(r for s in test_stats for r in s["radii"])
    assert all(max(r for d, r in evidence if d == q) >= largest_test for q in DIAGONALS)
    assert any(s["beam_needed"] and s["registration_needed"] for s in train_stats)
    assert {"rotation", "reflection"} <= {m for s in train_stats for m in s["moves"]}
    assert (max(s["receivers"] for s in train_stats) < min(s["receivers"] for s in test_stats) or
            (all(max(s["cap_counts"]) == 1 for s in train_stats) and
             all(max(s["cap_counts"]) >= 2 for s in test_stats))), "missing structural test shift"


def self_test():
    """42 streaming seeds; no saved puzzles, traces, or unbounded retry logs."""
    # Small geometric regression with explicit endpoint lengths and cutoffs.
    # Expected pixels below do not call the cap, ray, or docking predictors.
    fixture = _background((25, 25), (0, 1))
    stationary = {}
    for center in ((5, 3), (19, 18)):
        for dr, dc in ((-2, 0), (-1, -1), (-1, 1), (0, -2),
                       (0, 2), (1, -1), (1, 1), (2, 0)):
            stationary[(center[0]+dr, center[1]+dc)] = 2
    stationary.update({(6, 6): 5, (7, 5): 5, (8, 4): 5,
                       (18, 15): 6, (17, 16): 6, (16, 17): 6})
    receiver = {(r, c): 3 for r in range(10, 13) for c in range(10, 14)
                if (r, c) != (10, 12)}
    receiver.update({(11, 11): 4, (11, 12): 4, (9, 9): 5, (13, 14): 6})
    scrambled = _translate(_transform(receiver, 5), (2, 17))
    for cells in (stationary, scrambled, {(11, 11): 4, (11, 12): 4}):
        for p, v in cells.items():
            fixture[p] = v
    expected = _background((25, 25), (0, 1))
    explicit_rays = {(7, 7): 5, (8, 8): 5, (15, 16): 6, (14, 15): 6}
    explicit_rays.update({(9+k, 5+k): 5 for k in range(16)})
    explicit_rays.update({(17-k, 14-k): 6 for k in range(15)})
    for cells in (stationary, receiver, explicit_rays):
        for p, v in cells.items():
            expected[p] = v
    assert np.array_equal(solve(fixture), expected)
    assert expected[8, 6] == 0 and expected[16, 15] == 1
    assert expected[10, 10] == 3 and expected[10, 12] == 0
    shapes, patterns, masks, backgrounds, moves = set(), set(), set(), set(), set()
    counts, radii, adjacencies, fingerprints, emitter_counts = set(), set(), set(), set(), set()
    caps, lengths, overlaps, shared, shifts = set(), [], 0, 0, Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        local = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                _, _, s = _inspect(pair)
                shapes.add(s["shape"])
                patterns.update(s["patterns"])
                masks.update(s["cap_masks"])
                backgrounds.add(s["background"])
                moves.update(s["moves"])
                counts.add(s["receivers"])
                emitter_counts.add(s["emitters"])
                shared += s["shared_emitter"]
                radii.update(s["radii"])
                caps.update(s["cap_counts"])
                adjacencies.update(s["adjacent"])
                lengths.extend(s["lengths"])
                overlaps += s["old_ray_overlap"] > 0
                local.append((split, s))
                digest.update(bytes(pair["input"].shape))
                digest.update(pair["input"].tobytes())
                digest.update(pair["output"].tobytes())
        fingerprints.add(digest.digest())
        shifts["receivers" if max(s["receivers"] for k, s in local if k == "train") <
               min(s["receivers"] for k, s in local if k == "test") else "caps"] += 1
    assert len(fingerprints) == 42 and len(shapes) >= 35
    assert counts == {1, 2} and radii == {2, 3, 4} and adjacencies == {False, True}
    assert caps == {1, 2, 3, 4} and masks == set(range(1, 16))
    assert emitter_counts == {2, 3, 4} and shared >= 5
    assert len(patterns) >= 10 and len(backgrounds) >= 30
    assert moves == {"translation", "rotation", "reflection"}
    assert max(lengths) >= 15 and min(lengths) <= 2 and overlaps >= 10
    assert shifts["receivers"] and shifts["caps"]
    # Input-only solving is unaffected by generation calls between predictions.
    first = generate_puzzle(555, 2, 1)
    again = generate_puzzle(555, 2, 1)
    assert all(_key(a[k]) == _key(b[k]) for s in first for a, b in zip(first[s], again[s]) for k in a)
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(10000+a*10+b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # A damaged answer must be rejected by family-specific verification.
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in first[s]] for s in first}
    cell = tuple(np.argwhere(bad["test"][0]["input"] != bad["test"][0]["output"])[0])
    bad["test"][0]["output"][cell] = bad["test"][0]["input"][cell]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a corrupted answer")
