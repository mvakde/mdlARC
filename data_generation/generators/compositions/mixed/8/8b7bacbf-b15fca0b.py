"""A shortest orthogonal route selects terminal-owned loop cavities.

Reserved colors: 0 free space, 1 wall, 2 navigation terminals, 4 route.
All other colors are exchangeable outline/string and paint-source roles.
No file access, parent imports, hidden palettes, or retained random state.
"""
from __future__ import annotations

from collections import Counter, deque
import hashlib
import json

import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))
INKS = (3, 5, 6, 7, 8, 9)


class _Reject(Exception):
    """An internal bounded construction attempt did not fit."""


def _neighbors(cells, steps=D8):
    cells = set(cells)
    return {(r + dr, c + dc) for r, c in cells for dr, dc in steps} - cells


def _components(cells, steps=D8):
    left = set(cells)
    groups = []
    while left:
        first = min(left)
        left.remove(first)
        group, queue = {first}, [first]
        for r, c in queue:
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in left:
                    left.remove(q)
                    group.add(q)
                    queue.append(q)
        groups.append(group)
    return groups


def _cells(a, value):
    return {tuple(map(int, p)) for p in np.argwhere(a == value)}


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("grids must be integer arrays, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _shortest(free, start, goal):
    """BFS with minimum-path multiplicities capped at two, and one witness."""
    if start not in free or goal not in free:
        return [], 0
    distance, ways, previous = {start: 0}, {start: 1}, {start: None}
    queue = deque([start])
    while queue:
        p = queue.popleft()
        for dr, dc in D4:
            q = (p[0] + dr, p[1] + dc)
            if q not in free:
                continue
            nd = distance[p] + 1
            if q not in distance:
                distance[q], ways[q], previous[q] = nd, ways[p], p
                queue.append(q)
            elif distance[q] == nd:
                ways[q] = min(2, ways[q] + ways[p])
    if goal not in distance:
        return [], 0
    route, p = [], goal
    while p is not None:
        route.append(p)
        p = previous[p]
    return route[::-1], ways[goal]


def _turns(route):
    ds = [(q[0] - p[0], q[1] - p[1]) for p, q in zip(route, route[1:])]
    return sum(x != y for x, y in zip(ds, ds[1:]))


def _alternate_route(free, route):
    # On a unique shortest path, a surviving route after deleting one of its
    # intermediate cells must be strictly longer. Try every cell, bounded by 900.
    for p in route[1:-1]:
        other, _ = _shortest(free - {p}, route[0], route[-1])
        if other:
            return other
    return []


def _longer_route(free, route):
    other = _alternate_route(free, route)
    return len(other) - len(route) if other else 0


def _analyze(value):
    """Reconstruct every semantic object from this input alone."""
    a = _grid(value)
    if np.any(a == 4):
        raise ValueError("route color 4 must be absent from inputs")
    terminals = sorted(_cells(a, 2))
    if len(terminals) != 2:
        raise ValueError("exactly two navigation terminals of color 2 are required")
    free = _cells(a, 0) | set(terminals)
    route, count = _shortest(free, *terminals)
    if count != 1 or len(route) < 3:
        raise ValueError("the orthogonal terminal route must be uniquely shortest")

    systems = _components({tuple(map(int, p)) for p in np.argwhere(np.isin(a, INKS))})
    labels, owners = {}, []
    for index, cells in enumerate(systems):
        counts = Counter(int(a[p]) for p in cells)
        singleton = [col for col, n in counts.items() if n == 1]
        if len(counts) != 2 or len(singleton) != 1:
            raise ValueError("each disjoint loop system needs one distinct singleton source")
        paint = singleton[0]
        source = next(p for p in sorted(cells) if int(a[p]) == paint)
        if len(_neighbors({source}) & (cells - {source})) != 1:
            raise ValueError("paint sources must be terminal tips of their systems")
        body_color = next(col for col in counts if col != paint)
        owners.append({"source": source, "paint": paint, "body_color": body_color,
                       "cells": cells, "body": cells - {source}})
        labels.update((p, index) for p in cells)
    if not owners:
        raise ValueError("at least one terminal-owned loop system is required")

    h, w = a.shape
    route_cells = set(route[1:-1])
    contact_band = _neighbors(route_cells, D4)
    cavities = []
    for inner in _components(_cells(a, 0), D4):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in inner):
            continue
        shell = _neighbors(inner, D4)
        ownership = {labels.get(p, -1) for p in shell}
        if len(ownership) != 1 or -1 in ownership:
            continue  # A maze chamber bounded by wall 1 is not a loop cavity.
        owner = next(iter(ownership))
        if any(int(a[p]) != owners[owner]["body_color"] for p in shell):
            raise ValueError("a cavity must have a complete outline, without source gaps")
        cavities.append({"inner": inner, "shell": shell, "owner": owner,
                         "selected": bool(shell & contact_band)})
    if {c["owner"] for c in cavities} != set(range(len(owners))):
        raise ValueError("each paint system must own at least one enclosed cavity")
    return {"grid": a, "route": route, "free": free, "owners": owners,
            "cavities": cavities, "contact_band": contact_band}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Draw the unique route, then fill only edge-contacted cavity outlines."""
    info = _analyze(input_grid)
    out = info["grid"].copy()
    for p in info["route"][1:-1]:
        out[p] = 4
    for cavity in info["cavities"]:
        if cavity["selected"]:
            paint = info["owners"][cavity["owner"]]["paint"]
            for p in cavity["inner"]:
                out[p] = paint
    return out


def _maze(rng, expanded):
    """Random weighted lattice tree plus cycles, not a prescribed slalom."""
    nr = int(rng.integers(3, 5 if not expanded else 6))
    nc = int(rng.integers(3, 5 if not expanded else 6))
    rows, cols = [int(rng.integers(1, 4))], [int(rng.integers(1, 4))]
    for _ in range(nr - 1):
        rows.append(rows[-1] + int(rng.integers(5, 8)))
    for _ in range(nc - 1):
        cols.append(cols[-1] + int(rng.integers(5, 8)))
    shape = (rows[-1] + int(rng.integers(2, 4)), cols[-1] + int(rng.integers(2, 4)))
    if max(shape) > 30:
        raise _Reject()
    nodes = [(r, c) for r in range(nr) for c in range(nc)]
    all_edges = [(p, (p[0] + dr, p[1] + dc)) for p in nodes
                 for dr, dc in ((1, 0), (0, 1))
                 if p[0] + dr < nr and p[1] + dc < nc]
    first = nodes[int(rng.integers(len(nodes)))]
    seen, stack, edges = {first}, [first], []
    while stack:
        p = stack[-1]
        choices = [(p[0] + dr, p[1] + dc) for dr, dc in D4
                   if (p[0] + dr, p[1] + dc) in nodes
                   and (p[0] + dr, p[1] + dc) not in seen]
        if not choices:
            stack.pop()
            continue
        q = choices[int(rng.integers(len(choices)))]
        edges.append(tuple(sorted((p, q))))
        seen.add(q)
        stack.append(q)
    unused = [e for e in all_edges if tuple(sorted(e)) not in edges]
    rng.shuffle(unused)
    edges.extend(unused[:int(rng.integers(1, min(3, len(unused)) + 1))])
    a = np.ones(shape, dtype=np.int8)
    for p, q in edges:
        r0, r1 = sorted((rows[p[0]], rows[q[0]]))
        c0, c1 = sorted((cols[p[1]], cols[q[1]]))
        a[r0:r1 + 1, c0:c1 + 1] = 0
    free = _cells(a, 0)
    positions = [(rows[r], cols[c]) for r, c in nodes]
    for _ in range(36):
        ix = rng.choice(len(positions), size=2, replace=False)
        start, goal = (positions[int(i)] for i in ix)
        route, count = _shortest(free, start, goal)
        manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        if (count == 1 and len(route) >= 18 and _turns(route) >= 2
                and len(route) - 1 > manhattan):
            alternate = _alternate_route(free, route)
            if not alternate:
                continue
            a[start] = a[goal] = 2
            return a, route, alternate
    raise _Reject()


def _shape(rng):
    kind = int(rng.integers(5))
    h, w = int(rng.integers(1, 4)), int(rng.integers(1, 5))
    if kind == 0:
        radius = int(rng.integers(1, 3))
        inner = {(r, c) for r in range(-radius, radius + 1)
                 for c in range(-radius, radius + 1) if abs(r) + abs(c) <= radius}
    elif kind == 1 and h >= 2 and w >= 2:
        inner = {(r, c) for r in range(h) for c in range(w)
                 if r == h - 1 or c == 0}
    else:
        inner, left = set(), 0
        for r in range(h):
            if kind == 2 and r:
                left += int(rng.choice((-1, 0, 1)))
            width = max(1, w - (kind == 3 and r == 0))
            inner.update((r, c) for c in range(left, left + width))
    if len(_components(inner, D4)) != 1:
        inner = {(0, 0)}
    shell = _neighbors(inner, D4)
    r0, c0 = min(r for r, c in shell), min(c for r, c in shell)
    return {(r - r0, c - c0) for r, c in inner}


def _candidates(a, route, alternate, rng):
    walls = _cells(a, 1)
    near4, near8 = _neighbors(route[1:-1], D4), _neighbors(route[1:-1], D8)
    alternate_band = _neighbors(alternate[1:-1], D4)
    shapes = {tuple(sorted(_shape(rng))) for _ in range(18)}
    shapes.add(((1, 1),))
    result = {True: [], False: []}
    h, w = a.shape
    for normalized in sorted(shapes):
        inner = set(normalized)
        shell = _neighbors(inner, D4)
        footprint = inner | shell
        hh, ww = max(r for r, c in footprint) + 1, max(c for r, c in footprint) + 1
        for dr in range(1, h - hh):
            for dc in range(1, w - ww):
                fp = {(r + dr, c + dc) for r, c in footprint}
                if not fp <= walls:
                    continue
                ins = {(r + dr, c + dc) for r, c in inner}
                edge = fp - ins
                selected = bool(edge & near4)
                result[selected].append((ins, edge, bool(edge & near8), bool(edge & alternate_band)))
    if not result[True] or not result[False]:
        raise _Reject()
    return result


def _join(starts, goals, allowed, rng):
    queue = deque(sorted(starts))
    previous = {p: None for p in starts}
    steps = [D8[int(i)] for i in rng.permutation(8)]
    end = None
    while queue:
        p = queue.popleft()
        if p in goals:
            end = p
            break
        for dr, dc in steps:
            q = (p[0] + dr, p[1] + dc)
            if q in allowed and q not in previous:
                previous[q] = p
                queue.append(q)
    if end is None:
        raise _Reject()
    path = set()
    while end is not None:
        path.add(end)
        end = previous[end]
    return path


def _install_systems(a, route, alternate, plans, paints, rng, diagonal_decoy, alternate_decoy):
    candidates = _candidates(a, route, alternate, rng)
    wall = _cells(a, 1)
    occupied, bodies, sources, cavities = set(), [], [], []
    for owner, flags in enumerate(plans):
        blocked = occupied | _neighbors(occupied)
        body, holes = set(), set()
        for flag in flags:
            choices = candidates[flag]
            accepted = False
            for index in rng.permutation(len(choices))[:100]:
                inner, shell, diag, alt = choices[int(index)]
                fp = inner | shell
                if fp & (blocked | holes | body | _neighbors(body)):
                    continue
                if owner == 0 and not flag and diagonal_decoy and not diag:
                    continue
                if owner == 0 and not flag and alternate_decoy and not alt:
                    continue
                try:
                    wire = (_join(body, shell, wall - blocked - holes - inner, rng)
                            if body else set())
                except _Reject:
                    continue
                body |= shell | wire
                holes |= inner
                cavities.append((inner, shell, owner, flag))
                accepted = True
                break
            if not accepted:
                raise _Reject()
        tips = []
        # A single differently colored tip identifies paint without any hidden
        # key. Prefer a truly diagonal last link to teach eight-connectivity.
        for p in sorted((_neighbors(body) & wall) - blocked - holes - body):
            ns = _neighbors({p}) & body
            if len(ns) == 1:
                q = next(iter(ns))
                if p[0] != q[0] and p[1] != q[1]:
                    tips.append(p)
        if not tips:
            raise _Reject()
        source = tips[int(rng.integers(len(tips)))]
        bodies.append(body)
        sources.append(source)
        occupied |= body | holes | {source}
    out = a.copy()
    for owner, (body, source, paint) in enumerate(zip(bodies, sources, paints)):
        body_color = int(rng.choice([c for c in INKS if c != paint]))
        for p in body:
            a[p] = body_color
        a[source] = paint
    for inner, shell, owner, selected in cavities:
        for p in inner:
            a[p] = 0
    out[:] = a
    for p in route[1:-1]:
        out[p] = 4
    for inner, shell, owner, selected in cavities:
        if selected:
            for p in inner:
                out[p] = paints[owner]
    return a, out


def _make_pair(rng, mode, teaching_paints):
    expanded = mode == "test"
    if mode == "single":
        plans, paints = [[True, False]], [teaching_paints[0]]
    elif mode == "multi":
        plans = [[True, False], [True]]
        paints = [teaching_paints[1], teaching_paints[0]]
    elif expanded:
        plans = [[True, False], [True], [bool(rng.integers(2))]]
        if rng.random() < .45:
            plans[int(rng.integers(3))].append(bool(rng.integers(2)))
        paints = list(map(int, rng.choice(INKS, size=3, replace=False)))
    else:
        n = int(rng.integers(1, 3))
        plans = [[True, False]] + ([[True]] if n == 2 else [])
        if rng.random() < .4:
            plans[int(rng.integers(n))].append(bool(rng.integers(2)))
        paints = list(map(int, rng.choice(INKS, size=n, replace=False)))
    # Retry whole geometric attempts without relaxing any semantic requirement.
    for _ in range(100):
        try:
            a, route, alternate = _maze(rng, expanded)
            a, b = _install_systems(a, route, alternate, plans, paints, rng,
                                   mode == "single", mode in ("multi", "test"))
            k, flip = int(rng.integers(4)), bool(rng.integers(2))
            a, b = np.rot90(a, k), np.rot90(b, k)
            if flip:
                a, b = np.fliplr(a), np.fliplr(b)
            a, b = a.copy(), b.copy()
            # Expected outputs above use construction ownership; here the
            # input-only parser must reconstruct it without that metadata.
            info = _analyze(a)
            if len(info["cavities"]) != sum(map(len, plans)):
                continue
            if mode in ("multi", "test") and not _features(info)["alternate_decoys"]:
                continue
            if np.array_equal(solve(a), b):
                return {"input": a, "output": b}
        except (_Reject, ValueError):
            continue
    raise RuntimeError("bounded placement exhausted without changing the rule")


def _count(name, value, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count("num_train", num_train, 2, 10)
    ne = _count("num_test", num_test, 1, 4)
    if nt <= ne:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    teaching_paints = list(map(int, rng.choice(INKS, size=2, replace=False)))
    modes = ["single", "multi"] + ["varied"] * (nt - 2)
    rng.shuffle(modes)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, plan in (("train", modes), ("test", ["test"] * ne)):
        for mode in plan:
            for _ in range(12):
                pair = _make_pair(rng, mode, teaching_paints)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input attempts exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    cavities, owners, route = info["cavities"], info["owners"], info["route"]
    selected = [c for c in cavities if c["selected"]]
    rejected = [c for c in cavities if not c["selected"]]
    source_cuts = 0
    for i, owner in enumerate(owners):
        four = next(g for g in _components(owner["cells"], D4) if owner["source"] in g)
        source_cuts += any(c["owner"] == i and c["selected"] and not (four & c["shell"])
                           for c in cavities)
    diagonal_band = _neighbors(route[1:-1], D8)
    alternate = _alternate_route(info["free"], route)
    alternate_band = _neighbors(alternate[1:-1], D4)
    shaped = []
    for c in cavities:
        mr = min(r for r, col in c["inner"])
        mc = min(col for r, col in c["inner"])
        shaped.append(tuple(sorted((r - mr, col - mc) for r, col in c["inner"])))
    return {"shape": info["grid"].shape, "systems": len(owners), "cavities": len(cavities),
            "selected": len(selected), "unselected": len(rejected),
            "paints": sorted({owners[c["owner"]]["paint"] for c in selected}),
            "areas": [len(c["inner"]) for c in cavities], "contours": shaped,
            "route_length": len(route) - 1, "turns": _turns(route),
            "diagonal_decoys": sum(bool(c["shell"] & diagonal_band) for c in rejected),
            "alternate_decoys": sum(bool(c["shell"] & alternate_band) for c in rejected),
            "eight_neighbor_sources": source_cuts,
            "mixed_owner": any(any(c["owner"] == i for c in selected)
                               and any(c["owner"] == i for c in rejected)
                               for i in range(len(owners)))}


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise ValueError("puzzle must contain exactly train/test")
    if not all(isinstance(puzzle[s], list) for s in ("train", "test")):
        raise ValueError("splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _count("num_train", nt, 2, 10)
    _count("num_test", ne, 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    seen, measurements, training_paints = set(), {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            info = _analyze(a)
            f = _features(info)
            measurements[split].append(f)
            assert np.array_equal(solve(a), b), "input-derived rule disagrees with output"
            assert np.all(a[a != b] == 0), "only background may change"
            assert f["selected"] and f["unselected"] and f["mixed_owner"]
            assert f["eight_neighbor_sources"] >= 1
            assert f["turns"] >= 2 and f["route_length"] >= 17
            assert _longer_route(info["free"], info["route"]) > 0
            assert len(info["free"]) > len(info["route"])
            route_only = a.copy()
            for p in info["route"][1:-1]:
                route_only[p] = 4
            assert not np.array_equal(route_only, b), "omitting A must change the answer"
            fill_all = route_only.copy()
            for c in info["cavities"]:
                for p in c["inner"]:
                    fill_all[p] = info["owners"][c["owner"]]["paint"]
            assert not np.array_equal(fill_all, b), "route gating must change the cavity answer"
            if split == "train":
                training_paints.update(f["paints"])
                assert f["systems"] <= 2
            else:
                assert f["systems"] == 3
    fs = measurements["train"]
    assert len(training_paints) >= 2, "demonstrate copying, not one constant paint"
    assert any(f["systems"] == 1 and f["diagonal_decoys"] for f in fs)
    assert any(len(f["paints"]) >= 2 for f in fs)
    assert any(f["alternate_decoys"] for f in fs)
    assert all(f["alternate_decoys"] for f in measurements["test"])


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds; retain only metrics and hashes, never grid collections."""
    shapes, contours, palettes, signatures, areas, routes, counts = (set() for _ in range(7))
    totals = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                f = _features(info)
                shapes.add(f["shape"])
                contours.update(f["contours"])
                areas.update(f["areas"])
                routes.add(f["route_length"])
                counts.add(f["cavities"])
                palettes.add(tuple((o["body_color"], o["paint"]) for o in info["owners"]))
                totals["pairs"] += 1
                totals["diagonal_decoys"] += f["diagonal_decoys"]
                totals["alternate_decoys"] += f["alternate_decoys"]
                totals["selected_cavities"] += f["selected"]
                totals["unselected_cavities"] += f["unselected"]
                totals["eight_neighbor_sources"] += f["eight_neighbor_sources"]
        if seed in (0, 17, 41):
            assert _digest(puzzle) == _digest(generate_puzzle(seed))
    assert len(signatures) == 42
    assert len(shapes) >= 25 and len(contours) >= 12
    assert len(palettes) >= 50 and len(areas) >= 5 and len(routes) >= 20
    assert {2, 3, 4}.issubset(counts)
    assert totals["diagonal_decoys"] >= 42
    # Every legal explicit count combination, including minimum demonstrations.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + nt * 10 + ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Ties and disconnected endpoints must not acquire an arbitrary tiebreaker.
    assert _shortest({(r, c) for r in range(3) for c in range(3)}, (0, 0), (2, 2))[1] == 2
    assert _shortest({(0, 0), (2, 2)}, (0, 0), (2, 2))[1] == 0
    bad = generate_puzzle(0, 2, 1)
    p = bad["test"][0]
    info = _analyze(p["input"])
    target = next(c for c in info["cavities"] if c["selected"])
    q = min(target["inner"])
    p["output"][q] = 0
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("partial cavity fill accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shapes": len(shapes),
            "contour_shapes": len(contours), "palette_assignments": len(palettes),
            "cavity_areas": sorted(areas), "cavity_counts": sorted(counts),
            "route_lengths": len(routes), "totals": dict(totals),
            "independent_semantic_review": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
