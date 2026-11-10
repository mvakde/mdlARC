"""Self-programming constellations: reflected probes activate rotation counters.

Only the original input is traced. Reached mirror lengths are accumulated by
mirror color, then the corresponding *source* masks rotate around the board.
No parent generator, saved puzzle, seed, or builder metadata is used by solve.
"""
from __future__ import annotations

import numpy as np


_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(a, bg):
    left = set(map(tuple, np.argwhere(a != bg).tolist()))
    result = []
    while left:
        start = min(left)
        left.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in left and a[p] == a[start]:
                    left.remove(p)
                    stack.append(p)
        result.append((int(a[start]), tuple(sorted(cells))))
    return result


def _parse(a, *, require_matching=True):
    if a.shape[0] != a.shape[1]:
        raise ValueError("the constellation board must be square")
    hist = np.bincount(a.ravel(), minlength=10)
    bg = int(hist.argmax())
    if hist[bg] <= a.size / 2:
        raise ValueError("background must be a strict majority")
    sources, mirrors = [], []
    for color, cells in _components(a, bg):
        r0, c0 = np.min(cells, axis=0).tolist()
        r1, c1 = np.max(cells, axis=0).tolist()
        if len(cells) == 3 and r1-r0 == c1-c0 == 1:
            missing = next(p for p in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)) if p not in cells)
            elbow = (r0+r1-missing[0], c0+c1-missing[1])
            sources.append({"color": color, "cells": cells, "elbow": elbow,
                            "direction": (elbow[0]-missing[0], elbow[1]-missing[1])})
        elif (len(cells) in (1, 3, 5, 7, 9) and (r0 == r1 or c0 == c1)
              and max(r1-r0, c1-c0)+1 == len(cells)):
            mirrors.append({"color": color, "cells": cells, "length": len(cells),
                            "mid": ((r0+r1)//2, (c0+c1)//2),
                            "axis": None if len(cells) == 1 else (0 if r0 == r1 else 1)})
        else:
            raise ValueError("every monochromatic component must be an L or an odd straight bar")
    if not sources:
        raise ValueError("at least one L source is required")
    palette = {s["color"] for s in sources}
    if require_matching and any(m["color"] not in palette for m in mirrors):
        raise ValueError("each mirror must match a source color")
    return bg, sources, mirrors


def _face(m, p, d):
    for axis in (0, 1):
        if m["axis"] is not None and m["axis"] != axis:
            continue
        if p[1-axis] == m["mid"][1-axis] and m["mid"][axis]-p[axis] == d[axis]:
            return axis
    return None


def _trace(a, sources, mirrors, *, reflect=True):
    """Input-only ray events. Crossings between probes have no physical effect."""
    n = len(a)
    occupied = {p for o in sources+mirrors for p in o["cells"]}
    routes, reached = [], set()
    for s in sources:
        d = s["direction"]
        p = (s["elbow"][0]+d[0], s["elbow"][1]+d[1])
        states, events, path = set(), [], []
        while 0 <= p[0] < n and 0 <= p[1] < n:
            state = p, d
            if state in states:
                raise ValueError("a probe loops")
            states.add(state)
            if p in occupied:
                raise ValueError("a probe strikes an object rather than a midpoint face")
            hits = [(i, _face(m, p, d)) for i, m in enumerate(mirrors)]
            hits = [(i, axis) for i, axis in hits if axis is not None]
            if len(hits) > 1:
                raise ValueError("ambiguous simultaneous midpoint contacts")
            if hits:
                i, axis = hits[0]
                if reflect and i in reached:
                    raise ValueError("a mirror must be contacted at most once in the whole scene")
                reached.add(i)
                events.append((i, axis, p))
                if reflect:
                    d = (-d[0], d[1]) if axis == 0 else (d[0], -d[1])
            path.append(p)
            p = p[0]+d[0], p[1]+d[1]
        routes.append({"events": events, "path": path})
    return routes, reached


def _totals(mirrors, reached):
    counts = {}
    for i in sorted(reached):
        m = mirrors[i]
        counts[m["color"]] = counts.get(m["color"], 0) + m["length"]
    return counts


def _rotate_sources(a, bg, sources, counts):
    out = np.full(a.shape, bg, dtype=np.int8)
    source_only = out.copy()
    for s in sources:
        for p in s["cells"]:
            source_only[p] = s["color"]
    for color in sorted({s["color"] for s in sources}):
        mask = np.rot90(source_only == color, -counts.get(color, 0) % 4)
        if np.any(mask & (out != bg)):
            raise ValueError("different rotated source layers overlap")
        out[mask] = color
    return out


def _analyze(a):
    bg, sources, mirrors = _parse(a)
    routes, reached = _trace(a, sources, mirrors)
    counts = _totals(mirrors, reached)
    out = _rotate_sources(a, bg, sources, counts)
    return {"background": bg, "sources": sources, "mirrors": mirrors,
            "routes": routes, "reached": reached, "counts": counts, "output": out}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the original board, probe it, accumulate counters, rotate sources."""
    return _analyze(_grid(input_grid))["output"].copy()


def _inside(cells, n):
    return all(0 <= r < n and 0 <= c < n for r, c in cells)


def _L(elbow, d):
    r, c = elbow
    return {(r, c), (r-d[0], c), (r, c-d[1])}


def _bar(mid, axis, length):
    radius = length//2
    return {(mid[0]+(t if axis == 1 else 0), mid[1]+(t if axis == 0 else 0))
            for t in range(-radius, radius+1)}


def _put(a, cells, color, bg):
    if not _inside(cells, len(a)):
        return None
    for r, c in cells:
        if a[r, c] != bg:
            return None
        for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if p not in cells and _inside([p], len(a)) and a[p] == color:
                return None
    b = a.copy()
    for p in cells:
        b[p] = color
    return b


def _geometry(a):
    """Trace during construction without requiring final mask disjointness yet."""
    bg, sources, mirrors = _parse(a, require_matching=False)
    routes, reached = _trace(a, sources, mirrors)
    return bg, sources, mirrors, routes, reached


def _add_source(a, rng, color, *, clear=True):
    bg = int(np.bincount(a.ravel(), minlength=10).argmax())
    n = len(a)
    # Uniform cells and directions permit boundary, corner, and interior Ls.
    candidates = [(r, c, d) for r in range(n) for c in range(n) for d in _DIRS
                  if _inside(_L((r, c), d), n)
                  and _inside([(r+2*d[0], c+2*d[1])], n)]
    rng.shuffle(candidates)
    for r, c, d in candidates[:240]:
        b = _put(a, _L((r, c), d), color, bg)
        if b is None:
            continue
        try:
            _, ss, _, routes, _ = _geometry(b)
        except ValueError:
            continue
        route = next(rt for s, rt in zip(ss, routes) if s["elbow"] == (r, c))
        if len(route["path"]) >= 2 and (not clear or not route["events"]):
            return b, (r, c)
    return None


def _append_mirror(a, elbow, rng, color, length, forced_axis=None):
    bg, sources, mirrors, routes, _ = _geometry(a)
    s, route = next((s, rt) for s, rt in zip(sources, routes) if s["elbow"] == elbow)
    d = s["direction"]
    for _, axis, _ in route["events"]:
        d = (-d[0], d[1]) if axis == 0 else (d[0], -d[1])
    anchor = route["events"][-1][2] if route["events"] else elbow
    options = []
    for step in range(2, len(a)):
        contact = anchor[0]+step*d[0], anchor[1]+step*d[1]
        if not _inside([contact], len(a)):
            break
        for axis in ([forced_axis] if forced_axis is not None else [0, 1]):
            mid = contact[0]+(d[0] if axis == 0 else 0), contact[1]+(d[1] if axis == 1 else 0)
            outgoing = (-d[0], d[1]) if axis == 0 else (d[0], -d[1])
            if _inside([(contact[0]+2*outgoing[0], contact[1]+2*outgoing[1])], len(a)):
                options.append((contact, mid, axis))
    rng.shuffle(options)
    old_events = [(mirrors[i]["mid"], axis, p) for i, axis, p in route["events"]]
    for contact, mid, axis in options[:100]:
        b = _put(a, _bar(mid, axis, length), color, bg)
        if b is None:
            continue
        try:
            _, new_sources, new_mirrors, new_routes, reached = _geometry(b)
        except ValueError:
            continue
        # Keep all old contacts; append exactly one to this route. No source
        # may opportunistically activate the new counter a second time.
        if len(reached) != sum(len(rt["events"]) for rt in routes)+1:
            continue
        new_route = next(rt for s, rt in zip(new_sources, new_routes) if s["elbow"] == elbow)
        events = [(new_mirrors[i]["mid"], ax, p) for i, ax, p in new_route["events"]]
        if events == old_events+[(mid, axis, contact)]:
            old_other = sorted((mirrors[i]["mid"], ax, p) for rt in routes for i, ax, p in rt["events"])
            new_other = sorted((new_mirrors[i]["mid"], ax, p) for rt in new_routes for i, ax, p in rt["events"])
            if new_other == sorted(old_other+[(mid, axis, contact)]):
                return b
    return None


def _add_dormant(a, rng, color, length):
    bg, _, mirrors, routes, _ = _geometry(a)
    old = sorted((mirrors[i]["mid"], ax, p) for rt in routes for i, ax, p in rt["events"])
    n = len(a)
    options = [(r, c, axis) for r in range(n) for c in range(n) for axis in (0, 1)]
    rng.shuffle(options)
    for r, c, axis in options[:250]:
        b = _put(a, _bar((r, c), axis, length), color, bg)
        if b is None:
            continue
        try:
            _, _, mm, rr, _ = _geometry(b)
        except ValueError:
            continue
        new = sorted((mm[i]["mid"], ax, p) for rt in rr for i, ax, p in rt["events"])
        if new == old:
            return b
    return None


def _palette(rng, count):
    values = [int(c) for c in rng.permutation(10)]
    return values[0], values[1:count+1]


def _teaching(rng, index):
    """Random geometry with fixed evidence obligations, not fixed positions."""
    bg, colors = _palette(rng, 4)
    a = np.full((int(rng.integers(23, 31)),)*2, bg, dtype=np.int8)
    c0, c1, c2, c3 = colors
    order = [c1, c3, c0, c2, c1] if index == 0 else [c2, c0, c1, c3]
    plans = [[(c0, 1), (c0, 3)], [(c1, 3), (c2, 1)]] if index == 0 else [[(c0, 3), (c0, 3)], [(c1, 5)]]
    first_axis = int(rng.integers(2))
    for j, color in enumerate(order):
        added = _add_source(a, rng, color)
        if added is None:
            return None
        a, elbow = added
        if j < 2:
            for k, (shade, length) in enumerate(plans[j]):
                forced = (first_axis if j == 0 else 1-first_axis) if k == 0 else None
                b = _append_mirror(a, elbow, rng, shade, length, forced)
                if b is None:
                    return None
                a = b
    dormant_color = c3 if index == 0 else c2
    b = _add_dormant(a, rng, dormant_color, int(rng.choice([1, 3, 5])))
    if b is None:
        return None
    try:
        info = _analyze(b)
    except ValueError:
        return None
    if not _evidence_pair(b, info, index):
        return None
    return {"input": b, "output": info["output"]}


def _fallback(rng, index, *, long=False, seen=None):
    """Analytic two-route constellation; used only after bounded sampling."""
    a = np.zeros((28, 28), dtype=np.int8)
    source_specs = [((2, 3), (1, 1), 2 if index == 0 else 3),
                    ((2, 23), (1, -1), 4 if index == 0 else 1),
                    ((24, 26), (1, 1), 1 if index == 0 else 2),
                    ((26, 3), (-1, -1), 3)] if index == 0 else [
                    ((2, 3), (1, 1), 3), ((2, 23), (1, -1), 1),
                    ((20, 25), (1, 1), 2), ((23, 1), (1, 1), 4)]
    if index == 0:
        source_specs.append(((19, 24), (1, 1), 2))
    for elbow, d, color in source_specs:
        for p in _L(elbow, d):
            a[p] = color
    bar_specs = [((6, 8), 1, 1 if index == 0 else 3, 1),
                 ((10, 2), 1, 3, 1),
                 ((7, 19), 0, (int(rng.choice([7, 9])) if long else (3 if index == 0 else 5)), 2),
                 ((16, 24), 0, 3, 4 if index == 0 else 3)]
    if index == 0:
        bar_specs.append(((2, 16), 0, 1, 3))
    for mid, axis, length, color in bar_specs:
        for p in _bar(mid, axis, length):
            a[p] = color
    permutation = rng.permutation(10)
    angle = int(rng.integers(4))
    for shift in range(10):
        for turn in range(4):
            b = ((permutation[np.rot90(a, angle+turn)] + shift) % 10).astype(np.int8)
            if seen is None or _key(b) not in seen:
                info = _analyze(b)
                return {"input": b, "output": info["output"]}
    raise RuntimeError("fallback uniqueness space exhausted")


def _random_pair(rng, is_test):
    color_count = int(rng.integers(1, 6))
    bg, colors = _palette(rng, color_count)
    n = int(rng.integers(18 if is_test else 12, 31))
    a = np.full((n, n), bg, dtype=np.int8)
    count = int(rng.integers(max(2, color_count), 9))
    order = colors + [int(rng.choice(colors)) for _ in range(count-color_count)]
    rng.shuffle(order)
    elbows = []
    for color in order:
        added = _add_source(a, rng, color)
        if added is None:
            return None
        a, elbow = added
        elbows.append(elbow)
    rng.shuffle(elbows)
    for j, elbow in enumerate(elbows):
        desired = int(rng.integers(0, 4))
        if is_test and j == 0:
            desired = max(1, desired)
        for k in range(desired):
            length = int(rng.choice([7, 9])) if is_test and j == k == 0 else int(rng.choice([1, 3, 5, 7, 9] if is_test else [1, 3, 5]))
            b = _append_mirror(a, elbow, rng, int(rng.choice(colors)), length)
            if b is None:
                if is_test and j == k == 0:
                    return None
                break
            a = b
    for _ in range(int(rng.integers(1, 4))):
        b = _add_dormant(a, rng, int(rng.choice(colors)), int(rng.choice([1, 3, 5, 7, 9] if is_test else [1, 3, 5])))
        if b is not None:
            a = b
    try:
        info = _analyze(a)
    except ValueError:
        return None
    if not info["reached"]:
        return None
    if is_test and not any(info["mirrors"][i]["length"] >= 7 for i in info["reached"]):
        return None
    return {"input": a, "output": info["output"]}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not 2 <= num_train <= 10
            or not _integer(num_test) or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    for index in range(2):
        for _ in range(48):
            pair = _teaching(rng, index)
            if pair is not None and _key(pair["input"]) not in seen:
                break
        else:
            pair = _fallback(rng, index, seen=seen)
        puzzle["train"].append(pair)
        seen.add(_key(pair["input"]))
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        while len(puzzle[split]) < count:
            for _ in range(32):
                pair = _random_pair(rng, split == "test")
                if pair is not None and _key(pair["input"]) not in seen:
                    break
            else:
                pair = _fallback(rng, 1, long=split == "test", seen=seen)
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _evidence_pair(a, info, index):
    counts, ss, mm = info["counts"], info["sources"], info["mirrors"]
    totals = sorted(counts.values())
    target = [1, 3, 4] if index == 0 else [5, 6]
    if totals != target:
        return False
    if not any(s["color"] not in counts for s in ss):
        return False
    if not set(range(len(mm))) - info["reached"]:
        return False
    if {axis for rt in info["routes"] for _, axis, _ in rt["events"]} != {0, 1}:
        return False
    if not any(len(rt["events"]) >= 2 for rt in info["routes"]):
        return False
    # Omit A: count all bars, including the inactive bar. Omit reflection:
    # geometrically continue straight; an invalid straight route also refutes it.
    all_counts = _totals(mm, set(range(len(mm))))
    try:
        alternate = _rotate_sources(a, info["background"], ss, all_counts)
        if np.array_equal(alternate, info["output"]):
            return False
    except ValueError:
        pass
    if index == 0:
        grouped = {}
        for s in ss:
            grouped[s["color"]] = grouped.get(s["color"], 0)+1
        if not any(number >= 2 and counts.get(color, 0) % 4 for color, number in grouped.items()):
            return False
    return True


def _coordinate_expected(a, info):
    """Check the mask painter with direct coordinate rotation, no np.rot90."""
    n, bg = len(a), info["background"]
    out = np.full(a.shape, bg, dtype=np.int8)
    for s in info["sources"]:
        for r, c in s["cells"]:
            for _ in range(info["counts"].get(s["color"], 0) % 4):
                r, c = c, n-1-r
            assert out[r, c] == bg, "rotated sources overlap"
            out[r, c] = s["color"]
    return out


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(a) not in seen, "duplicate input"
            seen.add(_key(a))
            info = _analyze(a)
            assert out.shape == a.shape
            assert np.array_equal(out, _coordinate_expected(a, info)), "incorrect constellation"
            assert np.array_equal(out, solve(a.copy()))
            bg = info["background"]
            assert np.count_nonzero(out != bg) == 3*len(info["sources"])
            for color in {s["color"] for s in info["sources"]}:
                assert np.count_nonzero(out == color) == sum(3 for s in info["sources"] if s["color"] == color)
            assert not np.array_equal(a, out), "at least one bar must be removed"
            if split == "train":
                assert all(m["length"] <= 5 for m in info["mirrors"])
                training.append((a, info))
            else:
                assert any(info["mirrors"][i]["length"] in (7, 9) for i in info["reached"]), "missing active long-counter shift"
    assert any(_evidence_pair(a, info, 0) for a, info in training), "missing first teaching evidence"
    assert any(_evidence_pair(a, info, 1) for a, info in training), "missing second teaching evidence"


def _signature(puzzle):
    return tuple((_key(p["input"]), _key(p["output"])) for s in ("train", "test") for p in puzzle[s])


def _normalized(a):
    mapping = {}
    return a.shape, bytes(mapping.setdefault(int(c), len(mapping)) for c in a.flat)


def self_test():
    """Bounded 42-seed diversity, arithmetic, ablation, and rejection checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_signature(p) for p in puzzles}) == 42
    assert _signature(puzzles[0]) == _signature(generate_puzzle(0))
    infos, structures = [], set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                info = _analyze(a)
                infos.append((split, a, info))
                structures.add(_normalized(a))
                # Input-only solving must not mutate the caller's array.
                copied = a.copy()
                solve(copied)
                assert np.array_equal(copied, a)
        # Both parent omissions are actual counterexamples in every puzzle.
        assert any(_omission_changes(a, info, "all") for a, info in
                   [(p["input"], _analyze(p["input"])) for p in puzzle["train"]])
        assert any(_omission_changes(a, info, "none") for a, info in
                   [(p["input"], _analyze(p["input"])) for p in puzzle["train"]])
        assert any(_omission_changes(a, info, "straight") for a, info in
                   [(p["input"], _analyze(p["input"])) for p in puzzle["train"]])
    assert len(structures) >= 140
    assert len({a.shape for _, a, _ in infos}) >= 12
    assert {info["background"] for _, _, info in infos} == set(range(10))
    assert len({len(info["sources"]) for _, _, info in infos}) >= 5
    assert len({len(info["mirrors"]) for _, _, info in infos}) >= 5
    assert {s["direction"] for _, _, info in infos for s in info["sources"]} == set(_DIRS)
    assert {m["length"] for _, _, info in infos for m in info["mirrors"]} == {1, 3, 5, 7, 9}
    assert {k % 4 for _, _, info in infos for k in info["counts"].values()} == set(range(4))
    assert any(len(info["sources"]) >= 7 for _, _, info in infos)
    assert any(max(map(lambda rt: len(rt["events"]), info["routes"])) >= 3 for _, _, info in infos)
    assert any(any(p[0] in (0, len(a)-1) or p[1] in (0, len(a)-1)
                   for s in info["sources"] for p in s["cells"]) for _, a, info in infos)
    # Every legal explicit count is also exercised by verify.py.
    for train, test in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(420+train+test, train, test))
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for index in (0, 1):
        pair = _fallback(np.random.default_rng(index), index)
        assert _evidence_pair(pair["input"], _analyze(pair["input"]), index)
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ("train", "test")}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    _direct_regression()


def _omission_changes(a, info, mode):
    try:
        if mode == "all":
            counts = _totals(info["mirrors"], set(range(len(info["mirrors"]))))
        elif mode == "none":
            counts = {}
        else:
            _, reached = _trace(a, info["sources"], info["mirrors"], reflect=False)
            counts = _totals(info["mirrors"], reached)
        alternative = _rotate_sources(a, info["background"], info["sources"], counts)
    except ValueError:
        return True
    return not np.array_equal(alternative, info["output"])


def _direct_regression():
    # Hand-enumerated one-point activation, clockwise movement, and removal.
    a = np.zeros((12, 12), dtype=np.int8)
    a[0, 2] = a[1, 1] = a[1, 2] = 1
    a[9, 9] = a[9, 10] = a[8, 9] = 2
    a[5, 7] = 2  # Contact (5,6), reflects the column component.
    a[0, 9:12] = 1  # Dormant bar: it must not rotate the color-1 source.
    expected = np.zeros_like(a)
    expected[0, 2] = expected[1, 1] = expected[1, 2] = 1
    expected[9, 2] = expected[10, 2] = expected[9, 3] = 2
    assert np.array_equal(solve(a), expected)


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds)")
