"""Hollow channels fuse by keeping the exterior of their summed heading.

Only Python's standard library and NumPy are used. No parent imports, examples,
seed information, or construction metadata are used by solve().
"""

from __future__ import annotations

import json
from math import gcd

import numpy as np


DIRECTIONS = ((-1, -1), (-1, 1), (1, 1), (1, -1))
POLICIES = ("whole_cap", "independent", "keep_seams", "sign", "equal_clock", "one_each", "reverse_phase")


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid grid colors")
    return a.astype(np.int8, copy=False)


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _outline(center, radius):
    r, c = center
    return [(r + dr, c + dc) for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1) if abs(dr) + abs(dc) == radius]


def _cap(center, radius, direction):
    r, c = center
    dr, dc = direction
    return [(r + dr * (k + 1), c + dc * (radius - k + 1))
            for k in range(radius + 1)]


def _base(shape, background):
    return np.asarray(background, dtype=np.int8)[np.indices(shape).sum(axis=0) % 2]


def _parse(grid):
    """Recover local roles and complete outlines from visible input alone."""
    h, w = grid.shape
    parity = np.indices(grid.shape).sum(axis=0) % 2
    background = []
    for p in (0, 1):
        values = grid[parity == p]
        _require(values.size > 0, "checkerboard parity is absent")
        counts = np.bincount(values, minlength=10)
        color = int(counts.argmax())
        _require(counts[color] * 2 > values.size, "background parity has no majority")
        background.append(color)
    _require(background[0] != background[1], "background is not a two-color checkerboard")
    base = _base(grid.shape, background)
    foreground = set(map(tuple, np.argwhere(grid != base)))
    emitters = []
    for radius in (2, 3, 4):
        hh, ww = h - 2 * radius, w - 2 * radius
        if min(hh, ww) <= 0:
            continue
        top = grid[:hh, radius:radius + ww]
        mask = (top != background[0]) & (top != background[1])
        mask &= grid[2 * radius:2 * radius + hh, radius:radius + ww] == top
        mask &= grid[radius:radius + hh, :ww] == top
        mask &= grid[radius:radius + hh, 2 * radius:2 * radius + ww] == top
        for rr, cc in np.argwhere(mask):
            center = (int(rr) + radius, int(cc) + radius)
            color = int(top[rr, cc])
            if all(int(grid[p]) == color for p in _outline(center, radius)):
                emitters.append({"center": center, "radius": radius, "structure": color, "caps": {}})
    _require(1 <= len(emitters) <= 4, "need one to four unambiguous diamonds")
    emitters.sort(key=lambda e: e["center"])
    covered = set()
    for e in emitters:
        cells = set(_outline(e["center"], e["radius"]))
        _require(not covered.intersection(cells), "outlines overlap")
        covered.update(cells)
    for e in emitters:
        for v in DIRECTIONS:
            visible = [p for p in _cap(e["center"], e["radius"], v) if _inside(p, grid.shape)]
            if len(visible) < 2:
                continue
            colors = {int(grid[p]) for p in visible}
            if len(colors) == 1:
                color = next(iter(colors))
                if color not in background and color != e["structure"]:
                    _require(not covered.intersection(visible), "cap ownership is ambiguous")
                    e["caps"][v] = color
                    covered.update(visible)
        _require(e["caps"], "diamond has no recognizable cap")
    _require(covered == foreground, "unaccounted input foreground")
    return {"background": tuple(background), "emitters": emitters, "base": base}


def _walk(origin, heading, shape, reverse_phase=False):
    dr, dc = heading
    scale = max(abs(dr), abs(dc))
    result = []
    er = scale - 1 if reverse_phase and dr else 0
    ec = scale - 1 if reverse_phase and dc else 0
    # At least one coordinate advances every step, so at most max(shape) steps.
    for k in range(max(shape) + 1):
        point = (origin[0] + (1 if dr >= 0 else -1) * ((k * abs(dr) + er) // scale),
                 origin[1] + (1 if dc >= 0 else -1) * ((k * abs(dc) + ec) // scale))
        if not _inside(point, shape):
            break
        result.append(point)
    return result


def _sum_heading(a, b, sign_only=False):
    r, c = a[0] + b[0], a[1] + b[1]
    _require((r, c) != (0, 0), "head-on cancellation")
    if sign_only:
        return int(np.sign(r)), int(np.sign(c))
    d = gcd(abs(r), abs(c))
    return r // d, c // d


def _contact(paths_a, paths_b, equal_clock=False):
    """Contact of future pixels/diagonal segments, with separate travel clocks."""
    hits = []
    for a in paths_a:
        cells = {p: k for k, p in enumerate(a)}
        mids = {(p[0] + q[0], p[1] + q[1]): (k, p, q)
                for k, (p, q) in enumerate(zip(a, a[1:]))
                if p[0] != q[0] and p[1] != q[1]}
        for b in paths_b:
            for j, p in enumerate(b):
                if p in cells:
                    i = cells[p]
                    if not equal_clock or i == j:
                        hits.append((i - 1, j - 1, "cell"))
            for j, (p, q) in enumerate(zip(b, b[1:])):
                if p[0] == q[0] or p[1] == q[1]:
                    continue
                key = (p[0] + q[0], p[1] + q[1])
                if key in mids:
                    i, x, y = mids[key]
                    if {p, q} != {x, y} and (not equal_clock or i == j):
                        hits.append((i, j, "between"))
    if not hits:
        return None
    # The envelope's first contact must be first for both participants.
    ca, cb = min(x[0] for x in hits), min(x[1] for x in hits)
    first = [x for x in hits if x[:2] == (ca, cb)]
    _require(first, "ambiguous contact between channel envelopes")
    kinds = {x[2] for x in first}
    _require(len(kinds) == 1, "cell/midpoint contact tie")
    return ca, cb, first[0][2]


def _initial_channels(analysis, shape, policy):
    channels, serial = [], 0
    for e in analysis["emitters"]:
        for v in DIRECTIONS:
            if v not in e["caps"]:
                continue
            cap = _cap(e["center"], e["radius"], v)
            points = cap if policy == "whole_cap" else [cap[0], cap[-1]]
            members = []
            for p in points:
                if _inside(p, shape):
                    members.append((serial, p, e["caps"][v]))
                serial += 1
            if policy == "independent":
                channels.extend({"v": v, "members": [m], "depth": 0, "complete": True}
                                for m in members)
            elif members:
                channels.append({"v": v, "members": members, "depth": 0,
                                 "complete": len(members) == len(points)})
    return channels


def _trace(grid, analysis=None, policy="normal"):
    analysis = _parse(grid) if analysis is None else analysis
    channels = _initial_channels(analysis, grid.shape, policy)
    output = grid.copy()
    occupied, segments = {}, set()
    events = []
    initial_count = len(channels)

    def paint(member, path):
        sid, _, color = member
        for p in path[1:]:
            _require(int(grid[p]) == int(analysis["base"][p]), "rail overwrites an input object")
            _require(p not in occupied, "rail overlap or self-intersection")
            occupied[p] = sid
            output[p] = color
        for p, q in zip(path, path[1:]):
            if p[0] != q[0] and p[1] != q[1]:
                key = (p[0] + q[0], p[1] + q[1], (q[0] - p[0]) * (q[1] - p[1]))
                _require((key[0], key[1], -key[2]) not in segments, "rail crosses an old segment")
                segments.add(key)

    for _ in range(initial_count + 1):
        forecasts = [[_walk(m[1], ch["v"], grid.shape, policy == "reverse_phase")
                      for m in ch["members"]] for ch in channels]
        candidates = []
        for i in range(len(channels)):
            for j in range(i + 1, len(channels)):
                hit = _contact(forecasts[i], forecasts[j], policy == "equal_clock")
                if hit is not None:
                    _require(channels[i]["v"] != channels[j]["v"], "parallel channel overlap")
                    _require(channels[i]["complete"] and channels[j]["complete"],
                             "partly clipped channel cannot fuse")
                    candidates.append((i, j, hit))
        if not candidates:
            for ch, paths in zip(channels, forecasts):
                for member, path in zip(ch["members"], paths):
                    paint(member, path)
            break
        nearest = {}
        for i, j, hit in candidates:
            nearest[i] = min(nearest.get(i, 1000), hit[0])
            nearest[j] = min(nearest.get(j, 1000), hit[1])
        ready = [x for x in candidates if x[2][0] == nearest[x[0]] and x[2][1] == nearest[x[1]]]
        _require(ready, "cyclic encounter order")
        i, j, hit = min(ready, key=lambda x: (max(x[2][:2]), sum(x[2][:2]), x[0], x[1]))
        for a, b, other in candidates:
            if (a, b) != (i, j):
                _require(not ((a in (i, j) and other[0] == nearest[a]) or
                              (b in (i, j) and other[1] == nearest[b])), "multiway encounter tie")
        _require(min(hit[:2]) >= 0, "contact at a channel origin")
        a, b = channels[i], channels[j]
        heading = _sum_heading(a["v"], b["v"], policy == "sign")
        last_members = []
        for index, cut in ((i, hit[0]), (j, hit[1])):
            for member, path in zip(channels[index]["members"], forecasts[index]):
                _require(cut < len(path), "a rail exits before its partner fuses")
                paint(member, path[:cut + 1])
                last_members.append((member[0], path[cut], member[2]))
        projections = [heading[0] * m[1][1] - heading[1] * m[1][0] for m in last_members]
        low, high = min(projections), max(projections)
        _require(low != high and projections.count(low) == projections.count(high) == 1,
                 "exterior envelope has tied extremes")
        if policy == "keep_seams":
            survivors = last_members
        elif policy == "one_each":
            # Counterfactual: force one exterior from each incoming channel.
            # Prefer the cross-channel pair with maximum transverse separation.
            boundary = len(a["members"])
            options = [(abs(projections[x] - projections[y]), x, y)
                       for x in range(boundary) for y in range(boundary, len(last_members))]
            _, x, y = max(options)
            survivors = [last_members[x], last_members[y]]
        else:
            survivors = [last_members[projections.index(low)], last_members[projections.index(high)]]
        depth = max(a["depth"], b["depth"]) + 1
        incoming_kind = ("diagonal" if all(sum(map(abs, ch["v"])) == 2 and
                                         max(map(abs, ch["v"])) == 1 for ch in (a, b)) else "later")
        events.append({"incoming": (a["v"], b["v"]), "outgoing": heading,
                       "cuts": hit[:2], "intersection": hit[2], "depth": depth,
                       "kind": incoming_kind, "unequal": hit[0] != hit[1],
                       "last": tuple(last_members), "kept": tuple(m[0] for m in survivors),
                       "discarded": tuple(m for m in last_members if m not in survivors),
                       "positions": tuple(m[1] for m in survivors)})
        channels = [ch for k, ch in enumerate(channels) if k not in (i, j)]
        channels.append({"v": heading, "members": survivors, "depth": depth, "complete": True})
    else:
        raise ValueError("fusion failed to terminate")
    return output, {"analysis": analysis, "events": events, "terminal": channels,
                    "changed": int(np.count_nonzero(output != grid))}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only fixed-family solver. The caller's array is never mutated."""
    return _trace(_grid(input_grid))[0]


def _straight(grid, whole=False):
    """Counterfactual: overwrite only background with independent straight rays."""
    a = _parse(grid)
    out = grid.copy()
    for ch in _initial_channels(a, grid.shape, "whole_cap" if whole else "normal"):
        for _, point, color in ch["members"]:
            for p in _walk(point, ch["v"], grid.shape)[1:]:
                if grid[p] == a["base"][p]:
                    out[p] = color
    return out


def _alternative(grid, policy):
    try:
        return _trace(grid, policy=policy)[0]
    except ValueError:
        return None


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _spec_key(specs):
    return sorted((e["center"], e["radius"], e["structure"], tuple(sorted(e["caps"].items()))) for e in specs)


def _build(shape, specs, background=(0, 1)):
    grid = _base(shape, background)
    occupied = set()
    for e in specs:
        cells = _outline(e["center"], e["radius"])
        _require(all(_inside(p, shape) for p in cells), "clipped diamond outline")
        assignments = [(p, e["structure"]) for p in cells]
        for v, color in e["caps"].items():
            cap = [p for p in _cap(e["center"], e["radius"], v) if _inside(p, shape)]
            _require(len(cap) >= 2, "cap has fewer than two visible cells")
            _require(color != e["structure"] and color not in background, "invalid cap role")
            assignments.extend((p, color) for p in cap)
        for p, color in assignments:
            _require(p not in occupied and color not in background, "objects overlap or blend with background")
            occupied.add(p)
            grid[p] = color
    analysis = _parse(grid)
    _require(_spec_key(analysis["emitters"]) == _spec_key(specs), "construction cannot be recovered")
    out, info = _trace(grid, analysis)
    _require(info["changed"] > 0, "unchanged scene")
    return grid, out, info


def _endpoint_witness(grid, output, analysis):
    radii = set()
    for e in analysis["emitters"]:
        for v in e["caps"]:
            cap = _cap(e["center"], e["radius"], v)
            ends = [cap[0], cap[-1]]
            if not all(_inside(p, grid.shape) for p in ends):
                continue
            end_probes = [(p[0] + k * v[0], p[1] + k * v[1]) for p in ends for k in (1, 2)]
            interior = [(p[0] + k * v[0], p[1] + k * v[1]) for p in cap[1:-1] for k in (1, 2)]
            if (all(_inside(p, grid.shape) and grid[p] == analysis["base"][p] and
                    output[p] == e["caps"][v] for p in end_probes) and
                    all(_inside(p, grid.shape) and grid[p] == output[p] == analysis["base"][p]
                        for p in interior)):
                radii.add(e["radius"])
    return radii


def _event_tags(info):
    tags = set()
    for e in info["events"]:
        tags.add(e["kind"])
        if e["depth"] > 1:
            tags.add("chain")
        if max(map(abs, e["outgoing"])) > 1:
            tags.add("staircase")
    return tags


def _single(rng, max_radius, witness=False):
    for _ in range(80):
        radius = max_radius if witness else int(rng.integers(2, max_radius + 1))
        lo = 2 * radius + (9 if witness else 3)
        shape = tuple(map(int, rng.integers(lo, 31, 2)))
        center = tuple(int(rng.integers(radius + (3 if witness else 0),
                                        axis - radius - (3 if witness else 0))) for axis in shape)
        mask = int(rng.choice((5, 10, 7, 11, 13, 14, 15))) if witness else int(rng.integers(1, 16))
        directions = [v for i, v in enumerate(DIRECTIONS) if mask & (1 << i)]
        colors = list(map(int, rng.choice(np.arange(3, 10), int(rng.integers(1, len(directions) + 1)), replace=False)))
        caps = {v: colors[i % len(colors)] for i, v in enumerate(directions)}
        spec = {"center": center, "radius": radius, "structure": 2, "caps": caps}
        try:
            grid, out, info = _build(shape, [spec])
            if not witness or max_radius in _endpoint_witness(grid, out, info["analysis"]):
                return shape, [spec]
        except ValueError:
            continue
    raise RuntimeError("single-channel construction exhausted")


def _primary(rng, max_radius, half=None):
    for _ in range(160):
        ra, rb = map(int, rng.integers(2, max_radius + 1, 2))
        shape = tuple(map(int, rng.integers(max(21, 4 * max(ra, rb) + 9), 31, 2)))
        row = int(rng.integers(max(ra, rb) + 3, shape[0] - max(ra, rb) - 4))
        col = int(rng.integers(shape[1] // 2 - 2, shape[1] // 2 + 2))
        da, db = map(int, rng.integers(2, 8, 2))
        if da == db:
            continue
        between = bool(rng.integers(2)) if half is None else half
        if between:
            centers = ((row + da + 2, col - da - ra - 1),
                       (row + db + 2, col + db + rb + 2))
        else:
            centers = ((row + da + 1, col - da - ra - 1),
                       (row + db + 1, col + db + rb + 1))
        specs = [{"center": centers[0], "radius": ra, "structure": 2, "caps": {(-1, 1): 3}},
                 {"center": centers[1], "radius": rb, "structure": 2, "caps": {(-1, -1): 4}}]
        try:
            _, _, info = _build(shape, specs)
            _require(len(info["events"]) == 1 and info["events"][0]["unequal"], "missing unequal fusion")
            _require(min(len(_walk(m[1], ch["v"], shape)) for ch in info["terminal"]
                         for m in ch["members"]) >= 4, "fusion continuation too short")
            return shape, specs
        except ValueError:
            continue
    # A constructive fallback preserves the requested mechanism and palette freedom.
    specs = [{"center": (18, 6), "radius": 2, "structure": 2, "caps": {(-1, 1): 3}},
             {"center": (16, 21), "radius": 2, "structure": 2, "caps": {(-1, -1): 4}}]
    _build((30, 30), specs)
    return (30, 30), specs


def _attach(rng, shape, specs, max_radius):
    _, _, previous = _build(shape, specs)
    targets = [(ch, m, path) for ch in previous["terminal"] for m in ch["members"]
               if ch["depth"] > 0 and sum(map(abs, ch["v"])) == 1
               for path in [_walk(m[1], ch["v"], shape)] if len(path) >= 7]
    if not targets:
        return None
    for _ in range(220):
        ch, _, path = targets[int(rng.integers(len(targets)))]
        q = path[int(rng.integers(2, len(path) - 3))]
        # Canonical construction travels north; returned grids may use any transform.
        v = (-1, int(rng.choice((-1, 1))))
        distance = int(rng.integers(2, 10))
        radius = int(rng.integers(2, max_radius + 1))
        endpoint = int(rng.integers(2))
        p = (q[0] - distance * v[0], q[1] - distance * v[1])
        if endpoint == 0:
            center = (p[0] - v[0], p[1] - v[1] * (radius + 1))
        else:
            center = (p[0] - v[0] * (radius + 1), p[1] - v[1])
        new = {"center": center, "radius": radius, "structure": 2, "caps": {v: 5}}
        try:
            _, _, info = _build(shape, specs + [new])
            _require(len(info["events"]) == 2 and max(e["depth"] for e in info["events"]) == 2,
                     "attachment is not a two-stage fusion")
            _require("staircase" in _event_tags(info), "attachment did not produce a staircase")
            _require(min(len(_walk(m[1], c["v"], shape)) for c in info["terminal"]
                         for m in c["members"]) >= 4, "staircase too short")
            return specs + [new]
        except ValueError:
            continue
    return None


def _chain(rng, max_radius):
    for _ in range(24):
        shape, specs = _primary(rng, max_radius)
        attached = _attach(rng, shape, specs, max_radius)
        if attached is not None:
            return shape, attached
    raise RuntimeError("chain construction exhausted")


def _independent(rng, max_radius):
    """Four mouths, two outward-facing fusions; no visible compartments."""
    for _ in range(200):
        shape = tuple(map(int, rng.integers(26, 31, 2)))
        radii = list(map(int, rng.integers(2, min(max_radius, 3) + 1, 4)))
        left = int(rng.integers(4, 7))
        right = int(rng.integers(shape[1] - 8, shape[1] - 4))
        top = int(rng.integers(10, 13))
        bottom = int(rng.integers(shape[0] - 12, shape[0] - 9))
        centers = ((top, left), (top + int(rng.choice((-1, 0, 1))), right),
                   (bottom, left + int(rng.choice((-1, 0, 1)))),
                   (bottom + int(rng.choice((-1, 0, 1))), right))
        vectors = ((-1, 1), (-1, -1), (1, 1), (1, -1))
        specs = [{"center": c, "radius": r, "structure": 2, "caps": {v: 3 + i}}
                 for i, (c, r, v) in enumerate(zip(centers, radii, vectors))]
        try:
            _, _, info = _build(shape, specs)
            _require(len(info["events"]) == 2 and all(e["depth"] == 1 for e in info["events"]),
                     "independent fusions interfere")
            _require(min(len(_walk(m[1], ch["v"], shape)) for ch in info["terminal"]
                         for m in ch["members"]) >= 3, "independent continuations too short")
            return shape, specs
        except ValueError:
            continue
    # Algebraically separated north/south channels on the same checkerboard.
    specs = [{"center": c, "radius": 2, "structure": 2, "caps": {v: 3 + i}}
             for i, (c, v) in enumerate(zip(((12, 5), (11, 21), (19, 5), (20, 21)),
                                            ((-1, 1), (-1, -1), (1, 1), (1, -1))))]
    _build((30, 30), specs)
    return (30, 30), specs


def _extra(rng, shape, specs, max_radius, extra_object=False):
    """Add caps or a mouth only when the existing network remains readable."""
    _, _, before = _build(shape, specs)
    for _ in range(35):
        candidate = [{**e, "caps": dict(e["caps"])} for e in specs]
        if extra_object and len(specs) < 4:
            radius = int(rng.integers(2, max_radius + 1))
            center = tuple(int(rng.integers(radius, x - radius)) for x in shape)
            vs = [v for i, v in enumerate(DIRECTIONS) if int(rng.integers(1, 16)) & (1 << i)]
            if not vs:
                continue
            candidate.append({"center": center, "radius": radius, "structure": 2,
                              "caps": {v: 6 for v in vs}})
        else:
            e = candidate[int(rng.integers(len(candidate)))]
            choices = [v for v in DIRECTIONS if v not in e["caps"]]
            if not choices:
                continue
            v = choices[int(rng.integers(len(choices)))]
            e["caps"][v] = int(rng.integers(3, 10))
        try:
            _, _, info = _build(shape, candidate)
            _require(_event_tags(info) <= _event_tags(before), "extra introduced a new interaction type")
            _require(len(info["events"]) == len(before["events"]), "extra changed fusion topology")
            return candidate
        except ValueError:
            continue
    return specs


def _finish(rng, shape, specs, palette, stable):
    permutation = palette if stable else rng.permutation(10)
    background = tuple(map(int, permutation[:2]))
    bank = list(map(int, permutation[2:]))
    specs = [{**e, "caps": {v: bank[color - 3] for v, color in e["caps"].items()}} for e in specs]
    shared = bool(rng.random() < .4)
    for e in specs:
        choices = [c for c in bank if c not in e["caps"].values()]
        e["structure"] = bank[-1] if shared else int(rng.choice(choices))
    grid, _, _ = _build(shape, specs, background)
    rotation, reflection = int(rng.integers(4)), bool(rng.integers(2))
    grid = np.rot90(grid, rotation)
    if reflection:
        grid = np.fliplr(grid)
    grid = np.ascontiguousarray(grid)
    out, info = _trace(grid)
    return {"input": grid, "output": out}, info


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} is outside {low}..{high}")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 2, 10, "num_train")
    ne = _count(num_test, 1, 4, "num_test")
    if nt <= ne:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    max_radius = int(rng.integers(2, 5))
    palette = rng.permutation(10)
    stable = bool(rng.random() < .25)
    modes = ["endpoint", "fusion"] + (["chain"] if nt >= 3 else [])
    while len(modes) < nt:
        modes.append(str(rng.choice(("single", "fusion", "chain", "independent"))))
    rng.shuffle(modes)
    puzzle, seen, taught = {"train": [], "test": []}, set(), set()
    fitted = []
    for split, count in (("train", nt), ("test", ne)):
        if split == "test":
            fitted = [policy for policy in POLICIES if all(
                (pred := _alternative(p["input"], policy)) is not None and np.array_equal(pred, p["output"])
                for p in puzzle["train"])]
        for index in range(count):
            mode = modes[index] if split == "train" else str(rng.choice(
                ("fusion", "chain", "chain", "independent") if "chain" in taught else
                ("fusion", "fusion", "independent")))
            for attempt in range(30):
                if mode in ("single", "endpoint"):
                    shape, specs = _single(rng, max_radius, mode == "endpoint")
                elif mode == "chain":
                    shape, specs = _chain(rng, max_radius)
                elif mode == "independent":
                    shape, specs = _independent(rng, max_radius)
                else:
                    shape, specs = _primary(rng, max_radius)
                if mode != "endpoint":
                    for _ in range(int(rng.integers(0, 3))):
                        specs = _extra(rng, shape, specs, max_radius, rng.random() < .6)
                try:
                    pair, info = _finish(rng, shape, specs, palette, stable)
                    _require(_key(pair["input"]) not in seen, "duplicate input")
                    if mode == "endpoint":
                        _require(max_radius in _endpoint_witness(pair["input"], pair["output"], info["analysis"]),
                                 "returned endpoint witness is obscured")
                    if split == "test":
                        _require(_event_tags(info) <= taught, "untaught returned interaction")
                        _require(all((pred := _alternative(pair["input"], p)) is not None and
                                     np.array_equal(pred, pair["output"]) for p in fitted),
                                 "training leaves competing test predictions unresolved")
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    if split == "train":
                        taught.update(_event_tags(info))
                    break
                except ValueError:
                    continue
            else:
                raise RuntimeError(f"bounded final-grid construction exhausted: {split}/{mode}")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "invalid pair counts")
    seen, witnesses, taught, needed = set(), set(), set(), set()
    test_radii = set()
    fusion_witness = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            _require(grid.shape == output.shape, "dimensions changed")
            _require(_key(grid) not in seen, "duplicate input")
            seen.add(_key(grid))
            expected, info = _trace(grid)
            _require(np.array_equal(expected, output), "incorrect channel output")
            _require(info["changed"] > 0, "identity example")
            foreground = grid != info["analysis"]["base"]
            _require(np.array_equal(grid[foreground], output[foreground]), "input object changed")
            if split == "train":
                witnesses.update(_endpoint_witness(grid, output, info["analysis"]))
                taught.update(_event_tags(info))
                if info["events"] and any(e["unequal"] for e in info["events"]):
                    ignored_a = _alternative(grid, "whole_cap")
                    ignored_coupling = _alternative(grid, "independent")
                    ignored_envelope = _alternative(grid, "keep_seams")
                    fusion_witness |= (not np.array_equal(_straight(grid), output) and
                        (ignored_a is None or not np.array_equal(ignored_a, output)) and
                        (ignored_coupling is None or not np.array_equal(ignored_coupling, output)) and
                        (ignored_envelope is None or not np.array_equal(ignored_envelope, output)))
            else:
                needed.update(_event_tags(info))
                test_radii.update(e["radius"] for e in info["analysis"]["emitters"])
    _require(witnesses and max(witnesses) >= max(test_radii), "missing scalable endpoint witness")
    _require(fusion_witness, "training must distinguish both parents and envelope coupling")
    _require(needed <= taught, "test has an untaught interaction mechanism")
    for policy in POLICIES:
        if all((pred := _alternative(_grid(p["input"]), policy)) is not None and
               np.array_equal(pred, _grid(p["output"])) for p in puzzle["train"]):
            _require(all((pred := _alternative(_grid(p["input"]), policy)) is not None and
                         np.array_equal(pred, _grid(p["output"])) for p in puzzle["test"]),
                     "a training-consistent competing rule disagrees on test")


def _measure(pair):
    grid, out = _grid(pair["input"]), _grid(pair["output"])
    _, info = _trace(grid)
    es = info["analysis"]["emitters"]
    return {"shape": list(grid.shape), "mouths": len(es),
            "radii": sorted({e["radius"] for e in es}),
            "masks": [sum(1 << DIRECTIONS.index(v) for v in e["caps"]) for e in es],
            "background": list(info["analysis"]["background"]),
            "structures": sorted({e["structure"] for e in es}),
            "cap_color_counts": [len(set(e["caps"].values())) for e in es],
            "clipped_caps": sum(any(not _inside(p, grid.shape) for p in _cap(e["center"], e["radius"], v))
                                for e in es for v in e["caps"]),
            "fusions": len(info["events"]),
            "depth": max((e["depth"] for e in info["events"]), default=0),
            "contacts": [e["intersection"] for e in info["events"]],
            "headings": [list(e["outgoing"]) for e in info["events"]],
            "changed": int(np.count_nonzero(grid != out)),
            "layout": (grid.shape, tuple((e["center"], e["radius"], tuple(e["caps"])) for e in es))}


def _fingerprint(puzzle):
    return tuple((_key(_grid(p["input"])), _key(_grid(p["output"])))
                 for split in ("train", "test") for p in puzzle[split])


def self_test():
    """Bounded fixtures, counts, metamorphic checks, and 42-seed diversity."""
    # Explicit non-generated geometry: the inner rails terminate, outer rails turn.
    fixture = [{"center": (18, 6), "radius": 2, "structure": 2, "caps": {(-1, 1): 3}},
               {"center": (16, 21), "radius": 2, "structure": 2, "caps": {(-1, -1): 4}}]
    grid, out, info = _build((30, 30), fixture)
    assert len(info["events"]) == 1 and info["events"][0]["unequal"]
    assert info["events"][0]["outgoing"] == (-1, 0)
    assert len(info["events"][0]["kept"]) == len(info["events"][0]["discarded"]) == 2
    assert info["events"][0]["intersection"] == "between"
    assert set(info["events"][0]["positions"]) == {(10, 12), (10, 17)}
    assert out[0, 12] == 3 and out[0, 17] == 4
    assert out[12, 14] == 3 and out[12, 15] == 4
    assert out[11, 14] == grid[11, 14] and out[11, 15] == grid[11, 15]
    for color in (3, 4):
        assert np.count_nonzero(out[0] == color) == 1
    assert np.array_equal(solve(grid), out)
    assert _walk((7, 3), (-2, 1), (10, 10))[:5] == [(7, 3), (6, 3), (5, 4), (4, 4), (3, 5)]
    assert _walk((7, 3), (-2, 1), (10, 10), True)[:3] == [(7, 3), (6, 4), (5, 4)]
    assert _sum_heading((-1, -1), (-1, 1)) == (-1, 0)
    assert _sum_heading((-1, 0), (-1, 1)) == (-2, 1)
    for k in range(4):
        assert np.array_equal(solve(np.rot90(grid, k)), np.rot90(out, k))
    assert np.array_equal(solve(np.fliplr(grid)), np.fliplr(out))
    # A boundary-touching cap has zero endpoint continuation; its middle
    # must not extend into (0,0), while the opposite channel still grows.
    edge_grid, edge_out, _ = _build((13, 15), [{"center": (3, 3), "radius": 2,
        "structure": 2, "caps": {(-1, -1): 4, (1, 1): 3}}])
    assert edge_grid[0, 0] == edge_out[0, 0] == 0
    assert np.count_nonzero(edge_grid != edge_out) > 0
    assert np.array_equal(solve(edge_grid), edge_out)
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_fingerprint(p) for p in puzzles}) == 42
    assert _fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0))
    rows = [_measure(pair) for p in puzzles for split in ("train", "test") for pair in p[split]]
    assert len({tuple(f["shape"]) for f in rows}) >= 30
    assert len({f["layout"] for f in rows}) >= 120
    assert set(r for f in rows for r in f["radii"]) == {2, 3, 4}
    assert {1, 2, 3} <= {f["mouths"] for f in rows}
    assert {0, 1, 2} <= {f["depth"] for f in rows}
    assert {"cell", "between"} == {c for f in rows for c in f["contacts"]}
    assert len({tuple(v) for f in rows for v in f["headings"]}) >= 8
    assert len({tuple(f["background"]) for f in rows}) >= 20
    assert any(len(f["structures"]) > 1 for f in rows)
    assert any(f["clipped_caps"] for f in rows)
    assert any(max(f["cap_color_counts"]) >= 3 for f in rows)
    assert any(f["shape"][0] != f["shape"][1] for f in rows)
    assert any(set(event["kept"]) <= {m[0] for m in event["last"][:2]} or
               set(event["kept"]) <= {m[0] for m in event["last"][2:]}
               for p in puzzles for split in ("train", "test") for pair in p[split]
               for event in _trace(pair["input"])[1]["events"])
    # The two-example curriculum must never ask for an undemonstrated staircase.
    minimal = generate_puzzle(42, 2, 1)
    assert "staircase" not in _event_tags(_trace(minimal["test"][0]["input"])[1])
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(100 + 10 * nt + ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (2.0, 1), (None, 1), (4, False)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: _grid(v).copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    p = bad["test"][0]
    cell = tuple(np.argwhere(p["input"] != p["output"])[0])
    p["output"][cell] = p["input"][cell]
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    # Compact, descriptive report; no claim of independent semantic verification.
    return {"seeds": 42, "pairs": len(rows), "shapes": len({tuple(f["shape"]) for f in rows}),
            "layouts": len({f["layout"] for f in rows}), "mouth_counts": sorted({f["mouths"] for f in rows}),
            "radii": [2, 3, 4], "headings": len({tuple(v) for f in rows for v in f["headings"]}),
            "clipped_pairs": sum(bool(f["clipped_caps"]) for f in rows),
            "independent_solver": False, "blind_prediction": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
