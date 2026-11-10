"""Hidden-quarter waypoint walk.

Every input shows solid congruent squares on a uniform background: one start
square and many waypoint squares.  Together (start included) they are three of
the four quarter-turn copies of one constellation about a unique pivot; the
fourth quarter is invisible.  The solver restores the hidden squares in the
waypoint color, then walks greedily from the start through the *completed*
constellation: repeatedly go to the nearest unvisited, exactly aligned square
whose straight corridor is clear, painting a corridor as wide as the squares in
the start color, and stop when no such square remains.  Hidden squares are
real waypoints and real obstacles, so the walk differs from the walk that only
sees the visible squares.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Geometry shared by generation, solving and validation
# ---------------------------------------------------------------------------

def _rot(a: int, b: int, turns: int) -> tuple[int, int]:
    """Clockwise quarter turns of a doubled (row, column) offset."""
    for _ in range(turns % 4):
        a, b = b, -a
    return a, b


def _sector(a: int, b: int):
    """0 east, 1 south, 2 west, 3 north; None on a diagonal or the pivot."""
    if b > abs(a):
        return 0
    if a > abs(b):
        return 1
    if -b > abs(a):
        return 2
    if -a > abs(b):
        return 3
    return None


def _center2(pos, s):
    return (2 * pos[0] + s - 1, 2 * pos[1] + s - 1)


def _separate(p, q, s) -> bool:
    """At least one full background row or column lies between two squares."""
    return (p[0] + s < q[0] or q[0] + s < p[0]
            or p[1] + s < q[1] or q[1] + s < p[1])


def _relation(p, q, s):
    """Exact row-span or column-span alignment with a positive gap."""
    if p[0] == q[0] and abs(p[1] - q[1]) > s:
        return ("h", abs(p[1] - q[1]) - s, 1 if q[1] > p[1] else -1)
    if p[1] == q[1] and abs(p[0] - q[0]) > s:
        return ("v", abs(p[0] - q[0]) - s, 1 if q[0] > p[0] else -1)
    return None


def _segment(p, q, s, axis):
    if axis == "h":
        return tuple((r, c) for r in range(p[0], p[0] + s)
                     for c in range(min(p[1], q[1]) + s, max(p[1], q[1])))
    return tuple((r, c) for r in range(min(p[0], q[0]) + s, max(p[0], q[0]))
                 for c in range(p[1], p[1] + s))


def _cells(positions, s):
    return {(r, c) for pr, pc in positions
            for r in range(pr, pr + s) for c in range(pc, pc + s)}


def _trace(positions, s, blocking=True):
    """Greedy walk from positions[0]; nearest clear aligned unvisited square."""
    occupied = _cells(positions, s)
    remaining = set(range(1, len(positions)))
    current = 0
    records = []
    tie = False
    blocked_seen = False
    competition = False
    while True:
        eligible, blocked = [], []
        for index in sorted(remaining):
            rel = _relation(positions[current], positions[index], s)
            if rel is None:
                continue
            cells = _segment(positions[current], positions[index], s, rel[0])
            if blocking and any(cell in occupied for cell in cells):
                blocked.append(index)
            else:
                eligible.append((rel[1], index, rel, cells))
        blocked_seen = blocked_seen or bool(blocked)
        if not eligible:
            break
        gap = min(item[0] for item in eligible)
        best = [item for item in eligible if item[0] == gap]
        if len(best) != 1:
            tie = True
            break
        competition = competition or len(eligible) > 1
        _, index, rel, cells = best[0]
        records.append({"source": current, "target": index, "axis": rel[0],
                        "gap": rel[1], "sign": rel[2], "cells": cells,
                        "eligible": tuple(item[1] for item in eligible),
                        "blocked": tuple(blocked)})
        remaining.discard(index)
        current = index
    axes = [rec["axis"] for rec in records]
    return {"records": records, "tie": tie, "remaining": frozenset(remaining),
            "visited": frozenset([0] + [rec["target"] for rec in records]),
            "blocked": blocked_seen, "competition": competition,
            "turns": sum(x != y for x, y in zip(axes, axes[1:])),
            "connectors": frozenset(cell for rec in records for cell in rec["cells"])}


def _explanations(positions, s, shape):
    """All pivots that make the squares 3/4 of a quarter-turn constellation.

    Returns (pivot2, missing squares, missing sector) triples.  A complete
    fourfold constellation (no missing square) is reported with sector None so
    callers can reject it as an ambiguous alternative.
    """
    height, width = shape
    present = set(positions)
    anchor = _center2(positions[0], s)
    candidates = {anchor}
    for other in positions[1:]:
        y = _center2(other, s)
        for u, v in ((anchor, y), (y, anchor)):
            n1 = v[0] - u[1] + v[1] + u[0]
            n2 = v[1] + u[0] - v[0] + u[1]
            if n1 % 2 == 0 and n2 % 2 == 0:
                candidates.add((n1 // 2, n2 // 2))
    found = []
    for pivot in sorted(candidates):
        if (pivot[0] - pivot[1]) % 2:
            continue
        valid = True
        missing = set()
        direction = None
        for pos in positions:
            cr, cc = _center2(pos, s)
            a, b = cr - pivot[0], cc - pivot[1]
            orbit = set()
            for turn in range(4):
                ra, rb = _rot(a, b, turn)
                r2, c2 = pivot[0] + ra - s + 1, pivot[1] + rb - s + 1
                if r2 % 2 or c2 % 2:
                    valid = False
                    break
                orbit.add((r2 // 2, c2 // 2))
            if not valid:
                break
            if len(orbit) == 1:
                continue
            absent = orbit - present
            if len(absent) > 1:
                valid = False
                break
            for square in absent:
                if not (0 <= square[0] <= height - s and 0 <= square[1] <= width - s):
                    valid = False
                    break
                mr, mc = _center2(square, s)
                sector = _sector(mr - pivot[0], mc - pivot[1])
                if sector is None or (direction is not None and sector != direction):
                    valid = False
                    break
                direction = sector
                missing.add(square)
            if not valid:
                break
        if not valid:
            continue
        everything = list(positions) + sorted(missing)
        if any(not _separate(p, q, s) for i, p in enumerate(everything)
               for q in everything[i + 1:]):
            continue
        found.append((pivot, frozenset(missing), direction))
    return found


# ---------------------------------------------------------------------------
# Input parsing and the input-only solver
# ---------------------------------------------------------------------------

def _components(grid, color):
    height, width = grid.shape
    unseen = {(int(r), int(c)) for r, c in np.argwhere(grid == color)}
    groups = []
    while unseen:
        first = unseen.pop()
        group = {first}
        stack = [first]
        while stack:
            r, c = stack.pop()
            for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nb in unseen:
                    unseen.remove(nb)
                    group.add(nb)
                    stack.append(nb)
        groups.append(group)
    return groups


def _squares(grid, color, s):
    result = []
    for group in _components(grid, color):
        top = min(r for r, _ in group)
        left = min(c for _, c in group)
        expected = {(r, c) for r in range(top, top + s) for c in range(left, left + s)}
        if group != expected:
            raise ValueError("objects must be solid congruent squares")
        result.append((top, left))
    return sorted(result)


def _analyze(grid):
    """Everything the rule needs, derived from the input grid alone."""
    grid = np.asarray(grid)
    if grid.ndim != 2:
        raise ValueError("input must be a 2D grid")
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) != 3:
        raise ValueError("input must contain background, start and waypoint colors")
    order = np.argsort(-counts, kind="stable")
    if counts[order[0]] == counts[order[1]]:
        raise ValueError("background is ambiguous")
    background = int(colors[order[0]])
    start_color = int(colors[order[2]])
    waypoint_color = int(colors[order[1]])
    if counts[order[1]] == counts[order[2]]:
        raise ValueError("start color is ambiguous")
    area = int(counts[order[2]])
    s = int(round(area ** 0.5))
    if s not in (1, 2, 3) or s * s != area:
        raise ValueError("start must be one 1x1, 2x2 or 3x3 square")
    start = _squares(grid, start_color, s)
    waypoints = _squares(grid, waypoint_color, s)
    if len(start) != 1 or len(waypoints) < 2:
        raise ValueError("need one start square and several waypoint squares")
    positions = start + waypoints
    if any(not _separate(p, q, s) for i, p in enumerate(positions)
           for q in positions[i + 1:]):
        raise ValueError("squares must be separated by background")
    explanations = _explanations(positions, s, grid.shape)
    completing = [e for e in explanations if e[1]]
    if len(explanations) != 1 or len(completing) != 1:
        raise ValueError("no unique incomplete quarter-turn constellation")
    pivot, missing, direction = completing[0]
    hidden = sorted(missing)
    full = positions + hidden
    walk = _trace(full, s)
    if walk["tie"]:
        raise ValueError("nearest clear waypoint is tied")
    return {"background": background, "start_color": start_color,
            "waypoint_color": waypoint_color, "size": s, "positions": positions,
            "hidden": hidden, "pivot": pivot, "direction": direction,
            "full": full, "walk": walk}


def _render(grid, info, hidden=None, connectors=None):
    out = np.array(grid, dtype=int, copy=True)
    s = info["size"]
    for r, c in (info["hidden"] if hidden is None else hidden):
        out[r:r + s, c:c + s] = info["waypoint_color"]
    for r, c in (info["walk"]["connectors"] if connectors is None else connectors):
        if out[r, c] == info["background"]:
            out[r, c] = info["start_color"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(input_grid, dtype=int)
    info = _analyze(grid)
    return _render(grid, info)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _counts(rng, num_train, num_test):
    """Validate explicit counts; None (used by sampling tools) draws a legal one."""
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                   (num_test, 1, 4, "num_test")):
        if value is None:
            continue
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    if num_train is None:
        low = 2 if num_test is None else int(num_test) + 1
        num_train = int(rng.integers(max(low, 3), 7)) if low < 7 else low
    if num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train) - 1) + 1))
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _generic(lanes, q):
    """True when no two different lane pairs share a spacing (fewer ties)."""
    every = sorted(set(lanes) | {-v for v in lanes} | ({0} if q == 0 else set()))
    seen = {}
    for i, a in enumerate(every):
        for b in every[i + 1:]:
            key = tuple(sorted((abs(a), abs(b))))
            if seen.setdefault(b - a, key) != key:
                return False
    return True


def _lattice(rng, s, q, count, limit, spread=4):
    """Positive lane offsets (doubled units) with irregular spacing.

    Lanes are at least s + 1 cells apart, so two squares either share an
    exact row/column span or have disjoint spans: a corridor can only be
    blocked by a square that is itself aligned with it.  Generic spacings
    are preferred because they make nearest-choice ties rarer.
    """
    best = None
    for _ in range(12):
        lanes = []
        if q == 1:
            value = s + 1 if (s + 1) % 2 else s + 2
            value += 2 * int(rng.integers(0, 3))
            lanes.append(value)
        else:
            value = 0
        while len(lanes) < count:
            value += 2 * s + 2 + 2 * int(rng.integers(0, spread))
            lanes.append(value)
        lanes = [v for v in lanes if v + s <= limit]
        if best is None or len(lanes) > len(best):
            best = lanes
        if len(lanes) == count and _generic(lanes, q):
            return lanes
    return best


def _constellation(rng, s, parity, compact):
    """A motif of lattice squares strictly inside the east wedge (+ hub)."""
    q = (s - 1 - parity) % 2
    k_low = 2 if q == 0 else 3
    if compact:
        lanes = _lattice(rng, s, q, k_low, 30, spread=2)
    else:
        count = k_low + int(rng.choice(4, p=[0.3, 0.3, 0.25, 0.15]))
        lanes = _lattice(rng, s, q, count, 30, spread=3)
    every = sorted(set(lanes) | {-v for v in lanes} | ({0} if q == 0 else set()))
    wedge = [(a, b) for b in lanes for a in every if abs(a) < b]
    if len(wedge) < 3:
        return None
    choices = np.arange(3, min(len(wedge), 10) + 1)
    weights = choices - 1.5
    n = int(rng.choice(choices, p=weights / weights.sum()))
    picks = rng.choice(len(wedge), size=n, replace=False)
    motif = [wedge[int(i)] for i in picks]
    hub = q == 0 and rng.random() < 0.3
    return motif, hub


def _layout(rng, s, parity, phase, motif, hub, margin_cap):
    offsets = [(_rot(a, b, t), t) for a, b in motif for t in range(4)]
    if hub:
        offsets.append(((0, 0), None))
    squares = [(((parity + a - s + 1) // 2, (parity + b - s + 1) // 2), t)
               for (a, b), t in offsets]
    min_r = min(p[0] for p, _ in squares)
    min_c = min(p[1] for p, _ in squares)
    extent_r = max(p[0] for p, _ in squares) + s - min_r
    extent_c = max(p[1] for p, _ in squares) + s - min_c
    room_r = 30 - extent_r
    room_c = 30 - extent_c
    if room_r < 0 or room_c < 0:
        return None
    top = int(rng.integers(0, min(margin_cap, room_r) + 1))
    bottom = int(rng.integers(0, min(margin_cap, room_r - top) + 1))
    left = int(rng.integers(0, min(margin_cap, room_c) + 1))
    right = int(rng.integers(0, min(margin_cap, room_c - left) + 1))
    shape = (extent_r + top + bottom, extent_c + left + right)
    placed = [((r - min_r + top, c - min_c + left), t) for (r, c), t in squares]
    visible = [p for p, t in placed if t != phase]
    hidden = [p for p, t in placed if t == phase]
    return shape, visible, hidden


def _pair_features(info):
    walk = info["walk"]
    full = info["full"]
    s = info["size"]
    hidden = set(info["hidden"])
    visited = {full[i] for i in walk["visited"]}
    visible_only = _trace(info["positions"], s)
    no_block = _trace(full, s, blocking=False)
    all_links = set()
    occupied = _cells(full, s)
    for i, p in enumerate(full):
        for q in full[i + 1:]:
            rel = _relation(p, q, s)
            if rel is None:
                continue
            cells = _segment(p, q, s, rel[0])
            if not any(cell in occupied for cell in cells):
                all_links.update(cells)
    # Reflection alternative: mirror the opposite quarter instead of turning.
    pivot = info["pivot"]
    mirrored = set()
    for pos in info["positions"]:
        cr, cc = _center2(pos, s)
        a, b = cr - pivot[0], cc - pivot[1]
        if _sector(a, b) != (info["direction"] + 2) % 4:
            continue
        a, b = (a, -b) if info["direction"] in (0, 2) else (-a, b)
        mirrored.add(((pivot[0] + a - s + 1) // 2, (pivot[1] + b - s + 1) // 2))
    return {
        "links": len(walk["records"]),
        "mirror_differs": mirrored != hidden,
        "hidden": len(hidden),
        "hidden_visited": len(hidden & visited),
        "hidden_unvisited": len(hidden - visited),
        "unvisited": len(walk["remaining"]),
        "blocked": walk["blocked"],
        "competition": walk["competition"],
        "turns": walk["turns"],
        "ignore_hidden_differs": visible_only["tie"]
        or visible_only["connectors"] != walk["connectors"],
        "ignore_blocking_differs": no_block["tie"]
        or no_block["connectors"] != walk["connectors"],
        "connect_all_differs": frozenset(all_links) != walk["connectors"],
        "hub": any(_center2(p, s) == info["pivot"] for p in info["positions"]),
    }


def _make_pair(rng, spec, colors, used):
    s, parity, phase = spec["size"], spec["parity"], spec["phase"]
    lo, hi = spec["links"]
    background, start_color, waypoint_color = colors
    for attempt in range(5000):
        compact = spec.get("compact", False) and attempt < 600
        built = _constellation(rng, s, parity, compact)
        if built is None:
            continue
        motif, hub = built
        laid = _layout(rng, s, parity, phase, motif, hub, spec["margin"])
        if laid is None:
            continue
        shape, visible, hidden = laid
        everything = visible + hidden
        if any(not _separate(p, q, s) for i, p in enumerate(everything)
               for q in everything[i + 1:]):
            continue
        start = visible[int(rng.integers(0, len(visible)))]
        grid = np.full(shape, background, dtype=np.int64)
        for r, c in visible:
            grid[r:r + s, c:c + s] = waypoint_color
        grid[start[0]:start[0] + s, start[1]:start[1] + s] = start_color
        key = (grid.shape, grid.tobytes())
        if key in used:
            continue
        # Cheap route screening before the full input-only analysis.
        full = [start] + sorted(p for p in everything if p != start)
        walk = _trace(full, s)
        if walk["tie"] or not lo <= len(walk["records"]) <= hi:
            continue
        try:
            info = _analyze(grid)
        except ValueError:
            continue
        if set(info["hidden"]) != set(hidden):
            continue
        feats = _pair_features(info)
        if not (feats["ignore_hidden_differs"] and feats["hidden_visited"] >= 1
                and feats["connect_all_differs"]):
            continue
        if spec.get("need_unvisited_hidden") and feats["hidden_unvisited"] < 1:
            continue
        if spec.get("need_blocked") and not feats["ignore_blocking_differs"]:
            continue
        if spec.get("need_competition") and not feats["competition"]:
            continue
        if spec.get("need_mirror") and not feats["mirror_differs"]:
            continue
        if feats["hidden_visited"] < spec.get("min_hidden_visited", 1):
            continue
        used.add(key)
        return {"input": grid, "output": _render(grid, info)}
    raise RuntimeError("could not construct a pair for this specification")


def _plan(rng, num_train, num_test):
    total = num_train + num_test
    size_options = [(2,), (2, 3), (3,), (1, 2), (1, 2, 3), (1,)]
    size_set = size_options[int(rng.choice(6, p=[0.25, 0.3, 0.1, 0.15, 0.12, 0.08]))]
    if len(size_set) > num_train:
        size_set = tuple(sorted(int(x) for x in rng.choice(size_set, size=num_train, replace=False)))
    parity_mode = str(rng.choice(["stable", "mixed"], p=[0.35, 0.65]))
    phase_mode = str(rng.choice(["stable", "mixed"], p=[0.35, 0.65]))
    shift = str(rng.choice(["steady", "escalate", "novel"], p=[0.35, 0.4, 0.25]))
    if shift == "novel" and phase_mode == "stable":
        shift = "steady"
    order = [int(i) for i in rng.permutation(num_train)]
    # square sizes: every size used anywhere is demonstrated in training
    sizes = [int(rng.choice(size_set)) for _ in range(total)]
    for slot, size in zip(order, rng.permutation(size_set)):
        sizes[slot] = int(size)
    # pivot parities: 0 cell-centred pivot, 1 intersection pivot
    if parity_mode == "stable":
        parities = [int(rng.integers(0, 2))] * total
    else:
        parities = [int(rng.integers(0, 2)) for _ in range(total)]
        first = int(rng.integers(0, 2))
        parities[order[0]], parities[order[1]] = first, 1 - first
    # missing quarter
    if phase_mode == "stable":
        phases = [int(rng.integers(0, 4))] * total
    else:
        perm = [int(p) for p in rng.permutation(4)]
        if shift == "novel":
            train_set = perm[:int(rng.integers(2, 4))]
            phases = [train_set[i % len(train_set)] for i in range(num_train)]
            rng.shuffle(phases)
            unseen = [p for p in range(4) if p not in train_set]
            phases += [int(rng.choice(unseen)) for _ in range(num_test)]
        else:
            phases = [perm[i % 4] for i in range(num_train)]
            rng.shuffle(phases)
            phases += [int(rng.integers(0, 4)) for _ in range(num_test)]
    # route lengths: short 2-4, medium 5-8, long 9-16 links
    buckets = {"short": (2, 4), "medium": (5, 8), "long": (9, 16)}
    kinds = ["short", "medium"] if shift == "escalate" else ["short", "medium", "long"]
    cycle = [kinds[i] for i in rng.permutation(len(kinds))]
    links = [buckets[cycle[i % len(cycle)]] for i in range(num_train)]
    for _ in range(num_test):
        links.append(buckets["long"] if shift == "escalate"
                     else buckets[str(rng.choice(["short", "medium", "long"]))])
    # colors: background fixed per puzzle or varied per pair
    fixed_background = rng.random() < 0.5
    background = int(rng.integers(0, 10))
    colors = []
    previous = None
    for index in range(total):
        while True:
            bg = background if fixed_background else int(rng.integers(0, 10))
            rest = [c for c in range(10) if c != bg]
            start_color, waypoint_color = (int(x) for x in rng.choice(rest, size=2, replace=False))
            trio = (bg, start_color, waypoint_color)
            if index == 1 and previous is not None and (
                    start_color == previous[1] or waypoint_color == previous[2]):
                continue
            break
        previous = trio
        colors.append(trio)
    specs = []
    for index in range(total):
        specs.append({"size": sizes[index], "parity": parities[index],
                      "phase": phases[index], "links": links[index],
                      "margin": int(rng.choice([0, 1, 2, 3, 5])),
                      "compact": bool(links[index][0] > 2 and rng.random() < 0.45)})
    # Training lessons: an unvisited hidden square is still restored, a
    # visited square blocks a corridor, and a farther option is passed over.
    by_length = sorted(range(num_train), key=lambda i: (links[i][0], order.index(i)))
    specs[by_length[0]]["need_unvisited_hidden"] = True
    specs[by_length[-1]]["need_blocked"] = True
    specs[by_length[len(by_length) // 2]]["need_competition"] = True
    specs[by_length[len(by_length) // 2]]["need_mirror"] = True
    if shift == "escalate":
        for index in range(num_train, total):
            specs[index]["min_hidden_visited"] = 2
    return {"specs": specs, "colors": colors, "shift": shift,
            "size_set": size_set, "parity_mode": parity_mode,
            "phase_mode": phase_mode}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    plan = _plan(rng, num_train, num_test)
    used = set()
    pairs = []
    for spec, colors in zip(plan["specs"], plan["colors"]):
        pair_rng = np.random.default_rng(int(rng.integers(0, 2 ** 63 - 1)))
        pairs.append(_make_pair(pair_rng, spec, colors, used))
    puzzle = {"train": pairs[:num_train], "test": pairs[num_train:]}
    validate_generated_puzzle(puzzle)
    return puzzle


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _check_grid(grid):
    _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grids must be 2D arrays")
    _require(np.issubdtype(grid.dtype, np.integer), "grids must be integer arrays")
    _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30, "grid size out of range")
    _require(int(grid.min()) >= 0 and int(grid.max()) <= 9, "colors must be 0..9")


def _pair_report(pair):
    inp, out = (np.asarray(pair[k]) if isinstance(pair[k], list) else pair[k]
                for k in ("input", "output"))
    _check_grid(inp)
    _check_grid(out)
    _require(inp.shape == out.shape, "output must keep the input dimensions")
    info = _analyze(inp)
    expected = _render(inp, info)
    _require(np.array_equal(expected, out), "output is not the hidden-quarter walk")
    _require(set(np.unique(out).tolist()) == set(np.unique(inp).tolist()),
             "output must reuse the input palette")
    changed = out != inp
    _require(np.all(inp[changed] == info["background"]), "only background may change")
    _require(np.all(np.isin(out[changed], [info["start_color"], info["waypoint_color"]])),
             "additions use the start and waypoint colors only")
    feats = _pair_features(info)
    _require(feats["links"] >= 2, "route must have at least two links")
    _require(feats["hidden_visited"] >= 1, "route must use a hidden square")
    _require(feats["ignore_hidden_differs"], "ignoring the hidden quarter must change the route")
    _require(feats["connect_all_differs"], "connecting every aligned pair must differ")
    feats.update({"size": info["size"], "parity": info["pivot"][0] % 2,
                  "direction": info["direction"], "shape": inp.shape,
                  "roles": (info["background"], info["start_color"], info["waypoint_color"]),
                  "squares": len(info["positions"]) + len(info["hidden"])})
    return feats


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train and test")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list), "splits must be lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "illegal train/test counts")
    seen = set()
    reports = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pairs must have exactly input and output")
            grid = np.asarray(pair["input"], dtype=np.int64)
            key = (grid.shape, grid.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            reports[split].append(_pair_report(pair))
    tr, te = reports["train"], reports["test"]
    _require(any(r["hidden_unvisited"] >= 1 for r in tr),
             "training must reveal an unvisited hidden square")
    _require(any(r["ignore_blocking_differs"] for r in tr),
             "training must show a visited square blocking a corridor")
    _require(any(r["competition"] for r in tr),
             "training must show a farther eligible square being passed over")
    _require(any(r["mirror_differs"] for r in tr),
             "training must separate quarter-turn completion from reflection")
    _require(len({r["roles"][1] for r in tr}) >= 2,
             "training must vary the start color (connectors follow it)")
    train_sizes = {r["size"] for r in tr}
    _require({r["size"] for r in te} <= train_sizes, "test square size must be demonstrated")
    train_dirs = {r["direction"] for r in tr}
    if len(train_dirs) == 1:
        _require({r["direction"] for r in te} == train_dirs,
                 "a stable missing quarter must stay stable in test")


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------

def _signature(puzzle):
    return tuple((split, p["input"].shape, p["input"].tobytes(), p["output"].tobytes())
                 for split in ("train", "test") for p in puzzle[split])


_HAND_INPUT = ("000000500", "000000000", "500050000", "000000000", "005000000",
               "000000000", "000050000", "000000000", "002000000")
_HAND_OUTPUT = ("000000500", "000000200", "500050200", "000000200", "005222500",
                "002000000", "002050005", "002000000", "002000000")


def _hand_example():
    """Hand-solved 9x9 case: pivot (4,4), east quarter hidden, start at (8,2).

    Visible-only walking stops after one link at (4,2); the hidden dot (4,6)
    is the nearest continuation, then (0,6) is reached and the walk stops.
    The hidden dot (6,8) is never visited but is still restored.
    """
    grid = np.array([[int(c) for c in row] for row in _HAND_INPUT])
    want = np.array([[int(c) for c in row] for row in _HAND_OUTPUT])
    np.testing.assert_array_equal(solve(grid), want)
    info = _analyze(grid)
    _require(info["pivot"] == (8, 8) and info["hidden"] == [(4, 6), (6, 8)],
             "hand example pivot/hidden squares")
    _require(len(_trace(info["positions"], 1)["records"]) == 1,
             "hand example visible-only shortcut stops early")


def self_test() -> None:
    _hand_example()
    sizes, parities, directions, shapes, link_counts = set(), set(), set(), set(), set()
    hidden_visited, hidden_unvisited, roles, squares = set(), set(), set(), set()
    signatures = set()
    hubs = symmetric = escalations = novel = mixed_dirs = stable_dirs = 0
    varied_background = fixed_background = mixed_sizes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        reports = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                np.testing.assert_array_equal(solve(pair["input"]), pair["output"])
                rep = _pair_report(pair)
                reports[split].append(rep)
                sizes.add(rep["size"])
                parities.add(rep["parity"])
                directions.add(rep["direction"])
                shapes.add(rep["shape"])
                link_counts.add(rep["links"])
                hidden_visited.add(rep["hidden_visited"])
                hidden_unvisited.add(rep["hidden_unvisited"])
                roles.add(rep["roles"])
                squares.add(rep["squares"])
                hubs += rep["hub"]
                symmetric += not rep["mirror_differs"]
        train, test = reports["train"], reports["test"]
        train_dirs = {r["direction"] for r in train}
        mixed_dirs += len(train_dirs) > 1
        stable_dirs += len(train_dirs | {r["direction"] for r in test}) == 1
        novel += any(r["direction"] not in train_dirs for r in test)
        escalations += min(r["links"] for r in test) > max(r["links"] for r in train)
        backgrounds = {r["roles"][0] for r in train + test}
        varied_background += len(backgrounds) > 1
        fixed_background += len(backgrounds) == 1
        mixed_sizes += len({r["size"] for r in train}) > 1
    _require(len(signatures) == 42, "seeds must give distinct puzzles")
    _require(sizes == {1, 2, 3} and parities == {0, 1} and directions == {0, 1, 2, 3},
             "square size, pivot type and missing-quarter variety")
    _require(len(shapes) >= 45 and len(link_counts) >= 10 and len(squares) >= 8,
             "canvas, route-length and constellation-size variety")
    _require(len(hidden_visited) >= 4 and 0 in hidden_unvisited and len(hidden_unvisited) >= 4,
             "hidden squares used in several amounts, sometimes all and sometimes not")
    _require(len(roles) >= 80, "palette roles vary between pairs and puzzles")
    _require(hubs >= 2 and symmetric >= 5, "hub squares and mirror-symmetric motifs occur")
    _require(mixed_dirs >= 15 and stable_dirs >= 5 and novel >= 3,
             "stable, mixed and unseen missing quarters")
    _require(escalations >= 5, "some tests need longer walks than any training pair")
    _require(varied_background >= 5 and fixed_background >= 5 and mixed_sizes >= 10,
             "background and square-size regimes vary between puzzles")
    _require(_signature(generate_puzzle(7)) == _signature(generate_puzzle(7)),
             "generation must be deterministic")
    for a, b in ((2, 1), (10, 4), (5, 4), (3, 2)):
        p = generate_puzzle(1000 + a * 10 + b, a, b)
        _require(len(p["train"]) == a and len(p["test"]) == b, "explicit counts honoured")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (4, 5)):
        try:
            generate_puzzle(5, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed")
