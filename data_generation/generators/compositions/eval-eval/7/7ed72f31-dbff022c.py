"""Invisible symmetry molds. Only standard-library modules and NumPy are used.

The solver reads all conventions from the input. Construction never supplies
ownership, the palette, the operation pool, or RNG state to it.
"""
from __future__ import annotations

import json

import numpy as np


FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = FOUR + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape):
        raise ValueError("Grid dimensions must be 1..30")
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9)):
        raise ValueError("Grid cells must be integer colors 0..9")
    return a.astype(np.int64, copy=True)


def _components(mask, diagonal=False):
    todo = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    steps = EIGHT if diagonal else FOUR
    while todo:
        p = min(todo)
        todo.remove(p)
        stack, cells = [p], {p}
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in todo:
                    todo.remove(q)
                    cells.add(q)
                    stack.append(q)
        result.append(cells)
    return result


def _bbox(cells):
    rr, cc = zip(*cells)
    return min(rr), max(rr), min(cc), max(cc)


def _fingerprint(a):
    return a.shape, tuple(map(int, a.flat))


def _separated(a, b):
    # Inclusive boxes with at least one empty intervening row or column.
    return a[1] + 1 < b[0] or b[1] + 1 < a[0] or a[3] + 1 < b[2] or b[3] + 1 < a[2]


def _holes(wall, diagonal=False):
    """Complement components in a padded box, never using the canvas as a wall."""
    r0, r1, c0, c1 = _bbox(wall)
    h, w = r1 - r0 + 3, c1 - c0 + 3
    passable = np.ones((h, w), dtype=bool)
    for r, c in wall:
        passable[r - r0 + 1, c - c0 + 1] = False
    holes = []
    for cells in _components(passable, diagonal):
        if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells):
            holes.append({(r + r0 - 1, c + c0 - 1) for r, c in cells})
    return holes


def _read_key(a):
    colors, counts = np.unique(a, return_counts=True)
    if np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("Ambiguous background")
    bg = int(colors[counts.argmax()])
    h, w = a.shape
    candidates = []
    for cells in _components(a != bg, True):
        r0, r1, c0, c1 = _bbox(cells)
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        if len(cells) != hh * ww or len({int(a[q]) for q in cells}) < 2:
            continue
        lanes = []
        if hh == 2 and ww >= 3:
            if r0 == 0:
                lanes.append(("top", a[r0, c0:c1 + 1], a[r1, c0:c1 + 1]))
            if r1 == h - 1:
                lanes.append(("bottom", a[r1, c0:c1 + 1], a[r0, c0:c1 + 1]))
        if ww == 2 and hh >= 3:
            if c0 == 0:
                lanes.append(("left", a[r0:r1 + 1, c0], a[r0:r1 + 1, c1]))
            if c1 == w - 1:
                lanes.append(("right", a[r0:r1 + 1, c1], a[r0:r1 + 1, c0]))
        for side, outer, inner in lanes:
            outer, inner = list(map(int, outer)), list(map(int, inner))
            equal = [i for i, (s, t) in enumerate(zip(outer, inner)) if s == t]
            if len(equal) != 1 or equal[0] not in (0, len(outer) - 1):
                continue
            k = equal[0]
            marker = outer[k]
            sources = [s for i, s in enumerate(outer) if i != k]
            fills = [t for i, t in enumerate(inner) if i != k]
            if len(set(outer)) != len(outer) or marker in sources + fills:
                continue
            candidates.append({"background": bg, "marker": marker,
                               "mapping": dict(zip(sources, fills)), "side": side,
                               "cells": cells, "box": (r0, r1, c0, c1),
                               "header_end": k == 0})
    if len(candidates) != 1:
        raise ValueError("Expected one unambiguous edge key with an end swatch")
    return candidates[0]


def _classify(marker):
    r0, r1, c0, c1 = _bbox(marker)
    if len(marker) == 1:
        return "P"
    if r0 == r1 and len(marker) == c1 - c0 + 1:
        return "H"
    if c0 == c1 and len(marker) == r1 - r0 + 1:
        return "V"
    raise ValueError("A marker must be a point or straight contiguous segment")


def _echo(source, marker, kind):
    ar, ac = min(marker)
    if kind == "H":
        return {(2 * ar - r, c) for r, c in source}
    if kind == "V":
        return {(r, 2 * ac - c) for r, c in source}
    return {(2 * ar - r, 2 * ac - c) for r, c in source}


def _candidate(a, trace, marker, kind, key):
    r0, r1, c0, c1 = _bbox(trace)
    # Bounding-box containment plus one cell accommodates truncated cap ends.
    if not all(r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1 for r, c in marker):
        return None
    ar, ac = min(marker)
    if kind == "H" and not (r1 <= ar or r0 >= ar):
        return None
    if kind == "V" and not (c1 <= ac or c0 >= ac):
        return None
    if kind == "P" and not (r1 <= ar or r0 >= ar or c1 <= ac or c0 >= ac):
        return None
    echo = _echo(trace, marker, kind)
    h, w = a.shape
    if any(not (0 <= r < h and 0 <= c < w) for r, c in echo):
        return None
    if echo & marker or echo & key["cells"]:
        return None
    bg, color = key["background"], int(a[min(trace)])
    # An echo can coincide with its own original trace, never with other ink.
    if any(a[q] != bg and q not in trace for q in echo):
        return None
    return {"kind": kind, "marker_cells": marker, "source": trace,
            "echo": echo, "wall": trace | echo, "color": color,
            "box": _bbox(trace | echo | marker)}


def _read_scene(value):
    a = _grid(value)
    key = _read_key(a)
    bg, marker_color = key["background"], key["marker"]
    ink = a != bg
    for q in key["cells"]:
        ink[q] = False
    markers = _components(ink & (a == marker_color), True)
    traces = []
    for color in sorted(set(map(int, np.unique(a))) - {bg, marker_color}):
        traces.extend(_components(ink & (a == color), True))
    if not markers or len(markers) != len(traces):
        raise ValueError("Every trace needs exactly one marker")
    groups, used = [], set()
    for marker in markers:
        kind = _classify(marker)
        candidates = []
        for i, trace in enumerate(traces):
            g = _candidate(a, trace, marker, kind, key)
            if g is not None:
                candidates.append((i, g))
        if len(candidates) != 1 or candidates[0][0] in used:
            raise ValueError("Ambiguous or missing marker ownership")
        i, g = candidates[0]
        used.add(i)
        if _holes(g["source"]):
            raise ValueError("Original traces must be open")
        g["holes"] = _holes(g["wall"])
        if not _separated(g["box"], key["box"]):
            raise ValueError("Mold and key need a separating halo")
        if any(not _separated(g["box"], other["box"]) for other in groups):
            raise ValueError("Complete molds need separated bounding boxes")
        groups.append(g)
    return a, key, groups


def _paint(a, key, groups, override=None, ignore_mapping=False):
    out = a.copy()
    for i, g in enumerate(groups):
        color = g["color"]
        if not ignore_mapping and color not in key["mapping"]:
            continue
        fill = color if ignore_mapping else key["mapping"][color]
        holes = g["holes"] if override is None or i not in override else _holes(override[i])
        for cells in holes:
            for q in cells:
                if a[q] == key["background"]:
                    out[q] = fill
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the result from the supplied input alone; never mutate the input."""
    a, key, groups = _read_scene(input_grid)
    return _paint(a, key, groups)


def _translated_wall(g):
    r0, _, c0, _ = _bbox(g["source"])
    er0, _, ec0, _ = _bbox(g["echo"])
    return g["source"] | {(r + er0 - r0, c + ec0 - c0) for r, c in g["source"]}


def _cap_gap(g):
    r0, r1, c0, c1 = _bbox(g["wall"])
    ar, ac = min(g["marker_cells"])
    if g["kind"] == "V":
        axis_dim = 1
    elif g["kind"] == "H":
        axis_dim = 0
    else:
        sr0, sr1, sc0, sc1 = _bbox(g["source"])
        axis_dim = 1 if sc0 >= ac or sc1 <= ac else 0
    caps = ((r0, ac), (r1, ac)) if axis_dim == 1 else ((ar, c0), (ar, c1))
    return any(q not in g["wall"] for q in caps)


def _features(a, key, groups, out):
    mapped = [g for g in groups if g["color"] in key["mapping"]]
    active = [g for g in mapped if any(a[q] == key["background"] for cells in g["holes"] for q in cells)]
    witnesses = set()
    for i, g in enumerate(groups):
        if g in active and not np.array_equal(out, _paint(a, key, groups, {i: _translated_wall(g)})):
            witnesses.add(g["kind"])
    fills, sources = set(key["mapping"].values()), set(key["mapping"])
    all_holes = [cells for g in active for cells in g["holes"]]
    return {"shape": a.shape, "side": key["side"], "background": key["background"],
            "marker": key["marker"], "key_length": len(sources),
            "count": len(groups), "kinds": {g["kind"] for g in groups},
            "witnesses": witnesses,
            "active": bool(active),
            "chain": any(key["mapping"][g["color"]] in sources and
                         key["mapping"][key["mapping"][g["color"]]] != key["mapping"][g["color"]]
                         for g in active),
            "gap": any(not g["holes"] for g in mapped),
            "fill_only": any(g["holes"] and g["color"] in fills - sources for g in groups),
            "absent": any(g["holes"] and g["color"] not in sources | fills for g in groups),
            "multi": any(len(g["holes"]) >= 2 for g in active),
            "partial": any(g["holes"] and _cap_gap(g) for g in active),
            "irregular": any(len(cells) != (_bbox(cells)[1] - _bbox(cells)[0] + 1) *
                              (_bbox(cells)[3] - _bbox(cells)[2] + 1) for cells in all_holes),
            "diagonal": any(set().union(*g["holes"]) !=
                            set().union(set(), *_holes(g["wall"], True)) for g in active),
            "repeated": len({g["color"] for g in groups}) < len(groups),
            "cavity_counts": tuple(len(g["holes"]) for g in groups),
            "border": any(r in (0, a.shape[0] - 1) or c in (0, a.shape[1] - 1)
                          for g in groups for r, c in g["wall"]),
            "silhouettes": tuple(_normalized_shape(g["source"]) for g in groups)}


def _normalized_shape(cells):
    r0, r1, c0, c1 = _bbox(cells)
    return (r1 - r0 + 1, c1 - c0 + 1, tuple(sorted((r - r0, c - c0) for r, c in cells)))


def validate_generated_puzzle(puzzle):
    """Check rule invariants and actual demonstration coverage from returned grids."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _check_counts(nt, ns)
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, key, groups = _read_scene(pair["input"])
            out = _grid(pair["output"])
            assert a.shape == out.shape
            sig = _fingerprint(a)
            assert sig not in seen, "Duplicate input"
            seen.add(sig)
            bg = key["background"]
            assert np.array_equal(a[a != bg], out[a != bg]), "Changed input ink"
            assert np.array_equal(_paint(a, key, groups), out), "Incorrect virtual-cavity fill"
            for g in groups:
                for q in g["echo"] - g["source"]:
                    assert a[q] == bg and out[q] == bg, "Echo wall must remain invisible"
                if g["color"] not in key["mapping"]:
                    assert all(out[q] == a[q] for cells in g["holes"] for q in cells), "Unlisted mold filled"
            f = _features(a, key, groups, out)
            assert f["active"] and np.any(a != out), "Every pair must expose a mapped chamber"
            # Omitting A finds no cavities; omitting B's lookup paints the wrong colors.
            assert all(not _holes(g["source"]) for g in groups)
            assert not np.array_equal(_paint(a, key, groups, ignore_mapping=True), out)
            records[split].append(f)
    train, test = records["train"], records["test"]
    taught = set().union(*(f["witnesses"] for f in train))
    assert set().union(*(f["kinds"] for f in test)) <= taught, "Untaught symmetry or concealed orientation"
    for name in ("chain", "gap", "fill_only", "absent", "multi", "irregular", "diagonal"):
        assert any(f[name] for f in train), "Missing training lesson: " + name
    sides = {f["side"] for f in train}
    assert len(sides) >= 2, "Lookup direction requires two demonstrated sides"
    assert all(f["side"] not in sides or f["count"] > max(t["count"] for t in train) for f in test), "Missing train/test shift"
    assert test[0]["partial"], "First test must combine leakage with chamber splitting"


def _check_counts(num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _dilate(cells):
    return cells | {(r + dr, c + dc) for r, c in cells for dr, dc in FOUR}


def _prototype(rng, kind, bars=0, gap="none", compact=False):
    h = 3 if compact else int(rng.integers(3, 6))
    max_width = 3 if compact else int(rng.integers(3, 7))
    min_width = 2
    if kind == "P" and bars:
        # A point echo reverses an offset bar's row. Join its two axis ends
        # with a right-hand semicircle around the passable center. Its echo
        # completes an internal diamond, connecting the divided chambers.
        h = max(4, h)
        min_width, max_width = 4, max(5, max_width)
    widths = list(map(int, rng.integers(min_width, max_width + 1, 2 * h - 1)))
    # An unequal profile is a geometric witness, not just a changed palette.
    widths[0], widths[-1] = min_width, max_width
    if rng.random() < .5:
        widths.reverse()
    cavity = {(r, c) for r, width in zip(range(-h + 1, h), widths) for c in range(width)}
    thickness = 1 if compact else int(rng.choice([1, 1, 1, 2]))
    outer = cavity.copy()
    for _ in range(thickness):
        outer = _dilate(outer)
    source = {(r, c) for r, c in outer - cavity if c >= 0}
    bar_row = int(rng.choice([-2, 2] if kind == "P" else [-1, 1]))
    if bars:
        edge = max(c for r, c in source if r == bar_row)
        source |= {(bar_row, c) for c in range(edge + 1)}
        if kind == "P":
            source |= {(r, 2 - abs(r)) for r in range(-2, 3)}
    if gap != "none":
        # Remove only free cap ends, retaining one connected source arc.
        which = ("top", "bottom") if gap == "both" else (gap,)
        for side in which:
            source -= {(r, 0) for r, c in source if c == 0 and
                       (r < -h + 1 if side == "top" else r > h - 1)}
    if kind == "P":
        marker = {(0, 0)}
    else:
        usable = set(range(-h + 2, h - 1)) - ({bar_row} if bars else set())
        runs = []
        for first in sorted(usable):
            for last in sorted(usable):
                if last > first and all(x in usable for x in range(first, last + 1)):
                    runs.append((first, last))
        first, last = runs[int(rng.integers(len(runs)))]
        marker = {(r, 0) for r in range(first, last + 1)}
    # Construct a vertical mouth; H transposes it, P can face any direction.
    turns = 1 if kind == "H" else int(rng.integers(4)) if kind == "P" else 0
    side_sign = int(rng.choice([-1, 1]))
    def move(cells):
        result = set()
        for r, c in cells:
            c *= side_sign
            for _ in range(turns):
                r, c = -c, r
            result.add((r, c))
        return result
    source, marker = move(source), move(marker)
    echo = _echo(source, marker, kind)
    r0, r1, c0, c1 = _bbox(source | echo | marker)
    return {"kind": kind, "source": {(r - r0, c - c0) for r, c in source},
            "marker_cells": {(r - r0, c - c0) for r, c in marker},
            "height": r1 - r0 + 1, "width": c1 - c0 + 1}


def _palette(rng, need_chain=False):
    colors = list(map(int, rng.permutation(10)))
    bg, marker = colors[:2]
    n = int(rng.integers(2, 6))
    sources, outside = colors[2:2 + n], colors[2 + n:]
    fills = []
    style = int(rng.integers(3))
    common = int(rng.choice(outside))
    for s in sources:
        pool = outside if style == 0 else [c for c in colors[2:] if c != s]
        fills.append(common if style == 2 else int(rng.choice(pool)))
    # A fill-only color always exists, making inert and reversed-key evidence possible.
    fills[0] = outside[0]
    if need_chain:
        fills[0], fills[1] = sources[1], outside[0]
    # Keep one color absent from both lanes for the absent-color distractor.
    absent = outside[-1]
    fills = [outside[0] if c == absent else c for c in fills]
    return bg, marker, dict(zip(sources, fills)), sources[0], outside[0], absent


def _put_key(a, side, entries, start):
    for i, (source, fill) in enumerate(entries):
        k = start + i
        if side == "top":
            a[0, k], a[1, k] = source, fill
        elif side == "bottom":
            a[-1, k], a[-2, k] = source, fill
        elif side == "left":
            a[k, 0], a[k, 1] = source, fill
        else:
            a[k, -1], a[k, -2] = source, fill


def _pack(rng, prototypes, side, entries, bg, compact=False, deterministic=False):
    count = len(prototypes)
    largest_h = max(p["height"] for p in prototypes)
    largest_w = max(p["width"] for p in prototypes)
    for attempt in range(12 if not deterministic else 1):
        if deterministic:
            h = w = 30
        else:
            low = 12 if count <= 2 else 18 if count == 3 else 21
            h = int(rng.integers(max(low, largest_h), 31))
            w = int(rng.integers(max(low, largest_w), 31))
            if rng.random() < .16:
                h = w = max(h, w)
        a = np.full((h, w), bg, dtype=np.int64)
        axis = w if side in ("top", "bottom") else h
        start = int(rng.integers(axis - len(entries) + 1))
        _put_key(a, side, entries, start)
        blocked = np.zeros((h, w), dtype=bool)
        for r, c in np.argwhere(a != bg):
            blocked[max(0, r - 1):min(h, r + 2), max(0, c - 1):min(w, c + 2)] = True
        origins = [None] * count
        order = sorted(range(count), key=lambda i: -prototypes[i]["height"] * prototypes[i]["width"])
        for i in order:
            ph, pw = prototypes[i]["height"], prototypes[i]["width"]
            positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
            if not deterministic:
                rng.shuffle(positions)
                if rng.random() < .40:
                    positions.sort(key=lambda q: min(q[0], q[1], h - ph - q[0], w - pw - q[1]))
            for r, c in positions:
                if not blocked[r:r + ph, c:c + pw].any():
                    origins[i] = (r, c)
                    blocked[max(0, r - 1):min(h, r + ph + 1), max(0, c - 1):min(w, c + pw + 1)] = True
                    break
            else:
                break
        else:
            return a, origins
    return None


def _make_pair(rng, side, specs, required_witnesses=()):
    need_chain = any(s["role"] == "chain" for s in specs)
    for attempt in range(32):
        compact = attempt >= 12
        bg, marker, mapping, primary, fill_only, absent = _palette(rng, need_chain)
        prototypes = [_prototype(rng, s["kind"], s.get("bars", 0), s.get("gap", "none"), compact) for s in specs]
        entries = list(mapping.items())
        rng.shuffle(entries)
        entries.insert(0 if rng.random() < .5 else len(entries), (marker, marker))
        packed = _pack(rng, prototypes, side, entries, bg, compact, deterministic=attempt >= 24)
        if packed is None:
            continue
        a, origins = packed
        for spec, proto, (r0, c0) in zip(specs, prototypes, origins):
            role = spec["role"]
            color = primary if role == "chain" else fill_only if role == "fill_only" else absent if role == "absent" else int(rng.choice(list(mapping)))
            for r, c in proto["source"]:
                a[r + r0, c + c0] = color
            for r, c in proto["marker_cells"]:
                a[r + r0, c + c0] = marker
        try:
            a, key, groups = _read_scene(a)
        except ValueError:
            continue
        if key["background"] != bg or len(groups) != len(specs):
            continue
        out = _paint(a, key, groups)
        f = _features(a, key, groups, out)
        if not f["active"] or not set(required_witnesses) <= f["witnesses"]:
            continue
        if need_chain and not f["chain"]:
            continue
        if any(s.get("lesson") == "multi" for s in specs) and not f["multi"]:
            continue
        if any(s.get("lesson") == "partial" for s in specs) and not f["partial"]:
            continue
        if any(s["role"] == "fill_only" for s in specs) and not f["fill_only"]:
            continue
        if any(s["role"] == "absent" for s in specs) and not f["absent"]:
            continue
        if any(s.get("lesson") == "gap" for s in specs) and not f["gap"]:
            continue
        return {"input": a, "output": out}
    raise RuntimeError("Bounded mold construction failed")


def _spec(kind, role="active", bars=0, gap="none", lesson=None):
    return {"kind": kind, "role": role, "bars": bars, "gap": gap, "lesson": lesson}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    pools = [("H",), ("V",), ("P",), ("H", "V"), ("H", "P"), ("V", "P"), ("H", "V", "P")]
    kinds = pools[int(rng.integers(len(pools)))]
    ordered = list(map(str, rng.permutation(kinds)))
    sides = list(map(str, rng.permutation(["top", "left", "bottom", "right"])))
    novel_side = bool(rng.integers(2))
    plans = [[] for _ in range(int(num_train))]
    # At most two required kinds in the first scene and one in the second.
    for i, kind in enumerate(ordered):
        row = i % 2
        plans[row].append(_spec(kind, "chain" if i == 0 else "active",
                                bars=1 if i == 0 else 0, lesson="multi" if i == 0 else None))
    for row in range(2):
        if not plans[row]:
            plans[row].append(_spec(str(rng.choice(kinds))))
    # These color/topology contrasts survive even at the minimum count 2/1.
    plans[0].append(_spec(str(rng.choice(kinds)), "fill_only"))
    plans[1].append(_spec(str(rng.choice(kinds)), gap="both", lesson="gap"))
    plans[1].append(_spec(str(rng.choice(kinds)), "absent"))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            if split == "train":
                specs = list(plans[index])
                if not specs:
                    for j in range(int(rng.integers(1, 4))):
                        kind = str(rng.choice(kinds))
                        specs.append(_spec(kind, bars=int(rng.integers(2)) if j == 0 else 0))
                witnesses = {s["kind"] for s in specs if s["role"] in ("chain", "active") and s["gap"] == "none"}
                side = sides[index % 2]
            else:
                wanted = 4 if not novel_side else int(rng.integers(1 if index else 2, 5))
                specs = []
                if index == 0:
                    kind = str(rng.choice(kinds))
                    # P loses both cap endpoints; a line mold loses one cap.
                    specs.append(_spec(kind, bars=1, gap="both" if kind == "P" else "top", lesson="partial"))
                else:
                    specs.append(_spec(str(rng.choice(kinds)), bars=int(rng.integers(2))))
                while len(specs) < wanted:
                    kind = str(rng.choice(kinds))
                    roll = float(rng.random())
                    if roll < .23:
                        specs.append(_spec(kind, "fill_only" if rng.random() < .5 else "absent"))
                    elif roll < .43:
                        specs.append(_spec(kind, gap="both"))
                    else:
                        specs.append(_spec(kind, bars=int(rng.integers(2))))
                witnesses = ()
                side = sides[2 + index % 2] if novel_side else sides[index % 2]
            rng.shuffle(specs)
            for _ in range(10):
                pair = _make_pair(rng, side, specs, witnesses)
                sig = _fingerprint(pair["input"])
                if sig not in seen:
                    seen.add(sig)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not produce a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def _fixture(point=False, gap=False, bar=False, inert=False):
    """Hand-specified shapes and answers; no procedural profile or label solver."""
    a = np.zeros((19, 18), dtype=np.int64)
    _put_key(a, "top", [(9, 9), (2, 5), (3, 6)] if inert else [(9, 9), (1, 2), (2, 5)], 1)
    ar, ac = 9, 7
    if point:
        source = {(-3, 0), (-3, 1), (3, 0), (3, 1), (3, 2),
                  (-2, 2), (-1, 2), (0, 3), (1, 3), (2, 3)}
        marker = {(0, 0)}
        rows = {-2: range(-2, 2), -1: range(-2, 2), 0: range(-2, 3),
                1: range(-1, 3), 2: range(-1, 3)}
    else:
        source = {(-3, c) for c in range(3)} | {(3, c) for c in range(3)} | {(r, 3) for r in range(-2, 3)}
        marker = {(0, 0), (1, 0)}
        rows = {r: range(-2, 3) for r in range(-2, 3)}
        if bar:
            source |= {(-1, c) for c in range(4)}
            rows = {r: range(-2, 3) for r in range(0, 3)}
    if gap:
        source.remove((-3, 0))
        if not bar:
            rows = {}
    for r, c in source:
        a[ar + r, ac + c] = 1
    for r, c in marker:
        a[ar + r, ac + c] = 9
    expected = a.copy()
    if not inert:
        for r, cols in rows.items():
            for c in cols:
                if (r, c) not in marker:
                    expected[ar + r, ac + c] = 2
    return a, expected


def self_test():
    """Bounded regression suite plus measured structural diversity over 42 seeds."""
    for point, gap, bar, inert in ((False, False, False, False), (True, False, False, False),
                                    (False, True, False, False), (False, True, True, False),
                                    (False, False, False, True)):
        a, expected = _fixture(point, gap, bar, inert)
        before = a.copy()
        assert np.array_equal(solve(a), expected), "Hand-derived fixture failed"
        assert np.array_equal(a, before), "Solver mutated its input"
        for turn in range(4):
            palette = np.random.default_rng(101 + turn).permutation(10)
            rotated = palette[np.rot90(a, turn)]
            assert np.array_equal(solve(rotated), palette[np.rot90(expected, turn)])
    puzzles, facts, signatures = [], [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        signatures.add(tuple(_fingerprint(x["input"]) for split in ("train", "test") for x in p[split]))
        for split in ("train", "test"):
            for pair in p[split]:
                a, key, groups = _read_scene(pair["input"])
                assert np.array_equal(solve(a), pair["output"])
                facts.append(_features(a, key, groups, pair["output"]))
    assert len(signatures) == 42
    assert len({f["shape"] for f in facts}) >= 30
    assert {f["count"] for f in facts} == {1, 2, 3, 4}
    assert {f["side"] for f in facts} == {"top", "bottom", "left", "right"}
    assert {f["key_length"] for f in facts} == {2, 3, 4, 5}
    assert len({f["background"] for f in facts}) >= 8
    assert len({f["marker"] for f in facts}) >= 8
    assert len({s for f in facts for s in f["silhouettes"]}) >= 100
    assert set().union(*(f["kinds"] for f in facts)) == {"H", "V", "P"}
    assert {x for f in facts for x in f["cavity_counts"]} >= {0, 1, 2, 3}
    assert any(f["border"] for f in facts) and any(f["repeated"] for f in facts)
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(271828 + nt * 10 + ns, nt, ns)
            validate_generated_puzzle(p)
            assert (len(p["train"]), len(p["test"])) == (nt, ns)
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                   (4, False), (2.5, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    again = generate_puzzle(17)
    assert all(np.array_equal(a[k], b[k]) for split in ("train", "test")
               for a, b in zip(again[split], puzzles[17][split]) for k in ("input", "output"))
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in ("train", "test")}
    a = broken["train"][0]["input"]
    _, key, groups = _read_scene(a)
    ghost = min(set().union(*(g["echo"] - g["source"] for g in groups)))
    broken["train"][0]["output"][ghost] = next(g["color"] for g in groups if ghost in g["echo"])
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("Accepted a visible echo wall")
    return {"seeds": 42, "pairs": len(facts), "distinct_puzzles": len(signatures),
            "canvas_shapes": len({f["shape"] for f in facts}),
            "trace_silhouettes": len({s for f in facts for s in f["silhouettes"]}),
            "counts": sorted({f["count"] for f in facts}),
            "cavity_counts": sorted({x for f in facts for x in f["cavity_counts"]}),
            "features": {name: sum(bool(f[name]) for f in facts) for name in
                         ("chain", "gap", "fill_only", "absent", "multi", "partial", "irregular", "diagonal", "border", "repeated")}}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
