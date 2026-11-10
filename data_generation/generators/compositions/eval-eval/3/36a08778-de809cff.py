"""Diffraction repair: input-derived panels, defect diffraction, complementary etching.

No I/O, parent imports, saved examples, or persistent generation state. Grids in
flight are bounded by 30x30. See task_spec.md for the complete family contract.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)), "invalid grid colors")
    return a.astype(np.int8, copy=True)


def _turn(a, k):
    return np.rot90(a, k).copy()


def _components(mask):
    pending = set(map(tuple, np.argwhere(mask)))
    result = []
    while pending:
        start = min(pending)
        pending.remove(start)
        todo, cells = [start], [start]
        while todo:
            r, c = todo.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in pending:
                    pending.remove(p)
                    todo.append(p)
                    cells.append(p)
        result.append(cells)
    return result


def _decode(input_grid):
    """Recover roles, direction, rectangular scene and defects using input only."""
    grid = _grid(input_grid)
    colors = set(map(int, np.unique(grid)))
    _require(0 in colors and len(colors) == 4, "expected zero, two materials and guide")
    choices = []
    for k in range(4):
        a = _turn(grid, k)
        for guide in sorted(colors - {0}):
            entries = list(map(int, np.flatnonzero(a[0] == guide)))
            if not entries or any(b-a < 2 for a, b in zip(entries, entries[1:])):
                continue
            mask = np.zeros(a.shape, dtype=bool)
            lengths = []
            for c in entries:
                rows = np.flatnonzero(a[:, c] == guide)
                if not 2 <= len(rows) <= 4 or not np.array_equal(rows, np.arange(len(rows))):
                    break
                mask[:len(rows), c] = True
                lengths.append(len(rows))
            else:
                if np.array_equal(mask, a == guide):
                    choices.append((k, guide, entries, lengths))
    _require(len(choices) == 1, "guide direction must be unique")
    k, guide, entries, lengths = choices[0]
    a = _turn(grid, k)
    fills = tuple(sorted(colors - {0, guide}))
    clean = np.zeros(a.shape, dtype=np.int8)
    boxes = []
    for cells in _components((a != 0) & (a != guide)):
        if len(cells) == 1:  # isolated exterior speckle
            continue
        rows, cols = zip(*cells)
        r0, r1, c0, c1 = min(rows), max(rows)+1, min(cols), max(cols)+1
        area = (r1-r0)*(c1-c0)
        _require(min(r1-r0, c1-c0) >= 5 and len(cells) >= .70*area, "unsupported panel")
        counts = [sum(int(a[p]) == color for p in cells) for color in fills]
        _require(max(counts) > .75*area, "panel fill must have dense unique evidence")
        color = fills[int(np.argmax(counts))]
        _require(not np.any(clean[r0:r1, c0:c1]), "overlapping component bounds")
        clean[r0:r1, c0:c1] = color
        boxes.append((r0, c0, r1, c1, color))
    _require(2 <= len(boxes) <= 5 and set(map(int, np.unique(clean))) == {0, *fills}, "two to five panels, both materials required")
    for i, (r0, c0, r1, c1, _) in enumerate(boxes):
        for s0, t0, s1, t1, _ in boxes[:i]:
            _require(r1 < s0 or s1 < r0 or c1 < t0 or t1 < c0,
                     "panel bounds must retain a separating background alley")
    _require(all(r0 > max(lengths) for r0, _, _, _, _ in boxes), "panels need entry clearance")
    holes = (clean != 0) & (a == 0)
    _require(np.any(holes), "at least one erasure required")
    return a, clean, holes, fills, guide, entries, lengths, boxes, k


def _route(holes, entries, shortcut=None):
    """Immutable-defect layer simulation; bool footprints make merging idempotent."""
    h, w = holes.shape
    path = np.zeros((h, w), dtype=bool)
    centers = np.zeros((h, w), dtype=bool)
    active = set(entries)
    events, incoming = [], []
    shared = simultaneous = 0
    for r in range(h):
        incoming.append(frozenset(active))
        hits, following = {}, set()
        for c in sorted(active):
            if not holes[r, c]:
                path[r, c] = True
                following.add(c)
                continue
            l = u = c
            if shortcut != "individual":
                while l > 0 and holes[r, l-1]:
                    l -= 1
                while u+1 < w and holes[r, u+1]:
                    u += 1
            hits.setdefault((l, u), []).append(c)
        simultaneous += len(hits) > 1
        fresh = set()
        for (l, u), columns in sorted(hits.items()):
            centers[r, l:u+1] = True
            events.append((r, l, u, len(columns)))
            if r:
                path[r-1, max(0, l-1):min(w, u+2)] = True
            for c in (l-1, u+1):
                if 0 <= c < w:
                    shared += c in fresh or c in following
                    fresh.add(c)
                    following.add(c)
                    path[r, c] = True
        active = set(entries) if shortcut == "straight" else following
    return path, centers, {"events": events, "incoming": incoming,
                           "simultaneous": simultaneous, "shared_flanks": shared}


def _rings(clean, centers, fills):
    demand = np.full(clean.shape, -1, dtype=np.int8)
    h, w = clean.shape
    for r, c in np.argwhere(centers):
        color = fills[1] if clean[r, c] == fills[0] else fills[0]
        for y in range(max(0, r-1), min(h, r+2)):
            for x in range(max(0, c-1), min(w, c+2)):
                if centers[y, x]:
                    continue
                _require(demand[y, x] in (-1, color), "incompatible complementary halos")
                demand[y, x] = color
    return demand


def _render(a, clean, holes, fills, guide, entries, shortcut=None):
    route_mode = shortcut if shortcut in ("straight", "individual") else None
    path, centers, stats = _route(holes, entries, route_mode)
    if shortcut == "all_holes":
        centers = holes.copy()
    out = a.copy() if shortcut == "no_repair" else clean.copy()
    out[path & (clean == 0)] = guide
    for color in fills:
        out[path & (clean == color)] = guide if shortcut == "monochrome" else next(c for c in fills if c != color)
    demands = _rings(clean, centers, fills)
    out[demands >= 0] = guide if shortcut == "monochrome" else demands[demands >= 0]
    out[centers] = guide
    out[a == guide] = guide
    return out, path, centers, stats


def solve(input_grid: np.ndarray) -> np.ndarray:
    a, clean, holes, fills, guide, entries, _, _, k = _decode(input_grid)
    return _turn(_render(a, clean, holes, fills, guide, entries)[0], -k)


def _scene(rng, h, w, count, clearance):
    """Random guillotine free-space subdivision, then independent panel insets."""
    slots = [(clearance, 0, h, w)]
    for _ in range(count-1):
        candidates = [(i, axis) for i, (r0, c0, r1, c1) in enumerate(slots)
                      for axis, side in ((0, r1-r0), (1, c1-c0)) if side >= 11]
        if not candidates:
            return None
        i, axis = candidates[int(rng.integers(len(candidates)))]
        r0, c0, r1, c1 = slots.pop(i)
        side = (r1-r0) if axis == 0 else (c1-c0)
        gap = int(rng.integers(1, min(3, side-10)+1))
        cut = int(rng.integers(5, side-gap-4))
        if axis == 0:
            slots.extend([(r0, c0, r0+cut, c1), (r0+cut+gap, c0, r1, c1)])
        else:
            slots.extend([(r0, c0, r1, c0+cut), (r0, c0+cut+gap, r1, c1)])
    clean = np.zeros((h, w), dtype=np.int8)
    boxes = []
    rng.shuffle(slots)
    for i, (r0, c0, r1, c1) in enumerate(slots):
        rh = int(rng.integers(5, r1-r0+1))
        rw = int(rng.integers(5, c1-c0+1))
        y = int(rng.integers(r0, r1-rh+1))
        x = int(rng.integers(c0, c1-rw+1))
        color = 1+i % 2
        clean[y:y+rh, x:x+rw] = color
        boxes.append((y, x, y+rh, x+rw, color))
    return clean, boxes


def _witness(boxes, kind, rng):
    candidates = []
    for main in boxes:
        r0, c0, r1, c1, color = main
        if r1-r0 < 5 or c1-c0 < (8 if kind == 0 else 9):
            continue
        for p in range(c0+2, c1-3):
            for other in boxes:
                if other[4] == color:
                    continue
                for q in range(other[1]+1, other[3]-1):
                    if q < p-3 or q > p+5:
                        candidates.append((main, p, other, q))
    if not candidates:
        return None
    main, p, other, q = candidates[int(rng.integers(len(candidates)))]
    # A front-face erasure also teaches that its halo extends into exterior
    # space and takes precedence over the upstream guide-colored crosspiece.
    r = main[0]
    s = other[0]+1
    if kind == 0:
        length = int(rng.integers(2, 4))
        entries = [p, q]
        forced = [(r, p, p+length-1), (r+2, p-1, p-1), (s, q, q)]
    else:
        entries = [p, p+2, q]
        forced = [(r, p, p), (r, p+2, p+2), (r+2, p+1, p+1), (s, q, q)]
    return sorted(entries), forced


def _build(rng, h, w, count, kind):
    stub = int(rng.integers(2, 5))
    scene = _scene(rng, h, w, count, stub+1)
    if scene is None:
        return None
    clean, boxes = scene
    forced = []
    if kind is not None:
        witness = _witness(boxes, kind, rng)
        if witness is None:
            return None
        entries, forced = witness
    else:
        # Entries favor panel projections, but may also traverse empty alleys.
        candidates = list(map(int, rng.permutation(w)))
        entries = []
        target = int(rng.integers(1, 5))
        for c in candidates:
            if all(abs(c-d) >= 2 for d in entries):
                entries.append(c)
                if len(entries) == target:
                    break
        entries.sort()
    holes = np.zeros(clean.shape, dtype=bool)
    for r, l, u in forced:
        holes[r, l:u+1] = True
    # General cases combine simultaneous collisions and fresh causal cascades.
    # In witnesses, the reserved events remain plainly legible before extras.
    after = max((r for r, _, _ in forced), default=stub) + 1
    event_rows = set(map(int, rng.choice(np.arange(after, h),
                     size=min(h-after, int(rng.integers(1, 9))), replace=False)))
    active = set(entries)
    for r in range(h):
        if r in event_rows:
            possible = [c for c in sorted(active) if clean[r, c] != 0]
            rng.shuffle(possible)
            for c in possible[:int(rng.integers(1, 4))]:
                box = next(b for b in boxes if b[0] <= r < b[2] and b[1] <= c < b[3])
                n = int(rng.integers(1, min(4, box[3]-box[1]-2)+1))
                l = int(rng.integers(max(box[1], c-n+1), min(c, box[3]-n)+1))
                holes[r, l:l+n] = True
        # One row's transition, avoiding any dependence on painted output.
        following = {c for c in active if not holes[r, c]}
        for c in sorted(active):
            if holes[r, c]:
                l = u = c
                while l > 0 and holes[r, l-1]: l -= 1
                while u+1 < w and holes[r, u+1]: u += 1
                following.update(x for x in (l-1, u+1) if 0 <= x < w)
        active = following
    path, centers, stats = _route(holes, entries)
    if not np.any(centers):
        return None
    # Causally inert erasures use the same panels, sizes and colors as reached
    # erasures. Adding them away from incoming rays and hit-run endpoints cannot
    # alter any existing propagation decision.
    pool = [(int(r), int(c)) for r, c in np.argwhere((clean != 0) & ~holes)
            if c not in stats["incoming"][r]
            and not np.any(holes[r, max(0, c-1):min(w, c+2)])]
    rng.shuffle(pool)
    inert = []
    int_target = int(rng.integers(1, 6))
    for p in pool:
        if len(inert) >= int_target:
            break
        holes[p] = True
        inert.append(p)
    if not inert:
        return None
    try:
        _rings(clean, holes, (1, 2))
    except ValueError:
        return None
    # Input has guide stubs, two densely evidenced materials, holes and noise.
    a = clean.copy()
    a[holes] = 0
    for c in entries:
        a[:stub, c] = 3
    try:
        out, path, marked, _ = _render(a, clean, holes, (1, 2), 3, entries)
    except ValueError:
        return None
    rings = _rings(clean, marked, (1, 2))
    wrong_pool = list(map(tuple, np.argwhere((clean != 0) & ~holes & ~path & (rings < 0))))
    rng.shuffle(wrong_pool)
    for p in wrong_pool[:int(rng.integers(1, 5))]:
        a[p] = 3-int(clean[p])
    speckles = []
    speckle_target = int(rng.integers(0, 7))
    for r, c in rng.permutation(np.argwhere((clean == 0) & (a == 0))):
        if len(speckles) >= speckle_target:
            break
        r, c = int(r), int(c)
        if r < stub+1:
            continue
        if any(0 <= y < h and 0 <= x < w and a[y, x] != 0
               for y, x in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))):
            continue
        a[r, c] = int(rng.integers(1, 3))
        speckles.append((r, c))
        if len(speckles) >= speckle_target:
            break
    try:
        decoded = _decode(a)
        if not np.array_equal(decoded[1], clean) or sorted(decoded[7]) != sorted(boxes):
            return None
        expected = _render(a, clean, holes, (1, 2), 3, entries)[0]
        # Each witness visibly rejects shortcuts, rather than only possessing
        # a feature whose effect might be concealed by later drawing.
        if kind is not None:
            for shortcut in ("all_holes", "straight", "no_repair", "monochrome"):
                wrong = _render(a, clean, holes, (1, 2), 3, entries, shortcut)[0]
                if np.array_equal(wrong, expected):
                    return None
            if kind == 0 and np.array_equal(_render(a, clean, holes, (1, 2), 3, entries, "individual")[0], expected):
                return None
        return a, expected
    except ValueError:
        return None


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, lo, hi, name in ((num_train, 2, 10, "num_train"), (num_test, 1, 4, "num_test")):
        _require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
                 and lo <= value <= hi, f"illegal {name}")
    _require(num_train > num_test, "num_train must exceed num_test")
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_)) and seed >= 0,
             "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = np.array([0, *map(int, rng.choice(np.arange(1, 10), 3, replace=False))], dtype=np.int8)
    varied_palette = bool(rng.integers(2))
    direction = int(rng.integers(4))
    regime = int(rng.integers(3))  # scale, mixed recombination, orientation transfer
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for i in range(count):
            kind = i if split == "train" and i < 2 else None
            for attempt in range(160):
                if regime == 0:
                    lo, hi = (18, 25) if split == "train" else (25, 31)
                else:
                    lo, hi = 18, 31
                h, w = int(rng.integers(lo, hi)), int(rng.integers(lo, hi))
                panels = int(rng.integers(2, 5 if kind is not None else 6))
                # Late attempts use a spacious two-panel scene, still sampled
                # parametrically. No fixed example or saved answer is used.
                if attempt >= 100:
                    h = w = 30
                    panels = 2
                candidate = _build(rng, h, w, panels, kind)
                if candidate is None:
                    continue
                a, o = candidate
                if split == "test":
                    d = _decode(a)
                    _, marked, stats = _route(d[2], d[5])
                    # A test must use a genuinely propagated branch to reach
                    # a later defect, beyond all original straight columns.
                    off_entry = marked.copy()
                    off_entry[:, d[5]] = False
                    if len({r for r, _, _, _ in stats["events"]}) < 2 or not off_entry.any():
                        continue
                local = palette if not varied_palette else np.array(
                    [0, *map(int, rng.choice(np.arange(1, 10), 3, replace=False))], dtype=np.int8)
                k = (direction + (i % 2 if split == "train" else 2)) % 4 if regime == 2 else int(rng.integers(4))
                a, o = _turn(local[a], k), _turn(local[o], k)
                sig = (a.shape, a.tobytes())
                if sig in seen:
                    continue
                seen.add(sig)
                puzzle[split].append({"input": a, "output": o})
                break
            else:
                raise RuntimeError("bounded scene construction exhausted")
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal counts")
    seen, taught = set(), set()
    train_stats = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair")
            a, o = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == o.shape, "shape changed")
            sig = (a.shape, a.tobytes())
            _require(sig not in seen, "duplicate input")
            seen.add(sig)
            decoded = _decode(a)
            ca, clean, holes, fills, guide, entries, _, boxes, k = decoded
            expected, path, centers, stats = _render(ca, clean, holes, fills, guide, entries)
            _require(np.array_equal(o, _turn(expected, -k)), "wrong input-derived output")
            _require(not np.array_equal(a, o), "unchanged pair")
            _require(np.any(centers) and np.any(holes & ~centers), "need reached and inert erasures")
            _require(np.all(clean[holes] != 0), "exterior misidentified as hole")
            _require(np.all(expected[centers] == guide), "center precedence lost")
            _require(np.array_equal(expected[ca == guide], ca[ca == guide]), "stub changed")
            _require(2 <= len(boxes) <= 5, "panel count")
            _rings(clean, holes, fills)  # compatibility is input-wide, not label-derived
            if split == "test":
                off_entry = centers.copy()
                off_entry[:, entries] = False
                _require(len({r for r, _, _, _ in stats["events"]}) >= 2 and off_entry.any(),
                         "test lacks a descendant-activated defect")
            if split == "train":
                train_stats.append((stats, clean, centers))
                for shortcut in ("all_holes", "straight", "individual", "no_repair", "monochrome"):
                    wrong = _render(ca, clean, holes, fills, guide, entries, shortcut)[0]
                    if not np.array_equal(wrong, expected):
                        taught.add(shortcut)
    _require(taught == {"all_holes", "straight", "individual", "no_repair", "monochrome"},
             "training admits a designated shortcut")
    _require(any(s[0]["simultaneous"] and s[0]["shared_flanks"] for s in train_stats), "merging witness missing")
    _require(any(len(set(map(int, clean[centers]))) == 2 for _, clean, centers in train_stats), "both complementary colors untaught")


def self_test():
    """Stream 42 seeds; collect bounded hashes/counters, never retain puzzles."""
    # Hand-calculated diffraction fixture: a two-cell run followed by a hole
    # on its left descendant, plus an inert defect outside the causal tree.
    holes = np.zeros((8, 10), dtype=bool)
    holes[2, 3:5] = True
    holes[4, 2] = holes[6, 7] = True
    path, centers, stats = _route(holes, [3])
    _require(set(map(tuple, np.argwhere(centers))) == {(2, 3), (2, 4), (4, 2)}, "manual reached-set fixture")
    _require(stats["incoming"][3] == {2, 5} and stats["incoming"][5] == {1, 3, 5}, "manual flank fixture")
    _require(path[1, 2:6].all() and path[7, [1, 3, 5]].all(), "manual trace fixture")
    clean = np.ones((8, 10), dtype=np.int8)
    clean[:, 8:] = 2
    a = clean.copy(); a[holes] = 0
    expected = _render(a, clean, holes, (1, 2), 3, [3])[0]
    _require(expected[2, 3] == expected[2, 4] == expected[4, 2] == 3, "manual center precedence")
    _require(expected[3, 3] == expected[7, 5] == 2 and expected[7, 7] == 1, "manual material etching")
    conflicting = np.zeros((8, 10), dtype=bool)
    conflicting[4, 7:9] = True
    try:
        _rings(clean, conflicting, (1, 2))
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting material halos accepted")
    signatures, shapes, geometry, panels, lengths, directions, palettes = (set() for _ in range(7))
    totals = {"pairs": 0, "reached": 0, "inert": 0, "simultaneous": 0, "shared_flanks": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, o = pair["input"], pair["output"]
                _require(np.array_equal(solve(a.copy()), o), "solver mismatch")
                d = _decode(a)
                ca, clean, holes, fills, guide, entries, _, boxes, k = d
                _, path, centers, stats = _render(ca, clean, holes, fills, guide, entries)
                digest.update(bytes(a.shape)); digest.update(a.tobytes()); digest.update(o.tobytes())
                shapes.add(a.shape)
                geometry.add((clean.shape, tuple(b[:4] for b in boxes), tuple(entries)))
                panels.add(len(boxes)); directions.add(k); palettes.add((*fills, guide))
                lengths.update(u-l+1 for _, l, u, _ in stats["events"])
                totals["pairs"] += 1
                totals["reached"] += int(centers.sum())
                totals["inert"] += int((holes & ~centers).sum())
                totals["simultaneous"] += bool(stats["simultaneous"])
                totals["shared_flanks"] += bool(stats["shared_flanks"])
                # Remapping materials and guide must commute with the solver.
                remap = np.arange(10, dtype=np.int8)
                remap[1:] = np.roll(remap[1:], 2)
                _require(np.array_equal(solve(remap[a]), remap[o]), "literal color shortcut")
                _require(np.array_equal(solve(np.fliplr(a)), np.fliplr(o)), "reflection equivariance")
        signatures.add(digest.digest())
    _require(len(signatures) == 42 and len(geometry) >= 180, "structural diversity collapsed")
    _require(len(shapes) >= 50 and panels == {2, 3, 4, 5}, "size/count variation collapsed")
    _require({1, 2, 3, 4} <= lengths and directions == set(range(4)) and len(palettes) >= 35, "role/run variation collapsed")
    # All legal counts, determinism, invalid types and boundary values.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(1701, nt, nq))
    a, b = generate_puzzle(7, 2, 1), generate_puzzle(7, 2, 1)
    _require(all(np.array_equal(p[k], q[k]) for s in a for p, q in zip(a[s], b[s])
                 for k in ("input", "output")), "nondeterminism")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("mutated answer accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shapes": len(shapes),
            "clean_geometries": len(geometry), "panel_counts": sorted(panels),
            "run_lengths": sorted(lengths), "palettes": len(palettes), **totals}
