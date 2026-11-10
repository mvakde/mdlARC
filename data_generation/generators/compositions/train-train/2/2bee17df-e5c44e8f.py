"""ARC family: an expanding spiral in the ordinal coordinates of clear streets.

Only standard library and NumPy are used. All solving conventions are visible
in one input. Generation metadata is never consulted by solve().
"""

from __future__ import annotations

from collections import Counter
import copy
import json

import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _count(value, name, low, high):
    _require(not isinstance(value, (bool, np.bool_))
             and isinstance(value, (int, np.integer)), name + " must be an integer")
    value = int(value)
    _require(low <= value <= high, name + " is out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid size")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid grid colors")
    return a


def _perimeter(h, w):
    return ([(0, c) for c in range(w)]
            + [(r, w - 1) for r in range(1, h)]
            + [(h - 1, c) for c in range(w - 2, -1, -1)]
            + [(r, 0) for r in range(h - 2, 0, -1)])


def _parse(input_grid):
    a = _grid(input_grid)
    h, w = a.shape
    _require(h >= 3 and w >= 3, "a frame and an interior are required")
    walls = frozenset(int(a[r, c]) for r, c in _perimeter(h, w))
    _require(len(walls) == 2, "the frame identifies exactly two wall colors")
    counts = Counter(int(x) for x in a.flat if int(x) not in walls)
    _require(len(counts) == 3, "background, starter and stops are required")
    ordered = sorted(counts, key=lambda c: counts[c])
    path, stop, bg = ordered
    _require(counts[path] == 1 and 2 <= counts[stop] < counts[bg],
             "roles must have unique singleton, repeated stops, dominant background")
    structure = np.isin(a, tuple(walls))
    rows = tuple(int(x) for x in np.flatnonzero(~np.any(structure[1:-1, 1:-1], axis=1)) + 1)
    cols = tuple(int(x) for x in np.flatnonzero(~np.any(structure[1:-1, 1:-1], axis=0)) + 1)
    _require(bool(rows) and bool(cols), "both street axes must exist")
    positions = np.argwhere((a == path) | (a == stop))
    _require(all(int(r) in rows and int(c) in cols for r, c in positions),
             "annotations must be at street crossings")
    start_physical = tuple(int(x) for x in np.argwhere(a == path)[0])
    start = (rows.index(start_physical[0]), cols.index(start_physical[1]))
    return a, bg, path, stop, walls, rows, cols, start


def _coordinate(index, streets, size):
    # Ghost street -1 is on the near frame, street n on the far frame.
    # Still farther ghosts lie one pixel apart outside the canvas.
    if index < 0:
        return index + 1
    if index >= len(streets):
        return size - 1 + index - len(streets)
    return streets[index]


def _walk(rows, cols, shape, start):
    """Yield visible pixels, segment, interval offset, and prior off-canvas flag.

    One logical interval is expanded into unit physical pixels. Border pixels
    are clipping, not collisions. 4*(max axis length+1) sides suffice: after
    that complete outer ring all subsequent sides lie outside both axes.
    """
    h, w = shape
    lr, lc = start
    pr, pc = rows[lr], cols[lc]
    outside = False
    for segment in range(1, 4 * (max(len(rows), len(cols)) + 1) + 1):
        dr, dc = _DIRS[(segment - 1) % 4]
        for offset in range(1, 2 * ((segment + 1) // 2) + 1):
            lr, lc = lr + dr, lc + dc
            nr = _coordinate(lr, rows, h)
            nc = _coordinate(lc, cols, w)
            while (pr, pc) != (nr, nc):
                pr, pc = pr + dr, pc + dc
                if 0 < pr < h - 1 and 0 < pc < w - 1:
                    yield pr, pc, segment, offset, outside
                else:
                    outside = True


def _trace(a, bg, path, stop, rows, cols, start, *, ignore_stops=False):
    out = a.copy()
    collision = None
    for index, (r, c, segment, offset, outside) in enumerate(
            _walk(rows, cols, a.shape, start)):
        if int(a[r, c]) == stop and not ignore_stops:
            collision = (index, r, c, segment, offset, outside)
            break
        if int(a[r, c]) == bg:
            out[r, c] = path
    return out, collision


def solve(input_grid: np.ndarray) -> np.ndarray:
    a, bg, path, stop, walls, rows, cols, start = _parse(input_grid)
    return _trace(a, bg, path, stop, rows, cols, start)[0]


def _axes(rng, n, stretched=False):
    """Select n streets with empty slots before, between, and after them."""
    for _ in range(200):
        gaps = rng.choice(3, size=n - 1, p=(0.52, 0.32, 0.16)).astype(int)
        # Every example has adjacent streets and a genuinely wider interval.
        zero, wider = (int(x) for x in rng.choice(n - 1, 2, replace=False))
        gaps[zero] = 0
        gaps[wider] = int(rng.integers(3, 5)) if stretched else int(rng.integers(1, 3))
        before, after = (int(x) for x in rng.integers(0, 4, size=2))
        size = n + 2 + before + after + int(gaps.sum())
        if size > 30 or size < 8:
            continue
        streets = [1 + before]
        for gap in gaps:
            streets.append(streets[-1] + 1 + int(gap))
        blocked = set(range(1, size - 1)) - set(streets)
        if len(blocked) >= 2:
            return size, tuple(streets)
    raise RuntimeError("could not sample street positions")


def _wall_map(rng, h, w, rows, cols, colors):
    bg, wall_a, wall_b, path, stop = colors
    a = np.full((h, w), bg, dtype=np.int8)
    border = _perimeter(h, w)
    # Alternate between uneven patches, with no behavioral color distinction.
    patches = int(rng.choice((2, 4, 6, 8)))
    sizes = rng.multinomial(len(border) - patches,
                            rng.dirichlet(np.full(patches, 0.8))) + 1
    sequence = [colors[1 + i % 2] for i, size in enumerate(sizes)
                for _ in range(int(size))]
    shift = int(rng.integers(len(border)))
    for i, (r, c) in enumerate(border):
        a[r, c] = sequence[(i + shift) % len(border)]
    br = sorted(set(range(1, h - 1)) - set(rows))
    bc = sorted(set(range(1, w - 1)) - set(cols))
    mask = np.zeros_like(a, dtype=bool)
    # A sparse cover first, then local growth and optional denser masses.
    # Covering both projections guarantees precisely the intended clear axes.
    for r in br:
        mask[r, int(rng.choice(bc))] = True
    for c in bc:
        if not np.any(mask[:, c]):
            mask[int(rng.choice(br)), c] = True
    density = float(rng.choice((0.14, 0.25, 0.42, 0.65)))
    for r in br:
        for c in bc:
            adjacent = any(0 <= r + dr < h and 0 <= c + dc < w
                           and mask[r + dr, c + dc] for dr, dc in _DIRS)
            if rng.random() < density * (1.35 if adjacent else 0.55):
                mask[r, c] = True
    coords = np.argwhere(mask)
    wall_values = rng.choice((wall_a, wall_b), size=len(coords))
    for (r, c), value in zip(coords, wall_values):
        a[r, c] = int(value)
    return a


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _uses_long_interval(out, path, rows, cols):
    """A novel interval must actually be crossed by the visible path prefix."""
    return (any(b - a >= 4 and any(bool(np.all(out[a + 1:b, c] == path)) for c in cols)
                for a, b in zip(rows, rows[1:]))
            or any(b - a >= 4 and any(bool(np.all(out[r, a + 1:b] == path)) for r in rows)
                   for a, b in zip(cols, cols[1:])))


def _features(pair):
    a, bg, path, stop, walls, rows, cols, start = _parse(pair["input"])
    events = list(_walk(rows, cols, a.shape, start))
    expected, hit = _trace(a, bg, path, stop, rows, cols, start)
    unblocked = _trace(a, bg, path, stop, rows, cols, start, ignore_stops=True)[0]
    # Explicit wrong rules used for concept-ablation and shortcut witnesses.
    full = a.copy()
    for r in rows:
        full[r, 1:-1][a[r, 1:-1] == bg] = path
    for c in cols:
        full[1:-1, c][a[1:-1, c] == bg] = path
    dense_rows, dense_cols = tuple(range(1, a.shape[0] - 1)), tuple(range(1, a.shape[1] - 1))
    dense_start = (rows[start[0]] - 1, cols[start[1]] - 1)
    pixels = _trace(a, bg, path, stop, dense_rows, dense_cols, dense_start)[0]
    edge_rows = tuple(r for r in range(1, a.shape[0] - 1)
                      if int(a[r, 1]) not in walls and int(a[r, -2]) not in walls)
    edge_cols = tuple(c for c in range(1, a.shape[1] - 1)
                      if int(a[1, c]) not in walls and int(a[-2, c]) not in walls)
    # True streets always survive this weaker endpoint test.
    edge_start = (edge_rows.index(rows[start[0]]), edge_cols.index(cols[start[1]]))
    edges = _trace(a, bg, path, stop, edge_rows, edge_cols, edge_start)[0]
    stop_positions = {(int(r), int(c)) for r, c in np.argwhere(a == stop)}
    route_positions = {(e[0], e[1]) for e in events}
    later_stops = [] if hit is None else [p for p in stop_positions
        if any((e[0], e[1]) == p for e in events[hit[0] + 1:])]
    reentry_before_hit = any(e[4] for e in (events if hit is None else events[:hit[0]]))
    first_six_inside = (start[0] >= 4 and start[0] + 2 < len(rows)
                        and start[1] >= 2 and start[1] + 4 < len(cols))
    return dict(a=a, bg=bg, path=path, stop=stop, walls=walls, rows=rows,
                cols=cols, start=start, events=events, expected=expected,
                hit=hit, first_six_inside=first_six_inside,
                reentry=reentry_before_hit, later_stops=len(later_stops),
                inert=len(stop_positions - route_positions),
                ignore_a_diff=not np.array_equal(expected, pixels),
                ignore_b_diff=not np.array_equal(expected, full),
                edge_diff=not np.array_equal(expected, edges),
                ignore_stop_diff=not np.array_equal(expected, unblocked))


def _make_pair(rng, colors, mode, stretch_axis=None):
    # A collision at the very next crossing can expose a novel northward
    # interval. Later modes can expose either axis in their longer prefix.
    if mode == "first" and stretch_axis is not None:
        stretch_axis = 0
    for attempt in range(180):
        if mode == "anchor":
            nr, nc = (int(x) for x in rng.integers(7, 12, size=2))
        elif mode == "core_stop":
            nr, nc = (int(x) for x in rng.integers(7, 13, size=2))
        else:
            nr, nc = (int(x) for x in rng.integers(3, 13, size=2))
        h, rows = _axes(rng, nr, stretch_axis == 0)
        w, cols = _axes(rng, nc, stretch_axis == 1)
        if mode == "anchor":
            start = (int(rng.integers(4, nr - 2)), int(rng.integers(2, nc - 4)))
        else:
            start = (int(rng.integers(nr)), int(rng.integers(nc)))
            if mode in ("core_stop", "reentry") or rng.random() < 0.35:
                side = int(rng.integers(4))
                start = ((0, start[1]) if side == 0 else
                         (nr - 1, start[1]) if side == 1 else
                         (start[0], 0) if side == 2 else (start[0], nc - 1))
            if mode == "first" and stretch_axis is not None:
                long_gap = next(i for i in range(nr - 1) if rows[i + 1] - rows[i] >= 4)
                start = (long_gap + 1, start[1])
        a = _wall_map(rng, h, w, rows, cols, colors)
        bg, wall_a, wall_b, path, stop = colors
        origin = (rows[start[0]], cols[start[1]])
        a[origin] = path
        events = list(_walk(rows, cols, a.shape, start))
        crossing_indices = [i for i, e in enumerate(events)
                            if e[0] in rows and e[1] in cols]
        route_positions = {(e[0], e[1]) for e in events}
        crossings = {(r, c) for r in rows for c in cols} - {origin}
        inert = sorted(crossings - route_positions)
        target = None
        if mode not in ("anchor", "open"):
            choices = []
            for i in crossing_indices:
                r, c, seg, offset, outside = events[i]
                if mode == "core_stop":
                    good = (5 <= seg <= 14 and outside and i >= 12
                            and any(e[4] for e in events[:i])
                            and any(j > i for j in crossing_indices)
                            and len(events) - i >= 8)
                elif mode == "short":
                    good = seg <= 3
                elif mode == "mid":
                    good = 4 <= seg <= 7
                elif mode == "late":
                    good = seg >= 8 and len(events) - i >= 3
                elif mode == "reentry":
                    good = outside and i >= 2 and len(events) - i >= 3
                elif mode == "end":
                    good = offset == 2 * ((seg + 1) // 2)
                elif mode == "first":
                    good = i == crossing_indices[0]
                else:
                    raise ValueError("unknown mode")
                if good:
                    choices.append(i)
            if not choices:
                continue
            target = int(rng.choice(choices))
        selected = []
        if target is not None:
            selected.append(events[target][:2])
            later = [events[i][:2] for i in crossing_indices if i > target]
            if later:
                selected.append(later[int(rng.integers(len(later)))])
            safe = sorted(crossings - {(e[0], e[1]) for e in events[:target]})
        else:
            safe = inert
        desired = int(rng.integers(2, min(12, len(safe)) + 1)) if len(safe) >= 2 else 0
        if not desired:
            continue
        # At least one inert annotation also appears in the stopped witness.
        if mode == "core_stop" and inert:
            selected.append(inert[int(rng.integers(len(inert)))])
        selected = list(dict.fromkeys(selected))
        remainder = [p for p in safe if p not in selected]
        need = min(len(remainder), max(0, desired - len(selected)))
        if need:
            selected.extend(remainder[int(i)] for i in rng.choice(len(remainder), need, replace=False))
        for p in selected:
            a[p] = stop
        pair = {"input": a, "output": solve(a)}
        if stretch_axis is not None and not _uses_long_interval(pair["output"], path, rows, cols):
            continue
        if mode in ("anchor", "core_stop"):
            f = _features(pair)
            if not (f["ignore_a_diff"] and f["ignore_b_diff"] and f["edge_diff"]):
                continue
            if mode == "anchor" and (f["hit"] is not None or not f["first_six_inside"]):
                continue
            if mode == "core_stop" and not (f["hit"] is not None and f["reentry"]
                                              and f["later_stops"] and f["ignore_stop_diff"]):
                continue
        return pair
    raise RuntimeError("could not build " + mode)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)),
             "seed must be an integer")
    rng = np.random.default_rng(seed)
    colors = tuple(int(x) for x in rng.choice(10, size=5, replace=False))
    seen = set()

    def unique(mode, stretched=None):
        for _ in range(40):
            pair = _make_pair(rng, colors, mode, stretched)
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                return pair
        raise RuntimeError("could not produce distinct inputs")

    train = [unique("anchor"), unique("core_stop")]
    modes = ("open", "mid", "late", "short", "end", "reentry", "first")
    offset = int(rng.integers(len(modes)))
    for i in range(num_train - 2):
        train.append(unique(modes[(offset + i) % len(modes)]))
    rng.shuffle(train)
    test_modes = list(rng.permutation(modes))
    test = [unique(str(test_modes[i]), int(rng.integers(2))) for i in range(num_test)]
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _count(nt, "training count", 2, 10)
    _count(ns, "test count", 1, 4)
    _require(nt > ns, "training count must exceed test count")
    seen, palettes, info = set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == out.shape, "shape must be preserved")
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            f = _features(pair)
            palettes.add((f["bg"], f["walls"], f["path"], f["stop"]))
            _require(np.array_equal(out, f["expected"]), "wrong stretched spiral")
            changed = a != out
            _require(bool(np.all(a[changed] == f["bg"])) and bool(np.all(out[changed] == f["path"])),
                     "only background can become path")
            structure = np.isin(a, tuple(f["walls"]))
            _require(np.array_equal(a[structure], out[structure]), "walls changed")
            annotation = (a == f["path"]) | (a == f["stop"])
            _require(np.array_equal(a[annotation], out[annotation]), "annotation changed")
            gaps = [b - a for axis in (f["rows"], f["cols"]) for a, b in zip(axis, axis[1:])]
            _require(1 in gaps and any(g > 1 for g in gaps), "need adjacent and spaced streets")
            if split == "train":
                _require(max(gaps) <= 3, "training interval exceeds its sampling regime")
            else:
                _require(max(gaps) in (4, 5), "test must contain a novel stretched interval")
                _require(_uses_long_interval(out, f["path"], f["rows"], f["cols"]),
                         "test must draw across its novel interval")
            info[split].append(f)
    _require(len(palettes) == 1, "palette roles must agree within a puzzle")
    _require(any(f["hit"] is None and f["first_six_inside"] and f["inert"] >= 2
                 and f["ignore_a_diff"] and f["ignore_b_diff"] and f["edge_diff"]
                 for f in info["train"]), "missing open six-segment ordinal witness")
    _require(any(f["hit"] is not None and f["hit"][3] >= 5 and f["reentry"]
                 and f["later_stops"] and f["ignore_stop_diff"] and f["inert"] >= 1
                 and f["ignore_a_diff"] and f["ignore_b_diff"] and f["edge_diff"]
                 for f in info["train"]), "missing stopped re-entry witness")


def _same(p, q):
    return all(len(p[s]) == len(q[s]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(p[s], q[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test():
    """Bounded reproducibility, semantic invariants and 42-seed diversity audit."""
    # Hand-calculated physical strokes: N to (5,4), E to (5,8), S to
    # (12,8), W toward a stop at (12,4). The last bridge ends at column 5.
    # This expectation is explicit; it does not use the tracer to label itself.
    fixture = _wall_map(np.random.default_rng(12), 15, 16,
                        (2, 3, 5, 6, 9, 10, 12), (1, 3, 4, 7, 8, 11, 13),
                        (0, 1, 8, 3, 2))
    fixture[9, 4] = 3
    fixture[12, 4] = fixture[2, 13] = 2
    expected = fixture.copy()
    expected[5:9, 4] = 3
    expected[5, 5:9] = 3
    expected[6:13, 8] = 3
    expected[12, 5:8] = 3
    _require(np.array_equal(solve(fixture), expected), "manual stretched-stroke regression failed")
    shapes, palettes, layouts, starts, signatures, test_segments = (set() for _ in range(6))
    totals = Counter()
    for seed in range(42):
        p = generate_puzzle(seed)
        _require(_same(p, generate_puzzle(seed)), "nondeterminism")
        signatures.add(tuple(_key(x["input"]) for s in ("train", "test") for x in p[s]))
        for split in ("train", "test"):
            for pair in p[split]:
                f = _features(pair)
                a, rows, cols = f["a"], f["rows"], f["cols"]
                shapes.add(a.shape)
                palettes.add((f["bg"], f["walls"], f["path"], f["stop"]))
                layouts.add((a.shape, rows, cols))
                starts.add((len(rows), len(cols), f["start"]))
                _require(np.array_equal(solve(a.tolist()), pair["output"]), "list input solve failed")
                totals["edge_diff"] += f["edge_diff"]
                totals["wide"] += a.shape[1] >= 1.6 * a.shape[0]
                totals["tall"] += a.shape[0] >= 1.6 * a.shape[1]
                totals["thirty"] += 30 in a.shape
                totals["many_stops"] += np.count_nonzero(a == f["stop"]) >= 8
                if split == "test":
                    totals["open"] += f["hit"] is None
                    totals["reentry"] += f["reentry"]
                    totals["stopped"] += f["hit"] is not None
                    if f["hit"] is not None:
                        test_segments.add(f["hit"][3])
                        totals["early"] += f["hit"][3] <= 3
                        totals["late"] += f["hit"][3] >= 8
    _require(len(signatures) == 42 and len(palettes) >= 30, "palette or seed diversity missing")
    _require(len(shapes) >= 55 and len(layouts) >= 180 and len(starts) >= 100,
             "geometry diversity missing")
    _require(len(test_segments) >= 7 and all(totals[k] > 0 for k in
             ("open", "stopped", "reentry", "early", "late", "wide", "tall", "thirty", "many_stops")),
             "regime diversity missing")
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1900 + nt * 10 + ns, nt, ns)
            _require((len(p["train"]), len(p["test"])) == (nt, ns), "count mismatch")
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    bad = copy.deepcopy(generate_puzzle(7, 2, 1))
    pair = next(x for x in bad["train"] if np.any(x["input"] != x["output"]))
    r, c = np.argwhere(pair["input"] != pair["output"])[0]
    pair["output"][r, c] = pair["input"][r, c]
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
    return {"seeds": 42, "shapes": len(shapes), "palettes": len(palettes),
            "layouts": len(layouts), "origins": len(starts),
            "test_segments": sorted(test_segments),
            "features": {k: int(v) for k, v in totals.items()}}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
