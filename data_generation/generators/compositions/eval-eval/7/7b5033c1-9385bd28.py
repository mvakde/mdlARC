"""Route-operated sheets: trace, toggle, and composite the surviving stack.

No literal color has a special role. The unique long foreground chain is the
program; the isolated two-column table binds its cue colors to opaque paints.
Every program CELL toggles its sheet. An on event raises it, an off event removes
it. The output composites only the final stack over the original input.
"""

from collections import Counter
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with axes 1..30 and colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a)
    return (a.shape, a.tobytes())


def _inside(p, b):
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


def _overlap(a, b):
    return max(a[0], b[0]) <= min(a[2], b[2]) and max(a[1], b[1]) <= min(a[3], b[3])


def _contains(a, b):
    return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]


def _halo(cells):
    return set(cells) | {(r + dr, c + dc) for r, c in cells for dr, dc in DIRS}


def _markers(b, diagonal):
    r, c, rr, cc = b
    if diagonal == 0:
        return {(r, c), (r, c + 1), (r + 1, c),
                (rr, cc), (rr, cc - 1), (rr - 1, cc)}
    return {(r, cc), (r, cc - 1), (r + 1, cc),
            (rr, c), (rr, c + 1), (rr - 1, c)}


def _recognize(cells):
    if len(cells) != 6:
        raise ValueError("each sheet needs exactly two separated L triplets")
    rs, cs = zip(*cells)
    b = (min(rs), min(cs), max(rs), max(cs))
    h, w = b[2] - b[0] + 1, b[3] - b[1] + 1
    if min(h, w) < 2 or (min(h, w) == 2 and max(h, w) < 4):
        raise ValueError("invalid sheet dimensions")
    for d in (0, 1):
        if cells == _markers(b, d):
            return b, d
    raise ValueError("invalid opposite-corner pattern")


def _components(cells):
    todo = set(cells)
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        group, frontier = {start}, [start]
        while frontier:
            r, c = frontier.pop()
            for dr, dc in DIRS:
                q = (r + dr, c + dc)
                if q in todo:
                    todo.remove(q)
                    group.add(q)
                    frontier.append(q)
        result.append(group)
    return result


def _trace(cells):
    adj = {p: [(p[0] + dr, p[1] + dc) for dr, dc in DIRS
               if (p[0] + dr, p[1] + dc) in cells] for p in cells}
    ends = [p for p in cells if len(adj[p]) == 1]
    if len(ends) != 2 or any(len(v) not in (1, 2) for v in adj.values()):
        raise ValueError("program must be an induced simple orthogonal chain")
    if ends[0][0] == ends[1][0]:
        raise ValueError("program endpoints must have different rows")
    p, previous, path = min(ends, key=lambda q: q[0]), None, []
    while True:
        path.append(p)
        next_cells = [q for q in adj[p] if q != previous]
        if not next_cells:
            break
        previous, p = p, next_cells[0]
        if len(path) > len(cells):
            raise ValueError("cycle in program")
    if len(path) != len(cells):
        raise ValueError("disconnected program")
    return path


def _parse(value):
    g = _grid(value)
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] <= g.size // 2:
        raise ValueError("background must occupy a strict majority")
    fg = {tuple(map(int, p)) for p in np.argwhere(g != bg)}
    groups = _components(fg)
    long = [s for s in groups if len(s) >= 9]
    if len(long) != 1 or len(long[0]) > 48:
        raise ValueError("expected one program of 9..48 cells")
    path = _trace(long[0])
    scene = g.copy()
    for p in path:
        scene[p] = bg
    h, w = g.shape
    candidates = []
    for c in range(w - 1):
        rows = np.flatnonzero(scene[:, c] != bg)
        runs = np.split(rows, np.flatnonzero(np.diff(rows) != 1) + 1)
        for run in runs:
            if not 2 <= len(run) <= 4:
                continue
            r, rr = int(run[0]), int(run[-1])
            cues = list(map(int, scene[r:rr + 1, c]))
            if len(set(cues)) != len(cues):
                continue
            table = (r, c, rr, c + 1)
            # The moat is required in the ORIGINAL input, including the route.
            if any(g[y, x] != bg for y in range(max(0, r - 1), min(h, rr + 2))
                   for x in range(max(0, c - 1), min(w, c + 3))
                   if not _inside((y, x), table)):
                continue
            rest = scene.copy()
            rest[r:rr + 1, c:c + 2] = bg
            objects = {}
            try:
                for cue in set(map(int, np.unique(rest))) - {bg}:
                    cells = {tuple(map(int, p)) for p in np.argwhere(rest == cue)}
                    bounds, diagonal = _recognize(cells)
                    if _overlap(bounds, table):
                        raise ValueError("table intersects a sheet")
                    objects[cue] = dict(cue=cue, bounds=bounds, markers=cells,
                                        diagonal=diagonal)
            except ValueError:
                continue
            if set(objects) != set(cues) or not set(map(int, g[tuple(np.array(path).T)])) <= set(cues):
                continue
            fills = list(map(int, g[r:rr + 1, c + 1]))
            for cue, fill in zip(cues, fills):
                objects[cue]["fill"] = fill
            # Each fragment is a separate component; different fragments never
            # touch, including different sheets. This also guarantees parsing.
            marker_groups = _components({p for o in objects.values() for p in o["markers"]})
            if len(marker_groups) != 2 * len(cues) or any(len(s) != 3 for s in marker_groups):
                continue
            candidates.append(dict(background=bg, path=path, table=table,
                                   cues=cues, objects=objects,
                                   sequence=[int(g[p]) for p in path]))
    if len(candidates) != 1:
        raise ValueError("lookup table / scene is not uniquely readable")
    return candidates[0]


def _execute(sequence):
    """Stack is bottom to top. Removing a sheet does not erase older sheets."""
    stack = []
    for cue in sequence:
        if cue in stack:
            stack.remove(cue)
        else:
            stack.append(cue)
    return stack


def _render(g, objects, stack):
    out = np.asarray(g).copy()
    for cue in stack:
        o = objects[cue]
        r, c, rr, cc = o["bounds"]
        out[r:rr + 1, c:cc + 1] = o["fill"]
        for p in o["markers"]:
            out[p] = cue
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    g = _grid(input_grid)
    p = _parse(g)
    return _render(g, p["objects"], _execute(p["sequence"]))


def _witnesses(g, p):
    """Ablations are evaluated on visible arrays, never construction flags."""
    seq, objects = p["sequence"], p["objects"]
    stack = _execute(seq)
    out = _render(g, objects, stack)
    counts = Counter(seq)
    first = list(dict.fromkeys(seq))
    all_last = sorted(objects, key=lambda c: max((i for i, x in enumerate(seq) if x == c), default=-1))
    alternatives = {
        "toggle": all_last,
        "cells": _execute([c for i, c in enumerate(seq) if i == 0 or c != seq[i - 1]]),
        "reactivation": [c for c in first if c in stack],
        "direction": _execute(seq[::-1]),
        "table_order": [c for c in reversed(p["cues"]) if c in stack],
        "raster": _execute([int(g[q]) for q in sorted(p["path"])]),
        "last_only": stack[-1:],
    }
    found = {name for name, order in alternatives.items()
             if not np.array_equal(out, _render(g, objects, order))}
    cue_paints = {c: dict(o, fill=c) for c, o in objects.items()}
    if not np.array_equal(out, _render(g, cue_paints, stack)):
        found.add("lookup")
    if not np.array_equal(out, _render(g, objects, [c for c in stack if objects[c]["fill"] != p["background"]])):
        found.add("opaque")
    if not np.array_equal(out, g):
        found.add("completion")
    if np.any((g == p["background"]) & (out != p["background"])):
        found.add("area")
    if any(counts[c] >= 3 and counts[c] % 2 for c in stack):
        found.add("odd_repeat")
    if any(counts[c] > 0 and counts[c] % 2 == 0 for c in objects):
        found.add("even_repeat")
    if any(out[q] != g[q] for q in p["path"]):
        found.add("route_occlusion")
    if min(r for r, c in p["path"]) < p["path"][0][0]:
        found.add("above")
    for cue, o in objects.items():
        if any(out[q] != cue for q in o["markers"]):
            found.add("marker_occlusion")
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            layers = [objects[x] for x in stack if _inside((r, c), objects[x]["bounds"])]
            if not layers:
                continue
            top = layers[-1]
            if (r, c) in top["markers"] and top["cue"] != top["fill"]:
                found.add("retention")
            if len(layers) >= 2 and (r, c) not in top["markers"]:
                lower = layers[-2]
                below = lower["cue"] if (r, c) in lower["markers"] else lower["fill"]
                if top["fill"] == p["background"] and below != p["background"]:
                    found.add("clearing")
    return found


def _bounds(rng, h, w):
    style = int(rng.integers(5))
    if style == 0:
        hh, ww = int(rng.integers(2, 5)), int(rng.integers(4, w + 1))
    elif style == 1:
        hh, ww = int(rng.integers(4, h + 1)), int(rng.integers(2, 5))
    elif style == 2:
        hh = ww = int(rng.integers(3, min(h, w, 12) + 1))
    elif style == 3:
        hh, ww = int(rng.integers(3, min(h, 9) + 1)), int(rng.integers(3, min(w, 9) + 1))
    else:
        hh, ww = int(rng.integers(3, h + 1)), int(rng.integers(3, w + 1))
    r, c = int(rng.integers(h - hh + 1)), int(rng.integers(w - ww + 1))
    return (r, c, r + hh - 1, c + ww - 1)


def _geometry(rng, h, w, k, table, relation):
    table_moat = {(r, c) for r in range(table[0] - 1, table[2] + 2)
                  for c in range(table[1] - 1, table[3] + 2)}
    objects, occupied = [], set()
    for i in range(k):
        for _ in range(90):
            b = _bounds(rng, h, w)
            if i and relation == "nested":
                parent = objects[0]["bounds"]
                ph, pw = parent[2] - parent[0] - 1, parent[3] - parent[1] - 1
                if ph < 4 or pw < 4:
                    return None
                bb = _bounds(rng, ph, pw)
                b = (bb[0] + parent[0] + 1, bb[1] + parent[1] + 1,
                     bb[2] + parent[0] + 1, bb[3] + parent[1] + 1)
            if _overlap(b, table):
                continue
            if i:
                hits = [_overlap(b, o["bounds"]) for o in objects]
                if relation == "partial" and (not hits[0] or _contains(b, objects[0]["bounds"])
                                               or _contains(objects[0]["bounds"], b)):
                    continue
                if relation == "chain" and (not hits[-1] or any(hits[:-1])):
                    continue
                if relation == "disjoint" and any(hits):
                    continue
            d = int(rng.integers(2))
            cells = _markers(b, d)
            if cells & (occupied | table_moat):
                continue
            if len(_components(cells)) != 2:
                continue
            objects.append(dict(bounds=b, markers=cells, diagonal=d))
            occupied.update(_halo(cells))
            break
        else:
            return None
    return objects, occupied | table_moat


def _walk(rng, h, w, length, forbidden, persistence, allow_above, need_above, folded):
    free = [(r, c) for r in range(h) for c in range(w) if (r, c) not in forbidden]
    for _ in range(100):
        path = [free[int(rng.integers(len(free)))]]
        used, previous = set(path), None
        for _ in range(length - 1):
            choices, weights = [], []
            for dr, dc in DIRS:
                q = (path[-1][0] + dr, path[-1][1] + dc)
                if not (0 <= q[0] < h and 0 <= q[1] < w) or q in forbidden or q in used:
                    continue
                if sum((q[0] + a, q[1] + b) in used for a, b in DIRS) != 1:
                    continue
                choices.append((q, (dr, dc)))
                weights.append(persistence if previous == (dr, dc) else 1.0)
            if not choices:
                break
            weights = np.asarray(weights, dtype=float)
            q, previous = choices[int(rng.choice(len(choices), p=weights / weights.sum()))]
            path.append(q)
            used.add(q)
        if len(path) != length or path[0][0] == path[-1][0]:
            continue
        if path[0][0] > path[-1][0]:
            path.reverse()
        above = min(r for r, c in path) < path[0][0]
        steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
        internal_up = any(a == (-1, 0) and b != a for a, b in zip(steps, steps[1:]))
        if (above and not allow_above) or (need_above and not above):
            continue
        if folded and (not internal_up or not ((0, 1) in steps and (0, -1) in steps)):
            continue
        return path
    return None


def _sequence(rng, cues, live_count, length):
    # Every cue is visited. Counts are positive even for off sheets, positive
    # odd for on sheets. Extra pairs vary run lengths and recurrence locations.
    if length % 2 != live_count % 2:
        raise ValueError("incompatible parity")
    order = list(map(int, rng.permutation(cues)))
    counts = [1 if i < live_count else 2 for i in range(len(cues))]
    counts[0] += 2  # At least one on/off/on witness.
    slack = (length - sum(counts)) // 2
    counts = np.asarray(counts) + 2 * rng.multinomial(slack, np.full(len(cues), 1 / len(cues)))
    seq = [c for c, n in zip(order, counts) for _ in range(int(n))]
    rng.shuffle(seq)
    return seq


def _label(rng, bg, table, geometry, path, live_count, required, h, w):
    k = len(geometry)
    cues = list(map(int, rng.permutation([c for c in range(10) if c != bg])[:k]))
    table_order = list(map(int, rng.permutation(cues)))
    objects = {cue: dict(o, cue=cue, fill=int(rng.integers(10)))
               for cue, o in zip(cues, geometry)}
    for _ in range(180):
        seq = _sequence(rng, cues, live_count, len(path))
        stack = _execute(seq)
        # A background sheet is visibly opaque in the cancellation lesson.
        for cue in cues:
            objects[cue]["fill"] = int(rng.integers(10))
        if "clearing" in required:
            objects[stack[-1]]["fill"] = bg
            for cue in stack[:-1]:
                objects[cue]["fill"] = int(rng.choice([c for c in range(10) if c != bg]))
        g = np.full((h, w), bg, dtype=np.int8)
        for cue, o in objects.items():
            for q in o["markers"]:
                g[q] = cue
        for q, cue in zip(path, seq):
            g[q] = cue
        for i, cue in enumerate(table_order):
            g[table[0] + i, table[1]:table[1] + 2] = cue, objects[cue]["fill"]
        p = dict(background=bg, path=path, sequence=seq, table=table,
                 cues=table_order, objects=objects)
        if not required <= _witnesses(g, p):
            continue
        try:
            parsed = _parse(g)
        except ValueError:
            continue
        out = _render(g, objects, stack)
        if not np.array_equal(out, _render(g, parsed["objects"], _execute(parsed["sequence"]))):
            continue
        return {"input": g, "output": out}
    return None


def _fallback(rng, params, split, lesson, required):
    """A bounded safety net with overlapping sheets and an interior route.

    The sampler normally uses free geometry. This construction changes palettes,
    path prefixes, table row permutations and color sequences on every call.
    """
    k = 4 if split == "test" and params["shift"] == "layers" else 3
    live_count = 2 if lesson == "cancel" else (3 if k == 3 else int(rng.integers(3, 5)))
    boxes = [(6, 6, 17, 18), (10, 3, 21, 14), (3, 10, 11, 23), (19, 20, 23, 25)]
    geometry = [dict(bounds=b, markers=_markers(b, d), diagonal=d)
                for b, d in zip(boxes[:k], (0, 1, 1, 0))]
    # Two top-row variants give either parity while preserving the final
    # upward-then-right turn. Marker diagonals put lower/off Ls inside a cover.
    full_parity = 1 if params["above"] else 0
    top = 7 if full_parity == live_count % 2 else 8
    if params["above"]:
        path, segments = [(top + 3, 8)], [((-1, 0), 3), ((0, 1), 8), ((1, 0), 14 - top),
                                          ((0, -1), 7), ((1, 0), 3), ((0, 1), 6),
                                          ((-1, 0), 1), ((0, 1), 1)]
    else:
        path, segments = [(top, 8)], [((0, 1), 8), ((1, 0), 14 - top), ((0, -1), 7),
                                     ((1, 0), 3), ((0, 1), 6), ((-1, 0), 1), ((0, 1), 1)]
    for (dr, dc), n in segments:
        for _ in range(n):
            path.append((path[-1][0] + dr, path[-1][1] + dc))
    if params["above"] and lesson == "order":
        # A leftward finish makes raster reading visibly different near the
        # END of the program, instead of only rearranging an early prefix.
        path = [(8, 9)]
        for (dr, dc), n in [((-1, 0), 3), ((0, 1), 7), ((1, 0), 8), ((0, -1), 4)]:
            for _ in range(n):
                path.append((path[-1][0] + dr, path[-1][1] + dc))
    length = (30 if live_count % 2 == 0 else 29) if split == "test" else (18 if live_count % 2 == 0 else 19)
    if params["above"] and lesson == "order":
        length = 23
    # A folding-shift test includes the final upward turn.
    if split == "test" and params["shift"] == "fold":
        length = len(path)
    path = path[:length]
    bg = int(rng.integers(10))
    pair = _label(rng, bg, (1, 1, k, 2), geometry, path, live_count, required, 26, 28)
    if pair is None:
        raise RuntimeError("bounded fallback could not construct teaching evidence")
    return pair


def _requirements(params, split, lesson):
    required = {"completion", "retention"}
    if lesson == "cancel":
        required |= {"toggle", "cells", "lookup", "opaque", "last_only", "area", "reactivation", "table_order", "clearing", "even_repeat",
                     "odd_repeat", "route_occlusion", "marker_occlusion"}
    elif lesson == "order":
        required |= {"direction", "raster", "table_order", "route_occlusion", "last_only", "area"}
    elif split == "test":
        required |= {"table_order", "route_occlusion", "last_only"}
    if params["above"] and lesson in ("cancel", "order"):
        required.add("above")
    return required


def _make_pair(rng, params, split, lesson):
    required = _requirements(params, split, lesson)
    for _ in range(75):
        k = (int(rng.integers(3, 5)) if lesson == "cancel" else
             4 if split == "test" and params["shift"] == "layers" else
             3 if lesson == "order" else int(rng.integers(2, 5)))
        if split == "train" and params["shift"] == "layers":
            k = min(k, 3)
        live_count = k - 1 if lesson == "cancel" else k if lesson == "order" else int(rng.integers(2, k + 1))
        low, high = (12, 23) if split == "train" and params["shift"] == "length" else (24, 41) if split == "test" and params["shift"] == "length" else (12, 35)
        length = int(rng.integers(low, high))
        length += (live_count - length) % 2
        h, w = int(rng.integers(14, 31)), int(rng.integers(14, 31))
        tr, tc = int(rng.integers(h - k + 1)), int(rng.integers(w - 1))
        if rng.random() < .55:
            if rng.random() < .5:
                tc = int(rng.choice((0, w - 2)))
            else:
                tr = int(rng.choice((0, h - k)))
        table = (tr, tc, tr + k - 1, tc + 1)
        relation = str(rng.choice(("free", "partial", "nested", "chain", "disjoint")))
        if required & {"table_order", "clearing"} and relation == "disjoint":
            relation = "partial"
        result = _geometry(rng, h, w, k, table, relation)
        if result is None:
            continue
        geometry, forbidden = result
        persistence = float(rng.uniform(.4, 7))
        if params["shift"] == "fold":
            persistence = .7 if split == "test" else float(rng.uniform(4, 12))
        path = _walk(rng, h, w, length, forbidden, persistence, params["above"],
                     "above" in required, split == "test" and params["shift"] == "fold")
        if path is None:
            continue
        if "route_occlusion" in required and not any(_inside(q, o["bounds"]) for q in path for o in geometry):
            continue
        pair = _label(rng, int(rng.integers(10)), table, geometry, path, live_count, required, h, w)
        if pair is not None:
            return pair
    return _fallback(rng, params, split, lesson, required)


def _counts(num_train, num_test):
    for n, low, high, name in ((num_train, 2, 10, "num_train"), (num_test, 1, 4, "num_test")):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not low <= int(n) <= high:
            raise ValueError(f"{name} must be an integer from {low} through {high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    params = dict(shift=str(rng.choice(("length", "fold", "layers"))), above=bool(rng.integers(2)))
    lessons = ["cancel", "order"] + ["free"] * (num_train - 2)
    rng.shuffle(lessons)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            lesson = lessons[i] if split == "train" else "free"
            for _ in range(20):
                pair = _make_pair(rng, params, split, lesson)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded uniqueness construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, evidence, test_evidence = set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g, out = _grid(pair["input"]), _grid(pair["output"])
            assert g.shape == out.shape and _key(g) not in seen
            seen.add(_key(g))
            p = _parse(g)
            stack = _execute(p["sequence"])
            # Toggle semantics: odd counts exactly equal the surviving set.
            counts = Counter(p["sequence"])
            assert set(stack) == {c for c in p["cues"] if counts[c] % 2}
            assert stack == sorted(stack, key=lambda c: max(i for i, x in enumerate(p["sequence"]) if x == c))
            assert np.array_equal(out, _render(g, p["objects"], stack))
            assert np.array_equal(solve(g.copy()), out)
            tr, tc, rr, cc = p["table"]
            assert np.array_equal(out[tr:rr + 1, tc:cc + 1], g[tr:rr + 1, tc:cc + 1])
            covered = {q for o in (p["objects"][c] for c in stack)
                       for q in ((r, c) for r in range(o["bounds"][0], o["bounds"][2] + 1)
                                 for c in range(o["bounds"][1], o["bounds"][3] + 1))}
            assert all(out[r, c] == g[r, c] for r in range(g.shape[0]) for c in range(g.shape[1])
                       if (r, c) not in covered)
            witnesses = _witnesses(g, p)
            assert {"completion", "retention"} <= witnesses
            (evidence if split == "train" else test_evidence).update(witnesses)
    assert {"toggle", "cells", "lookup", "opaque", "last_only", "area", "reactivation", "direction", "raster", "table_order", "clearing",
            "odd_repeat", "even_repeat", "route_occlusion", "marker_occlusion"} <= evidence
    assert {"table_order", "route_occlusion", "last_only"} <= test_evidence
    if "above" in test_evidence:
        assert any({"above", "direction"} <= _witnesses(_grid(pair["input"]), _parse(pair["input"]))
                   for pair in puzzle["train"])


def self_test():
    """Bounded 42-seed semantic and structural variation audit."""
    fallback_cases = 0
    for shift in ("length", "fold", "layers"):
        for above in (False, True):
            params = dict(shift=shift, above=above)
            for split, lesson in (("train", "cancel"), ("train", "order"), ("test", "free")):
                required = _requirements(params, split, lesson)
                pair = _fallback(np.random.default_rng(100 + fallback_cases), params, split, lesson, required)
                p = _parse(pair["input"])
                assert required <= _witnesses(pair["input"], p)
                assert np.array_equal(solve(pair["input"]), pair["output"])
                if split == "test" and shift == "fold":
                    steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(p["path"], p["path"][1:])]
                    assert (0, 1) in steps and (0, -1) in steps
                    assert any(a == (-1, 0) and b != a for a, b in zip(steps, steps[1:]))
                if split == "train":
                    # Rule support beyond the demonstration sampler: unvisited
                    # cues and zero/one surviving sheet, without latent state.
                    single = pair["input"].copy()
                    cue = p["cues"][0]
                    for q in p["path"]:
                        single[q] = cue
                    expected = _render(single, p["objects"], [cue] if len(p["path"]) % 2 else [])
                    assert np.array_equal(solve(single), expected)
                fallback_cases += 1
    shapes, layouts, paths, palettes, lengths, live_counts, turns = set(), set(), set(), set(), set(), set(), set()
    signature, backgrounds, table_positions, diagonals = set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                g = pair["input"]
                p = _parse(g)
                shapes.add(g.shape)
                backgrounds.add(p["background"])
                table_positions.add(p["table"][:2])
                palettes.add(tuple(zip(p["cues"], [p["objects"][c]["fill"] for c in p["cues"]])))
                lengths.add(len(p["path"]))
                live_counts.add(len(_execute(p["sequence"])))
                origin = p["path"][0]
                paths.add(tuple((r - origin[0], c - origin[1]) for r, c in p["path"]))
                layouts.add(tuple(sorted(o["bounds"] for o in p["objects"].values())))
                diagonals.update(o["diagonal"] for o in p["objects"].values())
                steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(p["path"], p["path"][1:])]
                turns.add(sum(a != b for a, b in zip(steps, steps[1:])))
                # Table permutation is semantics-preserving, unlike path order.
                permuted = g.copy()
                r, c, rr, cc = p["table"]
                permuted[r:rr + 1, c:cc + 1] = permuted[r:rr + 1, c:cc + 1][::-1].copy()
                expected = pair["output"].copy()
                expected[r:rr + 1, c:cc + 1] = permuted[r:rr + 1, c:cc + 1]
                assert np.array_equal(solve(permuted), expected)
    assert len(signature) == 42
    assert len(shapes) >= 50 and len(layouts) >= 100 and len(paths) >= 120
    assert len(palettes) >= 150 and backgrounds == set(range(10))
    assert len(table_positions) >= 70 and diagonals == {0, 1}
    assert len(lengths) >= 20 and {2, 3, 4} <= live_counts and len(turns) >= 12
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(8128 + a * 10 + b, a, b))
    a, b = generate_puzzle(17, 2, 1), generate_puzzle(17, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in a[s]] for s in a}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return dict(seeds=42, distinct_puzzles=len(signature), shapes=len(shapes),
                sheet_layouts=len(layouts), program_geometries=len(paths),
                mappings=len(palettes), lengths=len(lengths), turn_counts=len(turns),
                fallback_cases=fallback_cases)


if __name__ == "__main__":
    print(self_test())
