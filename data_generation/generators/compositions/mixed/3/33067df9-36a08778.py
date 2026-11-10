"""Span gates: sparse rectangle reconstruction controls horizontal ray branching.

Only NumPy and the standard library are used. No files, examples, generation
metadata or random state are consulted by solve(). See task_spec.md for the rule.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


HOST_SEEDS = (0,114594124,837932326,281256049,797885408,367096197,880834480,
              888124346,93498226,359194807,810662356,545870816,211698713,
              580317451,934157187,245581114,645299402,19981525,53664887,
              243700773,541168872,859463030,589209222,529004138,695635737,
              68186735,798331210,240385843,504674160,806805928,823938718,
              97004320,714919790,37882651,951813961,580997867,24073583,
              853401009,955544547,474973579,989250378,436957672)
SHORTCUTS = ("no_merge", "overlap_merge", "no_rays", "straight", "outline_all",
             "one_per_layer", "ignore_color", "cellwise")


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}: {value}")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("grid must be an integer matrix of colors 0..9, size 1..30")
    return a.astype(np.int8, copy=True)


def _decode(value):
    a = _grid(value)
    h, w = a.shape
    if h % 2 != 1 or w % 2 != 1 or not (3 <= h <= 15 and 3 <= w <= 15):
        raise ValueError("expected a sparse lattice with 1..7 coarse rows/columns")
    bg = int(a[0, 0])
    if not (a[0, -1] == a[-1, 0] == a[-1, -1] == bg):
        raise ValueError("corners must be background")
    if np.any(a[0] != bg) or np.any(a[-1] != bg):
        raise ValueError("ports must be on a vertical edge")
    edges = [np.flatnonzero(a[1:-1, x] != bg) + 1 for x in (0, w-1)]
    if sum(bool(len(e)) for e in edges) != 1:
        raise ValueError("ports must occupy exactly one vertical edge")
    right = bool(len(edges[1]))
    ports = tuple(map(int, edges[int(right)]))
    edge = w-1 if right else 0
    guide = int(a[ports[0], edge])
    if np.any(a[list(ports), edge] != guide):
        raise ValueError("all ports must have one guide color")
    interior = a[1:-1, 1:-1]
    if np.any(interior[1::2] != bg) or np.any(interior[:, 1::2] != bg):
        raise ValueError("interior scaffold must be background")
    coarse = a[1::2, 1::2].copy()
    if np.any(coarse == guide) or not np.any(coarse != bg):
        raise ValueError("guide is reserved for ports; at least one payload is needed")
    return coarse, bg, guide, ports, right


def _runs(row, bg):
    result = []
    c = 0
    while c < len(row):
        color = int(row[c])
        if color == bg:
            c += 1
            continue
        end = c + 1
        while end < len(row) and row[end] == color:
            end += 1
        result.append((color, c, end))
        c = end
    return result


def _rectangles(coarse, bg, mode="exact"):
    """Return coarse (top,bottom-exclusive,left,right-exclusive,color) tuples."""
    runs = [_runs(row, bg) for row in coarse]
    if mode == "overlap_merge":
        # Deliberately wrong connected-run bounding boxes, used only for QA.
        nodes = [(r, color, a, b) for r, row in enumerate(runs) for color, a, b in row]
        parent = list(range(len(nodes)))
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i, (r, color, a, b) in enumerate(nodes):
            for j in range(i):
                s, other, c, d = nodes[j]
                if r == s+1 and color == other and max(a, c) < min(b, d):
                    parent[find(i)] = find(j)
        groups = {}
        for i, node in enumerate(nodes):
            groups.setdefault(find(i), []).append(node)
        return [(min(n[0] for n in g), max(n[0] for n in g)+1,
                 min(n[2] for n in g), max(n[3] for n in g), g[0][1])
                for g in groups.values()]
    result = []
    for r, row in enumerate(runs):
        for color, a, b in row:
            if mode != "no_merge" and r and (color, a, b) in runs[r-1]:
                continue
            bottom = r + 1
            if mode != "no_merge":
                while bottom < len(runs) and (color, a, b) in runs[bottom]:
                    bottom += 1
            result.append((r, bottom, a, b, color))
    return result


def _render(coarse, bg, mode="exact"):
    out = np.full((4*coarse.shape[0]+1, 4*coarse.shape[1]+1), bg, dtype=np.int8)
    rects = []
    for r, s, a, b, color in _rectangles(coarse, bg, mode):
        top, bottom, left, right = 4*r+1, 4*s, 4*a+1, 4*b
        out[top:bottom, left:right] = color
        rects.append((top, bottom, left, right, color))
    return out, rects


def _bridge_render(coarse, bg):
    """QA formulation: expand each coarse cell, then paint permitted bridges."""
    h, w = coarse.shape
    out = np.full((4*h+1, 4*w+1), bg, dtype=np.int8)
    for r in range(h):
        for c in range(w):
            color = int(coarse[r, c])
            if color == bg:
                continue
            out[4*r+1:4*r+4, 4*c+1:4*c+4] = color
            if c+1 < w and coarse[r, c+1] == color:
                out[4*r+1:4*r+4, 4*c+4] = color
    rows = [_runs(row, bg) for row in coarse]
    for r in range(h-1):
        for color, a, b in set(rows[r]) & set(rows[r+1]):
            out[4*r+4, 4*a+1:4*b] = color
    return out


def _flow(field, bg, guide, ports, right=False, mode="normal"):
    """Advance simultaneous rays on immutable rectangles; never use drawn ink as an obstacle."""
    a = np.fliplr(field) if right else field
    out = a.copy()
    h, w = a.shape
    active = {2*p for p in ports}
    out[list(active), 0] = guide
    events = []
    reached = set()
    simultaneous = multi_hit = shared = interrupted = 0
    for x in range(1, w):
        hits, following = {}, set()
        for y in sorted(active):
            if a[y, x] == bg:
                following.add(y)
                out[y, x] = guide
            else:
                top = bottom = y
                while top and a[top-1, x] != bg:
                    top -= 1
                while bottom+1 < h and a[bottom+1, x] != bg:
                    bottom += 1
                hits.setdefault((top, bottom), []).append(y)
        simultaneous += len(hits) > 1
        chosen = sorted(hits)
        if mode == "one_per_layer":
            chosen = chosen[:1]
        for top, bottom in chosen:
            columns = hits[top, bottom]
            multi_hit += len(columns) > 1
            lo, hi = max(0, top-1), min(h, bottom+2)
            interrupted += bool(np.any(a[lo:hi, x-1] != bg))
            upstream = out[lo:hi, x-1]
            upstream[a[lo:hi, x-1] == bg] = guide
            for y in (top-1, bottom+1):
                if 0 <= y < h:
                    shared += y in following
                    following.add(y)
                    out[y, x] = guide
            # Locate the forward extent of the struck rectangle for statistics.
            start = x
            while start > 0 and a[top, start-1] != bg:
                start -= 1
            reached.add((top, bottom, start))
            events.append((x, top, bottom, len(columns)))
        active = following
    stats = {"events": events, "hits": len(events), "reached_rectangles": len(reached),
             "simultaneous": simultaneous, "multi_hit": multi_hit, "mergers": shared,
             "interrupted": interrupted, "entry_count": len(ports),
             "collision_depths": len({e[0] for e in events})}
    return (np.fliplr(out).copy() if right else out), stats


def solve(input_grid: np.ndarray) -> np.ndarray:
    coarse, bg, guide, ports, right = _decode(input_grid)
    field, _ = _render(coarse, bg)
    return _flow(field, bg, guide, ports, right)[0]


def _wrong(input_grid, shortcut):
    """Finite counterfactual rules for demonstrating the causal dependency."""
    coarse, bg, guide, ports, right = _decode(input_grid)
    mode = shortcut if shortcut in ("no_merge", "overlap_merge") else "exact"
    field, rects = _render(coarse, bg, mode)
    if shortcut == "ignore_color":
        rows = [_runs(row, bg) for row in coarse]
        for r in range(len(rows)-1):
            for color, left, end in rows[r]:
                if any((left, end) == (a, b) for _, a, b in rows[r+1]):
                    field[4*r+4, 4*left+1:4*end] = color
    elif shortcut == "cellwise":
        field[:] = bg
        for r, c in zip(*np.nonzero(coarse != bg)):
            field[4*r+1:4*r+4, 4*c+1:4*c+4] = coarse[r, c]
    if shortcut in ("no_merge", "overlap_merge", "one_per_layer", "ignore_color", "cellwise"):
        return _flow(field, bg, guide, ports, right,
                     "one_per_layer" if shortcut == "one_per_layer" else "normal")[0]
    out = field.copy()
    out[[2*p for p in ports], -1 if right else 0] = guide
    if shortcut in ("straight", "outline_all"):
        for p in ports:
            out[2*p, field[2*p] == bg] = guide
    if shortcut == "outline_all":
        for top, bottom, left, end, _ in rects:
            for y in (top-1, bottom):
                cells = out[y, left-1:end+1]
                cells[field[y, left-1:end+1] == bg] = guide
            for x in (left-1, end):
                cells = out[top-1:bottom+1, x]
                cells[field[top-1:bottom+1, x] == bg] = guide
    return out


def _features(coarse, bg):
    rows = [_runs(row, bg) for row in coarse]
    horizontal = any(b-a > 1 for row in rows for _, a, b in row)
    exact = any(set(a) & set(b) for a, b in zip(rows, rows[1:]))
    unequal = any(c == d and (a, b) != (e, f) and max(a, e) < min(b, f)
                  for upper, lower in zip(rows, rows[1:])
                  for c, a, b in upper for d, e, f in lower)
    return bool(horizontal), bool(exact), bool(unequal)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _encode(coarse, bg, guide, ports, right):
    h, w = coarse.shape
    out = np.full((2*h+1, 2*w+1), bg, dtype=np.int8)
    out[1::2, 1::2] = coarse
    out[list(ports), -1 if right else 0] = guide
    return out


def _sample_coarse(rng, shape, bg, colors, role):
    h, w = shape
    a = np.full(shape, bg, dtype=np.int8)
    density = float(rng.uniform(.18, .70))
    for r in range(h):
        c = 0
        while c < w:
            if rng.random() < density:
                length = int(rng.integers(1, w-c+1))
                a[r, c:c+length] = int(rng.choice(colors))
                c += length
            else:
                c += 1
    # Repeat, interrupt, extend or return sampled runs: not a bank of whole layouts.
    for _ in range(int(rng.integers(4))):
        if h < 2:
            break
        r = int(rng.integers(h-1))
        row_runs = _runs(a[r], bg)
        if not row_runs:
            continue
        color, left, end = row_runs[int(rng.integers(len(row_runs)))]
        a[r+1] = bg
        a[r+1, left:end] = color
        if h > r+2 and rng.random() < .4:
            a[r+2] = a[r+1]
    ports = set()
    if role == "exact":
        depth = int(rng.integers(2, min(3, h-1)+1))
        top = int(rng.integers(h-depth+1))
        left = int(rng.integers(w-1))
        end = int(rng.integers(left+2, w+1))
        a[top:top+depth] = bg
        a[top:top+depth, left:end] = int(rng.choice(colors))
        ports.add(2*(top+int(rng.integers(1, depth))))
        # An outside-row rectangle teaches that a barrier need not be reached.
        outside = [r for r in range(h) if not top <= r < top+depth]
        r = int(rng.choice(outside))
        c = int(rng.integers(w))
        a[r, c:int(rng.integers(c+1, w+1))] = int(rng.choice(colors))
    elif role == "unequal":
        top = int(rng.integers(h-1))
        spans = [(l, e) for l in range(w) for e in range(l+1, w+1)]
        choices = [(s, t) for s in spans for t in spans
                   if s != t and max(s[0], t[0]) < min(s[1], t[1])]
        # Often align the near faces, exposing two simultaneous distinct hits.
        if rng.random() < .65:
            choices = [(s, t) for s, t in choices if s[0] == t[0]]
        first, second = choices[int(rng.integers(len(choices)))]
        color = int(rng.choice(colors))
        a[top:top+2] = bg
        a[top, first[0]:first[1]] = color
        a[top+1, second[0]:second[1]] = color
        ports.update((2*top+1, 2*top+2, 2*top+3))
    if not np.any(a != bg):
        a[int(rng.integers(h)), int(rng.integers(w))] = int(rng.choice(colors))
    desired = int(rng.integers(1, 4))
    available = [p for p in range(1, 2*h) if p not in ports]
    if desired > len(ports):
        ports.update(map(int, rng.choice(available, min(desired-len(ports), len(available)), replace=False)))
    return a, tuple(sorted(ports))


def _make_pair(rng, shape, role, palette, right, seen, consistent=()):
    bg, guide, *available = palette
    for _ in range(160):
        count = int(rng.integers(1, len(available)+1))
        colors = list(map(int, rng.choice(available, count, replace=False)))
        coarse, ports = _sample_coarse(rng, shape, bg, colors, role)
        if right:
            # Mirror the construction before encoding; A still groups horizontal runs.
            coarse = np.fliplr(coarse).copy()
        inp = _encode(coarse, bg, guide, ports, right)
        if _key(inp) in seen:
            continue
        field, rects = _render(coarse, bg)
        out, stats = _flow(field, bg, guide, ports, right)
        if not stats["hits"]:
            continue
        if role == "exact":
            if stats["reached_rectangles"] == len(rects):
                continue
            wrong = _wrong(inp, "no_merge")
            if np.array_equal(out == guide, wrong == guide):
                continue
        if role == "unequal":
            wrong = _wrong(inp, "overlap_merge")
            if np.array_equal(out == guide, wrong == guide):
                continue
        if any(not np.array_equal(_wrong(inp, m), out) for m in consistent):
            continue
        return {"input": inp, "output": out}
    raise RuntimeError(f"bounded construction exhausted for {shape}/{role}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 6, replace=False)))
    mixed_palette = bool(rng.integers(2))
    mixed_direction = bool(rng.integers(2))
    base_right = bool(rng.integers(2))
    puzzle = {"train": [], "test": []}
    seen = set()
    for i in range(nt):
        role = ("exact", "unequal")[i] if i < 2 else str(rng.choice(("exact", "unequal", "random")))
        shape = tuple(map(int, rng.integers(3, 7, 2))) if role != "random" else tuple(map(int, rng.integers(1, 7, 2)))
        local = tuple(map(int, rng.choice(10, 6, replace=False))) if mixed_palette else palette
        right = (bool((int(base_right)+i) % 2) if mixed_direction else base_right)
        pair = _make_pair(rng, shape, role, local, right, seen)
        seen.add(_key(pair["input"]))
        puzzle["train"].append(pair)
    # Do not challenge any tested shortcut that the requested demonstrations admit.
    consistent = [m for m in SHORTCUTS if all(np.array_equal(_wrong(p["input"], m), p["output"])
                                              for p in puzzle["train"])]
    for _ in range(nq):
        shape = (7, int(rng.integers(2, 8))) if rng.random() < .5 else (int(rng.integers(3, 8)), 7)
        role = str(rng.choice(("exact", "unequal")))
        local = tuple(map(int, rng.choice(10, 6, replace=False))) if mixed_palette else palette
        right = bool(rng.integers(2)) if mixed_direction else base_right
        pair = _make_pair(rng, shape, role, local, right, seen, consistent)
        seen.add(_key(pair["input"]))
        puzzle["test"].append(pair)
    order = rng.permutation(nt)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, flags, shapes = set(), [False]*3, {"train": [], "test": []}
    teaches_exact = teaches_unequal = inert = False
    training = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            coarse, bg, guide, ports, right = _decode(inp)
            field, rects = _render(coarse, bg)
            assert np.array_equal(field, _bridge_render(coarse, bg)), "rectangle bridge invariant"
            expected, stats = _flow(field, bg, guide, ports, right)
            assert out.shape == field.shape and np.array_equal(out, expected), "wrong solution"
            assert np.all(out[field != bg] == field[field != bg]), "payload was overwritten"
            assert np.all(out[out != field] == guide), "only guide may be added after rendering"
            assert np.all(out[[2*p for p in ports], -1 if right else 0] == guide)
            assert stats["hits"] > 0 and not np.array_equal(out, field)
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            shapes[split].append(coarse.shape)
            if split == "train":
                flags = [a or b for a, b in zip(flags, _features(coarse, bg))]
                teaches_exact |= not np.array_equal(out == guide, _wrong(inp, "no_merge") == guide)
                teaches_unequal |= not np.array_equal(out == guide, _wrong(inp, "overlap_merge") == guide)
                inert |= stats["reached_rectangles"] < len(rects)
                training.append((inp, out))
    assert all(flags) and teaches_exact and teaches_unequal and inert, "missing causal training evidence"
    max_rows = max(s[0] for s in shapes["train"])
    max_cols = max(s[1] for s in shapes["train"])
    assert all(r > max_rows or c > max_cols for r, c in shapes["test"]), "missing dimension transfer"
    for shortcut in SHORTCUTS:
        if all(np.array_equal(_wrong(a, shortcut), b) for a, b in training):
            assert all(np.array_equal(_wrong(p["input"], shortcut), p["output"])
                       for p in puzzle["test"]), f"undistinguished shortcut: {shortcut}"


def _puzzle_hash(puzzle):
    value = {s: [{k: np.asarray(v).tolist() for k, v in p.items()} for p in puzzle[s]]
             for s in ("train", "test")}
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def _worklist_check(input_grid):
    """Authored QA cross-check using individual ray positions, without layers."""
    coarse, bg, guide, ports, right = _decode(input_grid)
    field = _bridge_render(coarse, bg)
    out = field.copy()
    h, w = field.shape
    dx = -1 if right else 1
    pending = [(2*p, w-1 if right else 0) for p in ports]
    visited = set()
    while pending:
        y, x = pending.pop()
        if not (0 <= y < h and 0 <= x < w) or (y, x) in visited:
            continue
        visited.add((y, x))
        if field[y, x] == bg:
            out[y, x] = guide
            pending.append((y, x+dx))
            continue
        span = [y]
        for step in (-1, 1):
            t = y+step
            while 0 <= t < h and field[t, x] != bg:
                span.append(t)
                t += step
        top, bottom = min(span)-1, max(span)+1
        if 0 <= x-dx < w:
            for t in range(max(0, top), min(h, bottom+1)):
                if field[t, x-dx] == bg:
                    out[t, x-dx] = guide
        pending.extend(((top, x), (bottom, x)))
    return out


def self_test():
    """Bounded, streamed 42-seed coverage; no grid corpus or traces are saved."""
    shapes, layouts, directions, palettes, signatures = set(), set(), set(), set(), set()
    widths, depths, rectangle_counts, entries = set(), set(), set(), set()
    simultaneous = multi_hit = mergers = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_hash(puzzle))
        small = generate_puzzle(seed, 2, 1)
        validate_generated_puzzle(small)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                coarse, bg, guide, ports, right = _decode(pair["input"])
                field, rects = _render(coarse, bg)
                _, stats = _flow(field, bg, guide, ports, right)
                assert np.array_equal(solve(pair["input"]), pair["output"])
                assert np.array_equal(_worklist_check(pair["input"]), pair["output"])
                shapes.add(coarse.shape)
                directions.add(right)
                palettes.add((bg, guide))
                mapping = {bg: 0}
                labels = tuple(mapping.setdefault(int(c), len(mapping)) for c in coarse.flat)
                layouts.add((coarse.shape, labels, ports, right))
                entries.add(len(ports))
                rectangle_counts.add(len(rects))
                for r, s, a, b, _ in _rectangles(coarse, bg):
                    widths.add(b-a)
                    depths.add(s-r)
                simultaneous += stats["simultaneous"] > 0
                multi_hit += stats["multi_hit"] > 0
                mergers += stats["mergers"] > 0
    assert len(signatures) == 42 and len(layouts) >= 170 and len(shapes) >= 24
    assert directions == {False, True} and len(palettes) >= 25
    assert {1, 2, 3} <= entries and max(widths) >= 5 and max(depths) >= 3
    assert len(rectangle_counts) >= 8 and simultaneous >= 10 and multi_hit >= 10 and mergers >= 10
    # Hand-derived gate regressions. Exact spans block the inter-row channel;
    # unequal overlapping spans leave it open, even with the same payload color.
    exact = _encode(np.array([[2,2],[2,2]], dtype=np.int8), 0, 9, (2,), False)
    result = solve(exact)
    assert np.all(result[0] == 9) and np.all(result[8] == 9)
    assert np.all(result[1:8, 1:8] == 2) and np.all(result[:, 0] == 9)
    unequal = _encode(np.array([[2,2],[2,0]], dtype=np.int8), 0, 9, (1,2), False)
    result = solve(unequal)
    assert np.all(result[4] == 9) and np.all(result[0] == 9)
    assert np.all(result[5:8, 1:4] == 2) and np.all(result[8] == 0)
    for a in (exact, unequal):
        assert np.array_equal(solve(a), _worklist_check(a))
        assert np.array_equal(solve(np.fliplr(a)), np.fliplr(solve(a)))
    assert _puzzle_hash(generate_puzzle(0, 7, 3)) == _puzzle_hash(generate_puzzle(0, 7, 3))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(0, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(4.0,1),(True,1),(4,None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts were accepted")
    return {"seeds": len(HOST_SEEDS), "distinct_puzzles": len(signatures),
            "coarse_shapes": len(shapes), "normalized_layouts": len(layouts),
            "background_guide_roles": len(palettes), "rectangle_counts": sorted(rectangle_counts),
            "run_widths": sorted(widths), "stack_depths": sorted(depths),
            "simultaneous_pairs": simultaneous, "multi_hit_pairs": multi_hit,
            "merging_pairs": mergers, "stored_traces": 0}
