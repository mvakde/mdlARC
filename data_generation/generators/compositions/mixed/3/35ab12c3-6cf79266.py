"""Complete translated drawings, then mark the union of blank 3x3 windows.

Only standard Python and NumPy are used. No evidence files are loaded at runtime.
The topology inference follows the geometric convention in parent A's final
reference: a unique simple full cycle takes precedence over a unique full path.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np


HIGHLIGHT = 1
OFFSETS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
HOST_SEEDS = (0, 298135826, 38038339, 732405024, 462494316, 914787214,
              910604023, 422000147, 979275695, 385869559, 253855853,
              917182668, 951758856, 530362151, 347801559, 721760186,
              357075354, 823410114, 431963740, 994752928, 92625806,
              749200809, 790665910, 937224342, 823811563, 554760206,
              262868774, 355818921, 30851008, 844883602, 790974017,
              556708186, 869644189, 977225832, 573107972, 538488980,
              993104232, 171510106, 782675762, 729221823, 146721497, 693284569)


def _integer(value, name, lo, hi=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < lo or (hi is not None and value > hi)):
        raise ValueError(f"invalid {name}")
    return int(value)


def _sgn(x):
    return int(x > 0) - int(x < 0)


def _aligned(a, b):
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dr, dc) > 0 and (dr == 0 or dc == 0 or dr == dc)


def _line(a, b):
    if not _aligned(a, b):
        raise ValueError("unaligned link")
    dr, dc = b[0] - a[0], b[1] - a[1]
    return {(a[0] + i * _sgn(dr), a[1] + i * _sgn(dc))
            for i in range(max(abs(dr), abs(dc)) + 1)}


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _intersects(a, b, c, d):
    if (max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0])
            or max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1])):
        return False
    return _cross(a, b, c) * _cross(a, b, d) <= 0 and _cross(c, d, a) * _cross(c, d, b) <= 0


def _infer(points):
    """Recover edges from unordered visible vertices; reject ambiguous topology."""
    n = len(points)
    if not 2 <= n <= 10:
        raise ValueError("a drawing needs 2..10 landmarks")
    vertices = set(points)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)
             if _aligned(points[i], points[j])
             and not ((_line(points[i], points[j]) - {points[i], points[j]}) & vertices)]
    adj = [[] for _ in points]
    for e, (i, j) in enumerate(edges):
        adj[i].append((j, e))
        adj[j].append((i, e))
    if not all(adj):
        raise ValueError("disconnected landmarks")
    conflicts = [set() for _ in edges]
    for e, (a, b) in enumerate(edges):
        for f, (c, d) in enumerate(edges[:e]):
            if not ({a, b} & {c, d}) and _intersects(points[a], points[b], points[c], points[d]):
                conflicts[e].add(f)
                conflicts[f].add(e)

    def search(closed):
        found = {}
        visits = 0

        def walk(order, used, chosen):
            nonlocal visits
            visits += 1
            if visits > 30000:
                raise ValueError("topology search budget exceeded")
            if len(found) > 1:
                return
            if len(order) == n:
                if closed:
                    closing = [e for v, e in adj[order[-1]] if v == order[0]]
                    if not closing or conflicts[closing[0]] & set(chosen):
                        return
                    chosen = chosen + [closing[0]]
                found[tuple(sorted(chosen))] = order
                return
            for v, e in adj[order[-1]]:
                if v not in used and not conflicts[e] & set(chosen):
                    walk(order + [v], used | {v}, chosen + [e])

        for start in ([0] if closed else range(n)):
            walk([start], {start}, [])
            if len(found) > 1:
                break
        return found

    cycles = search(True) if n > 2 else {}
    if len(cycles) > 1:
        raise ValueError("ambiguous contour")
    found = cycles or search(False)
    if len(found) != 1:
        raise ValueError("ambiguous chain")
    selected, order = next(iter(found.items()))
    links = [edges[e] for e in selected]
    pixels, long_pixels = set(), set()
    for i, j in links:
        line = _line(points[i], points[j])
        pixels.update(line)
        if len(line) > 2:
            long_pixels.update(line)
    turns = [_sgn(_cross(points[order[i - 1]], points[order[i]], points[order[(i + 1) % n]]))
             for i in range(n)] if cycles else []
    return {"points": points, "edges": links, "pixels": pixels, "long": long_pixels,
            "closed": bool(cycles), "chords": len(edges) - len(links),
            "concave": -1 in turns and 1 in turns,
            "diagonal": any(points[i][0] != points[j][0] and points[i][1] != points[j][1]
                            for i, j in links)}


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9))):
        raise ValueError("grid must be an integer 1..30 by 1..30 array with colors 0..9")
    return a.astype(np.int8, copy=True)


def _complete(inp):
    values, counts = np.unique(inp, return_counts=True)
    bg = int(values[np.argmax(counts)])
    if int(max(counts)) * 2 <= inp.size or HIGHLIGHT in values:
        raise ValueError("background must be a strict majority; highlight 1 must be absent")
    groups = {int(c): [tuple(map(int, p)) for p in np.argwhere(inp == c)]
              for c in values if c != bg}
    sources = []
    for color, points in groups.items():
        if len(points) >= 2:
            info = _infer(points)
            info["color"] = color
            if not any(inp[p] == bg for p in info["pixels"]):
                raise ValueError("drawing is already complete")
            sources.append(info)
    if not sources:
        raise ValueError("no source drawing")
    foreground = set()
    for s in sources:
        if foreground & s["pixels"]:
            raise ValueError("sources collide")
        foreground.update(s["pixels"])
    markers, marked = [], set()
    for color, points in groups.items():
        if len(points) != 1:
            continue
        p = points[0]
        associations = [(i, v) for i, s in enumerate(sources) for v in s["points"]
                        if max(abs(p[0] - v[0]), abs(p[1] - v[1])) == 1]
        if len(associations) != 1:
            raise ValueError("singleton must unambiguously mark one vertex")
        i, v = associations[0]
        if i in marked:
            raise ValueError("multiple markers for one drawing")
        marked.add(i)
        d = p[0] - v[0], p[1] - v[1]
        copy = {(r + d[0], c + d[1]) for r, c in sources[i]["long"]}
        if p not in copy or not all(0 <= r < inp.shape[0] and 0 <= c < inp.shape[1] for r, c in copy):
            raise ValueError("invalid or clipped companion")
        markers.append({"color": color, "point": p, "source": i, "offset": d, "pixels": copy})
    completed, source_only = inp.copy(), inp.copy()
    claimed = set()
    for m in markers:
        for p in m["pixels"]:
            if inp[p] == bg and p not in foreground:
                if p in claimed:
                    raise ValueError("companion collision")
                claimed.add(p)
                completed[p] = m["color"]
    for s in sources:
        for p in s["pixels"]:
            if inp[p] == bg:
                completed[p] = source_only[p] = s["color"]
    return completed, source_only, {"background": bg, "sources": sources, "markers": markers}


def _windows(blank):
    """Simultaneous full-window detection. Never pad, clip, rank, or consume."""
    h, w = blank.shape
    mask = np.zeros((h, w), dtype=bool)
    if min(h, w) < 3:
        return [], mask
    counts = np.zeros((h - 2, w - 2), dtype=np.uint8)
    for dr in range(3):
        for dc in range(3):
            counts += blank[dr:h - 2 + dr, dc:w - 2 + dc]
    starts = [tuple(map(int, p)) for p in np.argwhere(counts == 9)]
    for r, c in starts:
        mask[r:r + 3, c:c + 3] = True
    return starts, mask


def solve(input_grid: np.ndarray) -> np.ndarray:
    inp = _grid(input_grid)
    completed, _, info = _complete(inp)
    _, mask = _windows(completed == info["background"])
    completed[mask] = HIGHLIGHT
    return completed


def _components(mask):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in remaining:
                    remaining.remove(p)
                    component.add(p)
                    stack.append(p)
        result.append(component)
    return result


def measure_pair(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    completed, source_only, info = _complete(inp)
    bg = info["background"]
    starts, mask = _windows(completed == bg)
    raw_starts, raw_mask = _windows(inp == bg)
    _, source_mask = _windows(source_only == bg)
    components = _components(mask)
    # Compare highlights on the SAME final background, so this proves dependency
    # beyond the trivial observation that a no-completion output lacks strokes.
    final_blank = completed == bg
    expected = completed.copy()
    expected[mask] = HIGHLIGHT
    assert out.shape == inp.shape and np.array_equal(out, expected), "incorrect output"
    component_fill_extra = sum(len(comp - set(map(tuple, np.argwhere(mask))))
                               for comp in _components(final_blank) if any(mask[p] for p in comp))
    return {"shape": list(inp.shape), "background": bg, "sources": len(info["sources"]),
            "markers": len(info["markers"]), "offsets": [list(m["offset"]) for m in info["markers"]],
            "vertices": [len(s["points"]) for s in info["sources"]],
            "cycles": sum(s["closed"] for s in info["sources"]),
            "paths": sum(not s["closed"] for s in info["sources"]),
            "diagonal": sum(s["diagonal"] for s in info["sources"]),
            "concave": sum(s["concave"] for s in info["sources"]),
            "chords": sum(s["chords"] for s in info["sources"]),
            "windows": len(starts), "raw_windows": len(raw_starts), "highlight_cells": int(mask.sum()),
            "overlap": int(mask.sum()) < 9 * len(starts),
            "isolated_squares": sum(len(c) == 9 and len({r for r, _ in c}) == 3
                                    and len({c0 for _, c0 in c}) == 3 for c in components),
            "highlight_components": len(components),
            "raw_mask_error": int(np.count_nonzero((raw_mask & final_blank) != mask)),
            "copy_mask_error": int(np.count_nonzero((source_mask & final_blank) != mask)),
            "component_fill_error": component_fill_extra,
            "boundary_window": any(r == 0 or c == 0 or r == inp.shape[0] - 3 or c == inp.shape[1] - 3
                                   for r, c in starts),
            "source_copy_overlap": sum(len(m["pixels"] & info["sources"][m["source"]]["pixels"])
                                       for m in info["markers"]),
            "visible_copy": sum(int(np.count_nonzero((completed == m["color"]) & (inp == bg)))
                                for m in info["markers"]),
            "residual_blank": int(np.count_nonzero(final_blank & ~mask))}


def _transform(points, code):
    result = []
    for r, c in points:
        if code & 4:
            c = -c
        for _ in range(code % 4):
            r, c = c, -r
        result.append((r, c))
    return result


def _spec(rng, points, closed, allowed, marked=True):
    info = _infer(points)
    intended = {tuple(sorted((i, i + 1))) for i in range(len(points) - 1)}
    if closed:
        intended.add((0, len(points) - 1))
    if set(info["edges"]) != intended or info["closed"] != closed or info["pixels"] != info["long"]:
        raise ValueError("template does not visibly specify intended drawing")
    options = []
    if marked:
        for d in allowed:
            copy = {(r + d[0], c + d[1]) for r, c in info["long"]}
            for v in points:
                p = v[0] + d[0], v[1] + d[1]
                if (p not in info["pixels"]
                        and sum(max(abs(p[0] - q[0]), abs(p[1] - q[1])) == 1 for q in points) == 1
                        and len(copy - info["pixels"] - {p}) >= 2):
                    options.append((p, copy, d))
        if not options:
            raise ValueError("no visible companion marker")
        marker, copy, offset = options[int(rng.integers(len(options)))]
    else:
        marker, copy, offset = None, set(), None
    return {"points": list(points), "pixels": info["pixels"], "copy": copy,
            "marker": marker, "offset": offset}


def _shift(s, dr, dc):
    shifted = dict(s)
    for k in ("points", "pixels", "copy"):
        q = [(r + dr, c + dc) for r, c in s[k]]
        shifted[k] = q if k == "points" else set(q)
    if s["marker"] is not None:
        shifted["marker"] = s["marker"][0] + dr, s["marker"][1] + dc
    return shifted


def _bounds(s):
    p = s["pixels"] | s["copy"]
    r0, c0 = min(r for r, _ in p), min(c for _, c in p)
    return r0, c0, max(r for r, _ in p) - r0 + 1, max(c for _, c in p) - c0 + 1


def _render(rng, specs, bg, padding):
    footprints = set().union(*(s["pixels"] | s["copy"] for s in specs))
    r0, c0 = min(r for r, _ in footprints), min(c for _, c in footprints)
    h, w = max(r for r, _ in footprints) - r0 + 1, max(c for _, c in footprints) - c0 + 1
    pads = []
    for size in (h, w):
        lo, hi = padding
        if size + 2 * lo > 30:
            raise ValueError("padding does not fit")
        before = int(rng.integers(lo, min(hi, 30 - size - lo) + 1))
        after = int(rng.integers(lo, min(hi, 30 - size - before) + 1))
        pads.append((before, after))
    specs = [_shift(s, pads[0][0] - r0, pads[1][0] - c0) for s in specs]
    h += sum(pads[0])
    w += sum(pads[1])
    inp = np.full((h, w), bg, dtype=np.int8)
    colors = [int(c) for c in rng.permutation(10) if c not in (bg, HIGHLIGHT)]
    for s in specs:
        s["color"] = colors.pop()
        for p in s["points"]:
            if inp[p] != bg:
                raise ValueError("landmark collision")
            inp[p] = s["color"]
        if s["marker"] is not None:
            s["copy_color"] = colors.pop()
            if inp[s["marker"]] != bg:
                raise ValueError("marker collision")
            inp[s["marker"]] = s["copy_color"]
    completed = inp.copy()
    for s in specs:
        if s["marker"] is not None:
            for p in s["copy"]:
                if inp[p] == bg:
                    completed[p] = s["copy_color"]
    for s in specs:
        for p in s["pixels"]:
            if inp[p] == bg:
                completed[p] = s["color"]
    # Label from intended geometry and a direct patch scan, separately from solve.
    out = completed.copy()
    for r in range(h - 2):
        for c in range(w - 2):
            if np.all(completed[r:r + 3, c:c + 3] == bg):
                out[r:r + 3, c:c + 3] = HIGHLIGHT
    pair = {"input": inp, "output": out}
    measure_pair(pair)  # verify that visible inference agrees with intended geometry
    return pair


def _isolated_demo(rng, bg, allowed):
    specs, cursor = [], 0
    for _ in range(int(rng.integers(2, 4))):
        d = allowed[int(rng.integers(len(allowed)))]
        h, w = 4 + abs(d[0]), 4 + abs(d[1])
        s = _spec(rng, [(0, 0), (0, w), (h, w), (h, 0)], True, (d,))
        r0, c0, sh, sw = _bounds(s)
        s = _shift(s, int(rng.integers(0, 3)) - r0, cursor - c0)
        specs.append(s)
        cursor += sw + int(rng.integers(1, 3))
    # Rotate/reflect the whole strip, including marker displacement.
    code = int(rng.integers(8))
    transformed = []
    for s in specs:
        # Only retain transforms whose actual marker directions are allowed.
        if _transform([s["offset"]], code)[0] not in allowed:
            code = 0
            break
    for s in specs:
        q = dict(s)
        for k in ("points", "pixels", "copy"):
            v = _transform(s[k], code)
            q[k] = v if k == "points" else set(v)
        q["marker"] = _transform([s["marker"]], code)[0]
        q["offset"] = _transform([s["offset"]], code)[0]
        transformed.append(q)
    return _render(rng, transformed, bg, (0, 2))


def _shape(rng, family):
    h, w = int(rng.integers(3, 10)), int(rng.integers(3, 10))
    if family == "rectangle":
        return [(0, 0), (0, w), (h, w), (h, 0)], True
    if family == "diamond":
        k = int(rng.integers(2, 6))
        return [(0, k), (k, 2 * k), (2 * k, k), (k, 0)], True
    if family == "bevel":
        h, w = max(h, 5), max(w, 5)
        a, b, c, d = [int(rng.integers(1, 3)) for _ in range(4)]
        return [(0, a), (0, w - b), (b, w), (h - c, w),
                (h, w - c), (h, d), (h - d, 0), (a, 0)], True
    if family == "notch":
        h, w = int(rng.integers(6, 11)), int(rng.integers(8, 13))
        a = int(rng.integers(2, 4))
        b, depth = a + int(rng.integers(2, 4)), int(rng.integers(1, 4))
        return [(0, 0), (0, a), (depth, a), (depth, b), (0, b),
                (0, w), (h, w), (h, 0)], True
    if family == "mixed":
        a, b = int(rng.integers(2, 6)), int(rng.integers(2, 6))
        return [(0, 0), (0, a), (b, a + b)], False
    if family == "segment":
        n = int(rng.integers(3, 12))
        return [(0, 0), (n, n)] if rng.random() < .4 else [(0, 0), (0, n)], False
    points = [(0, 0)]
    for _ in range(int(rng.integers(3, 7))):
        d = OFFSETS[int(rng.integers(8))]
        n = int(rng.integers(2, 6))
        points.append((points[-1][0] + n * d[0], points[-1][1] + n * d[1]))
    return points, False


def _random_spec(rng, family, allowed, marked=True):
    for _ in range(100):
        points, closed = _shape(rng, family)
        points = _transform(points, int(rng.integers(8)))
        if len(points) != len(set(points)):
            continue
        try:
            s = _spec(rng, points, closed, allowed, marked)
            if max(_bounds(s)[2:]) <= 22:
                return s
        except ValueError:
            pass
    raise ValueError("shape sampling exhausted")


def _pack(rng, specs):
    placed, blocked = [], set()
    for s in specs:
        r0, c0, h, w = _bounds(s)
        if max(h, w) > 24:
            raise ValueError("shape too large")
        options = [(r - r0, c - c0) for r in range(25 - h) for c in range(25 - w)]
        rng.shuffle(options)
        for dr, dc in options:
            q = _shift(s, dr, dc)
            cells = q["pixels"] | q["copy"]
            if not cells & blocked:
                placed.append(q)
                blocked.update((r + a, c + b) for r, c in cells
                               for a in (-1, 0, 1) for b in (-1, 0, 1))
                break
        else:
            raise ValueError("packing failed")
    return placed


def _scene(rng, bg, allowed, kind, forced=None):
    for _ in range(100):
        try:
            if kind == "dependency":
                specs = [_spec(rng, [(0, 0), (0, 4), (4, 4), (4, 0)], True, allowed),
                         _random_spec(rng, "mixed", allowed)]
                padding = (3, 5)
            else:
                n = int(rng.integers(1, 4))
                families = ("rectangle", "diamond", "bevel", "notch", "mixed", "segment", "walk")
                specs = []
                for i in range(n):
                    family = str(rng.choice(families))
                    specs.append(_random_spec(rng, family, (forced,) if i == 0 and forced else allowed,
                                              marked=(i == 0 or rng.random() < .65)))
                padding = (0, 4)
            pair = _render(rng, _pack(rng, specs), bg, padding)
            m = measure_pair(pair)
            if (m["windows"] and m["raw_mask_error"] and m["residual_blank"]
                    and (kind != "dependency" or (m["copy_mask_error"] and m["overlap"]
                                                 and m["boundary_window"] and m["component_fill_error"]))):
                return pair
        except ValueError:
            continue
    raise RuntimeError("bounded scene construction exhausted")


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    bg = int(rng.choice([c for c in range(10) if c != HIGHLIGHT]))
    held = OFFSETS[int(rng.integers(8))]
    allowed = tuple(d for d in OFFSETS if d != held)
    puzzle, seen = {"train": [], "test": []}, set()
    for index in range(nt + nq):
        split = "train" if index < nt else "test"
        for _ in range(30):
            if index == 0:
                pair = _isolated_demo(rng, bg, allowed)
                if measure_pair(pair)["isolated_squares"] < 2:
                    continue
            elif index == 1:
                pair = _scene(rng, bg, allowed, "dependency")
            else:
                pair = _scene(rng, bg, allowed if split == "train" else OFFSETS,
                              "free", forced=held if split == "test" else None)
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("distinct input construction exhausted")
    # Demonstrations are not assigned semantic roles by their index.
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    rows, seen, backgrounds = {"train": [], "test": []}, set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            m = measure_pair(pair)
            backgrounds.add(m["background"])
            assert 1 <= m["sources"] <= 3 and 1 <= m["markers"] <= m["sources"]
            assert m["windows"] and m["raw_mask_error"] and m["residual_blank"]
            assert m["visible_copy"] > 0
            completed, _, info = _complete(inp)
            bg = info["background"]
            _, target = _windows(completed == bg)
            assert np.array_equal(out[inp != bg], inp[inp != bg]), "existing cells changed"
            assert np.array_equal(out[~target], completed[~target]), "non-target background changed"
            assert np.all(out[target] == HIGHLIGHT) and not np.any(target & (completed != bg))
            rows[split].append(m)
    assert len(backgrounds) == 1
    train = rows["train"]
    assert any(m["isolated_squares"] >= 2 for m in train), "exhaustive detection not demonstrated"
    assert any(m["overlap"] and m["boundary_window"] for m in train), "union/boundary missing"
    assert any(m["copy_mask_error"] for m in train), "companion does not affect window eligibility"
    assert any(m["paths"] and m["diagonal"] for m in train), "mixed chain not demonstrated"
    assert any(m["chords"] and m["source_copy_overlap"] for m in train)
    assert any(m["component_fill_error"] for m in train), "component shortcut not excluded"
    known = {tuple(d) for m in train for d in m["offsets"]}
    assert all(any(tuple(d) not in known for d in m["offsets"]) for m in rows["test"]), "missing offset shift"


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        h.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Stream 42 seeds; keep only small sets/counters, never save sample batches."""
    # Hand-sized checks establish overlap, exact size and full-window boundaries.
    starts, footprint = _windows(np.ones((3, 4), dtype=bool))
    assert starts == [(0, 0), (0, 1)] and int(footprint.sum()) == 12
    assert not _windows(np.ones((2, 5), dtype=bool))[1].any()
    near = np.ones((3, 3), dtype=bool)
    near[1, 1] = False
    assert not _windows(near)[1].any()
    shapes, geometries, backgrounds, offsets, digests, counts = set(), set(), set(), set(), set(), set()
    totals = Counter()
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        digests.add(_digest(p))
        for split in ("train", "test"):
            for pair in p[split]:
                m = measure_pair(pair)
                shapes.add(tuple(m["shape"]))
                backgrounds.add(m["background"])
                offsets.update(tuple(d) for d in m["offsets"])
                counts.add(m["sources"])
                _, _, info = _complete(pair["input"])
                geometries.add(tuple(sorted((s["closed"], tuple(sorted(
                    (r - min(q[0] for q in s["points"]), c - min(q[1] for q in s["points"]))
                    for r, c in s["points"]))) for s in info["sources"])))
                totals["pairs"] += 1
                for metric in ("paths", "cycles", "diagonal", "concave", "chords", "overlap",
                               "boundary_window", "isolated_squares", "raw_mask_error", "copy_mask_error"):
                    totals[metric] += m[metric]
                # Alternative implementation of B's predicate: explicit patch loop.
                completed, _, info = _complete(pair["input"])
                slow = completed.copy()
                for r in range(completed.shape[0] - 2):
                    for c in range(completed.shape[1] - 2):
                        if np.all(completed[r:r + 3, c:c + 3] == info["background"]):
                            slow[r:r + 3, c:c + 3] = HIGHLIGHT
                assert np.array_equal(slow, solve(pair["input"]))
    assert len(digests) == 42 and len(shapes) >= 40 and len(geometries) >= 70
    assert len(backgrounds) >= 7 and offsets == set(OFFSETS) and counts == {1, 2, 3}
    assert all(totals[k] > 0 for k in totals)
    assert totals["concave"] > 0
    # Every explicit count combination, including the two-example minimum.
    for nt in range(2, 11):
        for nq in range(1, min(nt - 1, 4) + 1):
            p = generate_puzzle(20261023, nt, nq)
            assert (len(p["train"]), len(p["test"])) == (nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(1, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    assert _digest(generate_puzzle(1729, 2, 1)) == _digest(generate_puzzle(1729, 2, 1))
    # All geometry and marker directions transform together, while the 3x3
    # footprint remains axis aligned under the eight square-grid symmetries.
    example = generate_puzzle(1729, 2, 1)["test"][0]
    for turns in range(4):
        for reflected in (False, True):
            a, b = np.rot90(example["input"], turns), np.rot90(example["output"], turns)
            if reflected:
                a, b = np.fliplr(a), np.fliplr(b)
            assert np.array_equal(solve(a), b)
    p = generate_puzzle(0)
    pair = p["train"][0]
    old = pair["output"].copy()
    p0 = tuple(map(int, np.argwhere(pair["output"] == HIGHLIGHT)[0]))
    pair["output"][p0] = pair["input"][p0]
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing target cell accepted")
    pair["output"] = old
    return {"seeds": len(HOST_SEEDS), "pairs": totals["pairs"], "shapes": len(shapes),
            "geometry_signatures": len(geometries), "backgrounds": sorted(backgrounds),
            "offsets": len(offsets), "source_counts": sorted(counts), "totals": dict(totals)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
