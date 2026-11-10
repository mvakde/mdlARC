"""Complete sparse drawings, then read the parity-selected top 2x2 window.

No parent imports, files, hidden parameters, or random state are used by solve.
Generation and self_test keep diagnostics bounded and write no files.
"""
from __future__ import annotations

import hashlib
import numpy as np


OFFSETS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _need(bool(np.all((0 <= a) & (a <= 9))), "invalid colors")
    return a.astype(np.int8, copy=True)


def _aligned(a, b):
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dr, dc) > 0 and (dr == 0 or dc == 0 or dr == dc)


def _line(a, b):
    _need(_aligned(a, b), "unaligned link")
    dr, dc = b[0] - a[0], b[1] - a[1]
    sr, sc = (dr > 0) - (dr < 0), (dc > 0) - (dc < 0)
    return [(a[0] + i * sr, a[1] + i * sc) for i in range(max(abs(dr), abs(dc)) + 1)]


def _candidate_edges(points):
    occupied = set(points)
    return [(i, j) for i, a in enumerate(points) for j in range(i + 1, len(points))
            if _aligned(a, points[j]) and not occupied.intersection(_line(a, points[j])[1:-1])]


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _intersects(a, b, c, d):
    if (max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0])
            or max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1])):
        return False
    return _cross(a, b, c) * _cross(a, b, d) <= 0 and _cross(c, d, a) * _cross(c, d, b) <= 0


def _pixels(points, edges, long_only=False):
    result = set()
    for i, j in edges:
        cells = _line(points[i], points[j])
        if not long_only or len(cells) > 2:
            result.update(cells)
    return result


def _topology(points):
    """Unique noncrossing Hamiltonian cycle, otherwise unique undirected path.

    Edges cannot pass through another landmark. Reversals/cyclic starts count
    as the same drawing. Search is bounded, and ambiguous inputs are rejected.
    """
    _need(len(points) <= 12, "at most 12 landmarks per source")
    edges = _candidate_edges(points)
    if not edges:
        return None
    adjacency = [[] for _ in points]
    for k, (i, j) in enumerate(edges):
        adjacency[i].append((j, k))
        adjacency[j].append((i, k))
    _need(all(adjacency), "disconnected source landmarks")
    conflict = [set() for _ in edges]
    for k, (i, j) in enumerate(edges):
        for l, (a, b) in enumerate(edges[:k]):
            if not {i, j}.intersection((a, b)) and _intersects(points[i], points[j], points[a], points[b]):
                conflict[k].add(l)
                conflict[l].add(k)

    def find(closed):
        solutions, budget = {}, [0]

        def walk(order, used, chosen):
            budget[0] += 1
            _need(budget[0] <= 40000, "topology search budget exceeded")
            if len(solutions) >= 2:
                return
            if len(order) == len(points):
                if closed:
                    end = [e for v, e in adjacency[order[-1]] if v == order[0]]
                    if not end or conflict[end[0]].intersection(chosen):
                        return
                    chosen = chosen + [end[0]]
                solutions[tuple(sorted(chosen))] = order
                return
            for v, e in adjacency[order[-1]]:
                if v not in used and not conflict[e].intersection(chosen):
                    walk(order + [v], used | {v}, chosen + [e])

        for start in ([0] if closed else range(len(points))):
            walk([start], {start}, [])
            if len(solutions) >= 2:
                break
        return solutions

    cycles = find(True) if len(points) >= 3 else {}
    _need(len(cycles) <= 1, "ambiguous contours")
    solutions = cycles or find(False)
    _need(len(solutions) == 1, "ambiguous or impossible chain")
    chosen, order = next(iter(solutions.items()))
    selected = [edges[k] for k in chosen]
    pixels = _pixels(points, selected)
    long_pixels = _pixels(points, selected, True)
    _need(bool(long_pixels - set(points)), "source must have a gap")
    turns = [_cross(points[order[i - 1]], points[order[i]], points[order[(i + 1) % len(order)]])
             for i in range(len(order))] if cycles else []
    return {"points": points, "edges": selected, "pixels": pixels, "long_pixels": long_pixels,
            "all_pixels": _pixels(points, edges), "closed": bool(cycles),
            "chords": len(edges) - len(selected), "concave": any(t < 0 for t in turns) and any(t > 0 for t in turns)}


def _inside_canvas(pixels, n):
    return all(0 <= r < n and 0 <= c < n for r, c in pixels)


def _translate(pixels, delta):
    dr, dc = delta
    return {(r + dr, c + dc) for r, c in pixels}


def _analyze(a):
    n = a.shape[0]
    _need(a.shape == (n, n) and n >= 2, "input must be square, side at least two")
    colors, counts = np.unique(a, return_counts=True)
    background = int(colors[np.argmax(counts)])
    _need(int(max(counts)) > a.size // 2, "background must be a strict majority")
    groups = {int(c): [tuple(map(int, p)) for p in np.argwhere(a == c)]
              for c in colors if c != background}
    sources, others = [], []
    for color, points in groups.items():
        info = _topology(points)
        if info is None:
            others.append((color, points))
        else:
            info["color"] = color
            sources.append(info)
    _need(bool(sources), "no source drawing")
    source_pixels = set()
    for s in sources:
        _need(not source_pixels.intersection(s["pixels"]), "source/source collision")
        _need(any(a[p] == background for p in s["pixels"]), "source has no missing background cell")
        source_pixels.update(s["pixels"])
    markers, inert, marked = [], [], set()
    for color, points in others:
        associations = [(i, v, p) for p in points for i, s in enumerate(sources) for v in s["points"]
                        if max(abs(v[0] - p[0]), abs(v[1] - p[1])) == 1]
        if associations:
            _need(len(points) == len(associations) == 1, "ambiguous companion marker")
            i, v, p = associations[0]
            _need(i not in marked, "multiple markers on one source")
            marked.add(i)
            delta = (p[0] - v[0], p[1] - v[1])
            copy_pixels = _translate(sources[i]["long_pixels"], delta)
            _need(p in copy_pixels and _inside_canvas(copy_pixels, n), "invalid or clipped companion")
            _need(p not in source_pixels, "marker lies on a source stroke")
            markers.append({"color": color, "point": p, "source": i, "delta": delta, "pixels": copy_pixels})
        else:
            _need(not _candidate_edges(points), "inert landmarks must not form links")
            inert.extend(points)
    _need(not set(inert).intersection(source_pixels), "inert point lies on a source stroke")
    _need(all(not set(inert).intersection(m["pixels"]) for m in markers), "inert point lies on a companion")
    claimed = set()
    for m in markers:
        free = {p for p in m["pixels"] if a[p] == background and p not in source_pixels}
        _need(not free.intersection(claimed), "unresolved companion/companion collision")
        claimed.update(free)
    return {"background": background, "sources": sources, "markers": markers, "inert": inert}


def _render(a, info, copies=True, source_priority=True, all_alignments=False):
    out, background = a.copy(), info["background"]

    def sources():
        for s in info["sources"]:
            for p in s["all_pixels"] if all_alignments else s["pixels"]:
                if a[p] == background:
                    out[p] = s["color"]

    def companions():
        if copies:
            for m in info["markers"]:
                pixels = (_translate(info["sources"][m["source"]]["all_pixels"], m["delta"])
                          if all_alignments else m["pixels"])
                for p in pixels:
                    if 0 <= p[0] < a.shape[0] and 0 <= p[1] < a.shape[1] and a[p] == background:
                        out[p] = m["color"]

    if source_priority:
        companions()
        sources()
    else:
        sources()
        companions()
    return out


def _crop(a, parity=None):
    odd = a.shape[0] % 2 if parity is None else parity
    return a[:2, :2].copy() if odd else a[:2, -2:].copy()


def _bounding_boxes(a, info):
    """Plausible wrong model: replace each source by its bounding rectangle."""
    altered = dict(info)
    sources = []
    for s in info["sources"]:
        r0, r1 = min(r for r, c in s["points"]), max(r for r, c in s["points"])
        c0, c1 = min(c for r, c in s["points"]), max(c for r, c in s["points"])
        pixels = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                  if r in (r0, r1) or c in (c0, c1)}
        sources.append(dict(s, pixels=pixels, long_pixels=pixels))
    altered["sources"] = sources
    altered["markers"] = [dict(m, pixels=_translate(sources[m["source"]]["pixels"], m["delta"]))
                          for m in info["markers"]]
    return _render(a, altered)


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    return _crop(_render(a, _analyze(a)))


def _features(a, info=None, full=None):
    info = _analyze(a) if info is None else info
    full = _render(a, info) if full is None else full
    target = _crop(full)
    source_colors = {s["color"] for s in info["sources"]}
    raw = _crop(a)
    return {"side": a.shape[0], "background": info["background"], "sources": len(info["sources"]),
            "markers": len(info["markers"]), "inert": len(info["inert"]),
            "vertices": [len(s["points"]) for s in info["sources"]],
            "closed": sum(s["closed"] for s in info["sources"]),
            "concave": sum(s["concave"] for s in info["sources"]),
            "chords": sum(s["chords"] for s in info["sources"]),
            "diagonal": any(a0[0] != b0[0] and a0[1] != b0[1]
                            for s in info["sources"] for i, j in s["edges"]
                            for a0, b0 in [(s["points"][i], s["points"][j])]),
            "unit": any(max(abs(s["points"][i][0] - s["points"][j][0]),
                            abs(s["points"][i][1] - s["points"][j][1])) == 1
                        for s in info["sources"] for i, j in s["edges"]),
            "offsets": [m["delta"] for m in info["markers"]],
            "completion_matters": not np.array_equal(target, raw),
            "copy_matters": not np.array_equal(target, _crop(_render(a, info, copies=False))),
            "source_new": int(sum(raw[r, c] == info["background"] and int(target[r, c]) in source_colors
                                  for r in range(2) for c in range(2))),
            "priority_matters": not np.array_equal(target, _crop(_render(a, info, source_priority=False))),
            "chord_matters": not np.array_equal(target, _crop(_render(a, info, all_alignments=True))),
            "box_matters": not np.array_equal(target, _crop(_bounding_boxes(a, info))),
            "corners_differ": not np.array_equal(full[:2, :2], full[:2, -2:]),
            "both_corners_change": all(not np.array_equal(full[:2, c:c + 2], a[:2, c:c + 2])
                                       for c in (0, a.shape[0] - 2)),
            "palette": len(np.unique(a)), "output_colors": len(np.unique(target))}


def _ordered_edges(points, closed):
    return [(i, i + 1) for i in range(len(points) - 1)] + ([(len(points) - 1, 0)] if closed else [])


def _shape(rng, n, kind):
    """Procedural wide contour or open arch, not a completed parent puzzle."""
    top, left, right = int(rng.integers(2)), int(rng.integers(2)), n - 1 - int(rng.integers(2))
    bottom = int(rng.integers(max(4, n // 3), n))
    width = right - left
    if kind == "square":
        top, bottom = 0, n - 2
        left = int(rng.integers(2))
        right = left + n - 2
        return [(top, left), (top, right), (bottom, right), (bottom, left)], True
    if kind == "open":
        if rng.random() < .55:
            # An asymmetric valley has a diagonal at a reading corner. Its
            # bounding rectangle changes the answer, unlike an ordinary U.
            left = int(rng.integers(2))
            right = n - 1 - int(rng.integers(2))
            depth = int(rng.integers(3, max(4, right - left)))
            end = int(rng.integers(1, depth - 1))
            points = [(top, left), (top + depth, left + depth),
                      (top + depth, right), (top + end, right)]
            if rng.random() < .5:
                points = [(r, n - 1 - c) for r, c in points]
            return points, False
        h1, h2 = int(rng.integers(3, n)), int(rng.integers(3, n))
        a, b = int(rng.integers(3)), int(rng.integers(3))
        points = [(h1, left), (top + a, left), (top, left + a),
                  (top, right - b), (top + b, right), (h2, right)]
        closed = False
    else:
        maximum = max(1, min(3, (bottom - top) // 2, width // 3))
        a, b, c, d = [int(rng.integers(maximum + 1)) for _ in range(4)] if kind != "rectangle" else [0] * 4
        points = [(top, left + a), (top, right - b), (top + b, right),
                  (bottom - c, right), (bottom, right - c), (bottom, left + d),
                  (bottom - d, left), (top + a, left)]
        if kind == "notched" and width >= 6 and bottom - top >= 4:
            x = int(rng.integers(left + 2, right - 2))
            y = int(rng.integers(x + 1, right - 1))
            depth = int(rng.integers(1, bottom - top - 1))
            points[5:5] = [(bottom, y), (bottom - depth, y), (bottom - depth, x), (bottom, x)]
        closed = True
    # Remove duplicate corners where a chamfer has length zero.
    points = [p for i, p in enumerate(points) if p != points[i - 1]]
    return points, closed


def _make_object(points, closed, marker=None, delta=None):
    _need(len(points) == len(set(points)), "duplicate proposed vertex")
    intended = _ordered_edges(points, closed)
    info = _topology(points)
    _need(info is not None and set(info["edges"]) == {tuple(sorted(e)) for e in intended},
          "proposed geometry has another inferred order")
    # Unit links can occur, but no isolated unit-only stub needs a new copying convention.
    _need(info["pixels"] == info["long_pixels"], "unit-only stub excluded from generated inputs")
    return {"points": points, "pixels": _pixels(points, intended),
            "long_pixels": _pixels(points, intended, True), "marker": marker, "delta": delta,
            "all_pixels": info["all_pixels"], "closed": closed, "chords": info["chords"],
            "concave": info["concave"], "edges": intended}


def _marker_options(obj, n):
    result = []
    for delta in OFFSETS:
        copy_pixels = _translate(obj["long_pixels"], delta)
        if not _inside_canvas(copy_pixels, n):
            continue
        for v in obj["points"]:
            p = (v[0] + delta[0], v[1] + delta[1])
            if (p[0] >= 2 and p not in obj["pixels"] and p in copy_pixels
                    and sum(max(abs(p[0] - q[0]), abs(p[1] - q[1])) == 1 for q in obj["points"]) == 1):
                result.append((p, delta, copy_pixels))
    return result


def _paint(n, background, objects, palette):
    a = np.full((n, n), background, dtype=np.int8)
    source_info, marker_info = [], []
    for i, obj in enumerate(objects):
        color = int(palette[2 * i])
        for p in obj["points"]:
            _need(a[p] == background, "landmarks collide")
            a[p] = color
        s = dict(obj, color=color)
        source_info.append(s)
        if obj["marker"] is not None:
            p, delta = obj["marker"], obj["delta"]
            _need(a[p] == background, "marker collides")
            color = int(palette[2 * i + 1])
            a[p] = color
            marker_info.append({"color": color, "point": p, "source": i, "delta": delta,
                                "pixels": _translate(obj["long_pixels"], delta)})
    info = {"background": background, "sources": source_info, "markers": marker_info, "inert": []}
    return a, info


def _main_object(rng, n, kind, chord_witness=False, box_witness=False):
    for _ in range(110):
        try:
            points, closed = _shape(rng, n, kind)
            obj = _make_object(points, closed)
        except ValueError:
            continue
        options = _marker_options(obj, n)
        rng.shuffle(options)
        for p, delta, _ in options:
            obj["marker"], obj["delta"] = p, delta
            a, info = _paint(n, 0, [obj], [1, 2])
            f = _features(a, info)
            if (f["completion_matters"] and f["copy_matters"] and f["source_new"]
                    and f["corners_differ"] and f["both_corners_change"]
                    and (not box_witness or f["box_matters"])
                    and (not chord_witness or (f["chord_matters"] and f["priority_matters"]))):
                return obj
    # Guaranteed spacious witnesses, still inferred and validated from input.
    if kind == "open":
        depth = n - 3
        points = [(0, 0), (depth, depth), (depth, n - 2), (1, n - 2)]
        if n % 2:
            obj = _make_object(points, False, (depth, n - 1), (0, 1))
        else:
            points = [(r, n - 1 - c) for r, c in points]
            obj = _make_object(points, False, (depth, 0), (0, -1))
    else:
        left = 0 if n % 2 else 1
        points = [(0, left), (0, left + n - 2), (n - 2, left + n - 2), (n - 2, left)]
        delta = (0, 1) if n % 2 else (0, -1)
        marker = (n - 2, n - 1) if n % 2 else (n - 2, 0)
        obj = _make_object(points, True, marker, delta)
    return obj


def _extras(rng, n, main):
    objects = [main]
    blocked = main["pixels"] | _translate(main["long_pixels"], main["delta"])
    for _ in range(int(rng.integers(3))):
        for attempt in range(24):
            r, c = int(rng.integers(3, n - 1)), int(rng.integers(1, n - 1))
            dr, dc = OFFSETS[int(rng.integers(8))]
            length = int(rng.integers(2, max(3, n // 3)))
            points = [(r, c), (r + dr * length, c + dc * length)]
            if not _inside_canvas(points, n):
                continue
            obj = _make_object(points, False)
            marked = rng.random() < .5
            options = _marker_options(obj, n) if marked else [(None, None, set())]
            rng.shuffle(options)
            for p, delta, copy_pixels in options:
                occupied = obj["pixels"] | copy_pixels
                halo = {(a + x, b + y) for a, b in occupied for x in (-1, 0, 1) for y in (-1, 0, 1)}
                if not halo.intersection(blocked) and all(a >= 2 for a, _ in occupied):
                    obj["marker"], obj["delta"] = p, delta
                    objects.append(obj)
                    blocked.update(occupied)
                    break
            else:
                continue
            break
    return objects


def _pair(rng, n, kind, chord_witness=False, box_witness=False):
    main = _main_object(rng, n, kind, chord_witness, box_witness)
    objects = _extras(rng, n, main)
    palette = [int(x) for x in rng.permutation(10)]
    background = palette.pop()
    a, intended = _paint(n, background, objects, palette)
    # Inert colors are singleton points away from every vertex and full stroke.
    used = {int(x) for x in np.unique(a)}
    free_colors = [c for c in palette if c not in used]
    footprint = set().union(*(s["pixels"] for s in objects),
                            *(m["pixels"] for m in intended["markers"]))
    vertices = [v for s in objects for v in s["points"]]
    candidates = [(r, c) for r in range(2, n) for c in range(n)
                  if (r, c) not in footprint and a[r, c] == background
                  and all(max(abs(r - v[0]), abs(c - v[1])) > 1 for v in vertices)]
    rng.shuffle(candidates)
    count = min(int(rng.integers(3)), len(free_colors), len(candidates))
    for p, color in zip(candidates[:count], free_colors):
        a[p] = color
    expected_full = _render(a, intended)
    derived = _analyze(a)
    _need(np.array_equal(expected_full, _render(a, derived)), "geometry/solver disagreement")
    return {"input": a, "output": _crop(expected_full)}


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _fingerprint(a):
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    nt, nq = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    # Both branches even for two demonstrations; size and parity do not covary.
    first_parity = int(rng.integers(2))
    if nt >= 3:
        # An alternating parity sequence by size contradicts both directions of
        # a single size threshold. Order the witnesses independently of size.
        length = min(nt, 4)
        selected = None
        for _ in range(40):
            trial = sorted(int(x) for x in rng.choice(np.arange(6, 19), length, replace=False))
            if all(a % 2 != b % 2 for a, b in zip(trial, trial[1:])):
                selected = trial
                break
        if selected is None:
            start = int(rng.integers(6, 20 - length))
            selected = list(range(start, start + length))
        pools = {p: [n for n in selected if n % 2 == p] for p in (0, 1)}
        for pool in pools.values():
            rng.shuffle(pool)
        train_sides = [pools[first_parity].pop(), pools[1 - first_parity].pop()]
        remaining = pools[0] + pools[1]
        rng.shuffle(remaining)
        train_sides.extend(remaining)
    else:
        train_sides = [int(rng.choice([n for n in range(6, 19) if n % 2 == p]))
                       for p in (first_parity, 1 - first_parity)]
    while len(train_sides) < nt:
        parity = int(rng.integers(2))
        choices = [n for n in range(6, 19) if n % 2 == parity and n not in train_sides]
        train_sides.append(int(rng.choice(choices or [n for n in range(6, 19) if n % 2 == parity])))
    test_parities = [int(rng.integers(2))] if nq == 1 else [0, 1] + [int(rng.integers(2)) for _ in range(nq - 2)]
    rng.shuffle(test_parities)
    large_shift = bool(rng.integers(2))
    test_sides = []
    for parity in test_parities:
        choices = [n for n in (range(20, 31) if large_shift else range(6, 20))
                   if n % 2 == parity and n not in train_sides]
        if not choices:
            choices = [n for n in range(6, 31) if n % 2 == parity and n not in train_sides]
        test_sides.append(int(rng.choice(choices)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, sides in (("train", train_sides), ("test", test_sides)):
        for i, n in enumerate(sides):
            kind = "square" if split == "train" and i == 0 else "open" if split == "train" and i == 1 else str(
                rng.choice(["rectangle", "open", "chamfered", "notched", "square"]))
            for _ in range(12):
                pair = _pair(rng, n, kind, split == "train" and i == 0, split == "train" and i == 1)
                key = _fingerprint(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("unique input budget exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check semantics, necessity witnesses, topology, and visible conventions."""
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _need(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid counts")
    records, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            _need(out.shape == (2, 2), "output must be 2x2")
            key = _fingerprint(a)
            _need(key not in seen, "duplicate input")
            seen.add(key)
            info = _analyze(a)
            full = _render(a, info)
            _need(np.array_equal(out, _crop(full)), "wrong completed corner")
            _need(np.array_equal(full[a != info["background"]], a[a != info["background"]]), "existing cells changed")
            f = _features(a, info, full)
            _need(f["completion_matters"] and f["copy_matters"] and f["source_new"], "A has no effect in selected crop")
            _need(f["corners_differ"] and f["both_corners_change"], "B or decoy corner is uninformative")
            _need(all(m["point"][0] >= 2 for m in info["markers"]), "marker may not supply the output directly")
            _need(all(s["pixels"] == s["long_pixels"] for s in info["sources"]), "unit-only stub not demonstrated")
            records[split].append(f)
    train = records["train"]
    _need({f["side"] % 2 for f in train} == {0, 1}, "both parity branches need demonstrations")
    _need(any(f["chord_matters"] for f in train), "contour/chord distinction not visible")
    _need(any(f["priority_matters"] for f in train), "source priority not demonstrated")
    _need(any(f["box_matters"] for f in train), "bounding rectangle shortcut not distinguished")
    # The open-vs-closed distinction itself is visible in source landmark topology.
    _need(any(f["closed"] < f["sources"] for f in train), "open chain demonstration missing")
    if nt >= 4:
        _need(all(len({f["side"] for f in train if f["side"] % 2 == p}) >= 2 for p in (0, 1)),
              "need two sizes of each parity")
    if nt >= 3:
        sizes = sorted({f["side"] for f in train})
        _need(sum(a % 2 != b % 2 for a, b in zip(sizes, sizes[1:])) >= 2,
              "training permits a single size threshold")
    _need(not {f["side"] for f in train}.intersection(f["side"] for f in records["test"]),
          "test should require an unseen size")
    if nq >= 2:
        _need({f["side"] % 2 for f in records["test"]} == {0, 1}, "multi-test parity coverage")


def _puzzle_digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 seeds, streaming bounded diagnostics, count boundaries and mutations."""
    # Hand-worked corner regressions exercise the order of completion, priority,
    # and extraction without constructing labels through solve.
    odd = np.zeros((7, 7), dtype=np.int8)
    for p in ((0, 0), (0, 5), (5, 5), (5, 0)):
        odd[p] = 2
    odd[6, 0] = 3
    _need(np.array_equal(solve(odd), [[2, 2], [2, 3]]), "odd corner regression")
    even = np.zeros((8, 8), dtype=np.int8)
    for p in ((0, 0), (0, 6), (6, 6), (6, 0)):
        even[p] = 2
    even[6, 7] = 3
    _need(np.array_equal(solve(even), [[2, 3], [2, 3]]), "even corner regression")
    _need(not np.array_equal(solve(odd), _crop(_render(odd, _analyze(odd), all_alignments=True))),
          "corner regression must reject a diagonal chord")
    permutation = np.array([8, 4, 0, 9, 3, 6, 5, 2, 1, 7], dtype=np.int8)
    _need(np.array_equal(solve(permutation[odd]), permutation[np.array([[2, 2], [2, 3]])]),
          "literal color roles leaked")
    completed = _render(odd, _analyze(odd))
    _need(np.array_equal(solve(np.fliplr(odd)), _crop(np.fliplr(completed))), "reflection regression")
    hashes, shapes, structures, backgrounds, offsets, counts, palettes = (set() for _ in range(7))
    stats = {"pairs": 0, "chord_witnesses": 0, "priority_witnesses": 0, "box_witnesses": 0, "diagonal_pairs": 0,
             "unit_pairs": 0, "concave_pairs": 0, "open_pairs": 0, "test_odd": 0, "test_even": 0,
             "smaller_tests": 0, "larger_tests": 0}
    for seed in range(42):
        p = generate_puzzle(seed)
        hashes.add(_puzzle_digest(p))
        train_max = max(x["input"].shape[0] for x in p["train"])
        for split in ("train", "test"):
            for pair in p[split]:
                a, out = pair["input"], pair["output"]
                _need(np.array_equal(solve(a), out), "solve mismatch")
                f = _features(a)
                shapes.add(f["side"])
                backgrounds.add(f["background"])
                offsets.update(f["offsets"])
                counts.add(f["sources"])
                palettes.add(f["palette"])
                roles = {}
                canonical = bytes(roles.setdefault(int(c), len(roles)) for c in a.flat)
                structures.add(hashlib.sha256(bytes(a.shape) + canonical).hexdigest())
                stats["pairs"] += 1
                for k, feature in (("chord_witnesses", "chord_matters"), ("priority_witnesses", "priority_matters"),
                                   ("box_witnesses", "box_matters"), ("diagonal_pairs", "diagonal"),
                                   ("unit_pairs", "unit"), ("concave_pairs", "concave")):
                    stats[k] += bool(f[feature])
                stats["open_pairs"] += f["closed"] < f["sources"]
                if split == "test":
                    stats["test_odd" if f["side"] % 2 else "test_even"] += 1
                    stats["smaller_tests"] += f["side"] <= train_max
                    stats["larger_tests"] += f["side"] > train_max
    _need(len(hashes) == 42 and len(structures) >= 150, "structural variation collapsed")
    _need(len(shapes) >= 18 and backgrounds == set(range(10)), "scale/background coverage")
    _need(offsets == set(OFFSETS) and len(counts) >= 2 and len(palettes) >= 4, "geometry/palette/count coverage")
    _need(all(stats[k] > 0 for k in stats), "missing family-specific coverage")
    a = generate_puzzle(314159, 7, 3)
    _need(_puzzle_digest(a) == _puzzle_digest(generate_puzzle(314159, 7, 3)), "not reproducible")
    _need(_puzzle_digest(a) != _puzzle_digest(generate_puzzle(314160, 7, 3)), "seeds collapse")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            generate_puzzle(7000 + 10 * nt + nq, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                   (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    p = generate_puzzle(0)
    original = p["test"][0]["output"].copy()
    p["test"][0]["output"][0, 0] = (int(original[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    stats.update({"seeds": 42, "distinct_puzzles": len(hashes), "normalized_structures": len(structures),
                  "sides": sorted(shapes), "backgrounds": sorted(backgrounds), "offsets": sorted(offsets),
                  "source_counts": sorted(counts), "palette_sizes": sorted(palettes)})
    return stats
