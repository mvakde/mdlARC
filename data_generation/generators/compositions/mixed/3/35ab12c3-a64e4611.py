"""Complete translated drawings, then expose all locally empty background.

Only standard Python and NumPy are used. There is no file I/O, retained puzzle
state, parent import, or seed access in solve(). Highlight color 3 is fixed.
"""
from __future__ import annotations

import hashlib
import numpy as np


HIGHLIGHT = 3
OFFSETS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _segment(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    n = max(abs(dr), abs(dc))
    if not n or not (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
        raise ValueError("unaligned endpoints")
    sr, sc = int(dr > 0) - int(dr < 0), int(dc > 0) - int(dc < 0)
    return {(a[0] + k * sr, a[1] + k * sc) for k in range(n + 1)}


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _intersect(a, b, c, d):
    if (max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0])
            or max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1])):
        return False
    return _cross(a, b, c) * _cross(a, b, d) <= 0 and _cross(c, d, a) * _cross(c, d, b) <= 0


def _infer(points):
    """Unique simple full cycle first, otherwise unique simple full path.

    Edge-set identity removes reversal/start-point duplicates. Continuous
    intersection rejects crossings between diagonal pixels as well as on them.
    Ambiguous point sets are outside this family and raise ValueError.
    """
    points = tuple(sorted(points))
    n = len(points)
    if not 2 <= n <= 8:
        raise ValueError("a source needs 2..8 landmarks")
    point_set, edges, rasters = set(points), [], []
    for i, a in enumerate(points):
        for j in range(i + 1, n):
            b = points[j]
            dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
            if not (dr == 0 or dc == 0 or dr == dc):
                continue
            line = _segment(a, b)
            if not (line - {a, b}) & point_set:
                edges.append((i, j))
                rasters.append(line)
    adjacent = [[] for _ in points]
    for k, (i, j) in enumerate(edges):
        adjacent[i].append((j, k))
        adjacent[j].append((i, k))
    if not all(adjacent):
        raise ValueError("disconnected landmarks")
    incompatible = [set() for _ in edges]
    for k, (a, b) in enumerate(edges):
        for l, (c, d) in enumerate(edges[:k]):
            if not {a, b} & {c, d} and _intersect(points[a], points[b], points[c], points[d]):
                incompatible[k].add(l)
                incompatible[l].add(k)

    def enumerate_drawings(closed):
        answers = {}
        visits = 0

        def extend(order, visited, chosen):
            nonlocal visits
            visits += 1
            if visits > 50000:
                raise ValueError("topology search exceeds family budget")
            if len(answers) > 1:
                return
            if len(order) == n:
                if closed:
                    closing = [e for v, e in adjacent[order[-1]] if v == order[0]]
                    if not closing or incompatible[closing[0]] & chosen:
                        return
                    chosen = chosen | {closing[0]}
                answers[tuple(sorted(chosen))] = tuple(order)
                return
            for v, e in adjacent[order[-1]]:
                if v not in visited and not incompatible[e] & chosen:
                    extend(order + [v], visited | {v}, chosen | {e})

        for start in ([0] if closed else range(n)):
            extend([start], {start}, set())
            if len(answers) > 1:
                break
        return answers

    cycles = enumerate_drawings(True) if n > 2 else {}
    if len(cycles) > 1:
        raise ValueError("competing full contours")
    answers = cycles or enumerate_drawings(False)
    if len(answers) != 1:
        raise ValueError("no unique full drawing")
    chosen, order = next(iter(answers.items()))
    selected = [edges[k] for k in chosen]
    pixels = set().union(*(rasters[k] for k in chosen))
    lengths = [max(abs(points[i][0] - points[j][0]), abs(points[i][1] - points[j][1]))
               for i, j in selected]
    # Generated links are all nontrivial. Unit links are supported as existing
    # topology, but only pixels belonging to nontrivial links are copied (A v2).
    long_pixels = set().union(*(rasters[k] for k in chosen if len(rasters[k]) > 2))
    if not long_pixels:
        raise ValueError("no missing nontrivial link")
    turns = [_cross(points[order[i - 1]], points[order[i]], points[order[(i + 1) % n]])
             for i in range(n)] if cycles else []
    return {"points": points, "edges": selected, "pixels": pixels, "long": long_pixels,
            "kind": "cycle" if cycles else "path", "lengths": lengths,
            "extra": set().union(*rasters) - pixels,
            "concave": any(t < 0 for t in turns) and any(t > 0 for t in turns)}


def _completed(inp):
    if np.any(inp == HIGHLIGHT):
        raise ValueError("highlight color is absent from inputs")
    values, counts = np.unique(inp, return_counts=True)
    bg = int(values[np.argmax(counts)])
    if counts.max() * 2 <= inp.size:
        raise ValueError("background must be a strict majority")
    groups = {int(c): [tuple(map(int, p)) for p in np.argwhere(inp == c)]
              for c in values if c != bg}
    sources = []
    for color, points in groups.items():
        if len(points) >= 2:
            source = _infer(points)
            source["color"] = color
            if not any(inp[p] == bg for p in source["pixels"]):
                raise ValueError("source has no missing cells")
            sources.append(source)
    if not sources:
        raise ValueError("no source drawing")
    source_pixels = set()
    for source in sources:
        if source_pixels & source["pixels"]:
            raise ValueError("different source strokes intersect")
        source_pixels |= source["pixels"]
        if any(inp[p] not in (bg, source["color"]) for p in source["pixels"]):
            raise ValueError("a pre-existing marker obstructs a source stroke")
    markers, marked = [], set()
    for color, points in groups.items():
        if len(points) != 1:
            continue
        p = points[0]
        associations = [(i, v) for i, s in enumerate(sources) for v in s["points"]
                        if max(abs(p[0] - v[0]), abs(p[1] - v[1])) == 1]
        if len(associations) != 1:
            raise ValueError("marker must have exactly one adjacent source vertex")
        i, anchor = associations[0]
        if i in marked:
            raise ValueError("multiple markers on one source")
        marked.add(i)
        offset = p[0] - anchor[0], p[1] - anchor[1]
        copy = {(r + offset[0], c + offset[1]) for r, c in sources[i]["long"]}
        if p not in copy or not all(0 <= r < inp.shape[0] and 0 <= c < inp.shape[1] for r, c in copy):
            raise ValueError("complete translated links must fit the canvas")
        if copy & (source_pixels - sources[i]["pixels"]):
            raise ValueError("copy intersects a different construction")
        markers.append({"color": color, "point": p, "source": i, "offset": offset, "copy": copy})
    if not markers:
        raise ValueError("at least one companion is required")
    claimed = set()
    completed = inp.copy()
    for m in markers:
        if claimed & m["copy"]:
            raise ValueError("different companion copies intersect")
        claimed |= m["copy"]
        for p in m["copy"]:
            if inp[p] == bg and p not in source_pixels:
                completed[p] = m["color"]
    for s in sources:
        for p in s["pixels"]:
            if inp[p] == bg:
                completed[p] = s["color"]
    return completed, {"background": bg, "sources": sources, "markers": markers}


def _eligible(foreground):
    h, w = foreground.shape
    padded = np.pad(foreground, 1, constant_values=False)
    near = np.zeros((h, w), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            near |= padded[dr:dr + h, dc:dc + w]
    return ~near


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only reconstruction followed by one simultaneous local selection."""
    inp = _grid(input_grid)
    completed, info = _completed(inp)
    out = completed.copy()
    out[_eligible(completed != info["background"])] = HIGHLIGHT
    return out


def _transform(points, code):
    result = []
    for r, c in points:
        if code & 4:
            c = -c
        for _ in range(code % 4):
            r, c = c, -r
        result.append((r, c))
    lo_r, lo_c = min(r for r, c in result), min(c for r, c in result)
    return [(r - lo_r, c - lo_c) for r, c in result]


def _shape(rng, kind, small=False):
    """Parametric contours and filtered random walks, never parent puzzles."""
    for _ in range(100):
        a, b = (int(rng.integers(3, 7 if small else 10)) for _ in range(2))
        closed = kind not in ("segment", "elbow", "walk")
        if kind == "segment":
            points = [(0, 0), (a, a)] if rng.random() < .35 else [(0, 0), (0, a)]
        elif kind == "elbow":
            points = [(0, 0), (0, a), (b, a)]
        elif kind == "walk":
            points = [(0, 0)]
            for _ in range(int(rng.integers(2, 5 if small else 6))):
                dr, dc = OFFSETS[int(rng.integers(8))]
                length = int(rng.integers(2, 5 if small else 7))
                points.append((points[-1][0] + dr * length, points[-1][1] + dc * length))
        elif kind in ("rectangle", "square"):
            if kind == "square":
                a = b = int(rng.integers(5, 7))
            points = [(0, 0), (0, a), (b, a), (b, 0)]
        elif kind == "diamond":
            d = int(rng.integers(3, 5 if small else 7))
            points = [(0, d), (d, 2 * d), (2 * d, d), (d, 0)]
        elif kind == "bevel":
            h, w = (int(rng.integers(8, 11 if small else 15)) for _ in range(2))
            cuts = [int(rng.integers(2, 4)) for _ in range(4)]
            x, y, z, t = cuts
            points = [(0, x), (0, w - y), (y, w), (h - z, w),
                      (h, w - z), (h, t), (h - t, 0), (x, 0)]
        elif kind == "notch":
            x, gap, tail = (int(rng.integers(2, 4)) for _ in range(3))
            h = int(rng.integers(6, 10 if small else 14))
            depth = int(rng.integers(2, h - 2))
            points = [(0, 0), (0, x), (depth, x), (depth, x + gap),
                      (0, x + gap), (0, x + gap + tail), (h, x + gap + tail), (h, 0)]
        else:
            raise ValueError("unknown shape kind")
        if len(set(points)) != len(points):
            continue
        points = _transform(points, int(rng.integers(8)))
        if max(max(r for r, c in points), max(c for r, c in points)) > (12 if small else 22):
            continue
        intended_edges = list(zip(range(len(points) - 1), range(1, len(points))))
        if closed:
            intended_edges.append((len(points) - 1, 0))
        if any(len(_segment(points[i], points[j])) < 3 for i, j in intended_edges):
            continue
        intended = set().union(*(_segment(points[i], points[j]) for i, j in intended_edges))
        try:
            inferred = _infer(points)
        except ValueError:
            continue
        intended_links = {frozenset((points[i], points[j])) for i, j in intended_edges}
        inferred_links = {frozenset((inferred["points"][i], inferred["points"][j]))
                          for i, j in inferred["edges"]}
        if inferred_links == intended_links and inferred["kind"] == ("cycle" if closed else "path"):
            return points, intended
    raise ValueError("bounded shape sampling exhausted")


def _object(rng, points, pixels, offsets, marked=True, overlap=False):
    if not marked:
        return {"points": points, "pixels": pixels, "copy": set(), "marker": None}
    choices = []
    for dr, dc in offsets:
        copy = {(r + dr, c + dc) for r, c in pixels}
        for v in points:
            p = v[0] + dr, v[1] + dc
            if p in pixels or sum(max(abs(p[0] - q[0]), abs(p[1] - q[1])) == 1 for q in points) != 1:
                continue
            if len(copy - pixels - {p}) < 2:
                continue
            if overlap and not copy & (pixels - set(points)):
                continue
            choices.append((copy, p))
    if not choices:
        raise ValueError("no visible unambiguous companion")
    copy, marker = choices[int(rng.integers(len(choices)))]
    return {"points": points, "pixels": pixels, "copy": copy, "marker": marker}


def _shift(obj, dr, dc):
    result = {k: {(r + dr, c + dc) for r, c in obj[k]} for k in ("pixels", "copy")}
    result["points"] = [(r + dr, c + dc) for r, c in obj["points"]]
    result["marker"] = None if obj["marker"] is None else (obj["marker"][0] + dr, obj["marker"][1] + dc)
    return result


def _halo(points, radius=1):
    return {(r + dr, c + dc) for r, c in points
            for dr in range(-radius, radius + 1) for dc in range(-radius, radius + 1)}


def _place(rng, objects, nested=False, near=False):
    placed, occupied = [], set()
    for index, obj in enumerate(objects):
        footprint = obj["pixels"] | obj["copy"]
        lo_r, lo_c = min(r for r, c in footprint), min(c for r, c in footprint)
        h, w = max(r for r, c in footprint) - lo_r + 1, max(c for r, c in footprint) - lo_c + 1
        if h > 30 or w > 30:
            raise ValueError("footprint too large")
        close_zone = _halo(occupied, 2) if near else set()
        for _ in range(200):
            dr, dc = int(rng.integers(31 - h)) - lo_r, int(rng.integers(31 - w)) - lo_c
            shifted = {(r + dr, c + dc) for r, c in footprint}
            if shifted & occupied or (index and near and not shifted & close_zone):
                continue
            if nested and index == 1:
                outer = placed[0]["pixels"]
                if not all(min(r for r, c in outer) + 2 <= r <= max(r for r, c in outer) - 2
                           and min(c for r, c in outer) + 2 <= c <= max(c for r, c in outer) - 2
                           for r, c in shifted):
                    continue
            placed.append(_shift(obj, dr, dc))
            occupied |= shifted
            break
        else:
            raise ValueError("bounded placement exhausted")
    return placed


def _render(rng, objects, bg, diagnostic=False):
    occupied = set().union(*(o["pixels"] | o["copy"] for o in objects))
    low = [min(p[axis] for p in occupied) for axis in (0, 1)]
    extent = [max(p[axis] for p in occupied) - low[axis] + 1 for axis in (0, 1)]
    sizes, before = [], []
    for length in extent:
        if diagnostic:
            left, right = int(rng.integers(2, 6)), int(rng.integers(2, 6))
        else:
            extra = 30 - length if rng.random() < .18 else int(rng.integers(min(10, 30 - length) + 1))
            left = int(rng.integers(extra + 1))
            right = extra - left
        sizes.append(length + left + right)
        before.append(left)
    objects = [_shift(o, before[0] - low[0], before[1] - low[1]) for o in objects]
    palette = [int(c) for c in rng.permutation(10) if c not in (bg, HIGHLIGHT)]
    inp = np.full(tuple(sizes), bg, dtype=np.int8)
    complete = inp.copy()
    # Outputs are rendered from the sampled ordered geometry, not solve().
    # No construction footprints overlap; within one object source wins.
    for obj in objects:
        color = palette.pop()
        for p in obj["points"]:
            inp[p] = color
        for p in obj["pixels"]:
            complete[p] = color
        if obj["marker"] is not None:
            companion = palette.pop()
            inp[obj["marker"]] = companion
            for p in obj["copy"] - obj["pixels"]:
                complete[p] = companion
    out = complete.copy()
    out[_eligible(complete != bg)] = HIGHLIGHT
    return {"input": inp, "output": out}


def _components(mask):
    unseen = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while unseen:
        start = unseen.pop()
        component, todo = {start}, [start]
        while todo:
            r, c = todo.pop()
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in unseen:
                    unseen.remove(p)
                    component.add(p)
                    todo.append(p)
        result.append(component)
    return result


def _measure(inp, completed, info):
    bg = info["background"]
    foreground = completed != bg
    eligible = _eligible(foreground)
    remaining = ~foreground
    source_only = inp.copy()
    for s in info["sources"]:
        for p in s["pixels"]:
            source_only[p] = s["color"]
    copy_only = inp.copy()
    for m in info["markers"]:
        for p in m["copy"]:
            if inp[p] == bg:
                copy_only[p] = m["color"]
    axial_near = foreground.copy()
    axial_near[1:] |= foreground[:-1]
    axial_near[:-1] |= foreground[1:]
    axial_near[:, 1:] |= foreground[:, :-1]
    axial_near[:, :-1] |= foreground[:, 1:]
    halo_counts = np.zeros(inp.shape, dtype=np.int8)
    for i, s in enumerate(info["sources"]):
        footprint = s["pixels"] | set().union(*(m["copy"] for m in info["markers"] if m["source"] == i))
        for r, c in _halo(footprint):
            if 0 <= r < inp.shape[0] and 0 <= c < inp.shape[1]:
                halo_counts[r, c] += 1
    return {
        "shape": list(inp.shape), "background": bg, "sources": len(info["sources"]),
        "markers": len(info["markers"]), "offsets": [list(m["offset"]) for m in info["markers"]],
        "vertices": [len(s["points"]) for s in info["sources"]],
        "kinds": [s["kind"] for s in info["sources"]],
        "concave": sum(s["concave"] for s in info["sources"]),
        "rejected_chord_cells": sum(len(s["extra"]) for s in info["sources"]),
        "highlight": int(eligible.sum()), "highlight_components": len(_components(eligible)),
        "small_pockets": sum(1 <= len(c) <= 4 and all(0 < r < inp.shape[0] - 1 and 0 < q < inp.shape[1] - 1 for r, q in c)
                             for c in _components(eligible)),
        "edge_highlight": int(eligible[0].sum() + eligible[-1].sum() + eligible[:, 0].sum() + eligible[:, -1].sum()),
        "diagonal_fringe": int((remaining & ~axial_near & ~eligible).sum()),
        "completion_fringe_changes": int((remaining & (_eligible(inp != bg) != eligible)).sum()),
        "source_fringe_changes": int((remaining & (_eligible(copy_only != bg) != eligible)).sum()),
        "copy_fringe_changes": int((remaining & (_eligible(source_only != bg) != eligible)).sum()),
        "visible_copy_cells": sum(int(np.count_nonzero((completed == m["color"]) & (inp == bg))) for m in info["markers"]),
        "source_priority_cells": sum(len(m["copy"] & (info["sources"][m["source"]]["pixels"] - set(info["sources"][m["source"]]["points"]))) for m in info["markers"]),
        "shared_fringe": int((remaining & (halo_counts > 1)).sum()),
        "touches_boundary": bool(np.any(foreground[0]) or np.any(foreground[-1]) or np.any(foreground[:, 0]) or np.any(foreground[:, -1])),
    }


def _build_pair(rng, bg, allowed, mode, held=None):
    choices = ("segment", "elbow", "walk", "rectangle", "square", "diamond", "bevel", "notch")
    for _ in range(100):
        try:
            diagnostic = mode == "pocket"
            count = 1 if diagnostic else 2 if mode == "open" else int(rng.integers(2 if mode == "test" else 1, 4))
            nested = not diagnostic and mode != "open" and count == 2 and rng.random() < .25
            objects = []
            for index in range(count):
                if diagnostic:
                    kind = "square"
                elif mode == "open":
                    kind = "walk" if index == 0 else "segment"
                elif nested and index == 0:
                    kind = "rectangle"
                else:
                    kind = str(rng.choice(choices))
                if nested and index == 0:
                    h, w = int(rng.integers(17, 24)), int(rng.integers(17, 24))
                    points = [(0, 0), (0, w), (h, w), (h, 0)]
                    pixels = set().union(*(_segment(points[i], points[(i + 1) % 4]) for i in range(4)))
                else:
                    points, pixels = _shape(rng, kind, small=count > 1)
                offsets = (held,) if mode == "test" and index == 0 else allowed
                if diagnostic:
                    offsets = tuple(d for d in allowed if d[0] and d[1])
                marked = index == 0 or (mode != "open" and rng.random() < .65)
                objects.append(_object(rng, points, pixels, offsets, marked, overlap=diagnostic))
            placed = _place(rng, objects, nested=nested, near=mode in ("open", "test") and not nested)
            pair = _render(rng, placed, bg, diagnostic)
            complete, info = _completed(pair["input"])
            stats = _measure(pair["input"], complete, info)
            if not all(stats[k] > 0 for k in ("highlight", "source_fringe_changes", "copy_fringe_changes", "diagonal_fringe")):
                continue
            if diagnostic and not (stats["small_pockets"] and stats["source_priority_cells"] and stats["edge_highlight"]):
                continue
            if mode == "test" and not stats["shared_fringe"]:
                continue
            if not np.array_equal(solve(pair["input"]), pair["output"]):
                raise AssertionError("sampled geometry disagrees with input reconstruction")
            return pair
        except ValueError:
            continue
    raise RuntimeError("bounded scene generation exhausted")


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(a):
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    nt, nq = _count(num_train, 2, 10, "num_train"), _count(num_test, 1, 4, "num_test")
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    held = OFFSETS[int(rng.integers(8))]
    allowed = tuple(d for d in OFFSETS if d != held)
    backgrounds = [int(c) for c in rng.permutation([c for c in range(10) if c != HIGHLIGHT])]
    puzzle, seen = {"train": [], "test": []}, set()
    first_offset = None
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            mode = ("pocket" if index == 0 else "open" if index == 1 else "random") if split == "train" else "test"
            pair_allowed = tuple(d for d in allowed if d != first_offset) if mode == "open" else allowed
            for _ in range(12):
                bg = backgrounds[index % len(backgrounds)] if split == "train" else int(rng.choice(backgrounds))
                pair = _build_pair(rng, bg, pair_allowed, mode, held=held)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("distinct input generation exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
            if mode == "pocket":
                _, info = _completed(pair["input"])
                first_offset = info["markers"][0]["offset"]
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, measured, offsets = set(), {"train": [], "test": []}, {"train": set(), "test": set()}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            complete, info = _completed(inp)
            bg = info["background"]
            assert np.array_equal(out[inp != bg], inp[inp != bg]), "input foreground changed"
            assert np.array_equal(out[complete != bg], complete[complete != bg]), "completed foreground changed"
            # Scalar clipped windows audit the vectorized neighborhood kernel.
            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    window = complete[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
                    expected = HIGHLIGHT if np.all(window == bg) else int(complete[r, c])
                    assert int(out[r, c]) == expected, "incorrect clipped local selection"
            assert np.array_equal(solve(inp), out)
            stats = _measure(inp, complete, info)
            assert all(stats[k] > 0 for k in ("highlight", "source_fringe_changes", "copy_fringe_changes", "visible_copy_cells", "diagonal_fringe"))
            assert np.any(out == bg), "one-cell fringe must remain"
            measured[split].append(stats)
            offsets[split].update(m["offset"] for m in info["markers"])
    train = measured["train"]
    assert any(s["small_pockets"] and s["rejected_chord_cells"] and s["source_priority_cells"] for s in train)
    assert any(s["edge_highlight"] for s in train)
    assert any("path" in s["kinds"] and s["sources"] > s["markers"] for s in train)
    assert len({s["background"] for s in train}) >= 2
    assert len(offsets["train"]) >= 2
    assert offsets["test"] - offsets["train"], "query must include a held-out displacement"
    assert all(s["sources"] >= 2 and s["shared_fringe"] > 0 for s in measured["test"])


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 streamed seeds; return small measurements without writing any files."""
    assert np.all(_eligible(np.zeros((2, 4), dtype=bool))), "outside-grid cells are ignored"
    diagonal = np.zeros((4, 4), dtype=bool)
    diagonal[1, 1] = True
    assert not _eligible(diagonal)[0, 0] and _eligible(diagonal)[0, 3]
    signatures, shapes, geometries, offsets, backgrounds, kinds, counts = (set() for _ in range(7))
    concave = boundary = small = shared = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                complete, info = _completed(inp)
                m = _measure(inp, complete, info)
                shapes.add(inp.shape)
                backgrounds.add(info["background"])
                counts.add(len(info["sources"]))
                offsets.update(x["offset"] for x in info["markers"])
                for s in info["sources"]:
                    points = _transform(s["points"], 0)
                    geometries.add((s["kind"], tuple(sorted(points))))
                    kinds.add(s["kind"])
                concave += m["concave"]
                boundary += m["touches_boundary"]
                small += m["small_pockets"]
                shared += m["shared_fringe"] > 0
                # The rule must commute with all rotations and reflections.
                # One transformation per seed keeps this regression bounded.
                if split == "test":
                    # Cycle through all eight dihedral transforms across seeds.
                    a = np.rot90(inp, (seed // 2) % 4)
                    b = np.rot90(pair["output"], (seed // 2) % 4)
                    if seed % 2:
                        a, b = np.fliplr(a), np.fliplr(b)
                    assert np.array_equal(solve(a), b)
    assert len(signatures) == 42 and len(shapes) >= 40 and len(geometries) >= 80
    assert offsets == set(OFFSETS) and backgrounds == set(range(10)) - {HIGHLIGHT}
    assert counts == {1, 2, 3} and kinds == {"path", "cycle"}
    assert concave > 0 and boundary > 0 and small >= 42 and shared >= 42
    assert _signature(generate_puzzle(813, 5, 2)) == _signature(generate_puzzle(813, 5, 2))
    for nt, nq in ((2, 1), (10, 1), (10, 4), (5, 4)):
        p = generate_puzzle(900 + nt + nq, nt, nq)
        assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    p = generate_puzzle(0, 2, 1)
    pair = p["test"][0]
    saved = pair["output"].copy()
    complete, info = _completed(pair["input"])
    for mask, value in ((saved == HIGHLIGHT, info["background"]),
                        (saved == info["background"], HIGHLIGHT),
                        ((complete != info["background"]) & (pair["input"] == info["background"]), HIGHLIGHT)):
        r, c = np.argwhere(mask)[0]
        pair["output"] = saved.copy()
        pair["output"][r, c] = value
        try:
            validate_generated_puzzle(p)
        except AssertionError:
            continue
        raise AssertionError("incorrect output escaped validation")
    return {"seeds": 42, "distinct_puzzles": len(signatures),
            "canvas_shapes": len(shapes), "geometry_signatures": len(geometries),
            "offsets": [list(d) for d in sorted(offsets)],
            "backgrounds": sorted(backgrounds), "source_counts": sorted(counts),
            "kinds": sorted(kinds), "concave_drawings": concave,
            "boundary_pairs": boundary, "small_pockets": small,
            "shared_fringe_pairs": shared}

