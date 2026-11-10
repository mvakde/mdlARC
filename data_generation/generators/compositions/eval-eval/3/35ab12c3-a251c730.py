"""Stencil-lined contours: infer closed strokes, then sweep keyed masks inward.

No files, parent generators, seed, or training examples are used by solve().
All diagnostics are in memory; self_test retains only bounded feature sets.
"""
from collections import deque
import hashlib
import numpy as np

_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _D4:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        groups.append(cells)
    return groups


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _segment(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    assert (dr or dc) and (dr == 0 or dc == 0 or abs(dr) == abs(dc))
    sr, sc = int(np.sign(dr)), int(np.sign(dc))
    return [(a[0] + k * sr, a[1] + k * sc)
            for k in range(max(abs(dr), abs(dc)) + 1)]


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _intersects(a, b, c, d):
    if (max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0])
            or max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1])):
        return False
    return _cross(a, b, c) * _cross(a, b, d) <= 0 and _cross(c, d, a) * _cross(c, d, b) <= 0


def _contour(points):
    """Unique noncrossing full cycle, inferred from unordered visible vertices."""
    points = sorted(points)
    assert 4 <= len(points) <= 10 and len(points) == len(set(points))
    landmarks, edges = set(points), []
    for i, a in enumerate(points):
        for j in range(i + 1, len(points)):
            b = points[j]
            dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
            if not (dr == 0 or dc == 0 or dr == dc):
                continue
            if not set(_segment(a, b)[1:-1]) & landmarks:
                edges.append((i, j))
    adjacency = [[] for _ in points]
    for k, (i, j) in enumerate(edges):
        adjacency[i].append((j, k))
        adjacency[j].append((i, k))
    assert all(len(a) >= 2 for a in adjacency), "open or disconnected landmarks"
    conflicts = [set() for _ in edges]
    for k, (a, b) in enumerate(edges):
        for l, (c, d) in enumerate(edges[:k]):
            if not {a, b} & {c, d} and _intersects(points[a], points[b], points[c], points[d]):
                conflicts[k].add(l)
                conflicts[l].add(k)
    solutions, visits = {}, [0]

    def visit(order, used, selected):
        visits[0] += 1
        assert visits[0] <= 20000, "contour search bound exceeded"
        if len(solutions) > 1:
            return
        if len(order) == len(points):
            for neighbor, edge in adjacency[order[-1]]:
                if neighbor == 0 and not conflicts[edge] & set(selected):
                    signature = tuple(sorted(selected + [edge]))
                    solutions[signature] = order
            return
        for neighbor, edge in adjacency[order[-1]]:
            if neighbor not in used and not conflicts[edge] & set(selected):
                visit(order + [neighbor], used | {neighbor}, selected + [edge])
    visit([0], {0}, [])
    assert len(solutions) == 1, "contour is not uniquely recoverable"
    chosen, order = next(iter(solutions.items()))
    stroke = set().union(*(set(_segment(points[edges[e][0]], points[edges[e][1]])) for e in chosen))
    all_stroke = set().union(*(set(_segment(points[i], points[j])) for i, j in edges))
    assert stroke - landmarks, "already complete contour"
    turns = [_cross(points[order[i-1]], points[order[i]], points[order[(i+1) % len(order)]])
             for i in range(len(order))]
    return {"points": points, "stroke": stroke, "all_stroke": all_stroke,
            "chord_cells": len(all_stroke - stroke),
            "concave": any(t < 0 for t in turns) and any(t > 0 for t in turns)}


def _interior(shape, stroke):
    """Four-neighbor enclosure; diagonal one-cell walls block leakage."""
    h, w = shape
    assert all(0 <= r < h and 0 <= c < w for r, c in stroke)
    outside, queue = set(), deque()
    for r in range(h):
        for c in range(w):
            if r in (0, h-1) or c in (0, w-1):
                if (r, c) not in stroke:
                    outside.add((r, c))
                    queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in _D4:
            p = r + dr, c + dc
            if 0 <= p[0] < h and 0 <= p[1] < w and p not in outside and p not in stroke:
                outside.add(p)
                queue.append(p)
    return {(r, c) for r in range(h) for c in range(w)} - outside - stroke


def _panels(grid):
    # Frames have an exclusive local color and are isolated rectangular rings.
    panels = []
    for color in np.unique(grid):
        for component in _components(grid == color):
            rs, cs = zip(*component)
            y, x, bottom, right = min(rs), min(cs), max(rs), max(cs)
            h, w = bottom-y+1, right-x+1
            if h < 5 or w < 5 or len(component) != 2*h+2*w-4:
                continue
            if not all(r in (y, bottom) or c in (x, right) for r, c in component):
                continue
            crop = grid[y:bottom+1, x:right+1]
            inner = crop[1:-1, 1:-1]
            fill = int(inner[0, 0])
            if (fill == int(color) or not np.all(inner[0] == fill) or not np.all(inner[-1] == fill)
                    or not np.all(inner[:, 0] == fill) or not np.all(inner[:, -1] == fill)
                    or np.count_nonzero(inner == fill) <= inner.size // 2):
                continue
            panels.append({"box": (y, x, h, w), "crop": crop, "inner": inner,
                           "fill": fill, "frame": int(color)})
    # A dense square swatch can make the surrounding fill rim look like a
    # second small frame. Only the two outer scene panels define the crop.
    panels = [p for p in panels if not any(
        q is not p and q["box"][0] <= p["box"][0] and q["box"][1] <= p["box"][1]
        and q["box"][0]+q["box"][2] >= p["box"][0]+p["box"][2]
        and q["box"][1]+q["box"][3] >= p["box"][1]+p["box"][3]
        for q in panels)]
    assert len(panels) == 2, "expected two clear framed panels"
    a, b = [p["box"] for p in panels]
    assert a[0]+a[2] <= b[0] or b[0]+b[2] <= a[0] or a[1]+a[3] <= b[1] or b[1]+b[3] <= a[1]
    return panels


def _dictionary(panel):
    templates, repetitions = {}, {}
    for cells in _components(panel["inner"] != panel["fill"]):
        colors, counts = np.unique([panel["inner"][p] for p in cells], return_counts=True)
        assert len(colors) == 2 and sorted(counts) == [1, len(cells)-1] and len(cells) >= 3
        key = int(colors[np.argmin(counts)])
        decor = int(colors[np.argmax(counts)])
        anchor = next(p for p in cells if panel["inner"][p] == key)
        offsets = tuple(sorted((r-anchor[0], c-anchor[1]) for r, c in cells if (r, c) != anchor))
        value = (decor, offsets)
        assert key not in templates or templates[key] == value, "inconsistent reference swatches"
        templates[key] = value
        repetitions[key] = repetitions.get(key, 0) + 1
    assert 1 <= len(templates) <= 3
    assert not set(templates) & {v[0] for v in templates.values()}
    return templates, repetitions


def _render(target, templates, contours, mode="correct"):
    result = target["crop"].copy()
    inner = result[1:-1, 1:-1]
    for key, info in contours.items():
        stroke = info["all_stroke"] if mode == "all_alignments" else info["stroke"]
        if mode != "no_completion":
            for p in stroke:
                inner[p] = key
        if mode == "no_template":
            continue
        decor, offsets = templates[key]
        if mode == "rotated_stencil":
            offsets = [(-c, r) for r, c in offsets]
        if mode == "symmetric":
            radius = max(max(abs(r), abs(c)) for r, c in offsets)
            offsets = [(r, c) for r in range(-radius, radius+1) for c in range(-radius, radius+1) if r or c]
        origins = info["points"] if mode in ("vertices_only", "no_completion") else stroke
        if mode == "solid_fill":
            for p in info["inside"]:
                inner[p] = decor
            continue
        if mode == "opaque_patch":
            mask = ((0, 0),)+tuple(offsets)
            lowr, highr = min(r for r, c in mask), max(r for r, c in mask)
            lowc, highc = min(c for r, c in mask), max(c for r, c in mask)
            for r, c in origins:
                for dr in range(lowr, highr+1):
                    for dc in range(lowc, highc+1):
                        p = r+dr, c+dc
                        if p in info["inside"]:
                            inner[p] = target["reference_fill"]
        for r, c in origins:
            for dr, dc in offsets:
                p = r + dr, c + dc
                allowed = p in info["inside"]
                if mode == "unclipped":
                    allowed = 0 <= p[0] < inner.shape[0] and 0 <= p[1] < inner.shape[1] and p not in info["stroke"]
                if allowed:
                    inner[p] = decor
        # Source strokes are always in front, including at concave turns.
        if mode != "no_completion":
            for p in stroke:
                inner[p] = key
    return result


def _analyze(input_grid):
    grid = _grid(input_grid)
    panels = _panels(grid)
    candidates = []
    for reference, target in (panels, panels[::-1]):
        try:
            templates, repetitions = _dictionary(reference)
            inner = target["inner"]
            keys = {int(c) for c in np.unique(inner) if c != target["fill"]}
            assert keys == set(templates), "every swatch has one requested contour"
            assert reference["frame"] == target["frame"]
            reserved = {reference["fill"], target["fill"], reference["frame"]}
            assert not reserved & (keys | {v[0] for v in templates.values()})
            contours, occupied = {}, set()
            for key in sorted(keys):
                info = _contour([tuple(map(int, p)) for p in np.argwhere(inner == key)])
                info["inside"] = _interior(inner.shape, info["stroke"])
                assert info["inside"], "empty contour"
                footprint = info["stroke"] | info["inside"]
                assert not occupied & footprint, "overlapping enclosed regions"
                occupied.update(footprint)
                contours[key] = info
            expected = _render(target, templates, contours)
            candidates.append((expected, {"reference": reference, "target": target,
                                         "templates": templates, "repetitions": repetitions,
                                         "contours": contours}))
        except AssertionError:
            continue
    assert len(candidates) == 1, "ambiguous or invalid reference/target interpretation"
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the swatches, contour topology and inside mask from this input only."""
    return _analyze(input_grid)[0]


def _transform(points, code):
    result = []
    for r, c in points:
        if code & 4:
            c = -c
        for _ in range(code % 4):
            r, c = c, -r
        result.append((r, c))
    mr, mc = min(r for r, c in result), min(c for r, c in result)
    return [(r-mr, c-mc) for r, c in result]


def _shape(rng, max_h, max_w, family=None):
    """Procedural polygons; proposals survive input-only topology inference."""
    for _ in range(90):
        kind = family or str(rng.choice(["rectangle", "diamond", "clipped", "L", "notch"]))
        h, w = int(rng.integers(4, max_h)), int(rng.integers(4, max_w))
        if kind == "square":
            h = w = int(rng.integers(4, min(max_h, max_w)))
        if kind in ("rectangle", "square"):
            points = [(0, 0), (0, w), (h, w), (h, 0)]
        elif kind == "diamond":
            a = int(rng.integers(2, (min(max_h, max_w)-1)//2+1))
            points = [(0, a), (a, 2*a), (2*a, a), (a, 0)]
        elif kind == "clipped":
            limit = max(2, min(h, w)//2)
            a, b, c, d = [int(rng.integers(limit)) for _ in range(4)]
            points = [(0, a), (0, w-b), (b, w), (h-c, w), (h, w-c), (h, d), (h-d, 0), (a, 0)]
            points = [p for i, p in enumerate(points) if p != points[i-1]]
        elif kind == "L":
            a, b = int(rng.integers(2, w-1)), int(rng.integers(2, h-1))
            points = [(0, 0), (0, w), (b, w), (b, a), (h, a), (h, 0)]
        else:
            a, b = sorted(int(x) for x in rng.choice(range(1, w), 2, replace=False))
            depth = int(rng.integers(1, h-1))
            points = [(0, 0), (0, w), (h, w), (h, b), (depth, b), (depth, a), (h, a), (h, 0)]
        if len(set(points)) != len(points):
            continue
        points = _transform(points, int(rng.integers(8)))
        if max(r for r, c in points) >= max_h or max(c for r, c in points) >= max_w:
            continue
        try:
            info = _contour(points)
            intended = set().union(*(set(_segment(a, b)) for a, b in zip(points, points[1:]+points[:1])))
            assert info["stroke"] == intended
            if family in ("L", "notch"):
                assert info["concave"]
            return points
        except AssertionError:
            continue
    # A fixed *construction* guarantees progress; it is not a motif vocabulary.
    points = [(0, 0), (0, 6), (2, 6), (2, 3), (5, 3), (5, 0)] if family in ("L", "notch") else [(0, 0), (0, 6), (6, 6), (6, 0)]
    assert max(r for r, c in points) < max_h and max(c for r, c in points) < max_w
    _contour(points)
    return points


def _motif(rng, small=False):
    h, w = int(rng.integers(2, 4 if small else 5)), int(rng.integers(2, 4 if small else 5))
    size = int(rng.integers(3, min(8 if small else 11, h*w+1)))
    cells = {(int(rng.integers(h)), int(rng.integers(w)))}
    while len(cells) < size:
        frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _D4
                           if 0 <= r+dr < h and 0 <= c+dc < w} - cells)
        cells.add(frontier[int(rng.integers(len(frontier)))])
    ar, ac = sorted(cells)[int(rng.integers(len(cells)))]
    return tuple(sorted((r-ar, c-ac) for r, c in cells))


def _asymmetric(mask):
    original = set(mask)
    for code in range(1, 8):
        transformed = set()
        for r, c in mask:
            if code & 4:
                c = -c
            for _ in range(code % 4):
                r, c = c, -r
            transformed.add((r, c))
        if transformed == original:
            return False
    return True


def _make_pair(rng, count, witness=None):
    # All fitting arithmetic is constructive. At most 120 proposals are alive
    # one at a time; rejected arrays are neither saved nor logged.
    for _ in range(120):
        palette = [int(c) for c in rng.permutation(10)]
        exterior, frame, rf, tf = palette[:4]
        if witness is None and rng.random() < .3:
            tf = rf
        keys = palette[4:4+count]
        decors = palette[4+count:]
        templates, shapes = {}, {}
        for key in keys:
            mask = _motif(rng, small=witness is not None)
            decor = decors[int(rng.integers(len(decors)))]
            templates[key] = (decor, tuple(p for p in mask if p != (0, 0)))
            shapes[key] = mask
        if witness and not all(_asymmetric(mask) for mask in shapes.values()):
            continue
        # Single-object demos are spacious enough to expose an unpainted core.
        if count == 1:
            ih, iw = int(rng.integers(13, 20)), int(rng.integers(13, 23))
            slots = [(0, 0, ih, iw)]
        else:
            side = 9
            if count == 2:
                locations = [(0, 0), (0, side)] if rng.random() < .5 else [(0, 0), (side, 0)]
            else:
                locations = [(0, 0), (0, side), (side, 0), (side, side)]
                locations.pop(int(rng.integers(4)))
            ih = max(r for r, c in locations)+side
            iw = max(c for r, c in locations)+side
            rng.shuffle(locations)
            slots = [(r, c, side, side) for r, c in locations]
        vertices, contours = {}, {}
        for index, (key, (sy, sx, sh, sw)) in enumerate(zip(keys, slots)):
            family = ("square" if rng.random() < .5 else "diamond") if witness == "chord" and index == 0 else "L" if witness == "concave" and index == 0 else None
            points = _shape(rng, sh-2, sw-2, family)
            ph, pw = max(r for r, c in points)+1, max(c for r, c in points)+1
            dy = sy+int(rng.integers(1, sh-ph))
            dx = sx+int(rng.integers(1, sw-pw))
            points = [(r+dy, c+dx) for r, c in points]
            vertices[key] = points
            contours[key] = _contour(points)
            contours[key]["inside"] = _interior((ih, iw), contours[key]["stroke"])
        target = np.full((ih+2, iw+2), frame, dtype=np.int8)
        target[1:-1, 1:-1] = tf
        for key, points in vertices.items():
            for r, c in points:
                target[r+1, c+1] = key
        target_info = {"crop": target, "inner": target[1:-1, 1:-1], "fill": tf, "reference_fill": rf}
        output = _render(target_info, templates, contours)
        # Each contour must visibly use the template and retain a blank core.
        good = True
        for key, info in contours.items():
            values = [int(output[r+1, c+1]) for r, c in info["inside"]]
            if templates[key][0] not in values or tf not in values:
                good = False
        if not good:
            continue
        if witness:
            if any(np.array_equal(output, _render(target_info, templates, contours, mode))
                   for mode in ("vertices_only", "unclipped", "symmetric", "solid_fill", "rotated_stencil", "opaque_patch")):
                continue
            if witness == "chord" and not any(i["chord_cells"] for i in contours.values()):
                continue
        repeats = {k: 2 if witness else int(rng.integers(1, 3)) for k in keys}
        if count == 3:
            repeats = {k: 1 for k in keys}
        entries = [k for k in keys for _ in range(repeats[k])]
        rng.shuffle(entries)
        rh = max(max(r for r, c in shapes[k])-min(r for r, c in shapes[k])+1 for k in keys)+4
        gaps = [int(rng.integers(1, 3)) for _ in entries[:-1]]
        rw = sum(max(c for r, c in shapes[k])-min(c for r, c in shapes[k])+1 for k in entries)+sum(gaps)+4
        if rw > 30 or rh+1+target.shape[0] > 30:
            continue
        reference = np.full((rh, rw), frame, dtype=np.int8)
        reference[1:-1, 1:-1] = rf
        x = 2
        for index, key in enumerate(entries):
            mask = shapes[key]
            minr, minc = min(r for r, c in mask), min(c for r, c in mask)
            mh = max(r for r, c in mask)-minr+1
            y = int(rng.integers(2, rh-mh-1))
            for dr, dc in mask:
                reference[y+dr-minr, x+dc-minc] = key if (dr, dc) == (0, 0) else templates[key][0]
            x += max(c for r, c in mask)-minc+1+(gaps[index] if index < len(gaps) else 0)
        # Either panel can precede the other. Extra margins use only fitting slack.
        gap = int(rng.integers(1, min(3, 30-rh-target.shape[0])+1))
        total_h = rh+gap+target.shape[0]
        total_w = max(rw, target.shape[1])
        h = int(rng.integers(total_h, min(30, total_h+2)+1))
        w = int(rng.integers(total_w, min(30, total_w+3)+1))
        grid = np.full((h, w), exterior, dtype=np.int8)
        top = int(rng.integers(h-total_h+1))
        left_r, left_t = int(rng.integers(w-rw+1)), int(rng.integers(w-target.shape[1]+1))
        if rng.random() < .5:
            ry, ty = top, top+rh+gap
        else:
            ty, ry = top, top+target.shape[0]+gap
        grid[ry:ry+rh, left_r:left_r+rw] = reference
        grid[ty:ty+target.shape[0], left_t:left_t+target.shape[1]] = target
        # Global augmentation changes all offsets together, never a stamp alone.
        turns, reflect = int(rng.integers(4)), bool(rng.integers(2))
        grid, output = np.rot90(grid, turns).copy(), np.rot90(output, turns).copy()
        if reflect:
            grid, output = np.fliplr(grid).copy(), np.fliplr(output).copy()
        predicted, _ = _analyze(grid)
        assert np.array_equal(predicted, output), "rendering and visible reconstruction disagree"
        return {"input": grid, "output": output}
    raise RuntimeError("bounded informative-pair construction exhausted")


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    demos = list(map(int, rng.choice(nt, 2, replace=False)))
    count_cap = int(rng.integers(1, 3))
    train_max = 1
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            witness = "chord" if split == "train" and i == demos[0] else "concave" if split == "train" and i == demos[1] else None
            n = 1 if witness else int(rng.integers(1, count_cap+1)) if split == "train" else train_max+1
            for _ in range(8):
                pair = _make_pair(rng, n, witness)
                fingerprint = _key(pair["input"])
                if fingerprint not in seen:
                    break
            else:
                raise RuntimeError("distinct input retries exhausted")
            seen.add(fingerprint)
            puzzle[split].append(pair)
            if split == "train":
                train_max = max(train_max, n)
    validate_generated_puzzle(puzzle)
    return puzzle


def pair_features(pair):
    expected, info = _analyze(pair["input"])
    assert np.array_equal(expected, np.asarray(pair["output"]))
    target, templates, contours = {**info["target"], "reference_fill": info["reference"]["fill"]}, info["templates"], info["contours"]
    alternatives = {mode: int(np.count_nonzero(expected != _render(target, templates, contours, mode)))
                    for mode in ("no_completion", "no_template", "vertices_only", "all_alignments", "unclipped", "solid_fill", "symmetric", "rotated_stencil", "opaque_patch")}
    geometries = []
    for key, c in contours.items():
        points = c["points"]
        mr, mc = min(r for r, c in points), min(c for r, c in points)
        geometries.append(tuple(sorted((r-mr, col-mc) for r, col in points)))
    return {"input_shape": tuple(np.asarray(pair["input"]).shape), "output_shape": expected.shape,
            "types": len(contours), "vertices": tuple(len(i["points"]) for i in contours.values()),
            "concave": sum(i["concave"] for i in contours.values()),
            "chord_cells": sum(i["chord_cells"] for i in contours.values()),
            "motif_sizes": tuple(len(v[1])+1 for v in templates.values()),
            "motifs": tuple(tuple(sorted(((0, 0),)+v[1])) for v in templates.values()),
            "geometries": tuple(sorted(geometries)), "alternatives": alternatives,
            "equal_fills": target["fill"] == info["reference"]["fill"],
            "shared_decor": len({v[0] for v in templates.values()}) < len(templates),
            "asymmetric": any(_asymmetric(((0, 0),)+v[1]) for v in templates.values()),
            "repetitions": tuple(sorted(info["repetitions"].values())),
            "frame": target["frame"], "fill": target["fill"]}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            fingerprint = _key(inp)
            assert fingerprint not in seen
            seen.add(fingerprint)
            expected, info = _analyze(inp)
            assert np.array_equal(expected, out), "wrong contour, stencil, or crop"
            target = info["target"]
            original = target["crop"]
            assert np.array_equal(out[original != target["fill"]], original[original != target["fill"]])
            for key, c in info["contours"].items():
                values = [int(out[r+1, col+1]) for r, col in c["inside"]]
                assert info["templates"][key][0] in values and target["fill"] in values
                assert all(out[r+1, col+1] == key for r, col in c["stroke"])
            f = pair_features(pair)
            assert f["alternatives"]["no_completion"] > 0 and f["alternatives"]["no_template"] > 0
            features[split].append(f)
    train = features["train"]
    assert any(f["chord_cells"] for f in train), "chord rejection must be demonstrated"
    assert any(f["concave"] for f in train), "inward corners must be demonstrated"
    for name in ("vertices_only", "all_alignments", "unclipped", "solid_fill", "symmetric", "rotated_stencil", "opaque_patch"):
        assert any(f["alternatives"][name] > 0 for f in train), f"undistinguished shortcut: {name}"
    assert any(not f["equal_fills"] and f["asymmetric"] and min(f["repetitions"]) >= 2 for f in train)
    assert min(f["types"] for f in features["test"]) > max(f["types"] for f in train)


def self_test():
    """42 seeds, no saved puzzles, no logs, no unbounded retained payloads."""
    stats = {"input_shapes": set(), "output_shapes": set(), "motifs": set(), "geometries": set(),
             "types": set(), "sizes": set(), "frames": set(), "fills": set(), "repeats": set()}
    signatures, concave, chord, equal, shared, total = set(), 0, 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = pair_features(pair)
                digest.update(np.asarray(pair["input"], dtype=np.int8).tobytes())
                stats["input_shapes"].add(f["input_shape"])
                stats["output_shapes"].add(f["output_shape"])
                stats["motifs"].update(f["motifs"])
                stats["geometries"].update(f["geometries"])
                stats["types"].add(f["types"])
                stats["sizes"].update(f["motif_sizes"])
                stats["frames"].add(f["frame"])
                stats["fills"].add(f["fill"])
                stats["repeats"].update(f["repetitions"])
                concave += f["concave"]
                chord += f["chord_cells"] > 0
                equal += f["equal_fills"]
                shared += f["shared_decor"]
                total += 1
                # Equivariance tests input-only solving of transformed, unseen layouts.
                if split == "test":
                    transformed = np.rot90(pair["input"])
                    assert np.array_equal(solve(transformed), np.rot90(pair["output"]))
        signatures.add(digest.hexdigest())
    assert len(signatures) == 42
    assert len(stats["input_shapes"]) >= 30 and len(stats["output_shapes"]) >= 20
    assert len(stats["motifs"]) >= 80 and len(stats["geometries"]) >= 60
    assert stats["types"] == {1, 2, 3} and {3, 4, 5, 6, 7} <= stats["sizes"]
    assert stats["frames"] == set(range(10)) and stats["fills"] == set(range(10))
    assert stats["repeats"] == {1, 2} and min(concave, chord, equal, shared) > 0
    # Regression: a dense 3x3 swatch leaves a false 5x5 fill-colored ring
    # inside its true 7x7 frame. Select the two outer scene panels.
    scene = np.zeros((21, 17), dtype=np.int8)
    for y, x, h, w in ((1, 1, 7, 7), (9, 4, 11, 12)):
        scene[y:y+h, x:x+w] = 8
        scene[y+1:y+h-1, x+1:x+w-1] = 2
    scene[3:6, 3:6] = 4
    scene[4, 4] = 5
    scene[12, 7] = scene[12, 12] = scene[16, 7] = scene[16, 12] = 5
    assert len(_panels(scene)) == 2
    first, again = generate_puzzle(314159, 5, 3), generate_puzzle(314159, 5, 3)
    assert all(np.array_equal(a[k], b[k]) for s in first for a, b in zip(first[s], again[s]) for k in a)
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(919+a*10+b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in first[s]] for s in first}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt crop accepted")
    return {"seeds": 42, "pairs": total, "distinct_puzzles": len(signatures),
            **{k: len(v) for k, v in stats.items()}, "concave_contours": concave,
            "chord_pairs": chord, "equal_fill_pairs": equal, "shared_decoration_pairs": shared}
