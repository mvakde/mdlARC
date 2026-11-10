"""Echo vents: sparse, translated cutting tools change a body's enclosure.

Only standard Python and NumPy are used. No parent module is imported. The
input-only solver does not access generation state or training examples.
"""
from collections import Counter

import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "illegal grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(bool(((a >= 0) & (a <= 9)).all()), "illegal grid color")
    return a.astype(np.int8, copy=True)


def _regions(mask, neighbors=N4):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        cells, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    cells.add(q)
                    stack.append(q)
        result.append(cells)
    return result


def _holes(body, neighbors=N4):
    padded = np.pad(body, 1)
    outside = np.zeros(padded.shape, bool)
    outside[0, 0] = True
    stack = [(0, 0)]
    h, w = padded.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in neighbors:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not padded[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                stack.append((rr, cc))
    return (~padded & ~outside)[1:-1, 1:-1]


def _clear_topology(body):
    if len(_regions(body)) != 1 or len(_regions(body, N8)) != 1:
        return False
    a, b, c, d = body[:-1, :-1], body[:-1, 1:], body[1:, :-1], body[1:, 1:]
    if np.any((a == d) & (b == c) & (a != b)):
        return False
    holes = _holes(body)
    return (np.array_equal(holes, _holes(body, N8))
            and len(_regions(holes)) == len(_regions(holes, N8)))


def _line(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    length = max(abs(dr), abs(dc))
    _require(length > 0 and (dr == 0 or dc == 0 or abs(dr) == abs(dc)), "unaligned link")
    sr, sc = int(np.sign(dr)), int(np.sign(dc))
    return [(a[0] + i * sr, a[1] + i * sc) for i in range(length + 1)]


def _aligned(a, b):
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dr, dc) > 0 and (dr == 0 or dc == 0 or dr == dc)


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _intersects(a, b, c, d):
    if (max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0])
            or max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1])):
        return False
    return _cross(a, b, c) * _cross(a, b, d) <= 0 and _cross(c, d, a) * _cross(c, d, b) <= 0


def _infer_path(points):
    """Unique full noncrossing open path; reversal is the same edge set."""
    _require(2 <= len(points) <= 6 and len(set(points)) == len(points), "illegal landmarks")
    landmarks = set(points)
    edges = [(i, j) for i, a in enumerate(points) for j, b in enumerate(points)
             if i < j and _aligned(a, b) and not (set(_line(a, b)[1:-1]) & landmarks)]
    adjacency = [[] for _ in points]
    for k, (i, j) in enumerate(edges):
        adjacency[i].append((j, k))
        adjacency[j].append((i, k))
    conflicts = [set() for _ in edges]
    for i, (a, b) in enumerate(edges):
        for j, (c, d) in enumerate(edges[:i]):
            if not {a, b} & {c, d} and _intersects(points[a], points[b], points[c], points[d]):
                conflicts[i].add(j)
                conflicts[j].add(i)
    solutions, budget = set(), [0]

    def visit(last, used, selected):
        budget[0] += 1
        _require(budget[0] < 20000, "path search budget exceeded")
        if len(solutions) > 1:
            return
        if len(used) == len(points):
            solutions.add(tuple(sorted(selected)))
            return
        for q, edge in adjacency[last]:
            if q not in used and not conflicts[edge] & set(selected):
                visit(q, used | {q}, selected + [edge])

    for start in range(len(points)):
        visit(start, {start}, [])
        if len(solutions) > 1:
            break
    _require(len(solutions) == 1, "competing or absent open routes")
    selected = [edges[k] for k in next(iter(solutions))]
    pixels = set().union(*(set(_line(points[i], points[j])) for i, j in selected))
    _require(len(pixels) > len(points), "tool has no missing cells")
    return pixels, selected


def _cut(body, pixels):
    result = body.copy()
    h, w = body.shape
    for r, c in pixels:
        if 0 <= r < h and 0 <= c < w:
            result[r, c] = False
    return result


def _translated(pixels, offset):
    return {(r + offset[0], c + offset[1]) for r, c in pixels}


def _paint(body, bg, wall, guide, marker):
    holes = _holes(body)
    out = np.full(body.shape, bg, np.int8)
    out[body] = guide if holes.any() else wall
    out[holes] = marker
    return out


def _inspect(input_grid):
    inp = _grid(input_grid)
    colors, counts = np.unique(inp, return_counts=True)
    _require(len(colors) >= 4, "missing input roles")
    bg = int(colors[np.argmax(counts)])
    _require(int(counts.max()) * 2 > inp.size, "background must occupy a majority")
    non_bg = [(int(n), int(c)) for c, n in zip(colors, counts) if c != bg]
    wall = max(non_bg)[1]
    _require(sum(n == max(non_bg)[0] for n, c in non_bg) == 1, "ambiguous wall color")
    components = _regions(inp != bg)
    _require(len(components) == len(_regions(inp != bg, N8)), "diagonal body contact")
    out = np.full(inp.shape, bg, np.int8)
    occupied_boxes, objects = np.zeros(inp.shape, bool), []
    for cells in components:
        rr, cc = np.array(sorted(cells)).T
        r0, c0, r1, c1 = int(rr.min()), int(cc.min()), int(rr.max()) + 1, int(cc.max()) + 1
        _require(not occupied_boxes[r0:r1, c0:c1].any(), "nested/interlocking body boxes")
        occupied_boxes[r0:r1, c0:c1] = True
        body = np.zeros((r1 - r0, c1 - c0), bool)
        body[rr - r0, cc - c0] = True
        _require(_clear_topology(body), "ambiguous initial topology")
        local = inp[r0:r1, c0:c1]
        values = Counter(int(inp[p]) for p in cells)
        _require(wall in values and len(values) == 3, "body must contain wall, landmarks and marker")
        other = {c: n for c, n in values.items() if c != wall}
        guide_candidates = [c for c, n in other.items() if n >= 2]
        marker_candidates = [c for c, n in other.items() if n == 1]
        _require(len(guide_candidates) == len(marker_candidates) == 1, "unclear local color roles")
        guide, marker_color = guide_candidates[0], marker_candidates[0]
        points = [tuple(map(int, p)) for p in np.argwhere((local == guide) & body)]
        marker = tuple(map(int, np.argwhere((local == marker_color) & body)[0]))
        _require(values[wall] >= 2 * len(points), "insufficient wall material")
        source, edges = _infer_path(points)
        associations = [p for p in points if max(abs(p[0] - marker[0]), abs(p[1] - marker[1])) == 1]
        _require(len(associations) == 1, "ambiguous marker association")
        anchor = associations[0]
        offset = (marker[0] - anchor[0], marker[1] - anchor[1])
        echo = _translated(source, offset)
        remaining = _cut(body, source | echo)
        _require(_clear_topology(remaining), "disconnected or ambiguous residual body")
        holes = _holes(remaining)
        result = _paint(remaining, bg, wall, guide, marker_color)
        # Boxes do not overlap; outside remains background, including cuts.
        out[r0:r1, c0:c1] = result
        objects.append({"origin": (r0, c0), "body": body, "remaining": remaining,
                        "holes": holes, "initial_holes": _holes(body), "points": points,
                        "source": source, "echo": echo, "edges": edges, "marker": marker,
                        "offset": offset, "guide_color": guide, "marker_color": marker_color})
    _require(objects, "no bodies")
    return out, {"background": bg, "wall": wall, "objects": objects}


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _inspect(input_grid)[0]


def _finish(body, points, offset):
    """Check an intended local construction against visible-input inference."""
    source, edges = _infer_path(points)
    intended = set().union(*(set(_line(a, b)) for a, b in zip(points, points[1:])))
    _require(source == intended, "intended route is not recoverable")
    h, w = body.shape
    _require(all(0 <= r < h and 0 <= c < w and body[r, c] for r, c in points), "landmark off material")
    candidates = []
    for p in points:
        q = (p[0] + offset[0], p[1] + offset[1])
        if (0 <= q[0] < h and 0 <= q[1] < w and body[q] and q not in points
                and sum(max(abs(v[0] - q[0]), abs(v[1] - q[1])) == 1 for v in points) == 1):
            candidates.append(q)
    _require(candidates, "no visible translation marker")
    marker = candidates[-1]
    echo = _translated(source, offset)
    remaining = _cut(body, source | echo)
    _require(_clear_topology(body) and _clear_topology(remaining), "bad body topology")
    _require(int(body.sum()) - len(points) - 1 >= 2 * len(points), "body too small for guide")
    return {"body": body, "points": points, "marker": marker, "offset": offset,
            "source": source, "remaining": remaining, "holes": _holes(remaining)}


def _decorate(rng, body, points):
    """Exterior steps/tabs vary independently of pocket position and tool."""
    body = body.copy()
    initial = _holes(body)
    for _ in range(int(rng.integers(4))):
        candidate = body.copy()
        h, w = body.shape
        dh, dw = int(rng.integers(1, 3)), int(rng.integers(1, 4))
        side = int(rng.integers(4))
        rs = slice(0, dh) if side < 2 else slice(h - dh, h)
        cs = slice(0, dw) if side % 2 == 0 else slice(w - dw, w)
        candidate[rs, cs] = False
        if (all(candidate[p] for p in points) and _clear_topology(candidate)
                and np.array_equal(_holes(candidate), initial)):
            body = candidate
    if rng.random() < .65:
        padded = np.pad(body, 1)
        h, w = body.shape
        side = int(rng.integers(4))
        if side < 2:
            start = int(rng.integers(1, w + 1))
            end = min(w + 1, start + int(rng.integers(1, 5)))
            padded[0 if side == 0 else h + 1, start:end] = True
        else:
            start = int(rng.integers(1, h + 1))
            end = min(h + 1, start + int(rng.integers(1, 5)))
            padded[start:end, 0 if side == 2 else w + 1] = True
        if _clear_topology(padded) and np.array_equal(_holes(padded), np.pad(initial, 1)):
            rr, cc = np.nonzero(padded)
            r0, c0 = int(rr.min()), int(cc.min())
            body = padded[r0:rr.max() + 1, c0:cc.max() + 1].copy()
            points = [(r + 1 - r0, c + 1 - c0) for r, c in points]
    return body, points


def _orient_spec(spec, code):
    h, w = spec["body"].shape
    points = spec["points"]
    marker = spec["marker"]
    body = spec["body"].copy()
    if code & 4:
        body = np.fliplr(body)
        points = [(r, w - 1 - c) for r, c in points]
        marker = (marker[0], w - 1 - marker[1])
    for _ in range(code % 4):
        body = np.rot90(body)
        points = [(w - 1 - c, r) for r, c in points]
        marker = (w - 1 - marker[1], marker[0])
        h, w = w, h
    anchor = next(p for p in points if max(abs(p[0] - marker[0]), abs(p[1] - marker[1])) == 1)
    offset = (marker[0] - anchor[0], marker[1] - anchor[1])
    result = _finish(body, points, offset)
    result["marker"] = marker
    return result


def _irregular_pocket(rng, body, top, bottom, left, right):
    body[top:bottom, left:right] = False
    # A step at a lower corner leaves the top edge of the chamber intact.
    if bottom - top >= 2 and right - left >= 2 and rng.random() < .5:
        dh = int(rng.integers(1, bottom - top))
        dw = int(rng.integers(1, right - left))
        body[bottom - dh:bottom, right - dw:right] = True


def _contrast(rng):
    for _ in range(60):
        t = int(rng.integers(5, 7))
        h, w = int(rng.integers(t + 3, 14)), int(rng.integers(8, 14))
        body = np.ones((h, w), bool)
        _irregular_pocket(rng, body, t, h - 1, 1, w - 1)
        col = int(rng.integers(2, w - 2))
        points = [(0, col), (t - 2, col)]
        body, points = _decorate(rng, body, points)
        try:
            opened = _finish(body, points, (1, 1))
            sealed = _finish(body, points, (-1, 1))
            _require(len(_regions(opened["holes"])) == 0 and len(_regions(sealed["holes"])) == 1, "lost contrast")
            _require(_holes(_cut(body, opened["source"])).any(), "source alone vents contrast")
            code = int(rng.integers(8))
            return [_orient_spec(opened, code), _orient_spec(sealed, code)]
        except AssertionError:
            continue
    body = np.ones((9, 10), bool)
    body[5:8, 1:9] = False
    points = [(0, 3), (3, 3)]
    return [_finish(body, points, (1, 1)), _finish(body, points, (-1, 1))]


def _make_object(rng, kind, size=12):
    for attempt in range(100):
        try:
            if kind == "tiny":
                if rng.random() < .5:
                    w = int(rng.integers(3, 7))
                    body = np.ones((3, w), bool)
                    body[1, 1:-1] = False
                    points, offset, expected = [(0, 0), (0, w - 1)], (1, 0), 0
                else:
                    h, w = int(rng.integers(4, 6)), int(rng.integers(5, 8))
                    body = np.ones((h, w), bool)
                    points, offset, expected = [(1, 1), (1, w - 2)], (1, 0), 1
            elif kind in ("partial", "all_echo", "miss_multi"):
                t = 5 if size <= 10 else int(rng.integers(5, 7))
                h = int(rng.integers(t + 5 if kind == "partial" else t + 3, size + 1))
                w = int(rng.integers(10, size + 1))
                body = np.ones((h, w), bool)
                split = w // 2
                intervals = [(1, split), (split + 1, w - 1)]
                if kind == "all_echo" and w >= 12 and rng.random() < .45:
                    intervals = [(1, 4), (5, 8), (9, w - 1)]
                for i, (left, right) in enumerate(intervals):
                    top = t + 2 if kind == "partial" and i == 1 else t
                    _irregular_pocket(rng, body, top, h - 2, left, right)
                points = [(0, 2), (t - 2, 2), (t - 2, w - 3)]
                offset = (-1, 1) if kind == "miss_multi" else (1, 1)
                expected = 1 if kind == "partial" else 2 if kind == "miss_multi" else 0
                if kind == "miss_multi":
                    points = points[:2]
            elif kind in ("echo", "miss", "vent"):
                t = int(rng.integers(5, min(7, size - 1))) if kind != "vent" else int(rng.integers(3, 6))
                h, w = int(rng.integers(max(t + 2, 7), size + 1)), int(rng.integers(7, size + 1))
                body = np.ones((h, w), bool)
                _irregular_pocket(rng, body, t, h - 1, 1, w - 1)
                col = int(rng.integers(2, w - 2))
                depth = t - 1 if kind == "vent" else t - 2
                points = [(0, col), (depth, col)]
                offset = (0, 1) if kind == "vent" else (1, 1) if kind == "echo" else (-1, 1)
                expected = 1 if kind == "miss" else 0
            elif kind == "bay":
                h, w = int(rng.integers(7, size + 1)), int(rng.integers(7, size + 1))
                body = np.ones((h, w), bool)
                depth = int(rng.integers(3, h - 2))
                body[:depth, 2:w - 2] = False
                points = [(0, w - 1), (depth, w - 1)]
                offset, expected = (0, -1), 0
            else:
                _require(kind == "pocket", "unknown body kind")
                h, w = int(rng.integers(7, size + 1)), int(rng.integers(7, size + 1))
                body = np.ones((h, w), bool)
                style = int(rng.integers(4))
                if style == 0:
                    points, offset = [(2, 2), (2, w - 3)], (1, 0)
                elif style == 1:
                    length = int(rng.integers(2, min(h, w) - 4))
                    points, offset = [(2, 2), (2 + length, 2 + length)], (0, 1)
                elif style == 2:
                    points, offset = [(2, 2), (2, w - 3), (h - 3, w - 3)], (1, -1)
                else:
                    points = [(2, 2)]
                    for i in range(int(rng.integers(3, 6))):
                        r, c = points[-1]
                        length = int(rng.integers(1, 4))
                        q = (r, c + length) if i % 2 == 0 else (r + length, c)
                        if q[0] <= h - 3 and q[1] <= w - 3:
                            points.append(q)
                    offset = (1, 0)
                expected = 1
            initial_count = len(_regions(_holes(body)))
            if kind != "tiny":
                body, points = _decorate(rng, body, points)
            spec = _finish(body, points, offset)
            _require(len(_regions(spec["holes"])) == expected, "unexpected cavity effect")
            _require(len(_regions(_holes(body))) == initial_count, "decoration changed initial cavities")
            if kind in ("echo", "partial", "all_echo"):
                _require(len(_regions(_holes(_cut(body, spec["source"])))) > expected, "echo has no topological effect")
            return _orient_spec(spec, int(rng.integers(8)))
        except (AssertionError, ValueError):
            continue
    # Fixed valid geometry is an emergency finite fallback, not a seed template.
    if kind == "partial":
        body = np.ones((10, 10), bool)
        body[5:8, 1:5] = False
        body[7:8, 6:9] = False
        return _finish(body, [(0, 2), (3, 2), (3, 7)], (1, 1))
    if kind in ("all_echo", "miss_multi"):
        body = np.ones((10, 10), bool)
        body[5:8, 1:5] = False
        body[5:8, 6:9] = False
        return (_finish(body, [(0, 2), (3, 2)], (-1, 1)) if kind == "miss_multi"
                else _finish(body, [(0, 2), (3, 2), (3, 7)], (1, 1)))
    if kind == "tiny":
        body = np.ones((3, 4), bool)
        body[1, 1:3] = False
        return _finish(body, [(0, 0), (0, 3)], (1, 0))
    if kind == "pocket":
        return _finish(np.ones((8, 10), bool), [(2, 2), (2, 7), (5, 7)], (1, -1))
    if kind == "bay":
        body = np.ones((8, 9), bool)
        body[:4, 2:7] = False
        return _finish(body, [(0, 8), (4, 8)], (0, -1))
    body = np.ones((9, 10), bool)
    body[5:8, 1:9] = False
    return _finish(body, [(0, 3), (4 if kind == "vent" else 3, 3)],
                   (0, 1) if kind == "vent" else (1, 1) if kind == "echo" else (-1, 1))


def _teaching_pocket(rng):
    # A unit stub, a longer straight link, a diagonal, and a final bend make
    # copying every link observable even in the minimum two-example request.
    for _ in range(60):
        h, w = int(rng.integers(8, 13)), int(rng.integers(12, 14))
        first = int(rng.integers(2, 4))
        diagonal = int(rng.integers(2, 4))
        end = 3 + first + diagonal + int(rng.integers(1, 3))
        if end > w - 3 or 2 + diagonal > h - 3:
            continue
        points = [(2, 2), (2, 3), (2, 3 + first),
                  (2 + diagonal, 3 + first + diagonal), (2 + diagonal, end)]
        body, points = _decorate(rng, np.ones((h, w), bool), points)
        try:
            spec = _finish(body, points, (1, 0))
            _require(len(_regions(spec["holes"])) == 1 and not _holes(body).any(), "bad teaching pocket")
            return _orient_spec(spec, int(rng.integers(8)))
        except AssertionError:
            continue
    return _finish(np.ones((8, 12), bool), [(2, 2), (2, 3), (2, 5), (4, 7), (4, 9)], (1, 0))


def _pack(rng, specs):
    order = sorted(range(len(specs)), key=lambda i: -specs[i]["body"].size)
    for _ in range(16):
        blocked = np.zeros((30, 30), bool)
        anchors = [None] * len(specs)
        for i in order:
            h, w = specs[i]["body"].shape
            candidates = [(r, c) for r in range(31 - h) for c in range(31 - w)
                          if not blocked[r:r + h, c:c + w].any()]
            if not candidates:
                break
            r, c = candidates[int(rng.integers(len(candidates)))]
            anchors[i] = (r, c)
            blocked[max(0, r - 1):min(30, r + h + 1), max(0, c - 1):min(30, c + w + 1)] = True
        else:
            return anchors
    # At most four bodies; busy scenes use <=11-cell axes, teaching twins <=14.
    anchors = []
    if len(specs) <= 2:
        if sum(s["body"].shape[1] for s in specs) + len(specs) - 1 <= 30:
            c = 0
            for s in specs:
                anchors.append((0, c))
                c += s["body"].shape[1] + 1
            return anchors
        r = 0
        for s in specs:
            anchors.append((r, 0))
            r += s["body"].shape[0] + 1
        _require(r - 1 <= 30, "fallback cannot fit bodies")
    else:
        anchors = [(0 if i < 2 else 15, 0 if i % 2 == 0 else 15) for i in range(len(specs))]
    return anchors


def _render(rng, specs, background=None, twin_colors=False):
    anchors = _pack(rng, specs)
    r0, c0 = min(p[0] for p in anchors), min(p[1] for p in anchors)
    h = max(r + s["body"].shape[0] for (r, c), s in zip(anchors, specs)) - r0
    w = max(c + s["body"].shape[1] for (r, c), s in zip(anchors, specs)) - c0
    population = sum(int(s["body"].sum()) for s in specs)
    _require(population < 450, "too much material for a majority background")
    while h * w <= 2 * population:
        if (h <= w and h < 30) or w == 30:
            h += 1
        else:
            w += 1
    h += int(rng.integers(min(9, 31 - h)))
    w += int(rng.integers(min(9, 31 - w)))
    extent_h = max(r + s["body"].shape[0] for (r, c), s in zip(anchors, specs)) - r0
    extent_w = max(c + s["body"].shape[1] for (r, c), s in zip(anchors, specs)) - c0
    dr = int(rng.integers(h - extent_h + 1)) - r0
    dc = int(rng.integers(w - extent_w + 1)) - c0
    palette = list(map(int, rng.permutation(10)))
    if background is not None:
        palette.remove(background)
        palette.insert(0, background)
    bg, wall = palette[:2]
    common = tuple(map(int, rng.choice(palette[2:], 2, replace=False)))
    inp, out = np.full((h, w), bg, np.int8), np.full((h, w), bg, np.int8)
    for i, ((r, c), s) in enumerate(zip(anchors, specs)):
        r, c = r + dr, c + dc
        bh, bw = s["body"].shape
        guide, marker = common if twin_colors and i < 2 else map(int, rng.choice(palette[2:], 2, replace=False))
        view = inp[r:r + bh, c:c + bw]
        view[s["body"]] = wall
        for p in s["points"]:
            view[p] = guide
        view[s["marker"]] = marker
        out[r:r + bh, c:c + bw] = _paint(s["remaining"], bg, wall, guide, marker)
    return {"input": inp, "output": out}


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt, nq = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    puzzle, seen = {"train": [], "test": []}, set()
    teaching = [int(x) for x in rng.choice(nt, 2, replace=False)]
    backgrounds = list(map(int, rng.integers(10, size=nt + nq)))
    backgrounds[teaching[1]] = (backgrounds[teaching[0]] + int(rng.integers(1, 10))) % 10
    ordinary = ("echo", "miss", "vent", "bay", "pocket", "partial", "miss_multi", "tiny")
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            for retry in range(96):
                twins = split == "train" and index == teaching[0]
                if twins:
                    specs = _contrast(rng)
                elif split == "train" and index == teaching[1]:
                    specs = [_make_object(rng, "partial", 13), _teaching_pocket(rng),
                             _make_object(rng, "miss_multi", 10)]
                else:
                    n = int(rng.integers(1, 5))
                    size = 10 if n >= 3 else int(rng.integers(10, 14))
                    kinds = ["all_echo"] if split == "test" else []
                    kinds += [str(rng.choice(ordinary)) for _ in range(n - len(kinds))]
                    specs = [_make_object(rng, kind, size) for kind in kinds]
                    rng.shuffle(specs)
                bg = backgrounds[index if split == "train" else nt + index]
                pair = _render(rng, specs, bg, twins)
                fp = (pair["input"].shape, pair["input"].tobytes())
                if fp not in seen:
                    break
            else:
                raise AssertionError("distinct-input construction exhausted")
            seen.add(fp)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _counterfactual(inp, info, mode):
    out = np.full(inp.shape, info["background"], np.int8)
    for o in info["objects"]:
        if mode == "source_only":
            remaining = _cut(o["body"], o["source"])
        elif mode == "marks_only":
            remaining = _cut(o["body"], set(o["points"]) | {o["marker"]})
        elif mode == "rectangle":
            rr, cc = np.array(sorted(o["source"] | o["echo"])).T
            pixels = {(r, c) for r in range(int(rr.min()), int(rr.max()) + 1)
                      for c in range(int(cc.min()), int(cc.max()) + 1)}
            remaining = _cut(o["body"], pixels)
        elif mode == "long_links_only":
            long_pixels = set()
            for i, j in o["edges"]:
                line = _line(o["points"][i], o["points"][j])
                if len(line) > 2:
                    long_pixels.update(line)
            remaining = _cut(o["body"], o["source"] | _translated(long_pixels, o["offset"]))
        else:
            remaining = o["remaining"]
        local = (_paint(remaining, info["background"], info["wall"], o["guide_color"], o["marker_color"])
                 if mode != "no_topology" else np.where(remaining, info["wall"], info["background"]).astype(np.int8))
        r, c = o["origin"]
        out[r:r + local.shape[0], c:c + local.shape[1]] = local
    return out


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "illegal pair counts")
    seen, training, queries = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "changed dimensions")
            fp = (inp.shape, inp.tobytes())
            _require(fp not in seen, "duplicate input")
            seen.add(fp)
            expected, info = _inspect(inp)
            _require(np.array_equal(out, expected), "incorrect cutting/classification/filling")
            _require(not np.array_equal(inp, out), "identity example")
            _require(1 <= len(info["objects"]) <= 4, "unsupported body count")
            (training if split == "train" else queries).append((inp, out, info))
    objects = [o for a, b, info in training for o in info["objects"]]
    _require(any(o["initial_holes"].any() and not o["holes"].any()
                 and _holes(_cut(o["body"], o["source"])).any() for o in objects), "no echo-dependent vent evidence")
    _require(any(not o["initial_holes"].any() and o["holes"].any() for o in objects), "no created pocket evidence")
    _require(any(len(_regions(o["initial_holes"])) == 2 and len(_regions(o["holes"])) == 1 for o in objects),
             "no partial vent evidence")
    _require(any(len(_regions(o["holes"])) >= 2 for o in objects), "filling several sealed cavities not shown")
    _require(any(len(o["points"]) >= 3 for o in objects), "no bent guide evidence")
    _require(any(o["holes"].any() and np.any(o["remaining"][:-1, :-1] & o["remaining"][1:, :-1]
                 & o["remaining"][:-1, 1:] & o["remaining"][1:, 1:]) for o in objects), "no thick-body highlight evidence")
    for mode in ("source_only", "marks_only", "rectangle", "no_topology", "long_links_only"):
        _require(any(not np.array_equal(out, _counterfactual(inp, info, mode)) for inp, out, info in training),
                 "training does not distinguish " + mode)
    _require(len({info["background"] for a, b, info in training}) >= 2, "background-role variation not shown")
    _require(any(any(o["points"][i][0] != o["points"][j][0]
                     and o["points"][i][1] != o["points"][j][1] for i, j in o["edges"])
                 for o in objects), "diagonal tools not demonstrated")
    _require(all(any(len(_regions(o["initial_holes"])) >= 2 and not o["holes"].any()
                     and len(_regions(_holes(_cut(o["body"], o["source"])))) >= 2
                     and len(o["points"]) >= 3 for o in info["objects"]) for a, b, info in queries),
             "missing several-chamber echo recombination")


def _features(pair):
    inp = np.asarray(pair["input"])
    out, info = _inspect(inp)
    objects = info["objects"]
    directions = set()
    geometry, effects, silhouettes = [], [], []
    for o in objects:
        for i, j in o["edges"]:
            a, b = o["points"][i], o["points"][j]
            directions.add("horizontal" if a[0] == b[0] else "vertical" if a[1] == b[1] else "diagonal")
        geometry.append(tuple(sorted(o["points"])))
        effects.append((len(_regions(o["initial_holes"])), len(_regions(o["holes"]))))
        silhouettes.append((o["body"].shape, o["body"].tobytes()))
    occupied = inp != info["background"]
    return {"shape": inp.shape, "bodies": len(objects), "offsets": {o["offset"] for o in objects},
            "vertices": {len(o["points"]) for o in objects}, "directions": directions,
            "effects": set(effects), "silhouettes": silhouettes, "geometry": tuple(geometry),
            "background": info["background"], "wall": info["wall"],
            "boundary": bool(occupied[0].any() or occupied[-1].any() or occupied[:, 0].any() or occupied[:, -1].any()),
            "irregular": any(len(region) < (max(r for r, c in region) - min(r for r, c in region) + 1)
                             * (max(c for r, c in region) - min(c for r, c in region) + 1)
                             for o in objects for region in _regions(o["initial_holes"]))}


def self_test():
    """Bounded 42-seed semantic/diversity checks; returns descriptive metrics."""
    _regressions()
    records, signatures, shapes, silhouettes, geometries = [], set(), set(), set(), set()
    palettes, offsets, vertices, effects, counts = set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple((np.asarray(p["input"]).shape, np.asarray(p["input"]).tobytes())
                             for s in ("train", "test") for p in puzzle[s]))
        for split in puzzle:
            for pair in puzzle[split]:
                f = _features(pair)
                records.append(f)
                shapes.add(f["shape"])
                silhouettes.update(f["silhouettes"])
                geometries.add(f["geometry"])
                palettes.add((f["background"], f["wall"]))
                offsets.update(f["offsets"])
                vertices.update(f["vertices"])
                effects.update(f["effects"])
                counts.add(f["bodies"])
    _require(len(signatures) == 42, "repeated puzzles across seeds")
    _require(len(shapes) >= 35 and len(silhouettes) >= 100 and len(geometries) >= 100, "insufficient structural variation")
    _require(len(palettes) >= 35 and offsets == set(N8), "insufficient palette/displacement variation")
    _require({1, 2, 3, 4} <= counts and {2, 3, 4} <= vertices, "counts or bends collapsed")
    _require({(0, 1), (1, 0), (1, 1), (2, 1), (2, 0), (2, 2)} <= effects, "missing cavity effects")
    _require(any(max(shape) <= 6 for shape, cells in silhouettes), "small bodies absent")
    _require(set().union(*(f["directions"] for f in records)) == {"horizontal", "vertical", "diagonal"}, "missing direction")
    _require(any(f["boundary"] for f in records) and any(not f["boundary"] for f in records), "placement collapse")
    _require(sum(f["irregular"] for f in records) >= 10, "irregular cavities absent")
    # Count boundary cases, type errors, deterministic whole-puzzle reproduction.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(123456 + nt * 10 + nq, nt, nq)
            _require(len(p["train"]) == nt and len(p["test"]) == nq, "counts not honored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p, q = generate_puzzle(73, 3, 1), generate_puzzle(73, 3, 1)
    _require(all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s])
                 for k in ("input", "output")), "nondeterminism")
    # Single-input inference commutes with whole-scene spatial and color changes.
    perm = np.random.default_rng(19).permutation(10)
    for pair in p["train"] + p["test"]:
        inp, out = pair["input"], pair["output"]
        _require(np.array_equal(solve(perm[inp]), perm[out]), "palette equivariance failed")
        for k in range(4):
            a, b = np.rot90(inp, k), np.rot90(out, k)
            _require(np.array_equal(solve(a), b) and np.array_equal(solve(np.fliplr(a)), np.fliplr(b)), "spatial equivariance failed")
    # Corruption in each meaningful output zone must be detected.
    for zone in ("remaining", "holes", "exterior"):
        mutated = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        changed = False
        for pair in mutated["train"]:
            expected, info = _inspect(pair["input"])
            for o in info["objects"]:
                mask = o[zone] if zone != "exterior" else ~o["remaining"] & ~o["holes"]
                if mask.any():
                    r, c = np.argwhere(mask)[0]
                    r, c = int(r) + o["origin"][0], int(c) + o["origin"][1]
                    pair["output"][r, c] = (int(pair["output"][r, c]) + 1) % 10
                    changed = True
                    break
            if changed:
                break
        _require(changed, "mutation zone absent")
        try:
            validate_generated_puzzle(mutated)
        except AssertionError:
            pass
        else:
            raise AssertionError("corruption accepted: " + zone)
    return {"seeds": 42, "pairs": len(records), "canvas_shapes": len(shapes),
            "body_masks": len(silhouettes), "guide_geometries": len(geometries),
            "palettes": len(palettes), "offsets": sorted(offsets), "vertices": sorted(vertices),
            "body_counts": sorted(counts), "cavity_effects": sorted(effects),
            "irregular_cavity_scenes": sum(f["irregular"] for f in records),
            "boundary_scenes": sum(f["boundary"] for f in records),
            "semantic_verification": "generator-owned invariant and counterfactual checks"}


def _regressions():
    """Small explicit geometric fixtures, with expected pixels set by hand."""
    body = np.ones((9, 10), bool)
    body[5:8, 1:9] = False
    for offset, marker, vented in (((1, 1), (4, 4), True), ((-1, 1), (2, 4), False)):
        local = np.where(body, 5, 0).astype(np.int8)
        local[0, 3] = local[3, 3] = 2
        local[marker] = 3
        expected = np.where(body, 5 if vented else 2, 0).astype(np.int8)
        if not vented:
            expected[5:8, 1:9] = 3
        expected[0:4, 3] = 0
        expected[1:5 if vented else 3, 4] = 0
        if not vented:
            expected[0, 4] = 0
        if vented:
            expected[5:8, 1:9] = 0
        inp, out = np.pad(local, 4), np.pad(expected, 4)
        _require(np.array_equal(solve(inp), out), "matched inward/outward fixture failed")
    # An internal horizontal source plus its downward echo creates a 2x4 hole.
    local = np.full((7, 9), 5, np.int8)
    local[2, 2] = local[2, 5] = 2
    local[3, 5] = 3
    expected = np.full(local.shape, 2, np.int8)
    expected[2:4, 2:6] = 3
    _require(np.array_equal(solve(np.pad(local, 4)), np.pad(expected, 4)), "created-pocket fixture failed")
