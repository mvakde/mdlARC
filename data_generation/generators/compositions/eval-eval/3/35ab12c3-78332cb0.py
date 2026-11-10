"""Shadow relay jigsaw: companion exits connect to source entries.

Standard library + NumPy only. No parent imports, seed-dependent solver roles,
stored examples, external data, or generation metadata in solve().
"""
from __future__ import annotations

from itertools import combinations
import numpy as np


def _check_grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _segment(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    assert (dr or dc) and (not dr or not dc or abs(dr) == abs(dc))
    n = max(abs(dr), abs(dc))
    sr, sc = int(np.sign(dr)), int(np.sign(dc))
    return tuple((a[0] + i * sr, a[1] + i * sc) for i in range(n + 1))


def _aligned(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    return bool((dr or dc) and (not dr or not dc or abs(dr) == abs(dc)))


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _intersect(a, b, c, d):
    if (max(a[0], b[0]) < min(c[0], d[0]) or
            max(c[0], d[0]) < min(a[0], b[0]) or
            max(a[1], b[1]) < min(c[1], d[1]) or
            max(c[1], d[1]) < min(a[1], b[1])):
        return False
    return _cross(a, b, c) * _cross(a, b, d) <= 0 and _cross(c, d, a) * _cross(c, d, b) <= 0


def _infer_path(points):
    """Unique simple full path whose axial coordinate (column) never decreases."""
    points = sorted(points, key=lambda p: (p[1], p[0]))
    assert 2 <= len(points) <= 8 and len(set(points)) == len(points)
    lo, hi = points[0][1], points[-1][1]
    assert lo < hi
    assert sum(c == lo for r, c in points) == sum(c == hi for r, c in points) == 1
    landmarks = set(points)
    adjacency = [[] for _ in points]
    candidates = []
    for i, j in combinations(range(len(points)), 2):
        if _aligned(points[i], points[j]):
            if not landmarks.intersection(_segment(points[i], points[j])[1:-1]):
                candidates.append((i, j))
                if points[j][1] >= points[i][1]:
                    adjacency[i].append(j)
                if points[i][1] >= points[j][1]:
                    adjacency[j].append(i)
    solutions = []

    def visit(order, used):
        if len(solutions) >= 2:
            return
        if len(order) == len(points):
            if order[-1] == len(points) - 1:
                solutions.append(tuple(order))
            return
        a = points[order[-1]]
        for j in adjacency[order[-1]]:
            if j in used or (j == len(points) - 1 and len(used) != len(points) - 1):
                continue
            b = points[j]
            # Shared consecutive endpoint is allowed; all other intersections,
            # including diagonal half-cell crossings, are excluded.
            if any(_intersect(a, b, points[u], points[v])
                   for u, v in zip(order[:-2], order[1:-1])):
                continue
            visit(order + [j], used | {j})

    visit([0], {0})
    assert len(solutions) == 1, "source landmarks must have one monotone full path"
    ordered = [points[i] for i in solutions[0]]
    strokes = set().union(*(set(_segment(a, b)) for a, b in zip(ordered, ordered[1:])))
    all_strokes = set().union(*(set(_segment(points[i], points[j])) for i, j in candidates))
    return ordered, strokes, all_strokes - strokes


def _partition(grid):
    candidates = []
    h, w = grid.shape
    for div in np.unique(grid):
        rows = list(map(int, np.flatnonzero(np.all(grid == div, axis=1))))
        cols = list(map(int, np.flatnonzero(np.all(grid == div, axis=0))))
        if not rows and not cols:
            continue
        rr, cc = [-1] + rows + [h], [-1] + cols + [w]
        heights = [b - a - 1 for a, b in zip(rr, rr[1:])]
        widths = [b - a - 1 for a, b in zip(cc, cc[1:])]
        if min(heights + widths) < 4 or len(set(heights)) != 1 or len(set(widths)) != 1:
            continue
        panels = [grid[r + 1:rr[i + 1], c + 1:cc[j + 1]].copy()
                  for i, r in enumerate(rr[:-1]) for j, c in enumerate(cc[:-1])]
        if not 2 <= len(panels) <= 6 or any(np.any(p == div) for p in panels):
            continue
        corners = {int(p[r, c]) for p in panels for r, c in ((0, 0), (0, -1), (-1, 0), (-1, -1))}
        if len(corners) != 1:
            continue
        bg = next(iter(corners))
        if bg == div or any(len(np.unique(p)) != 3 for p in panels):
            continue
        candidates.append(dict(panels=panels, background=bg, divider=int(div),
                               layout=(len(heights), len(widths))))
    assert len(candidates) == 1, "one visible equal-card partition required"
    return candidates[0]


def _panel_roles(panel, bg):
    roles = [(int(c), int(np.count_nonzero(panel == c))) for c in np.unique(panel) if c != bg]
    sources = [c for c, n in roles if n >= 2]
    markers = [c for c, n in roles if n == 1]
    assert len(sources) == len(markers) == 1
    return sources[0], markers[0]


def _complete_card(panel, bg, axis):
    p = panel if axis == "H" else panel.T
    h, w = p.shape
    source, companion = _panel_roles(p, bg)
    points = [tuple(map(int, q)) for q in np.argwhere(p == source)]
    ordered, stroke, chords = _infer_path(points)
    marker = tuple(map(int, np.argwhere(p == companion)[0]))
    anchors = [q for q in points if max(abs(q[0] - marker[0]), abs(q[1] - marker[1])) == 1]
    assert len(anchors) == 1, "marker must select one landmark"
    dr, dc = marker[0] - anchors[0][0], marker[1] - anchors[0][1]
    assert dr in (-1, 1) and dc == 0, "offset must be perpendicular to the relay"
    assert marker not in stroke
    copy = {(r + dr, c) for r, c in stroke}
    assert marker in copy
    assert all(0 <= r < h and 0 <= c < w for r, c in stroke | copy)
    assert all(0 < r < h - 1 for r, c in stroke | copy), "non-chain sides must stay clear"
    incoming = tuple(sorted(r for r, c in stroke if c == 0))
    raw_exit = tuple(sorted(r for r, c in stroke if c == w - 1))
    outgoing = tuple(r + dr for r in raw_exit)
    assert len(incoming) <= 1 and len(outgoing) <= 1
    assert incoming or outgoing
    completed = p.copy()
    for q in copy - stroke:
        if p[q] == bg:
            completed[q] = companion
    for q in stroke:
        if p[q] == bg:
            completed[q] = source
    visible = sum(p[q] == bg for q in copy - stroke)
    overlap = sum(p[q] == bg for q in copy & stroke)
    long_stroke = set()
    for a, b in zip(ordered, ordered[1:]):
        if len(_segment(a, b)) > 2:
            long_stroke.update(_segment(a, b))
    long_copy = {(r + dr, c) for r, c in long_stroke}
    unit_copy_difference = sum(p[q] == bg for q in copy - long_copy - stroke)
    assert visible > 0
    info = dict(source=source, companion=companion, points=ordered, stroke=stroke,
                copy=copy, offset=dr, incoming=incoming, outgoing=outgoing,
                raw_exit=raw_exit, chords=len(chords), overlap=overlap,
                visible=visible, marker=marker,
                unit_copy_difference=unit_copy_difference,
                vertical=any(a[1] == b[1] for a, b in zip(ordered, ordered[1:])),
                diagonal=any(a[0] != b[0] and a[1] != b[1] for a, b in zip(ordered, ordered[1:])),
                unit_links=sum(len(_segment(a, b)) == 2 for a, b in zip(ordered, ordered[1:])))
    return completed if axis == "H" else completed.T.copy(), info


def _orders(signatures):
    starts = [i for i, (a, b) in enumerate(signatures) if not a and b]
    ends = [i for i, (a, b) in enumerate(signatures) if a and not b]
    if len(starts) != 1 or len(ends) != 1 or any(not a and not b for a, b in signatures):
        return []
    answers = []

    def visit(order, remaining):
        if len(answers) >= 2:
            return
        if not remaining:
            if order[-1] == ends[0]:
                answers.append(tuple(order))
            return
        outgoing = signatures[order[-1]][1]
        if not outgoing:
            return
        for j in remaining:
            if signatures[j][0] == outgoing:
                visit(order + [j], tuple(k for k in remaining if k != j))

    visit([starts[0]], tuple(i for i in range(len(signatures)) if i != starts[0]))
    return answers


def _decode(input_grid):
    inp = _check_grid(input_grid)
    parsed = _partition(inp)
    panels, bg = parsed["panels"], parsed["background"]
    axes = []
    for axis in ("H", "V"):
        oriented = [p if axis == "H" else p.T for p in panels]
        source_masks = [p == _panel_roles(p, bg)[0] for p in oriented]
        if (all(not m[0].any() and not m[-1].any() for m in source_masks)
                and any(m[:, 0].any() or m[:, -1].any() for m in source_masks)):
            axes.append(axis)
    assert len(axes) == 1, "source ports must determine one relay axis"
    axis = axes[0]
    finished, infos = [], []
    for panel in panels:
        complete, info = _complete_card(panel, bg, axis)
        finished.append(complete)
        infos.append(info)
    signatures = [(i["incoming"], i["outgoing"]) for i in infos]
    orders = _orders(signatures)
    assert len(orders) == 1, "unique full companion-to-source chain required"
    parsed.update(axis=axis, finished=finished, infos=infos, signatures=signatures, order=orders[0])
    return parsed


def _layout(panels, order, layout, divider):
    nr, nc = layout
    h, w = panels[0].shape
    assert nr * nc == len(panels)
    grid = np.full((nr * (h + 1) - 1, nc * (w + 1) - 1), divider, dtype=np.int8)
    assert all(n <= 30 for n in grid.shape)
    for slot, j in enumerate(order):
        r, c = divmod(slot, nc)
        grid[r * (h + 1):r * (h + 1) + h, c * (w + 1):c * (w + 1) + w] = panels[j]
    return grid


def _relay_output(panels, infos, order, axis, divider):
    n = len(panels)
    strip = _layout(panels, order, (1, n) if axis == "H" else (n, 1), divider)
    oriented = strip if axis == "H" else strip.T
    width = panels[0].shape[1] if axis == "H" else panels[0].shape[0]
    for pos, j in enumerate(order[:-1]):
        for row in infos[j]["outgoing"]:
            oriented[row, (pos + 1) * (width + 1) - 1] = infos[j]["companion"]
    return strip


def solve(input_grid: np.ndarray) -> np.ndarray:
    parsed = _decode(input_grid)
    return _relay_output(parsed["finished"], parsed["infos"], parsed["order"],
                         parsed["axis"], parsed["divider"])


def _compress(points):
    points = [p for i, p in enumerate(points) if i == 0 or p != points[i - 1]]
    result = []
    for p in points:
        result.append(p)
        while len(result) >= 3:
            a, b, c = result[-3:]
            if _cross(a, b, c) != 0:
                break
            result.pop(-2)
    return result


def _walk(rng, a, b, left, right, h):
    """Random bounded diagonal/horizontal bridge, compressed to landmarks."""
    row = a
    dense = [(a, left)]
    for col in range(left + 1, right + 1):
        remaining = right - col
        choices = [r for r in (row - 1, row, row + 1)
                   if 2 <= r <= h - 3 and abs(b - r) <= remaining]
        row = int(_choose(rng, choices))
        dense.append((row, col))
    return _compress(dense)


def _stairs(rng, a, b, left, right, h):
    slots = list(range(left + 1, right))
    assert slots
    k = int(rng.integers(1, min(3, len(slots)) + 1))
    columns = sorted(map(int, rng.choice(slots, k, replace=False)))
    points, row = [(a, left)], a
    for index, col in enumerate(columns):
        points.append((row, col))
        new = b if index == k - 1 else int(rng.integers(2, h - 2))
        if new != row:
            points.append((new, col))
        row = new
    points.append((b, right))
    return _compress(points)


def _marker_options(points, stroke, d):
    return [(r + d, c) for r, c in points
            if (r + d, c) not in stroke and
            sum(max(abs(r + d - a), abs(c - b)) == 1 for a, b in points) == 1]


def _make_path(rng, a, b, left, right, h, d, kind=None, overlap=False):
    for attempt in range(100):
        if kind == "bow":
            assert a == b and right - left >= 4
            t = int(rng.integers(2, (right - left) // 2 + 1))
            signs = [s for s in (-1, 1) if 2 <= a + s * t <= h - 3]
            if not signs:
                continue
            s = int(_choose(rng, signs))
            points = [(a, left), (a + s * t, left + t), (a, left + 2 * t)]
            if left + 2 * t < right:
                points.append((a, right))
        else:
            stairs = overlap or kind == "stairs" or (kind is None and rng.random() < .50)
            points = _stairs(rng, a, b, left, right, h) if stairs else _walk(rng, a, b, left, right, h)
        try:
            ordered, stroke, chords = _infer_path(points)
            if ordered != points or not any(q not in points for q in stroke):
                continue
            markers = _marker_options(points, stroke, d)
            if not markers:
                continue
            copy = {(r + d, c) for r, c in stroke}
            if overlap and (not any(q not in points for q in stroke & copy) or not copy - stroke):
                continue
            if overlap:
                long = set().union(*(set(_segment(p, q)) for p, q in zip(points, points[1:])
                                     if len(_segment(p, q)) > 2))
                if not copy - {(r + d, c) for r, c in long} - stroke - set(markers):
                    continue
            return points, stroke, copy, _choose(rng, markers)
        except AssertionError:
            continue
    if overlap:
        # A guaranteed staircase witness: unit-only end stubs, a long vertical
        # run with visible overlap, and an unambiguous marker beyond its apex.
        assert b == a - d and right - left >= 4
        points = [(a, left), (a, left + 1), (a + 2 * d, left + 1),
                  (a + 2 * d, right - 1), (b, right - 1), (b, right)]
        _, stroke, _ = _infer_path(points)
        marker = (a + 3 * d, left + 1)
        assert marker in _marker_options(points, stroke, d)
        return points, stroke, {(r + d, c) for r, c in stroke}, marker
    if kind == "bow":
        points = [(a, left), (a + 2, left + 2), (a, left + 4)]
        if left + 4 < right:
            points.append((a, right))
        _, stroke, _ = _infer_path(points)
        markers = _marker_options(points, stroke, d)
        assert markers
        return points, stroke, {(r + d, c) for r, c in stroke}, _choose(rng, markers)
    # Try both positions of a diagonal transition. The late transition is
    # needed for a two-step terminal whose early bend crowds the marker.
    options = [[(a, left), (b, right)]] if a == b or abs(a - b) == right - left else [
        [(a, left), (b, left + abs(a - b)), (b, right)],
        [(a, left), (a, right - abs(a - b)), (b, right)]]
    for points in options:
        _, stroke, _ = _infer_path(points)
        markers = _marker_options(points, stroke, d)
        if markers:
            return points, stroke, {(r + d, c) for r, c in stroke}, _choose(rng, markers)
    raise AssertionError("no simple bridge marker")


def _links(rng, n, h, w, ds, repeated):
    pool = list(range(3, h - 3))
    for _ in range(100):
        links = [int(_choose(rng, pool)) for _ in range(n - 1)]
        if repeated and len(set(links)) == len(links):
            continue
        if not repeated and len(set(links)) < len(links):
            continue
        if any(abs(links[i - 1] - (links[i] - ds[i])) > w - 1 for i in range(1, n - 1)):
            continue
        sigs = list(zip([()] + [(k,) for k in links], [(k,) for k in links] + [()]))
        if len(_orders(sigs)) == 1:
            return links
    # Distinct consecutive blocks with at most one loop per key are unique.
    count = (n - 1 + 1) // 2 if repeated else n - 1
    assert count <= len(pool)
    start = int(rng.integers(0, len(pool) - count + 1))
    selected = pool[start:start + count]
    if rng.random() < .5:
        selected.reverse()
    lengths = [1] * count
    for i in map(int, rng.choice(count, n - 1 - count, replace=False)):
        lengths[i] += 1
    return [k for k, size in zip(selected, lengths) for _ in range(size)]


def _build_pair(rng, axis, plan, witness_sign):
    witness = plan in ("bow", "overlap")
    n = 3 if plan == "bow" else 4 if plan == "overlap" else int(_choose(rng, [5, 6] if plan == "test" else [2, 3, 4]))
    max_width = min(10, 31 // n - 1)
    w = int(rng.integers(5 if witness else 4, max_width + 1))
    h = int(rng.integers(10 if witness else max(8, n + 3), 15))
    ds = [int(_choose(rng, (-1, 1))) for _ in range(n)]
    if plan == "bow":
        ds[1] = witness_sign
        a = int(rng.integers(4, h - 4))
        links = [a, a + ds[1]]
    elif plan == "overlap":
        ds[1] = witness_sign
        ds[2] = witness_sign  # guarantees a join where occupied-union matching fails
        a = int(rng.integers(4, h - 4))
        b = a + int(_choose(rng, (-1, 1)))
        links = [a, a, b]
    else:
        # Small transverse heights have fewer keys; admit repetitions rather
        # than requiring impossible distinct signatures.
        repeated = bool(rng.random() < .55 or h - 6 < n - 1)
        links = _links(rng, n, h, w, ds, repeated)
    palette = list(map(int, rng.permutation(10)))
    bg, div = palette[:2]
    colors = palette[2:]
    mode = "shared" if plan == "overlap" else str(_choose(rng, ("shared", "partial", "distinct", "role_swap")))
    if mode == "shared":
        source_colors = [colors[0]] * n
        companion_colors = [colors[1]] * n
    elif mode == "role_swap":
        source_colors = [colors[i % 2] for i in range(n)]
        companion_colors = [colors[1 - i % 2] for i in range(n)]
    elif mode == "distinct":
        source_colors = colors[:n]
        companion_colors = [int(_choose(rng, [c for c in colors if c != s])) for s in source_colors]
    else:
        source_colors = [int(_choose(rng, colors[:3])) for _ in range(n)]
        companion_colors = [int(_choose(rng, [c for c in colors if c != s])) for s in source_colors]
    sparse, finished, infos = [], [], []
    for i in range(n):
        a = links[0] - ds[0] if i == 0 else links[i - 1]
        b = links[-1] if i == n - 1 else links[i] - ds[i]
        left, right = (1, w - 1) if i == 0 else (0, w - 2) if i == n - 1 else (0, w - 1)
        # Terminal closed endpoints may bend, but remain reachable and in bounds.
        if i == 0:
            a = int(_choose(rng, [r for r in range(2, h - 2) if abs(r - b) <= right - left]))
        elif i == n - 1:
            b = int(_choose(rng, [r for r in range(2, h - 2) if abs(r - a) <= right - left]))
        kind = "bow" if plan == "bow" and i == 1 else None
        points, stroke, copy, marker = _make_path(rng, a, b, left, right, h, ds[i], kind,
                                                overlap=plan == "overlap" and i == 1)
        panel = np.full((h, w), bg, dtype=np.int8)
        for q in points:
            panel[q] = source_colors[i]
        panel[marker] = companion_colors[i]
        completed = panel.copy()
        for q in copy - stroke:
            if panel[q] == bg:
                completed[q] = companion_colors[i]
        for q in stroke:
            if panel[q] == bg:
                completed[q] = source_colors[i]
        sparse.append(panel if axis == "H" else panel.T.copy())
        finished.append(completed if axis == "H" else completed.T.copy())
        infos.append(dict(outgoing=() if i == n - 1 else (links[i],), companion=companion_colors[i]))
    ph, pw = sparse[0].shape
    layouts = [(nr, n // nr) for nr in range(1, n + 1) if n % nr == 0
               and nr * (ph + 1) - 1 <= 30 and (n // nr) * (pw + 1) - 1 <= 30]
    layout = _choose(rng, layouts)
    permutation = list(map(int, rng.permutation(n)))
    if witness and permutation == list(range(n)):
        permutation.reverse()
    inp = _layout(sparse, permutation, layout, div)
    out = _relay_output(finished, infos, list(range(n)), axis, div)
    decoded = _decode(inp)
    assert np.array_equal(out, _relay_output(decoded["finished"], decoded["infos"], decoded["order"], axis, div))
    return {"input": inp, "output": out}


def _measure(pair):
    parsed = _decode(pair["input"])
    infos, order = parsed["infos"], parsed["order"]
    assert np.array_equal(pair["output"], _relay_output(parsed["finished"], infos, order, parsed["axis"], parsed["divider"]))
    # Check content conservation in the translated card slots, independently
    # of whether whole input and output happen to have the same dimensions.
    oriented = pair["output"] if parsed["axis"] == "H" else pair["output"].T
    w = parsed["panels"][0].shape[1] if parsed["axis"] == "H" else parsed["panels"][0].shape[0]
    for pos, j in enumerate(order):
        original = parsed["panels"][j] if parsed["axis"] == "H" else parsed["panels"][j].T
        result = oriented[:, pos * (w + 1):pos * (w + 1) + w]
        assert np.array_equal(result[original != parsed["background"]], original[original != parsed["background"]])
        for r, c in infos[j]["stroke"]:
            assert result[r, c] == infos[j]["source"]
    source_only_diff = any(infos[j]["raw_exit"] != infos[k]["incoming"] for j, k in zip(order, order[1:]))
    union_diff = any(tuple(sorted(set(infos[j]["raw_exit"] + infos[j]["outgoing"]))) !=
                     tuple(sorted(set(infos[k]["incoming"] + tuple(r + infos[k]["offset"] for r in infos[k]["incoming"]))))
                     for j, k in zip(order, order[1:]))
    joins = [infos[j]["outgoing"] for j in order[:-1]]
    return dict(n=len(infos), axis=parsed["axis"], layout=parsed["layout"],
                panel_shape=parsed["panels"][0].shape, background=parsed["background"], divider=parsed["divider"],
                reordered=order != tuple(range(len(infos))), source_only_diff=source_only_diff,
                union_diff=union_diff, repeated=len(set(joins)) < len(joins),
                shared=len({i["source"] for i in infos}) < len(infos),
                signs={i["offset"] for i in infos}, chords=sum(i["chords"] for i in infos),
                overlap=sum(i["overlap"] for i in infos), diagonal=any(i["diagonal"] for i in infos),
                vertical=any(i["vertical"] for i in infos),
                unit_links=sum(i["unit_links"] for i in infos),
                unit_copy_difference=sum(i["unit_copy_difference"] for i in infos),
                geometry=tuple(sorted(tuple(i["points"]) for i in infos)),
                role_swaps=bool({i["source"] for i in infos} & {i["companion"] for i in infos}))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _check_grid(pair["input"]), _check_grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            features[split].append(_measure(dict(input=inp, output=out)))
    train = features["train"]
    assert any(f["n"] == 3 and f["chords"] and f["diagonal"] and f["reordered"] for f in train)
    assert any(f["n"] == 4 and f["repeated"] and f["overlap"] and f["vertical"] and
               f["unit_copy_difference"] and f["reordered"] for f in train)
    assert any(f["source_only_diff"] and f["union_diff"] for f in train)
    assert any(f["shared"] for f in train)
    assert set.union(*(f["signs"] for f in train)) == {-1, 1}
    assert len({f["background"] for f in train}) >= 2
    assert all(f["n"] <= 4 for f in train) and all(f["n"] >= 5 for f in features["test"])


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    first_axis = str(_choose(rng, ("H", "V")))
    mixed = bool(rng.random() < .7)
    sign = int(_choose(rng, (-1, 1)))
    witnesses = list(map(int, rng.choice(num_train, 2, replace=False)))
    plans = ["ordinary"] * num_train + ["test"] * num_test
    plans[witnesses[0]], plans[witnesses[1]] = "bow", "overlap"
    puzzle, seen, witness_bg = {"train": [], "test": []}, set(), None
    # Background diversity is checked locally; retries change visible content,
    # never the rule. Bounded fresh per-pair attempts avoid unlucky rich paths.
    for index, plan in enumerate(plans):
        axis = str(_choose(rng, ("H", "V"))) if mixed else first_axis
        local_sign = sign if plan == "bow" else -sign if plan == "overlap" else sign
        for _ in range(120):
            try:
                pair = _build_pair(rng, axis, plan, local_sign)
                feature = _measure(pair)
                if _key(pair["input"]) in seen:
                    continue
                if plan in ("bow", "overlap") and witness_bg is not None and feature["background"] == witness_bg:
                    continue
                if plan == "overlap" and not feature["union_diff"]:
                    continue
                break
            except AssertionError:
                continue
        else:
            raise AssertionError("bounded distinct-pair generation exhausted")
        if plan in ("bow", "overlap"):
            witness_bg = feature["background"]
        seen.add(_key(pair["input"]))
        puzzle["train" if index < num_train else "test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s]) for s in ("train", "test"))


def self_test():
    """Bounded 42-seed variation and invariant checks, plus targeted mutations."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    rows, structures, shapes = [], set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _measure(pair)
                f["split"] = split
                rows.append(f)
                shapes.add(pair["input"].shape)
                structures.add((f["axis"], f["layout"], f["geometry"]))
                assert np.array_equal(solve(pair["input"]), pair["output"])
    assert {r["n"] for r in rows} == {2, 3, 4, 5, 6}
    assert {r["axis"] for r in rows} == {"H", "V"}
    assert len({r["panel_shape"] for r in rows}) >= 20
    assert len(shapes) >= 30 and len(structures) >= 150
    assert {r["background"] for r in rows} == set(range(10))
    assert {r["divider"] for r in rows} == set(range(10))
    assert any(min(r["layout"]) > 1 for r in rows)
    assert any(r["role_swaps"] for r in rows)
    assert any(r["unit_links"] for r in rows)
    assert any(not r["reordered"] for r in rows)
    for split in ("train", "test"):
        selected = [r for r in rows if r["split"] == split]
        for field in ("repeated", "shared", "vertical", "diagonal", "overlap", "chords"):
            assert any(r[field] for r in selected), (split, field)
    for i, (a, b) in enumerate(((2, 1), (10, 1), (10, 4))):
        p = generate_puzzle(1000 + i, a, b)
        assert (len(p["train"]), len(p["test"])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Exact transformation symmetries are meaningful input-only checks.
    pair = puzzles[0]["test"][0]
    assert np.array_equal(solve(pair["input"].T), pair["output"].T)
    mapping = np.array([3, 6, 1, 9, 0, 8, 2, 5, 7, 4], dtype=np.int8)
    assert np.array_equal(solve(mapping[pair["input"]]), mapping[pair["output"]])
    a, b = (3,), (4,)
    assert len(_orders([((), a), (a, a), (a, b), (b, ())])) == 1
    assert len(_orders([((), a), (a, a), (a, a), (a, ())])) == 2
    # Delete an original card cell, erase a bridge, or reverse occlusion.
    original = puzzles[0]
    for mutation in ("original", "bridge", "priority"):
        clone = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in original[s]] for s in original}
        candidates = clone["train"] if mutation == "priority" else clone["test"]
        for target in candidates:
            parsed = _decode(target["input"])
            order, infos = parsed["order"], parsed["infos"]
            out = target["output"] if parsed["axis"] == "H" else target["output"].T
            w = parsed["panels"][0].shape[1] if parsed["axis"] == "H" else parsed["panels"][0].shape[0]
            if mutation == "bridge":
                out[infos[order[0]]["outgoing"][0], w] = parsed["divider"]
                break
            found = False
            for pos, j in enumerate(order):
                panel = parsed["panels"][j] if parsed["axis"] == "H" else parsed["panels"][j].T
                cells = list(infos[j]["points"]) if mutation == "original" else [q for q in infos[j]["stroke"] & infos[j]["copy"] if panel[q] == parsed["background"]]
                if cells:
                    r, c = sorted(cells)[0]
                    out[r, pos * (w + 1) + c] = parsed["background"] if mutation == "original" else infos[j]["companion"]
                    found = True
                    break
            if found:
                break
        try:
            validate_generated_puzzle(clone)
        except AssertionError:
            continue
        raise AssertionError("bad output mutation accepted: " + mutation)


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds)")
