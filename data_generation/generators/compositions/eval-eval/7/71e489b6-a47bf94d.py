"""Two-ended fault signatures: rectify panels, trace cables, superpose stencils.

Structural colors are fixed: 0 field, 5 panel, 8 cable, 9 crossing/conflict.
The remaining six colors label the diamond and checker endpoints. Construction
uses explicit paths and masks; solve parses only its input. No parent imports.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIAMOND = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
CHECKER = ~DIAMOND
LABELS = (1, 2, 3, 4, 6, 7)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=True)


def _components(points):
    left = set(points)
    result = []
    while left:
        p = min(left)
        left.remove(p)
        found, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                q = (r + dr, c + dc)
                if q in left:
                    left.remove(q)
                    found.add(q)
                    stack.append(q)
        result.append(found)
    return result


def _signature(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _rect_cells(rect):
    r0, r1, c0, c1 = rect
    return {(r, c) for r in range(r0, r1) for c in range(c0, c1)}


def _terminals(g):
    """Recognize the visible endpoint grammar, with no bank assumption."""
    found, occupied = [], set()
    for r in range(g.shape[0] - 2):
        for c in range(g.shape[1] - 2):
            b = g[r:r + 3, c:c + 3]
            for name, stencil in (("diamond", DIAMOND), ("checker", CHECKER)):
                label = int(b[stencil][0])
                if (label in LABELS and np.all(b[stencil] == label)
                        and np.all(b[~stencil] == 0)):
                    cells = _rect_cells((r, r + 3, c, c + 3))
                    if occupied & cells:
                        raise ValueError("overlapping terminal footprints")
                    occupied |= cells
                    center = (r + 1, c + 1)
                    ports = [(center[0] + 2 * dr, center[1] + 2 * dc)
                             for dr, dc in DIRS]
                    ports = [p for p in ports if 0 <= p[0] < g.shape[0]
                             and 0 <= p[1] < g.shape[1] and g[p] == 8]
                    if len(ports) != 1:
                        raise ValueError("terminal needs exactly one cable port")
                    found.append(dict(kind=name, color=label, port=ports[0],
                                      center=center, cells=cells))
    label_cells = set(map(tuple, np.argwhere(np.isin(g, LABELS))))
    if label_cells - occupied or not 4 <= len(found) <= 8:
        raise ValueError("unrecognized endpoint labels")
    return found, occupied


def _panels(g):
    """Long supported rectangle sides distinguish panels from <=2-cell specks."""
    panels, extras = [], set()
    for comp in _components(map(tuple, np.argwhere(g == 5))):
        if len(comp) <= 2:
            extras |= comp
            continue
        rr, cc = zip(*comp)
        rect = (min(rr), max(rr) + 1, min(cc), max(cc) + 1)
        r0, r1, c0, c1 = rect
        if r1 - r0 < 3 or c1 - c0 < 3:
            raise ValueError("unsupported solid fragment")
        cells = _rect_cells(rect)
        holes = cells - comp
        corners = {(r0, c0), (r0, c1 - 1), (r1 - 1, c0), (r1 - 1, c1 - 1)}
        if (not corners <= comp or len(holes) > len(cells) // 4
                or any(g[p] != 0 for p in holes)):
            raise ValueError("panel lacks intact rectangular evidence")
        panels.append(dict(rect=rect, cells=cells, holes=holes))
    if not panels:
        raise ValueError("no diagnostic panels")
    return panels, extras


def _parse(g):
    terminals, footprints = _terminals(g)
    panels, extras = _panels(g)
    visible = set(map(tuple, np.argwhere((g == 8) | (g == 9))))
    graph = {p: set() for p in visible}
    for r, c in visible:
        for dr, dc in DIRS:
            q = (r + dr, c + dc)
            if q in visible:
                graph[r, c].add(q)
    ports = {}
    for i, t in enumerate(terminals):
        p = t["port"]
        if p in ports:
            raise ValueError("two terminals share a port")
        ports[p] = i
    for i, panel in enumerate(panels):
        node = (-1, i)
        candidates = set()
        for r, c in panel["cells"]:
            for dr, dc in DIRS:
                q = (r + dr, c + dc)
                # Only an actual interrupted strand enters a panel. A parallel
                # degree-two wire beside it is not a port.
                if q in visible and len(graph[q]) in (1, 3) and q not in ports:
                    candidates.add(q)
        if len(candidates) != 2:
            raise ValueError("panel must hide exactly one two-port connection")
        graph[node] = candidates
        panel["ports"] = candidates
        for p in candidates:
            graph[p].add(node)
    for p in visible:
        degree = len(graph[p])
        if p in ports:
            if degree != 1 or g[p] != 8:
                raise ValueError("broken endpoint lead")
        elif g[p] == 9:
            if degree != 4 or any(q[0] < 0 for q in graph[p]):
                raise ValueError("crossing must expose all four straight arms")
        elif degree != 2:
            raise ValueError("broken or branching cable")
    routes, paired, owners, used, used_edges = [], set(), {}, set(), set()
    for ti, terminal in enumerate(terminals):
        if ti in paired:
            continue
        current, previous = terminal["port"], None
        nodes, states = [], set()
        while True:
            state = (previous, current)
            if state in states:
                raise ValueError("cable cycle")
            states.add(state)
            nodes.append(current)
            used.add(current)
            if current in ports and ports[current] != ti:
                tj = ports[current]
                break
            edges = graph[current]
            if current[0] >= 0 and len(edges) == 4:
                if previous is None or previous[0] < 0:
                    raise ValueError("unreadable crossing approach")
                nxt = (2 * current[0] - previous[0], 2 * current[1] - previous[1])
                if nxt not in edges:
                    raise ValueError("crossing has no straight continuation")
            else:
                options = edges - ({previous} if previous is not None else set())
                if len(options) != 1:
                    raise ValueError("ambiguous strand")
                nxt = next(iter(options))
            used_edges.add(frozenset((current, nxt)))
            previous, current = current, nxt
        other = terminals[tj]
        if tj in paired or {terminal["kind"], other["kind"]} != {"diamond", "checker"}:
            raise ValueError("route must pair complementary endpoint forms")
        d = terminal["color"] if terminal["kind"] == "diamond" else other["color"]
        c = other["color"] if other["kind"] == "checker" else terminal["color"]
        if d == c:
            raise ValueError("the two endpoint channels must have different colors")
        ri = len(routes)
        for p in nodes:
            if p[0] < 0:
                if p[1] in owners:
                    raise ValueError("panel belongs to multiple cables")
                owners[p[1]] = ri
        paired |= {ti, tj}
        routes.append(dict(terminals=(ti, tj), nodes=nodes, diamond=d, checker=c))
    all_edges = {frozenset((p, q)) for p, adjacent in graph.items() for q in adjacent}
    if used != set(graph) or used_edges != all_edges or len(owners) != len(panels):
        raise ValueError("unattached routing or panels")
    holes = set().union(*(p["holes"] for p in panels))
    for r, c in holes:
        if any((r + dr, c + dc) in footprints or
               (0 <= r + dr < g.shape[0] and 0 <= c + dc < g.shape[1]
                and g[r + dr, c + dc] == 9)
               for dr in (-1, 0, 1) for dc in (-1, 0, 1)):
            raise ValueError("fault halo would obscure a terminal or crossing")
    return dict(terminals=terminals, footprints=footprints, panels=panels,
                extras=extras, routes=routes, owners=owners, holes=holes)


def _predict(g, info):
    result = g.copy()
    for p in info["extras"]:
        result[p] = 0
    signals = np.zeros(g.shape, dtype=np.uint16)
    for pi, panel in enumerate(info["panels"]):
        route = info["routes"][info["owners"][pi]]
        for r, c in panel["holes"]:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if (not (dr or dc) or q in info["holes"]
                            or not (0 <= q[0] < g.shape[0] and 0 <= q[1] < g.shape[1])):
                        continue
                    color = route["checker" if dr and dc else "diamond"]
                    signals[q] |= np.uint16(1 << color)
    for r, c in np.argwhere(signals):
        value = int(signals[r, c])
        result[r, c] = value.bit_length() - 1 if value & (value - 1) == 0 else 9
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover all information from input pixels; no learned or latent state."""
    g = _grid(input_grid)
    return _predict(g, _parse(g))


def _path(a, b):
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    step = 1 if b[0] >= a[0] else -1
    return [(r, a[1]) for r in range(a[0], b[0] + step, step)]


def _routing(rng, n, lesson):
    # Unequal, offset terminal banks and arbitrary ordering of turn levels.
    # Start and finish columns occupy disjoint ranges, preventing shared runs.
    pitches = [(a, b) for a in range(3 if n == 4 else 4, 8)
               for b in range(3 if n == 4 else 4, 8)
               if (n - 1) * (a + b) + 11 <= 30]
    p1, p2 = pitches[int(rng.integers(len(pitches)))]
    spare = 30 - ((n - 1) * (p1 + p2) + 11)
    gap = 6 + int(rng.integers(min(4, spare) + 1))
    slack = 30 - ((n - 1) * (p1 + p2) + gap + 5)
    left = int(rng.integers(slack + 1))
    starts = [2 + left + i * p1 for i in range(n)]
    ends = [starts[-1] + gap + i * p2 for i in range(n)]
    w = ends[-1] + 3 + int(rng.integers(30 - ends[-1] - 2))
    h = int(rng.integers({2: 19, 3: 23, 4: 27}[n], 31))
    level_slack = h - 7 - (6 + 4 * (n - 1))
    levels = np.arange(n) * 4 + 6 + np.sort(rng.integers(level_slack + 1, size=n))
    rng.shuffle(levels)
    perm = list(map(int, rng.permutation(n)))
    if lesson == 0:
        perm = list(reversed(range(n)))
    paths, terminals = [], []
    for i in range(n):
        top = int(rng.integers(2))
        bottom = h - 3 - int(rng.integers(2))
        r, s, e = int(levels[i]), starts[i], ends[perm[i]]
        path = _path((top + 3, s), (r, s))
        path += _path((r, s), (r, e))[1:]
        path += _path((r, e), (bottom - 1, e))[1:]
        paths.append(path)
        terminals.append(((top, s - 1), (bottom, e - 1)))
    visits = Counter(p for path in paths for p in path)
    field = np.zeros((h, w), dtype=np.int8)
    for p, count in visits.items():
        field[p] = 9 if count == 2 else 8
    return field, paths, terminals, starts[-1], ends[0]


def _panel_ok(rect, ri, field, paths, rectangles, terminal_cells):
    r0, r1, c0, c1 = rect
    if not (0 <= r0 < r1 <= field.shape[0] and 0 <= c0 < c1 <= field.shape[1]):
        return False
    cells = _rect_cells(rect)
    if cells & terminal_cells or any(field[p] == 9 for p in cells):
        return False
    for old in rectangles:
        a, b, x, y = old["rect"]
        if not (r1 < a or b < r0 or c1 < x or y < c0):
            return False
    if any(cells & set(path) for j, path in enumerate(paths) if j != ri):
        return False
    path = paths[ri]
    inside = [p in cells for p in path]
    transitions = [k for k in range(len(path) - 1) if inside[k] != inside[k + 1]]
    if inside[0] or inside[-1] or len(transitions) != 2:
        return False
    # A nearby parallel wire must not touch a panel's perimeter: this keeps
    # exactly the two path-cut ports readable, including hidden elbows.
    boundary = {q for r, c in cells for dr, dc in DIRS
                for q in [(r + dr, c + dc)] if q not in cells}
    ports = {p for p in boundary if 0 <= p[0] < field.shape[0]
             and 0 <= p[1] < field.shape[1] and field[p] in (8, 9)}
    true_ports = {path[transitions[0]], path[transitions[1] + 1]}
    return ports == true_ports and all(field[p] == 8 for p in ports)


def _make_panels(rng, field, paths, sites, left, right, lesson, is_test):
    rectangles = []
    terminal_cells = set().union(*(_rect_cells((r, r + 3, c, c + 3))
                                  for pair in sites for r, c in pair))
    for ri, path in enumerate(paths):
        horizontal = [p for a, p, b in zip(path, path[1:], path[2:])
                      if a[0] == p[0] == b[0]]
        row = horizontal[0][0]
        # Always retain one supported panel on each route. Try corner panels
        # too; their two ports make an occluded elbow rather than a junction.
        selected = None
        for attempt in range(40):
            if attempt < 24 and rng.random() < .6:
                center = path[min(range(len(path)), key=lambda k: abs(path[k][0] - row))]
                if rng.random() < .5:
                    center = next(p for p in reversed(path) if p[0] == row)
                hh, ww = int(rng.integers(3, 6)), int(rng.integers(3, 6))
                r0 = center[0] - int(rng.integers(1, hh - 1))
                c0 = center[1] - int(rng.integers(1, ww - 1))
            else:
                hh = 3
                ww = int(rng.integers(3, right - left - 2))
                r0 = row - 1
                c0 = int(rng.integers(left + 2, right - ww))
            rect = (r0, r0 + hh, c0, c0 + ww)
            if _panel_ok(rect, ri, field, paths, rectangles, terminal_cells):
                selected = dict(rect=rect, route=ri)
                break
        if selected is None:
            raise RuntimeError("primary panel did not fit")
        rectangles.append(selected)
    # Additional panels can cover straight leads, tails or bends, at variable
    # distances. Proposals are bounded and never delete a primary panel.
    wanted = int(rng.integers(0, 3 if is_test else 2))
    for _ in range(50):
        if wanted == 0:
            break
        ri = int(rng.integers(len(paths)))
        p = paths[ri][int(rng.integers(len(paths[ri])))]
        hh, ww = map(int, rng.integers(3, 7, size=2))
        r0, c0 = p[0] - hh // 2, p[1] - ww // 2
        rect = (r0, r0 + hh, c0, c0 + ww)
        if _panel_ok(rect, ri, field, paths, rectangles, terminal_cells):
            rectangles.append(dict(rect=rect, route=ri))
            wanted -= 1
    return rectangles, terminal_cells


def _choose_holes(rng, rect, mode, field, terminal_cells):
    r0, r1, c0, c1 = rect
    cells = _rect_cells(rect)
    corners = {(r0, c0), (r0, c1 - 1), (r1 - 1, c0), (r1 - 1, c1 - 1)}
    available = []
    for p in sorted(cells - corners):
        if all((p[0] + dr, p[1] + dc) not in terminal_cells and
               not (0 <= p[0] + dr < field.shape[0] and
                    0 <= p[1] + dc < field.shape[1] and field[p[0] + dr, p[1] + dc] == 9)
               for dr in (-1, 0, 1) for dc in (-1, 0, 1)):
            available.append(p)
    motifs = {"single": [(0, 0)], "edge": [(0, 0)],
              "orth": [(0, 0), (0, 1)], "diag": [(0, 0), (1, 1)],
              "L": [(0, 0), (1, 0), (1, 1)],
              "separate": [(0, 0), (0, 2)], "clean": []}
    if mode == "clean":
        return set()
    offsets = motifs[mode]
    if rng.random() < .5:
        offsets = [(c, r) for r, c in offsets]
    choices = []
    allowed = set(available)
    for r, c in available:
        holes = {(r + dr, c + dc) for dr, dc in offsets}
        if not holes <= allowed or len(holes) > len(cells) // 4:
            continue
        if mode == "edge" and not any(y in (r0, r1 - 1) or x in (c0, c1 - 1) for y, x in holes):
            continue
        remaining = cells - holes
        if len(_components(remaining)) == 1:
            choices.append(holes)
    if not choices:
        if mode != "single":
            return _choose_holes(rng, rect, "single", field, terminal_cells)
        raise RuntimeError("no readable fault location")
    return choices[int(rng.integers(len(choices)))]


def _constructed_output(inp, rectangles, channel_colors, extras):
    """Ground truth from construction, separate from input graph and bit masks."""
    out = inp.copy()
    for p in extras:
        out[p] = 0
    missing = set().union(*(p["holes"] for p in rectangles))
    votes = {}
    for panel in rectangles:
        diamond, checker = channel_colors[panel["route"]]
        for y, x in panel["holes"]:
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)):
                q = (y + dy, x + dx)
                if q not in missing and 0 <= q[0] < out.shape[0] and 0 <= q[1] < out.shape[1]:
                    votes.setdefault(q, set()).add(checker if dy * dx else diamond)
    for q, colors in votes.items():
        out[q] = next(iter(colors)) if len(colors) == 1 else 9
    return out


def _make_pair(rng, n, lesson, is_test, rotation, combined_faults=False):
    field, paths, sites, left, right = _routing(rng, n, lesson)
    rectangles, terminal_cells = _make_panels(rng, field, paths, sites, left, right, lesson, is_test)
    inp = field.copy()
    channel_colors = []
    for ri, pair in enumerate(sites):
        colors = list(map(int, rng.choice(LABELS, 2, replace=False)))
        if ri == 1 and colors == channel_colors[0]:
            colors.reverse()
        channel_colors.append(tuple(colors))
        reverse = bool(rng.integers(2))
        for side, (r, c) in enumerate(pair):
            di = (side == 0) != reverse
            stencil, color = (DIAMOND, colors[0]) if di else (CHECKER, colors[1])
            inp[r:r + 3, c:c + 3] = np.where(stencil, color, 0)
    for pi, panel in enumerate(rectangles):
        r0, r1, c0, c1 = panel["rect"]
        if lesson == 0:
            mode = "single" if pi < 2 else "clean"
        elif lesson == 1:
            mode = ("orth", "edge", "diag")[pi % 3]
        else:
            choices = ("single", "orth", "diag", "edge", "separate", "clean")
            if is_test:
                choices += ("L",)
            mode = str(rng.choice(choices))
        if combined_faults and pi < 2:
            mode = "L" if pi == 0 else "orth"
        panel["holes"] = _choose_holes(rng, panel["rect"], mode, field, terminal_cells)
        if combined_faults and pi == 0 and len(panel["holes"]) != 3:
            raise RuntimeError("test needs a readable three-cell fault composition")
        inp[r0:r1, c0:c1] = 5
        for p in panel["holes"]:
            inp[p] = 0
    if not any(p["holes"] for p in rectangles):
        rectangles[0]["holes"] = _choose_holes(rng, rectangles[0]["rect"], "single", field, terminal_cells)
        for p in rectangles[0]["holes"]:
            inp[p] = 0
    extras = set()
    all_panel_cells = set().union(*(_rect_cells(p["rect"]) for p in rectangles))
    forbidden = all_panel_cells | terminal_cells
    # Specks do not touch panels, cables or terminal footprints. Their small
    # size, rather than position on the canvas, identifies them as corruption.
    wanted = int(rng.integers(1 if lesson in (0, 1) else 0, 6))
    candidates = []
    for r, c in np.argwhere(inp == 0):
        p = (int(r), int(c))
        if p in forbidden:
            continue
        if any((r + dr, c + dc) in forbidden or
               (0 <= r + dr < inp.shape[0] and 0 <= c + dc < inp.shape[1]
                and inp[r + dr, c + dc] in (8, 9)) for dr, dc in DIRS):
            continue
        candidates.append(p)
    allowed = set(candidates)
    for ix in rng.permutation(len(candidates)):
        if len(extras) >= wanted:
            break
        p = candidates[int(ix)]
        if p in extras or any((p[0] + dr, p[1] + dc) in extras for dr, dc in DIRS):
            continue
        group = {p}
        if len(extras) + 2 <= wanted and rng.random() < .4:
            neighbors = [(p[0] + dr, p[1] + dc) for dr, dc in DIRS]
            neighbors = [q for q in neighbors if q in allowed and q not in extras
                         and not any((q[0] + dr, q[1] + dc) in extras for dr, dc in DIRS)]
            if neighbors:
                group.add(neighbors[int(rng.integers(len(neighbors)))])
        extras |= group
    if lesson in (0, 1) and not extras:
        raise RuntimeError("no space for teaching speck erasure")
    for p in extras:
        inp[p] = 5
    out = _constructed_output(inp, rectangles, channel_colors, extras)
    info = _parse(inp)
    # Check intended geometry against the returned pixels, before accepting.
    assert {p["rect"] for p in rectangles} == {p["rect"] for p in info["panels"]}
    assert info["extras"] == extras
    for panel in info["panels"]:
        original = next(p for p in rectangles if p["rect"] == panel["rect"])
        route = info["routes"][info["owners"][info["panels"].index(panel)]]
        assert (route["diamond"], route["checker"]) == channel_colors[original["route"]]
    if lesson == 0 and not _nearest_wrong(info):
        raise RuntimeError("need a routing counterexample to proximity matching")
    if lesson == 1:
        features = _features(inp, info)
        if not all(features[k] for k in ("conflicts", "orthogonal", "diagonal", "boundary", "elbows")):
            raise RuntimeError("need visible superposition, boundary, diagonal and elbow lessons")
    if rng.random() < .5:
        inp, out = np.fliplr(inp), np.fliplr(out)
    return {"input": np.rot90(inp, rotation).copy(), "output": np.rot90(out, rotation).copy()}


def _nearest_wrong(info):
    for pi, panel in enumerate(info["panels"]):
        if not panel["holes"]:
            continue
        route = info["routes"][info["owners"][pi]]
        r0, r1, c0, c1 = panel["rect"]
        center = ((r0 + r1 - 1) / 2, (c0 + c1 - 1) / 2)
        for kind in ("diamond", "checker"):
            candidates = [t for t in info["terminals"] if t["kind"] == kind]
            distances = [sum((a - b) ** 2 for a, b in zip(center, t["center"])) for t in candidates]
            winners = [t for t, d in zip(candidates, distances) if d == min(distances)]
            if all(t["color"] != route[kind] for t in winners):
                return True
    return False


def _features(g, info=None):
    info = _parse(g) if info is None else info
    holes = info["holes"]
    boundary = clipped = orth = diagonal = elbows = 0
    for panel in info["panels"]:
        r0, r1, c0, c1 = panel["rect"]
        for r, c in panel["holes"]:
            boundary += int(r in (r0, r1 - 1) or c in (c0, c1 - 1))
            clipped += int(r in (0, g.shape[0] - 1) or c in (0, g.shape[1] - 1))
            orth += int((r + 1, c) in holes or (r, c + 1) in holes)
            diagonal += int((r + 1, c + 1) in holes or (r + 1, c - 1) in holes)
        a, b = sorted(panel["ports"])
        elbows += int(a[0] != b[0] and a[1] != b[1])
    predicted = _predict(g, info)
    return dict(routes=len(info["routes"]), panels=len(info["panels"]),
                holes=len(holes), extras=len(info["extras"]), boundary=boundary,
                clipped=clipped, orthogonal=orth, diagonal=diagonal, elbows=elbows,
                crossings=int(np.count_nonzero(g == 9)),
                conflicts=int(np.count_nonzero((predicted == 9) & (g != 9))),
                nearest_wrong=_nearest_wrong(info),
                panel_shapes=tuple((p["rect"][1] - p["rect"][0],
                                    p["rect"][3] - p["rect"][2]) for p in info["panels"]),
                channel_pairs=tuple((r["diamond"], r["checker"]) for r in info["routes"]))


def _counts(num_train, num_test):
    for name, value, lower, upper in (("num_train", num_train, 2, 10),
                                      ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lower <= value <= upper):
            raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))  # count, combined faults, orientation
    base_rotation = int(rng.integers(4))
    result, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            lesson = index if split == "train" and index < 2 else -1
            is_test = split == "test"
            n = (2 if lesson == 0 else 3 if lesson == 1 else
                 4 if is_test and regime == 0 else int(rng.integers(2, 4 if regime == 0 else 5)))
            rotation = (base_rotation + (1 if is_test else 0) + 2 * int(rng.integers(2))) % 4 if regime == 2 else int(rng.integers(4))
            for _ in range(100):
                try:
                    pair = _make_pair(rng, n, lesson, is_test, rotation,
                                      combined_faults=is_test and regime == 1)
                except (RuntimeError, ValueError):
                    continue
                if is_test:
                    f = _features(pair["input"])
                    if f["clipped"] and not any(_features(p["input"])["clipped"] for p in result["train"]):
                        continue
                if _signature(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded generation exhausted")
            seen.add(_signature(pair["input"]))
            result[split].append(pair)
    validate_generated_puzzle(result)
    return result


def validate_generated_puzzle(puzzle):
    """Validate pixels, topology, exact edits and the two demonstration anchors."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features, test_features = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _signature(inp)
            assert key not in seen
            seen.add(key)
            info = _parse(inp)
            predicted = _predict(inp, info)
            assert np.array_equal(predicted, out), "input-derived fault signature disagrees"
            assert not np.array_equal(inp, out)
            assert all(out[p] == 0 for p in info["holes"]), "a hole was filled"
            assert all(out[p] == inp[p] for p in info["footprints"]), "endpoint clue changed"
            halo = set()
            for r, c in info["holes"]:
                halo |= {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                         if 0 <= r + dr < inp.shape[0] and 0 <= c + dc < inp.shape[1]}
            changes = set(map(tuple, np.argwhere(inp != out)))
            assert changes <= halo | info["extras"]
            assert all(out[p] == 0 for p in info["extras"] - halo)
            assert info["holes"] and len(info["routes"]) in (2, 3, 4)
            assert len(info["panels"]) >= len(info["routes"])
            if split == "train":
                features.append(_features(inp, info))
            else:
                test_features.append(_features(inp, info))
    assert any(f["nearest_wrong"] and f["crossings"] for f in features), "teach connectivity over proximity"
    assert any(f["conflicts"] and f["orthogonal"] for f in features), "teach channel superposition"
    assert any(f["extras"] for f in features), "teach speck erasure"
    assert any(f["boundary"] for f in features), "teach halos outside panels"
    assert any(f["diagonal"] for f in features), "teach diagonal fault groups"
    assert any(f["elbows"] for f in features), "teach hidden turns"
    assert any(len(set(f["channel_pairs"])) >= 2 for f in features), "teach route-specific colors"
    assert not any(f["clipped"] for f in test_features) or any(f["clipped"] for f in features), "teach clipping before testing it"


def self_test():
    """Stream 42 seeds; retain aggregate counts/hashes, never a sample corpus."""
    shapes, geometries, colors, sizes, hole_counts, extra_counts, hashes = (set() for _ in range(7))
    totals = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                g = pair["input"]
                info = _parse(g)
                f = _features(g, info)
                shapes.add(g.shape)
                # Structural signature ignores all endpoint labels and damage.
                structural = np.where(np.isin(g, LABELS), 0, g).copy()
                for panel in info["panels"]:
                    for p in panel["cells"]:
                        structural[p] = 5
                for p in info["extras"]:
                    structural[p] = 0
                geometries.add(_signature(structural))
                colors.update(f["channel_pairs"])
                sizes.update(f["panel_shapes"])
                hole_counts.add(f["holes"])
                extra_counts.add(f["extras"])
                totals.update({k: int(bool(f[k])) for k in
                               ("elbows", "boundary", "clipped", "conflicts", "diagonal", "nearest_wrong")})
                totals[f"routes_{f['routes']}"] += 1
                digest.update(g.tobytes())
                digest.update(pair["output"].tobytes())
                if seed < 3:
                    for k in range(4):
                        transformed = np.rot90(g.T, k).copy()
                        assert np.array_equal(solve(transformed), np.rot90(pair["output"].T, k))
        hashes.add(digest.hexdigest())
        if seed < 2:
            replay = generate_puzzle(seed)
            assert all(np.array_equal(a[k], b[k]) for s in puzzle
                       for a, b in zip(puzzle[s], replay[s]) for k in a)
    assert len(hashes) == 42 and len(shapes) >= 25 and len(geometries) >= 150
    assert len(colors) >= 20 and len(sizes) >= 8
    assert len(hole_counts) >= 6 and len(extra_counts) >= 5
    assert all(totals[k] > 0 for k in ("elbows", "boundary", "conflicts", "diagonal", "nearest_wrong",
                                      "routes_2", "routes_3", "routes_4"))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            generate_puzzle(1703, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    fixture = generate_puzzle(0)
    # Counterfactual label intervention: changing one endpoint must change a
    # fault halo, beyond merely preserving that edited endpoint in the output.
    g = fixture["train"][0]["input"].copy()
    info = _parse(g)
    ri = next(info["owners"][pi] for pi, p in enumerate(info["panels"]) if p["holes"])
    route = info["routes"][ri]
    ti = next(t for t in route["terminals"] if info["terminals"][t]["kind"] == "diamond")
    terminal = info["terminals"][ti]
    new = next(c for c in LABELS if c not in (route["diamond"], route["checker"]))
    for p in terminal["cells"]:
        if g[p] == terminal["color"]:
            g[p] = new
    changed = solve(g) != fixture["train"][0]["output"]
    assert any(changed[p] for p in map(tuple, np.argwhere(changed)) if p not in info["footprints"])
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in fixture[s]] for s in fixture}
    hole = min(_parse(bad["test"][0]["input"])["holes"])
    bad["test"][0]["output"][hole] = 5
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("filled missing cell accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(hashes),
            "shapes": len(shapes), "clean_structures": len(geometries),
            "channel_pairs": len(colors), "panel_sizes": len(sizes),
            "hole_counts": sorted(hole_counts), "extra_counts": sorted(extra_counts),
            "feature_pairs": dict(totals)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
