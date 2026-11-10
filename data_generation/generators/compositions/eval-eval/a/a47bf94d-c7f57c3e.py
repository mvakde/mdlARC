"""Wired anti-copy: complement the remote input state in the local motif frame.

Rendering uses explicit construction paths and states. solve() independently
reads motifs and traces the visible directional cable graph; it has no RNG or
access to examples. Only standard-library modules and NumPy are used.
"""
from itertools import permutations
import json
import numpy as np


WIRE, CROSS, MASK = 8, 9, 7
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or not np.all((g >= 0) & (g <= 9))):
        raise ValueError("expected an integer ARC grid of size 1..30")
    return g.astype(int, copy=True)


def _components(points, diagonal=False):
    remaining, result = set(points), []
    steps = ([(y, x) for y in (-1, 0, 1) for x in (-1, 0, 1) if y or x]
             if diagonal else DIRS)
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        found, stack = {start}, [start]
        while stack:
            y, x = stack.pop()
            for dy, dx in steps:
                p = y + dy, x + dx
                if p in remaining:
                    remaining.remove(p)
                    found.add(p)
                    stack.append(p)
        result.append(found)
    return result


def _bounds(points):
    return (min(y for y, x in points), max(y for y, x in points),
            min(x for y, x in points), max(x for y, x in points))


def _turn(p, code):
    y, x = p
    for _ in range(code % 4):
        y, x = x, -y
    return y, -x if code >= 4 else x


def _unturn(p, code):
    y, x = p
    if code >= 4:
        x = -x
    for _ in range(code % 4):
        y, x = -x, y
    return y, x


def _sig(mapping):
    return tuple(sorted((y, x, int(c)) for (y, x), c in mapping.items()))


def _primitive(crop):
    lengths = []
    for a in (crop, crop.T):
        for row in a:
            cuts = np.r_[0, np.flatnonzero(row[1:] != row[:-1]) + 1, len(row)]
            lengths.extend(map(int, np.diff(cuts)))
    s = int(np.gcd.reduce(lengths))
    small = crop[::s, ::s]
    assert np.array_equal(crop, np.repeat(np.repeat(small, s, 0), s, 1))
    return s, small


def _canonical_mask(points):
    variants = []
    for code in range(8):
        turned = {_turn(p, code) for p in points}
        y0, _, x0, _ = _bounds(turned)
        variants.append(tuple(sorted((y - y0, x - x0) for y, x in turned)))
    return min(variants), len(set(variants)) == 8


def _read_motifs(g, bg):
    cells = set(map(tuple, np.argwhere(~np.isin(g, [bg, WIRE, CROSS, MASK]))))
    objects = []
    for group in _components(cells, diagonal=True):
        y0, y1, x0, x1 = _bounds(group)
        crop = g[y0:y1 + 1, x0:x1 + 1]
        # Structural strokes cannot pass through a motif's bounding rectangle.
        assert not np.any(np.isin(crop, [WIRE, CROSS, MASK]))
        assert np.count_nonzero(crop != bg) == len(group)
        s, small = _primitive(crop)
        raw = {(int(y), int(x)): int(small[y, x])
               for y, x in np.argwhere(small != bg)}
        objects.append(dict(box=(y0, x0), scale=s, raw=raw, cells=group))
    assert 4 <= len(objects) <= 8 and len(objects) % 2 == 0
    common_colors = set.intersection(*(set(o["raw"].values()) for o in objects))
    assert common_colors, "no invariant scaffold color"
    sizes = {c: min(sum(v == c for v in o["raw"].values()) for o in objects)
             for c in common_colors}
    maximum = max(sizes.values())
    largest = [c for c, n in sizes.items() if n == maximum]
    assert len(largest) == 1 and maximum >= 5, "scaffold role is ambiguous"
    carrier = largest[0]
    canonical, asymmetric = _canonical_mask({p for p, c in objects[0]["raw"].items() if c == carrier})
    assert asymmetric, "scaffold must define a unique local frame"
    signatures = set()
    for o in objects:
        assert sum(v == carrier for v in o["raw"].values()) > max(
            [sum(v == c for v in o["raw"].values())
             for c in set(o["raw"].values()) - {carrier}] or [0])
        choices = []
        for code in range(8):
            turned = {_turn(p, code): c for p, c in o["raw"].items()}
            scaffold = {p for p, c in turned.items() if c == carrier}
            y0, _, x0, _ = _bounds(scaffold)
            shape = tuple(sorted((y - y0, x - x0) for y, x in scaffold))
            if shape == canonical:
                normalized = {(y - y0, x - x0): c for (y, x), c in turned.items()}
                choices.append((code, (y0, x0), normalized))
        assert len(choices) == 1
        code, origin, normalized = choices[0]
        o.update(code=code, origin=origin, signature=_sig(normalized))
        signatures.add(o["signature"])
    signatures = sorted(signatures)
    assert len(signatures) == 2, "exactly two local states must be visible"
    reference = min(o["scale"] for o in objects)
    assert reference == 1
    assert {o["signature"] for o in objects if o["scale"] == reference} == set(signatures)
    templates = [{(y, x): c for y, x, c in sig} for sig in signatures]
    union = set(templates[0]) | set(templates[1])
    stable = {p: c for p, c in templates[0].items() if templates[1].get(p) == c}
    owner, reserved = {}, set()
    for i, o in enumerate(objects):
        o["state"] = signatures.index(o["signature"])
        physical = [_pixel(o, p) for p in union]
        y0, y1, x0, x1 = _bounds(physical)
        y1 += o["scale"] - 1
        x1 += o["scale"] - 1
        assert 0 <= y0 <= y1 < g.shape[0] and 0 <= x0 <= x1 < g.shape[1], "replacement clipped"
        rectangle = {(y, x) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)}
        assert not (rectangle & reserved), "replacement rectangles overlap"
        assert not np.any(np.isin(g[y0:y1 + 1, x0:x1 + 1], [WIRE, CROSS, MASK]))
        assert (rectangle & cells) == o["cells"], "replacement would erase another motif"
        reserved.update(rectangle)
        o["rectangle"] = rectangle
        for p in o["cells"]:
            owner[p] = i
        stable_pixels = set()
        for p in stable:
            y, x = _pixel(o, p)
            stable_pixels.update((y + dy, x + dx) for dy in range(o["scale"]) for dx in range(o["scale"]))
        o["stable_pixels"] = stable_pixels
    for i, a in enumerate(objects):
        for b in objects[:i]:
            ay0, ay1, ax0, ax1 = _bounds(a["rectangle"])
            by0, by1, bx0, bx1 = _bounds(b["rectangle"])
            assert ay1 + 1 < by0 or by1 + 1 < ay0 or ax1 + 1 < bx0 or bx1 + 1 < ax0
    return objects, templates, carrier, owner


def _pixel(o, point):
    p = point[0] + o["origin"][0], point[1] + o["origin"][1]
    y, x = _unturn(p, o["code"])
    return o["box"][0] + y * o["scale"], o["box"][1] + x * o["scale"]


def _mask_links(rect, visible):
    y0, y1, x0, x1 = rect
    candidates = []
    for x in range(x0, x1 + 1):
        candidates.extend([((y0 - 1, x), 2), ((y1 + 1, x), 0)])
    for y in range(y0, y1 + 1):
        candidates.extend([((y, x0 - 1), 1), ((y, x1 + 1), 3)])
    ports = []
    for p, inward in candidates:
        degree = sum((p[0] + dy, p[1] + dx) in visible for dy, dx in DIRS)
        if p in visible and degree in (1, 3):
            ports.append((p, inward))
    assert len(ports) >= 2 and len(ports) % 2 == 0
    if len(ports) == 2:
        return [(ports[0], ports[1])]
    left, links = set(ports), []
    while left:
        a = min(left)
        left.remove(a)
        p, d = a
        matches = [b for b in left if b[1] == (d + 2) % 4
                   and b[0][1 if d % 2 == 0 else 0] == p[1 if d % 2 == 0 else 0]]
        assert len(matches) == 1, "mask hides an ambiguous pairing"
        b = matches[0]
        left.remove(b)
        links.append((a, b))
    return links


def _wire_graph(g):
    visible = set(map(tuple, np.argwhere(np.isin(g, [WIRE, CROSS]))))
    graph = {p: {} for p in visible}
    for p in sorted(visible):
        for d, (dy, dx) in enumerate(DIRS):
            q = p[0] + dy, p[1] + dx
            if q in visible:
                graph[p][d] = (q, (d + 2) % 4)
    masks = []
    for group in _components(map(tuple, np.argwhere(g == MASK))):
        rect = _bounds(group)
        y0, y1, x0, x1 = rect
        assert y1 > y0 and x1 > x0 and len(group) == (y1 - y0 + 1) * (x1 - x0 + 1)
        links = _mask_links(rect, visible)
        for (p, dp), (q, dq) in links:
            assert dp not in graph[p] and dq not in graph[q]
            graph[p][dp], graph[q][dq] = (q, dq), (p, dp)
        masks.append((rect, links))
    for p, edges in graph.items():
        assert len(edges) in (1, 2, 4), "broken or branching cable"
        assert (g[p] == CROSS) == (len(edges) == 4), "crossing must continue straight on both axes"
    return graph, masks


def _read(g):
    bg = int(g[0, 0])
    assert bg not in (WIRE, CROSS, MASK)
    assert all(int(g[p]) == bg for p in ((0, -1), (-1, 0), (-1, -1)))
    objects, templates, carrier, owner = _read_motifs(g, bg)
    graph, masks = _wire_graph(g)
    endpoint_owner = {}
    for p, edges in graph.items():
        if len(edges) != 1:
            continue
        neighbors = {(p[0] + dy, p[1] + dx) for dy, dx in DIRS} & set(owner)
        ids = {owner[q] for q in neighbors}
        assert len(ids) == 1, "a cable end must dock to one motif"
        i = next(iter(ids))
        assert neighbors <= objects[i]["stable_pixels"], "docking must use invariant material"
        endpoint_owner[p] = i
    assert len(endpoint_owner) == len(objects)
    assert set(endpoint_owner.values()) == set(range(len(objects)))
    partners, routes, covered_edges = {}, {}, set()
    for start, index in sorted(endpoint_owner.items()):
        p, incoming, states, route = start, None, set(), []
        while True:
            assert (p, incoming) not in states, "closed cable with no remote endpoint"
            states.add((p, incoming))
            route.append(p)
            if p != start and p in endpoint_owner:
                remote = endpoint_owner[p]
                assert remote != index
                partners[index] = remote
                routes[index] = route
                break
            edges = graph[p]
            if incoming is None:
                d = next(iter(edges))
            elif len(edges) == 4:
                d = (incoming + 2) % 4
            else:
                assert incoming in edges and len(edges) == 2
                d = next(d for d in edges if d != incoming)
            covered_edges.add((p, d))
            p, incoming = edges[d]
    assert all(partners[partners[i]] == i for i in partners)
    assert covered_edges == {(p, d) for p, edges in graph.items() for d in edges}, "unattached strand"
    equal = [(i, j) for i, j in partners.items() if i < j and objects[i]["state"] == objects[j]["state"]]
    unlike = [(i, j) for i, j in partners.items() if i < j and objects[i]["state"] != objects[j]["state"]]
    return dict(background=bg, objects=objects, templates=templates, carrier=carrier,
                partners=partners, routes=routes, masks=masks, equal=equal, unlike=unlike,
                endpoints={i: p for p, i in endpoint_owner.items()},
                endpoint_axes={i: next(iter(graph[p])) % 2 for p, i in endpoint_owner.items()},
                crossings=sum(g[p] == CROSS for p in graph))


def _answer(g, info, ignore_routes=False, ordinary_copy=False, toggle_all=False):
    out = g.copy()
    templates, objects, bg = info["templates"], info["objects"], info["background"]
    union = set(templates[0]) | set(templates[1])
    for i, o in enumerate(objects):
        j = info["partners"][i]
        if ignore_routes:
            # A plausible layout shortcut: pair nearest objects across the
            # two physical banks, as identified by their wire endpoints.
            endpoints = info["endpoints"]
            axis = info["endpoint_axes"][i]
            side = endpoints[i][axis]
            candidates = [k for k in endpoints if endpoints[k][axis] != side]
            j = min(candidates, key=lambda k: (abs(endpoints[i][1 - axis] - endpoints[k][1 - axis]), k))
        desired = o["state"] ^ 1 if toggle_all else objects[j]["state"] ^ int(not ordinary_copy)
        for p in union:
            y, x = _pixel(o, p)
            out[y:y + o["scale"], x:x + o["scale"]] = bg
        for p, c in templates[desired].items():
            y, x = _pixel(o, p)
            out[y:y + o["scale"], x:x + o["scale"]] = c
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive remote anti-copy from input cells alone; do not modify the input."""
    g = _grid(input_grid)
    return _answer(g, _read(g))


def _design(rng, change, small=False):
    for _ in range(120):
        h = 3 if small else int(rng.integers(3, 5))
        w = 3 if small else int(rng.integers(3, 5))
        core = {(h - 1, x) for x in range(w)} | {(y, 0) for y in range(h)}
        # Connected procedural growth gives an asymmetric orientation cue.
        for _ in range(int(rng.integers(1, max(2, h * w - len(core) - 1)))):
            frontier = sorted({(y + dy, x + dx) for y, x in core for dy, dx in DIRS
                               if 0 <= y + dy < h and 0 <= x + dx < w} - core)
            if frontier:
                core.add(frontier[int(rng.integers(len(frontier)))])
        _, asymmetric = _canonical_mask(core)
        if not asymmetric or len(core) < 5:
            continue
        interior = sorted({(y, x) for y in range(h - 1) for x in range(w)} - core)
        interior = [p for p in interior if any((p[0] + dy, p[1] + dx) in core for dy, dx in DIRS)]
        if change == "combined" and not interior:
            continue
        palette = list(map(int, rng.permutation(7)))
        bg, carrier, accent0, accent1, extra, stable_color = palette[:6]
        common = {p: carrier for p in core}
        if rng.random() < .35 and interior:
            common[interior[-1]] = stable_color
        # Never extend below the docking row; state support may grow upward or
        # sideways. The complete union is reserved before placing any object.
        a = {(-1, 0)}
        if rng.random() < .5:
            a.add((-1, 1))
        if not small and h == 3 and rng.random() < .25:
            a.add((-2, 0))
        b = a if change == "recolor" else {(1, -1)}
        if change != "recolor" and rng.random() < .5:
            b = b | {(0, -1)}
        if not small and w == 3 and change != "recolor" and rng.random() < .25:
            b = b | {(1, -2)}
        maps = []
        moving = interior[:int(rng.integers(1, min(2, len(interior)) + 1))] if change == "combined" else []
        moving = [p for p in moving if p not in common]
        if change == "combined" and not moving:
            continue
        for state, patch in enumerate((a, b)):
            mapping = dict(common)
            color = accent0 if state == 0 or change == "relocate" else accent1
            mapping.update({p: color for p in patch | set(moving)})
            # A second accent may distinguish the interior from the endpoint.
            if change == "combined" and rng.random() < .5:
                mapping[moving[0]] = extra if state else accent0
            maps.append(mapping)
        if all(len(_components(m, diagonal=True)) == 1 for m in maps):
            canonical, _ = _canonical_mask(core)
            for code in range(8):
                turned_core = {_turn(p, code) for p in core}
                y0, _, x0, _ = _bounds(turned_core)
                if tuple(sorted((y - y0, x - x0) for y, x in turned_core)) == canonical:
                    def template_key(mapping):
                        transformed = {_turn(p, code): c for p, c in mapping.items()}
                        return _sig({(y - y0, x - x0): c for (y, x), c in transformed.items()})
                    maps.sort(key=template_key)
                    break
            return bg, carrier, tuple(maps), (h - 1, int(rng.integers(w)))
    raise AssertionError("bounded asymmetric motif construction failed")


def _moves(target, rng):
    n = len(target)
    state, moves = list(range(n)) + [None], []
    for destination in map(int, rng.permutation(n)):
        if state[destination] == target[destination]:
            continue
        saved = state[destination]
        moves.append((destination, n))
        state[n], state[destination] = state[destination], None
        vacant = destination
        while target[vacant] != saved:
            old = state.index(target[vacant])
            moves.append((old, vacant))
            state[vacant], state[old] = state[old], None
            vacant = old
        moves.append((n, vacant))
        state[vacant], state[n] = state[n], None
    assert state[:n] == target
    return moves


def _routing(rng, centers, spare, start, end, moves, shape, bg):
    columns = centers + [spare]
    rows = [start + 2 + 2 * k for k in range(len(moves))]
    # Distribute slack over gaps instead of freezing crossing locations.
    if moves:
        slack = end - rows[-1] - 2
        for _ in range(slack):
            k = int(rng.integers(len(rows) + 1))
            for j in range(k, len(rows)):
                rows[j] += 1
    paths = [[(start, c)] for c in centers]
    state = list(range(len(centers))) + [None]
    previous = start
    for r, (old, new) in zip(rows, moves):
        for lane, route in enumerate(state):
            if route is not None:
                paths[route].extend((y, columns[lane]) for y in range(previous + 1, r + 1))
        route = state[old]
        step = 1 if columns[new] > columns[old] else -1
        paths[route].extend((r, x) for x in range(columns[old] + step, columns[new] + step, step))
        state[new], state[old] = state[old], None
        previous = r
    for lane, route in enumerate(state):
        if route is not None:
            paths[route].extend((r, columns[lane]) for r in range(previous + 1, end + 1))
    grid = np.full(shape, bg, dtype=int)
    visits = {}
    for i, path in enumerate(paths):
        for p in path:
            visits.setdefault(p, set()).add(i)
            grid[p] = WIRE
    for p, owners in visits.items():
        if len(owners) == 2:
            grid[p] = CROSS
        else:
            assert len(owners) == 1
    return grid, paths


def _occlude(rng, grid, paths, start, end, wanted):
    original = grid.copy()
    occupied = set(map(tuple, np.argwhere(np.isin(original, [WIRE, CROSS]))))
    used, points = [], sorted(occupied)
    for _ in range(55):
        if len(used) >= wanted:
            break
        r, c = points[int(rng.integers(len(points)))]
        hh = int(rng.integers(2, min(4, end - start - 1) + 1))
        ww = int(rng.integers(2, min(8, grid.shape[1] - 1) + 1))
        y0 = max(start + 1, min(end - hh, r - int(rng.integers(hh))))
        x0 = max(0, min(grid.shape[1] - ww, c - int(rng.integers(ww))))
        rect = y0, y0 + hh - 1, x0, x0 + ww - 1
        a, b, x, z = rect
        if any(not (b + 1 < p or q + 1 < a or z + 1 < u or v + 1 < x) for p, q, u, v in used):
            continue
        inside = lambda p: a <= p[0] <= b and x <= p[1] <= z
        actual = set()
        for path in paths:
            entry = None
            for p, q in zip(path, path[1:]):
                if not inside(p) and inside(q):
                    entry = p
                if inside(p) and not inside(q):
                    assert entry is not None
                    actual.add(frozenset((entry, q)))
                    entry = None
        if not actual:
            continue
        try:
            links = _mask_links(rect, {p for p in occupied if not inside(p)})
            # Two exposed ports determine a hidden connection even around an
            # elbow. This is the occlusion convention taught by the lessons.
            assert len(links) == 1
            assert {frozenset((p, q)) for (p, _), (q, _) in links} == actual
            candidate = grid.copy()
            candidate[a:b + 1, x:z + 1] = MASK
            _wire_graph(candidate)
        except AssertionError:
            continue
        grid = candidate
        used.append(rect)
    return grid


def _paint(grid, template, placement):
    y0, x0, scale, code, socket = placement
    for p, color in template.items():
        y, x = _turn((p[0] - socket[0], p[1] - socket[1]), code)
        y, x = y0 + y * scale, x0 + x * scale
        assert 0 <= y and y + scale <= grid.shape[0] and 0 <= x and x + scale <= grid.shape[1]
        grid[y:y + scale, x:x + scale] = color


def _make_pair(rng, n, change, lesson, enlarged, mask_wanted, orientation, target=None):
    # Geometry proposals may be too large, but a small design at the same
    # requested count/scale is a bounded fitting fallback, not a rule exception.
    requested_target = None if target is None else list(target)
    for attempt in range(90):
        small = enlarged == 3 or attempt >= 50
        bg, carrier, templates, socket = _design(rng, change, small)
        target = list(map(int, rng.permutation(n))) if requested_target is None else list(requested_target)
        if lesson is not None:
            target = list(range(n))
            target[0], target[-1] = target[-1], target[0]
        top_states = list(map(int, rng.integers(2, size=n)))
        bottom_states = list(map(int, rng.integers(2, size=n)))
        # Lessons show equal 0/0 and equal 1/1 respectively, plus unlike pairs.
        if lesson is not None:
            bit = lesson % 2
            top_states[0] = bottom_states[target.index(0)] = bit
            top_states[-1], bottom_states[target.index(n - 1)] = 0, 1
        elif all(top_states[target[j]] != bottom_states[j] for j in range(n)):
            bottom_states[0] = top_states[target[0]]
        scales = [[1] * n, [1] * n]
        if enlarged:
            # Retain at least one unit reference of each state.
            eligible = []
            states = top_states + bottom_states
            for k, bit in enumerate(states):
                remote = bottom_states[target.index(k)] if k < n else top_states[target[k - n]]
                if states.count(bit) >= 2 and bit == remote:
                    eligible.append(k)
            k = eligible[int(rng.integers(len(eligible)))]
            scales[k // n][k % n] = enlarged
        if not enlarged and n <= 3 and attempt < 35 and rng.random() < .18:
            states = top_states + bottom_states
            options = [k for k, bit in enumerate(states) if states.count(bit) > 1]
            k = options[int(rng.integers(len(options)))]
            scales[k // n][k % n] = 2
        codes = [[4 * int(rng.integers(2)) for _ in range(n)],
                 [2 + 4 * int(rng.integers(2)) for _ in range(n)]]
        union = set(templates[0]) | set(templates[1])
        extents, heights = [[], []], [0, 0]
        for bank in range(2):
            for i in range(n):
                offsets = {_turn((p[0] - socket[0], p[1] - socket[1]), codes[bank][i]) for p in union}
                y0, y1, x0, x1 = _bounds(offsets)
                s = scales[bank][i]
                assert y1 == 0 if bank == 0 else y0 == 0
                heights[bank] = max(heights[bank], (y1 - y0 + 1) * s)
                extents[bank].append((s // 2 - x0 * s, x1 * s + s - 1 - s // 2))
        left = [max(extents[bank][i][0] for bank in range(2)) for i in range(n)]
        right = [max(extents[bank][i][1] for bank in range(2)) for i in range(n)]
        margin_left, margin_right = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        centers = [margin_left + left[0]]
        for i in range(1, n):
            centers.append(centers[-1] + right[i - 1] + left[i] + int(rng.integers(2, 4)))
        spare = centers[-1] + right[-1] + int(rng.integers(2, 4))
        width = spare + margin_right + 1
        moves = _moves(target, rng)
        field_height = 2 * len(moves) + 3 if moves else int(rng.integers(4, 8))
        top_margin, bottom_margin = int(rng.integers(2)), int(rng.integers(2))
        minimum = heights[0] + heights[1] + field_height + top_margin + bottom_margin
        if width > 30 or minimum > 30:
            continue
        extra = int(rng.integers(min(3, 30 - minimum) + 1))
        start = top_margin + heights[0]
        end = start + field_height + extra - 1
        shape = end + 1 + heights[1] + bottom_margin, width
        field, paths = _routing(rng, centers, spare, start, end, moves, shape, bg)
        if mask_wanted:
            field = _occlude(rng, field, paths, start, end, mask_wanted)
        inp, out = field.copy(), field.copy()
        for bank in range(2):
            for i in range(n):
                s = scales[bank][i]
                placement = (start - s if bank == 0 else end + 1, centers[i] - s // 2,
                             s, codes[bank][i], socket)
                state = top_states[i] if bank == 0 else bottom_states[i]
                remote = bottom_states[target.index(i)] if bank == 0 else top_states[target[i]]
                _paint(inp, templates[state], placement)
                _paint(out, templates[1 - remote], placement)
        if orientation >= 4:
            inp, out = np.fliplr(inp), np.fliplr(out)
        inp, out = np.rot90(inp, orientation % 4).copy(), np.rot90(out, orientation % 4).copy()
        try:
            info = _read(inp)
            assert info["equal"] and np.array_equal(_answer(inp, info), out)
            if lesson is not None:
                assert info["unlike"]
                assert not np.array_equal(_answer(inp, info, ignore_routes=True), out)
                if lesson == 1:
                    assert info["masks"], "the second lesson must demonstrate occlusion"
            return {"input": inp, "output": out}
        except AssertionError:
            continue
    raise AssertionError(f"bounded packing exhausted: {n}/{change}/{enlarged}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 2, 10, "num_train")
    nq = _count(num_test, 1, 4, "num_test")
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(["count", "scale", "combine"]))
    orientation_base = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            lesson = index if split == "train" and index < 2 else None
            if lesson is not None:
                n = 2
                change = "recolor" if index == 0 else "relocate"
                enlarged = 2 if index == 0 else 0
                masks = index  # First teach a visible crossing, then occlusion.
            elif split == "test" and shift == "count":
                n, change, enlarged, masks = 4, str(rng.choice(["recolor", "relocate", "combined"])), 0, int(rng.integers(3))
            elif split == "test" and shift == "scale":
                n, change, enlarged, masks = 2, str(rng.choice(["recolor", "relocate", "combined"])), 3, int(rng.integers(3))
            else:
                n = int(rng.integers(2, 4 if split == "train" else 5))
                change = "combined" if split == "test" else str(rng.choice(["recolor", "relocate"]))
                enlarged = 0
                masks = int(rng.integers(3))
            orientation = (orientation_base + (index % 2 if split == "train" else int(rng.integers(4)))) % 4
            orientation += 4 * int(rng.integers(2))
            for _ in range(20):
                pair = _make_pair(rng, n, change, lesson, enlarged, masks, orientation)
                key = pair["input"].shape, pair["input"].tobytes()
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("distinct input construction exhausted")
    return puzzle


def _features(g, info):
    a, b = info["templates"]
    recolor = any(a[p] != b[p] for p in set(a) & set(b))
    relocate = set(a) != set(b)
    return dict(shape=list(g.shape), cables=len(info["objects"]) // 2,
                scales=sorted(o["scale"] for o in info["objects"]),
                equal=len(info["equal"]), unlike=len(info["unlike"]),
                crossings=int(info["crossings"]), masks=len(info["masks"]),
                mask_shapes=[(r[1] - r[0] + 1, r[3] - r[2] + 1) for r, _ in info["masks"]],
                change="combined" if recolor and relocate else "recolor" if recolor else "relocate",
                scaffold=tuple(sorted(p for p, c in a.items() if c == info["carrier"])),
                background=info["background"], carrier=info["carrier"],
                palette=tuple(map(int, np.unique(g))),
                frames=tuple(o["code"] for o in info["objects"]),
                unequal_support=len(a) != len(b),
                template_boxes=_bounds(a) != _bounds(b))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    _count(len(puzzle["train"]), 2, 10, "num_train")
    _count(len(puzzle["test"]), 1, 4, "num_test")
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, lessons, changes, scale_evidence = set(), set(), set(), False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g, out = _grid(pair["input"]), _grid(pair["output"])
            assert g.shape == out.shape
            key = g.shape, g.tobytes()
            assert key not in seen
            seen.add(key)
            info = _read(g)
            expected = _answer(g, info)
            assert np.array_equal(expected, out), "incorrect simultaneous remote complement"
            assert info["equal"] and not np.array_equal(g, out)
            mutable = set().union(*(o["rectangle"] for o in info["objects"]))
            outside = np.ones(g.shape, dtype=bool)
            for p in mutable:
                outside[p] = False
            assert np.array_equal(g[outside], out[outside])
            assert np.array_equal(g == info["carrier"], out == info["carrier"])
            # Direct endpoint assertions also check fixed points and direction.
            for i, o in enumerate(info["objects"]):
                if o["state"] != info["objects"][info["partners"][i]]["state"]:
                    for p in o["rectangle"]:
                        assert g[p] == out[p], "unlike terminals should be fixed"
            if split == "train":
                feature = _features(g, info)
                changes.add(feature["change"])
                scale_evidence |= any(o["scale"] > 1 and o["state"] ==
                                      info["objects"][info["partners"][i]]["state"]
                                      for i, o in enumerate(info["objects"]))
                for i, j in info["equal"]:
                    lessons.add(info["objects"][i]["state"])
                if info["unlike"]:
                    lessons.add("fixed")
                if info["masks"]:
                    lessons.add("occlusion")
                if not np.array_equal(_answer(g, info, ignore_routes=True), out):
                    lessons.add("routing")
                assert not np.array_equal(_answer(g, info, ordinary_copy=True), out)
    assert lessons >= {0, 1, "fixed", "routing", "occlusion"}
    assert scale_evidence and {"recolor", "relocate"} <= changes


def self_test():
    """Bounded 42-seed audit plus counterexamples and malformed-input checks."""
    features, fingerprints, geometries, palettes = [], set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        again = generate_puzzle(seed)
        fingerprint = tuple((p["input"].shape, p["input"].tobytes(), p["output"].tobytes())
                            for split in ("train", "test") for p in puzzle[split])
        assert fingerprint == tuple((p["input"].shape, p["input"].tobytes(), p["output"].tobytes())
                                    for split in ("train", "test") for p in again[split])
        fingerprints.add(fingerprint)
        for split in ("train", "test"):
            for p in puzzle[split]:
                before = p["input"].copy()
                assert np.array_equal(solve(before), p["output"])
                assert np.array_equal(before, p["input"])
                info = _read(before)
                f = _features(before, info)
                features.append(f)
                geometries.add(f["scaffold"])
                palettes.add(f["palette"])
    assert len(fingerprints) == 42
    assert {f["cables"] for f in features} == {2, 3, 4}
    assert {s for f in features for s in f["scales"]} == {1, 2, 3}
    assert {f["change"] for f in features} == {"recolor", "relocate", "combined"}
    assert len({tuple(f["shape"]) for f in features}) >= 25
    assert len(geometries) >= 8 and len(palettes) >= 15
    assert {f["background"] for f in features} == set(range(7))
    assert {f["carrier"] for f in features} == set(range(7))
    assert {f["masks"] for f in features} >= {0, 1, 2}
    assert any(f["crossings"] == 0 for f in features) and any(f["crossings"] >= 5 for f in features)
    assert any(f["unequal_support"] for f in features) and any(f["template_boxes"] for f in features)
    assert any(len(set(f["frames"])) > 2 for f in features)
    routing_rng = np.random.default_rng(410)
    for n in (2, 3, 4):
        for target in permutations(range(n)):
            pair = _make_pair(routing_rng, n, "recolor", None, 0, 0, 0, target)
            info = _read(pair["input"])
            endpoints = info["endpoints"]
            top = sorted((p[1], i) for i, p in endpoints.items() if p[0] == min(q[0] for q in endpoints.values()))
            bottom = sorted((p[1], i) for i, p in endpoints.items() if p[0] == max(q[0] for q in endpoints.values()))
            bottom_index = {i: k for k, (_, i) in enumerate(bottom)}
            assert [bottom_index[info["partners"][i]] for _, i in top] == [target.index(i) for i in range(n)]
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(1000 + 10 * nt + nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    p = generate_puzzle(88, 2, 1)
    first = p["train"][0]
    info = _read(first["input"])
    assert not np.array_equal(_answer(first["input"], info, ignore_routes=True), first["output"])
    assert not np.array_equal(_answer(first["input"], info, toggle_all=True), first["output"])
    # Updating a partner from the already-updated state fails on an equal pair.
    i, j = info["equal"][0]
    old = info["objects"][i]["state"]
    first_new = 1 - old
    sequential_second = 1 - first_new
    assert sequential_second != 1 - old
    for point in ((0, 0), tuple(np.argwhere(first["input"] != first["output"])[0])):
        bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        bad["train"][0]["output"][point] = (int(bad["train"][0]["output"][point]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            continue
        raise AssertionError("output corruption accepted")
    # A marked crossing cannot silently be interpreted as a junction or bend.
    bad_input = first["input"].copy()
    q = tuple(np.argwhere(bad_input == CROSS)[0])
    bad_input[q] = WIRE
    try:
        solve(bad_input)
    except AssertionError:
        pass
    else:
        raise AssertionError("unmarked crossing accepted")
    return {"seeds": 42, "pairs": len(features), "shapes": len({tuple(f["shape"]) for f in features}),
            "scaffolds": len(geometries), "palette_sets": len(palettes)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
