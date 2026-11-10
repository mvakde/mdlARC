"""Wired mirror consensus: erase pixels without a routed reflected partner.

Only standard-library modules and NumPy are used. Construction records paths
and clean packets; prediction reconstructs connectivity from input cells alone.
"""
from itertools import permutations
import json
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _array(grid):
    a = np.asarray(grid)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _components(cells):
    unseen, result = set(cells), []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        stack, found = [first], {first}
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    found.add(q)
                    stack.append(q)
        result.append(found)
    return result


def _box(cells):
    rs, cs = zip(*cells)
    return min(rs), max(rs), min(cs), max(cs)


def _outline(box):
    r0, r1, c0, c1 = box
    return ({(r, c) for r in (r0, r1) for c in range(c0, c1 + 1)} |
            {(r, c) for r in range(r0, r1 + 1) for c in (c0, c1)})


def _external(box):
    r0, r1, c0, c1 = box
    return ([((r0 - 1, c), 0) for c in range(c0 + 1, c1)] +
            [((r, c1 + 1), 1) for r in range(r0 + 1, r1)] +
            [((r1 + 1, c), 2) for c in range(c0 + 1, c1)] +
            [((r, c0 - 1), 3) for r in range(r0 + 1, r1)])


def _rectangles(grid, color):
    result = []
    for comp in _components(map(tuple, np.argwhere(grid == color).tolist())):
        box = _box(comp)
        r0, r1, c0, c1 = box
        assert r1 > r0 and c1 > c0
        assert len(comp) == (r1 - r0 + 1) * (c1 - c0 + 1)
        result.append(box)
    return result


def _mask_links(box, visible, endpoints):
    r0, r1, c0, c1 = box
    candidates = ([((r0 - 1, c), 2) for c in range(c0, c1 + 1)] +
                  [((r1 + 1, c), 0) for c in range(c0, c1 + 1)] +
                  [((r, c0 - 1), 1) for r in range(r0, r1 + 1)] +
                  [((r, c1 + 1), 3) for r in range(r0, r1 + 1)])
    ports = []
    for p, inward in candidates:
        if p not in visible:
            continue
        degree = sum((p[0] + dr, p[1] + dc) in visible for dr, dc in DIRS)
        if degree % 2 == 1 or endpoints.get(p) == inward:
            ports.append((p, inward))
    assert len(ports) >= 2 and len(ports) % 2 == 0
    if len(ports) == 2:
        return [(ports[0], ports[1])]
    remaining, links = set(ports), []
    while remaining:
        a = min(remaining)
        remaining.remove(a)
        p, d = a
        aligned = [b for b in remaining if b[1] == (d + 2) % 4 and
                   b[0][1 if d % 2 == 0 else 0] == p[1 if d % 2 == 0 else 0]]
        assert len(aligned) == 1, "mask pairing is not visibly determined"
        b = aligned[0]
        remaining.remove(b)
        links.append((a, b))
    return links


def _trace(field, endpoints, wire, cross=None, mask=None):
    """A directional graph, including straight-through vertices and mask jumps."""
    visible = set(map(tuple, np.argwhere(field == wire).tolist()))
    if cross is not None:
        visible |= set(map(tuple, np.argwhere(field == cross).tolist()))
    assert set(endpoints) <= visible
    graph = {p: {} for p in visible}
    for p in visible:
        for d, (dr, dc) in enumerate(DIRS):
            q = (p[0] + dr, p[1] + dc)
            if q in visible:
                graph[p][d] = (q, (d + 2) % 4)
    masks = []
    for box in ([] if mask is None else _rectangles(field, mask)):
        links = _mask_links(box, visible, endpoints)
        for (p, dp), (q, dq) in links:
            assert dp not in graph[p] and dq not in graph[q]
            graph[p][dp], graph[q][dq] = (q, dq), (p, dp)
        masks.append((box, links))
    crossings = []
    for p, edges in graph.items():
        if p in endpoints:
            assert len(edges) == 1 and endpoints[p] in edges
        elif field[p] == cross:
            assert len(edges) == 4, "crossing is not four-way"
            crossings.append(p)
        else:
            assert len(edges) == 2, "unmarked junction or broken cable"
    pairs, routes, visited_ends, used_edges = [], [], set(), set()
    for start in sorted(endpoints):
        if start in visited_ends:
            continue
        p, incoming = start, None
        route, states = [], set()
        while True:
            assert (p, incoming) not in states, "closed cable strand"
            states.add((p, incoming))
            route.append(p)
            if p != start and p in endpoints:
                assert p not in visited_ends
                pairs.append((start, p))
                routes.append(route)
                visited_ends |= {start, p}
                break
            edges = graph[p]
            if incoming is None:
                outgoing = endpoints[p]
            elif len(edges) == 4:
                assert incoming in edges
                outgoing = (incoming + 2) % 4
            else:
                assert incoming in edges
                outgoing = next(d for d in edges if d != incoming)
            q, new_incoming = edges[outgoing]
            used_edges |= {(p, outgoing), (q, new_incoming)}
            p, incoming = q, new_incoming
    assert visited_ends == set(endpoints)
    assert used_edges == {(p, d) for p, edges in graph.items() for d in edges}, "untraced strand"
    return pairs, routes, masks, crossings


def _parse(input_grid):
    g = _array(input_grid)
    bg = int(g[0, 0])
    assert all(np.all(edge == bg) for edge in (g[0], g[-1], g[:, 0], g[:, -1]))
    assert np.count_nonzero(g == bg) > g.size / 2
    candidates = []
    for frame_color in sorted(set(map(int, np.unique(g))) - {bg}):
        comps = _components(map(tuple, np.argwhere(g == frame_color).tolist()))
        if not 4 <= len(comps) <= 8 or len(comps) % 2:
            continue
        try:
            frames, footprints, border = [], np.zeros(g.shape, bool), np.zeros(g.shape, bool)
            for comp in comps:
                box = _box(comp)
                r0, r1, c0, c1 = box
                assert r1 - r0 >= 3 and c1 - c0 >= 3 and comp == _outline(box)
                assert 0 < r0 < r1 < g.shape[0] - 1 and 0 < c0 < c1 < g.shape[1] - 1
                ports = [(p, d) for p, d in _external(box) if int(g[p]) != bg]
                assert len(ports) == 1
                port, side = ports[0]
                inside = np.s_[r0 + 1:r1, c0 + 1:c1]
                assert np.any(g[inside] != bg)
                assert not np.any(footprints[r0:r1 + 1, c0:c1 + 1])
                footprints[inside] = True
                for p in comp:
                    border[p] = True
                frames.append(dict(box=box, inside=inside, port=port, side=side,
                                   shape=(r1 - r0 - 1, c1 - c0 - 1)))
            wire_colors = {int(g[f["port"]]) for f in frames}
            assert len(wire_colors) == 1
            wire = wire_colors.pop()
            assert wire not in (bg, frame_color)
            field = g.copy()
            field[footprints | border] = bg
            other = set(map(int, np.unique(field))) - {bg, wire}
            assert len(other) <= 2
            cross, mask = None, None
            for color in sorted(other):
                pieces = _components(map(tuple, np.argwhere(field == color).tolist()))
                if all(len(p) == 1 for p in pieces):
                    assert cross is None
                    cross = color
                else:
                    _rectangles(field, color)
                    assert mask is None
                    mask = color
            labels = set(map(int, np.unique(g[footprints]))) - {bg}
            assert labels.isdisjoint({frame_color, wire, cross, mask})
            endpoints = {f["port"]: f["side"] for f in frames}
            assert len(endpoints) == len(frames)
            pp, routes, masks, crossings = _trace(field, endpoints, wire, cross, mask)
            by_port = {f["port"]: i for i, f in enumerate(frames)}
            pairs = [(by_port[a], by_port[b]) for a, b in pp]
            for i, j in pairs:
                assert frames[i]["side"] == (frames[j]["side"] + 2) % 4
                assert frames[i]["shape"] == frames[j]["shape"]
            candidates.append(dict(grid=g, bg=bg, frame=frame_color, wire=wire, cross=cross,
                                   mask=mask, frames=frames, footprint=footprints, pairs=pairs,
                                   routes=routes, masks=masks, crossings=crossings))
        except (AssertionError, IndexError, StopIteration):
            continue
    assert len(candidates) == 1, f"expected one framed-network interpretation, got {len(candidates)}"
    return candidates[0]


def _reflect(a, side):
    return a[:, ::-1] if side % 2 == 0 else a[::-1, :]


def _answer(info, pairs=None, *, reflect=True, occupancy=False):
    g, bg, frames = info["grid"], info["bg"], info["frames"]
    out = g.copy()
    for i, j in (info["pairs"] if pairs is None else pairs):
        fi, fj = frames[i], frames[j]
        a, b = g[fi["inside"]], g[fj["inside"]]
        assert a.shape == b.shape
        rb = _reflect(b, fi["side"]) if reflect else b
        ra = _reflect(a, fj["side"]) if reflect else a
        good_a = (a != bg) & ((rb != bg) if occupancy else (a == rb))
        good_b = (b != bg) & ((ra != bg) if occupancy else (b == ra))
        out[fi["inside"]] = np.where(good_a, a, bg)
        out[fj["inside"]] = np.where(good_b, b, bg)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Predict from input only; no seed, examples, or construction state."""
    return _answer(_parse(input_grid))


def _positional_pairs(info):
    frames = info["frames"]
    sides = sorted({f["side"] for f in frames})
    assert len(sides) == 2 and sides[1] == (sides[0] + 2) % 4
    groups = []
    for side in sides:
        ids = [i for i, f in enumerate(frames) if f["side"] == side]
        groups.append(sorted(ids, key=lambda i: frames[i]["port"][1 if side % 2 == 0 else 0]))
    assert len(groups[0]) == len(groups[1])
    return list(zip(*groups))


def _local_answer(info):
    out, bg = info["grid"].copy(), info["bg"]
    for f in info["frames"]:
        a = info["grid"][f["inside"]]
        out[f["inside"]] = np.where((a != bg) & (a == _reflect(a, f["side"])), a, bg)
    return out


def _features(pair, info=None):
    info = _parse(pair["input"]) if info is None else info
    out = _array(pair["output"])
    fs, bg, g = info["frames"], info["bg"], info["grid"]
    sides = {f["side"] % 2 for f in fs}
    assert len(sides) == 1
    widths = {f["shape"][1 if f["side"] % 2 == 0 else 0] for f in fs}
    shapes = {f["shape"] for f in fs}
    positional = _answer(info, _positional_pairs(info)) if len(shapes) == 1 else None
    return dict(shape=g.shape, channels=len(info["pairs"]), axis=next(iter(sides)),
                transverse=tuple(sorted(widths)), masks=len(info["masks"]),
                mask_shapes=tuple((b[1]-b[0]+1, b[3]-b[2]+1) for b, _ in info["masks"]),
                elbows=sum(len(links) == 1 and links[0][0][1] % 2 != links[0][1][1] % 2
                           for _, links in info["masks"]),
                crossings=len(info["crossings"]), noise=int(np.count_nonzero(g != out)),
                roles=(bg, info["frame"], info["wire"], info["cross"], info["mask"]),
                payload_colors=len(set(map(int, np.unique(g[info["footprint"]]))) - {bg}),
                ignore_a=not np.array_equal(g, out),
                ignore_b=positional is not None and not np.array_equal(positional, out),
                local=not np.array_equal(_local_answer(info), out),
                direct=not np.array_equal(_answer(info, reflect=False), out),
                occupancy=not np.array_equal(_answer(info, occupancy=True), out))


def _moves(target, rng):
    n = len(target)
    state, moves = list(range(n)) + [None], []
    for j in map(int, rng.permutation(n)):
        if state[j] == target[j]:
            continue
        saved = state[j]
        moves.append((j, n))
        state[n], state[j] = state[j], None
        vacant = j
        while target[vacant] != saved:
            old = state.index(target[vacant])
            moves.append((old, vacant))
            state[vacant], state[old] = state[old], None
            vacant = old
        moves.append((n, vacant))
        state[vacant], state[n] = state[n], None
    assert state[:n] == list(target)
    return moves


def _geometry(rng, n, w, target, wanted, min_h=2):
    moves = _moves(target, rng)
    if rng.random() < .25:
        lane = int(rng.integers(n))
        if 2 * (min_h + 2) + 2 + (6 if wanted else 5) + 2 * (len(moves) + 1) <= 30:
            moves = [(lane, n), (n, lane)] + moves
    lead, tail = (3 if wanted else 2), 2
    gaps = [2] * max(0, len(moves) - 1)
    field_h = lead + tail + sum(gaps) + 1
    max_h = min(6, (30 - field_h - 2) // 2 - 2)
    assert max_h >= min_h
    h = int(rng.integers(min_h, max_h + 1))
    fh, fw = h + 2, w + 2
    # A blank boundary, one-cell minimum bank gaps, and an exterior spare lane.
    x_budget = 30 - (n * fw + n - 1 + 4)
    assert x_budget >= 0
    extras = [0] * (n + 1)
    for _ in range(int(rng.integers(x_budget + 1))):
        extras[int(rng.integers(len(extras)))] += 1
    lefts, x = [], 1 + extras[0]
    for i in range(n):
        lefts.append(x)
        if i < n - 1:
            x += fw + 1 + extras[i + 1]
    centers = [x + 1 + (w - 1) // 2 for x in lefts]
    spare = lefts[-1] + fw + 1
    width = spare + 2 + extras[-1]
    y_budget = 30 - (2 * fh + field_h + 2)
    pieces = [1, 1, lead, tail] + gaps
    for _ in range(int(rng.integers(y_budget + 1))):
        pieces[int(rng.integers(len(pieces)))] += 1
    top, bottom, lead, tail, gaps = pieces[0], pieces[1], pieces[2], pieces[3], pieces[4:]
    start = top + fh
    move_rows, row = [], start + lead
    for i in range(len(moves)):
        move_rows.append(row)
        if i < len(gaps):
            row += gaps[i]
    end = row + tail if moves else start + lead + tail
    height = end + 1 + fh + bottom
    field = np.zeros((height, width), dtype=np.int8)
    paths = [[(start, c)] for c in centers]
    columns, state, previous = centers + [spare], list(range(n)) + [None], start
    for row, (old, new) in zip(move_rows, moves):
        for lane, owner in enumerate(state):
            if owner is not None:
                paths[owner].extend((r, columns[lane]) for r in range(previous + 1, row + 1))
        owner = state[old]
        step = 1 if columns[new] > columns[old] else -1
        paths[owner].extend((row, c) for c in range(columns[old] + step, columns[new] + step, step))
        state[new], state[old] = state[old], None
        previous = row
    for lane, owner in enumerate(state):
        if owner is not None:
            paths[owner].extend((r, columns[lane]) for r in range(previous + 1, end + 1))
    visits = {}
    for i, path in enumerate(paths):
        for p in path:
            visits.setdefault(p, set()).add(i)
            field[p] = 2
    for p, owners in visits.items():
        if len(owners) > 1:
            assert len(owners) == 2
            field[p] = 3
    endpoints = {(start, c): 2 for c in centers}
    endpoints.update({(end, c): 0 for c in centers})
    inferred, _, _, _ = _trace(field, endpoints, 2, 3)
    assert {frozenset((a, b)) for a, b in inferred} == {frozenset((p[0], p[-1])) for p in paths}
    boxes = ([(top, top + fh - 1, x, x + fw - 1) for x in lefts] +
             [(end + 1, end + fh, x, x + fw - 1) for x in lefts])
    return field, boxes, paths, endpoints, centers, start, end, h


def _add_masks(rng, original, paths, endpoints, centers, start, end, wanted):
    field, used = original.copy(), []
    points = sorted(map(tuple, np.argwhere(original != 0).tolist()))
    for _ in range(90):
        if len(used) == wanted:
            return field
        r, c = points[int(rng.integers(len(points)))]
        hh = int(rng.integers(2, min(5, end - start - 1) + 1))
        ww = int(rng.integers(2, min(9, field.shape[1] - 2) + 1))
        r0 = max(start + 1, min(end - hh, r - int(rng.integers(hh))))
        c0 = max(1, min(field.shape[1] - 1 - ww, c - int(rng.integers(ww))))
        box = (r0, r0 + hh - 1, c0, c0 + ww - 1)
        r0, r1, c0, c1 = box
        if any(not (r1 + 1 < a or b + 1 < r0 or c1 + 1 < x or y + 1 < c0)
               for a, b, x, y in used):
            continue
        inside = lambda p: r0 <= p[0] <= r1 and c0 <= p[1] <= c1
        actual = set()
        for path in paths:
            entry = None
            for a, b in zip(path, path[1:]):
                if not inside(a) and inside(b):
                    entry = a
                elif inside(a) and not inside(b):
                    assert entry is not None
                    actual.add(frozenset((entry, b)))
                    entry = None
        if not actual:
            continue
        candidate = field.copy()
        candidate[r0:r1 + 1, c0:c1 + 1] = 4
        visible = set(map(tuple, np.argwhere((candidate == 2) | (candidate == 3)).tolist()))
        try:
            links = _mask_links(box, visible, endpoints)
            assert {frozenset((a[0], b[0])) for a, b in links} == actual
            _trace(candidate, endpoints, 2, 3, 4)
        except (AssertionError, StopIteration):
            continue
        field, used = candidate, used + [box]
    # Guaranteed two-row straight lead stretches on widely separated lanes.
    field = original.copy()
    for c in [centers[0], centers[-1]][:wanted]:
        field[start + 1:start + 3, c - 1:c + 1] = 4
    _trace(field, endpoints, 2, 3, 4)
    assert len(_rectangles(field, 4)) == wanted
    return field


def _packets(rng, n, h, w, label_count, budget, lesson):
    # Canonical palette roles: 0 background, 1 frame, 2 wire, 3 cross,
    # 4 mask, 5..7 payload. Far arrays below are normalized into near coordinates.
    labels = list(range(5, 5 + label_count))
    for _ in range(40):
        cores = []
        style = int(rng.integers(4))
        for i in range(n):
            density = float(rng.uniform(.15, .62))
            a = np.zeros((h, w), dtype=np.int8)
            for r in range(h):
                for c in range(w):
                    present = (rng.random() < density if style == 0 else
                               (r + c + i) % int(rng.integers(2, 5)) == 0 if style == 1 else
                               (r in (0, h - 1) or c in (0, w - 1)) and rng.random() < .75 if style == 2 else
                               c <= int(rng.integers(w)) and rng.random() < .65)
                    if present:
                        a[r, c] = int(rng.choice(labels))
            a[0, 0], a[0, -1] = labels[0], 0
            a[-1, int(rng.integers(w))] = int(rng.choice(labels))
            cores.append(a)
        holes = sum(np.count_nonzero(a == 0) for a in cores)
        capacity = holes * (2 if label_count > 1 else 1) - (1 if label_count > 1 else 0)
        if capacity >= budget:
            break
    else:
        cores = [np.zeros((h, w), np.int8) for _ in range(n)]
        for a in cores:
            a[0, 0], a[-1, 0] = labels[0], labels[-1]
    near, far = [a.copy() for a in cores], [a.copy() for a in cores]
    near[0][0, -1] = labels[0]  # Locally symmetric-looking, remotely unsupported.
    remaining = budget - 1
    reserved = (0, 0, w - 1)
    if lesson == 1:
        assert label_count >= 2
        holes = [(i, r, c) for i, a in enumerate(cores) for r, c in np.argwhere(a == 0)
                 if (i, int(r), int(c)) != reserved]
        i, r, c = holes[int(rng.integers(len(holes)))]
        near[i][r, c], far[i][r, c] = labels[0], labels[1]
        remaining -= 2
    # Ensure deletion is demonstrated on both banks, even with one payload color.
    if not any(np.any(a != b) for a, b in zip(far, cores)):
        holes = [(i, r, c) for i, a in enumerate(cores) for r, c in np.argwhere(a == 0)
                 if (i, int(r), int(c)) != reserved and near[i][r, c] == 0]
        assert holes and remaining > 0
        i, r, c = holes[int(rng.integers(len(holes)))]
        far[i][r, c] = int(rng.choice(labels))
        remaining -= 1
    slots = [(i, int(r), int(c), side) for i, a in enumerate(cores)
             for r, c in np.argwhere(a == 0) for side in (0, 1)
             if (i, int(r), int(c)) != reserved]
    for k in rng.permutation(len(slots)):
        if remaining == 0:
            break
        i, r, c, side = slots[int(k)]
        a, b = (near[i], far[i]) if side == 0 else (far[i], near[i])
        if a[r, c] != 0:
            continue
        options = [color for color in labels if color != b[r, c]]
        if not options:
            continue
        a[r, c] = int(rng.choice(options))
        remaining -= 1
    assert remaining == 0, "packet noise capacity exhausted"
    assert sum(np.count_nonzero(a != c) + np.count_nonzero(b != c)
               for a, b, c in zip(near, far, cores)) == budget
    assert all(np.array_equal(np.where((a != 0) & (a == b), a, 0), c)
               for a, b, c in zip(near, far, cores))
    return cores, near, far


def _make_pair(rng, shift, split, lesson=None):
    test = split == "test"
    if shift == "count":
        n = int(rng.integers(3, 5)) if test else 2
    elif shift == "size" and test:
        n = int(rng.integers(2, 4))
    else:
        n = int(rng.integers(2, 5))
    max_w = {2: 7, 3: 5, 4: 3}[n]
    if shift == "size":
        choices = list(range(5, max_w + 1)) if test else list(range(2, min(4, max_w) + 1))
    else:
        choices = list(range(2, max_w + 1))
    if lesson is not None:
        choices = [w for w in choices if w % 2 == (1 if lesson == 0 else 0)]
    if shift == "noise" and test:
        choices = [w for w in choices if w >= 3]
    w = int(rng.choice(choices))
    target = list(map(int, rng.permutation(n)))
    if lesson is not None and target == list(range(n)):
        target = list(reversed(target))
    elif lesson is None and rng.random() < .18:
        target = list(range(n))
    wanted = 0 if lesson == 0 else 1 if lesson == 1 else int(rng.integers(3 if test else 2))
    if shift == "occlusion" and test:
        wanted = 2
    min_h = 4 if shift == "noise" and test else 2
    field, boxes, paths, endpoints, centers, start, end, h = _geometry(rng, n, w, target, wanted, min_h)
    if wanted:
        field = _add_masks(rng, field, paths, endpoints, centers, start, end, wanted)
    label_count = int(rng.integers(1, 4))
    if lesson == 1 or shift == "noise" and test:
        label_count = max(2, label_count)
    # Keep the two lessons and all non-noise-shift examples within a safe hole budget.
    capacity_floor = n * (h * w - 2) * (2 if label_count > 1 else 1) - (1 if label_count > 1 else 0)
    budget = int(rng.integers(24, 31)) if shift == "noise" and test else int(rng.integers(4, min(8, capacity_floor) + 1))
    cores, near, far = _packets(rng, n, h, w, label_count, budget, lesson)
    inp, out = field.copy(), field.copy()
    for box in boxes:
        for p in _outline(box):
            inp[p] = out[p] = 1
    for i, path in enumerate(paths):
        j = centers.index(path[-1][1])
        for box, a, b in ((boxes[i], near[i], cores[i]),
                          (boxes[n + j], far[i][:, ::-1], cores[i][:, ::-1])):
            r0, r1, c0, c1 = box
            inp[r0 + 1:r1, c0 + 1:c1] = a
            out[r0 + 1:r1, c0 + 1:c1] = b
    palette = np.asarray(rng.permutation(10), dtype=np.int8)
    inp, out = palette[inp], palette[out]
    k = int(rng.choice((0, 2))) if lesson == 0 else int(rng.choice((1, 3))) if lesson == 1 else int(rng.integers(4))
    return {"input": np.rot90(inp, k).copy(), "output": np.rot90(out, k).copy()}


def _key(g):
    a = np.asarray(g, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 2, 10, "num_train")
    nq = _count(num_test, 1, 4, "num_test")
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(("count", "size", "noise", "occlusion")))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            lesson = i if split == "train" and i < 2 else None
            for _ in range(80):
                pair = _make_pair(rng, shift, split, lesson)
                try:
                    info = _parse(pair["input"])
                    assert np.array_equal(_answer(info), pair["output"])
                    m = _features(pair, info)
                    if lesson is not None:
                        assert all(m[k] for k in ("ignore_a", "ignore_b", "local", "direct"))
                        assert m["crossings"] > 0 if lesson == 0 else m["occupancy"]
                    assert _key(pair["input"]) not in seen
                except AssertionError:
                    continue
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
                break
            else:
                raise AssertionError("bounded generation did not produce an informative example")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, measured = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _array(pair["input"]), _array(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _parse(inp)
            assert np.array_equal(_answer(info), out), "answer disagrees with input connectivity and support"
            assert np.all((inp == out) | (out == info["bg"])), "must only delete"
            assert np.array_equal(inp[~info["footprint"]], out[~info["footprint"]]), "changed network"
            for i, j in info["pairs"]:
                fi, fj = info["frames"][i], info["frames"][j]
                a, b = out[fi["inside"]], out[fj["inside"]]
                assert np.any(a != info["bg"])
                assert np.array_equal(a, _reflect(b, fi["side"])), "retained data must corroborate"
            measured[split].append(_features(pair, info))
    a, b = measured["train"][:2]
    assert a["axis"] != b["axis"]
    assert a["transverse"][0] % 2 == 1 and b["transverse"][0] % 2 == 0
    assert a["masks"] == 0 and a["crossings"] > 0 and b["masks"] == 1
    assert all(m[k] for m in (a, b) for k in ("ignore_a", "ignore_b", "local", "direct"))
    assert b["occupancy"]
    tr, te = measured["train"], measured["test"]
    shifts = (min(m["channels"] for m in te) > max(m["channels"] for m in tr),
              min(min(m["transverse"]) for m in te) > max(max(m["transverse"]) for m in tr),
              min(m["noise"] for m in te) > max(m["noise"] for m in tr),
              min(m["masks"] for m in te) == 2 and max(m["masks"] for m in tr) <= 1)
    assert any(shifts), "no measured train/test shift"


def _mixed_size_fixture():
    g = np.zeros((20, 18), np.int8)
    boxes = [(1, 4, 1, 4), (14, 17, 1, 4), (1, 5, 9, 13), (13, 17, 9, 13)]
    for box in boxes:
        for p in _outline(box):
            g[p] = 1
    g[5:14, 2], g[6:13, 11] = 2, 2
    first = np.array([[5, 0], [5, 5]], np.int8)
    second = np.array([[5, 6, 0], [0, 5, 5], [6, 0, 0]], np.int8)
    out = g.copy()
    for x, (a, i, j) in enumerate(((first, 0, 1), (second, 2, 3))):
        for k, data in ((i, a), (j, a[:, ::-1])):
            r0, r1, c0, c1 = boxes[k]
            g[r0+1:r1, c0+1:c1] = out[r0+1:r1, c0+1:c1] = data
        r0, r1, c0, c1 = boxes[i]
        g[r0+1, c1-1] = 5 + x
    return {"input": g, "output": out}


def self_test():
    metrics, signatures, structures = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                g, out = pair["input"], pair["output"]
                assert np.array_equal(solve(g), out)
                assert np.array_equal(solve(out), out), "consensus is idempotent"
                for k in range(4):
                    assert np.array_equal(solve(np.rot90(g, k)), np.rot90(out, k))
                assert np.array_equal(solve(g[:, ::-1]), out[:, ::-1])
                m = _features(pair)
                m["split"] = split
                metrics.append(m)
                roles = {}
                structures.add((g.shape, tuple(roles.setdefault(int(c), len(roles)) for c in g.flat)))
        if seed < 3:
            again = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle for p, q in zip(puzzle[s], again[s])
                       for k in ("input", "output"))
    assert len(signatures) == 42 and len(structures) >= 150
    assert len({m["shape"] for m in metrics}) >= 35
    assert {m["channels"] for m in metrics} == {2, 3, 4}
    assert {m["axis"] for m in metrics} == {0, 1}
    assert {m["masks"] for m in metrics} == {0, 1, 2}
    assert {m["payload_colors"] for m in metrics} >= {1, 2, 3}
    assert len({m["roles"] for m in metrics}) >= 100
    assert len({s for m in metrics for s in m["mask_shapes"]}) >= 10
    assert any(m["elbows"] for m in metrics)
    assert any(m["crossings"] == 0 for m in metrics)
    assert max(m["noise"] for m in metrics) >= 24
    # Exhaust every four-channel permutation, using an input-visible graph.
    rng = np.random.default_rng(23091)
    for target in permutations(range(4)):
        field, _, paths, endpoints, _, _, _, _ = _geometry(rng, 4, 2, list(target), 0)
        pp, _, _, _ = _trace(field, endpoints, 2, 3)
        assert {frozenset(x) for x in pp} == {frozenset((p[0], p[-1])) for p in paths}
    # Hidden elbow, with no privileged assumption of aligned straight ports.
    field = np.zeros((6, 7), np.int8)
    field[0:2, 1], field[3, 4:6], field[2:4, 1:4] = 2, 2, 4
    pp, _, masks, _ = _trace(field, {(0, 1): 2, (3, 5): 3}, 2, None, 4)
    assert pp == [((0, 1), (3, 5))]
    assert {a[0] for a in masks[0][1][0]} == {(1, 1), (3, 4)}
    fixture = _mixed_size_fixture()
    assert np.array_equal(solve(fixture["input"]), fixture["output"])
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(172, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for interior in (False, True):
        p = generate_puzzle(5, 2, 1)
        pair = p["test"][0]
        info = _parse(pair["input"])
        pos = tuple(np.argwhere(info["footprint"] & (pair["output"] != info["bg"]))[0]) if interior else (0, 0)
        pair["output"][pos] = (int(pair["output"][pos]) + 1) % 10
        try:
            validate_generated_puzzle(p)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted an output mutation")
    return {"seeds": 42, "pairs": len(metrics), "distinct_puzzles": len(signatures),
            "color_normalized_inputs": len(structures), "grid_shapes": len({m["shape"] for m in metrics}),
            "channels": sorted({m["channels"] for m in metrics}),
            "mask_shapes": len({s for m in metrics for s in m["mask_shapes"]}),
            "elbow_examples": sum(bool(m["elbows"]) for m in metrics),
            "all_four_channel_permutations": 24, "independent_semantic_review": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
