"""Rectangular chamber mazes with a uniquely determined terminal route."""

import hashlib
import numpy as np


_SOURCE_DIGESTS = {
    "9df6c0bc7e1e8f5c9ee4ba86dbb76bd1a1d26f260c2e4b7827d74d2787344855",
    "555e53177b83f130507941ad4dc34bb8fa936f36ebec9e18955a1659e8c855cb",
    "1780bd9f2fede96b42eb03de1285b54acdac15492efffb484c9589491660b903",
    "f976117ce3b1db40785454d5636abc8bf5d331e8319de4dbf8c597bb9ef02c02",
}


def _key(a):
    return (a.shape, a.tobytes())


def _digest(a, b):
    return hashlib.sha256(bytes(a.shape) + a.astype(np.uint8).tobytes()
                          + b.astype(np.uint8).tobytes()).hexdigest()


def _edge(a, b):
    return (a, b) if a < b else (b, a)


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), bool)
    result = []
    for rr, cc in zip(*np.nonzero(mask)):
        r, c = int(rr), int(cc)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack = [(r, c)]
        component = []
        while stack:
            p = stack.pop()
            component.append(p)
            for q in ((p[0]-1, p[1]), (p[0]+1, p[1]),
                      (p[0], p[1]-1), (p[0], p[1]+1)):
                if 0 <= q[0] < h and 0 <= q[1] < w and mask[q] and not seen[q]:
                    seen[q] = True
                    stack.append(q)
        result.append(component)
    return result


def _spec(rng, small=False):
    nr = int(rng.integers(1, 4 if small else 5))
    nc = int(rng.integers(1, 4 if small else 6))
    if nr * nc == 1:
        nc = 2
    sizes = [1, 2, 2, 3, 3, 4, 5, 6]
    heights = [int(rng.choice(sizes)) for _ in range(nr)]
    widths = [int(rng.choice(sizes)) for _ in range(nc)]
    if rng.random() < 0.28:
        heights = [heights[0]] * nr
        widths = [widths[0]] * nc
    wr = [0]
    wc = [0]
    for size in heights:
        wr.append(wr[-1] + size + 1)
    for size in widths:
        wc.append(wc[-1] + size + 1)
    occupied = {(i, j) for i in range(nr) for j in range(nc)}
    if nr >= 2 and nc >= 2 and rng.random() < 0.42:
        if rng.random() < 0.5:
            occupied.remove((int(rng.choice([0, nr-1])), int(rng.choice([0, nc-1]))))
        else:
            at_bottom = rng.random() < 0.5
            stem_length = int(rng.integers(1, nr))
            keep = int(rng.integers(nc))
            stem_rows = (range(nr-stem_length,nr) if at_bottom
                         else range(stem_length))
            occupied = {p for p in occupied if p[0] not in stem_rows
                        or p[1] == keep}
    return {"wr": wr, "wc": wc, "cells": occupied, "top": 0, "left": 0}


def _side_candidates(spec, side):
    nr, nc = len(spec["wr"])-1, len(spec["wc"])-1
    if side == "top":
        return [(0, j) for j in range(nc) if (0, j) in spec["cells"]]
    if side == "bottom":
        return [(nr-1, j) for j in range(nc) if (nr-1, j) in spec["cells"]]
    if side == "left":
        return [(i, 0) for i in range(nr) if (i, 0) in spec["cells"]]
    return [(i, nc-1) for i in range(nr) if (i, nc-1) in spec["cells"]]


def _tree(rng, cells):
    adjacent = []
    for i, j in sorted(cells):
        for q in ((i+1, j), (i, j+1)):
            if q in cells:
                adjacent.append(((i, j), q))
    parent = {p: p for p in cells}

    def root(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    result = []
    for k in rng.permutation(len(adjacent)):
        a, b = adjacent[int(k)]
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[ra] = rb
            result.append((a, b))
    assert len(result) == len(cells)-1
    return result


def _path(adjacency, start, goal):
    parent = {start: None}
    stack = [start]
    while stack:
        p = stack.pop()
        if p == goal:
            break
        for q in adjacency[p]:
            if q not in parent:
                parent[q] = p
                stack.append(q)
    assert goal in parent
    result = []
    p = goal
    while p is not None:
        result.append(p)
        p = parent[p]
    return result[::-1]


def _draw_spec(grid, spec, wall):
    top, left = spec["top"], spec["left"]
    wr, wc = spec["wr"], spec["wc"]
    for i, j in spec["cells"]:
        r0, r1 = top+wr[i], top+wr[i+1]
        c0, c1 = left+wc[j], left+wc[j+1]
        grid[r0, c0:c1+1] = wall
        grid[r1, c0:c1+1] = wall
        grid[r0:r1+1, c0] = wall
        grid[r0:r1+1, c1] = wall


def _box(spec, cell):
    i, j = cell
    return (spec["top"]+spec["wr"][i]+1,
            spec["top"]+spec["wr"][i+1]-1,
            spec["left"]+spec["wc"][j]+1,
            spec["left"]+spec["wc"][j+1]-1)


def _internal_door(rng, spec, a, b):
    ai, aj = a
    bi, bj = b
    if ai == bi:
        c = spec["left"] + spec["wc"][max(aj, bj)]
        r = int(rng.integers(spec["top"]+spec["wr"][ai]+1,
                             spec["top"]+spec["wr"][ai+1]))
    else:
        r = spec["top"] + spec["wr"][max(ai, bi)]
        c = int(rng.integers(spec["left"]+spec["wc"][aj]+1,
                             spec["left"]+spec["wc"][aj+1]))
    return r, c


def _external_door(rng, spec, cell, side):
    i, j = cell
    if side == "top":
        return (spec["top"]+spec["wr"][i],
                spec["left"]+int(rng.integers(spec["wc"][j]+1, spec["wc"][j+1])))
    if side == "bottom":
        return (spec["top"]+spec["wr"][i+1],
                spec["left"]+int(rng.integers(spec["wc"][j]+1, spec["wc"][j+1])))
    if side == "left":
        return (spec["top"]+int(rng.integers(spec["wr"][i]+1, spec["wr"][i+1])),
                spec["left"]+spec["wc"][j])
    return (spec["top"]+int(rng.integers(spec["wr"][i]+1, spec["wr"][i+1])),
            spec["left"]+spec["wc"][j+1])


def _make_pair(rng, palette, style, arrays=None, span_mode=None,
               orientation=None, unused_array=False):
    for attempt in range(180):
        count = arrays if arrays is not None else int(rng.choice([1, 2, 2, 2, 3]))
        if unused_array:
            assert count == 3
        specs = [_spec(rng, small=count == 3) for _ in range(count)]
        if rng.random() < 0.18:
            uniform_h = int(rng.integers(1, 6))
            uniform_w = int(rng.integers(1, 6))
            for s in specs:
                nr, nc = len(s["wr"])-1, len(s["wc"])-1
                s["wr"] = [i*(uniform_h+1) for i in range(nr+1)]
                s["wc"] = [j*(uniform_w+1) for j in range(nc+1)]
        gateways = []
        local_gates = []
        for k in range(count-1):
            ca = _side_candidates(specs[k], "bottom")
            cb = _side_candidates(specs[k+1], "top")
            a = ca[int(rng.integers(len(ca)))]
            b = cb[int(rng.integers(len(cb)))]
            xa = int(rng.integers(specs[k]["wc"][a[1]]+1,
                                  specs[k]["wc"][a[1]+1]))
            xb = int(rng.integers(specs[k+1]["wc"][b[1]]+1,
                                  specs[k+1]["wc"][b[1]+1]))
            gateways.append((a, b))
            local_gates.append((xa, xb))
        specs[0]["left"] = 0
        for k in range(count-1):
            # Adjacent gate coordinates give a two-cell-wide rectangle.
            if span_mode == "aligned":
                displacement = 0
            elif span_mode == "broad":
                displacement = int(rng.choice([2, -2, 3, -3, 4, -4,
                                               5, -5, 7, -7, 10, -10, 14, -14]))
            elif span_mode == "offset":
                displacement = int(rng.choice([1, -1, 2, -2, 3, -3,
                                               4, -4, 7, -7, 10, -10]))
            else:
                displacement = int(rng.choice([0, 0, 1, -1, 2, -2, 3, -3,
                                               4, -4, 5, -5, 7, -7, 10, -10, 14, -14]))
            specs[k+1]["left"] = (specs[k]["left"]+local_gates[k][0]
                                  + displacement-local_gates[k][1])
        for k in range(1, count):
            gap = int(rng.integers(2, 8))
            specs[k]["top"] = specs[k-1]["top"]+specs[k-1]["wr"][-1]+gap+1
        minx = min(s["left"] for s in specs)
        maxx = max(s["left"]+s["wc"][-1] for s in specs)
        raw_h = specs[-1]["top"]+specs[-1]["wr"][-1]+1
        raw_w = maxx-minx+1
        if raw_h > 29 or raw_w > 29:
            continue
        margin_top = int(rng.integers(0, min(5, 31-raw_h)))
        margin_left = int(rng.integers(0, min(5, 31-raw_w)))
        margin_bottom = int(rng.integers(0, min(5, 31-raw_h-margin_top)))
        margin_right = int(rng.integers(0, min(5, 31-raw_w-margin_left)))
        # A single array keeps exterior background for role recognition.
        if count == 1 and margin_top+margin_bottom < 2:
            if raw_h+2 <= 30:
                margin_bottom = 2-margin_top
            else:
                continue
        for s in specs:
            s["top"] += margin_top
            s["left"] += margin_left-minx
        height = raw_h+margin_top+margin_bottom
        width = raw_w+margin_left+margin_right
        bg, wall, route = palette
        inp = np.full((height, width), bg, dtype=np.int8)
        for s in specs:
            _draw_spec(inp, s, wall)
        nodes = [(k, p) for k, s in enumerate(specs) for p in sorted(s["cells"])]
        edges = []
        for k, s in enumerate(specs):
            edges.extend((((k, a), (k, b), "door", _internal_door(rng, s, a, b))
                          for a, b in _tree(rng, s["cells"])))
        for k, ((a, b), (xa, xb)) in enumerate(zip(gateways, local_gates)):
            if unused_array and k == 1:
                continue
            pa = (specs[k]["top"]+specs[k]["wr"][-1], specs[k]["left"]+xa)
            pb = (specs[k+1]["top"], specs[k+1]["left"]+xb)
            edges.append(((k, a), (k+1, b), "connector", (pa, pb)))
        adj = {node: [] for node in nodes}
        for a, b, _, _ in edges:
            adj[a].append(b)
            adj[b].append(a)
        first = [(0, p) for p in specs[0]["cells"]]
        terminal_last = count-2 if unused_array else count-1
        last = [(terminal_last, p) for p in specs[terminal_last]["cells"]]
        if count == 1:
            pair = rng.choice(len(first), size=2, replace=False)
            start, goal = first[int(pair[0])], first[int(pair[1])]
        else:
            start = first[int(rng.integers(len(first)))]
            goal = last[int(rng.integers(len(last)))]
        path = _path(adj, start, goal)
        path_edges = {_edge(a, b) for a, b in zip(path, path[1:])}
        if style == "branch" and len(path) == len(nodes):
            continue
        if style == "branch":
            off = [e for e in edges if _edge(e[0], e[1]) not in path_edges]
            attached = [e for e in off if e[0] in path or e[1] in path]
            if not attached:
                continue
            retain = set()
            chosen = attached[int(rng.integers(len(attached)))]
            retain.add(_edge(chosen[0], chosen[1]))
            for e in off:
                if rng.random() < float(rng.uniform(0.15, 0.85)):
                    retain.add(_edge(e[0], e[1]))
        else:
            retain = set()
            for e in edges:
                if (e[0] not in path and e[1] not in path
                        and rng.random() < 0.45):
                    retain.add(_edge(e[0], e[1]))
        retain |= path_edges
        for a, b, kind, place in edges:
            if _edge(a, b) not in retain:
                continue
            if kind == "door":
                inp[place] = bg
            else:
                inp[place[0]] = bg
                inp[place[1]] = bg
        # The route markers are literal color regions; full chambers are common.
        for node in (start, goal):
            r0, r1, c0, c1 = _box(specs[node[0]], node[1])
            full = rng.random() < 0.72
            mh = r1-r0+1 if full else int(rng.integers(1, r1-r0+2))
            mw = c1-c0+1 if full else int(rng.integers(1, c1-c0+2))
            rr = int(rng.integers(r0, r1-mh+2))
            cc = int(rng.integers(c0, c1-mw+2))
            inp[rr:rr+mh, cc:cc+mw] = route
        out = inp.copy()
        for node in path:
            r0, r1, c0, c1 = _box(specs[node[0]], node[1])
            view = out[r0:r1+1, c0:c1+1]
            view[view == bg] = route
        for a, b, kind, place in edges:
            if _edge(a, b) not in path_edges:
                continue
            if kind == "door":
                out[place] = route
            else:
                p, q = place
                r0, r1 = sorted((p[0], q[0]))
                c0, c1 = sorted((p[1], q[1]))
                view = out[r0:r1+1, c0:c1+1]
                view[view == bg] = route
        # Rotate entire construction, keeping relative placement and scale independent.
        if orientation == "horizontal" or (orientation is None and rng.random() < 0.5):
            inp, out = inp.T.copy(), out.T.copy()
        if rng.random() < 0.5:
            inp, out = inp[:, ::-1].copy(), out[:, ::-1].copy()
        # Remove an outer boundary line occasionally, leaving readable clipped rooms.
        if inp.shape[0] > 5 and np.any(inp[0] == wall) and rng.random() < 0.34:
            inp, out = inp[1:].copy(), out[1:].copy()
        if inp.shape[0] > 5 and np.any(inp[-1] == wall) and rng.random() < 0.34:
            inp, out = inp[:-1].copy(), out[:-1].copy()
        if inp.shape[1] > 5 and np.any(inp[:, 0] == wall) and rng.random() < 0.34:
            inp, out = inp[:, 1:].copy(), out[:, 1:].copy()
        if inp.shape[1] > 5 and np.any(inp[:, -1] == wall) and rng.random() < 0.34:
            inp, out = inp[:, :-1].copy(), out[:, :-1].copy()
        if np.array_equal(inp, out) or _digest(inp, out) in _SOURCE_DIGESTS:
            continue
        try:
            expected, meta = _solve(inp)
            if (np.array_equal(expected, out)
                    and meta["arrays"] == count
                    and meta["rooms"] == len(nodes)
                    and sorted(meta["array_room_counts"]) ==
                    sorted(len(s["cells"]) for s in specs)
                    and sorted(meta["array_layouts"]) == sorted(
                        ((len(s["wc"])-1,len(s["wr"])-1)
                         if meta["orientation"] == "horizontal" else
                         (len(s["wr"])-1,len(s["wc"])-1)) for s in specs)
                    and all(max(size) <= 6 for size in meta["room_sizes"])
                    and (style != "branch" or meta["branch"])):
                return inp, out, meta
        except (AssertionError, ValueError, IndexError):
            pass
    raise AssertionError("Could not construct a readable chamber maze")


def _bands(active):
    positions = [int(i) for i in np.flatnonzero(active)]
    if not positions:
        return []
    groups = [[positions[0]]]
    for p in positions[1:]:
        if p-groups[-1][-1] >= 3:
            groups.append([p])
        else:
            groups[-1].append(p)
    return [(x[0], x[-1]) for x in groups]


def _parse_array(inp, wall, bounds):
    r0, c0, r1, c1 = bounds
    mask = inp == wall
    wr = [r for r in range(r0, r1+1)
          if np.any(mask[r, c0:c1] & mask[r, c0+1:c1+1])]
    wc = [c for c in range(c0, c1+1)
          if np.any(mask[r0:r1, c] & mask[r0+1:r1+1, c])]
    if r0 == 0 and 0 not in wr and np.any(mask[0, c0:c1+1]):
        wr.insert(0, -1)
    if r1 == inp.shape[0]-1 and r1 not in wr and np.any(mask[r1, c0:c1+1]):
        wr.append(inp.shape[0])
    if c0 == 0 and 0 not in wc and np.any(mask[r0:r1+1, 0]):
        wc.insert(0, -1)
    if c1 == inp.shape[1]-1 and c1 not in wc and np.any(mask[r0:r1+1, c1]):
        wc.append(inp.shape[1])
    wr = sorted(set(wr))
    wc = sorted(set(wc))
    assert len(wr) >= 2 and len(wc) >= 2
    assert all(b-a >= 2 for a, b in zip(wr, wr[1:]))
    assert all(b-a >= 2 for a, b in zip(wc, wc[1:]))
    cells = {}
    for i in range(len(wr)-1):
        for j in range(len(wc)-1):
            a, b, x, y = wr[i], wr[i+1], wc[j], wc[j+1]
            corners = [(a, x), (a, y), (b, x), (b, y)]
            four_corners = all(0 <= r < inp.shape[0] and 0 <= c < inp.shape[1]
                               and mask[r, c] for r, c in corners)
            sides = 0
            if 0 <= a < inp.shape[0] and np.any(mask[a, max(0,x+1):min(inp.shape[1],y)]):
                sides += 1
            if 0 <= b < inp.shape[0] and np.any(mask[b, max(0,x+1):min(inp.shape[1],y)]):
                sides += 1
            if 0 <= x < inp.shape[1] and np.any(mask[max(0,a+1):min(inp.shape[0],b), x]):
                sides += 1
            if 0 <= y < inp.shape[1] and np.any(mask[max(0,a+1):min(inp.shape[0],b), y]):
                sides += 1
            if four_corners or sides >= 3:
                box = (max(0,a+1), min(inp.shape[0]-1,b-1),
                       max(0,x+1), min(inp.shape[1]-1,y-1))
                if box[0] <= box[1] and box[2] <= box[3]:
                    cells[i, j] = box
    assert cells
    return {"wr": wr, "wc": wc, "cells": cells}


def _solve(inp):
    assert isinstance(inp, np.ndarray) and inp.ndim == 2
    values = [int(v) for v in np.unique(inp)]
    assert len(values) == 3
    h, w = inp.shape
    # Inter-array gaps (or the single-array margin) identify neutral background.
    flat = {v: sum(np.all(inp == v, axis=0)) + sum(np.all(inp == v, axis=1))
            for v in values}
    bg = max(values, key=lambda v: flat[v])
    assert flat[bg] >= 2
    remaining = [v for v in values if v != bg]
    routes = []
    for v in remaining:
        parts = _components(inp == v)
        if len(parts) == 2 and all(len(p) == (max(r for r,c in p)-min(r for r,c in p)+1)
                                           *(max(c for r,c in p)-min(c for r,c in p)+1)
                                   for p in parts):
            routes.append(v)
    assert len(routes) == 1
    route = routes[0]
    wall = next(v for v in remaining if v != route)
    mask = inp == wall
    row_bands = _bands(np.any(mask, axis=1))
    col_bands = _bands(np.any(mask, axis=0))
    assert not (len(row_bands) > 1 and len(col_bands) > 1)
    if len(row_bands) > 1:
        orientation = "vertical"
        bounds = [(a, int(np.flatnonzero(np.any(mask[a:b+1], axis=0))[0]), b,
                   int(np.flatnonzero(np.any(mask[a:b+1], axis=0))[-1])) for a,b in row_bands]
    elif len(col_bands) > 1:
        orientation = "horizontal"
        bounds = [(int(np.flatnonzero(np.any(mask[:,a:b+1], axis=1))[0]), a,
                   int(np.flatnonzero(np.any(mask[:,a:b+1], axis=1))[-1]), b)
                  for a,b in col_bands]
    else:
        orientation = "single"
        rs, cs = np.nonzero(mask)
        bounds = [(int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max()))]
    arrays = [_parse_array(inp, wall, bound) for bound in bounds]
    chambers = {(k, p): box for k, arr in enumerate(arrays)
                for p, box in arr["cells"].items()}
    adj = {node: [] for node in chambers}
    edge_info = {}
    def add(a, b, kind, place):
        key = _edge(a,b)
        assert key not in edge_info
        edge_info[key] = (kind, place)
        adj[a].append(b)
        adj[b].append(a)
    exterior = []
    for k, arr in enumerate(arrays):
        wr, wc = arr["wr"], arr["wc"]
        for (i,j), box in arr["cells"].items():
            node = (k,(i,j))
            for side, q in (("top",(i-1,j)),("bottom",(i+1,j)),
                            ("left",(i,j-1)),("right",(i,j+1))):
                if q in arr["cells"] and node < (k,q):
                    if side == "bottom":
                        pts = [(wr[i+1], c) for c in range(box[2], box[3]+1)]
                    elif side == "right":
                        pts = [(r, wc[j+1]) for r in range(box[0], box[1]+1)]
                    else:
                        continue
                    holes = [p for p in pts if inp[p] != wall]
                    assert len(holes) <= 1
                    if holes:
                        assert inp[holes[0]] == bg
                        add(node, (k,q), "door", holes[0])
                elif q not in arr["cells"]:
                    if side == "top":
                        pts = [(wr[i], c) for c in range(box[2], box[3]+1)]
                    elif side == "bottom":
                        pts = [(wr[i+1], c) for c in range(box[2], box[3]+1)]
                    elif side == "left":
                        pts = [(r, wc[j]) for r in range(box[0], box[1]+1)]
                    else:
                        pts = [(r, wc[j+1]) for r in range(box[0], box[1]+1)]
                    pts = [p for p in pts if 0 <= p[0] < h and 0 <= p[1] < w]
                    if not pts:
                        continue
                    holes = [p for p in pts if inp[p] != wall]
                    assert len(holes) <= 1 or len(pts) == len(holes)
                    if len(holes) == 1 and len(pts) != 1:
                        assert inp[holes[0]] == bg
                        exterior.append((k, side, node, holes[0]))
    connectors = []
    for k in range(len(arrays)-1):
        facing = ("bottom", "top") if orientation == "vertical" else ("right", "left")
        a = [x for x in exterior if x[0] == k and x[1] == facing[0]]
        b = [x for x in exterior if x[0] == k+1 and x[1] == facing[1]]
        if not a and not b:
            continue
        assert len(a) == len(b) == 1
        add(a[0][2], b[0][2], "connector", (a[0][3], b[0][3]))
        connectors.append((a[0][3], b[0][3]))
    assert len(exterior) == 2*len(connectors)
    markers = _components(inp == route)
    assert len(markers) == 2
    terminal_nodes = []
    full = []
    terminal_clipped = False
    marker_sizes = []
    marker_offsets = []
    for part in markers:
        hits = [node for node, box in chambers.items()
                if all(box[0] <= r <= box[1] and box[2] <= c <= box[3]
                       for r,c in part)]
        assert len(hits) == 1
        terminal_nodes.append(hits[0])
        box = chambers[hits[0]]
        full.append(len(part) == (box[1]-box[0]+1)*(box[3]-box[2]+1))
        marker_sizes.append((max(r for r,c in part)-min(r for r,c in part)+1,
                             max(c for r,c in part)-min(c for r,c in part)+1))
        marker_offsets.append((min(r for r,c in part)-box[0],
                               min(c for r,c in part)-box[2]))
        terminal_clipped |= (box[0] == 0 or box[1] == h-1
                             or box[2] == 0 or box[3] == w-1)
    assert terminal_nodes[0] != terminal_nodes[1]
    path = _path(adj, *terminal_nodes)
    # A forest gives exactly one terminal route, even with isolated corridors.
    assert len(edge_info) <= len(chambers)-1
    seen_nodes = set()
    for node in chambers:
        if node in seen_nodes:
            continue
        seen_nodes.add(node)
        stack = [(node, None)]
        while stack:
            current, parent = stack.pop()
            for nxt in adj[current]:
                if nxt == parent:
                    continue
                assert nxt not in seen_nodes
                seen_nodes.add(nxt)
                stack.append((nxt, current))
    expected = inp.copy()
    for node in path:
        r0,r1,c0,c1 = chambers[node]
        view = expected[r0:r1+1,c0:c1+1]
        view[view == bg] = route
    for a,b in zip(path,path[1:]):
        kind, place = edge_info[_edge(a,b)]
        if kind == "door":
            expected[place] = route
        else:
            p,q = place
            r0,r1 = sorted((p[0],q[0]))
            c0,c1 = sorted((p[1],q[1]))
            view = expected[r0:r1+1,c0:c1+1]
            view[view == bg] = route
    reached = set(path)
    stack = list(path)
    while stack:
        for q in adj[stack.pop()]:
            if q not in reached:
                reached.add(q)
                stack.append(q)
    branch = len(reached) > len(path)
    depths = {node:0 for node in path}
    frontier = list(path)
    while frontier:
        node = frontier.pop(0)
        for nxt in adj[node]:
            if nxt not in depths:
                depths[nxt] = depths[node]+1
                frontier.append(nxt)
    directions = []
    for a,b in zip(path,path[1:]):
        aa,bb = chambers[a],chambers[b]
        dr = (bb[0]+bb[1])-(aa[0]+aa[1])
        dc = (bb[2]+bb[3])-(aa[2]+aa[3])
        directions.append("v" if abs(dr) >= abs(dc) else "h")
    turns = sum(a != b for a,b in zip(directions,directions[1:]))
    sizes = {(b[1]-b[0]+1,b[3]-b[2]+1) for b in chambers.values()}
    gaps = tuple((bounds[k+1][0]-bounds[k][2]-1) if orientation == "vertical"
                 else (bounds[k+1][1]-bounds[k][3]-1)
                 for k in range(len(bounds)-1))
    connector_nodes = {node for a,b in edge_info for node in (a,b)
                       if edge_info[a,b][0] == "connector"}
    clip_sides = tuple(side for side, condition in (
        ("top", any(-1 in arr["wr"] for arr in arrays)),
        ("bottom", any(h in arr["wr"] for arr in arrays)),
        ("left", any(-1 in arr["wc"] for arr in arrays)),
        ("right", any(w in arr["wc"] for arr in arrays))) if condition)
    meta = {"bg":bg,"wall":wall,"route":route,"arrays":len(arrays),
            "orientation":orientation,"connector_spans":tuple(
                abs(p[1]-q[1])+1 if orientation == "vertical"
                else abs(p[0]-q[0])+1 for p,q in connectors),
            "rooms":len(chambers),"route_rooms":len(path),"branch":branch,
            "branch_depth":max(depths.values()),
            "full_markers":sum(full),"clipped":bool(clip_sides),
            "marker_sizes":tuple(sorted(marker_sizes)),
            "marker_offsets":tuple(marker_offsets),
            "unequal_terminals":marker_sizes[0] != marker_sizes[1],
            "clip_sides":clip_sides,
            "array_layouts":tuple((len(arr["wr"])-1,len(arr["wc"])-1)
                                   for arr in arrays),
            "array_room_counts":tuple(len(arr["cells"]) for arr in arrays),
            "room_sizes":tuple(sorted(sizes)),
            "regular_rooms":len(sizes) == 1,
            "stepped":any(len(arr["cells"]) < (len(arr["wr"])-1)*(len(arr["wc"])-1)
                          for arr in arrays),
            "single_strip":any(len(arr["wr"]) == 2 or len(arr["wc"]) == 2
                               for arr in arrays),
            "gaps":gaps,"turns":turns,
            "terminal_at_gate":any(node in connector_nodes for node in terminal_nodes),
            "terminal_clipped":terminal_clipped,
            "unused_corridor":any(a not in reached and b not in reached
                                   for a,b in edge_info),
            "unused_array":any(all(node not in reached for node in chambers
                                    if node[0] == k) for k in range(len(arrays))),
            "path":tuple(path)}
    return expected, meta


def _resolve_counts(seed, num_train, num_test):
    def legal(v):
        return isinstance(v,(int,np.integer)) and not isinstance(v,(bool,np.bool_))
    if num_train is not None and (not legal(num_train) or not 2 <= num_train <= 10):
        raise ValueError("num_train must be 2..10")
    if num_test is not None and (not legal(num_test) or not 1 <= num_test <= 4):
        raise ValueError("num_test must be 1..4")
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("train count must exceed test count")
        return int(num_train),int(num_test)
    if num_train is None and num_test is None:
        num_test = 1+seed%4
        lo = max(2,num_test+1)
        num_train = lo+(seed//4)%(11-lo)
    elif num_train is None:
        lo = max(2,num_test+1)
        num_train = lo+seed%(11-lo)
    else:
        num_test = 1+seed%min(4,num_train-1)
    return int(num_train),int(num_test)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if not isinstance(seed,(int,np.integer)) or isinstance(seed,(bool,np.bool_)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    ntrain, ntest = _resolve_counts(seed,num_train,num_test)
    # Only the invariant graph-and-rectangle rule is shared. Literal colors are
    # sometimes stable and sometimes independently assigned to each pair.
    stable = rng.random() < 0.28
    palette = tuple(int(x) for x in rng.choice(10,size=3,replace=False))
    branch_puzzle = rng.random() < 0.55
    family = str(rng.choice(["recombine", "single", "chain", "unused"],
                            p=[0.25, 0.25, 0.27, 0.23]))
    orientation_preference = (str(rng.choice(["vertical","horizontal"]))
                              if rng.random() < 0.45 else None)
    if family == "recombine":
        demonstrations = [(2,"aligned","vertical",False),
                          (2,"broad","horizontal",False)]
    elif family == "single":
        demonstrations = [(1,None,None,False),
                          (2,"broad",None,False)]
    elif family == "chain":
        demonstrations = [(3,"broad",None,False),
                          (2,"aligned",None,False)]
    else:
        demonstrations = [(3,"broad",None,True),
                          (2,"offset",None,False)]
    puzzle = {"train":[],"test":[]}
    seen = set()
    for section,count in (("train",ntrain),("test",ntest)):
        for index in range(count):
            if section == "train" and index < 2:
                arrays, span_mode, orientation, unused_array = demonstrations[index]
            elif family == "recombine":
                arrays, span_mode, orientation, unused_array = (2,
                    str(rng.choice(["aligned","offset","broad"])),
                    "vertical" if section == "test" else
                    str(rng.choice(["vertical","horizontal"])), False)
            elif family == "single":
                arrays = int(rng.choice([1,1,2] if section == "train" else [1,2,2]))
                span_mode, orientation, unused_array = "offset", None, False
            elif family == "chain":
                arrays = int(rng.choice([2,3] if section == "train" else [3,3,2]))
                span_mode, orientation, unused_array = "offset", None, False
            else:
                arrays = int(rng.choice([2,3] if section == "train" else [3,3,2]))
                span_mode, orientation, unused_array = "offset", None, arrays == 3
            if section == "test" and family == "recombine":
                span_mode = "offset"
            if orientation is None and arrays > 1:
                orientation = orientation_preference
            style = ("branch" if branch_puzzle and (index == 0 or
                      (section == "test" and rng.random() < 0.65) or rng.random() < 0.35)
                     else "isolated")
            for _ in range(20):
                colors = palette if stable else tuple(int(x) for x in rng.choice(10,size=3,replace=False))
                inp,out,_ = _make_pair(rng,colors,style,arrays,span_mode,
                                       orientation,unused_array)
                key = _key(inp)
                if key not in seen:
                    seen.add(key)
                    puzzle[section].append({"input":inp,"output":out})
                    break
            else:
                raise AssertionError("duplicate puzzle inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        assert isinstance(puzzle,dict) and set(puzzle) == {"train","test"}
        assert all(isinstance(puzzle[s],list) for s in ("train","test"))
        assert 2 <= len(puzzle["train"]) <= 10
        assert 1 <= len(puzzle["test"]) <= 4
        assert len(puzzle["train"]) > len(puzzle["test"])
        seen_inputs,seen_pairs = set(),set()
        train_meta,test_meta = [],[]
        for section in ("train","test"):
            for pair in puzzle[section]:
                assert isinstance(pair,dict) and set(pair) == {"input","output"}
                inp,out = pair["input"],pair["output"]
                for grid in (inp,out):
                    assert isinstance(grid,np.ndarray) and grid.ndim == 2
                    assert np.issubdtype(grid.dtype,np.integer)
                    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                    assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
                assert inp.shape == out.shape and not np.array_equal(inp,out)
                ik,pk = _key(inp),(_key(inp),_key(out))
                assert ik not in seen_inputs and pk not in seen_pairs
                seen_inputs.add(ik);seen_pairs.add(pk)
                assert _digest(inp,out) not in _SOURCE_DIGESTS
                expected,meta = _solve(inp)
                assert np.array_equal(expected,out)
                changed = inp != out
                assert np.all(inp[changed] == meta["bg"])
                assert np.all(out[changed] == meta["route"])
                assert np.all(out[inp == meta["wall"]] == meta["wall"])
                assert np.all(out[inp == meta["route"]] == meta["route"])
                assert len(_components(inp == meta["route"])) == 2
                assert len(_components(out == meta["route"])) == 1
                (train_meta if section == "train" else test_meta).append(meta)
        if any(m["branch"] for m in test_meta):
            assert any(m["branch"] for m in train_meta)
        if any(m["arrays"] == 1 for m in test_meta):
            assert any(m["arrays"] == 1 for m in train_meta)
        if any(m["arrays"] == 3 and not m["unused_array"] for m in test_meta):
            assert any(m["arrays"] == 3 and not m["unused_array"] for m in train_meta)
        if any(m["unused_array"] for m in test_meta):
            assert any(m["unused_array"] for m in train_meta)
        if any(any(span > 1 for span in m["connector_spans"]) for m in test_meta):
            assert any(any(span > 1 for span in m["connector_spans"]) for m in train_meta)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError("invalid generated chamber maze") from exc


def _equal(a,b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[role],y[role]) for x,y in zip(a[s],b[s])
        for role in ("input","output")) for s in ("train","test"))


def self_test() -> None:
    a = generate_puzzle(123,6,2)
    assert _equal(a,generate_puzzle(123,6,2))
    assert not _equal(a,generate_puzzle(124,6,2))
    for t,u in ((2,1),(10,1),(10,4),(6,2),(None,1),(None,4),(2,None),(10,None)):
        p = generate_puzzle(567+t if t is not None else 579,t,u)
        validate_generated_puzzle(p)
        if t is not None: assert len(p["train"]) == t
        if u is not None: assert len(p["test"]) == u
    for t,u in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4)):
        try:
            generate_puzzle(0,t,u)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    profiles = []
    metas = []
    stable_palettes = varied_palettes = 0
    stable_orientations = mixed_orientations = 0
    source_shapes_seen = 0
    counts = set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        counts.add((len(p["train"]),len(p["test"])))
        puzzle_roles = set()
        puzzle_orientations = set()
        for section in ("train","test"):
            for pair in p[section]:
                expected,meta = _solve(pair["input"].copy())
                assert np.array_equal(expected,pair["output"])
                metas.append(meta)
                puzzle_roles.add((meta["bg"],meta["wall"],meta["route"]))
                if meta["orientation"] != "single":
                    puzzle_orientations.add(meta["orientation"])
                source_shapes_seen += pair["input"].shape in {
                    (22,26),(29,24),(24,28),(28,28)}
                profiles.append((pair["input"].shape,meta["orientation"],
                                 meta["arrays"],meta["rooms"],meta["room_sizes"],
                                 meta["connector_spans"],meta["branch"],
                                 meta["full_markers"],meta["clipped"]))
        stable_palettes += len(puzzle_roles) == 1
        varied_palettes += len(puzzle_roles) > 1
        stable_orientations += len(puzzle_orientations) == 1
        mixed_orientations += len(puzzle_orientations) > 1
    assert len({t for t,u in counts}) >= 4
    assert len({u for t,u in counts}) >= 3
    assert len(set(profiles)) >= 100
    assert {p[2] for p in profiles} >= {1,2,3}
    assert {p[1] for p in profiles} >= {"single","vertical","horizontal"}
    assert any(p[6] for p in profiles) and any(not p[6] for p in profiles)
    assert any(p[7] == 2 for p in profiles)
    assert any(p[8] for p in profiles)
    assert any(2 in p[5] for p in profiles)
    assert any(any(span >= 8 for span in p[5]) for p in profiles)
    assert stable_palettes >= 3 and varied_palettes >= 10
    assert stable_orientations >= 3 and mixed_orientations >= 10
    assert source_shapes_seen >= 1
    assert sum(m["regular_rooms"] for m in metas) >= 10
    assert sum(m["stepped"] for m in metas) >= 20
    assert sum(m["clipped"] for m in metas) >= 10
    assert sum(m["terminal_clipped"] for m in metas) >= 1
    assert sum(m["unused_array"] for m in metas) >= 5
    assert sum(m["single_strip"] for m in metas) >= 10
    assert sum(m["branch"] for m in metas) >= 15
    assert sum(m["full_markers"] == 2 for m in metas) >= 20
    assert sum(m["full_markers"] < 2 for m in metas) >= 20
    assert sum(m["rooms"]-m["route_rooms"] == 1 for m in metas) >= 1
    assert max(m["turns"] for m in metas) >= 4
    assert {1,2,5,6} <= {size for m in metas for box in m["room_sizes"] for size in box}
    assert {2,6,7} <= {gap for m in metas for gap in m["gaps"]}
    assert any(m["arrays"] == 3 and m["branch"] for m in metas)
    assert any(m["stepped"] and m["branch"] and m["bg"] != 0 for m in metas)
