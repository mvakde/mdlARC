"""Optical stencil registration: input-only ray selection and scaled registration.

Only Python's standard library and NumPy are used. No parent code is imported.
"""
from __future__ import annotations

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("grids must be 2D integer ARC grids of size 1..30")
    return a.astype(int, copy=False)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _components(a, bg):
    """Orthogonal connectivity within one color; adjacent colors stay distinct."""
    todo = set(map(tuple, np.argwhere(a != bg)))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, cells = [start], []
        while stack:
            p = stack.pop()
            cells.append(p)
            for q in ((p[0]-1, p[1]), (p[0]+1, p[1]),
                      (p[0], p[1]-1), (p[0], p[1]+1)):
                if q in todo and a[q] == a[start]:
                    todo.remove(q)
                    stack.append(q)
        result.append((int(a[start]), tuple(sorted(cells))))
    return result


def _perimeter(r0, c0, r1, c1):
    return ({(r0, c) for c in range(c0, c1+1)}
            | {(r1, c) for c in range(c0, c1+1)}
            | {(r, c0) for r in range(r0, r1+1)}
            | {(r, c1) for r in range(r0, r1+1)})


def _l_cells(elbow, direction):
    r, c = elbow
    dr, dc = direction
    return {elbow, (r-dr, c), (r, c-dc)}


def _bar_cells(mid, axis, length=3):
    return {(mid[0] + (axis == 1)*k, mid[1] + (axis == 0)*k)
            for k in range(-(length//2), length//2+1)}


def _decode_object(color, cells):
    rows, cols = {p[0] for p in cells}, {p[1] for p in cells}
    if len(cells) == 3 and len(rows) == len(cols) == 2:
        absent = next(iter({(r, c) for r in rows for c in cols} - set(cells)))
        elbow = (min(rows)+max(rows)-absent[0], min(cols)+max(cols)-absent[1])
        return "source", {"color": color, "cells": set(cells), "elbow": elbow,
                          "direction": (elbow[0]-absent[0], elbow[1]-absent[1])}
    _need(len(cells) >= 3 and len(cells) % 2 == 1
          and (len(rows) == 1 or len(cols) == 1), "invalid optical glyph")
    _need(max(max(rows)-min(rows), max(cols)-min(cols))+1 == len(cells),
          "a mirror must be a continuous odd bar")
    axis = 0 if len(rows) == 1 else 1  # component of direction to reverse
    return "mirror", {"color": color, "cells": set(cells), "axis": axis,
                      "mid": ((min(rows)+max(rows))//2, (min(cols)+max(cols))//2)}


def _trace(h, w, sources, mirrors, *, recolor=True):
    """Pixel-level optical trace. The border itself absorbs the ray.

    Returns exposures at all inner pixels and explicit route/contact evidence.
    The swatches are transparent and are deliberately absent from occupied.
    """
    occupied = set().union(*(o["cells"] for o in sources+mirrors))
    exposure, routes, used = {}, [], set()
    for source in sources:
        dr, dc = source["direction"]
        r, c = source["elbow"]
        color = source["color"]
        r, c = r+dr, c+dc
        seen, path, contacts = set(), [], []
        while 0 < r < h-1 and 0 < c < w-1:
            state = r, c, dr, dc
            _need(state not in seen, "cyclic ray")
            seen.add(state)
            _need((r, c) not in occupied, "ray strikes an optical glyph")
            hits = []
            for i, mirror in enumerate(mirrors):
                mr, mc = mirror["mid"]
                axis = mirror["axis"]
                if ((axis == 0 and c == mc and mr-r == dr)
                        or (axis == 1 and r == mr and mc-c == dc)):
                    hits.append((i, axis))
            _need(len(hits) <= 1, "competing midpoint contacts")
            if hits:
                i, axis = hits[0]
                incoming = color
                if recolor:
                    color = mirrors[i]["color"]
                used.add(i)
                contacts.append((i, axis, (r, c), incoming, color))
                if axis == 0:
                    dr = -dr
                else:
                    dc = -dc
            exposure.setdefault((r, c), set()).add(color)
            path.append(((r, c), color))
            r, c = r+dr, c+dc
        routes.append({"path": path, "contacts": contacts, "exit_direction": (dr, dc)})
    _need(used == set(range(len(mirrors))), "unused mirror")
    return exposure, routes


def _parse_optics(inp):
    counts = np.bincount(inp.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _need(counts[bg] > inp.size/2, "background must be a strict majority")
    components = _components(inp, bg)
    frames = []
    for color, cells in components:
        r0, c0 = np.min(cells, axis=0)
        r1, c1 = np.max(cells, axis=0)
        ph, pw = int(r1-r0+1), int(c1-c0+1)
        if ph >= 11 and pw >= 11 and set(cells) == _perimeter(r0, c0, r1, c1):
            frames.append((color, (int(r0), int(c0), int(r1), int(c1))))
    _need(len(frames) == 1, "one optical frame is required")
    frame_color, (r0, c0, r1, c1) = frames[0]
    ph, pw = r1-r0+1, c1-c0+1
    _need((ph-3) % 4 == (pw-3) % 4 == 0, "invalid swatch lattice extent")
    nh, nw = (ph-3)//4, (pw-3)//4
    _need(2 <= nh <= 6 and 2 <= nw <= 6, "invalid miniature dimensions")
    sites = {(3+4*r, 3+4*c): (r, c) for r in range(nh) for c in range(nw)}
    plan = np.full((nh, nw), bg, dtype=int)
    sources, mirrors, squares = [], [], []
    for color, cells in components:
        if color == frame_color and set(cells) == _perimeter(r0, c0, r1, c1):
            continue
        inside = [r0 < r < r1 and c0 < c < c1 for r, c in cells]
        if any(inside):
            _need(all(inside) and color != frame_color, "glyph straddles the frame")
            local = tuple((r-r0, c-c0) for r, c in cells)
            if len(local) == 1:
                _need(local[0] in sites, "a singleton must be a lattice swatch")
                plan[sites[local[0]]] = color
            else:
                _need(not set(local).intersection(sites), "glyph occupies a swatch site")
                kind, obj = _decode_object(color, local)
                (sources if kind == "source" else mirrors).append(obj)
        else:
            _need(all(not (r0 <= r <= r1 and c0 <= c <= c1) for r, c in cells),
                  "object lies on the frame")
            a, b = np.min(cells, axis=0), np.max(cells, axis=0)
            side = int(b[0]-a[0]+1)
            _need(side >= 2 and b[1]-a[1]+1 == side and len(cells) == side**2,
                  "registration glyphs must be solid squares")
            _need(np.all(inp[a[0]:b[0]+1, a[1]:b[1]+1] == color), "non-solid seed")
            squares.append((int(a[0]), int(a[1]), side, color))
    _need(sources and mirrors, "sources and mirrors are both required")
    for obj in sources+mirrors:
        _need(all(abs(p[0]-q[0])+abs(p[1]-q[1]) > 1
                  for p in obj["cells"] for q in sites), "glyph touches a swatch site")
    exposure, routes = _trace(ph, pw, sources, mirrors)
    active = np.zeros(plan.shape, dtype=bool)
    for p, coord in sites.items():
        active[coord] = int(plan[coord]) != bg and int(plan[coord]) in exposure.get(p, ())
    return dict(background=bg, frame_color=frame_color, frame=(r0, c0, r1, c1),
                panel_shape=(ph, pw), plan=plan, active=active, sites=sites,
                sources=sources, mirrors=mirrors, exposure=exposure,
                routes=routes, squares=squares)


def _segments(plan, active, axis, length):
    h, w = plan.shape
    for r in range(h-(length-1 if axis == 0 else 0)):
        for c in range(w-(length-1 if axis == 1 else 0)):
            cells = tuple((r+(axis == 0)*k, c+(axis == 1)*k) for k in range(length))
            if all(active[p] for p in cells):
                yield (r, c), tuple(int(plan[p]) for p in cells), cells


def _overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _recover(inp):
    info = _parse_optics(inp)
    squares = info["squares"]
    _need(len(squares) >= 2, "at least two registration squares are required")
    if len({s[0] for s in squares}) == 1:
        axis = 1
    else:
        _need(len({s[1] for s in squares}) == 1, "registration must be a straight run")
        axis = 0
    squares.sort(key=lambda s: s[axis])
    side = squares[0][2]
    _need(all(s[2] == side for s in squares), "square sizes disagree")
    spacing = np.diff([s[axis] for s in squares])
    stride = int(spacing[0])
    _need(stride > side and np.all(spacing == stride), "nonuniform or nonpositive gutters")
    sequence = tuple(s[3] for s in squares)
    length = len(squares)
    matches = [(p, cells) for p, seq, cells in
               _segments(info["plan"], info["active"], axis, length) if seq == sequence]
    _need(len(matches) == 1, "seed must match exactly one exposed sequence")
    match, seed_sites = matches[0]
    base = (squares[0][0]-match[0]*stride, squares[0][1]-match[1]*stride)
    h, w = info["plan"].shape
    br, bc = base
    er, ec = br+(h-1)*stride+side, bc+(w-1)*stride+side
    _need(0 <= br < er <= inp.shape[0] and 0 <= bc < ec <= inp.shape[1],
          "full enlarged lattice must fit")
    fr = info["frame"]
    _need(not _overlap((br, bc, er, ec), (fr[0], fr[1], fr[2]+1, fr[3]+1)),
          "representations overlap")
    expected = inp.copy()
    seed_sites = set(seed_sites)
    for p in np.ndindex(info["plan"].shape):
        y, x = br+p[0]*stride, bc+p[1]*stride
        value = int(info["plan"][p])
        _need(np.all(inp[y:y+side, x:x+side]
                     == (value if p in seed_sites else info["background"])),
              "unexpected content on enlarged lattice")
        if info["active"][p]:
            expected[y:y+side, x:x+side] = value
    info.update(axis=axis, side=side, stride=stride, gutter=stride-side,
                sequence=sequence, length=length, match=match, base=base,
                seed_sites=seed_sites)
    return expected, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive selection, registration, and completion from this input alone."""
    return _recover(_grid(input_grid))[0]


def _geometry_options():
    result = []
    for h in range(3, 7):
        for w in range(3, 7):
            panel = (4*h+3, 4*w+3)
            for side in range(2, 5):
                for gutter in range(1, 4):
                    box = tuple((n-1)*(side+gutter)+side for n in (h, w))
                    for sep in (0, 1):
                        required = [max(panel[d], box[d]) for d in (0, 1)]
                        required[sep] = panel[sep]+box[sep]+1
                        if max(required) <= 30:
                            result.append((h, w, side, gutter, sep))
    return tuple(result)


_GEOMETRIES = _geometry_options()


def _bare_objects(scene, bg, frame):
    sources, mirrors = [], []
    for color, cells in _components(scene, bg):
        if color == frame:
            continue
        kind, obj = _decode_object(color, cells)
        (sources if kind == "source" else mirrors).append(obj)
    return sources, mirrors


def _put_object(scene, cells, color, bg, frame, sites):
    h, w = scene.shape
    for r, c in cells:
        if not (0 < r < h-1 and 0 < c < w-1) or scene[r, c] != bg:
            return None
        if any(abs(r-p[0])+abs(c-p[1]) <= 1 for p in sites):
            return None
        for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if q not in cells and scene[q] == color:
                return None
    out = scene.copy()
    for p in cells:
        out[p] = color
    try:
        sources, mirrors = _bare_objects(out, bg, frame)
        _trace(h, w, sources, mirrors)
    except ValueError:
        return None
    return out


def _add_mirror(scene, rng, bg, frame, palette, sites):
    sources, mirrors = _bare_objects(scene, bg, frame)
    _, routes = _trace(*scene.shape, sources, mirrors)
    options = []
    for route in routes:
        dr, dc = route["exit_direction"]
        after = route["contacts"][-1][2] if route["contacts"] else None
        tail = route["path"]
        if after is not None:
            last_index = max(i for i, (p, _) in enumerate(tail) if p == after)
            tail = tail[last_index+1:]
        for p, color in tail:
            if (p[0]-3) % 4 == (p[1]-3) % 4 == 2:
                for axis in (0, 1):
                    mid = (p[0]+(dr if axis == 0 else 0), p[1]+(dc if axis == 1 else 0))
                    options.append((mid, axis, color))
    rng.shuffle(options)
    for mid, axis, incoming in options:
        color = incoming if rng.random() < .25 else int(rng.choice(palette))
        trial = _put_object(scene, _bar_cells(mid, axis), color, bg, frame, sites)
        if trial is not None:
            return trial
    return None


def _add_source(scene, rng, bg, frame, palette, sites):
    options = [(p, d, k) for p in sites for d in ((-1,-1), (-1,1), (1,-1), (1,1))
               for k in (1, 2)]
    rng.shuffle(options)
    before_sources, before_mirrors = _bare_objects(scene, bg, frame)
    before, _ = _trace(*scene.shape, before_sources, before_mirrors)
    for p, d, k in options[:120]:
        elbow = (p[0]-k*d[0], p[1]-k*d[1])
        color = int(rng.choice(palette))
        trial = _put_object(scene, _l_cells(elbow, d), color, bg, frame, sites)
        if trial is None:
            continue
        sources, mirrors = _bare_objects(trial, bg, frame)
        exposure, _ = _trace(*scene.shape, sources, mirrors)
        if any(exposure.get(q, set())-before.get(q, set()) for q in sites):
            return trial
    return None


def _initial_scene(rng, h, w, bg, frame, palette, axis):
    # Start with a randomly positioned half-cell reflection. Transpose is a
    # geometrical construction aid, never a hidden solution convention.
    ah, aw = (h, w) if axis == 1 else (w, h)
    scene = np.full((4*ah+3, 4*aw+3), bg, dtype=int)
    for p in _perimeter(0, 0, scene.shape[0]-1, scene.shape[1]-1):
        scene[p] = frame
    r, c = int(rng.integers(ah-1)), int(rng.integers(aw-1))
    lead = int(rng.integers(min(r, c)+1))
    first = (3+4*(r-lead), 3+4*(c-lead))
    k = 1 if min(first) == 3 else int(rng.integers(1, 3))
    elbow = (first[0]-k, first[1]-k)
    for p in _l_cells(elbow, (1, 1)):
        scene[p] = int(palette[0])
    mid = (3+4*r+3, 3+4*c+2)
    for p in _bar_cells(mid, 0):
        scene[p] = int(palette[1])
    if axis == 0:
        scene = scene.T.copy()
    if rng.integers(2):
        scene = scene[::-1].copy()
    if rng.integers(2):
        scene = scene[:, ::-1].copy()
    sources, mirrors = _bare_objects(scene, bg, frame)
    _trace(*scene.shape, sources, mirrors)
    return scene


def _layout(rng, panel, box, sep):
    slack = 30-panel[sep]-box[sep]
    gap = int(rng.integers(1, min(3, slack)+1))
    required = [max(panel[d], box[d]) for d in (0, 1)]
    required[sep] = panel[sep]+box[sep]+gap
    if rng.random() < .22:
        n = int(rng.integers(max(required), 31))
        shape = (n, n)
    else:
        shape = tuple(int(rng.integers(n, min(30, n+4)+1)) for n in required)
    mini, base = [0, 0], [0, 0]
    first = int(rng.integers(shape[sep]-required[sep]+1))
    if rng.integers(2):
        mini[sep], base[sep] = first, first+panel[sep]+gap
    else:
        base[sep], mini[sep] = first, first+box[sep]+gap
    other = 1-sep
    mini[other] = int(rng.integers(shape[other]-panel[other]+1))
    base[other] = int(rng.integers(shape[other]-box[other]+1))
    return shape, tuple(mini), tuple(base)


def _feature_masks(info):
    """Two concrete competing optical rules, derived only from the input."""
    no_color, _ = _trace(*info["panel_shape"], info["sources"], info["mirrors"], recolor=False)
    no_recolor = np.zeros(info["plan"].shape, dtype=bool)
    straight = np.zeros(info["plan"].shape, dtype=bool)
    for pixel, coord in info["sites"].items():
        value = int(info["plan"][coord])
        no_recolor[coord] = value != info["background"] and value in no_color.get(pixel, ())
        if value == info["background"]:
            continue
        for source in info["sources"]:
            dr, dc = source["direction"]
            step = (pixel[0]-source["elbow"][0])*dr
            if (step > 0 and (pixel[1]-source["elbow"][1])*dc == step
                    and value == source["color"]):
                straight[coord] = True
    return no_recolor, straight


def _union_witness(info):
    """At least two sources each uniquely explain an unseeded exposed site."""
    contributions = []
    for route in info["routes"]:
        contributions.append({info["sites"][p] for p, color in route["path"]
                              if p in info["sites"] and info["active"][info["sites"][p]]
                              and info["plan"][info["sites"][p]] == color})
    essential = 0
    for i, own in enumerate(contributions):
        others = set().union(*(s for j, s in enumerate(contributions) if j != i))
        essential += bool(own-others-info["seed_sites"])
    return essential >= 2


def _transparency_witness(info):
    """An opaque wrong-color swatch would remove an unseeded square."""
    absorbing = np.zeros(info["plan"].shape, dtype=bool)
    for route in info["routes"]:
        for pixel, color in route["path"]:
            if pixel not in info["sites"]:
                continue
            p = info["sites"][pixel]
            value = int(info["plan"][p])
            if value != info["background"] and value != color:
                break
            if value != info["background"]:
                absorbing[p] = True
    difference = absorbing != info["active"]
    for p in info["seed_sites"]:
        difference[p] = False
    return bool(np.any(difference))


def _geometric_matches(info, shape):
    """Visible matches that survive canvas/frame constraints without optics."""
    result = []
    h, w = info["plan"].shape
    side, stride = info["side"], info["stride"]
    fr = info["frame"]
    for match, sequence, _ in _segments(info["plan"], info["plan"] != info["background"],
                                        info["axis"], info["length"]):
        if sequence != info["sequence"]:
            continue
        br = info["squares"][0][0]-match[0]*stride
        bc = info["squares"][0][1]-match[1]*stride
        er, ec = br+(h-1)*stride+side, bc+(w-1)*stride+side
        if (0 <= br < er <= shape[0] and 0 <= bc < ec <= shape[1]
                and not _overlap((br, bc, er, ec), (fr[0], fr[1], fr[2]+1, fr[3]+1))):
            result.append(match)
    return result


def _make_pair(rng, axis, *, test=False, chain=False, ambiguous=False, avoid=None):
    options = _GEOMETRIES
    if avoid is not None:
        options = [g for g in options if g[2] != avoid["side"]
                   and (g[0], g[1]) != avoid["plan"].shape]
    for attempt in range(160):
        h, w, side, gutter, sep = options[int(rng.integers(len(options)))]
        colors = rng.permutation(10)
        bg, frame = int(colors[0]), int(colors[1])
        if avoid is not None and bg == avoid["background"]:
            bg, frame = frame, bg
        palette = [int(v) for v in colors[2:2+int(rng.integers(3, 7))]]
        scene = _initial_scene(rng, h, w, bg, frame, palette, axis)
        sites = {(3+4*r, 3+4*c): (r, c) for r in range(h) for c in range(w)}
        desired_mirrors = int(rng.choice([1, 2, 3], p=[.45, .40, .15]))
        if chain or test:
            desired_mirrors = int(rng.choice([2, 3], p=[.75, .25]))
        for _ in range(desired_mirrors-1):
            trial = _add_mirror(scene, rng, bg, frame, palette, sites)
            if trial is None:
                break
            scene = trial
        desired_sources = int(rng.choice([1, 2, 3], p=[.32, .52, .16]))
        if chain:
            desired_sources = max(2, desired_sources)
        for _ in range(desired_sources-1):
            trial = _add_source(scene, rng, bg, frame, palette, sites)
            if trial is not None:
                scene = trial
        sources, mirrors = _bare_objects(scene, bg, frame)
        exposure, routes = _trace(*scene.shape, sources, mirrors)
        potential = {p: exposure.get(pixel, set()) for pixel, p in sites.items()}
        if len([v for v in potential.values() if v]) < 4:
            trial = _add_source(scene, rng, bg, frame, palette, sites)
            if trial is None:
                continue
            scene = trial
            sources, mirrors = _bare_objects(scene, bg, frame)
            exposure, routes = _trace(*scene.shape, sources, mirrors)
            potential = {p: exposure.get(pixel, set()) for pixel, p in sites.items()}
        if chain and (max(len(r["contacts"]) for r in routes) < 2 or len(sources) < 2):
            continue
        if chain and len({s["direction"] for s in sources}) < 2:
            continue
        if test and len(mirrors) < 2:
            continue
        # Colors are sampled after the optics, so a mixture of exposed,
        # wrong-color traversed, blank, and unvisited swatches is explicit.
        plan = rng.choice(palette, size=(h, w)).astype(int)
        active = np.zeros((h, w), dtype=bool)
        for p in np.ndindex(plan.shape):
            allowed = potential[p]
            if allowed and rng.random() < .72:
                plan[p] = int(rng.choice(sorted(allowed)))
                active[p] = True
            elif rng.random() < .18:
                plan[p] = bg
            else:
                other = [v for v in palette if v not in allowed]
                plan[p] = int(rng.choice(other)) if other else bg
        if int(active.sum()) < 3:
            continue
        visited_wrong = [p for p in np.ndindex(plan.shape)
                         if potential[p] and not active[p] and plan[p] != bg]
        if not visited_wrong:
            continue
        candidates = []
        for length in range(2, min(4, plan.shape[axis])+1):
            occurrences = {}
            for match, seq, cells in _segments(plan, active, axis, length):
                occurrences.setdefault(seq, []).append((match, cells))
            for seq, found in occurrences.items():
                if len(found) != 1:
                    continue
                match, seed_sites = found[0]
                if active.sum() <= length:
                    continue
                if chain and match == (0, 0):
                    continue
                if test and not 0 < match[1-axis] < plan.shape[1-axis]-1:
                    continue
                dark_runs = [(p, cells) for p, _, cells in
                             _segments(plan, np.array([[not potential[(r, c)] for c in range(w)]
                                                       for r in range(h)]), axis, length)]
                if not dark_runs:
                    continue
                candidates.append((match, seq, seed_sites, dark_runs))
        if not candidates:
            continue
        # Longer runs occur where the geometry permits; two-cell clues remain.
        weights = np.array([1.0 if len(c[1]) == 2 else .65 for c in candidates])
        chosen = candidates[int(rng.choice(len(candidates), p=weights/weights.sum()))]
        match, sequence, seed_sites, dark_runs = chosen
        _, decoy_cells = dark_runs[int(rng.integers(len(dark_runs)))]
        for p, value in zip(decoy_cells, sequence):
            plan[p] = value
        # A color change must affect a square that needs completing, not only
        # the already supplied registration marks.
        temporary = dict(plan=plan, active=active, sites=sites, sources=sources,
                         mirrors=mirrors, panel_shape=scene.shape, background=bg,
                         routes=routes, seed_sites=set(seed_sites))
        if chain and (not _union_witness(temporary) or not _transparency_witness(temporary)):
            continue
        no_color, straight = _feature_masks(temporary)
        remainder = np.ones(plan.shape, dtype=bool)
        for p in seed_sites:
            remainder[p] = False
        if not np.any((active != no_color) & remainder):
            continue
        if not np.any((active != straight) & remainder):
            continue
        for pixel, coord in sites.items():
            scene[pixel] = plan[coord]
        stride = side+gutter
        box = ((h-1)*stride+side, (w-1)*stride+side)
        shape, mini, base = _layout(rng, scene.shape, box, sep)
        inp = np.full(shape, bg, dtype=int)
        inp[mini[0]:mini[0]+scene.shape[0], mini[1]:mini[1]+scene.shape[1]] = scene
        out = inp.copy()
        # Ground truth is drawn from the sampled stencil, not solve().
        for p in np.ndindex(plan.shape):
            y, x = base[0]+p[0]*stride, base[1]+p[1]*stride
            if p in seed_sites:
                inp[y:y+side, x:x+side] = int(plan[p])
            if active[p]:
                out[y:y+side, x:x+side] = int(plan[p])
        try:
            predicted, info = _recover(inp)
        except ValueError:
            continue
        if ambiguous and len(_geometric_matches(info, inp.shape)) < 2:
            continue
        if np.array_equal(predicted, out):
            return {"input": inp, "output": out}, info
    return _constructive_pair(rng, axis, avoid=avoid, ambiguous=ambiguous)


def _constructive_pair(rng, axis, *, avoid=None, ambiguous=False):
    """Guaranteed clear two-bounce stencil, randomized within the same rule.

    Its first ray bounces twice on horizontal bars, then leaves through the
    bottom/right boundary. The second ray exposes a separate top-edge swatch.
    Transposition supplies vertical versions; reflections supply all rotations.
    """
    dimensions = {(3, 4), (3, 5)} if axis == 1 else {(4, 3), (5, 3)}
    options = [g for g in _GEOMETRIES if g[:2] in dimensions]
    if ambiguous:
        # A same-row dark copy is two sites from the true match. Make room
        # along the non-separation axis for both translated large rectangles.
        options = [g for g in options if g[4] == 1-axis
                   and (g[axis]-1)*(g[2]+g[3])+g[2]+2*(g[2]+g[3]) <= 30]
    if avoid is not None:
        options = [g for g in options if g[:2] != avoid["plan"].shape and g[2] != avoid["side"]]
    h, w, side, gutter, sep = options[int(rng.integers(len(options)))]
    cw = w if axis == 1 else h
    scene = np.zeros((15, 4*cw+3), dtype=int)
    for p in _perimeter(0, 0, 14, scene.shape[1]-1):
        scene[p] = 9
    for p in ((6, 2), (5, 2), (6, 1)):
        scene[p] = 1
    scene[10, 4:7] = 2
    scene[4, 8:11] = 5
    for p in ((4, 14), (5, 14), (4, 13)):
        scene[p] = 3
    plan = np.zeros((3, cw), dtype=int)
    plan[0, 3] = 3
    plan[1, :3] = [4, 2, 5]
    plan[2, :4] = [2, 5, 0, 5]
    active = np.zeros(plan.shape, dtype=bool)
    for p in ((0, 3), (1, 1), (1, 2), (2, 3)):
        active[p] = True
    seed_sites = {(1, 1), (1, 2)}
    if ambiguous:
        plan[1, :4] = [1, 2, 1, 2]
        plan[2, :3] = 0
        active[1] = False
        active[1, :2] = True
        seed_sites = {(1, 0), (1, 1)}
    for p in np.ndindex(plan.shape):
        scene[3+4*p[0], 3+4*p[1]] = plan[p]
    if axis == 0:
        scene, plan, active = scene.T.copy(), plan.T.copy(), active.T.copy()
        seed_sites = {(c, r) for r, c in seed_sites}
    for dim in (0, 1):
        if not ambiguous and rng.integers(2):
            scene, plan, active = (np.flip(a, dim).copy() for a in (scene, plan, active))
            seed_sites = {tuple(plan.shape[d]-1-p[d] if d == dim else p[d] for d in (0, 1))
                          for p in seed_sites}
    # An injective recoloring makes every visible role local and preserves
    # geometry; background/frame never share a payload color.
    permutation = rng.permutation(10)
    if avoid is not None and int(permutation[0]) == avoid["background"]:
        permutation[0], permutation[9] = permutation[9], permutation[0]
    scene, plan = permutation[scene], permutation[plan]
    bg = int(permutation[0])
    stride = side+gutter
    box = ((h-1)*stride+side, (w-1)*stride+side)
    shape, mini, base = _layout(rng, scene.shape, box, sep)
    if ambiguous:
        other = 1-sep
        shape, mini, base = list(shape), list(mini), list(base)
        shape[other] = max(shape[other], box[other]+2*stride)
        mini[other] = int(rng.integers(shape[other]-scene.shape[other]+1))
        base[other] = 2*stride+int(rng.integers(shape[other]-box[other]-2*stride+1))
        shape, mini, base = tuple(shape), tuple(mini), tuple(base)
    inp = np.full(shape, bg, dtype=int)
    inp[mini[0]:mini[0]+scene.shape[0], mini[1]:mini[1]+scene.shape[1]] = scene
    out = inp.copy()
    for p in np.ndindex(plan.shape):
        y, x = base[0]+p[0]*stride, base[1]+p[1]*stride
        if p in seed_sites:
            inp[y:y+side, x:x+side] = int(plan[p])
        if active[p]:
            out[y:y+side, x:x+side] = int(plan[p])
    if ambiguous:
        for dim in (0, 1):
            if rng.integers(2):
                inp, out = np.flip(inp, dim).copy(), np.flip(out, dim).copy()
    expected, info = _recover(inp)
    _need(np.array_equal(out, expected), "constructive stencil mismatch")
    _need(not ambiguous or len(_geometric_matches(info, inp.shape)) >= 2,
          "constructed dark registration origin does not fit")
    return {"input": inp, "output": out}, info


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _check_counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    axes = [0, 1]
    rng.shuffle(axes)
    previous = None
    for index in range(num_train):
        axis = axes[index] if index < 2 else int(rng.integers(2))
        for _ in range(12):
            pair, info = _make_pair(rng, axis, chain=index == 1, ambiguous=index == 0,
                                    avoid=previous if index == 1 else None)
            if _key(pair["input"]) not in seen:
                break
        else:
            raise RuntimeError("duplicate construction exhausted")
        seen.add(_key(pair["input"]))
        puzzle["train"].append(pair)
        if index == 0:
            previous = info
    for _ in range(num_test):
        for _ in range(12):
            pair, _ = _make_pair(rng, int(rng.integers(2)), test=True)
            if _key(pair["input"]) not in seen:
                break
        else:
            raise RuntimeError("duplicate construction exhausted")
        seen.add(_key(pair["input"]))
        puzzle["test"].append(pair)
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _need(inp.shape == out.shape and _key(inp) not in seen, "duplicate input or changed shape")
            seen.add(_key(inp))
            expected, info = _recover(inp)
            _need(np.array_equal(out, expected), "incorrect completion")
            bg, plan, active = info["background"], info["plan"], info["active"]
            _need(not np.array_equal(inp, out), "no additions")
            _need(np.array_equal(inp[inp != bg], out[inp != bg]), "original objects changed")
            _need(set(np.unique(out)) <= set(np.unique(inp)), "new color introduced")
            # Registration cannot be solved by a visible-plan match alone.
            visible = plan != bg
            full_matches = [p for p, seq, _ in _segments(plan, visible, info["axis"], info["length"])
                            if seq == info["sequence"]]
            _need(len(full_matches) >= 2, "missing dark registration decoy")
            _need(np.any(visible & ~active), "no dark visible swatches")
            _need(any(info["exposure"].get(pixel) and plan[p] != bg and not active[p]
                      for pixel, p in info["sites"].items()), "missing transparent wrong-color swatch")
            no_color, straight = _feature_masks(info)
            remainder = np.ones(plan.shape, dtype=bool)
            for p in info["seed_sites"]:
                remainder[p] = False
            _need(np.any((active != no_color) & remainder), "recoloring does not affect additions")
            _need(np.any((active != straight) & remainder), "reflection does not affect additions")
            _need(any(a != b for route in info["routes"]
                      for _, _, _, a, b in route["contacts"]), "no color-changing reflection")
            if split == "test":
                _need(0 < info["match"][1-info["axis"]] < plan.shape[1-info["axis"]]-1,
                      "test registration must lie on an interior line")
                _need(len(info["mirrors"]) >= 2, "test must recombine reflections")
            records[split].append(info)
    train = records["train"]
    _need({i["axis"] for i in train} == {0, 1}, "both registration axes must be demonstrated")
    _need({m["axis"] for i in train for m in i["mirrors"]} == {0, 1}, "both mirror axes must be taught")
    _need(any(_union_witness(i) and max(len(r["contacts"]) for r in i["routes"]) >= 2
              for i in train), "missing multiple-source mirror-chain demonstration")
    _need(len({s["direction"] for i in train for s in i["sources"]}) >= 2,
          "training must demonstrate more than one launch direction")
    for feature in ("background", "side"):
        _need(len({i[feature] for i in train}) >= 2, "two demonstrations must vary "+feature)
    _need(len({i["plan"].shape for i in train}) >= 2, "miniature shapes must vary")
    _need(any(i["match"] != (0, 0) for i in train), "training must rule out top-left registration")
    _need(any(len(_geometric_matches(i, _grid(pair["input"]).shape)) >= 2
              for i, pair in zip(train, puzzle["train"])),
          "training must contain two geometrically feasible registration matches")
    _need(any(_transparency_witness(i) for i in train),
          "training must distinguish transparent from absorbing wrong-color swatches")


def _signature(puzzle):
    return tuple((_key(pair["input"]), _key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test():
    """Bounded 42-seed structural, semantic-witness, and negative checks."""
    _rule_regressions()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _need(len({_signature(p) for p in puzzles}) == 42, "seed diversity collapsed")
    _need(_signature(puzzles[0]) == _signature(generate_puzzle(0)), "nondeterminism")
    records = []
    structures = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                expected, info = _recover(inp)
                _need(np.array_equal(solve(inp.copy()), pair["output"]), "input-only solve mismatch")
                records.append((split, info, inp.shape))
                mapping, normalized = {}, []
                for value in inp.flat:
                    normalized.append(mapping.setdefault(int(value), len(mapping)))
                structures.add((inp.shape, bytes(normalized)))
    for key, minimum in (("background", 8), ("side", 3), ("gutter", 3),
                         ("length", 2), ("match", 12), ("base", 30)):
        _need(len({i[key] for _, i, _ in records}) >= minimum, "insufficient variation: "+key)
    _need(len({i["plan"].shape for _, i, _ in records}) >= 7, "miniature geometry diversity")
    _need(len({shape for _, _, shape in records}) >= 30, "canvas diversity")
    _need(len(structures) >= 190, "only palette variation")
    _need({s["direction"] for _, i, _ in records for s in i["sources"]}
          == {(-1,-1), (-1,1), (1,-1), (1,1)}, "missing launch rotation")
    _need(any(len(i["sources"]) == 1 for _, i, _ in records), "missing single-source scenes")
    _need(any(len(i["sources"]) >= 3 for _, i, _ in records), "missing multi-source variation")
    _need(any(len(i["mirrors"]) >= 3 for _, i, _ in records), "missing longer mirror arrangements")
    _need(any(np.any(i["plan"] == i["background"]) for _, i, _ in records), "missing blank swatches")
    # The host verifier exhausts all legal count combinations. Keep this
    # generator-owned self-test bounded while exercising the count boundaries.
    for a, b in ((2, 1), (10, 1), (10, 4), (6, 3)):
        p = generate_puzzle(700+a*10+b, a, b)
        _need(len(p["train"]) == a and len(p["test"]) == b, "explicit counts")
    # Exercise the fallback directly; random occurrence is not relied upon.
    for seed in range(8):
        rng = np.random.default_rng(10000+seed)
        first, info = _constructive_pair(rng, seed % 2, ambiguous=True)
        second, _ = _constructive_pair(rng, 1-seed % 2, avoid=info)
        test, _ = _constructive_pair(rng, seed % 2)
        validate_generated_puzzle({"train": [first, second], "test": [test]})
    for counts in ((1,1), (11,1), (4,0), (6,5), (2,2), (3,4), (True,1), (4,False), (3.5,1), (None,1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
           for s in ("train", "test")}
    p = bad["test"][0]
    changed = tuple(np.argwhere(p["input"] != p["output"])[0])
    p["output"][changed] = p["input"][changed]
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("missing completion cell accepted")
    return {"seeds": 42, "pairs": len(records), "structures": len(structures),
            "canvas_shapes": len({shape for _, _, shape in records})}


def _rule_regressions():
    """A hand-specified stencil, with expected squares enumerated directly.

    Swatch (1,0) has color 4 and is traversed by color 1: it remains dark,
    and the ray still reaches the horizontal color-2 mirror. The enlarged
    [2,3] clue selects (1,1), rather than its unvisited duplicate at (2,0).
    """
    inp = np.zeros((25, 19), dtype=int)
    for p in _perimeter(0, 0, 14, 18):
        inp[p] = 9
    for p in ((6, 2), (5, 2), (6, 1)):
        inp[p] = 1
    inp[10, 4:7] = 2
    for p in ((6, 10), (5, 10), (6, 9)):
        inp[p] = 3
    plan = [[0, 0, 2, 0], [4, 2, 3, 0], [2, 3, 0, 3]]
    for r in range(3):
        for c in range(4):
            inp[3+4*r, 3+4*c] = plan[r][c]
    inp[20:22, 4:6] = 2
    inp[20:22, 7:9] = 3
    expected = inp.copy()
    expected[17:19, 7:9] = 2
    expected[23:25, 10:12] = 3
    for a, b in ((inp, expected), (inp.T.copy(), expected.T.copy())):
        _need(np.array_equal(solve(a), b), "hand-enumerated rule regression")
    info = _parse_optics(inp)
    expected_active = np.zeros((3, 4), dtype=bool)
    for p in ((0, 2), (1, 1), (1, 2), (2, 3)):
        expected_active[p] = True
    _need(np.array_equal(info["active"], expected_active), "transparent-filter mask regression")
    _need(info["exposure"][(9, 5)] == {2}, "contact must take the mirror color")
    _need(info["exposure"][(7, 3)] == {1}, "wrong-color swatch must not recolor light")
    # The full visible plan has two registration matches, but the exposed plan
    # has exactly one. This fixture is independent of the random sampler.
    matches = [p for p, seq, _ in _segments(info["plan"], info["active"], 1, 2)
               if seq == (2, 3)]
    _need(matches == [(1, 1)], "optical registration regression")
    ambiguous = inp.copy()
    # Two extra sources expose the formerly dark [2,3] copy. No arbitrary
    # origin tie-break is allowed once both locations qualify.
    for p in ((10, 2), (9, 2), (10, 1)):
        ambiguous[p] = 2
    for p in ((12, 6), (13, 6), (12, 5)):
        ambiguous[p] = 3
    _need(_parse_optics(ambiguous)["active"][2, 0]
          and _parse_optics(ambiguous)["active"][2, 1], "ambiguous fixture is not exposed")
    try:
        solve(ambiguous)
    except ValueError as error:
        _need("exactly one exposed sequence" in str(error), "wrong ambiguity rejection")
    else:
        raise AssertionError("two exposed registration matches accepted")
