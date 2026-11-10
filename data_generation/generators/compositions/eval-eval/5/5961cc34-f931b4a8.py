"""Ink-sensitive stretch relays: input-only geometry, rendering, and routing.

Only NumPy and the standard library are used. No parent files are read at runtime.
The reference rule, including the broader compatible subset, is task_spec.md.
"""
from __future__ import annotations

import copy
import json

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
REQUIRED = frozenset({
    "frame_collision", "ignore_stencil", "tile_carrier", "scale_stencil",
    "early_fusion", "straight", "all_receivers", "opaque_background",
    "global_phase", "width_one",
    "visible_carrier", "visible_overwrite", "chain", "turn", "wide_port",
    "route_ignore_stencil", "route_tile_carrier", "route_scale_stencil",
})


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= v <= 30 for v in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int64, copy=False)


def _key(a):
    return tuple(a.shape), a.astype(np.uint8).tobytes()


def _components(cells):
    unseen, result = set(cells), []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        part, todo = {first}, [first]
        while todo:
            r, c = todo.pop()
            for dr, dc in DIRS:
                q = r + dr, c + dc
                if q in unseen:
                    unseen.remove(q)
                    part.add(q)
                    todo.append(q)
        result.append(part)
    return result


def _cells(mask):
    return {tuple(map(int, p)) for p in np.argwhere(mask)}


def _perimeter(h, w):
    return {(r, c) for r in range(h) for c in range(w)
            if r in (0, h - 1) or c in (0, w - 1)}


def _move(p, d, amount=1):
    dr, dc = DIRS[d]
    return p[0] + amount * dr, p[1] + amount * dc


def _port(h, w, side, start, width):
    if side == 0:
        return [(0, start + i) for i in range(width)]
    if side == 2:
        return [(h - 1, start + i) for i in range(width)]
    if side == 3:
        return [(start + i, 0) for i in range(width)]
    return [(start + i, w - 1) for i in range(width)]


def _render_fast(carrier, stencil, k, axis, bg):
    """Construction's vectorized replication and native stencil overlay."""
    base = np.repeat(carrier, k, axis=axis)
    rr, cc = np.indices(base.shape)
    n = carrier.shape[0]
    tile = stencil[rr % n, cc % n]
    return np.where(tile == bg, base, tile).astype(np.int64)


def _render_read(carrier, stencil, k, axis, bg, mode="normal"):
    """Solver rendering reads each target cell in local chamber coordinates."""
    n = len(carrier)
    h, w = (n * k, n) if axis == 0 else (n, n * k)
    result = np.empty((h, w), dtype=np.int64)
    for r in range(h):
        for c in range(w):
            sr, sc = (r // k, c) if axis == 0 else (r, c // k)
            base = int(carrier[r % n, c % n]) if mode == "tile_carrier" else int(carrier[sr, sc])
            ink = int(stencil[sr, sc]) if mode in ("scale_stencil", "early_fusion") else int(stencil[r % n, c % n])
            if mode == "ignore_stencil":
                ink = bg
            result[r, c] = ink if ink != bg or mode == "opaque_background" else base
    return result


def _parse_frame(inp, component, frame, bg, signal, cap_color):
    r0, c0 = min(p[0] for p in component), min(p[1] for p in component)
    r1, c1 = max(p[0] for p in component), max(p[1] for p in component)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    ih, iw = h - 2, w - 2
    n, long = min(ih, iw), max(ih, iw)
    assert 2 <= n <= 4 and long % n == 0 and 2 <= long // n <= 4
    assert 0 < r0 and r1 < inp.shape[0] - 1
    assert 0 < c0 and c1 < inp.shape[1] - 1
    rim = {(r0 + r, c0 + c) for r, c in _perimeter(h, w)}
    assert component <= rim
    assert all((r, c) in component for r, c in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)))
    ports = rim - component
    assert 1 <= len(ports) <= 3
    inks = {int(inp[p]) for p in ports}
    assert len(inks) == 1
    pc = next(iter(inks))
    assert pc not in (bg, signal, cap_color, frame)
    possible = [d for d, ok in enumerate((all(r == r0 for r, c in ports),
                 all(c == c1 for r, c in ports), all(r == r1 for r, c in ports),
                 all(c == c0 for r, c in ports))) if ok]
    assert len(possible) == 1
    side = possible[0]
    positions = sorted(c if side % 2 == 0 else r for r, c in ports)
    assert positions == list(range(positions[0], positions[0] + len(ports)))
    interior = inp[r0 + 1:r1, c0 + 1:c1]
    axis = 0 if ih > iw else 1
    if axis == 1:
        carrier, stencil = interior[:, :n].copy(), interior[:, n:2*n].copy()
        rest = interior[:, 2*n:]
    else:
        carrier, stencil = interior[:n].copy(), interior[n:2*n].copy()
        rest = interior[2*n:]
    assert np.all(carrier != bg) and np.all(rest == bg)
    forbidden = {frame, pc, cap_color}
    assert not forbidden & set(map(int, np.unique(interior)))
    k = long // n
    render = _render_read(carrier, stencil, k, axis, bg)
    return dict(bbox=(r0, c0, h, w), port=sorted(ports), side=side,
                carrier=carrier, stencil=stencil, n=n, k=k, axis=axis,
                render=render, port_color=pc)


def _decode(input_grid):
    """All conventions are read from this one input; no training/seed state."""
    inp = _grid(input_grid)
    h, w = inp.shape
    bg = int(inp[0, 0])
    assert all(int(inp[p]) == bg for p in ((0, w - 1), (h - 1, 0), (h - 1, w - 1)))
    parts = _components(_cells(inp != bg))
    boundary_parts = [p for p in parts if any(r in (0, h - 1) or c in (0, w - 1) for r, c in p)]
    assert len(boundary_parts) == 1
    launcher = boundary_parts[0]
    ends = [p for p in launcher if p[0] in (0, h - 1) or p[1] in (0, w - 1)]
    assert len(ends) == 1 and 3 <= len(launcher) <= 5
    start = ends[0]
    d = 2 if start[0] == 0 else 0 if start[0] == h - 1 else 1 if start[1] == 0 else 3
    ordered = [_move(start, d, i) for i in range(len(launcher))]
    assert set(ordered) == launcher
    signal, cap_color = int(inp[start]), int(inp[ordered[-1]])
    assert signal != cap_color and all(int(inp[p]) == signal for p in ordered[:-1])
    cap, tail = ordered[-1], ordered[:-1]
    candidates = []
    for frame in sorted(set(map(int, np.unique(inp))) - {bg, signal, cap_color}):
        try:
            components = _components(_cells(inp == frame))
            assert components
            objects = [_parse_frame(inp, part, frame, bg, signal, cap_color) for part in components]
            assert len({o["port_color"] for o in objects}) == 1
            rectangles, occupied = [], set(launcher)
            for obj in objects:
                r, c, hh, ww = obj["bbox"]
                box = {(rr, cc) for rr in range(r, r + hh) for cc in range(c, c + ww)}
                assert _separated(box, occupied)
                rectangles.append(box)
                occupied |= box
            assert _cells(inp != bg) <= occupied
            candidates.append((frame, objects))
        except (AssertionError, IndexError):
            continue
    assert len(candidates) == 1, "ambiguous or invalid receiver frames"
    frame, objects = candidates[0]
    return dict(inp=inp, bg=bg, signal=signal, cap_color=cap_color,
                frame=frame, tail=tail, cap=cap, initial=d, objects=objects)


def _sensors(obj, signal, frame_collision=False):
    r0, c0, h, w = obj["bbox"]
    if frame_collision:
        return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}
    return {(r0 + 1 + int(r), c0 + 1 + int(c))
            for r, c in np.argwhere(obj["render"] == signal)}


def _trace(scene, *, frame_collision=False, straight=False, width_one=False):
    """Input solver projects sensors onto lanes and chooses nearest distance."""
    h, w = scene["inp"].shape
    objects, signal = scene["objects"], scene["signal"]
    sensors = [_sensors(o, signal, frame_collision) for o in objects]
    trace = set(scene["tail"]) | {scene["cap"]}
    origins, d = [scene["cap"]], scene["initial"]
    active, transitions, segments, distances = [], [], [], []
    for _ in range(len(objects) + 1):
        dr, dc = DIRS[d]
        limit = (origins[0][0] + 1 if d == 0 else w - origins[0][1] if d == 1
                 else h - origins[0][0] if d == 2 else origins[0][1] + 1)
        hits = []
        for index, points in enumerate(sensors):
            for r, c in origins:
                for rr, cc in points:
                    distance = (rr - r) * dr + (cc - c) * dc
                    cross = (rr - r) * dc - (cc - c) * dr
                    if cross == 0 and 0 < distance < limit:
                        hits.append((distance, index))
        if not hits:
            segment = {_move(p, d, i) for p in origins for i in range(1, limit)}
            trace |= segment
            segments.append(segment)
            return dict(active=active, trace=trace, transitions=transitions,
                        segments=segments, distances=distances)
        distance = min(t[0] for t in hits)
        targets = {i for dist, i in hits if dist == distance}
        assert len(targets) == 1, "tied first receivers"
        index = next(iter(targets))
        assert index not in active, "routing cycle"
        segment = {_move(p, d, i) for p in origins for i in range(1, distance + 1)}
        trace |= segment
        segments.append(segment)
        active.append(index)
        distances.append(distance)
        obj = objects[index]
        if straight:
            r0, c0, hh, ww = obj["bbox"]
            if d in (0, 2):
                origins = [(r0 if d == 0 else r0 + hh - 1, c) for r, c in origins]
            else:
                origins = [(r, c0 + ww - 1 if d == 1 else c0) for r, c in origins]
            transitions.append((d, d, len(origins)))
        else:
            transitions.append((d, obj["side"], len(obj["port"])))
            origins, d = obj["port"], obj["side"]
            if width_one:
                origins = origins[:1]
    raise AssertionError("unterminated route")


def _paint(scene, route, *, all_receivers=False):
    out = np.full_like(scene["inp"], scene["bg"])
    for p in route["trace"]:
        out[p] = scene["signal"]
    active = range(len(scene["objects"])) if all_receivers else route["active"]
    for index in active:
        obj = scene["objects"][index]
        r, c, h, w = obj["bbox"]
        out[r:r+h, c:c+w] = scene["signal"]
        out[r+1:r+h-1, c+1:c+w-1] = obj["render"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the result entirely from visible input roles and local geometry."""
    scene = _decode(input_grid)
    return _paint(scene, _trace(scene))


def _separated(cells, occupied):
    return not any((r + dr, c + dc) in occupied for r, c in cells
                   for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def _box(obj):
    r, c, h, w = obj["bbox"]
    return {(rr, cc) for rr in range(r, r+h) for cc in range(c, c+w)}


def _simulate(scene):
    """Construction uses advancing slices, separately from solver projection."""
    objects, signal = scene["objects"], scene["signal"]
    owner = {}
    for index, obj in enumerate(objects):
        for p in _sensors(obj, signal):
            assert p not in owner
            owner[p] = index
    h, w = scene["inp"].shape
    d = scene["initial"]
    front = [_move(scene["cap"], d)]
    trace = set(scene["tail"]) | {scene["cap"]}
    active, transitions = [], []
    for _ in range((len(objects) + 1) * max(h, w) + 1):
        inside = [0 <= r < h and 0 <= c < w for r, c in front]
        if not any(inside):
            return dict(active=active, trace=trace, transitions=transitions)
        assert all(inside)
        trace.update(front)
        hits = {owner[p] for p in front if p in owner}
        if hits:
            assert len(hits) == 1
            index = next(iter(hits))
            assert index not in active
            active.append(index)
            obj = objects[index]
            transitions.append((d, obj["side"], len(obj["port"])))
            d = obj["side"]
            front = [_move(p, d) for p in obj["port"]]
        else:
            front = [_move(p, d) for p in front]
    raise AssertionError("unterminated construction route")


def _swatches(rng, n, palette, bg):
    inks = np.asarray(palette)
    rr, cc = np.indices((n, n))
    kind = int(rng.integers(6))
    if kind == 0:
        labels = np.zeros((n, n), dtype=int)
    elif kind == 1:
        line = rng.integers(len(inks), size=n)
        labels = np.tile(line, (n, 1))
    elif kind == 2:
        cut = int(rng.integers(1, n))
        labels = np.tile(np.where(np.arange(n) < cut, 0, 1), (n, 1))
    elif kind == 3:
        labels = (rr + cc) % len(inks)
    elif kind == 4:
        a, b = int(rng.integers(1, n)), int(rng.integers(1, n))
        labels = ((rr >= a) + 2 * (cc >= b)) % len(inks)
    else:
        labels = rng.integers(len(inks), size=(n, n))
    carrier = inks[labels]
    carrier = np.rot90(carrier, int(rng.integers(4))).copy()
    style = int(rng.integers(7))
    if style == 0:
        mask = np.zeros((n, n), dtype=bool)
    elif style == 1:
        mask = rng.random((n, n)) < float(rng.uniform(.10, .75))
    elif style == 2:
        mask = rr == int(rng.integers(n))
    elif style == 3:
        mask = (rr + cc + int(rng.integers(2))) % 2 == 0
    elif style == 4:
        mask = np.minimum.reduce([rr, cc, n-1-rr, n-1-cc]) == 0
    elif style == 5:
        mask = np.maximum(abs(rr-int(rng.integers(n))), abs(cc-int(rng.integers(n)))) <= 1
    else:
        mask = rng.random((n, n)) > float(rng.uniform(.15, .55))
    mask = np.rot90(mask, int(rng.integers(4))).copy()
    stencil = np.full((n, n), bg, dtype=np.int64)
    m = int(rng.integers(1, len(inks) + 1))
    stencil[mask] = rng.choice(inks[:m], size=int(mask.sum()))
    return carrier.astype(np.int64), stencil


def _local_object(rng, palette, bg, incoming, out_width, *, factor=None,
                  force_axis=None, suppression=False):
    n = int(rng.choice([2, 3, 4], p=[.50, .35, .15]))
    k = int(rng.choice([2, 3])) if factor is None else factor
    axis = int(rng.integers(2)) if force_axis is None else force_axis
    if suppression:
        n, k, axis = int(rng.choice([3, 4])), 2, (1 if incoming % 2 else 0)
    ih, iw = (n * k, n) if axis == 0 else (n, n * k)
    h, w = ih + 2, iw + 2
    carrier, stencil = _swatches(rng, n, palette, bg)
    lane = None
    if suppression:
        # The relation varies in row, n, other payload, orientation, and ports.
        # n=3,j=1 and n=4,j=1/2 all suppress the enlarged sensitive run.
        signal = palette[0]
        other = int(rng.choice(palette[1:]))
        lane = int(rng.integers(n))
        j = 1 if n == 3 else int(rng.choice([1, 2]))
        sensitive_phases = {(k*j+t) % n for t in range(k)}
        assert j not in sensitive_phases
        line_c = np.full(n, other, dtype=np.int64)
        line_c[j] = signal
        line_s = np.full(n, bg, dtype=np.int64)
        for t in sensitive_phases:
            line_s[t] = other
        if axis == 1:
            carrier[lane], stencil[lane] = line_c, line_s
        else:
            carrier[:, lane], stencil[:, lane] = line_c, line_s
    outgoing = int(rng.integers(4)) if suppression else (incoming + int(rng.choice([-1, 0, 1]))) % 4
    side_length = w - 2 if outgoing % 2 == 0 else h - 2
    width = min(out_width, side_length)
    start = int(rng.integers(1, side_length - width + 2))
    ports = _port(h, w, outgoing, start, width)
    return dict(bbox=(0, 0, h, w), port=ports, side=outgoing,
                carrier=carrier, stencil=stencil, n=n, k=k, axis=axis,
                render=_render_fast(carrier, stencil, k, axis, bg), lane=lane)


def _translated(obj, r0, c0):
    result = obj.copy()
    _, _, h, w = obj["bbox"]
    result["bbox"] = r0, c0, h, w
    result["port"] = [(r + r0, c + c0) for r, c in obj["port"]]
    return result


def _position(obj, origins, incoming, entry, gap):
    """Gap is distance from current origins to the near outside frame edge."""
    _, _, h, w = obj["bbox"]
    r, c = origins[0]
    if incoming == 1:
        return _translated(obj, r - entry, c + gap)
    if incoming == 3:
        return _translated(obj, r - entry, c - gap - w + 1)
    if incoming == 2:
        return _translated(obj, r + gap, c - entry)
    return _translated(obj, r - gap - h + 1, c - entry)


def _fits(obj, shape):
    r, c, h, w = obj["bbox"]
    return 0 < r and 0 < c and r + h < shape[0] and c + w < shape[1]


def _blank_scene(shape, roles, initial, lane, tail_length):
    bg, signal, cap_color, frame, pc = roles
    h, w = shape
    start = (h - 1, lane) if initial == 0 else (lane, 0) if initial == 1 else (0, lane) if initial == 2 else (lane, w - 1)
    tail = [_move(start, initial, i) for i in range(tail_length)]
    cap = _move(start, initial, tail_length)
    return dict(inp=np.full(shape, bg, dtype=np.int64), bg=bg, signal=signal,
                cap_color=cap_color, frame=frame, port_color=pc,
                tail=tail, cap=cap, initial=initial, objects=[])


def _encode(scene):
    inp = np.full_like(scene["inp"], scene["bg"])
    for obj in scene["objects"]:
        r, c, h, w = obj["bbox"]
        for rr, cc in _perimeter(h, w):
            inp[r + rr, c + cc] = scene["frame"]
        for p in obj["port"]:
            inp[p] = scene["port_color"]
        n = obj["n"]
        inp[r+1:r+1+n, c+1:c+1+n] = obj["carrier"]
        if obj["axis"] == 1:
            inp[r+1:r+1+n, c+1+n:c+1+2*n] = obj["stencil"]
        else:
            inp[r+1+n:r+1+2*n, c+1:c+1+n] = obj["stencil"]
    for p in scene["tail"]:
        inp[p] = scene["signal"]
    inp[scene["cap"]] = scene["cap_color"]
    scene["inp"] = inp
    return inp


def _try_scene(rng, roles, palette, active_count, decoys, skip, factors, wide):
    initial = int(rng.integers(4))
    lower = 15 if active_count <= 1 and not skip else 23 if skip else 19
    h, w = map(int, rng.integers(lower, 31, size=2))
    if skip:
        if initial % 2:
            w = int(rng.integers(27, 31))
        else:
            h = int(rng.integers(27, 31))
    transverse = w if initial % 2 == 0 else h
    lane = int(rng.integers(5, transverse - 5))
    scene = _blank_scene((h, w), roles, initial, lane, int(rng.integers(2, 5)))
    occupied = set(scene["tail"]) | {scene["cap"]}
    origins, incoming = [scene["cap"]], initial
    if skip:
        obj = _local_object(rng, palette, roles[0], incoming, int(rng.integers(1, 4)), suppression=True)
        placed = _position(obj, origins, incoming, obj["lane"] + 1, int(rng.integers(2, 4)))
        if not _fits(placed, (h, w)) or not _separated(_box(placed), occupied):
            return None
        scene["objects"].append(placed)
        occupied |= _box(placed)
    intended = []
    previous_trace = set(scene["tail"]) | {scene["cap"]}
    for index in range(active_count):
        success = False
        # This finite search preserves the intended active count and factors.
        for _ in range(140):
            out_width = (2 if index == 0 else 1) if wide else int(rng.choice([1, 2, 3], p=[.55, .35, .10]))
            obj = _local_object(rng, palette, roles[0], incoming, out_width, factor=factors[index])
            transverse_length = obj["bbox"][2 if incoming % 2 else 3] - 2
            width = len(origins)
            if transverse_length < width:
                continue
            entry = int(rng.integers(1, transverse_length - width + 2))
            gaps = rng.permutation(np.arange(2, 23))
            for gap in gaps:
                placed = _position(obj, origins, incoming, entry, int(gap))
                box = _box(placed)
                if (not _fits(placed, (h, w)) or not _separated(box, occupied)
                        or box & previous_trace):
                    continue
                test_scene = scene.copy()
                test_scene["objects"] = scene["objects"] + [placed]
                try:
                    route = _simulate(test_scene)
                except AssertionError:
                    continue
                target = len(scene["objects"])
                if route["active"] != intended + [target]:
                    continue
                scene["objects"].append(placed)
                intended.append(target)
                occupied |= box
                # Reserve earlier approaches, not the last unbounded exit ray.
                prefix_scene = scene.copy()
                prefix = _trace(prefix_scene)
                previous_trace = set(scene["tail"]) | {scene["cap"]}
                for segment in prefix["segments"][:-1]:
                    previous_trace |= segment
                origins, incoming = placed["port"], placed["side"]
                success = True
                break
            if success:
                break
        if not success:
            return None
    try:
        route = _simulate(scene)
    except AssertionError:
        return None
    if route["active"] != intended:
        return None
    for _ in range(decoys):
        success = False
        for _ in range(180):
            obj = _local_object(rng, palette, roles[0], int(rng.integers(4)), int(rng.integers(1, 4)))
            _, _, oh, ow = obj["bbox"]
            if oh + 2 > h or ow + 2 > w:
                continue
            placed = _translated(obj, int(rng.integers(1, h-oh)), int(rng.integers(1, w-ow)))
            if not _separated(_box(placed), occupied):
                continue
            if _sensors(placed, roles[1]) & route["trace"]:
                continue
            scene["objects"].append(placed)
            occupied |= _box(placed)
            success = True
            break
        if not success:
            return None
    _encode(scene)
    route = _simulate(scene)
    assert route["active"] == intended
    return {"input": scene["inp"], "output": _paint(scene, route)}


def _make_pair(rng, roles, palette, *, active_count, decoys=0, skip=False,
               factors=None, wide=False):
    factors = list(map(int, rng.choice([2, 3], size=active_count))) if factors is None else factors
    for _ in range(100):
        result = _try_scene(rng, roles, palette, active_count, decoys, skip, factors, wide)
        if result is not None:
            return result
    raise RuntimeError("bounded packing search exhausted for the intended route")


def _counterfactual(scene, mode):
    alternative = scene.copy()
    alternative["objects"] = [o.copy() for o in scene["objects"]]
    if mode in ("ignore_stencil", "tile_carrier", "scale_stencil", "early_fusion", "opaque_background"):
        for obj in alternative["objects"]:
            obj["render"] = _render_read(obj["carrier"], obj["stencil"], obj["k"], obj["axis"], scene["bg"], mode)
    elif mode == "global_phase":
        for obj in alternative["objects"]:
            base = np.repeat(obj["carrier"], obj["k"], axis=obj["axis"])
            rr, cc = np.indices(base.shape)
            r0, c0, _, _ = obj["bbox"]
            tile = obj["stencil"][(rr+r0+1) % obj["n"], (cc+c0+1) % obj["n"]]
            obj["render"] = np.where(tile == scene["bg"], base, tile)
    route = _trace(alternative, frame_collision=mode == "frame_collision",
                   straight=mode == "straight", width_one=mode == "width_one")
    return _paint(alternative, route, all_receivers=mode == "all_receivers"), route


def _evidence(pair):
    scene = _decode(pair["input"])
    route = _trace(scene)
    expected = _paint(scene, route)
    assert np.array_equal(expected, pair["output"])
    covered = set()
    for mode in ("frame_collision", "ignore_stencil", "tile_carrier", "scale_stencil",
                 "early_fusion", "straight", "all_receivers", "opaque_background", "global_phase", "width_one"):
        try:
            alternative, alt_route = _counterfactual(scene, mode)
            changed = not np.array_equal(expected, alternative)
            route_changed = alt_route["active"] != route["active"]
        except AssertionError:
            # An invalid mutant falsifies its full rule, but does not count as
            # evidence of a valid alternative activation sequence.
            changed, route_changed = True, False
        if changed:
            covered.add(mode)
        if route_changed and mode in ("ignore_stencil", "tile_carrier", "scale_stencil"):
            covered.add("route_" + mode)
    if len(route["active"]) >= 2:
        covered.add("chain")
    if any(a != b for a, b, width in route["transitions"]):
        covered.add("turn")
    if any(width >= 2 for a, b, width in route["transitions"]):
        covered.add("wide_port")
    for index in route["active"]:
        obj = scene["objects"][index]
        base = np.repeat(obj["carrier"], obj["k"], axis=obj["axis"])
        rr, cc = np.indices(base.shape)
        tile = obj["stencil"][rr % obj["n"], cc % obj["n"]]
        if np.any(tile == scene["bg"]):
            covered.add("visible_carrier")
        if np.any((tile != scene["bg"]) & (tile != base)):
            covered.add("visible_overwrite")
    return covered, scene, route


def _roles(rng, base, vary):
    colors = list(map(int, rng.permutation(10))) if vary else base
    roles = colors[:5]
    pool = [roles[1]] + colors[5:5 + int(rng.integers(1, 5))]
    return roles, pool


def _legal_count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _legal_count(num_train, "num_train", 2, 10)
    _legal_count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    base, vary = list(map(int, rng.permutation(10))), bool(rng.integers(2))
    train = None
    # Select two jointly informative witnesses, rather than fixing their images.
    for _ in range(40):
        roles, palette = _roles(rng, base, vary)
        first = _make_pair(rng, roles, palette, active_count=2, skip=True,
                           factors=[2, int(rng.choice([2, 3]))], wide=True)
        roles, palette = _roles(rng, base, vary)
        second = _make_pair(rng, roles, palette, active_count=int(rng.choice([2, 3], p=[.8, .2])),
                            factors=None, wide=True)
        a, sa, ra = _evidence(first)
        b, sb, rb = _evidence(second)
        factors = {s["objects"][i]["k"] for s, r in ((sa, ra), (sb, rb)) for i in r["active"]}
        widths = {v[2] for r in (ra, rb) for v in r["transitions"]}
        if REQUIRED <= a | b and factors == {2, 3} and {1, 2} <= widths:
            train = [first, second]
            break
    if train is None:
        raise RuntimeError("bounded evidence search exhausted")
    puzzle, seen = {"train": train, "test": []}, {_key(p["input"]) for p in train}
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        attempts = 0
        while len(puzzle[split]) < count:
            attempts += 1
            if attempts > 20 * count:
                raise RuntimeError("bounded distinct-input search exhausted")
            roles, palette = _roles(rng, base, vary)
            if split == "test":
                active_count = int(rng.choice([1, 2, 3], p=[.30, .55, .15]))
                factors = list(map(int, rng.choice([2, 3], size=active_count)))
                factors[int(rng.integers(active_count))] = 4
                decoys, skip = int(rng.choice([0, 1, 2], p=[.55, .35, .10])), False
            else:
                active_count = int(rng.choice([0, 1, 2, 3, 4], p=[.12, .27, .36, .22, .03]))
                factors = None
                decoys = int(rng.choice([0, 1, 2, 3], p=[.42, .35, .18, .05]))
                decoys = max(1, decoys) if active_count == 0 else decoys
                skip = False
            pair = _make_pair(rng, roles, palette, active_count=active_count,
                              decoys=decoys, skip=skip, factors=factors)
            if _key(pair["input"]) in seen:
                continue
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    for split in puzzle:
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Validate schema, local conventions, routing, labels, and teaching evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, evidence, training_factors, widths = set(), set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            covered, scene, route = _evidence({"input": inp, "output": out})
            assert len(route["active"]) <= 4 and len(scene["objects"]) - len(route["active"]) <= 3
            assert set(map(int, np.unique(out))) <= {scene["bg"], scene["signal"]} | set().union(*(
                set(map(int, np.unique(scene["objects"][i]["render"]))) for i in route["active"]))
            # Check the distinct cell-front implementation against actual input.
            simulated = _simulate(scene)
            assert simulated["active"] == route["active"] and simulated["trace"] == route["trace"]
            if split == "train":
                evidence |= covered
                training_factors.update(scene["objects"][i]["k"] for i in route["active"])
                widths.update(t[2] for t in route["transitions"])
            else:
                assert any(scene["objects"][i]["k"] == 4 for i in route["active"])
    assert REQUIRED <= evidence, "missing demonstration evidence: " + str(REQUIRED - evidence)
    assert {2, 3} <= training_factors and {1, 2} <= widths


def _normalized(a):
    mapping = {}
    return tuple(a.shape), bytes(mapping.setdefault(int(v), len(mapping)) for v in a.flat)


def _puzzle_key(p):
    return tuple((_key(x["input"]), _key(x["output"])) for s in ("train", "test") for x in p[s])


def self_test():
    """Bounded 42-seed diversity, counterfactual, and boundary regression audit."""
    puzzles, records = [], []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                before = pair["input"].copy()
                result = solve(before)
                assert np.array_equal(result, pair["output"])
                assert np.array_equal(before, pair["input"]), "solver mutated input"
                _, scene, route = _evidence(pair)
                records.append((split, scene, route, pair))
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len({_key(p["test"][0]["input"]) for p in puzzles}) == 42
    assert len({_normalized(p["test"][0]["output"]) for p in puzzles}) >= 38
    assert len({tuple(s["inp"].shape) for _, s, _, _ in records}) >= 30
    assert {s["initial"] for _, s, _, _ in records} == {0, 1, 2, 3}
    objects = [o for _, s, _, _ in records for o in s["objects"]]
    assert {o["axis"] for o in objects} == {0, 1}
    assert {o["n"] for o in objects} == {2, 3, 4}
    assert {o["k"] for o in objects} == {2, 3, 4}
    assert len({_normalized(o["carrier"]) for o in objects}) >= 40
    assert len({_normalized(o["stencil"]) for o in objects}) >= 40
    assert len({_key(o["stencil"] != s["bg"]) for _, s, _, _ in records for o in s["objects"]}) >= 40
    assert len({o["bbox"][2:] for o in objects}) >= 12
    assert any(np.all(o["stencil"] == s["bg"]) for _, s, _, _ in records for o in s["objects"])
    assert any(s["bg"] != 0 and np.any(p["output"] == 0) for _, s, _, p in records)
    transitions = [t for _, _, r, _ in records for t in r["transitions"]]
    assert {0, 1, 3} <= {(b-a) % 4 for a, b, width in transitions}
    assert {1, 2, 3} <= {width for a, b, width in transitions}
    assert {0, 1, 2, 3} <= {len(r["active"]) for _, _, r, _ in records}
    assert len({(s["bg"], s["signal"], s["frame"], s["cap_color"]) for _, s, _, _ in records}) >= 35
    assert _puzzle_key(generate_puzzle(314159, 2, 1)) == _puzzle_key(generate_puzzle(314159, 2, 1))
    # Every legal count pair, with a different seed, including the two-pair edge.
    for a in range(2, 11):
        for b in range(1, min(4, a-1) + 1):
            p = generate_puzzle(10000 + 10*a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = copy.deepcopy(puzzles[0])
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt label accepted")
    # Simple numeric rendering witness does not depend on the random sampler.
    carrier = np.array([[2, 1, 2], [3, 3, 3], [2, 2, 2]])
    stencil = np.array([[2, 0, 2], [0, 0, 0], [0, 0, 0]])
    normal = _render_read(carrier, stencil, 2, 1, 0)
    assert normal[0].tolist() == [2, 2, 2, 2, 2, 2]
    for mode in ("ignore_stencil", "tile_carrier", "scale_stencil"):
        assert 1 in _render_read(carrier, stencil, 2, 1, 0, mode)[0]
    # A complete hand-derived routing witness: a suppressed nearer frame is
    # crossed, a farther receiver activates, then its top port exits at col 18.
    # Expected output rows are numeric constants, not a constructor prediction.
    manual = _blank_scene((9, 25), [0, 1, 7, 8, 9], 1, 4, 2)
    near_carrier = np.array([[2, 2, 2], [2, 1, 2], [2, 2, 2]])
    near_stencil = np.array([[0, 0, 0], [2, 0, 2], [0, 0, 0]])
    far_carrier, far_stencil = np.array([[1, 2], [3, 2]]), np.zeros((2, 2), dtype=int)
    manual["objects"] = [
        dict(bbox=(3, 5, 5, 8), port=[(3, 7)], side=0, n=3, k=2, axis=1,
             carrier=near_carrier, stencil=near_stencil,
             render=_render_fast(near_carrier, near_stencil, 2, 1, 0)),
        dict(bbox=(3, 16, 4, 6), port=[(3, 18)], side=0, n=2, k=2, axis=1,
             carrier=far_carrier, stencil=far_stencil,
             render=_render_fast(far_carrier, far_stencil, 2, 1, 0)),
    ]
    expected_rows = [
        "0000000000000000001000000",
        "0000000000000000001000000",
        "0000000000000000001000000",
        "0000000000000000111111000",
        "1111111111111111111221000",
        "0000000000000000133221000",
        "0000000000000000111111000",
        "0000000000000000000000000",
        "0000000000000000000000000",
    ]
    expected = np.array([[int(v) for v in row] for row in expected_rows])
    assert np.array_equal(solve(_encode(manual)), expected)
    assert _trace(_decode(manual["inp"]))["active"] == [1]
    # Input-only solving is broader than the generator's factor-four test
    # schedule: this fixed same-factor scene, and any palette permutation, work.
    permutation = np.array([5, 0, 3, 2, 9, 4, 6, 1, 8, 7])
    assert np.array_equal(solve(permutation[manual["inp"]]), permutation[expected])
    return {"seeds": 42, "distinct_puzzles": 42,
            "canvas_shapes": len({tuple(s["inp"].shape) for _, s, _, _ in records}),
            "carrier_structures": len({_normalized(o["carrier"]) for o in objects}),
            "stencil_structures": len({_normalized(o["stencil"]) for o in objects}),
            "receiver_shapes": len({o["bbox"][2:] for o in objects})}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
