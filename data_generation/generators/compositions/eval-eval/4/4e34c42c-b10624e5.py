"""Calibrated scrap maps: register scaled/mirrored partial views by landmarks.

Only 8 (landmark flags) and 9 (the reference frame) have fixed color roles.
The solver reads pixels, not generation coordinates or random state.
"""
from collections import deque
import numpy as np


FLAG = 8
FRAME = 9
FLAG_CELLS = ((0, 1), (1, 0), (1, 1), (2, 0))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer color grid")
    return a.astype(np.int8, copy=False)


def _groups(points):
    remaining = set(points)
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    group.add(p)
                    stack.append(p)
        groups.append(group)
    return groups


def _bounds(points):
    rs, cs = zip(*points)
    return min(rs), max(rs)+1, min(cs), max(cs)+1


def _tight(cells, background):
    r0, r1, c0, c1 = _bounds(cells)
    result = np.full((r1-r0, c1-c0), background, dtype=np.int8)
    for (r, c), color in cells.items():
        result[r-r0, c-c0] = color
    return result


def _render(cells, background, unit=1, signs=(1, 1)):
    fy, fx = signs
    raw = _tight({(fy*r, fx*c): v for (r, c), v in cells.items()}, background)
    return np.repeat(np.repeat(raw, unit, axis=0), unit, axis=1)


def _key_records(grid, component, background):
    """Decode complete flags and their identity squares, including handedness."""
    records = []
    flags = {p for p in component if grid[p] == FLAG}
    for region in _groups(flags):
        r0, r1, c0, c1 = _bounds(region)
        unit = (r1-r0)//3
        assert unit >= 1 and r1-r0 == 3*unit and c1-c0 == 2*unit, "malformed flag"
        candidates = []
        for fy, fx in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            ar = r0 if fy == 1 else r0+2*unit
            ac = c0 if fx == 1 else c0+unit
            expected = {(ar+fy*dr*unit+y, ac+fx*dc*unit+x)
                        for dr, dc in FLAG_CELLS
                        for y in range(unit) for x in range(unit)}
            if expected != region:
                continue
            square = {(ar+y, ac+x) for y in range(unit) for x in range(unit)}
            notch = {(ar+fy*2*unit+y, ac+fx*unit+x)
                     for y in range(unit) for x in range(unit)}
            if not square <= component or any(grid[p] != background for p in notch):
                continue
            colors = {int(grid[p]) for p in square}
            if len(colors) != 1 or next(iter(colors)) in (background, FLAG, FRAME):
                continue
            candidates.append((next(iter(colors)), ar, ac, unit, fy, fx))
        assert len(candidates) == 1, "flag does not identify one anchor/orientation"
        records.append(candidates[0])
    assert records, "every scrap needs a landmark"
    assert len({k[0] for k in records}) == len(records), "repeated identity within a scrap"
    assert len({k[3:] for k in records}) == 1, "inconsistent scrap scale/orientation"
    return sorted(records)


def _calibrate(grid, component, records, background, target_side):
    """Exact rational pixel scaling, then undo the observed axis reflections.

    A destination pixel is legal only if all contributing source subpixels
    agree. Padding on anchor-side boundaries makes translations integral.
    """
    _, ar, ac, side, fy, fx = records[0]
    relative = {(int(r)-ar, int(c)-ac): int(grid[r, c]) for r, c in component}
    r0, r1, c0, c1 = _bounds(relative)
    top, left = r0//side*side, c0//side*side
    bottom, right = ((r1+side-1)//side)*side, ((c1+side-1)//side)*side
    raw = np.full((bottom-top, right-left), background, dtype=np.int8)
    for (r, c), color in relative.items():
        raw[r-top, c-left] = color
    expanded = np.repeat(np.repeat(raw, target_side, axis=0), target_side, axis=1)
    h, w = expanded.shape[0]//side, expanded.shape[1]//side
    blocks = expanded.reshape(h, side, w, side)
    low, high = blocks.min(axis=(1, 3)), blocks.max(axis=(1, 3))
    assert np.array_equal(low, high), "nonintegral color boundary at reference scale"
    offset = [-top*target_side//side, -left*target_side//side]
    if fy == -1:
        low = low[::-1, :]
        offset[0] = h-offset[0]-target_side
    if fx == -1:
        low = low[:, ::-1]
        offset[1] = w-offset[1]-target_side
    return {(int(r)-offset[0], int(c)-offset[1]): int(low[r, c])
            for r, c in np.argwhere(low != background)}


def _decode(input_grid):
    grid = _grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[counts.argmax()])
    assert counts.max()*2 > grid.size and background not in (FLAG, FRAME), "unclear background"
    frame_points = set(map(tuple, np.argwhere(grid == FRAME)))
    assert frame_points, "missing reference frame"
    r0, r1, c0, c1 = _bounds(frame_points)
    assert r1-r0 >= 7 and c1-c0 >= 6
    perimeter = {(r, c) for r in range(r0, r1) for c in range(c0, c1)
                 if r in (r0, r1-1) or c in (c0, c1-1)}
    assert frame_points == perimeter, "expected one rectangular frame"
    components = _groups(map(tuple, np.argwhere((grid != background) & (grid != FRAME))))
    assert len(components) >= 2, "need several scraps"
    roots = [i for i, p in enumerate(components)
             if all(r0 < r < r1-1 and c0 < c < c1-1 for r, c in p)]
    assert len(roots) == 1, "frame must enclose exactly one scrap"
    root = roots[0]
    assert all(i == root or all(not (r0 <= r < r1 and c0 <= c < c1) for r, c in p)
               for i, p in enumerate(components)), "scrap crosses reference frame"
    all_records = [_key_records(grid, component, background) for component in components]
    target_side = int(all_records[root][0][3])
    parts = []
    for component, records in zip(components, all_records):
        _, ar, ac, unit, fy, fx = records[0]
        logical = _calibrate(grid, component, records, background, target_side)
        assert all((r-ar)*target_side % unit == 0 and (c-ac)*target_side % unit == 0
                   for _, r, c, *_ in records), "nonintegral landmark location"
        keys = {identity: (fy*(r-ar)*target_side//unit, fx*(c-ac)*target_side//unit)
                for identity, r, c, *_ in records}
        parts.append({"cells": logical, "keys": keys, "unit": unit,
                      "signs": (fy, fx), "physical": component, "records": records})
    identities = set().union(*(set(p["keys"]) for p in parts))
    for part in parts:
        for color in identities:
            positions = {p for p, v in part["cells"].items() if v == color}
            expected = ({(part["keys"][color][0]+y, part["keys"][color][1]+x)
                         for y in range(target_side) for x in range(target_side)}
                        if color in part["keys"] else set())
            assert positions == expected, "landmark identity also used as payload"
    return {"grid": grid, "background": background, "root": root,
            "parts": parts, "frame": (r0, r1, c0, c1), "side": target_side}


def _register(data):
    """Propagate exact key equalities; there is no overlap-size optimization."""
    parts, root = data["parts"], data["root"]
    offsets = {root: (0, 0)}
    world_keys = dict(parts[root]["keys"])
    while len(offsets) < len(parts):
        progress = False
        for i, part in enumerate(parts):
            if i in offsets:
                continue
            common = sorted(set(part["keys"]) & set(world_keys))
            if not common:
                continue
            deltas = {(world_keys[k][0]-part["keys"][k][0],
                       world_keys[k][1]-part["keys"][k][1]) for k in common}
            assert len(deltas) == 1, "inconsistent landmark translations"
            dr, dc = next(iter(deltas))
            offsets[i] = (dr, dc)
            for identity, (r, c) in part["keys"].items():
                pos = (r+dr, c+dc)
                assert identity not in world_keys or world_keys[identity] == pos
                world_keys[identity] = pos
            progress = True
        assert progress, "disconnected registration graph"
    union = {}
    for i, part in enumerate(parts):
        dr, dc = offsets[i]
        for (r, c), color in part["cells"].items():
            pos = r+dr, c+dc
            assert pos not in union or union[pos] == color, "unequal-color overlap"
            union[pos] = color
    side = data["side"]
    for identity, (r, c) in world_keys.items():
        assert all(union.get((r+y, c+x)) == identity for y in range(side) for x in range(side))
        assert all(union.get((r+dy*side+y, c+dx*side+x)) == FLAG
                   for dy, dx in FLAG_CELLS for y in range(side) for x in range(side))
        assert all((r+2*side+y, c+side+x) not in union
                   for y in range(side) for x in range(side)), "landmark notch filled"
    assert len(_groups(union)) == 1, "assembled map must be connected"
    return union, offsets


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the tightly cropped union in the framed scrap's pixel frame."""
    data = _decode(input_grid)
    union, _ = _register(data)
    root = data["parts"][data["root"]]
    output = _render(union, data["background"], 1, root["signs"])
    assert max(output.shape) <= 30, "assembled map exceeds ARC bounds"
    return output


def _ablation(data, *, scale=True, horizontal=True, vertical=True, focus_only=False):
    """Diagnostic counterfactual: align first keys but omit a required mechanism."""
    _, offsets = _register(data)
    root = data["parts"][data["root"]]
    ry, rx = root["signs"]
    union = {}
    order = [data["root"]] + [i for i in range(len(data["parts"])) if i != data["root"]]
    for i in order[:1] if focus_only else order:
        part = data["parts"][i]
        size = data["side"] if scale else int(part["unit"])
        cells = (part["cells"] if scale else
                 _calibrate(data["grid"], part["physical"], part["records"], data["background"], size))
        fy = ry if vertical else part["signs"][0]
        fx = rx if horizontal else part["signs"][1]
        dr, dc = offsets[i]
        for (r, c), color in cells.items():
            y = r if fy == 1 else size-1-r
            x = c if fx == 1 else size-1-c
            union.setdefault((ry*dr+y, rx*dc+x), color)
    return _tight(union, data["background"])


def _analysis(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    data = _decode(inp)
    union, offsets = _register(data)
    root = data["parts"][data["root"]]
    expected = _render(union, data["background"], 1, root["signs"])
    assert np.array_equal(out, expected), "incorrect calibrated union/crop"
    assert np.array_equal(out, _ablation(data)), "counterfactual baseline inconsistent"
    assert FRAME not in out and set(np.unique(inp))-{FRAME} == set(np.unique(out)), "palette changed"
    assert not np.array_equal(out, _ablation(data, focus_only=True)), "reference already complete"
    adjacency = [{j for j, b in enumerate(data["parts"]) if j != i and set(a["keys"]) & set(b["keys"])}
                 for i, a in enumerate(data["parts"])]
    distances = {data["root"]: 0}
    queue = deque(distances)
    while queue:
        i = queue.popleft()
        for j in sorted(adjacency[i]):
            if j not in distances:
                distances[j] = distances[i]+1
                queue.append(j)
    placed = [{(r+offsets[i][0], c+offsets[i][1]) for r, c in p["cells"]}
              for i, p in enumerate(data["parts"])]
    redundant = sum(not (p-set().union(*(q for j, q in enumerate(placed) if i != j)))
                    for i, p in enumerate(placed))
    links = {tuple(sorted(p["keys"])) for p in data["parts"] if len(p["keys"]) == 2}
    degrees = {k: sum(k in edge for edge in links) for k in set().union(*(set(p["keys"]) for p in data["parts"]))}
    boundary_fine = any(offset % data["side"] and union.get((r+dy, c+dx), data["background"]) != color
                        for (r, c), color in union.items()
                        for dy, dx, offset in ((-1, 0, r), (1, 0, r+1), (0, -1, c), (0, 1, c+1)))
    return {"background": data["background"], "parts": len(placed),
            "landmarks": len(set().union(*(set(p["keys"]) for p in data["parts"]))),
            "sides": tuple(int(p["unit"]) for p in data["parts"]),
            "signs": tuple(p["signs"] for p in data["parts"]),
            "root_side": int(root["unit"]), "root_signs": root["signs"],
            "hops": max(distances.values()), "redundant": redundant, "fine_geometry": boundary_fine,
            "branch": max(degrees.values()) >= 3, "cycle": len(links) >= len(degrees),
            "scale_witness": not np.array_equal(out, _ablation(data, scale=False)),
            "h_witness": not np.array_equal(out, _ablation(data, horizontal=False)),
            "v_witness": not np.array_equal(out, _ablation(data, vertical=False)),
            "union": union}


def _route(start, finish, blocked, shape, rng):
    queue, previous = deque([start]), {start: None}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    rng.shuffle(directions)
    while queue:
        p = queue.popleft()
        if p == finish:
            path = []
            while p is not None:
                path.append(p)
                p = previous[p]
            return path
        for dy, dx in directions:
            q = p[0]+dy, p[1]+dx
            if (0 <= q[0] < shape[0] and 0 <= q[1] < shape[1]
                    and q not in blocked and q not in previous):
                previous[q] = p
                queue.append(q)
    raise AssertionError("no connector route")


def _make_fragments(rng, n, topology, fallback=False, allow_fine=True):
    anchor_base = (1 if fallback or n == 5 or not allow_fine else
                   2 if rng.random() < (.65 if n == 3 else .30) else 1)
    kh, kw = 3*anchor_base, 2*anchor_base
    if fallback:
        shape = (10, 11)
        positions = [(1, 1), (1, 5), (5, 5), (5, 1), (1, 9)][:n]
    else:
        shape = ((15, 15) if anchor_base == 2 else
                 (int(rng.integers(9, 15)), int(rng.integers(9, 15))))
        if anchor_base == 2:
            # Fine observations need compact ports to leave room for several
            # views. Unequal row/column offsets still come from the landmarks.
            row = int(rng.integers(1, 3))
            col = int(rng.integers(1, 3))
            gap = int(rng.integers(1, 4))
            shift = int(rng.integers(-1, 2))
            positions = [(row, col), (row, col+kw+gap),
                         (row+kh+1, col+kw+gap+shift), (row+kh+1, col+shift)][:n]
            assert all(0 <= r and r+kh <= shape[0] and 0 <= c and c+kw <= shape[1]
                       for r, c in positions)
        else:
            positions = []
            for _ in range(n):
                candidates = [(r, c) for r in range(1, shape[0]-kh) for c in range(1, shape[1]-kw)
                              if all(r+kh < y or y+kh < r or c+kw < x or x+kw < c for y, x in positions)]
                assert candidates, "landmarks do not fit"
                positions.append(candidates[int(rng.integers(len(candidates)))])
        # Start with a short edge; longer edges can use a smaller pixel unit.
        a, b = min(((i, j) for i in range(n) for j in range(i+1, n)),
                   key=lambda ij: sum(abs(positions[ij[0]][k]-positions[ij[1]][k]) for k in (0, 1)))
        order = [a, b]
        while len(order) < n:
            rest = [i for i in range(n) if i not in order]
            order.append(min(rest, key=lambda i: sum(abs(positions[i][k]-positions[order[-1]][k]) for k in (0, 1))))
        positions = [positions[i] for i in order]
    palette = [int(x) for x in rng.permutation(8)]
    background, identities = palette[0], palette[1:n+1]
    details = palette[n+1:n+1+int(rng.integers(1, min(2, 7-n)+1))]
    keys, reserved = [], set()
    for (r, c), identity in zip(positions, identities):
        cells = {(r+y, c+x): identity for y in range(anchor_base) for x in range(anchor_base)}
        cells.update({(r+dy*anchor_base+y, c+dx*anchor_base+x): FLAG
                      for dy, dx in FLAG_CELLS for y in range(anchor_base) for x in range(anchor_base)})
        keys.append(cells)
        reserved.update((r+dy, c+dx) for dy in range(kh) for dx in range(kw))
    edges = [(i-1, i) for i in range(1, n)]
    if topology == "branch" and n == 5:
        edges[-1] = (2, 4)
    if topology == "cycle" and n >= 4:
        edges.append((n-1, 0))
    payloads = []
    for a, b in edges:
        def ports(index):
            r, c = positions[index]
            return [(y, x) for y, x in ((r-1, c), (r+kh, c), (r+anchor_base, c-1), (r+anchor_base, c+kw))
                    if 0 <= y < shape[0] and 0 <= x < shape[1] and (y, x) not in reserved]
        pa, pb = ports(a), ports(b)
        if anchor_base == 2:
            start, finish = min(((p, q) for p in pa for q in pb),
                                key=lambda pq: abs(pq[0][0]-pq[1][0])+abs(pq[0][1]-pq[1][1]))
        else:
            start, finish = pa[int(rng.integers(len(pa)))], pb[int(rng.integers(len(pb)))]
        payload = set(_route(start, finish, reserved, shape, rng))
        if not fallback:
            for _ in range(int(rng.integers(1, 10))):
                lo_r = min(positions[a][0], positions[b][0])
                hi_r = max(positions[a][0], positions[b][0])+kh
                lo_c = min(positions[a][1], positions[b][1])
                hi_c = max(positions[a][1], positions[b][1])+kw
                frontier = sorted({(r+dy, c+dx) for r, c in payload
                                   for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                                   if 0 <= r+dy < shape[0] and 0 <= c+dx < shape[1]
                                   and (anchor_base == 1 or (lo_r <= r+dy < hi_r and lo_c <= c+dx < hi_c))
                                   and (r+dy, c+dx) not in reserved and (r+dy, c+dx) not in payload})
                if frontier:
                    payload.add(frontier[int(rng.integers(len(frontier)))])
        payloads.append(payload)
    # A globally defined pattern makes all crossing connector cells agree.
    axis, band, phase = int(rng.integers(2)), int(rng.integers(1, 4)), int(rng.integers(3))
    fragments = []
    for (a, b), payload in zip(edges, payloads):
        cells = dict(keys[a])
        cells.update(keys[b])
        cells.update({p: details[(p[axis]//band+phase) % len(details)] for p in payload})
        assert len(_groups(cells)) == 1
        fragments.append(cells)
    return fragments, keys, background


def _pack(rasters, background, rng, fallback=False):
    root = rasters[0]
    framed = np.full((root.shape[0]+4, root.shape[1]+4), background, dtype=np.int8)
    framed[0, :] = framed[-1, :] = FRAME
    framed[:, 0] = framed[:, -1] = FRAME
    framed[2:-2, 2:-2] = root
    assert max(framed.shape) <= 29, "reference frame too large"
    canvases = [(30, 30)] if fallback else [
        (int(rng.integers(max(framed.shape[0], 16), 31)),
         int(rng.integers(max(framed.shape[1], 16), 31))) for _ in range(14)] + [(30, 30)]*6
    for height, width in canvases:
        inp = np.full((height, width), background, dtype=np.int8)
        blocked = np.zeros((height, width), dtype=bool)
        if framed.shape[0] > height or framed.shape[1] > width:
            continue
        top = int(rng.integers(height-framed.shape[0]+1))
        left = int(rng.integers(width-framed.shape[1]+1))
        inp[top:top+framed.shape[0], left:left+framed.shape[1]] = framed
        blocked[max(0, top-1):min(height, top+framed.shape[0]+1),
                max(0, left-1):min(width, left+framed.shape[1]+1)] = True
        order = sorted(range(1, len(rasters)), key=lambda i: -np.count_nonzero(rasters[i] != background))
        success = True
        for i in order:
            raster = rasters[i]
            ph, pw = raster.shape
            if ph > height or pw > width:
                success = False
                break
            points = [tuple(p) for p in np.argwhere(raster != background)]
            allowed = np.ones((height-ph+1, width-pw+1), dtype=bool)
            for r, c in points:
                allowed &= ~blocked[r:r+allowed.shape[0], c:c+allowed.shape[1]]
            choices = np.argwhere(allowed)
            if len(choices) == 0:
                success = False
                break
            top, left = map(int, choices[int(rng.integers(len(choices)))])
            for r, c in points:
                inp[r+top, c+left] = raster[r, c]
                blocked[max(0, r+top-1):min(height, r+top+2),
                        max(0, c+left-1):min(width, c+left+2)] = True
        if success and np.count_nonzero(inp == background)*2 > inp.size:
            return inp
    raise AssertionError("scraps do not fit")


def _build_pair(rng, n, topology, witness, redundant, fallback=False, allow_fine=True):
    fragments, keys, background = _make_fragments(rng, n, topology, fallback, allow_fine)
    # A redundant observation can be a glyph alone or a glyph with a retained
    # irregular portion of its connecting material.
    if redundant:
        contained = dict(keys[1])
        if rng.random() < .5 and not fallback:
            for _ in range(int(rng.integers(1, 5))):
                candidates = sorted(p for p in fragments[0] if p not in contained and fragments[0][p] != FLAG
                                    and fragments[0][p] not in {v for key in keys for v in key.values()}
                                    and any(abs(p[0]-q[0])+abs(p[1]-q[1]) == 1 for q in contained))
                if candidates:
                    p = candidates[int(rng.integers(len(candidates)))]
                    contained[p] = fragments[0][p]
        fragments.append(contained)
    root_unit = 1 if witness == 0 else 2 if witness == 1 else int(rng.integers(1, 3))
    signs = [(int(rng.choice([-1, 1])), int(rng.choice([-1, 1])))]
    units = [root_unit]
    for i in range(1, len(fragments)):
        # Both axes differ in at least one training scrap. Other scraps vary
        # independently, including unchanged scale and singly reflected cases.
        if i == 1:
            signs.append((-signs[0][0], -signs[0][1]))
        elif witness == 0 and i == len(fragments)-1:
            signs.append((signs[0][0], -signs[0][1]))
        elif witness == 1 and i == 2:
            signs.append((-signs[0][0], signs[0][1]))
        else:
            signs.append((int(rng.choice([-1, 1])), int(rng.choice([-1, 1]))))
        if i == 1:
            units.append(2 if root_unit == 1 else 1)
        elif fallback:
            units.append(1)
        else:
            span = max(_tight(fragments[i], background).shape)
            maximum = min(3, 18//span)
            units.append(int(rng.integers(1, max(1, maximum)+1)))
    rasters = [_render(part, background, unit, flip)
               for part, unit, flip in zip(fragments, units, signs)]
    inp = _pack(rasters, background, rng, fallback)
    union = {}
    for part in fragments:
        for p, color in part.items():
            assert p not in union or union[p] == color
            union[p] = color
    out = _render(union, background, root_unit, signs[0])
    assert max(out.shape) <= 30
    pair = {"input": inp, "output": out}
    info = _analysis(pair)
    assert info["scale_witness"] and info["h_witness"] and info["v_witness"]
    if n >= 4 and topology != "cycle":
        assert info["hops"] >= 2, "need indirect registration"
    return pair


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    num_train, num_test = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if seed < 0 or not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError("nonnegative seed and legal counts (2..10 train > 1..4 test) required")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    fine_seen = False
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            witness = index if split == "train" and index < 2 else None
            n = 3 if witness == 0 else 4 if witness == 1 else int(rng.integers(3, 5) if split == "train" else rng.integers(4, 6))
            topology = "path" if witness is not None else str(rng.choice(["path", "branch", "cycle"]))
            # Tests use at least two-hop registration; cycles are also taught
            # in optional training examples but are not a required new rule.
            if split == "test" and topology == "cycle":
                topology = "path"
            redundant = witness == 0 or bool(rng.random() < .4)
            pair = None
            for attempt in range(80):
                try:
                    candidate = _build_pair(rng, n, topology, witness, redundant,
                                            allow_fine=split == "train" or fine_seen)
                    key = _grid_key(candidate["input"])
                    if key in seen:
                        continue
                    pair = candidate
                    break
                except AssertionError:
                    continue
            if pair is None:
                # Finite compact fallback retains the same landmarks and rule.
                # Its n=4 path still demonstrates an indirect join.
                for _ in range(60):
                    try:
                        candidate = _build_pair(rng, n, "path", witness, redundant, True)
                        key = _grid_key(candidate["input"])
                        if key not in seen:
                            pair = candidate
                            break
                    except AssertionError:
                        continue
                if pair is None:
                    raise RuntimeError("bounded construction exhausted")
            seen.add(key)
            puzzle[split].append(pair)
            if split == "train":
                fine_seen |= _analysis(pair)["fine_geometry"]
        rng.shuffle(puzzle[split])
    return puzzle


def _grid_key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def _map_signature(cells, background):
    """Remove translation, whole-map enlargement, and literal palette roles."""
    raw = _tight(cells, background)
    quantum = 1
    for k in range(2, min(raw.shape)+1):
        if raw.shape[0] % k or raw.shape[1] % k:
            continue
        blocks = raw.reshape(raw.shape[0]//k, k, raw.shape[1]//k, k)
        if np.array_equal(blocks.min(axis=(1, 3)), blocks.max(axis=(1, 3))):
            quantum = k
    raw = raw[::quantum, ::quantum]
    colors = {background: 0, FLAG: 1}
    return tuple(tuple(colors.setdefault(int(v), len(colors)) for v in row) for row in raw)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, info = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _grid_key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            a = _analysis({"input": inp, "output": out})
            assert 3 <= a["landmarks"] <= 5 and 2 <= a["parts"] <= 6
            info[split].append(a)
    # Semantic evidence, not merely a sampling-parameter promise.
    train = info["train"]
    assert any(a["scale_witness"] for a in train), "scale not demonstrated"
    assert any(a["h_witness"] for a in train) and any(a["v_witness"] for a in train)
    relative_signs = {(sy*a["root_signs"][0], sx*a["root_signs"][1])
                      for a in train for sy, sx in a["signs"]}
    assert relative_signs >= {(-1, 1), (1, -1), (-1, -1)}, "axis reflections not separately shown"
    assert any(max(a["sides"]) > a["root_side"] for a in train), "reduction not shown"
    assert any(min(a["sides"]) < a["root_side"] for a in train), "enlargement not shown"
    assert any(a["hops"] >= 2 for a in train), "indirect registration not shown"
    assert any(a["redundant"] for a in train), "absorption not shown"
    assert all(a["hops"] >= 2 for a in info["test"]), "test shift missing"
    if any(a["fine_geometry"] for a in info["test"]):
        assert any(a["fine_geometry"] for a in train), "fine detail not demonstrated"


def self_test():
    """Bounded pixel checks, ablations, symmetries and 42-seed diversity."""
    puzzles, pairs, infos = [], [], []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pairs.append(pair)
                infos.append(_analysis(pair))
                assert np.array_equal(solve(pair["input"]), pair["output"])
        if seed < 3:
            assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len({p["input"].shape for p in pairs}) >= 25
    assert len({p["output"].shape for p in pairs}) >= 25
    assert {a["background"] for a in infos} == set(range(8))
    assert {a["root_signs"] for a in infos} == {(-1, -1), (-1, 1), (1, -1), (1, 1)}
    assert {a["root_side"] for a in infos} >= {1, 2}
    assert {a["landmarks"] for a in infos} == {3, 4, 5}
    assert len({a["parts"] for a in infos}) >= 4
    assert any(3 in a["sides"] for a in infos), "third-size calibration not generated"
    assert sum(a["fine_geometry"] for a in infos) >= 8, "sub-anchor geometry not generated"
    assert any(a["hops"] >= 3 for a in infos)
    assert any(a["branch"] for a in infos) and any(a["cycle"] for a in infos)
    assert len({_map_signature(a["union"], a["background"]) for a in infos}) >= 120
    for i, (n, witness) in enumerate(((3, 0), (4, 1), (5, None))):
        fallback = _build_pair(np.random.default_rng(930+i), n, "path", witness, True, True)
        assert np.array_equal(solve(fallback["input"]), fallback["output"])
    # Repacking and full-scene axis reflection change no registration content.
    pair = puzzles[0]["train"][0]
    data = _decode(pair["input"])
    order = [data["root"]] + [i for i in range(len(data["parts"])) if i != data["root"]]
    rasters = [_tight({p: int(data["grid"][p]) for p in data["parts"][i]["physical"]}, data["background"])
               for i in order]
    repacked = _pack(rasters, data["background"], np.random.default_rng(743))
    assert np.array_equal(solve(repacked), pair["output"])
    for axes in ((0,), (1,), (0, 1)):
        inp, out = np.flip(pair["input"], axes), np.flip(pair["output"], axes)
        assert np.array_equal(solve(inp), out)
    # Reassign all non-marker roles, preserving the two visual markers.
    mapping = np.array([3, 6, 0, 7, 1, 5, 4, 2, 8, 9], dtype=np.int8)
    assert np.array_equal(solve(mapping[pair["input"]]), mapping[pair["output"]])
    # A well-formed but unshared landmark cannot arbitrarily be placed.
    observation = next(p for p in puzzles[0]["train"]
                       if _analysis(p)["redundant"] and _analysis(p)["landmarks"] == 3)
    decoded = _decode(observation["input"])
    contained = next(p for p in decoded["parts"] if len(p["keys"]) == 1)
    old_identity = next(iter(contained["keys"]))
    new_identity = next(c for c in range(8) if c not in observation["input"])
    disconnected = observation["input"].copy()
    for pos in contained["physical"]:
        if disconnected[pos] == old_identity:
            disconnected[pos] = new_identity
    try:
        solve(disconnected)
    except AssertionError:
        pass
    else:
        raise AssertionError("disconnected landmark graph accepted")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(300+a*10+b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    out = broken["test"][0]["output"]
    out[0, 0] = FRAME
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds)")
