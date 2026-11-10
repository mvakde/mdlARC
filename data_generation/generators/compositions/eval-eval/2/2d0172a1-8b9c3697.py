"""Topological freight: canonical containment diagrams are mechanical keys.

Only NumPy and the standard library are used. No parent generator is imported.
Construction paints expected answers from abstract cargo and receiver geometry;
solve reconstructs containment, layout and docking from the visible input.
"""

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= d <= 30 for d in a.shape)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid of size 1..30")
    return a.astype(np.int8, copy=False)


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack, cells = [(int(y), int(x))], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        result.append(np.array(cells, dtype=int))
    return result


def _holes(mask):
    """Exact enclosure, flooding only the drawing's tight bounding box.

    All pixels outside this box are empty and connected to the unbounded
    exterior. Consequently every empty box-edge cell is an exterior seed;
    visiting the rest of the full canvas cannot change an enclosed region.
    This local computation preserves arbitrary closed/stepped contour behavior.
    """
    result = np.zeros_like(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return result
    y0, y1, x0, x1 = int(ys.min()), int(ys.max())+1, int(xs.min()), int(xs.max())+1
    wall = mask[y0:y1, x0:x1]
    if wall.all():
        return result
    outside = np.zeros_like(wall)
    edge = np.zeros_like(wall)
    edge[0] = edge[-1] = True
    edge[:, 0] = edge[:, -1] = True
    rows, cols = np.nonzero(edge & ~wall)
    outside[rows, cols] = True
    stack = list(zip(rows.tolist(), cols.tolist()))
    h, w = wall.shape
    while stack:
        r, c = stack.pop()
        for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= rr < h and 0 <= cc < w and not wall[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                stack.append((rr, cc))
    result[y0:y1, x0:x1] = ~wall & ~outside
    return result


def _tile(mask, leaves):
    return {"mask": mask, "leaves": leaves}


def _join(tiles, axis, offsets, gaps):
    cross = 1-axis
    low = min(offsets)
    starts, position, shape = [], 0, [0, 0]
    for i, tile in enumerate(tiles):
        start = [0, 0]
        start[axis], start[cross] = position, offsets[i]-low
        starts.append(start)
        position += tile["mask"].shape[axis]
        if i < len(gaps):
            position += gaps[i]
        shape[cross] = max(shape[cross], start[cross]+tile["mask"].shape[cross])
    shape[axis] = position
    mask, leaves = np.zeros(shape, bool), []
    for tile, (y, x) in zip(tiles, starts):
        h, w = tile["mask"].shape
        mask[y:y+h, x:x+w] |= tile["mask"]
        leaves.extend((key, r+y, c+x) for key, r, c in tile["leaves"])
    return _tile(mask, leaves)


def _frame(content, pads, outline=None):
    t, b, l, r = pads
    h, w = content["mask"].shape
    mask = np.zeros((h+t+b, w+l+r), bool) if outline is None else outline.copy()
    if outline is None:
        mask[0] = mask[-1] = True
        mask[:, 0] = mask[:, -1] = True
    mask[t:t+h, l:l+w] |= content["mask"]
    return _tile(mask, [(key, y+t, x+l) for key, y, x in content["leaves"]])


def _packet_tile(records, children, root):
    """Normalize one actual containment tree using whitespace and leaf anchors.

    Both possible slicing axes are checked; incompatible placements are invalid.
    A layout group never creates an extra frame. This supports mixed layouts and
    alignment to leaves inside descendant frames, rather than center heuristics.
    """
    cache = {}

    def group(ids):
        ids = tuple(sorted(ids))
        if ids in cache:
            return cache[ids]
        if len(ids) == 1:
            i = ids[0]
            if records[i]["hole"].any():
                assert children[i], "empty contour"
                result = _frame(group(children[i]), (2, 2, 2, 2))
            else:
                result = _tile(np.ones((1, 1), bool), [(i, 0, 0)])
        else:
            candidates = []
            for axis in (0, 1):
                partitions, end = [], -1
                for i in sorted(ids, key=lambda j: int(records[j]["lo"][axis])):
                    if int(records[i]["lo"][axis]) > end+1:
                        partitions.append([])
                    partitions[-1].append(i)
                    end = max(end, int(records[i]["hi"][axis]))
                if len(partitions) < 2:
                    continue
                try:
                    tiles = [group(part) for part in partitions]
                    edges, cross = [], 1-axis
                    for i in range(len(tiles)):
                        for j in range(i+1, len(tiles)):
                            shifts = set()
                            for a, ar, ac in tiles[i]["leaves"]:
                                for b, br, bc in tiles[j]["leaves"]:
                                    if max(records[a]["lo"][cross], records[b]["lo"][cross]) <= min(records[a]["hi"][cross], records[b]["hi"][cross]):
                                        shifts.add((ar, ac)[cross]-(br, bc)[cross])
                            assert len(shifts) <= 1, "competing leaf anchors"
                            if shifts:
                                edges.append((i, j, shifts.pop()))
                    offsets = {0: 0}
                    for _ in tiles:
                        for i, j, delta in edges:
                            if i in offsets and j in offsets:
                                assert offsets[j] == offsets[i]+delta
                            elif i in offsets:
                                offsets[j] = offsets[i]+delta
                            elif j in offsets:
                                offsets[i] = offsets[j]-delta
                    assert len(offsets) == len(tiles), "unanchored diagonal layout"
                    candidates.append(_join(tiles, axis, [offsets[i] for i in range(len(tiles))], [1]*(len(tiles)-1)))
                except AssertionError:
                    continue
            assert candidates, "no clear cardinal layout"
            result = candidates[0]
            for other in candidates[1:]:
                assert np.array_equal(result["mask"], other["mask"])
                assert sorted(result["leaves"]) == sorted(other["leaves"])
        cache[ids] = result
        return result

    return group([root])["mask"]


def _parse_fork(cells):
    parses = []
    for out in ((1, 0), (0, 1), (-1, 0), (0, -1)):
        perp = (-out[1], out[0])
        ks, qs = cells @ np.array(out), cells @ np.array(perp)
        kmin, qmin = int(ks.min()), int(qs.min())
        local = set(zip((ks-kmin).tolist(), (qs-qmin).tolist()))
        width, depth = int(qs.max()-qmin+1), int(ks.max()-kmin)
        if {q for k, q in local if k == 0} != set(range(width)):
            continue
        lengths = [sum((k, q) in local for k in range(1, depth+1)) for q in range(width)]
        expected = {(0, q) for q in range(width)}
        expected |= {(k, q) for q, length in enumerate(lengths) for k in range(1, length+1)}
        if expected != local:
            continue
        runs, q = [], 0
        while q < width:
            if lengths[q] == 0:
                q += 1
                continue
            start, length = q, lengths[q]
            while q < width and lengths[q] == length:
                q += 1
            runs.append((start, q, length))
        if (len(runs) != 3 or any(runs[i][1] >= runs[i+1][0] for i in (0, 1))
                or runs[1][0]+runs[1][1] != width):
            continue
        parses.append(dict(out=out, perp=perp, kmin=kmin, qmin=qmin,
                           width=width, runs=tuple(runs), q0=runs[1][0],
                           span=runs[1][1]-runs[1][0], stop=runs[1][2]+1,
                           cells=set(map(tuple, cells.tolist()))))
    assert len(parses) == 1, "receiver needs a unique base and three positive spokes"
    return parses[0]


def _cell(fork, k, q):
    k, q = k+fork["kmin"], q+fork["qmin"]
    return (k*fork["out"][0]+q*fork["perp"][0],
            k*fork["out"][1]+q*fork["perp"][1])


def _local_mask(mask, fork):
    cells = np.argwhere(mask)
    ks, qs = cells @ np.array(fork["out"]), cells @ np.array(fork["perp"])
    result = np.zeros((int(ks.max()-ks.min()+1), int(qs.max()-qs.min()+1)), bool)
    result[ks-ks.min(), qs-qs.min()] = True
    return result


def _analyze(input_grid):
    grid = _grid(input_grid)
    values, counts = np.unique(grid, return_counts=True)
    bg = int(values[np.argmax(counts)])
    assert int(counts.max()) > grid.size//2, "background must be a strict majority"
    assert all(np.all(edge == bg) for edge in (grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    records = []
    for color in values:
        if int(color) == bg:
            continue
        for cells in _components(grid == color):
            mask = np.zeros(grid.shape, bool)
            mask[cells[:, 0], cells[:, 1]] = True
            hole = _holes(mask)
            assert not hole.any() or len(_components(hole)) == 1
            records.append(dict(cells=cells, mask=mask, hole=hole, color=int(color),
                                lo=cells.min(axis=0), hi=cells.max(axis=0)))
    assert records
    labels = np.zeros(grid.shape, int)
    for i, obj in enumerate(records):
        labels[obj["mask"]] = i+1
    for a, b in ((labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:]),
                 (labels[:-1, :-1], labels[1:, 1:]), (labels[:-1, 1:], labels[1:, :-1])):
        assert not np.any((a > 0) & (b > 0) & (a != b)), "touching input components"
    parents, children = [], {i: [] for i in range(len(records))}
    for i, child in enumerate(records):
        possible = []
        for j, parent in enumerate(records):
            if i == j or not parent["hole"].any():
                continue
            inside = parent["hole"][child["cells"][:, 0], child["cells"][:, 1]]
            assert inside.all() or not inside.any(), "crossing components"
            if inside.all():
                possible.append(j)
        parent = min(possible, key=lambda j: int(records[j]["hole"].sum())) if possible else None
        parents.append(parent)
        if parent is not None:
            children[parent].append(i)
    packets, forks = [], []
    for i, obj in enumerate(records):
        if parents[i] is not None:
            assert obj["color"] == records[parents[i]]["color"], "one drawing color per packet"
            if not obj["hole"].any():
                assert len(obj["cells"]) >= .45*np.prod(obj["hi"]-obj["lo"]+1)
            continue
        if obj["hole"].any():
            normalized = _packet_tile(records, children, i)
            descendants, stack = [], [i]
            while stack:
                j = stack.pop()
                descendants.append(j)
                stack.extend(children[j])
            packets.append(dict(root=i, normalized=normalized, ids=descendants,
                                color=obj["color"], lo=obj["lo"], hi=obj["hi"]))
        else:
            fork = _parse_fork(obj["cells"])
            fork["color"] = obj["color"]
            forks.append(fork)
    assert packets and forks
    for i, obj in enumerate(records):
        if obj["hole"].any():
            assert children[i], "empty packet frame"
    structural = set().union(*(f["cells"] for f in forks))
    matches, used, swept = [], set(), set()
    for fi, fork in enumerate(forks):
        choices = []
        for pi, packet in enumerate(packets):
            local = _local_mask(packet["normalized"], fork)
            if local.shape[1] != fork["span"]:
                continue
            corners = np.array([(r, c) for r in (packet["lo"][0], packet["hi"][0])
                                for c in (packet["lo"][1], packet["hi"][1])])
            ks = corners @ np.array(fork["out"]) - fork["kmin"]
            qs = corners @ np.array(fork["perp"]) - fork["qmin"]
            if int(qs.min()+qs.max()) != 2*fork["q0"]+fork["span"]-1:
                continue
            far, length = int(ks.max()), local.shape[0]
            start = far-length+1
            if int(ks.min()) <= fork["stop"] or int(ks.max()-ks.min()+1) < length or start <= fork["stop"]:
                continue
            route = {_cell(fork, k, q) for k in range(fork["stop"], far+1)
                     for q in range(fork["q0"], fork["q0"]+fork["span"])}
            if any(not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]) for r, c in route):
                continue
            if route & structural:
                continue
            envelope = {_cell(fork, k, q) for k in range(fork["stop"], fork["stop"]+length)
                        for q in range(fork["q0"], fork["q0"]+fork["span"])}
            final = {_cell(fork, fork["stop"]+int(k), fork["q0"]+int(q))
                     for k, q in np.argwhere(local)}
            choices.append(dict(fi=fi, pi=pi, local=local, route=route,
                                envelope=envelope, final=final, distance=start-fork["stop"]))
        assert len(choices) <= 1, "two topologically matching packets on one axis"
        if choices:
            match = choices[0]
            assert match["pi"] not in used, "one packet matches two receivers"
            assert not swept & match["route"], "overlapping active conveyor envelopes"
            used.add(match["pi"])
            swept |= match["route"]
            matches.append(match)
    return dict(grid=grid, background=bg, records=records, parents=parents,
                children=children, packets=packets, forks=forks, matches=matches,
                unmatched=len(packets)-len(used))


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _answer(_analyze(input_grid))


def _answer(info):
    """Render an analysis obtained solely from the visible input."""
    output = info["grid"].copy()
    for packet in info["packets"]:
        for i in packet["ids"]:
            output[info["records"][i]["mask"]] = info["background"]
    for match in info["matches"]:
        fork, packet = info["forks"][match["fi"]], info["packets"][match["pi"]]
        for cell in match["route"]-match["envelope"]:
            output[cell] = fork["color"]
        for cell in match["final"]:
            output[cell] = packet["color"]
    return output


# Abstract nodes: B (blob), C (contour), and G (spatial group, not an object).
def _group(axis, *nodes):
    return nodes[0] if len(nodes) == 1 else ("G", axis, tuple(nodes))


def _scene(rng, kind=None):
    b = ("B",)
    if kind is None:
        def contents(depth):
            n = int(rng.choice([1, 2, 3, 4], p=[.35, .35, .23, .07]))
            nodes = [("C", contents(depth+1)) if depth < 3 and rng.random() < .30 else b
                     for _ in range(n)]
            rng.shuffle(nodes)
            axis = int(rng.integers(2))
            if n >= 3 and rng.random() < .75:
                cut = int(rng.integers(1, n))
                return _group(axis, _group(1-axis, *nodes[:cut]),
                              _group(1-axis, *nodes[cut:]))
            return _group(axis, *nodes)
        return ("C", contents(1))
    axis = int(rng.integers(2))
    if kind == 0:
        return ("C", b)
    if kind == 1:
        return ("C", _group(axis, *([b]*int(rng.integers(2, 4)))))
    if kind == 2:
        return ("C", _group(axis, b, _group(1-axis, b, b)))
    if kind == 3:
        return ("C", ("C", b))
    if kind == 4:
        return ("C", ("C", _group(axis, b, b)))
    if kind == 5:
        # Two axes inside one region, with a descendant alignment anchor.
        inner = ("C", b)
        line = _group(1-axis, b, inner) if rng.integers(2) else _group(1-axis, inner, b)
        return ("C", _group(axis, b, line) if rng.integers(2) else _group(axis, line, b))
    inner = ("C", _group(1-axis, b, b))
    return ("C", _group(axis, inner, b) if rng.integers(2) else _group(axis, b, inner))


def _outline(rng, h, w, clearance):
    """A single closed orthogonal loop with independently stepped side profiles."""
    cap = max(0, clearance-2)
    left = np.zeros(h, int)
    right = np.full(h, w-1, int)
    if cap and h >= 7:
        for profile, sign in ((left, 1), (right, -1)):
            a = int(rng.integers(2, max(3, h//2)))
            b = int(rng.integers(max(a+1, h//2), h-1))
            profile[a:b] += sign*int(rng.integers(0, cap+1))
    path = [(0, 0), (0, w-1)]
    path += [(r, int(right[r])) for r in range(1, h)]
    path += [(h-1, 0)]
    path += [(r, int(left[r])) for r in range(h-2, -1, -1)]
    path.append(path[0])
    mask = np.zeros((h, w), bool)
    for (r, c), (rr, cc) in zip(path, path[1:]):
        mask[min(r, rr):max(r, rr)+1, c] = True
        mask[rr, min(c, cc):max(c, cc)+1] = True
    if rng.integers(2):
        mask = np.rot90(mask, 2)
    return mask


def _draw_scene(node, rng, compact=True):
    serial = [0]

    def draw(obj):
        if obj[0] == "B":
            key = serial[0]
            serial[0] += 1
            if compact:
                h, w = (1, 1) if rng.random() < .8 else (3, 3)
            else:
                h, w = (int(rng.choice([1, 3])), int(rng.choice([1, 3])))
            mask = np.ones((h, w), bool)
            if h == w == 3 and rng.random() < .7:
                for y, x in ((0, 0), (0, 2), (2, 0), (2, 2)):
                    if rng.random() < .5:
                        mask[y, x] = False
            return (_tile(mask, [(key, h//2, w//2)]),
                    _tile(np.ones((1, 1), bool), [(key, 0, 0)]))
        if obj[0] == "G":
            pairs = [draw(child) for child in obj[2]]
            axis, cross = obj[1], 1-obj[1]
            anchors = [int(rng.integers(len(pair[0]["leaves"]))) for pair in pairs]
            versions = []
            for v in (0, 1):
                tiles = [pair[v] for pair in pairs]
                offsets = [-tile["leaves"][anchor][cross+1] for tile, anchor in zip(tiles, anchors)]
                gaps = [int(rng.choice([1, 3])) if v == 0 else 1 for _ in tiles[:-1]]
                versions.append(_join(tiles, axis, offsets, gaps))
            return tuple(versions)
        metric, canonical = draw(obj[1])
        py = int(rng.choice([2, 3], p=[.7, .3])) if compact else int(rng.integers(2, 5))
        px = int(rng.choice([2, 3], p=[.7, .3])) if compact else int(rng.integers(2, 5))
        h, w = metric["mask"].shape
        # Side distortions stay inside safe corridors and retain one real hole.
        outline = _outline(rng, h+2*py, w+2*px, min(py, px))
        return (_frame(metric, (py, py, px, px), outline),
                _frame(canonical, (2, 2, 2, 2)))

    metric, canonical = draw(node)
    return metric["mask"], canonical["mask"]


def _fork_mask(rng, span, long_center=False, small=False):
    side = 1 if small else int(rng.integers(1, 3))
    gap = 1 if small else int(rng.integers(1, 3))
    width, q0 = span+2*(side+gap), side+gap
    if long_center:
        center = int(rng.integers(3, 5))
        left, right = int(rng.integers(1, center)), int(rng.integers(1, center))
    elif small:
        center = int(rng.integers(1, 3))
        left, right = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    else:
        left, center, right = map(int, rng.integers(1, 6, 3))
    mask = np.zeros((max(left, center, right)+1, width), bool)
    mask[0] = True
    mask[1:left+1, :side] = True
    mask[1:center+1, q0:q0+span] = True
    mask[1:right+1, -side:] = True
    return mask, center+1, q0


def _module(rng, bg, fork_color, cargo_color, kind=None, long_center=False,
            witness=False, active=True, compact=True):
    if witness:
        count = int(rng.integers(2, 4))
        span = 3+2*count
        raw_h, raw_w = int(rng.choice([7, 9])), span+int(rng.choice([4, 6]))
        active_raw = _outline(rng, raw_h, raw_w, 3)
        wrong_raw = active_raw.copy()
        center = raw_w//2
        for q in range(center-(count-1)*2, center+(count-1)*2+1, 4):
            active_raw[raw_h//2, q] = True
        wrong_raw[raw_h//2, center] = True
        norm = np.zeros((5, span), bool)
        norm[0] = norm[-1] = True
        norm[:, 0] = norm[:, -1] = True
        norm[2, 2:-2:2] = True
        fork, stop, q0 = _fork_mask(rng, span, small=True)
        near = fork.shape[0]+2
        top = near+raw_h+int(rng.integers(2, 4))
        cargos = [(active_raw, top), (wrong_raw, near)]
    else:
        raw, norm = _draw_scene(_scene(rng, kind), rng, compact)
        if raw.shape[1] % 2:
            span = norm.shape[1] if active else norm.shape[1]+2
        else:
            raise AssertionError("half-cell packet axis")
        fork, stop, q0 = _fork_mask(rng, span, long_center, small=compact)
        top = fork.shape[0]+int(rng.integers(2, 5))
        cargos = [(raw, top)]
    low = min(0, *(q0+(span-m.shape[1])//2 for m, _ in cargos))
    high = max(fork.shape[1], *(q0+(span-m.shape[1])//2+m.shape[1] for m, _ in cargos))
    height = max(top+m.shape[0] for m, top in cargos)
    grid = np.full((height, high-low), bg, np.int8)
    output = grid.copy()
    grid[:fork.shape[0], -low:fork.shape[1]-low][fork] = fork_color
    output[:fork.shape[0], -low:fork.shape[1]-low][fork] = fork_color
    for raw, top in cargos:
        col = q0+(span-raw.shape[1])//2-low
        grid[top:top+raw.shape[0], col:col+raw.shape[1]][raw] = cargo_color
    if active:
        far = cargos[0][1]+cargos[0][0].shape[0]-1
        assert far-norm.shape[0]+1 > stop
        output[stop+norm.shape[0]:far+1, q0-low:q0-low+span] = fork_color
        output[stop:stop+norm.shape[0], q0-low:q0-low+span][norm] = cargo_color
    return grid, output


def _add_decoys(rng, grid, bg, cargo_color, limit):
    grid = grid.copy()
    for _ in range(limit):
        options = []
        for size in (5, 7):
            for r in range(1, grid.shape[0]-size):
                for c in range(1, grid.shape[1]-size):
                    if np.all(grid[r-1:r+size+1, c-1:c+size+1] == bg):
                        options.append((r, c, size))
        if not options:
            break
        unmatched = None
        for index in rng.permutation(len(options))[:30]:
            r, c, size = options[int(index)]
            candidate = grid.copy()
            patch = candidate[r:r+size, c:c+size]
            patch[0] = patch[-1] = cargo_color
            patch[:, 0] = patch[:, -1] = cargo_color
            patch[size//2, size//2] = cargo_color
            try:
                info = _analyze(candidate)
                if unmatched is None:
                    unmatched = _analyze(grid)["unmatched"]
                if info["unmatched"] > unmatched:
                    grid = candidate
                    break
            except AssertionError:
                continue
    return grid


def _make_pair(rng, style, orientation):
    for attempt in range(160):
        palette = list(map(int, rng.choice(10, 5, replace=False)))
        bg, fc, cc, fc2, cc2 = palette
        compact = attempt >= 5 or style in ("contrast", "mixed", "test")
        try:
            if style == "contrast":
                modules = [_module(rng, bg, fc, cc, witness=True)]
            elif style == "mixed":
                modules = [_module(rng, bg, fc, cc, kind=5, long_center=True, compact=True)]
            elif style == "test":
                kind = None if rng.random() < .55 else int(rng.choice([4, 6]))
                modules = [_module(rng, bg, fc, cc, kind=kind, compact=True),
                           _module(rng, bg, fc2, cc2, kind=int(rng.choice([0, 1, 2])), compact=True)]
            else:
                modules = [_module(rng, bg, fc, cc, compact=compact)]
                if rng.random() < .45:
                    modules.append(_module(rng, bg, fc2, cc2, kind=int(rng.choice([0, 1, 2])),
                                           active=bool(rng.integers(2)), compact=True))
            if len(modules) == 2:
                # Mix directions when space allows; routes are reserved in tiles.
                turn = int(rng.choice([0, 0, 1, 2, 3]))
                modules[1] = tuple(np.rot90(a, turn) for a in modules[1])
                axis = int(rng.integers(2))
                cross = 1-axis
                gap = int(rng.integers(2, 5))
                shape = [0, 0]
                shape[axis] = sum(m[0].shape[axis] for m in modules)+gap
                shape[cross] = max(m[0].shape[cross] for m in modules)
                if max(shape) > 28:
                    continue
                arrays = [np.full(shape, bg, np.int8) for _ in range(2)]
                start = 0
                for module in modules:
                    offset = int(rng.integers(shape[cross]-module[0].shape[cross]+1))
                    pos = [0, 0]
                    pos[axis], pos[cross] = start, offset
                    y, x = pos
                    h, w = module[0].shape
                    for v in (0, 1):
                        arrays[v][y:y+h, x:x+w] = module[v]
                    start += module[0].shape[axis]+gap
                grid, output = arrays
            else:
                grid, output = modules[0]
            if max(grid.shape) > 28:
                continue
            grid, output = np.rot90(grid, orientation), np.rot90(output, orientation)
            # Variable margins, placement and aspect ratios within both splits.
            height = int(rng.integers(grid.shape[0]+2, 31))
            width = int(rng.integers(grid.shape[1]+2, 31))
            y = int(rng.integers(1, height-grid.shape[0]))
            x = int(rng.integers(1, width-grid.shape[1]))
            inp, out = np.full((height, width), bg, np.int8), np.full((height, width), bg, np.int8)
            inp[y:y+grid.shape[0], x:x+grid.shape[1]] = grid
            out[y:y+output.shape[0], x:x+output.shape[1]] = output
            inp = _add_decoys(rng, inp, bg, cc, int(rng.integers(0, 3)))
            info = _analyze(inp)
            assert np.array_equal(_answer(info), out), "constructed geometry must be visibly recoverable"
            assert info["matches"]
            if style == "contrast":
                assert _features(info)["same_box_wrong_key"]
            if style == "mixed":
                assert _features(info)["mixed"] and _features(info)["depth"] >= 2
            if style == "test":
                assert len(info["matches"]) == len(info["forks"]) == 2
                assert _features(info)["depth"] >= 2
            return dict(input=inp, output=out)
        except (AssertionError, ValueError):
            continue
    raise RuntimeError("bounded packing search exhausted")


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(name+" must be an integer in its legal range")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    orientation = int(rng.integers(4))
    puzzle, seen = dict(train=[], test=[]), set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            if split == "test":
                style, turn = "test", (orientation+int(rng.choice([2, 3]))) % 4
            elif i == 0:
                style, turn = "contrast", orientation
            elif i == 1:
                style, turn = "mixed", (orientation+1) % 4
            else:
                style, turn = "random", int(rng.integers(4))
            for _ in range(24):
                pair = _make_pair(rng, style, turn)
                if _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input search exhausted")
        rng.shuffle(puzzle[split])
    return puzzle


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _features(info):
    depth, mixed, same_box = 0, False, False
    topologies, signatures, raw_shapes = [], [], []

    def tree(i):
        nonlocal depth
        if not info["records"][i]["hole"].any():
            return "B"
        d, parent = 1, info["parents"][i]
        while parent is not None:
            d += 1
            parent = info["parents"][parent]
        depth = max(depth, d)
        ids = info["children"][i]
        return ("C", tuple(sorted((tree(j) for j in ids), key=repr)))

    for packet in info["packets"]:
        topologies.append(repr(tree(packet["root"])))
        signatures.append(_key(packet["normalized"]))
        raw_shapes.append(tuple((packet["hi"]-packet["lo"]+1).tolist()))
        for i in packet["ids"]:
            ids = info["children"][i]
            if len(ids) >= 3:
                centers = np.array([info["records"][j]["lo"]+info["records"][j]["hi"] for j in ids])
                mixed |= bool(np.ptp(centers[:, 0]) and np.ptp(centers[:, 1]))
    for match in info["matches"]:
        fork, target = info["forks"][match["fi"]], info["packets"][match["pi"]]
        for other in info["packets"]:
            if other is target or not np.array_equal(other["hi"]-other["lo"], target["hi"]-target["lo"]):
                continue
            center = other["lo"]+other["hi"]
            axial = int(center @ np.array(fork["perp"])-2*fork["qmin"])
            def axial_limits(packet):
                corners = np.array([(r, c) for r in (packet["lo"][0], packet["hi"][0])
                                    for c in (packet["lo"][1], packet["hi"][1])])
                ks = corners @ np.array(fork["out"])-fork["kmin"]
                return int(ks.min()), int(ks.max())
            nearer = fork["stop"] < axial_limits(other)[0] and axial_limits(other)[1] < axial_limits(target)[0]
            if (nearer and axial == 2*fork["q0"]+fork["span"]-1
                    and _local_mask(other["normalized"], fork).shape[1] != fork["span"]):
                same_box = True
    return dict(depth=depth, mixed=mixed, same_box_wrong_key=same_box,
                topologies=tuple(topologies), signatures=tuple(signatures), raw_shapes=tuple(raw_shapes))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _analyze(inp)
            assert info["matches"] and not np.array_equal(inp, out)
            assert np.array_equal(_answer(info), out)
            changed_allowed = set()
            for packet in info["packets"]:
                for i in packet["ids"]:
                    changed_allowed |= set(map(tuple, info["records"][i]["cells"].tolist()))
            for fork in info["forks"]:
                assert all(int(out[p]) == fork["color"] for p in fork["cells"])
                assert all(length >= 1 for _, _, length in fork["runs"])
            for match in info["matches"]:
                changed_allowed |= match["route"]
                fork, packet = info["forks"][match["fi"]], info["packets"][match["pi"]]
                # No trail in canonical holes, central-tip contact, exact width.
                assert all(int(out[p]) == packet["color"] for p in match["final"])
                assert all(int(out[p]) == info["background"] for p in match["envelope"]-match["final"])
                assert all(int(out[p]) == fork["color"] for p in match["route"]-match["envelope"])
                leading = {_cell(fork, fork["stop"], q) for q in range(fork["q0"], fork["q0"]+fork["span"])}
                assert leading <= match["final"]
                assert match["distance"] > 0
            assert set(map(tuple, np.argwhere(inp != out).tolist())) <= changed_allowed
            if split == "train":
                training.append((_features(info), info))
            else:
                assert len(info["forks"]) == len(info["matches"]) == 2
                assert _features(info)["depth"] >= 2
    assert any(f["same_box_wrong_key"] for f, _ in training)
    assert any(f["mixed"] and f["depth"] >= 2 for f, _ in training)
    assert any(fork["runs"][1][2] > max(fork["runs"][0][2], fork["runs"][2][2])
               for _, info in training for fork in info["forks"])


def self_test():
    """Bounded semantic, metamorphic, API and variation checks on 42 seeds."""
    # Hand-specified equal-box regression, independent of the scene renderer.
    # Two 7x11 boxes on one axis: nearer singleton key 5, farther two-point key 7.
    fixture = np.zeros((26, 15), np.int8)
    fixture[1, 1:12] = 1
    fixture[2:5, 1] = fixture[2:5, 11] = 1
    fixture[2, 3:10] = 1
    for top in (7, 16):
        fixture[top, 1:12] = fixture[top+6, 1:12] = 2
        fixture[top:top+7, 1] = fixture[top:top+7, 11] = 2
    fixture[10, 6] = 2
    fixture[19, 4] = fixture[19, 8] = 2
    expected = np.zeros_like(fixture)
    expected[fixture == 1] = 1
    expected[3, 3:10] = expected[7, 3:10] = 2
    expected[3:8, 3] = expected[3:8, 9] = 2
    expected[5, 5] = expected[5, 7] = 2
    expected[8:23, 3:10] = 1
    assert np.array_equal(solve(fixture), expected), "manual topology-key/docking regression"
    # Hold the outline and blob count fixed but rotate the sibling arrangement.
    vertical = fixture.copy()
    vertical[19, 4] = vertical[19, 8] = 0
    vertical[18, 6] = vertical[20, 6] = 2
    receivers_only = np.where(fixture == 1, 1, 0).astype(np.int8)
    assert np.array_equal(solve(vertical), receivers_only), "arrangement must change the mechanical key"
    assert not np.array_equal(receivers_only, expected), "omitting A changes the answer"
    normalized_in_place = np.where(fixture == 1, 1, 0).astype(np.int8)
    # A-only abstraction retains both diagrams away from the central tip.
    for top, width in ((8, 5), (17, 7)):
        left = 6-width//2
        normalized_in_place[top, left:left+width] = normalized_in_place[top+4, left:left+width] = 2
        normalized_in_place[top:top+5, left] = normalized_in_place[top:top+5, left+width-1] = 2
        if width == 5:
            normalized_in_place[top+2, 6] = 2
        else:
            normalized_in_place[top+2, left+2] = normalized_in_place[top+2, left+4] = 2
    assert not np.array_equal(normalized_in_place, expected), "omitting B changes the answer"
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, diagrams, topologies, geometry, directions, backgrounds = set(), set(), set(), set(), set(), set()
    metric_realizations, packets_by_box = {}, {}
    multi, inactive, distorted = 0, 0, 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in puzzle:
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                features = _features(info)
                shapes.add(pair["input"].shape)
                backgrounds.add(info["background"])
                diagrams.update(features["signatures"])
                topologies.update(features["topologies"])
                multi += len(info["forks"]) > 1
                inactive += len(info["forks"]) > len(info["matches"])
                for fork in info["forks"]:
                    geometry.add((fork["width"], fork["runs"]))
                    directions.add(fork["out"])
                for packet, signature, shape in zip(info["packets"], features["signatures"], features["raw_shapes"]):
                    metric_realizations.setdefault(signature, set()).add(shape)
                    packets_by_box.setdefault(shape, set()).add(signature)
                for obj in info["records"]:
                    if obj["hole"].any():
                        y, x = obj["lo"]; yy, xx = obj["hi"]
                        local = obj["mask"][y:yy+1, x:xx+1]
                        distorted += not (local[0].all() and local[-1].all() and local[:, 0].all() and local[:, -1].all())
        # Input-only rotation, reflection, color permutation and translation.
        pair = puzzle["test"][0]
        inp, out = pair["input"], pair["output"]
        permutation = np.random.default_rng(701+len(shapes)).permutation(10)
        assert np.array_equal(solve(permutation[inp]), permutation[out])
        assert np.array_equal(solve(np.rot90(inp)), np.rot90(out))
        assert np.array_equal(solve(inp[:, ::-1]), out[:, ::-1])
        if max(inp.shape) <= 28:
            bg = _analyze(inp)["background"]
            assert np.array_equal(solve(np.pad(inp, 1, constant_values=bg)), np.pad(out, 1, constant_values=bg))
    assert len(shapes) >= 40 and len(diagrams) >= 15 and len(topologies) >= 10, (len(shapes), len(diagrams), len(topologies))
    assert len(geometry) >= 40 and len(directions) == 4 and len(backgrounds) >= 8, (len(geometry), directions, backgrounds)
    assert multi >= 42 and inactive >= 1 and distorted >= 15, (multi, inactive, distorted)
    assert sum(len(v) >= 2 for v in metric_realizations.values()) >= 8
    assert sum(len(v) >= 2 for v in packets_by_box.values()) >= 8
    assert _puzzle_key(generate_puzzle(0)) == _puzzle_key(puzzles[0])
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    # All count combinations, with minimum demonstrations retained at 2/1.
    for n in range(2, 11):
        for m in range(1, min(4, n-1)+1):
            puzzle = generate_puzzle(500+n*10+m, n, m)
            assert (len(puzzle["train"]), len(puzzle["test"])) == (n, m)
            validate_generated_puzzle(puzzle)
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Corrupt a central contact cell; schema alone would accept this answer.
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    info = _analyze(bad["test"][0]["input"])
    cell = min(info["matches"][0]["final"])
    bad["test"][0]["output"][cell] = info["background"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("damaged canonical cargo accepted")
    return dict(seeds=42, pairs=sum(len(p[s]) for p in puzzles for s in p),
                input_shapes=len(shapes), canonical_diagrams=len(diagrams),
                containment_trees=len(topologies), receiver_geometries=len(geometry),
                multi_receiver_pairs=multi, inactive_receiver_pairs=inactive,
                distorted_contours=distorted,
                diagrams_with_multiple_metric_boxes=sum(len(v) >= 2 for v in metric_realizations.values()),
                boxes_with_different_diagrams=sum(len(v) >= 2 for v in packets_by_box.values()))


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s]) for s in ("train", "test"))
