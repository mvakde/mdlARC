"""Registered constellation jigsaw: pixel-derived placement and scaled rendering.

Only standard-library modules and NumPy are used. No parent generator, stored
example, seed-dependent solver convention, or construction state is consulted.
"""

from itertools import combinations
import numpy as np


# Fixed direction order: up, right, down, left. Opposites differ by two.
_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _connected(points):
    points = set(points)
    if not points:
        return False
    seen = {min(points)}
    stack = list(seen)
    while stack:
        r, c = stack.pop()
        for dr, dc in _DIRS:
            q = r + dr, c + dc
            if q in points and q not in seen:
                seen.add(q)
                stack.append(q)
    return seen == points


def _signatures(mask):
    return tuple(tuple(map(int, np.flatnonzero(edge))) for edge in
                 (mask[0], mask[:, -1], mask[-1], mask[:, 0]))


def _gap(a, b):
    """Rectangles have exclusive bottom/right; leave a full background band."""
    return a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]


def _parse_gallery(a, bg, divider):
    coords = np.argwhere(a == divider)
    gt, gl = map(int, coords.min(axis=0))
    gb, gr = map(int, coords.max(axis=0) + 1)
    box = a[gt:gb, gl:gr]
    rows = list(map(int, np.flatnonzero(np.all(box == divider, axis=1))))
    cols = list(map(int, np.flatnonzero(np.all(box == divider, axis=0))))
    assert len(rows) >= 2 and len(cols) >= 2
    assert rows[0] == cols[0] == 0
    assert rows[-1] == box.shape[0] - 1 and cols[-1] == box.shape[1] - 1
    hh, ww = np.diff(rows), np.diff(cols)
    assert len(set(hh)) == len(set(ww)) == 1
    h, w = int(hh[0] - 1), int(ww[0] - 1)
    assert 3 <= h <= 8 and 3 <= w <= 8
    frame = np.zeros(box.shape, dtype=bool)
    frame[rows, :] = True
    frame[:, cols] = True
    assert np.array_equal(box == divider, frame)
    tiles, positions, bays = [], [], []
    for ri, r in enumerate(rows[:-1]):
        for ci, c in enumerate(cols[:-1]):
            tile = box[r+1:r+1+h, c+1:c+1+w].copy()
            if np.all(tile == bg):
                continue
            mask = tile != bg
            assert not any(mask[p] for p in ((0, 0), (0, -1), (-1, 0), (-1, -1)))
            assert _connected(tuple(map(tuple, np.argwhere(mask))))
            tiles.append(tile)
            positions.append((gt+r+1, gl+c+1))
            bays.append((ri, ci))
    assert 2 <= len(tiles) <= 8
    inside_colors, inside_counts = np.unique(
        np.concatenate([t[t != bg] for t in tiles]), return_counts=True)
    assert len(inside_colors) == 2 and np.count_nonzero(inside_counts == 1) == 1
    marker = int(inside_colors[np.argmin(inside_counts)])
    ink = int(inside_colors[np.argmax(inside_counts)])
    registered = next(i for i, tile in enumerate(tiles) if np.any(tile == marker))
    accent = tuple(map(int, np.argwhere(tiles[registered] == marker)[0]))
    assert 0 < accent[0] < h-1 and 0 < accent[1] < w-1
    outside = a.copy()
    outside[gt:gb, gl:gr] = bg
    point_coords = [tuple(map(int, p)) for p in np.argwhere(outside != bg)]
    assert len(point_coords) == len(tiles)
    assert all(int(outside[p]) in (ink, marker) for p in point_coords)
    anchors = [p for p in point_coords if outside[p] == ink]
    assert len(anchors) == 1
    pr, pc = map(int, np.min(point_coords, axis=0))
    er, ec = map(int, np.max(point_coords, axis=0))
    assert all((r-pr) % 2 == (c-pc) % 2 == 0 for r, c in point_coords)
    points = tuple(sorted(((r-pr)//2, (c-pc)//2) for r, c in point_coords))
    assert _connected(points)
    mh, mw = (er-pr)//2+1, (ec-pc)//2+1
    assert _gap((gt, gl, gb, gr), (pr, pc, er+1, ec+1))
    anchor = (anchors[0][0]-pr)//2, (anchors[0][1]-pc)//2
    tr, tc = positions[registered]
    origin = tr-anchor[0]*h, tc-anchor[1]*w
    assert origin[0] >= 0 and origin[1] >= 0
    assert origin[0]+mh*h <= a.shape[0] and origin[1]+mw*w <= a.shape[1]
    return dict(shape=a.shape, roles=(bg, divider, ink, marker), tiles=tiles,
                masks=[t != bg for t in tiles], positions=positions, bays=bays,
                gallery_shape=(len(rows)-1, len(cols)-1), tile_shape=(h, w),
                gallery_rect=(gt, gl, gb, gr), map_points=point_coords,
                points=points, map_shape=(mh, mw), anchor=anchor,
                registered=registered, origin=origin,
                ports=[_signatures(t != bg) for t in tiles])


def _decode(value):
    a = _grid(value)
    colors, counts = np.unique(a, return_counts=True)
    assert len(colors) == 4, "four semantic input colors required"
    assert counts.max() > sorted(counts)[-2], "background must be uniquely most frequent"
    bg = int(colors[np.argmax(counts)])
    candidates = []
    for divider in colors:
        if divider == bg:
            continue
        try:
            candidates.append(_parse_gallery(a, bg, int(divider)))
        except (AssertionError, StopIteration, ValueError, IndexError):
            continue
    assert len(candidates) == 1, "one valid closed gallery and constellation required"
    return candidates[0]


def _placements(info):
    """At most two complete all-piece assignments; exact signatures on all sides."""
    points, ports = info["points"], info["ports"]
    n = len(points)
    index = {p: i for i, p in enumerate(points)}
    neighbors = [tuple(index.get((r+dr, c+dc), -1) for dr, dc in _DIRS)
                 for r, c in points]
    anchor_slot = index[info["anchor"]]
    registered = info["registered"]
    domains = []
    for slot, adjacent in enumerate(neighbors):
        domains.append(tuple(tile for tile in range(n)
                             if (tile == registered) == (slot == anchor_slot)
                             and all(bool(ports[tile][d]) == (other >= 0)
                                     for d, other in enumerate(adjacent))))
    assigned, used, answers = [-1]*n, set(), []

    def available(slot):
        return [tile for tile in domains[slot] if tile not in used
                and all(other < 0 or assigned[other] < 0
                        or ports[tile][d] == ports[assigned[other]][(d+2) % 4]
                        for d, other in enumerate(neighbors[slot]))]

    def visit():
        if len(answers) >= 2:
            return
        pending = [i for i in range(n) if assigned[i] < 0]
        if not pending:
            answers.append(tuple(assigned))
            return
        options = [(available(i), i) for i in pending]
        choices, slot = min(options, key=lambda item: (len(item[0]), item[1]))
        for tile in choices:
            assigned[slot] = tile
            used.add(tile)
            visit()
            used.remove(tile)
            assigned[slot] = -1

    visit()
    return answers


def _render(info, assignment, inversion=True):
    bg, _, ink, marker = info["roles"]
    h, w = info["tile_shape"]
    top, left = info["origin"]
    out = np.full(info["shape"], bg, dtype=np.int8)
    for point, tile in zip(info["points"], assignment):
        r, c = point
        mask = info["masks"][tile]
        color = marker if inversion and point == info["anchor"] else ink
        out[top+r*h:top+(r+1)*h, left+c*w:left+(c+1)*w][mask] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _decode(input_grid)
    answers = _placements(info)
    if len(answers) != 1:
        raise ValueError("input must have exactly one registered port-matching tiling")
    return _render(info, answers[0])


def _kind(signature):
    if len(signature) == 1:
        return "single"
    return "thick" if signature[-1]-signature[0]+1 == len(signature) else "separated"


def _topology(points):
    points = set(points)
    degrees = [sum((r+dr, c+dc) in points for dr, dc in _DIRS) for r, c in points]
    return max(degrees), sum(degrees)//2-len(points)+1


def _reading_assignment(info):
    anchor_slot = info["points"].index(info["anchor"])
    remaining = iter(i for i in range(len(info["tiles"])) if i != info["registered"])
    return tuple(info["registered"] if i == anchor_slot else next(remaining)
                 for i in range(len(info["points"])))


def _position_blind_placements(info):
    """Counterfactual: retain open sides but discard transverse port positions."""
    reduced = dict(info)
    reduced["ports"] = [tuple((1,) if s else () for s in p) for p in info["ports"]]
    return _placements(reduced)


def _analyze_pair(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    assert inp.shape == out.shape
    info = _decode(inp)
    answers = _placements(info)
    assert len(answers) == 1, "ambiguous or impossible graph assembly"
    assignment = answers[0]
    assert np.array_equal(_render(info, assignment), out), "incorrect reconstruction"
    bg, divider, ink, marker = info["roles"]
    assert divider not in np.unique(out), "source frames must disappear"
    assert _connected(tuple(map(tuple, np.argwhere(out != bg)))), "expanded graph must connect"
    assert np.count_nonzero(out != bg) == sum(int(m.sum()) for m in info["masks"])
    assert np.count_nonzero(out == marker) == int(info["masks"][info["registered"]].sum())
    assert not np.array_equal(inp, out)
    # Check each output macro-cell, its blank cells, and its four adjacency ports.
    h, w = info["tile_shape"]
    top, left = info["origin"]
    lookup = dict(zip(info["points"], assignment))
    for r in range(info["map_shape"][0]):
        for c in range(info["map_shape"][1]):
            block = out[top+r*h:top+(r+1)*h, left+c*w:left+(c+1)*w]
            if (r, c) not in lookup:
                assert np.all(block == bg)
                continue
            tile = lookup[r, c]
            assert np.array_equal(block != bg, info["masks"][tile])
            for d, (dr, dc) in enumerate(_DIRS):
                other = lookup.get((r+dr, c+dc))
                if other is None:
                    assert not info["ports"][tile][d]
                else:
                    assert info["ports"][tile][d] == info["ports"][other][(d+2) % 4]
    blind = _position_blind_placements(info)
    info.update(assignment=assignment, n=len(assignment),
                kinds={_kind(s) for p in info["ports"] for s in p if s},
                topology=_topology(info["points"]),
                reading_fails=not np.array_equal(_render(info, _reading_assignment(info)), out),
                repetition_fails=not np.array_equal(
                    _render(info, (info["registered"],)*len(assignment)), out),
                positions_needed=len(blind) > 1 and any(
                    not np.array_equal(_render(info, alternative), out) for alternative in blind))
    assert not np.array_equal(_render(info, assignment, inversion=False), out)
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, groups = set(), {}
    for split in ("train", "test"):
        groups[split] = []
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            key = _key(_grid(pair["input"]))
            assert key not in seen, "duplicate input"
            seen.add(key)
            groups[split].append(_analyze_pair(pair))
    train, test = groups["train"], groups["test"]
    teaching = [m for m in train if m["reading_fails"] and m["repetition_fails"]
                and min(m["map_shape"]) >= 2]
    assert len(teaching) >= 2, "two demonstrations must defeat both parent shortcuts"
    assert any(m["positions_needed"] for m in train), "demonstrate exact contact positions"
    assert len({m["tile_shape"][0] for m in train}) >= 2
    assert len({m["tile_shape"][1] for m in train}) >= 2
    assert len({m["anchor"][0] for m in train}) >= 2
    assert len({m["anchor"][1] for m in train}) >= 2
    assert any(m["anchor"] != (0, 0) for m in train)
    taught = set().union(*(m["kinds"] for m in train))
    assert set().union(*(m["kinds"] for m in test)) <= taught
    assert max(m["n"] for m in train) < min(m["n"] for m in test)
    if any(m["topology"][0] >= 3 for m in test):
        assert any(m["topology"][0] >= 3 for m in train)
    if any(m["topology"][1] > 0 for m in test):
        assert any(m["topology"][1] > 0 for m in train)
    return None


def _counts(num_train, num_test):
    for name, count, lo, hi in (("num_train", num_train, 2, 10),
                               ("num_test", num_test, 1, 4)):
        if (isinstance(count, (bool, np.bool_))
                or not isinstance(count, (int, np.integer)) or not lo <= count <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")


def _normalize(points):
    r0, c0 = np.min(list(points), axis=0)
    return tuple(sorted((int(r-r0), int(c-c0)) for r, c in points))


def _sample_map(rng, n, regime, required, two_axes):
    for _ in range(160):
        cells = {(0, 0)}
        while len(cells) < n:
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _DIRS} - cells)
            if regime in ("path", "tree"):
                frontier = [p for p in frontier if sum((p[0]+dr, p[1]+dc) in cells
                                                       for dr, dc in _DIRS) == 1]
            if regime == "path":
                frontier = [p for p in frontier if all(
                    sum((r+dr, c+dc) in cells for dr, dc in _DIRS) < 2
                    for r, c in cells if abs(r-p[0])+abs(c-p[1]) == 1)]
            if not frontier:
                break
            cells.add(_pick(rng, frontier))
        if len(cells) != n:
            continue
        points = _normalize(cells)
        mh, mw = np.max(points, axis=0) + 1
        degree, cycles = _topology(points)
        if max(mh, mw) > 7 or (two_axes and min(mh, mw) == 1):
            continue
        if required == "branch" and degree < 3:
            continue
        if required == "cycle" and cycles == 0:
            continue
        return points
    # Finite connected witnesses, not a bank used for normal generation.
    if regime == "path":
        base = ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (2, 3), (3, 3), (4, 3))
    elif required == "cycle" or regime == "general":
        base = ((0, 0), (0, 1), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3))
    else:
        base = ((0, 1), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4))
    points = base[:n]
    if rng.integers(2):
        points = tuple((c, r) for r, c in points)
    if rng.integers(2):
        points = tuple((-r, c) for r, c in points)
    return _normalize(points)


def _pool(size, capability):
    return [s for k in range(1, min(3, size-2)+1)
            for s in combinations(range(1, size-1), k)
            if capability == 2 or k == 1 or (capability == 1 and _kind(s) == "thick")]


def _edge_ports(rng, points, dims, capability, required):
    index = {p: i for i, p in enumerate(points)}
    edges = [(i, index[r+dr, c+dc], d) for i, (r, c) in enumerate(points)
             for d, (dr, dc) in enumerate(_DIRS) if d in (1, 2) and (r+dr, c+dc) in index]
    ports = [[()] * 4 for _ in points]
    forced = {}
    indices = list(map(int, rng.permutation(len(edges))))
    for kind, edge in zip(sorted(required), indices):
        forced[edge] = kind
    used = {1: set(), 2: set()}
    for e, (i, j, d) in enumerate(edges):
        pool = _pool(dims[0 if d == 1 else 1], capability)
        if e in forced:
            pool = [s for s in pool if _kind(s) == forced[e]]
        fresh = [s for s in pool if s not in used[d]]
        signature = _pick(rng, fresh if fresh and rng.random() < .85 else pool)
        used[d].add(signature)
        ports[i][d] = ports[j][(d+2) % 4] = signature
    return [tuple(p) for p in ports]


def _mask(rng, dims, ports):
    h, w = dims
    boundary = ([(0, c) for c in ports[0]] + [(r, w-1) for r in ports[1]]
                + [(h-1, c) for c in ports[2]] + [(r, 0) for r in ports[3]])
    roots = [(min(h-2, max(1, r)), min(w-2, max(1, c))) for r, c in boundary]
    hub = int(rng.integers(1, h-1)), int(rng.integers(1, w-1))
    body = {hub}
    for r, c in roots:
        body.add((r, c))
        while (r, c) != hub:
            axes = [axis for axis in (0, 1) if (r, c)[axis] != hub[axis]]
            if _pick(rng, axes) == 0:
                r += 1 if r < hub[0] else -1
            else:
                c += 1 if c < hub[1] else -1
            body.add((r, c))
    target = max(len(body), int(round((h-2)*(w-2)*_pick(rng, (.2, .4, .6, .8, 1.0)))))
    while len(body) < target:
        frontier = sorted({(r+dr, c+dc) for r, c in body for dr, dc in _DIRS
                           if 0 < r+dr < h-1 and 0 < c+dc < w-1} - body)
        body.add(_pick(rng, frontier))
    mask = np.zeros(dims, dtype=bool)
    for point in body | set(boundary):
        mask[point] = True
    assert _signatures(mask) == ports
    return mask


def _gallery_options(n, dims, map_shape, anchor):
    h, w = dims
    oh, ow = map_shape[0]*h, map_shape[1]*w
    options = []
    for nr in range(1, 7):
        for nc in range(1, 7):
            if not n <= nr*nc <= n+2:
                continue
            gh, gw = nr*(h+1)+1, nc*(w+1)+1
            if max(gh, gw) > 30:
                continue
            for sr in range(nr):
                for sc in range(nc):
                    gt, gl = anchor[0]*h-1-sr*(h+1), anchor[1]*w-1-sc*(w+1)
                    if max(oh, gt+gh)-min(0, gt) <= 30 and max(ow, gl+gw)-min(0, gl) <= 30:
                        options.append((nr, nc, sr, sc, gt, gl, gh, gw))
    return options


def _map_positions(option, dims, map_shape):
    _, _, _, _, gt, gl, gh, gw = option
    oh, ow = dims[0]*map_shape[0], dims[1]*map_shape[1]
    ph, pw = 2*map_shape[0]-1, 2*map_shape[1]-1
    base_top, base_left = min(0, gt), min(0, gl)
    base_bottom, base_right = max(oh, gt+gh), max(ow, gl+gw)
    rr, cc = np.meshgrid(np.arange(base_bottom-30, base_top+31-ph),
                         np.arange(base_right-30, base_left+31-pw), indexing="ij")
    valid = (rr+ph < gt) | (gt+gh < rr) | (cc+pw < gl) | (gl+gw < cc)
    minr, minc = np.minimum(base_top, rr), np.minimum(base_left, cc)
    maxr, maxc = np.maximum(base_bottom, rr+ph), np.maximum(base_right, cc+pw)
    valid &= (maxr-minr <= 30) & (maxc-minc <= 30)
    indices = np.argwhere(valid)
    return rr, cc, minr, minc, maxr, maxc, indices


def _place(rng, masks, points, anchor, dims, colors):
    h, w = dims
    n = len(points)
    map_shape = tuple(map(int, np.max(points, axis=0)+1))
    options = _gallery_options(n, dims, map_shape, anchor)
    rng.shuffle(options)
    for option in options:
        rr, cc, minr, minc, maxr, maxc, indices = _map_positions(option, dims, map_shape)
        if len(indices):
            break
    else:
        return None
    if rng.random() < .65:
        areas = (maxr-minr)*(maxc-minc)
        minimum = min(areas[tuple(indices.T)])
        indices = np.array([p for p in indices if areas[tuple(p)] <= minimum*1.20])
    i, j = map(int, _pick(rng, indices))
    pr, pc = int(rr[i, j]), int(cc[i, j])
    base_h, base_w = int(maxr[i, j]-minr[i, j]), int(maxc[i, j]-minc[i, j])
    nr, nc, sr, sc, gt, gl, gh, gw = option
    bg, divider, ink, marker = colors
    occupied = sum(int(m.sum()) for m in masks)
    div_count = gh*gw - nr*nc*h*w
    # The outside anchor adds one ink cell back to the gallery's ink count.
    min_area = occupied+n+div_count+max(occupied, n, div_count)+1
    canvas_options = [(a, b) for a in range(base_h, 31) for b in range(base_w, 31)
                      if a*b >= min_area]
    if not canvas_options:
        return None
    if rng.random() < .60:
        smallest = min(a*b for a, b in canvas_options)
        canvas_options = [(a, b) for a, b in canvas_options if a*b <= smallest*1.15]
    ch, cw = _pick(rng, canvas_options)
    top = int(rng.integers(ch-base_h+1))-int(minr[i, j])
    left = int(rng.integers(cw-base_w+1))-int(minc[i, j])
    inp = np.full((ch, cw), bg, dtype=np.int8)
    library = np.full((gh, gw), bg, dtype=np.int8)
    library[::h+1, :] = divider
    library[:, ::w+1] = divider
    registered = points.index(anchor)
    slots = [(r, c) for r in range(nr) for c in range(nc) if (r, c) != (sr, sc)]
    rng.shuffle(slots)
    others = [tile for tile in range(n) if tile != registered]
    rng.shuffle(others)
    assignment = [(registered, (sr, sc))] + list(zip(others, slots[:n-1]))
    for tile, (r, c) in assignment:
        block = library[r*(h+1)+1:r*(h+1)+1+h, c*(w+1)+1:c*(w+1)+1+w]
        block[masks[tile]] = ink
        if tile == registered:
            interior = np.argwhere(masks[tile][1:-1, 1:-1])+1
            accent = tuple(map(int, _pick(rng, interior)))
            block[accent] = marker
    inp[top+gt:top+gt+gh, left+gl:left+gl+gw] = library
    out = np.full_like(inp, bg)
    # Labels use the unshuffled construction plan, not the input-only solver.
    for tile, (r, c) in enumerate(points):
        inp[top+pr+2*r, left+pc+2*c] = ink if (r, c) == anchor else marker
        block = out[top+r*h:top+(r+1)*h, left+c*w:left+(c+1)*w]
        block[masks[tile]] = marker if (r, c) == anchor else ink
    return {"input": inp, "output": out}


def _make_pair(rng, regime, capability, colors, *, test=False, teaching=0, previous=None):
    required_topology = ("cycle" if regime == "general" and teaching == 1 else
                         "branch" if regime == "tree" and teaching == 1
                         or regime == "general" and teaching == 2 else None)
    kinds = ({"single", "thick", "separated"} if capability == 2 else
             {"single", "thick"} if capability == 1 else {"single"}) if teaching == 1 else set()
    for _ in range(160):
        minimum = 4 if required_topology or len(kinds) == 3 else 3
        n = int(rng.integers(7, 9) if test else rng.integers(minimum, 7))
        local_regime = "tree" if regime == "general" and teaching == 2 else regime
        points = _sample_map(rng, n, local_regime, required_topology, bool(teaching))
        map_shape = tuple(map(int, np.max(points, axis=0)+1))
        dimensions = [(h, w) for h in range(4, 7) for w in range(4, 7)
                      if map_shape[0]*h <= 28 and map_shape[1]*w <= 28
                      and (not kinds or capability < 2 or min(h, w) >= 5)
                      and (teaching != 2 or (h != previous["tile_shape"][0]
                                            and w != previous["tile_shape"][1]))]
        if not dimensions:
            continue
        dims = _pick(rng, dimensions)
        anchors = [p for p in points if (teaching != 1 or p != (0, 0))
                   and (teaching != 2 or (p[0] != previous["anchor"][0]
                                         and p[1] != previous["anchor"][1]))]
        if not anchors:
            continue
        anchor = _pick(rng, anchors)
        # The second witness uses singleton contacts: port cardinality then
        # provides no substitute for exact positions in the ambiguous shape.
        local_capability = 0 if teaching == 2 else capability
        ports = _edge_ports(rng, points, dims, local_capability, kinds)
        # Uniqueness is first checked cheaply on geometry, then on final pixels.
        prototype = dict(points=points, ports=ports, anchor=anchor, registered=points.index(anchor))
        if len(_placements(prototype)) != 1:
            continue
        if teaching == 2 and len(_position_blind_placements(prototype)) < 2:
            continue
        masks = [_mask(rng, dims, p) for p in ports]
        pair = _place(rng, masks, points, anchor, dims, colors)
        if pair is None:
            continue
        info = _analyze_pair(pair)
        if teaching and not (info["reading_fails"] and info["repetition_fails"]):
            continue
        if teaching == 2 and not info["positions_needed"]:
            continue
        if kinds - info["kinds"]:
            continue
        return pair, info
    raise RuntimeError("bounded construction could not find a valid witnessed pair")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime = _pick(rng, ("path", "tree", "general"))
    capability = int(rng.integers(3))
    consistent = bool(rng.random() < .25)
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    puzzle, seen, previous = {"train": [], "test": []}, set(), None
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            teaching = i+1 if split == "train" and i < 2 else 0
            for _ in range(20):
                colors = palette if consistent else tuple(map(int, rng.choice(10, 4, replace=False)))
                pair, info = _make_pair(rng, regime, capability, colors, test=split == "test",
                                        teaching=teaching, previous=previous)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded duplicate avoidance exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
            if teaching == 1:
                previous = info
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded 42-seed semantic and variation checks; returns measured coverage."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[17]) == _puzzle_key(generate_puzzle(17))
    records, masks, maps, roles, gallery_shapes, shapes = [], set(), set(), set(), set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_pair(pair)
                records.append(info)
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                masks.update(_key(m.astype(np.int8)) for m in info["masks"])
                maps.add(info["points"])
                roles.add(info["roles"])
                gallery_shapes.add(info["gallery_shape"])
                shapes.add(info["shape"])
    assert {m["n"] for m in records} == set(range(3, 9))
    assert len(masks) >= 200 and len(maps) >= 90 and len(shapes) >= 60
    assert len({m["tile_shape"] for m in records}) >= 8
    assert len({m["map_shape"] for m in records}) >= 12
    assert len({m["anchor"] for m in records}) >= 12
    assert len(gallery_shapes) >= 8 and len(roles) >= 80
    assert any(m["topology"][1] > 0 for m in records)
    assert any(m["topology"][0] >= 3 for m in records)
    assert any(max(m["topology"]) <= 2 and m["topology"][1] == 0 for m in records)
    assert any(m["gallery_shape"][0]*m["gallery_shape"][1] > m["n"] for m in records)
    assert {"single", "thick", "separated"} == set().union(*(m["kinds"] for m in records))
    for role in range(4):
        assert {colors[role] for colors in roles} == set(range(10))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(1000+nt*10+nq, nt, nq))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Non-generation family witnesses: a two-tile map with 3x3 tiles, and
    # distinct same-port intermediate pieces. The decoder supports both;
    # the latter must be rejected rather than tied by source order.
    rng = np.random.default_rng(91)
    points = ((0, 0), (0, 1))
    dims, colors = (3, 3), (0, 5, 2, 8)
    ports = [((), (1,), (), ()), ((), (), (), (1,))]
    pair = _place(rng, [_mask(rng, dims, p) for p in ports], points, (0, 1), dims, colors)
    assert pair is not None and np.array_equal(solve(pair["input"]), pair["output"])
    square = ((0, 0), (0, 1), (1, 0), (1, 1))
    for dims in ((8, 3), (3, 8), (8, 8)):
        ports = _edge_ports(rng, square, dims, 2, set())
        pair = _place(rng, [_mask(rng, dims, p) for p in ports], square, (1, 0), dims, colors)
        assert pair is not None and np.array_equal(solve(pair["input"]), pair["output"])
    ring = tuple((r, c) for r in range(3) for c in range(3) if (r, c) != (1, 1))
    ring_ports = [[()] * 4 for _ in ring]
    ring_index = {p: i for i, p in enumerate(ring)}
    codes = ((1,), (2,), (3,), (1, 2))
    for d in (1, 2):
        edges = [(i, ring_index[r+_DIRS[d][0], c+_DIRS[d][1]])
                 for i, (r, c) in enumerate(ring)
                 if (r+_DIRS[d][0], c+_DIRS[d][1]) in ring_index]
        for (i, j), signature in zip(edges, codes):
            ring_ports[i][d] = ring_ports[j][(d+2) % 4] = signature
    pair = _place(rng, [_mask(rng, (5, 5), tuple(p)) for p in ring_ports],
                  ring, (2, 1), (5, 5), colors)
    assert pair is not None
    ring_info = _analyze_pair(pair)
    assert ring_info["map_shape"] == (3, 3) and (1, 1) not in ring_info["points"]
    assert np.array_equal(solve(pair["input"]), pair["output"])
    ambiguous_masks = []
    for cells in (((1, 2), (1, 3)), tuple((1, c) for c in range(4)),
                  tuple((1, c) for c in range(4)) + ((2, 1),), ((1, 0), (1, 1))):
        mask = np.zeros((4, 4), dtype=bool)
        for cell in cells:
            mask[cell] = True
        ambiguous_masks.append(mask)
    ambiguous = _place(rng, ambiguous_masks, tuple((0, c) for c in range(4)),
                       (0, 0), (4, 4), colors)
    assert ambiguous is not None and len(_placements(_decode(ambiguous["input"]))) == 2
    try:
        solve(ambiguous["input"])
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous same-port pieces accepted")
    pair = puzzles[0]["train"][0]
    damaged = {k: v.copy() for k, v in pair.items()}
    info = _decode(pair["input"])
    damaged["output"][info["origin"]] = info["roles"][1]
    try:
        _analyze_pair(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    # A tile's orientation and port pattern must rule out at least one swapped
    # placement in every teaching example, checked by explicit wrong outputs.
    return dict(seeds=42, pairs=len(records), distinct_masks=len(masks),
                distinct_maps=len(maps), canvas_shapes=len(shapes),
                tile_shapes=len({m["tile_shape"] for m in records}),
                map_shapes=len({m["map_shape"] for m in records}),
                anchors=len({m["anchor"] for m in records}),
                role_palettes=len(roles), gallery_shapes=len(gallery_shapes),
                cyclic_pairs=sum(m["topology"][1] > 0 for m in records),
                branching_pairs=sum(m["topology"][0] >= 3 for m in records))
