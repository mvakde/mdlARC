"""Route loom: stretch a carrier and continuously tile a stencil along a route.

Only standard-library modules and NumPy are used. The input frame identifies
background; the bottom legend supplies the key and both swatches explicitly.
"""
from __future__ import annotations

import copy
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= d <= 30 for d in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("grids must be integer 2D arrays, dimensions 1..30, colors 0..9")
    return a.astype(np.int64, copy=False)


def _signature(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def _normalized(a):
    mapping = {}
    return a.shape, tuple(mapping.setdefault(int(x), len(mapping)) for x in a.flat)


def _components(mask):
    pending = set((int(r), int(c)) for r, c in zip(*np.nonzero(mask)))
    result = []
    while pending:
        start = min(pending)
        pending.remove(start)
        stack, points = [start], []
        while stack:
            r, c = stack.pop()
            points.append((r, c))
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in pending:
                    pending.remove(q)
                    stack.append(q)
        result.append(points)
    return result


def _bounds(points):
    rs, cs = zip(*points)
    return min(rs), min(cs), max(rs)+1, max(cs)+1


def _edge(a, b, size, width):
    """Cells indexed [progress, traveler's left-to-right cross section, rc]."""
    r, c = a[:2]
    y, x = b[:2]
    off = (size-width)//2
    if r == y and abs(c-x) > size:
        if x > c:
            columns = range(c+size, x)
            rows = list(range(r+off, r+off+width))
            direction = "R"
        else:
            columns = range(c-1, x+size-1, -1)
            rows = list(range(r+off+width-1, r+off-1, -1))
            direction = "L"
        cells = [[(yy, xx) for yy in rows] for xx in columns]
    elif c == x and abs(r-y) > size:
        if y > r:
            rows = range(r+size, y)
            columns = list(range(c+off+width-1, c+off-1, -1))
            direction = "D"
        else:
            rows = range(r-1, y+size-1, -1)
            columns = list(range(c+off, c+off+width))
            direction = "U"
        cells = [[(yy, xx) for xx in columns] for yy in rows]
    else:
        return None
    return np.asarray(cells, dtype=int), direction


def _routes(stations, key, edges, limit=2):
    answers = []
    choices = [[i for i, s in enumerate(stations) if s[2] == color] for color in key]

    def visit(path, used, painted):
        if len(answers) >= limit:
            return
        if len(path) == len(key):
            answers.append(tuple(path))
            return
        for i in choices[len(path)]:
            if i in used:
                continue
            cells = set()
            if path:
                e = edges.get((path[-1], i))
                if e is None:
                    continue
                cells = set(map(tuple, e[0].reshape(-1, 2)))
                if cells & painted:
                    continue
            visit(path+[i], used | {i}, painted | cells)

    visit([], set(), set())
    return answers


def _decode(input_grid):
    g = _grid(input_grid)
    bg = int(g[0, 0])
    assert all(np.all(x == bg) for x in (g[0], g[-1], g[:, 0], g[:, -1])), "background frame"
    separators = [r for r in range(1, g.shape[0]-1)
                  if g[r, 1] != bg and np.all(g[r, 1:-1] == g[r, 1])]
    assert len(separators) == 1, "one full-width interior divider"
    d = separators[0]
    frame = int(g[d, 1])
    stations = []
    geometry = None
    rebuilt = np.full(g.shape, bg, dtype=int)
    rebuilt[d, 1:-1] = frame
    for points in _components(g[:d] == frame):
        r, c, y, x = _bounds(points)
        size = y-r
        assert 3 <= size <= 8 and x-c == size
        block = g[r:y, c:x]
        rr, cc = np.nonzero(block != frame)
        assert len(rr)
        off = int(rr.min())
        width = int(rr.max())-off+1
        assert 1 <= width <= 4 and off > 0 and 2*off+width == size
        assert int(cc.min()) == off and int(cc.max()) == off+width-1
        assert len(rr) == width*width
        label = int(block[off, off])
        assert label not in (bg, frame)
        assert np.all(block[off:off+width, off:off+width] == label)
        assert geometry is None or geometry == (size, width)
        geometry = size, width
        stations.append((r, c, label))
        rebuilt[r:y, c:x] = block
    assert geometry is not None and 3 <= len(stations) <= 16
    size, width = geometry
    stations.sort()
    assert max(r+size for r, c, label in stations) <= d-1
    key_y = d+2
    assert key_y < g.shape[0]
    key_xs = np.flatnonzero(g[key_y] != bg)
    assert 3 <= len(key_xs) <= 10 and np.all(np.diff(key_xs) == 2)
    key = [int(g[key_y, c]) for c in key_xs]
    assert all(c not in (bg, frame) for c in key)
    rebuilt[key_y, key_xs] = key
    plaques = []
    for points in _components(g[d+3:] == frame):
        r, c, y, x = _bounds(points)
        r, y = r+d+3, y+d+3
        assert r == d+4 and y-r == width+2 and 3 <= x-c <= 9
        block = g[r:y, c:x]
        assert all(np.all(edge == frame) for edge in (block[0], block[-1], block[:, 0], block[:, -1]))
        assert not np.any(block[1:-1, 1:-1] == frame)
        plaques.append((c, block[1:-1, 1:-1].copy()))
        rebuilt[r:y, c:x] = block
    assert len(plaques) == 2
    plaques.sort(key=lambda p: p[0])
    carrier, stencil = plaques[0][1], plaques[1][1]
    assert np.all(carrier != bg)
    assert np.array_equal(g, rebuilt), "unexpected ink or legend layout"
    edges = {}
    for i, a in enumerate(stations):
        for j, b in enumerate(stations):
            e = _edge(a, b, size, width)
            if e is not None and np.all(g[e[0][..., 0], e[0][..., 1]] == bg):
                edges[i, j] = e
    routes = _routes(stations, key, edges)
    assert len(routes) == 1, "the key must have exactly one noncrossing distinct-station route"
    route = routes[0]
    links = [edges[i, j] for i, j in zip(route, route[1:])]
    coordinates = np.concatenate([e[0] for e in links], axis=0)
    length = len(coordinates)
    k = carrier.shape[1]
    assert length >= k and length % k == 0, "carrier requires integral replication"
    return dict(grid=g, bg=bg, frame=frame, divider=d, size=size, width=width,
                key=key, stations=stations, edges=edges, route=route, links=links,
                coordinates=coordinates, length=length, carrier=carrier, stencil=stencil)


def _render(info, variant="correct"):
    g = info["grid"]
    carrier, stencil = info["carrier"], info["stencil"]
    length, bg = info["length"], info["bg"]
    k, p = carrier.shape[1], stencil.shape[1]
    t = np.arange(length)
    base = carrier[:, t//(length//k)]
    tiled = stencil[:, t % p]
    if variant == "tile_carrier":
        base = carrier[:, t % k]
    elif variant == "local_stretch":
        base = np.concatenate([carrier[:, np.arange(len(e[0]))*k//len(e[0])]
                               for e in info["links"]], axis=1)
    elif variant == "stretch_stencil":
        tiled = stencil[:, t*p//length]
    elif variant == "overlay_before_stretch":
        tiled = stencil[:, (t//(length//k)) % p]
    elif variant == "ignore_stencil":
        tiled = np.full_like(base, bg)
    elif variant == "opaque_background":
        base = np.full_like(base, bg)
    elif variant == "reset_stencil":
        local = np.concatenate([np.arange(len(e[0])) for e in info["links"]])
        tiled = stencil[:, local % p]
    elif variant == "count_station_phase":
        skipped = np.concatenate([np.full(len(e[0]), i*info["size"], dtype=int)
                                  for i, e in enumerate(info["links"])])
        tiled = stencil[:, (t+skipped) % p]
    elif variant == "shift_stencil":
        tiled = stencil[:, (t+1) % p]
    ribbon = np.where(tiled == bg, base, tiled).T
    coordinates = info["coordinates"].copy()
    if variant == "screen_cross_section":
        at = 0
        for cells, direction in info["links"]:
            if direction in ("L", "D"):
                coordinates[at:at+len(cells)] = cells[:, ::-1]
            at += len(cells)
    elif variant == "reverse_route":
        links = [info["edges"][j, i] for i, j in zip(info["route"], info["route"][1:])][::-1]
        coordinates = np.concatenate([e[0] for e in links])
    elif variant == "source_color":
        ribbon = np.concatenate([np.full(e[0].shape[:2], info["stations"][i][2], dtype=int)
                                 for i, e in zip(info["route"], info["links"])])
    elif variant == "world_phase":
        tile = stencil[coordinates[..., 0] % info["width"], coordinates[..., 1] % p]
        ribbon = np.where(tile == bg, base.T, tile)
    out = g.copy()
    out[coordinates[..., 0], coordinates[..., 1]] = ribbon
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from the visible grid alone; neither seed nor demonstrations enter."""
    return _render(_decode(input_grid))


_ALTERNATIVES = ("tile_carrier", "stretch_stencil", "overlay_before_stretch",
                 "ignore_stencil", "opaque_background", "reset_stencil",
                 "shift_stencil", "screen_cross_section", "reverse_route",
                 "source_color", "world_phase", "local_stretch", "count_station_phase")
_REQUIRED = set(_ALTERNATIVES) | {"repeat", "unused_gap", "visible_carrier", "overwrite",
                                "R", "D", "L", "U"}


def _evidence(info, output):
    cover = {v for v in _ALTERNATIVES if not np.array_equal(output, _render(info, v))}
    cover.update(e[1] for e in info["links"])
    if len(set(info["key"])) < len(info["key"]):
        cover.add("repeat")
    selected = {frozenset((i, j)) for i, j in zip(info["route"], info["route"][1:])}
    if any(frozenset(e) not in selected for e in info["edges"]):
        cover.add("unused_gap")
    t = np.arange(info["length"])
    tile = info["stencil"][:, t % info["stencil"].shape[1]]
    base = info["carrier"][:, t//(info["length"]//info["carrier"].shape[1])]
    if np.any(tile == info["bg"]):
        cover.add("visible_carrier")
    if np.any((tile != info["bg"]) & (tile != base)):
        cover.add("overwrite")
    return cover


def _coordinates(rng, count, size, limit):
    gaps = np.ones(count-1, dtype=int)
    slack = limit-count*size-(count-1)
    for i in rng.permutation(count-1):
        extra = int(rng.integers(min(6, slack)+1))
        gaps[i] += extra
        slack -= extra
    return [0]+list(np.cumsum(size+gaps).astype(int))


def _random_path(rng, positions, edges, length):
    budget = 1500

    def walk(path, used_cells):
        nonlocal budget
        budget -= 1
        if budget < 0:
            return None
        if len(path) == length:
            return path
        candidates = [j for i, j in edges if i == path[-1] and j not in path]
        for idx in rng.permutation(len(candidates)):
            j = candidates[int(idx)]
            cells = set(map(tuple, edges[path[-1], j][0].reshape(-1, 2)))
            if cells & used_cells:
                continue
            result = walk(path+[j], used_cells | cells)
            if result is not None:
                return result
        return None

    for i in rng.permutation(len(positions)):
        result = walk([int(i)], set())
        if result is not None:
            return result
        if budget < 0:
            break
    return None


def _label(rng, positions, edges, path, palette):
    distinct = rng.random() < .35
    n = len(path)
    for attempt in range(60):
        count = n if distinct else int(rng.integers(2, min(n, 5)+1))
        colors = list(rng.permutation(palette)[:count])
        key = colors+[int(rng.choice(colors)) for _ in range(n-count)]
        key = list(map(int, rng.permutation(key)))
        labels = [int(rng.choice(palette if rng.random() < .4 else key)) for _ in positions]
        for i, color in zip(path, key):
            labels[i] = color
        stations = [(r, c, color) for (r, c), color in zip(positions, labels)]
        if _routes(stations, key, edges) == [tuple(path)]:
            return labels, key
        if attempt == 40:
            distinct = True
    return None


def _swatches(rng, width, k, p, palette, bg):
    inks = rng.choice(palette, size=int(rng.integers(1, 6)), replace=False)
    rr, cc = np.indices((width, k))
    kind = int(rng.integers(5))
    if kind == 0:
        labels = rng.integers(len(inks), size=(width, k))
    elif kind == 1:
        labels = np.tile(rng.integers(len(inks), size=k), (width, 1))
    elif kind == 2:
        labels = np.tile(rng.integers(len(inks), size=(width, 1)), (1, k))
    elif kind == 3:
        labels = (rr//int(rng.integers(1, width+1))+cc//int(rng.integers(1, k+1))) % len(inks)
    else:
        centers = rng.choice(width*k, size=min(width*k, len(inks)+1), replace=False)
        distance = abs(rr[..., None]-centers//k)+abs(cc[..., None]-centers % k)
        labels = rng.integers(len(inks), size=len(centers))[distance.argmin(axis=2)]
    carrier = inks[labels].astype(int)
    stencil = np.full((width, p), bg, dtype=int)
    if rng.random() >= .12:
        rows, columns = np.indices(stencil.shape)
        mode = int(rng.integers(5))
        if mode == 0:
            mask = rng.random(stencil.shape) < rng.uniform(.15, .8)
        elif mode == 1:
            mask = (rows+columns+int(rng.integers(2))) % 2 == 0
        elif mode == 2:
            mask = np.tile(rng.random(p) < .5, (width, 1))
        elif mode == 3:
            mask = (rows == int(rng.integers(width))) | (columns == int(rng.integers(p)))
        else:
            mask = np.ones(stencil.shape, dtype=bool)
            mask.ravel()[rng.choice(width*p, size=max(1, width*p//2), replace=False)] = False
        if not mask.any():
            mask.ravel()[int(rng.integers(width*p))] = True
        if mask.all():
            mask.ravel()[int(rng.integers(width*p))] = False
        paints = rng.choice(palette, size=int(rng.integers(1, min(4, int(mask.sum()))+1)), replace=False)
        stencil[mask] = rng.choice(paints, size=int(mask.sum()))
    return carrier, stencil


def _assemble(rng, positions, path, labels, key, size, width, carrier, stencil, roles):
    bg, frame = roles
    bh = max(r for r, c in positions)+size
    bw = max(c for r, c in positions)+size
    k, p = carrier.shape[1], stencil.shape[1]
    hmin = bh+width+9
    wmin = max(bw+2, 2*len(key)+1, k+p+7)
    assert hmin <= 30 and wmin <= 30
    h, w = int(rng.integers(hmin, 31)), int(rng.integers(wmin, 31))
    top = int(rng.integers(1, h-hmin+2))
    left = int(rng.integers(1, w-bw))
    positions = [(r+top, c+left) for r, c in positions]
    divider = h-width-7
    g = np.full((h, w), bg, dtype=int)
    off = (size-width)//2
    for (r, c), color in zip(positions, labels):
        g[r:r+size, c:c+size] = frame
        g[r+off:r+off+width, c+off:c+off+width] = color
    g[divider, 1:-1] = frame
    key_x = int(rng.integers(1, w-(2*len(key)-1)))
    g[divider+2, key_x:key_x+2*len(key):2] = key
    plaque_x = int(rng.integers(1, w-(k+p+5)))
    for x, image in ((plaque_x, carrier), (plaque_x+k+3, stencil)):
        y, cols = divider+4, image.shape[1]
        g[y:y+width+2, x:x+cols+2] = frame
        g[y+1:y+width+1, x+1:x+cols+1] = image
    # Construction expands runs and rotates whole rectangular pieces. The
    # input-only decoder instead indexes individual intrinsic ribbon cells.
    links = [_edge(positions[i], positions[j], size, width) for i, j in zip(path, path[1:])]
    length = sum(len(e[0]) for e in links)
    base = np.repeat(carrier, length//k, axis=1)
    tiled = np.tile(stencil, (1, (length+p-1)//p))[:, :length]
    ribbon = np.where(tiled == bg, base, tiled)
    out, at = g.copy(), 0
    for (i, j), (cells, direction) in zip(zip(path, path[1:]), links):
        count = len(cells)
        piece = ribbon[:, at:at+count]
        a, b = positions[i], positions[j]
        if direction == "R":
            out[a[0]+off:a[0]+off+width, a[1]+size:b[1]] = piece
        elif direction == "L":
            out[a[0]+off:a[0]+off+width, b[1]+size:a[1]] = piece[::-1, ::-1]
        elif direction == "D":
            out[a[0]+size:b[0], a[1]+off:a[1]+off+width] = np.rot90(piece, -1)
        else:
            out[b[0]+size:a[0], a[1]+off:a[1]+off+width] = np.rot90(piece, 1)
        at += count
    return {"input": g, "output": out}


def _sample_pair(rng, roles, test=False):
    bg, frame = roles
    palette = [c for c in range(10) if c not in roles]
    for attempt in range(160):
        width = int(rng.choice((1, 2, 3, 4), p=(.15, .3, .35, .2)))
        size = width+(2 if rng.random() < .83 else 4)
        limit_h = 30-width-9
        length = int(rng.integers(6, 9) if test else rng.integers(3, 6))
        options = [(r, c) for r in range(1, 5) for c in range(1, 7)
                   if r*size+r-1 <= limit_h and c*size+c-1 <= 28 and r*c >= length]
        if not options:
            continue
        weights = np.array([1/(r*c-length+1)**.7 for r, c in options])
        nr, nc = options[int(rng.choice(len(options), p=weights/weights.sum()))]
        count = length+int(rng.integers(min(3, nr*nc-length)+1))
        ys, xs = _coordinates(rng, nr, size, limit_h), _coordinates(rng, nc, size, 28)
        universe = [(y, x) for y in ys for x in xs]
        positions = sorted(universe[int(i)] for i in rng.choice(len(universe), count, replace=False))
        occupancy = np.zeros((limit_h, 28), dtype=bool)
        for r, c in positions:
            occupancy[r:r+size, c:c+size] = True
        edges = {}
        for i, a in enumerate(positions):
            for j, b in enumerate(positions):
                e = _edge(a, b, size, width)
                if e is not None and not occupancy[e[0][..., 0], e[0][..., 1]].any():
                    edges[i, j] = e
        path = _random_path(rng, positions, edges, length)
        if path is None:
            continue
        total = sum(len(edges[i, j][0]) for i, j in zip(path, path[1:]))
        divisors = [k for k in range(2, 8) if total % k == 0 and total//k >= 2]
        if not divisors:
            continue
        assigned = _label(rng, positions, edges, path, palette)
        if assigned is None:
            continue
        labels, key = assigned
        k, p = int(rng.choice(divisors)), int(rng.integers(2, 7))
        carrier, stencil = _swatches(rng, width, k, p, palette, bg)
        return _assemble(rng, positions, path, labels, key, size, width, carrier, stencil, roles)
    return _fallback(rng, roles, test=test)


def _fallback(rng, roles, test=False, mirrored=False, serial=0):
    """Finite constructive witnesses obey exactly the normal rule and grammar."""
    bg, frame = roles
    colors = list(map(int, rng.permutation([c for c in range(10) if c not in roles])))
    a, b, c, paint = colors[:4]
    if test:
        width, size = 2, 4
        xs = [0, 5, 11, 18]
        positions = [(r, x) for r in (0, 7) for x in xs]
        path = [0, 1, 2, 3, 7, 6]
        key = colors[:6]
        labels = [colors[6]]*8
        for i, color in zip(path, key):
            labels[i] = color
        p = 5
    else:
        width = 3 if mirrored else 2
        size = width+2
        positions = [(0, 0), (0, size+2), (size+3, size+2),
                     (size+3, 0), (size+3, 2*size+6)]
        path, key, labels, p = [0, 1, 2, 4], [a, b, b, c], [a, b, b, a, c], 4
    if mirrored:
        maxr, maxc = max(r for r, c in positions), max(c for r, c in positions)
        positions = [(maxr-r, maxc-c) for r, c in positions]
    carrier = np.array([[colors[(row+col) % 3] for col in range(3)] for row in range(width)])
    # Serial changes an actual carrier cell, so even a duplicate fallback is
    # avoidable without changing the key, route, or structural-role convention.
    carrier.flat[serial % carrier.size] = colors[(serial//carrier.size) % 8]
    stencil = np.full((width, p), bg, dtype=int)
    for row in range(width):
        stencil[row, (1+2*row) % p] = paint
    return _assemble(rng, positions, path, labels, key, size, width, carrier, stencil, roles)


def _counts(num_train, num_test):
    for value, lo, hi, name in ((num_train, 2, 10, "num_train"), (num_test, 1, 4, "num_test")):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    roles = tuple(map(int, rng.choice(10, 2, replace=False)))
    candidates, covers, seen = [], [], set()
    train = None
    for _ in range(96):
        pair = _sample_pair(rng, roles)
        signature = _signature(pair["input"])
        if signature in seen:
            continue
        info = _decode(pair["input"])
        cover = _evidence(info, pair["output"])
        for j in rng.permutation(len(candidates)):
            j = int(j)
            if _REQUIRED <= (cover | covers[j]):
                train = [candidates[j], pair]
                break
        if train is not None:
            break
        candidates.append(pair)
        covers.append(cover)
        seen.add(signature)
    if train is None:
        train = [_fallback(rng, roles), _fallback(rng, roles, mirrored=True)]
    puzzle = {"train": train, "test": []}
    seen = {_signature(pair["input"]) for pair in train}
    for split, count in (("train", nt), ("test", nq)):
        while len(puzzle[split]) < count:
            for _ in range(32):
                pair = _sample_pair(rng, roles, test=split == "test")
                if _signature(pair["input"]) not in seen:
                    break
            else:
                for serial in range(128):
                    pair = _fallback(rng, roles, test=split == "test", serial=serial)
                    if _signature(pair["input"]) not in seen:
                        break
                else:
                    raise AssertionError("distinct constructive witnesses exhausted")
            puzzle[split].append(pair)
            seen.add(_signature(pair["input"]))
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, roles, cover = set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g, out = _grid(pair["input"]), _grid(pair["output"])
            assert g.shape == out.shape
            sig = _signature(g)
            assert sig not in seen, "duplicate inputs"
            seen.add(sig)
            info = _decode(g)
            roles.add((info["bg"], info["frame"]))
            expected = _render(info)
            assert np.array_equal(out, expected), "incorrect ribbon"
            coords = info["coordinates"]
            footprint = np.zeros(g.shape, dtype=bool)
            footprint[coords[..., 0], coords[..., 1]] = True
            assert footprint.sum() == info["length"]*info["width"], "overlapping links"
            assert np.all(g[footprint] == info["bg"])
            assert np.all(out[footprint] != info["bg"]), "transparent cells retain the carrier"
            assert np.array_equal(g[~footprint], out[~footprint]), "nonroute cells must be preserved"
            if split == "train":
                assert 3 <= len(info["key"]) <= 5
                cover |= _evidence(info, out)
            else:
                assert 6 <= len(info["key"]) <= 8
    assert len(roles) == 1, "background and structural frame are constant within a puzzle"
    assert _REQUIRED <= cover, "missing demonstration evidence: "+str(_REQUIRED-cover)


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in a)


def self_test():
    """Bounded 42-seed audit, counterfactual witnesses, and boundary checks."""
    # A hand-calculated R,D,L witness: gaps 3,4,3, global factor 5,
    # stencil period 3. The second and third bands start mid-stencil.
    witness = _assemble(np.random.default_rng(73),
                        [(0, 0), (0, 7), (8, 7), (8, 0)], [0, 1, 2, 3],
                        [5, 6, 7, 8], [5, 6, 7, 8], 4, 2,
                        np.array([[2, 3], [3, 2]]),
                        np.array([[0, 4, 0], [4, 0, 0]]), (0, 1))
    a = _decode(witness["input"])
    start = a["stations"][a["route"][0]]
    r, c = start[:2]
    manual = witness["input"].copy()
    manual[r+1:r+3, c+4:c+7] = [[2, 4, 2], [4, 3, 3]]
    manual[r+4:r+8, c+8:c+10] = [[4, 2], [3, 4], [2, 3], [4, 3]]
    manual[r+9:r+11, c+4:c+7] = [[4, 2, 2], [3, 3, 4]]
    assert np.array_equal(manual, witness["output"])
    assert np.array_equal(solve(witness["input"]), manual)
    # Rule-level members beyond normal sampling: identity expansion, period
    # seven, singleton swatches, and fully empty/opaque masks.
    boundary = _assemble(np.random.default_rng(74),
                         [(0, 0), (0, 5), (6, 5)], [0, 1, 2],
                         [5, 6, 8], [5, 6, 8], 4, 2,
                         np.array([[2, 3, 4], [3, 4, 2]]),
                         np.array([[0, 4, 0, 0, 2, 0, 3], [3, 0, 0, 4, 0, 2, 0]]), (0, 1))
    assert _decode(boundary["input"])["length"] == 3
    assert np.array_equal(solve(boundary["input"]), boundary["output"])
    for stencil in (np.array([[0], [0]]), np.array([[4], [5]]), np.array([[0], [4]])):
        boundary = _assemble(np.random.default_rng(75),
                             [(0, 0), (0, 7), (8, 7), (8, 0)], [0, 1, 2, 3],
                             [5, 6, 7, 8], [5, 6, 7, 8], 4, 2,
                             np.array([[2], [3]]), stencil, (0, 1))
        assert np.array_equal(solve(boundary["input"]), boundary["output"])
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert _equal(puzzles[0], generate_puzzle(0))
    assert not _equal(puzzles[0], puzzles[1])
    assert len({tuple(_signature(pair["input"]) for s in p for pair in p[s]) for p in puzzles}) == 42
    metrics = {s: [] for s in ("train", "test")}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in metrics:
            for pair in puzzle[split]:
                a = _decode(pair["input"])
                metrics[split].append(a)
                # Palette permutation cannot change geometry or transparency.
                mapping = (3*np.arange(10)+7) % 10
                assert np.array_equal(solve(mapping[pair["input"]]), mapping[pair["output"]])
    all_info = metrics["train"]+metrics["test"]
    assert {a["width"] for a in all_info} == {1, 2, 3, 4}
    assert len({a["grid"].shape for a in all_info}) >= 25
    assert len({a["length"] for a in all_info}) >= 20
    assert len({a["bg"] for a in all_info}) >= 8
    assert len({_normalized(a["carrier"]) for a in all_info}) >= 80
    assert len({_normalized(a["stencil"]) for a in all_info}) >= 80
    assert len({tuple((r-min(s[0] for s in a["stations"]), c-min(s[1] for s in a["stations"]))
                      for r, c, label in a["stations"]) for a in all_info}) >= 80
    for split, infos in metrics.items():
        assert any(np.all(a["stencil"] == a["bg"]) for a in infos)
        assert any(a["length"] % a["stencil"].shape[1] for a in infos)
        assert any(len(a["stations"]) == len(a["key"]) for a in infos)
        assert any(len(a["stations"]) > len(a["key"]) for a in infos)
        assert any(len(set(a["key"])) < len(a["key"]) for a in infos)
        assert any(len(set(a["key"])) == len(a["key"]) for a in infos)
        assert set().union(*(set(e[1] for e in a["links"]) for a in infos)) == set("RDLU")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(20261007, nt, nq))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Exercise the constructive teaching fallback even if normal sampling
    # never invokes it, including every possible structural color assignment.
    for bg in range(10):
        rng = np.random.default_rng(1000+bg)
        roles = (bg, (bg+1) % 10)
        pair1 = _fallback(rng, roles)
        pair2 = _fallback(rng, roles, mirrored=True)
        test = _fallback(rng, roles, test=True)
        validate_generated_puzzle({"train": [pair1, pair2], "test": [test]})
    damaged = copy.deepcopy(puzzles[0])
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupted output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds and all legal explicit counts")
