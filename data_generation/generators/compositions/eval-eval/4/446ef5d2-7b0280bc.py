"""Minimum-stop jigsaw relays: edge words are connections and tiles are stations.

Only a selected route is assembled. No parent generator, generation metadata,
training label, cache, or seed is consulted by the input-only solver.
"""
from collections import deque
from itertools import product
import math

import numpy as np


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= d <= 30 for d in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return g.astype(np.int8, copy=False)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        cells = [first]
        for r, c in cells:
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    cells.append(p)
        result.append(cells)
    return result


def _ports(tile, carrier):
    """An entire non-carrier run is a word, read top-down or left-right."""
    h, w = tile.shape
    result = []
    for side, values in (("T", tile[0]), ("B", tile[-1]),
                         ("L", tile[:, 0]), ("R", tile[:, -1])):
        i = 0
        while i < len(values):
            if values[i] == carrier:
                i += 1
                continue
            start = i
            while i < len(values) and values[i] != carrier:
                i += 1
            word = tuple(int(x) for x in values[start:i])
            if start == 0 or i == len(values) or len(word) not in (3, 4):
                raise ValueError("ports need three/four symbols and carrier end caps")
            if len(set(word)) < 2:
                raise ValueError("uniform runs are not relay words")
            result.append((side, start, word))
    return result


def _compatibility(tiles, ports, ordered=True):
    adjacency = [[] for _ in tiles]
    for a in range(len(tiles)):
        for b in range(a+1, len(tiles)):
            for sa, pa, wa in ports[a]:
                for sb, pb, wb in ports[b]:
                    equal = wa == wb if ordered else sorted(wa) == sorted(wb)
                    if not equal or (sa, sb) not in (("R", "L"), ("L", "R"), ("B", "T"), ("T", "B")):
                        continue
                    if sa == "R": delta = (pa-pb, tiles[a].shape[1])
                    elif sa == "L": delta = (pa-pb, -tiles[b].shape[1])
                    elif sa == "B": delta = (tiles[a].shape[0], pa-pb)
                    else: delta = (-tiles[b].shape[0], pa-pb)
                    adjacency[a].append((b, delta))
                    adjacency[b].append((a, (-delta[0], -delta[1])))
    return adjacency


def _read(g):
    bg = int(np.bincount(g.ravel(), minlength=10).argmax())
    if np.count_nonzero(g == bg)*2 <= g.size:
        raise ValueError("background must occupy more than half the input")
    candidates = []
    for color in range(10):
        q = np.argwhere(g == color)
        if color == bg or len(q) != 3:
            continue
        r, c = q.min(0)
        y, x = q.max(0)
        if (y-r, x-c) != (1, 1):
            continue
        missing = {(int(r), int(c)), (int(r), int(x)),
                   (int(y), int(c)), (int(y), int(x))} - set(map(tuple, q.tolist()))
        point = missing.pop()
        if g[point] not in (bg, color):
            candidates.append((color, point, q))
    if len(candidates) != 1:
        raise ValueError("one external L marker is required")
    marker, point, marker_cells = candidates[0]
    tiles, old, anchor = [], [], None
    for cells in _components((g != bg) & (g != marker)):
        q = np.array(cells)
        r, c = map(int, q.min(0))
        y, x = map(int, q.max(0)+1)
        if len(cells) != (y-r)*(x-c) or min(y-r, x-c) < 5:
            raise ValueError("relay tiles must be separated filled rectangles, sides >=5")
        if point in cells:
            anchor = len(tiles)
        tiles.append(g[r:y, c:x].copy())
        old.append((r, c))
    if not 5 <= len(tiles) <= 12 or anchor is None:
        raise ValueError("expected 5..12 tiles and a marked terminal")
    carrier = int(tiles[0][0, 0])
    stations, ports = [], []
    for tile in tiles:
        h, w = tile.shape
        if any(tile[p] != carrier for p in ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1))):
            raise ValueError("tile corners must be carrier")
        q = np.argwhere(tile[1:-1, 1:-1] != carrier)+1
        if len(q) != 4:
            raise ValueError("each tile needs one solid 2x2 station stamp")
        r, c = map(int, q.min(0))
        y, x = map(int, q.max(0))
        stamp = tile[r:r+2, c:c+2]
        color = int(stamp[0, 0])
        if ((y-r, x-c) != (1, 1) or abs(2*r+2-h) > 1 or abs(2*c+2-w) > 1
                or color in (bg, marker, carrier) or np.any(stamp != color)):
            raise ValueError("station stamp must be central, allowing either middle on odd sides")
        stations.append(color)
        ports.append(_ports(tile, carrier))
    colors, counts = np.unique(stations, return_counts=True)
    if len(colors) != 2 or sorted(counts.tolist()) != [2, len(tiles)-2]:
        raise ValueError("exactly two endpoint stamps and at least three ordinary stamps")
    endpoint_color = int(colors[np.flatnonzero(counts == 2)[0]])
    terminals = [i for i, c in enumerate(stations) if c == endpoint_color]
    if anchor not in terminals:
        raise ValueError("the marked tile must be a terminal")
    reserved = {bg, marker, carrier, *map(int, colors)}
    if any(set(word) & reserved for pp in ports for _, _, word in pp):
        raise ValueError("port symbols must have distinct roles")
    ar, ac = old[anchor]
    ah, aw = tiles[anchor].shape
    corner = (point[0] == ar+ah-1, point[1] == ac+aw-1)
    if point != (ar+(ah-1 if corner[0] else 0), ac+(aw-1 if corner[1] else 0)):
        raise ValueError("L marker must complete a tile corner")
    mr, mc = map(int, marker_cells.min(0))
    if corner != (point[0] == mr, point[1] == mc):
        raise ValueError("L marker must be external to that corner")
    adjacency = _compatibility(tiles, ports)
    reached, queue = {anchor}, [anchor]
    for a in queue:
        for b, _ in adjacency[a]:
            if b not in reached:
                reached.add(b)
                queue.append(b)
    if len(reached) != len(tiles):
        raise ValueError("the compatibility network must be connected")
    return dict(bg=bg, marker=marker, marker_cells=marker_cells, point=point,
                tiles=tiles, old=old, anchor=anchor, corner=corner,
                target=next(t for t in terminals if t != anchor), ports=ports,
                adjacency=adjacency, carrier=carrier, stations=stations)


def _overlap(a, ashape, b, bshape):
    return (a[0] < b[0]+bshape[0] and b[0] < a[0]+ashape[0]
            and a[1] < b[1]+bshape[1] and b[1] < a[1]+ashape[1])


def _routes(data, shape, shortest_only=False):
    """Simple routes and exact word-derived translations; no arbitrary tie break.

    A route must fit the canvas, avoid occupied-cell overlaps, and stay inward
    of the marked corner. BFS may stop after the entire first winning depth.
    """
    tiles, anchor = data["tiles"], data["anchor"]
    start = data["old"][anchor]
    queue = deque([((anchor,), (start,))])
    routes, best = [], None
    pr, pc = data["point"]
    bottom, right = data["corner"]
    while queue:
        nodes, positions = queue.popleft()
        if shortest_only and best is not None and len(nodes) > best:
            break
        a = nodes[-1]
        if a == data["target"]:
            routes.append((nodes, positions))
            best = len(nodes)
            continue
        for b, (dr, dc) in data["adjacency"][a]:
            if b in nodes:
                continue
            p = positions[-1][0]+dr, positions[-1][1]+dc
            h, w = tiles[b].shape
            if p[0] < 0 or p[1] < 0 or p[0]+h > shape[0] or p[1]+w > shape[1]:
                continue
            if (p[0]+h-1 > pr if bottom else p[0] < pr):
                continue
            if (p[1]+w-1 > pc if right else p[1] < pc):
                continue
            if any(_overlap(p, (h, w), q, tiles[n].shape) for n, q in zip(nodes, positions)):
                continue
            queue.append((nodes+(b,), positions+(p,)))
    return routes


def _paint(data, shape, route):
    output = np.full(shape, data["bg"], dtype=np.int8)
    for n, (r, c) in zip(*route):
        tile = data["tiles"][n]
        output[r:r+tile.shape[0], c:c+tile.shape[1]] = tile
    return output


def _infer(g):
    data = _read(g)
    routes = _routes(data, g.shape, shortest_only=True)
    if not routes:
        raise ValueError("no feasible terminal route")
    keys = {_paint(data, g.shape, r).tobytes() for r in routes}
    if len(routes) != 1 or len(keys) != 1:
        raise ValueError("minimum-stop relay is tied")
    return _paint(data, g.shape, routes[0]), data, routes[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the unique minimum-stop assembly, deriving every role from input."""
    return _infer(_grid(input_grid))[0]


def _order_witness(g):
    """Does ignoring word order admit a different minimum-stop answer?"""
    expected, d, _ = _infer(g)
    d["adjacency"] = _compatibility(d["tiles"], d["ports"], ordered=False)
    return any(not np.array_equal(_paint(d, g.shape, r), expected)
               for r in _routes(d, g.shape, shortest_only=True))


def _contacts(rects):
    contacts = []
    for a, (r, c, h, w) in enumerate(rects):
        for b, (rr, cc, hh, ww) in enumerate(rects):
            if c+w == cc:
                lo, hi = max(r, rr), min(r+h, rr+hh)
                if hi-lo >= 5:
                    contacts.append((a, b, "R", lo-r, lo-rr, hi-lo))
            if r+h == rr:
                lo, hi = max(c, cc), min(c+w, cc+ww)
                if hi-lo >= 5:
                    contacts.append((a, b, "B", lo-c, lo-cc, hi-lo))
    return contacts


def _partition(rng, larger=False, witness=False):
    if witness:
        heights = [int(rng.integers(7, 9)), 5, 5]
        widths = [int(x) for x in rng.integers(5, 7, size=4)]
        x = np.cumsum([0]+widths).tolist()
        a, b, c = heights
        rects = [(0, 0, a, widths[0]), (0, x[1], a, x[4]-x[1]),
                 (a, x[3], b+c, widths[3]), (a, 0, b, widths[0]),
                 (a, x[1], b, widths[1]), (a, x[2], b, widths[2])]
        # Independent bottom subdivisions supply branches and optional extra loops.
        cuts = [0]+[i for i in (1, 2) if rng.random() < .65]+[3]
        rects += [(a+b, x[i], c, x[j]-x[i]) for i, j in zip(cuts, cuts[1:])]
        return rects, (a+b+c, x[-1]), 0, 2
    nr, nc = (3, 4) if larger else (int(rng.integers(2, 4)), 3)
    heights = [int(x) for x in rng.integers(5, 7 if larger else 8, size=nr)]
    widths = [int(x) for x in rng.integers(5, 7 if larger else 8, size=nc)]
    yy, xx = np.cumsum([0]+heights), np.cumsum([0]+widths)
    rects = [(int(yy[r]), int(xx[c]), heights[r], widths[c])
             for r in range(nr) for c in range(nc)]
    target = int(rng.integers(9, 13)) if larger else int(rng.integers(5, min(8, len(rects))+1))
    while len(rects) > target:
        choices = []
        for i, first in enumerate(rects):
            for j in range(i+1, len(rects)):
                r, c, h, w = first
                rr, cc, hh, ww = rects[j]
                if r == rr and h == hh and (c+w == cc or cc+ww == c):
                    choices.append((i, j, (r, min(c, cc), h, w+ww)))
                if c == cc and w == ww and (r+h == rr or rr+hh == r):
                    choices.append((i, j, (min(r, rr), c, h+hh, w)))
        if not choices:
            break
        i, j, merged = choices[int(rng.integers(len(choices)))]
        rects = [v for k, v in enumerate(rects) if k not in (i, j)]+[merged]
    anchor = next(i for i, (r, c, _, _) in enumerate(rects) if r == c == 0)
    return rects, (int(yy[-1]), int(xx[-1])), anchor, None


def _choose_graph(rng, rects, anchor, target, witness, min_hops=2):
    contacts = _contacts(rects)
    if witness:
        required = {frozenset(e) for e in ((0, 1), (1, 2), (0, 3), (3, 4), (4, 5), (5, 2))}
        selected = [e for e in contacts if frozenset(e[:2]) in required]
        reached = set(range(6))
    else:
        selected, reached = [], {anchor}
    while len(reached) < len(rects):
        options = [e for e in contacts if (e[0] in reached) != (e[1] in reached)]
        if not options:
            raise ValueError("disconnected partition")
        e = options[int(rng.integers(len(options)))]
        selected.append(e)
        reached.update(e[:2])
    density = .18 if witness else float(rng.uniform(.15, .65))
    for e in contacts:
        if e not in selected and rng.random() < density:
            selected.append(e)
    adjacency = [[] for _ in rects]
    for a, b, *_ in selected:
        adjacency[a].append(b)
        adjacency[b].append(a)
    distance, ways = {anchor: 0}, {anchor: 1}
    queue = [anchor]
    for a in queue:
        for b in adjacency[a]:
            if b not in distance:
                distance[b], ways[b] = distance[a]+1, 0
                queue.append(b)
            if distance[b] == distance[a]+1:
                ways[b] += ways[a]
    targets = [target] if target is not None else list(range(len(rects)))
    candidates = [end for end in targets if end != anchor and
                  min_hops <= distance[end] < len(rects)-1 and ways[end] == 1]
    if candidates:
        wanted = int(rng.integers(min_hops, max(min_hops+1, len(rects)-1)))
        gap = min(abs(distance[end]-wanted) for end in candidates)
        candidates = [end for end in candidates if abs(distance[end]-wanted) == gap]
        return selected, int(rng.choice(candidates))
    raise ValueError("no unique positive route with unused tiles")


def _tile_matrices(rng, rects, edges, terminals, roles):
    _, carrier, ordinary, endpoint, _, *symbols = roles
    tiles = []
    for i, (_, _, h, w) in enumerate(rects):
        t = np.full((h, w), carrier, dtype=np.int8)
        r, c = (h-2)//2, (w-2)//2
        t[r:r+2, c:c+2] = endpoint if i in terminals else ordinary
        tiles.append(t)
    used = {"R": set(), "B": set()}
    # Repeated words on perpendicular axes are legal: facing orientation matters.
    for a, b, side, pa, pb, length in edges:
        size = int(rng.integers(3, min(4, length-2)+1))
        words = [q for q in product(symbols, repeat=size) if len(set(q)) >= 2 and q not in used[side]]
        word = words[int(rng.integers(len(words)))]
        used[side].add(word)
        offset = int(rng.integers(1, length-size))
        pa, pb = pa+offset, pb+offset
        if side == "R":
            tiles[a][pa:pa+size, -1] = word
            tiles[b][pb:pb+size, 0] = word
        else:
            tiles[a][-1, pa:pa+size] = word
            tiles[b][0, pb:pb+size] = word
    return tiles


def _scatter(rng, tiles, anchor, fixed, shape):
    remaining = sorted((i for i in range(len(tiles)) if i != anchor), key=lambda i: -tiles[i].size)
    for _ in range(16):
        placed = {anchor: fixed}
        for i in remaining:
            h, w = tiles[i].shape
            locations = [(r, c) for r in range(shape[0]-h+1) for c in range(shape[1]-w+1)]
            rng.shuffle(locations)
            for p in locations:
                # Expanded rectangles give a blank row/column, including diagonals.
                if all(not _overlap((p[0]-1, p[1]-1), (h+2, w+2), q, tiles[j].shape)
                       for j, q in placed.items()):
                    placed[i] = p
                    break
            else:
                break
        if len(placed) == len(tiles):
            return placed
    raise ValueError("bounded scatter failed")


def _features(g):
    output, d, selected = _infer(g)
    routes = _routes(d, g.shape)
    selected_area = sum(d["tiles"][n].size for n in selected[0][1:-1])
    area_witness = any(len(rr[0]) > len(selected[0]) and
                       sum(d["tiles"][n].size for n in rr[0][1:-1]) < selected_area for rr in routes)
    unequal = False
    for a, b, pa, pb in zip(selected[0], selected[0][1:], selected[1], selected[1][1:]):
        horizontal = pa[1]+d["tiles"][a].shape[1] == pb[1] or pb[1]+d["tiles"][b].shape[1] == pa[1]
        axis = 0 if horizontal else 1
        unequal |= d["tiles"][a].shape[axis] != d["tiles"][b].shape[axis]
    return dict(tiles=len(d["tiles"]), selected=len(selected[0]), routes=len(routes),
                area_witness=area_witness, subedge=unequal, corner=d["corner"],
                sizes=tuple(sorted(t.shape for t in d["tiles"])),
                degrees=tuple(sorted(len(a) for a in d["adjacency"])), shape=g.shape)


def _make_pair(rng, corner, witness=False, larger=False, need_subedge=False):
    for _ in range(100):
        try:
            rects, target_shape, anchor, target = _partition(rng, larger, witness)
            min_hops = 2 if witness else 4 if larger else 3
            edges, target = _choose_graph(rng, rects, anchor, target, witness, min_hops)
            roles = [int(x) for x in rng.choice(10, 8, replace=False)]
            tiles = _tile_matrices(rng, rects, edges, (anchor, target), roles)
            area = sum(t.size for t in tiles)
            wlow = max(target_shape[1]+2, 22, math.ceil((2*area+6)/30))
            width = int(rng.integers(wlow, 31))
            hlow = max(target_shape[0]+2, 22, math.ceil((2*area+6)/width))
            height = int(rng.integers(hlow, 31))
            shape = height, width
            origin = int(rng.integers(1, height-target_shape[0])), int(rng.integers(1, width-target_shape[1]))
            fixed = origin  # The canonical anchor is the top-left tile.
            positions = _scatter(rng, tiles, anchor, fixed, shape)
            g = np.full(shape, roles[0], dtype=np.int8)
            for i, (r, c) in positions.items():
                tile = tiles[i]
                g[r:r+tile.shape[0], c:c+tile.shape[1]] = tile
            r, c = origin
            marker_cells = ((r-1, c-1), (r-1, c), (r, c-1))
            if any(g[p] != roles[0] for p in marker_cells):
                continue
            for p in marker_cells:
                g[p] = roles[4]
            # Reflect the entire construction, never an individual moved tile.
            if corner[0]: g = g[::-1].copy()
            if corner[1]: g = g[:, ::-1].copy()
            output, data, selected = _infer(g)
            f = _features(g)
            if witness and not f["area_witness"]:
                continue
            if need_subedge and not f["subedge"]:
                continue
            if need_subedge and not _order_witness(g):
                continue
            if f["selected"] == f["tiles"]:
                continue
            # Labels are constructed from the input-derived route, then subjected
            # to conservation and anchoring checks by the public validator.
            return {"input": g, "output": output}
        except ValueError:
            continue
    raise RuntimeError("relay construction exhausted 100 bounded attempts")


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name+" must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name+" out of range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    phase = int(rng.integers(4))
    corners = ((False, False), (True, True), (False, True), (True, False))
    train_corners = (corners[phase], corners[(phase+1) % 4])
    test_corners = (corners[(phase+2) % 4], corners[(phase+3) % 4])
    regime = int(rng.integers(3))
    if regime == 1:
        test_corners = train_corners
    elif regime == 2:
        test_corners = corners
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            for _ in range(8):
                choices = train_corners if split == "train" else test_corners
                corner = choices[i % len(choices)] if split == "train" else choices[int(rng.integers(len(choices)))]
                pair = _make_pair(rng, corner, witness=(split == "train" and i == 0),
                                  larger=split == "test", need_subedge=(split == "train" and i == 1))
                key = pair["input"].shape, pair["input"].tobytes()
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not obtain distinct relay inputs")
    rng.shuffle(puzzle["train"])
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g, out = _grid(pair["input"]), _grid(pair["output"])
            key = g.shape, g.tobytes()
            assert key not in seen, "duplicate input"
            seen.add(key)
            predicted, d, route = _infer(g)
            assert out.shape == g.shape and np.array_equal(predicted, out), "incorrect relay output"
            ar, ac = d["old"][d["anchor"]]
            tile = d["tiles"][d["anchor"]]
            assert np.array_equal(out[ar:ar+tile.shape[0], ac:ac+tile.shape[1]], tile), "anchor moved"
            assert not np.any(out == d["marker"]), "marker was retained"
            assert np.all(out[tuple(d["marker_cells"].T)] == d["bg"])
            assert int(np.count_nonzero(out != d["bg"])) == sum(d["tiles"][n].size for n in route[0])
            actual = np.bincount(out.ravel(), minlength=10)
            expected = sum((np.bincount(d["tiles"][n].ravel(), minlength=10) for n in route[0]), np.zeros(10, dtype=int))
            for c in range(10):
                if c != d["bg"]:
                    assert actual[c] == expected[c], "selected tile colors were altered"
            assert 3 <= len(route[0]) < len(d["tiles"])
            assert any(d["old"][n] != p for n, p in zip(*route)), "route did not require assembly"
            features[split].append(_features(g))
    assert any(f["area_witness"] for f in features["train"]), "training lacks a count-versus-area witness"
    assert any(f["subedge"] for f in features["train"]), "training lacks unequal-side alignment"
    assert any(_order_witness(_grid(p["input"])) for p in puzzle["train"]), "training does not distinguish word order"


def _semantic_fixtures():
    """An analytically specified six-tile graph: a loop and a two-tile tail."""
    rng = np.random.default_rng(196)
    rects = [(r*6, c*6, 6, 6) for r in range(2) for c in range(3)]
    wanted = {frozenset(e) for e in ((0, 1), (1, 4), (4, 3), (3, 0), (1, 2), (2, 5))}
    edges = [e for e in _contacts(rects) if frozenset(e[:2]) in wanted]
    for target in (5, 4):
        tiles = _tile_matrices(rng, rects, edges, (0, target), list(range(8)))
        inp = np.zeros((20, 26), dtype=np.int8)
        # Scramble the source slots; the anchor remains at (1,1).
        for n, slot in enumerate((0, 5, 3, 2, 1, 4)):
            r, c = 1+(slot//3)*9, 1+(slot % 3)*8
            inp[r:r+6, c:c+6] = tiles[n]
        inp[0, 0] = inp[0, 1] = inp[1, 0] = 4
        if target == 5:
            expected = np.zeros_like(inp)
            for n in (0, 1, 2, 5):
                r, c, _, _ = rects[n]
                expected[r+1:r+7, c+1:c+7] = tiles[n]
            assert np.array_equal(solve(inp), expected), "analytic loop-and-tail regression failed"
        else:
            try:
                solve(inp)
            except ValueError as error:
                assert "tied" in str(error)
            else:
                raise AssertionError("equal-length terminal routes were accepted")


def self_test():
    """42 seeds, all legal count combinations, metamorphic and corruption checks.

    No files or traces are saved; only small sets of feature signatures survive
    each iteration. Peak returned puzzle storage is below 26 KB with int8 grids.
    """
    _semantic_fixtures()
    shapes, partitions, counts, corners, lengths, fingerprints = set(), set(), set(), set(), set(), set()
    witnesses, subedges, branches = 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(tuple(p["input"].tobytes() for s in puzzle for p in puzzle[s]))
        if seed < 3:
            again = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle for p, q in zip(puzzle[s], again[s]) for k in p)
        for split in puzzle:
            for pair in puzzle[split]:
                f = _features(pair["input"])
                shapes.add(f["shape"]); partitions.add((f["sizes"], f["degrees"]))
                counts.add(f["tiles"]); corners.add(f["corner"]); lengths.add(f["selected"])
                witnesses += f["area_witness"]; subedges += f["subedge"]
                branches += max(f["degrees"]) >= 3
                assert np.array_equal(solve(pair["input"]), pair["output"])
    assert len(fingerprints) == 42 and len(shapes) >= 15 and len(partitions) >= 90
    assert len(counts) >= 5 and len(corners) == 4 and len(lengths) >= 4
    assert witnesses >= 42 and subedges >= 42 and branches >= 42
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(404+a*10+b, a, b)
            validate_generated_puzzle(p)
            assert (len(p["train"]), len(p["test"])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (4, False), ("4", 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    puzzle = generate_puzzle(91, 2, 1)
    pair = puzzle["test"][0]
    # Every rotation/reflection and every palette permutation preserves meaning.
    permutation = np.random.default_rng(918).permutation(10)
    for k in range(4):
        for reflect in (False, True):
            g, out = np.rot90(pair["input"], k), np.rot90(pair["output"], k)
            if reflect: g, out = g[:, ::-1], out[:, ::-1]
            assert np.array_equal(solve(permutation[g]), permutation[out])
    # The same input predicts identically after unrelated generation calls.
    saved = solve(pair["input"])
    generate_puzzle(992, 2, 1)
    assert np.array_equal(solve(pair["input"]), saved)
    # Wrong labels and duplicated demonstrations must be detected.
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupted output accepted")
    damaged["train"][1] = damaged["train"][0]
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("duplicate demonstration accepted")
    return {"seeds": 42, "shapes": len(shapes), "partitions": len(partitions), "order_witness_puzzles": 42,
            "tile_counts": sorted(counts), "selected_counts": sorted(lengths),
            "area_witnesses": witnesses, "subedge_routes": subedges, "branched_networks": branches}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
