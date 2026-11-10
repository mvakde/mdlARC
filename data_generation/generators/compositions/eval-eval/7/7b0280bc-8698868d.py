"""Count-addressed routing stencils and a minimum-station cable route.

The input encodes every solving convention. Generation does not import either
parent, and solve does not read training examples, files, seeds, or generator state.
"""
from itertools import combinations
import json

import numpy as np


STEPS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # N E S W
FOREGROUNDS = tuple(range(2, 8))


def _components(mask, diagonal=False):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    steps = STEPS + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else STEPS
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                point = r + dr, c + dc
                if point in unseen:
                    unseen.remove(point)
                    cells.add(point)
                    stack.append(point)
        result.append(cells)
    return result


def _box(cells):
    r, c = zip(*cells)
    return min(r), min(c), max(r) + 1, max(c) + 1


def _perimeter(block):
    return np.concatenate((block[0], block[-1], block[1:-1, 0], block[1:-1, -1]))


def _pins(s):
    m = s // 2
    return ((0, m), (m, s - 1), (s - 1, m), (m, 0))


def _probes(s):
    m = s // 2
    return ((1, m), (m, s - 2), (s - 2, m), (m, 1))


def _labels(mask):
    labels = np.full(mask.shape, -1, dtype=int)
    components = _components(mask)
    for i, cells in enumerate(components):
        for p in cells:
            labels[p] = i
    return labels, len(components)


def _stencil(block, box=None):
    labels, count = _labels(block == 0)
    probes = tuple(int(labels[p]) for p in _probes(len(block)))
    return {"block": block.copy(), "count": count, "probes": probes, "box": box}


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(int, copy=False)


def _read_input(value):
    """Recover library, stations, midpoint ports, and cable edges from pixels."""
    grid = _grid(value)
    assert not np.any(grid == 9), "9 is reserved for the output route"
    endpoints = _components(grid == 1)
    assert len(endpoints) == 2
    end_boxes = [_box(c) for c in endpoints]
    sizes = {b[2] - b[0] for b in end_boxes}
    assert len(sizes) == 1
    t = sizes.pop()
    assert t in (7, 9)
    for cells, (r, c, rr, cc) in zip(endpoints, end_boxes):
        assert rr - r == cc - c == t and len(cells) == t * t
    s = t - 2

    cards = {}
    for cells in _components(grid != 0):
        r, c, rr, cc = _box(cells)
        if rr - r != s or cc - c != s:
            continue
        block = grid[r:rr, c:cc]
        color = int(block[0, 0])
        if color not in FOREGROUNDS:
            continue
        expected = np.full((s, s), color, dtype=int)
        for p in _pins(s):
            expected[p] = 8
        if not np.array_equal(_perimeter(block), _perimeter(expected)):
            continue
        assert set(map(int, np.unique(block[1:-1, 1:-1]))) <= {0, color}
        assert int(np.sum(block == 8)) == 4
        card = _stencil(block, (r, c, rr, cc))
        assert 0 <= card["count"] <= 4 and card["count"] not in cards
        cards[card["count"]] = card
    assert 2 <= len(cards) <= 5

    nodes = [{"box": b, "endpoint": True, "kind": None, "carrier": 1}
             for b in end_boxes]
    for color in FOREGROUNDS:
        for cells in _components(grid == color):
            r, c, rr, cc = _box(cells)
            if rr - r != t or cc - c != t:
                continue
            block = grid[r:rr, c:cc]
            assert np.all(_perimeter(block) == color)
            assert set(map(int, np.unique(block))) <= {0, color}
            holes = block == 0
            assert not np.any(holes[:-1] & holes[1:])
            assert not np.any(holes[:, :-1] & holes[:, 1:])
            k = int(holes.sum())
            assert k in cards
            assert int(cards[k]["block"][0, 0]) != color
            nodes.append({"box": (r, c, rr, cc), "endpoint": False,
                          "kind": k, "carrier": color, "clues": holes[1:-1, 1:-1]})
    nodes.sort(key=lambda node: node["box"])
    assert 4 <= len(nodes) <= 9
    assert {n["kind"] for n in nodes if not n["endpoint"]} == set(cards)
    endpoint_ids = [i for i, n in enumerate(nodes) if n["endpoint"]]
    occupied = np.zeros(grid.shape, dtype=bool)
    station_map = np.full(grid.shape, -1, dtype=int)
    boxes = [(n["box"], "station") for n in nodes]
    boxes += [(card["box"], "card") for card in cards.values()]
    for i, (box, _) in enumerate(boxes):
        r, c, rr, cc = box
        for other, _ in boxes[i + 1:]:
            a, b, aa, bb = other
            assert rr < a or aa < r or cc < b or bb < c, "objects need a black gap"
        occupied[r:rr, c:cc] = True
    for i, node in enumerate(nodes):
        r, c, rr, cc = node["box"]
        station_map[r:rr, c:cc] = i
    for card in cards.values():
        r, c, rr, cc = card["box"]
        pad = grid[max(0, r - 1):min(len(grid), rr + 1),
                   max(0, c - 1):min(grid.shape[1], cc + 1)].copy()
        pad[r - max(0, r - 1):rr - max(0, r - 1),
            c - max(0, c - 1):cc - max(0, c - 1)] = 0
        assert not np.any(pad), "stencils are separate objects, including their pins"

    cable_mask = (grid == 8) & ~occupied
    assert np.all((grid == 0) | occupied | cable_mask), "unrecognized foreground"
    edges, adjacency, used_ports = [], [[] for _ in nodes], set()
    for cells in _components(cable_mask):
        assert all(sum((r + dr, c + dc) in cells for dr, dc in STEPS) <= 2
                   for r, c in cells), "cable has an unmarked junction"
        touches = set()
        for r, c in cells:
            for dr, dc in STEPS:
                a, b = r + dr, c + dc
                if not (0 <= a < grid.shape[0] and 0 <= b < grid.shape[1]):
                    continue
                n = int(station_map[a, b])
                if n < 0:
                    continue
                r0, c0, r1, c1 = nodes[n]["box"]
                m = t // 2
                ports = ((r0, c0 + m), (r0 + m, c1 - 1),
                         (r1 - 1, c0 + m), (r0 + m, c0))
                assert (a, b) in ports, "cables attach only at side midpoints"
                touches.add((n, ports.index((a, b))))
        assert len(touches) == 2 and len({n for n, _ in touches}) == 2
        ends = tuple(sorted(touches))
        for n, side in ends:
            assert (n, side) not in used_ports
            used_ports.add((n, side))
        e = len(edges)
        edges.append({"ends": ends, "cells": cells})
        for n, side in ends:
            other, other_side = next(x for x in ends if x[0] != n)
            adjacency[n].append((other, e, side, other_side))
    assert len({tuple(n for n, _ in e["ends"]) for e in edges}) == len(edges)
    assert all(1 <= len(a) <= 4 for a in adjacency)
    reached, stack = {0}, [0]
    while stack:
        for n, _, _, _ in adjacency[stack.pop()]:
            if n not in reached:
                reached.add(n)
                stack.append(n)
    assert len(reached) == len(nodes)
    return {"grid": grid, "nodes": nodes, "cards": cards, "edges": edges,
            "adjacency": adjacency, "endpoints": endpoint_ids, "t": t}


def _routes(data, gates=True, cards=None):
    cards = data["cards"] if cards is None else cards
    start, target = data["endpoints"]
    found = []
    stack = [(start, None, [start], [])]
    while stack:
        current, entry, nodes, edges = stack.pop()
        if current == target:
            found.append((nodes, edges))
            continue
        for other, edge, exit_side, arrival_side in data["adjacency"][current]:
            if other in nodes:
                continue
            if gates and entry is not None:
                probes = cards[data["nodes"][current]["kind"]]["probes"]
                if probes[entry] < 0 or probes[entry] != probes[exit_side]:
                    continue
            stack.append((other, arrival_side, nodes + [other], edges + [edge]))
    return sorted(found, key=lambda path: (len(path[0]), path[0], path[1]))


def _unique_route(paths):
    assert paths, "no admissible endpoint route"
    assert len(paths) == 1 or len(paths[0][0]) < len(paths[1][0]), "minimum route is tied"
    return paths[0]


def _paint(data, route, cards=None):
    cards = data["cards"] if cards is None else cards
    out = data["grid"].copy()
    for n in route[0][1:-1]:
        node = data["nodes"][n]
        r, c, rr, cc = node["box"]
        out[r:rr, c:cc] = node["carrier"]
        card = cards[node["kind"]]["block"]
        out[r + 1:rr - 1, c + 1:cc - 1] = np.where(card == 0, node["carrier"], card)
    for edge in route[1]:
        for p in data["edges"][edge]["cells"]:
            out[p] = 9
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    data = _read_input(input_grid)
    return _paint(data, _unique_route(_routes(data)))


def _gate_witness(data, routes):
    raw = _routes(data, gates=False)
    return (len(raw) == 1 or len(raw[0][0]) < len(raw[1][0])) and len(raw[0][0]) < len(routes[0][0])


def _area_witness(data, routes):
    chosen = routes[0]
    area = lambda path: sum(len(data["edges"][e]["cells"]) for e in path[1])
    return any(len(path[0]) > len(chosen[0]) and area(path) < area(chosen) for path in routes)


def _address_cost_witness(data, routes):
    cost = lambda path: sum(data["nodes"][n]["kind"] for n in path[0][1:-1])
    return any(len(path[0]) > len(routes[0][0]) and cost(path) < cost(routes[0]) for path in routes)


def _area_rank_refuted(data, expected):
    ordered = sorted(data["cards"])
    areas = {k: int((card["block"] == 0).sum()) for k, card in data["cards"].items()}
    if len(set(areas.values())) != len(areas):
        return True  # Area alone cannot define the unique stencil assignment.
    assignment = dict(zip(ordered, [data["cards"][k] for k in sorted(ordered, key=areas.get)]))
    routes = _routes(data, cards=assignment)
    if not routes or (len(routes) > 1 and len(routes[0][0]) == len(routes[1][0])):
        return True
    return not np.array_equal(_paint(data, routes[0], cards=assignment), expected)


def _transform(a, index):
    return np.rot90(a, index) if index < 4 else np.fliplr(np.rot90(a, index - 4))


def _symmetry_refutations(data, expected):
    result = set()
    for index in range(1, 8):
        cards = {k: _stencil(_transform(card["block"], index)) for k, card in data["cards"].items()}
        paths = _routes(data, cards=cards)
        if not paths or (len(paths) > 1 and len(paths[0][0]) == len(paths[1][0])):
            result.add(index)
        elif not np.array_equal(_paint(data, paths[0], cards), expected):
            result.add(index)
    return result


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train, symmetry = set(), [], set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            data = _read_input(inp)
            routes = _routes(data)
            expected = _paint(data, _unique_route(routes))
            assert np.array_equal(out, expected), "incorrect route, installation, or preservation"
            assert not np.array_equal(inp, out)
            if split == "train":
                train.append((data, routes))
                symmetry.update(_symmetry_refutations(data, expected))
    assert any(_gate_witness(d, p) for d, p in train), "B's routing effect is not demonstrated"
    assert any(_area_witness(d, p) for d, p in train), "A's station cost is not demonstrated"
    assert any(_address_cost_witness(d, p) for d, p in train), "addresses could be mistaken for route costs"
    assert any(_area_rank_refuted(d, _paint(d, p[0])) for d, p in train), "area-rank matching is not refuted"
    assert symmetry == set(range(1, 8)), "visible orientation is underdetermined"
    assert any(any(a < b and int((ca["block"] == 0).sum()) > int((cb["block"] == 0).sum())
                   for a, ca in d["cards"].items() for b, cb in d["cards"].items()) for d, _ in train)
    assert any(any(len(_components(card["block"] == 0, True)) < k
                   for k, card in d["cards"].items()) for d, _ in train)
    assert any(any(int((d["cards"][d["nodes"][n]["kind"]]["block"] == 0).sum())
                       > d["nodes"][n]["kind"] for n in paths[0][0][1:-1]) for d, paths in train)
    assert any(len({int(c["block"][0, 0]) for c in d["cards"].values()}) < len(d["cards"])
               for d, _ in train), "color alone could address the library"


def _choice(rng, values):
    return values[int(rng.integers(len(values)))]


def _corner_mask(s, pairs):
    """Connect chosen neighboring compass probes around interior corners."""
    mask = np.zeros((s, s), dtype=bool)
    for p in _probes(s):
        mask[p] = True
    m = s // 2
    for corner in pairs:
        if corner == 0:  # NE
            mask[1, m:s - 1] = True
            mask[1:m + 1, s - 2] = True
        elif corner == 1:  # SE
            mask[m:s - 1, s - 2] = True
            mask[s - 2, m:s - 1] = True
        elif corner == 2:  # SW
            mask[s - 2, 1:m + 1] = True
            mask[m:s - 1, 1] = True
        else:  # NW
            mask[1:m + 1, 1] = True
            mask[1, 1:m + 1] = True
    return mask


def _mask(rng, s, k, mode="random"):
    if k == 0:
        return np.zeros((s, s), dtype=bool)
    if mode == "gate":
        if k == 1:
            return np.pad(np.ones((s - 2, s - 2), dtype=bool), 1)
        mask = _corner_mask(s, (0, 2) if k == 2 else (0,) if k == 3 else ())
        return _transform(mask, int(rng.integers(8))).copy()
    if mode == "area" and k == 3:
        return _transform(_corner_mask(s, (0,)), int(rng.integers(8))).copy()
    if k == 1:
        if mode not in ("fallback_area", "area") and rng.random() < .25:
            # A single chamber need not reach every pin. This varies gate
            # behavior independently of the stencil's numerical address.
            for _ in range(64):
                trial = np.pad(rng.random((s - 2, s - 2)) < rng.uniform(.3, .8), 1)
                if len(_components(trial)) == 1 and sum(bool(trial[p]) for p in _probes(s)) >= 2:
                    return trial
        mask = np.zeros((s, s), dtype=bool)
        m = s // 2
        mask[1:-1, m] = True
        mask[m, 1:-1] = True
        if mode == "fallback_area":
            mask[1, 1] = True  # NW; distinguishes the gate fallback's remaining symmetry.
            return mask
        wanted = int(rng.integers(int(mask.sum()), (s - 2) ** 2 + 1))
        for _ in range((s - 2) ** 2):
            choices = [(r, c) for r in range(1, s - 1) for c in range(1, s - 1)
                       if not mask[r, c] and any(mask[r + dr, c + dc] for dr, dc in STEPS)]
            if int(mask.sum()) >= wanted or not choices:
                break
            mask[_choice(rng, choices)] = True
        return mask
    for _ in range(100):
        mask = np.pad(rng.random((s - 2, s - 2)) < rng.uniform(.18, .72), 1)
        if len(_components(mask)) == k:
            return mask
    return _transform(_corner_mask(s, (0, 2) if k == 2 else (0,) if k == 3 else ()),
                      int(rng.integers(8))).copy()


def _library(rng, t, counts, mode="random", fixed=False):
    s = t - 2
    pool = list(map(int, rng.choice(FOREGROUNDS, int(rng.integers(2, 7)), replace=False)))
    shared = _choice(rng, pool)
    cards = {}
    for k in counts:
        foreground = shared if mode == "gate" or fixed else _choice(rng, pool)
        mask = _mask(rng, s, k, mode)
        if fixed and k in (2, 3, 4):
            mask = _corner_mask(s, (0, 2) if k == 2 else (0,) if k == 3 else ())
        block = np.full((s, s), foreground, dtype=int)
        block[mask] = 0
        for p in _pins(s):
            block[p] = 8
        cards[k] = _stencil(block)
        assert cards[k]["count"] == k
    return cards, pool


def _axis(rng, n, t, maximum):
    minimum = n * t + n - 1
    slack = int(rng.integers(maximum - minimum + 1))
    gaps = np.ones(n - 1, dtype=int) + rng.multinomial(slack, [1 / (n - 1)] * (n - 1))
    return [0] + list(map(int, np.cumsum(t + gaps)))


def _line(a, b):
    assert a[0] == b[0] or a[1] == b[1]
    dr, dc = np.sign(b[0] - a[0]), np.sign(b[1] - a[1])
    return {(a[0] + int(dr) * i, a[1] + int(dc) * i)
            for i in range(abs(b[0] - a[0]) + abs(b[1] - a[1]) + 1)}


def _raw_graph(rng, nr, nc, purpose):
    vertices = [(r, c) for r in range(nr) for c in range(nc)]
    candidates = [(a, b) for a, b in combinations(vertices, 2)
                  if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1]
    target = len(vertices) if purpose in ("area", "small") else int(rng.integers(max(5, len(vertices) - 2), len(vertices) + 1))
    active, edges = {vertices[int(rng.integers(len(vertices)))]}, []
    while len(active) < target:
        options = [e for e in candidates if (e[0] in active) != (e[1] in active)]
        e = _choice(rng, options)
        edges.append(e)
        active.update(e)
    extras = [e for e in candidates if e not in edges and set(e) <= active]
    rng.shuffle(extras)
    probability = .85 if purpose == "area" else .70
    edges += [e for e in extras if rng.random() < probability]
    degree = {v: sum(v in e for e in edges) for v in active}
    visible = {v for v in active if degree[v] != 2 or rng.random() > .40}
    minimum = 4 if purpose == "small" else 5
    for v in rng.permutation(len(vertices)):
        if len(visible) >= minimum:
            break
        if vertices[int(v)] in active:
            visible.add(vertices[int(v)])
    return active, edges, visible


def _embedding(rng, t, nr, nc, raw_edges, visible):
    # A full side band always fits the largest template; the other axis can stretch.
    rows = _axis(rng, nr, t, 30 - (t - 2) - 1)
    cols = _axis(rng, nc, t, 30)
    m = t // 2
    centers = {(r, c): (rows[r] + m, cols[c] + m) for r in range(nr) for c in range(nc)}
    raw_adjacency = {v: [] for e in raw_edges for v in e}
    pixels = {}
    for a, b in raw_edges:
        raw_adjacency[a].append(b)
        raw_adjacency[b].append(a)
        pa, pb = centers[a], centers[b]
        points = [pa, pb]
        # Safe one-cell detours only in wide corridors, beyond midpoint contacts.
        horizontal = a[0] == b[0]
        gap = (abs(pb[1] - pa[1]) if horizontal else abs(pb[0] - pa[0])) - t
        if gap >= 5 and rng.random() < .35:
            if pa > pb:
                pa, pb = pb, pa
            shift = _choice(rng, [-1, 1])
            if horizontal:
                x, y = pa[1] + m + 2, pb[1] - m - 2
                points = [pa, (pa[0], x), (pa[0] + shift, x),
                          (pb[0] + shift, y), (pb[0], y), pb]
            else:
                x, y = pa[0] + m + 2, pb[0] - m - 2
                points = [pa, (x, pa[1]), (x, pa[1] + shift),
                          (y, pb[1] + shift), (y, pb[1]), pb]
        pixels[frozenset((a, b))] = set().union(*(_line(p, q) for p, q in zip(points, points[1:])))
    ordered = sorted(visible)
    ids = {v: i for i, v in enumerate(ordered)}
    boxes = [(rows[v[0]], cols[v[1]], rows[v[0]] + t, cols[v[1]] + t) for v in ordered]
    node_cells = set().union(*({(r, c) for r in range(a, aa) for c in range(b, bb)}
                              for a, b, aa, bb in boxes))
    adjacency, edges, used = [[] for _ in ordered], [], set()
    for a in ordered:
        for nxt in raw_adjacency[a]:
            chain, previous, current = [frozenset((a, nxt))], a, nxt
            first = nxt
            while current not in visible:
                choices = [v for v in raw_adjacency[current] if v != previous]
                assert len(choices) == 1
                following = choices[0]
                chain.append(frozenset((current, following)))
                previous, current = current, following
            if chain[0] in used:
                continue
            assert a != current
            used.update(chain)
            side_a = STEPS.index((first[0] - a[0], first[1] - a[1]))
            side_b = STEPS.index((previous[0] - current[0], previous[1] - current[1]))
            ends = ((ids[a], side_a), (ids[current], side_b))
            cells = set().union(*(pixels[e] for e in chain)) - node_cells
            edge = len(edges)
            edges.append({"ends": ends, "cells": cells})
            adjacency[ids[a]].append((ids[current], edge, side_a, side_b))
            adjacency[ids[current]].append((ids[a], edge, side_b, side_a))
    assert len({tuple(sorted(n for n, _ in e["ends"])) for e in edges}) == len(edges)
    height = max(b[2] for b in boxes)
    width = max(b[3] for b in boxes)
    all_pixels = set().union(*(e["cells"] for e in edges))
    height = max(height, max(r for r, _ in all_pixels) + 1)
    width = max(width, max(c for _, c in all_pixels) + 1)
    return {"nodes": [{"box": b} for b in boxes], "edges": edges, "adjacency": adjacency,
            "t": t, "shape": (height, width)}, ids


def _clues(rng, size, k):
    mask = np.zeros((size, size), dtype=bool)
    for index in rng.permutation(size * size):
        r, c = divmod(int(index), size)
        if any(0 <= r + dr < size and 0 <= c + dc < size and mask[r + dr, c + dc] for dr, dc in STEPS):
            continue
        if int(mask.sum()) == k:
            break
        mask[r, c] = True
    assert int(mask.sum()) == k
    return mask


def _draw_network(rng, data, cards, pool, endpoints, kinds, route):
    grid = np.zeros(data["shape"], dtype=int)
    for e in data["edges"]:
        for p in e["cells"]:
            grid[p] = 8
    for i, node in enumerate(data["nodes"]):
        r, c, rr, cc = node["box"]
        node["endpoint"] = i in endpoints
        node["kind"] = None if i in endpoints else kinds[i]
        if i in endpoints:
            node["carrier"] = 1
            grid[r:rr, c:cc] = 1
        else:
            fg = int(cards[kinds[i]]["block"][0, 0])
            carrier = int(_choice(rng, [p for p in pool if p != fg]))
            node["carrier"] = carrier
            grid[r:rr, c:cc] = carrier
            grid[r + 1:rr - 1, c + 1:cc - 1][_clues(rng, data["t"] - 2, kinds[i])] = 0
    data["grid"], data["cards"], data["endpoints"] = grid, cards, list(endpoints)
    return grid, _paint(data, route)


def _free_origins(boxes, s):
    r, c = np.indices((31 - s, 31 - s))
    free = np.ones(r.shape, dtype=bool)
    for a, b, aa, bb in boxes:
        free &= (r + s < a) | (r > aa) | (c + s < b) | (c > bb)
    return np.argwhere(free)


def _pack(rng, inp, out, cards):
    h, w = inp.shape
    s = len(next(iter(cards.values()))["block"])
    objects = [cards[int(k)]["block"] for k in rng.permutation(list(cards))]
    boxes = None
    for _ in range(24):
        r, c = int(rng.integers(31 - h)), int(rng.integers(31 - w))
        trial = [(r, c, r + h, c + w)]
        for obj in objects:
            free = _free_origins(trial, s)
            if not len(free):
                break
            a, b = map(int, free[int(rng.integers(len(free)))])
            trial.append((a, b, a + s, b + s))
        if len(trial) == len(objects) + 1:
            boxes = trial
            break
    if boxes is None:
        assert h + 1 + s <= 30 and len(objects) * s + len(objects) - 1 <= 30
        boxes = [(0, 0, h, w)] + [(h + 1, i * (s + 1), h + 1 + s, i * (s + 1) + s)
                                         for i in range(len(objects))]
    a, b = np.zeros((30, 30), dtype=int), np.zeros((30, 30), dtype=int)
    r, c, rr, cc = boxes[0]
    a[r:rr, c:cc], b[r:rr, c:cc] = inp, out
    for (r, c, rr, cc), obj in zip(boxes[1:], objects):
        a[r:rr, c:cc] = obj
        b[r:rr, c:cc] = obj
    low_r, low_c = min(x[0] for x in boxes), min(x[1] for x in boxes)
    high_r, high_c = max(x[2] for x in boxes), max(x[3] for x in boxes)
    low_r, low_c = int(rng.integers(low_r + 1)), int(rng.integers(low_c + 1))
    high_r, high_c = int(rng.integers(high_r, 31)), int(rng.integers(high_c, 31))
    symmetry = int(rng.integers(8))
    return {"input": _transform(a[low_r:high_r, low_c:high_c], symmetry).copy(),
            "output": _transform(b[low_r:high_r, low_c:high_c], symmetry).copy()}


def _fallback(rng, purpose, new_count=4):
    t = 9 if purpose == "newscale" else 7
    counts = [1, 2, new_count] if purpose == "newcount" else [1, 2] if purpose == "small" else [1, 2, 3]
    mode = "fallback_area" if purpose == "area" else "gate"
    cards, pool = _library(rng, t, counts, mode, fixed=True)
    if purpose == "area":
        cards[3] = _stencil(_transform(cards[3]["block"], 2))  # SW joins the long sparse route.
    if purpose == "small":
        nr, nc = 2, 2
        vertices = {(r, c) for r in range(2) for c in range(2)}
        raw = [(a, b) for a, b in combinations(sorted(vertices), 2)
               if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1]
        data, ids = _embedding(rng, t, nr, nc, raw, vertices)
        cards[2] = _stencil(_transform(cards[2]["block"], 1))  # blocks W-to-S.
        endpoints = (ids[(0, 0)], ids[(1, 1)])
        kinds = {ids[(1, 0)]: 1, ids[(0, 1)]: 2}
    else:
        nr, nc = (3, 3) if purpose == "area" else (2, 3)
        active = {(r, c) for r in (0, 1) for c in range(3)}
        raw = [((r, c), (r, c + 1)) for r in (0, 1) for c in range(2)]
        raw += [((0, 0), (1, 0)), ((0, 2), (1, 2))]
        if purpose == "area":
            active.update(((2, 0), (2, 1)))
            raw += [((1, 0), (2, 0)), ((1, 1), (2, 1))]
        visible = active - {(0, 1)}
        data, ids = _embedding(rng, t, nr, nc, raw, visible)
        endpoints = (ids[(0, 0)], ids[(1, 2)])
        if purpose == "area":
            kinds = {ids[v]: 1 for v in visible if ids[v] not in endpoints}
            kinds[ids[(0, 2)]], kinds[ids[(2, 0)]] = 3, 2
        else:
            kinds = {ids[(0, 2)]: new_count if purpose == "newcount" else 3,
                     ids[(1, 0)]: 2, ids[(1, 1)]: 1}
    for i, node in enumerate(data["nodes"]):
        node["kind"] = kinds.get(i)
    data["cards"], data["endpoints"] = cards, endpoints
    route = _unique_route(_routes(data))
    inp, out = _draw_network(rng, data, cards, pool, endpoints, kinds, route)
    return _pack(rng, inp, out, cards)


def _make_pair(rng, purpose, allowed_sizes=(7, 9), new_count=4, allowed_counts=(0, 1, 2, 3, 4)):
    for _ in range(100):
        t = 7 if purpose in ("gate", "area") else 9 if purpose == "newscale" else _choice(rng, allowed_sizes)
        if purpose == "small":
            nr, nc = 2, 2
            counts = [1, _choice(rng, [2, 3])]
        else:
            nr, nc = (3, 3) if purpose == "area" else ((2, 3) if t == 9 else _choice(rng, [(2, 3), (3, 3)]))
            if purpose in ("gate", "area"):
                counts = [1, 2, 3]
            elif purpose == "newcount":
                counts = sorted([1, 2, new_count] + ([3] if t == 7 and rng.random() < .35 else []))
            else:
                pool = [k for k in allowed_counts if k != 1]
                maximum = min(5 if t == 7 else 3, len(pool) + 1)
                number = int(rng.integers(2, maximum + 1))
                counts = sorted([1] + list(map(int, rng.choice(pool, number - 1, replace=False))))
            if t == 7 and len(counts) == 5:
                nr, nc = 3, 3
        cards, pool = _library(rng, t, counts, purpose if purpose in ("gate", "area") else "random")
        active, raw, visible = _raw_graph(rng, nr, nc, purpose)
        if len(visible) < len(counts) + 2:
            continue
        try:
            data, _ = _embedding(rng, t, nr, nc, raw, visible)
        except AssertionError:
            continue
        n = len(data["nodes"])
        for _ in range(4):
            probabilities = [0.55 if k == 1 else .45 / (len(counts) - 1) for k in counts]
            kinds = list(map(int, rng.choice(counts, n, p=probabilities)))
            for k, i in zip(counts, rng.permutation(n)):
                kinds[int(i)] = k
            for i, node in enumerate(data["nodes"]):
                node["kind"] = kinds[i]
            data["cards"] = cards
            endpoint_pairs = list(combinations(range(n), 2))
            rng.shuffle(endpoint_pairs)
            for endpoints in endpoint_pairs:
                if {kinds[i] for i in range(n) if i not in endpoints} != set(counts):
                    continue
                data["endpoints"] = endpoints
                routes = _routes(data)
                if not routes or (len(routes) > 1 and len(routes[0][0]) == len(routes[1][0])):
                    continue
                if len(routes[0][0]) == 2 and (purpose != "free" or rng.random() < .9):
                    continue
                if purpose == "gate" and not _gate_witness(data, routes):
                    continue
                if purpose == "area" and not _area_witness(data, routes):
                    continue
                if purpose == "area" and not _address_cost_witness(data, routes):
                    continue
                inp, out = _draw_network(rng, data, cards, pool, endpoints, kinds, routes[0])
                pair = _pack(rng, inp, out, cards)
                try:
                    recovered = _read_input(pair["input"])
                    paths = _routes(recovered)
                    assert np.array_equal(_paint(recovered, _unique_route(paths)), pair["output"])
                    assert purpose != "gate" or _gate_witness(recovered, paths)
                    assert purpose != "area" or _area_witness(recovered, paths)
                    assert purpose != "area" or _address_cost_witness(recovered, paths)
                except AssertionError:
                    break
                return pair
            else:
                continue
            break
    return _fallback(rng, purpose, new_count)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime, new_count = int(rng.integers(3)), _choice(rng, [0, 4])
    puzzle, seen = {"train": [], "test": []}, set()
    # The same two mandatory mechanisms survive every legal requested count.
    first = _make_pair(rng, "gate")
    first_data = _read_input(first["input"])
    missing = set(range(1, 8)) - _symmetry_refutations(first_data, first["output"])
    first_rank_refuted = _area_rank_refuted(first_data, first["output"])
    for _ in range(16):
        second = _make_pair(rng, "area")
        second_data = _read_input(second["input"])
        if (missing <= _symmetry_refutations(second_data, second["output"])
                and (first_rank_refuted or _area_rank_refuted(second_data, second["output"]))):
            break
    else:
        # The gate construction leaves one diagonal reflection undecided;
        # the area construction's corner resolves it in a suitable orientation.
        first, base_second = _fallback(rng, "gate"), _fallback(rng, "area")
        first_refuted = _symmetry_refutations(_read_input(first["input"]), first["output"])
        for index in range(8):
            second = {k: _transform(v, index).copy() for k, v in base_second.items()}
            refuted = first_refuted | _symmetry_refutations(_read_input(second["input"]), second["output"])
            if refuted == set(range(1, 8)):
                break
        else:
            raise AssertionError("orientation witness construction exhausted")
    puzzle["train"].extend((first, second))
    seen.update(_key(p["input"]) for p in puzzle["train"])
    for split, number in (("train", num_train - 2), ("test", num_test)):
        for _ in range(number):
            purpose = "free" if split == "train" else ("newscale", "newcount", "small")[regime]
            for _ in range(20):
                pair = _make_pair(rng, purpose, allowed_sizes=(7,) if regime == 0 else (7, 9),
                                  new_count=new_count,
                                  allowed_counts=tuple(k for k in range(5) if regime != 1 or split == "test" or k != new_count))
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                # Thirty injective recolorings of two present ordinary colors
                # exceed the at-most-thirteen previous inputs. Markers stay fixed.
                present = [c for c in FOREGROUNDS if np.any(pair["input"] == c)]
                assert len(present) >= 2
                order = present[:2] + [c for c in FOREGROUNDS if c not in present[:2]]
                for a in FOREGROUNDS:
                    for b in FOREGROUNDS:
                        if a == b:
                            continue
                        target = [a, b] + [c for c in FOREGROUNDS if c not in (a, b)]
                        lookup = np.arange(10)
                        lookup[order] = target
                        recolored = {k: lookup[v] for k, v in pair.items()}
                        key = _key(recolored["input"])
                        if key not in seen:
                            break
                    else:
                        continue
                    break
                else:
                    raise AssertionError("injective recoloring bound violated")
                seen.add(key)
                puzzle[split].append(recolored)
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test():
    shapes, topologies, palettes, masks, normalized, sizes, counts, shifts = set(), set(), set(), set(), set(), set(), set(), set()
    totals = {"pairs": 0, "gate_witnesses": 0, "area_witnesses": 0,
              "address_cost_witnesses": 0, "branches": 0, "multiple_routes": 0, "zero_cards": 0, "four_cards": 0}
    signatures = set()
    gate_partitions = {k: set() for k in range(5)}
    library_sizes, station_counts, selected_counts = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p["input"]) for s in puzzle for p in puzzle[s]))
        analyses = {s: [_read_input(p["input"]) for p in puzzle[s]] for s in puzzle}
        train_sizes = {d["t"] for d in analyses["train"]}
        train_counts = set().union(*(set(d["cards"]) for d in analyses["train"]))
        train_min = min(len(d["nodes"]) for d in analyses["train"])
        for d in analyses["test"]:
            if d["t"] not in train_sizes:
                shifts.add("scale")
            if set(d["cards"]) - train_counts:
                shifts.add("count")
            if len(d["nodes"]) < train_min:
                shifts.add("smaller")
        assert any(d["t"] not in train_sizes or bool(set(d["cards"]) - train_counts)
                   or len(d["nodes"]) < train_min for d in analyses["test"])
        for split in puzzle:
            for pair, d in zip(puzzle[split], analyses[split]):
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                paths = _routes(d)
                totals["pairs"] += 1
                totals["gate_witnesses"] += _gate_witness(d, paths)
                totals["area_witnesses"] += _area_witness(d, paths)
                totals["address_cost_witnesses"] += _address_cost_witness(d, paths)
                totals["branches"] += any(len(a) >= 3 for a in d["adjacency"])
                totals["multiple_routes"] += len(paths) > 1
                totals["zero_cards"] += 0 in d["cards"]
                totals["four_cards"] += 4 in d["cards"]
                shapes.add(pair["input"].shape)
                sizes.add(d["t"])
                counts.add(tuple(sorted(d["cards"])))
                library_sizes.add(len(d["cards"]))
                station_counts.add(len(d["nodes"]))
                selected_counts.add(len(paths[0][0]) - 2)
                degrees = tuple(sorted(len(a) for a in d["adjacency"]))
                topologies.add((len(d["nodes"]), len(d["edges"]), degrees,
                                tuple(len(p[0]) - 2 for p in paths)))
                palettes.add(tuple(map(int, np.unique(pair["input"]))))
                roles = {}
                normalized.add((pair["input"].shape, tuple(roles.setdefault(int(c), len(roles)) for c in pair["input"].flat)))
                for card in d["cards"].values():
                    mask = card["block"] == 0
                    masks.add((mask.shape, min(_transform(mask, i).tobytes() for i in range(8))))
                    names = {}
                    gate_partitions[card["count"]].add(tuple(-1 if label < 0 else names.setdefault(label, len(names))
                                                             for label in card["probes"]))
    assert len(signatures) == 42 and len(normalized) >= 180
    assert len(shapes) >= 20 and len(topologies) >= 20 and len(masks) >= 35
    assert sizes == {7, 9} and len(counts) >= 5 and len(palettes) >= 10
    assert shifts == {"scale", "count", "smaller"}
    assert totals["gate_witnesses"] >= 42 and totals["area_witnesses"] >= 42
    assert totals["address_cost_witnesses"] >= 42
    assert totals["branches"] >= 20 and totals["zero_cards"] and totals["four_cards"]
    assert library_sizes == {2, 3, 4, 5} and len(station_counts) >= 5
    assert len(gate_partitions[1]) >= 2 and len(gate_partitions[2]) >= 3 and len(gate_partitions[3]) >= 3
    for seed, nt, ne in ((0, 2, 1), (9, 10, 4), (27, 7, 3)):
        a, b = generate_puzzle(seed, nt, ne), generate_puzzle(seed, nt, ne)
        assert all(np.array_equal(p[k], q[k]) for s in a for p, q in zip(a[s], b[s]) for k in p)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Equivariance checks change the input geometry, not latent parameters.
    example = generate_puzzle(33, 2, 1)["test"][0]
    for i in range(8):
        assert np.array_equal(solve(_transform(example["input"], i)), _transform(example["output"], i))
    # A changed output pixel must be detected, including off-route preservation.
    broken = generate_puzzle(7, 2, 1)
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    malformed = example["input"].copy()
    card = next(iter(_read_input(malformed)["cards"].values()))
    r, c, _, _ = card["box"]
    pr, pc = _pins(len(card["block"]))[0]
    malformed[r + pr, c + pc] = int(card["block"][0, 0])
    try:
        solve(malformed)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing stencil pin accepted")
    # Construct an even cycle with two equal-cost admissible routes. Its two
    # off-cycle leaves carry the other stencil addresses, so parsing remains valid.
    rng = np.random.default_rng(991)
    raw = [((r, c), (r, c + 1)) for r in (0, 1) for c in range(2)]
    raw += [((0, 0), (1, 0)), ((0, 2), (1, 2)), ((1, 0), (2, 0)), ((1, 1), (2, 1))]
    visible = {v for edge in raw for v in edge}
    data, ids = _embedding(rng, 7, 3, 3, raw, visible)
    cards, pool = _library(rng, 7, [1, 2, 3], "gate", fixed=True)
    endpoints = (ids[(0, 0)], ids[(1, 2)])
    kinds = {ids[v]: 1 for v in visible if ids[v] not in endpoints}
    kinds[ids[(2, 0)]], kinds[ids[(2, 1)]] = 2, 3
    for i, node in enumerate(data["nodes"]):
        node["kind"] = kinds.get(i)
    data["cards"], data["endpoints"] = cards, endpoints
    routes = _routes(data)
    assert len(routes[0][0]) == len(routes[1][0])
    inp, out = _draw_network(rng, data, cards, pool, endpoints, kinds, routes[0])
    tied = _pack(rng, inp, out, cards)["input"]
    _read_input(tied)  # It is structurally legitimate; only the route is ambiguous.
    try:
        solve(tied)
    except AssertionError as error:
        assert "tied" in str(error)
    else:
        raise AssertionError("tied route accepted")
    # Correct individual labels are insufficient if a training set omits a
    # parent's actual routing mechanism. Test both omissions explicitly.
    for purpose, message in (("area", "B's routing effect"), ("gate", "A's station cost")):
        insufficient = {"train": [_fallback(rng, purpose), _fallback(rng, purpose)],
                        "test": [example]}
        try:
            validate_generated_puzzle(insufficient)
        except AssertionError as error:
            assert message in str(error)
        else:
            raise AssertionError("training without a parent witness accepted")
    return {"seeds": 42, **totals, "input_shapes": len(shapes), "topology_signatures": len(topologies),
            "cavity_shapes_up_to_symmetry": len(masks), "color_normalized_inputs": len(normalized),
            "count_sets": len(counts), "palettes": len(palettes), "test_shifts": sorted(shifts),
            "library_sizes": sorted(library_sizes), "station_counts": sorted(station_counts),
            "selected_station_counts": sorted(selected_counts),
            "port_partitions_per_cavity_count": {k: len(v) for k, v in gate_partitions.items()}}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
