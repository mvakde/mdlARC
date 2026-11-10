"""Itinerary relay: a color-word walk in a locally color-directed rail network.

The solver reads only pixels. Construction uses a small embedded tree and its
known itinerary to place the answer; validation reconstructs the network afresh.
No files, execution histories, parent generators, or persistent caches are used.
"""
from hashlib import sha256
import json

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _need(bool(np.all((a >= 0) & (a <= 9))), "ARC colors required")
    return a


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(g, p):
    return 0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1]


def _signature(g):
    return g.shape, np.asarray(g, dtype=np.int8).tobytes()


def _parse(g):
    """Recognize square label frames and their sockets, then the bottom word."""
    g = _grid(g)
    bg = int(g[0, 0])
    _need(all(np.all(edge == bg) for edge in (g[0], g[-1], g[:, 0], g[:, -1])),
          "a clear background border is required")
    occupied_rows = np.flatnonzero(np.any(g != bg, axis=1))
    _need(len(occupied_rows) > 0, "empty scene")
    key_y = int(occupied_rows[-1])
    key_xs = np.flatnonzero(g[key_y] != bg)
    _need(3 <= len(key_xs) <= 12 and np.all(np.diff(key_xs) == 2), "isolated itinerary")
    _need(key_y >= 3 and np.all(g[key_y - 1] == bg), "word clearance")
    key = tuple(map(int, g[key_y, key_xs]))
    stations = []
    mask = np.full(g.shape, -1, dtype=np.int16)
    for size in (3, 5, 7):
        radius = size // 2
        for y in range(1, key_y - size):
            for x in range(1, g.shape[1] - size):
                label = int(g[y, x])
                if label == bg:
                    continue
                block = g[y:y + size, x:x + size]
                if not (np.all(block[0] == label) and np.all(block[-1] == label)
                        and np.all(block[:, 0] == label) and np.all(block[:, -1] == label)):
                    continue
                interior = block[1:-1, 1:-1].copy()
                center = int(block[radius, radius])
                interior[radius - 1, radius - 1] = bg
                if not np.all(interior == bg) or center == label:
                    continue
                _need(np.all(mask[y:y + size, x:x + size] == -1), "overlapping stations")
                index = len(stations)
                mask[y:y + size, x:x + size] = index
                stations.append(dict(center=(y + radius, x + radius), size=size,
                                     radius=radius, label=label, occupant=center))
    _need(len(stations) >= 3, "at least three square stations")
    loaded = [i for i, s in enumerate(stations) if s["occupant"] != bg]
    _need(len(loaded) == 1, "one occupied socket")
    start = loaded[0]
    token = stations[start]["occupant"]
    _need(np.count_nonzero(g == token) == 1 and token not in key,
          "the token color occurs only in its socket")
    _need(all(s["label"] not in (bg, token) for s in stations), "distinct color roles")
    _need(stations[start]["label"] == key[0], "word includes the starting station")
    _need(len(set(key)) >= 2, "nonconstant itinerary")
    return dict(grid=g, background=bg, token=token, stations=stations,
                station_mask=mask, start=start, key=key, key_y=key_y)


def _flanks(a, p, d):
    g, bg, token = a["grid"], a["background"], a["token"]
    normal = (-d[1], d[0])
    q, r = _add(p, normal), _add(p, (-normal[0], -normal[1]))
    if not (_inside(g, q) and _inside(g, r)):
        return None
    c, e = int(g[q]), int(g[r])
    if c in (bg, token) or e in (bg, token):
        return None
    return c, e


def _trace(a, source, d):
    """Trace physical geometry, retaining rail evidence separately from validity.

    Motion keeps its direction until blocked. At a blockage a unique orthogonal
    continuation must have two visible flanks. Colors are evaluated afterward,
    so the same geometry reader can expose convincing mismatched decoy tracks.
    """
    g, bg, mask = a["grid"], a["background"], a["station_mask"]
    station = a["stations"][source]
    cy, cx = station["center"]
    radius = station["radius"]
    p = cy + (radius + 1) * d[0], cx + (radius + 1) * d[1]
    if not _inside(g, p) or int(g[p]) != bg or _flanks(a, p, d) is None:
        return None
    path, supports, seen, turns = [], [], set(), 0
    while True:
        _need(p not in seen, "a rail channel must not cycle or revisit")
        seen.add(p)
        path.append(p)
        pair = _flanks(a, p, d)
        if pair is not None:
            supports.append(pair)
        nxt = _add(p, d)
        _need(_inside(g, nxt) and nxt[0] < a["key_y"] - 1, "channel runs out of network")
        target = int(mask[nxt])
        if target >= 0:
            dest = a["stations"][target]
            midpoint = (dest["center"][0] - dest["radius"] * d[0],
                        dest["center"][1] - dest["radius"] * d[1])
            _need(target != source and nxt == midpoint, "channel must meet a side midpoint")
            _need(bool(supports), "channel needs rail evidence")
            return dict(source=source, target=target, path=tuple(path),
                        supports=tuple(supports), turns=turns, direction=d)
        if int(g[nxt]) == bg:
            p = nxt
            continue
        choices = []
        for e in ((-d[1], d[0]), (d[1], -d[0])):
            q = _add(p, e)
            if _inside(g, q) and int(g[q]) == bg and _flanks(a, q, e) is not None:
                choices.append(e)
        _need(len(choices) == 1, "one visible turn is required inside a channel")
        d = choices[0]
        p = _add(p, d)
        turns += 1


def _network(a, mode="normal"):
    graph = [[] for _ in a["stations"]]
    tracks = {}
    for i, s in enumerate(a["stations"]):
        for d in DIRS:
            track = _trace(a, i, d)
            if track is None:
                continue
            j = track["target"]
            _need((i, j) not in tracks, "parallel tracks between a station pair are excluded")
            tracks[i, j] = track
            supports = track["supports"]
            if mode == "normal":
                active = all(c == e == s["label"] for c, e in supports)
            elif mode == "ignore_matching":
                # Still require one source-colored side; relax paired equality.
                active = all(s["label"] in (c, e) for c, e in supports)
            elif mode == "ignore_source":
                active = all(c == e for c, e in supports)
            else:
                raise ValueError("unknown rail ablation")
            if active:
                graph[i].append(j)
    for (i, j), track in tracks.items():
        _need((j, i) in tracks and tracks[j, i]["path"] == track["path"][::-1],
              "rail geometry must connect the same stations in both directions")
    return graph, tracks


def _word_paths(a, graph, limit=64):
    answers = []
    key, stations = a["key"], a["stations"]
    def walk(path):
        if len(answers) >= limit:
            return
        if len(path) == len(key):
            answers.append(tuple(path))
            return
        for j in graph[path[-1]]:
            if j not in path and stations[j]["label"] == key[len(path)]:
                walk(path + [j])
    walk([a["start"]])
    return answers


def _analyse(input_grid):
    a = _parse(np.asarray(input_grid))
    graph, tracks = _network(a)
    answers = _word_paths(a, graph, limit=2)
    _need(len(answers) == 1, "the itinerary must select exactly one distinct-station walk")
    a.update(graph=graph, tracks=tracks, route=answers[0])
    return a


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Move the single token along the unique visible word-compatible rail walk."""
    a = _analyse(input_grid)
    out = a["grid"].copy()
    out[a["stations"][a["start"]]["center"]] = a["background"]
    out[a["stations"][a["route"][-1]]["center"]] = a["token"]
    return out


def _tree(rng, rows, cols):
    vertices = [(r, c) for r in range(rows) for c in range(cols)]
    edges = [(p, (p[0] + dr, p[1] + dc)) for p in vertices
             for dr, dc in ((0, 1), (1, 0))
             if p[0] + dr < rows and p[1] + dc < cols]
    parents = {p: p for p in vertices}
    def root(p):
        while parents[p] != p:
            parents[p] = parents[parents[p]]
            p = parents[p]
        return p
    graph = {p: [] for p in vertices}
    for ix in rng.permutation(len(edges)):
        p, q = edges[int(ix)]
        a, b = root(p), root(q)
        if a != b:
            parents[a] = b
            graph[p].append(q)
            graph[q].append(p)
    return graph


def _suppress(rng, tree, keep_probability):
    """Degree-two vacancies become true bends or longer links, never switches."""
    nodes = sorted(p for p in tree if len(tree[p]) != 2 or rng.random() < keep_probability)
    node_set = set(nodes)
    paths = {}
    graph = {p: [] for p in nodes}
    for p in nodes:
        for q in tree[p]:
            order = [p, q]
            while order[-1] not in node_set:
                order.append(next(t for t in tree[order[-1]] if t != order[-2]))
            target = order[-1]
            graph[p].append(target)
            if p < target:
                paths[p, target] = order
    return nodes, graph, paths


def _candidates(graph, length, same_depth=False):
    """A final fork supplies a rail decoy; an earlier fork supplies a word decoy."""
    answers = []
    def walk(path):
        if len(path) == length:
            u = path[-2]
            wrong_ends = [w for w in graph[u] if w not in path]
            early = []
            def branch(k, off_path):
                hops = k + len(off_path) - 1
                if not same_depth or hops == length - 1:
                    early.append((k, tuple(off_path)))
                if same_depth and hops >= length - 1:
                    return
                for q in graph[off_path[-1]]:
                    if q not in path and q not in off_path:
                        branch(k, off_path + [q])
            for k in range(length - 2):
                for q in graph[path[k]]:
                    if q not in path:
                        branch(k, [path[k], q])
            for w in wrong_ends:
                for k, off_path in early:
                    if w not in off_path:
                        answers.append((tuple(path), w, k, off_path))
            return
        for q in graph[path[-1]]:
            if q not in path:
                walk(path + [q])
    for p in graph:
        walk([p])
    return answers


def _coordinates(rng, levels, size, span_limit):
    minimum = size + 3 if size == 3 else size + 2
    gaps = np.full(levels - 1, minimum, dtype=int)
    spare = span_limit - int(gaps.sum())
    _need(spare >= 0, "embedding cannot fit")
    for i in rng.permutation(len(gaps)):
        extra = int(rng.integers(0, min(3, spare) + 1))
        gaps[int(i)] += extra
        spare -= extra
    return [size // 2 + 1] + list((np.cumsum(gaps) + size // 2 + 1).astype(int))


def _polyline(points):
    path = [points[0]]
    for q in points[1:]:
        p = path[-1]
        _need((p[0] == q[0]) != (p[1] == q[1]), "orthogonal edge")
        d = int(np.sign(q[0] - p[0])), int(np.sign(q[1] - p[1]))
        while path[-1] != q:
            path.append(_add(path[-1], d))
    return path


def _walls(path, all_paths, station_mask, left, right):
    walls = {}
    for i, p in enumerate(path):
        directions = []
        if i:
            directions.append((p[0] - path[i - 1][0], p[1] - path[i - 1][1]))
        if i + 1 < len(path):
            directions.append((path[i + 1][0] - p[0], path[i + 1][1] - p[1]))
        for d in directions:
            n = -d[1], d[0]
            for delta, color in ((n, left), ((-n[0], -n[1]), right)):
                q = _add(p, delta)
                if q in all_paths or station_mask[q] >= 0:
                    continue
                _need(q not in walls or walls[q] == color, "conflicting sides of one rail")
                walls[q] = color
    return walls


def _draw(rng, nodes, lattice_paths, route, labels, rail_colors,
          size, bg, token, rows, cols):
    ys = _coordinates(rng, rows, size, 25 - size)
    xs = _coordinates(rng, cols, size, 28 - size)
    centers = {p: (int(ys[p[0]]), int(xs[p[1]])) for p in nodes}
    radius = size // 2
    bottom = max(ys) + radius
    width = max(max(xs) + radius + 2, 2 * len(route) + 1)
    height = bottom + 5
    # Small independent margins keep the input distinct without a dimension ban.
    height += int(rng.integers(0, 31 - height))
    width += int(rng.integers(0, 31 - width))
    g = np.full((height, width), bg, dtype=np.int8)
    station_mask = np.full(g.shape, -1, dtype=np.int16)
    for index, p in enumerate(nodes):
        y, x = centers[p]
        g[y - radius:y + radius + 1, x - radius:x + radius + 1] = labels[p]
        g[y - radius + 1:y + radius, x - radius + 1:x + radius] = bg
        station_mask[y - radius:y + radius + 1, x - radius:x + radius + 1] = index
    g[centers[route[0]]] = token
    pixel_paths, all_paths = {}, set()
    for edge, order in lattice_paths.items():
        points = [(int(ys[p[0]]), int(xs[p[1]])) for p in order]
        path = _polyline(points)[radius + 1:-radius - 1]
        _need(len(path) >= 2 and not all_paths.intersection(path), "separated channels")
        pixel_paths[edge] = path
        all_paths.update(path)
    for edge, path in pixel_paths.items():
        left, right = rail_colors[edge]
        for q, color in _walls(path, all_paths, station_mask, left, right).items():
            _need(int(g[q]) in (bg, color), "rails of separate channels conflict")
            g[q] = color
    key_y = height - 2
    key_width = 2 * len(route) - 1
    key_x = int(rng.integers(1, width - key_width))
    g[key_y, key_x:key_x + key_width:2] = [labels[p] for p in route]
    out = g.copy()
    out[centers[route[0]]] = bg
    out[centers[route[-1]]] = token
    return dict(input=g, output=out), centers


def _sample_pair(rng, roles, index, is_test):
    bg, token = roles[:2]
    pool = roles[2:]
    for attempt in range(240):
        size = int(rng.choice((3, 3, 3, 5)))
        rows, cols = ((3, 4) if size == 5 else (4, 4)) if is_test else (
            (3, 3) if rng.random() < .7 else (3, 4))
        if size == 3 and not is_test and rng.random() < .3:
            rows, cols = cols, rows
        tree = _tree(rng, rows, cols)
        nodes, graph, paths = _suppress(rng, tree, .8 if is_test else .55)
        if is_test and len(nodes) < 7:
            continue
        length = int(rng.integers(5, min(8, len(nodes) - 2) + 1)) if is_test else (
            3 if index == 0 else 4 if index == 1 else int(rng.integers(3, 5)))
        candidates = _candidates(graph, length, same_depth=not is_test and index == 0)
        if not is_test and index == 1:
            candidates = [c for c in candidates if any(
                q not in c[0] and q != c[1] and q not in c[3] for q in graph[c[0][-2]])]
        if not candidates:
            continue
        route, wrong, early_step, early_path = candidates[int(rng.integers(len(candidates)))]
        early = early_path[-1]
        foreign_options = [q for q in graph[route[-2]]
                           if q not in route and q != wrong and q not in early_path]
        foreign = foreign_options[0] if foreign_options else None
        # The final color has at least three stations. One is behind a mismatch,
        # one is reachable by a different word, and one is the answer.
        final = int(rng.choice(pool))
        internal_pool = [int(c) for c in pool if c != final]
        labels = {p: int(rng.choice(internal_pool)) for p in nodes}
        key_colors = [int(rng.choice(internal_pool)) for _ in route[:-1]]
        if length >= 4 and rng.random() < .6:
            key_colors[-1] = key_colors[0]
        if not is_test and index == 1:
            key_colors[1] = int(rng.choice([c for c in internal_pool if c != key_colors[0]]))
        for p, color in zip(route[:-1], key_colors):
            labels[p] = color
        if len(early_path) > 2:
            # A competing walk of the correct length must have a wrong letter,
            # not merely an endpoint excluded by an arbitrary sampling rule.
            next_color = labels[route[early_step + 1]]
            labels[early_path[1]] = int(rng.choice([c for c in internal_pool if c != next_color]))
        for p in (route[-1], wrong, early):
            labels[p] = final
        if foreign is not None:
            labels[foreign] = final
        rail_colors = {}
        for edge in paths:
            u, v = edge
            identity = int(rng.choice((labels[u], labels[v], int(rng.choice(pool)))))
            other = int(rng.choice([c for c in pool if c != identity])) if rng.random() < .23 else identity
            rail_colors[edge] = (identity, other)
        for u, v in zip(route, route[1:]):
            rail_colors[tuple(sorted((u, v)))] = (labels[u], labels[u])
        source = route[-2]
        other = int(rng.choice([c for c in pool if c != labels[source]]))
        rail_colors[tuple(sorted((source, wrong)))] = (labels[source], other)
        if foreign is not None:
            identity = int(rng.choice([c for c in pool if c not in (labels[source], final)]))
            rail_colors[tuple(sorted((source, foreign)))] = (identity, identity)
        for source, target in zip(early_path, early_path[1:]):
            rail_colors[tuple(sorted((source, target)))] = (labels[source], labels[source])
        try:
            pair, centers = _draw(rng, nodes, paths, route, labels, rail_colors,
                                  size, bg, token, rows, cols)
            a = _analyse(pair["input"])
            if {s["center"] for s in a["stations"]} != set(centers.values()):
                continue
            if tuple(a["stations"][i]["center"] for i in a["route"]) != tuple(centers[p] for p in route):
                continue
            # Introduce curved channels before testing longer walks through them.
            if not is_test and index < 2 and not any(
                    a["tracks"][i, j]["turns"] for i, j in zip(a["route"], a["route"][1:])):
                continue
            _validate_pair(pair)
            return pair
        except AssertionError:
            continue
    # Deterministic bounded fallback with the same two diagnostic forks. It is
    # a fresh construction, never a parent pair or an exception to the rule.
    return _fallback_pair(rng, roles, is_test, index)


def _fallback_pair(rng, roles, is_test, index):
    bg, token, *pool = roles
    # A meandering backbone; the two spare leaves implement the same ablations.
    backbone = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 3)]
    if is_test:
        route = tuple(backbone)
        orders = [[u, v] for u, v in zip(route, route[1:])]
    elif index == 0:
        route = (backbone[0], backbone[3], backbone[5])
        orders = [backbone[:4], backbone[3:]]
    else:
        route = ((0, 0), (1, 0), (1, 1), (2, 2))
        orders = [[route[0], route[1]], [route[1], route[2]],
                  [route[2], (1, 2), route[3]]]
    wrong, early = (3, 2), (1, 0)
    early_step = 0
    early_path = [route[0], early]
    foreign = (2, 1) if is_test else None
    # The wrong leaf for short demonstrations needs a clear straight arm.
    if not is_test:
        if index == 0:
            wrong = (1, 3)
            early = (2, 0)
            early_path = [route[0], (1, 0), early]
        else:
            wrong, early, foreign, early_step = (0, 1), (2, 0), (2, 1), 1
            early_path = [route[early_step], early]
    orders.append([route[-2], wrong])
    orders.extend([[u, v] for u, v in zip(early_path, early_path[1:])])
    if foreign is not None:
        orders.append([route[-2], foreign])
    paths = {}
    for order in orders:
        if order[0] > order[-1]:
            order = order[::-1]
        paths[order[0], order[-1]] = order
    nodes = sorted(set(route) | {wrong, early} | set(early_path))
    if foreign is not None:
        nodes.append(foreign)
        nodes.sort()
    final = pool[-1]
    labels = {p: int(pool[i % (len(pool) - 1)]) for i, p in enumerate(route[:-1])}
    labels.update({p: final for p in (route[-1], wrong, early)})
    if len(early_path) > 2:
        labels[early_path[1]] = next(c for c in pool[:-1] if c != labels[route[early_step + 1]])
    if foreign is not None:
        labels[foreign] = final
    rails = {}
    for u, v in zip(route, route[1:]):
        rails[tuple(sorted((u, v)))] = (labels[u], labels[u])
    for source, target in zip(early_path, early_path[1:]):
        rails[tuple(sorted((source, target)))] = (labels[source], labels[source])
    rails[tuple(sorted((route[-2], wrong)))] = (labels[route[-2]], final)
    if foreign is not None:
        identity = next(c for c in pool if c not in (labels[route[-2]], final))
        rails[tuple(sorted((route[-2], foreign)))] = (identity, identity)
    pair, _ = _draw(rng, nodes, paths, route, labels, rails, 3, bg, token,
                    4 if is_test else 3, 4)
    _validate_pair(pair)
    return pair


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " outside legal range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    colors = tuple(map(int, rng.permutation(10)))
    varied_roles = bool(rng.integers(0, 2))
    palette_size = int(rng.integers(3, 9))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            order = tuple(map(int, rng.permutation(10))) if varied_roles else colors
            roles = order[:2 + palette_size]
            pair = _sample_pair(rng, roles, index, split == "test")
            if _signature(pair["input"]) in seen:
                # There are more bijective affine remappings than prior pairs.
                found = False
                for multiplier in (1, 3, 7, 9):
                    for shift in range(10):
                        remap = (multiplier * np.arange(10) + shift) % 10
                        candidate = {k: remap[v].astype(np.int8) for k, v in pair.items()}
                        if _signature(candidate["input"]) not in seen:
                            pair, found = candidate, True
                            break
                    if found:
                        break
                _need(found, "distinct input remapping")
            seen.add(_signature(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _reachable(graph, start):
    seen, pending = {start}, [start]
    while pending:
        for v in graph[pending.pop()]:
            if v not in seen:
                seen.add(v)
                pending.append(v)
    return seen


def _distances(graph, start):
    distance, queue = {start: 0}, [start]
    for u in queue:
        for v in graph[u]:
            if v not in distance:
                distance[v] = distance[u] + 1
                queue.append(v)
    return distance


def _validate_pair(pair):
    _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    _need(inp.shape == out.shape, "dimensions are preserved")
    a = _analyse(inp)
    expected = inp.copy()
    start = a["stations"][a["start"]]["center"]
    end = a["stations"][a["route"][-1]]["center"]
    expected[start], expected[end] = a["background"], a["token"]
    _need(np.array_equal(expected, out), "incorrect relay transfer or altered scenery")
    _need(np.count_nonzero(inp != out) == 2, "only the two endpoint sockets change")
    _need(np.count_nonzero(out == a["token"]) == 1, "token conservation")
    sizes = {s["size"] for s in a["stations"]}
    _need(len(sizes) == 1 and sizes <= {3, 5}, "generated station geometry")
    _need(3 <= len(a["key"]) <= 8 and a["key"][-1] not in a["key"][:-1],
          "generated word conventions")
    physical = {tuple(sorted((i, j))) for i, j in a["tracks"]}
    _need(len(physical) == len(a["stations"]) - 1, "generated network is a tree")
    undirected = [[] for _ in a["stations"]]
    used_cells = set()
    covered = a["station_mask"] >= 0
    covered[a["key_y"]] = True
    for i, j in physical:
        undirected[i].append(j)
        undirected[j].append(i)
        cells = set(a["tracks"][i, j]["path"])
        _need(len(set(a["tracks"][i, j]["supports"])) == 1,
              "each rail side has a constant guide identity")
        _need(not used_cells.intersection(cells), "channels cannot cross or merge")
        used_cells.update(cells)
        for p in cells:
            for d in DIRS:
                q = _add(p, d)
                if _inside(inp, q):
                    covered[q] = True
    _need(np.all((inp == a["background"]) | covered), "unexplained guide cells")
    _need(len(_reachable(undirected, 0)) == len(undirected), "connected physical network")
    relaxed, _ = _network(a, "ignore_matching")
    relaxed_ends = {p[-1] for p in _word_paths(a, relaxed)}
    _need(len(relaxed_ends) >= 2, "rail equality must affect the answer")
    final = a["key"][-1]
    wordless = {i for i in _reachable(a["graph"], a["start"])
                if a["stations"][i]["label"] == final}
    _need(len(wordless) >= 2, "the itinerary must disambiguate reachable final labels")
    _need(a["route"][-1] in relaxed_ends & wordless, "ablation consistency")
    distance = _distances(a["graph"], a["start"])
    a["same_depth_decoy"] = any(i != a["route"][-1] and distance[i] == len(a["key"]) - 1
                                for i in wordless)
    foreign_graph, _ = _network(a, "ignore_source")
    a["source_sensitive"] = len({p[-1] for p in _word_paths(a, foreign_graph)}) >= 2
    return a


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _need(2 <= n <= 10 and 1 <= m <= 4 and n > m, "pair counts")
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            a = _validate_pair(pair)
            sig = _signature(np.asarray(pair["input"]))
            _need(sig not in seen, "duplicate inputs")
            seen.add(sig)
            infos[split].append(a)
    _need(max(len(a["key"]) for a in infos["train"]) <
          min(len(a["key"]) for a in infos["test"]), "longer test itineraries")
    for a in infos["train"][:2]:
        _need(any(a["tracks"][i, j]["turns"] for i, j in zip(a["route"], a["route"][1:])),
              "training must demonstrate curved interstation channels")
    _need(all(len(a["key"]) in (3, 4) for a in infos["train"]), "training word lengths")
    _need(all(5 <= len(a["key"]) <= 8 for a in infos["test"]), "test word lengths")
    _need(any(len(set(a["key"][:-1])) >= 2 for a in infos["train"]),
          "training must demonstrate a change of departure identity")
    _need(any(a["source_sensitive"] for a in infos["train"]),
          "training must distinguish equal rails from source-colored equal rails")
    _need(infos["train"][0]["same_depth_decoy"],
          "the first demonstration must distinguish the word from its length")


def _metrics(a):
    physical = [v for (i, j), v in a["tracks"].items() if i < j]
    return dict(shape=a["grid"].shape, stations=len(a["stations"]), word=len(a["key"]),
                size=tuple(sorted({s["size"] for s in a["stations"]})),
                turns=tuple(t["turns"] for t in physical),
                lengths=tuple(len(t["path"]) for t in physical),
                repeated=len(set(a["key"])) < len(a["key"]),
                roles=(a["background"], a["token"]),
                layout=tuple((s["center"], s["size"]) for s in a["stations"]))


def self_test():
    """A streaming, bounded 42-seed sweep plus ablations and corruptions."""
    shapes, layouts, roles, stations, words, sizes, turns, lengths = (set() for _ in range(8))
    digests, repeated = set(), 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _validate_pair(pair)
                before = pair["input"].copy()
                _need(np.array_equal(solve(pair["input"]), pair["output"]), "solve agreement")
                _need(np.array_equal(pair["input"], before), "solver must not mutate its input")
                m = _metrics(a)
                shapes.add(m["shape"])
                layouts.add(m["layout"])
                roles.add(m["roles"])
                stations.add(m["stations"])
                words.add(m["word"])
                sizes.update(m["size"])
                turns.update(m["turns"])
                lengths.update(m["lengths"])
                repeated += m["repeated"]
                digest.update(repr(pair["input"].shape).encode())
                digest.update(pair["input"].tobytes())
        digests.add(digest.hexdigest())
        if seed == 0:
            again = generate_puzzle(seed)
            _need(all(np.array_equal(p[k], q[k]) for s in puzzle
                      for p, q in zip(puzzle[s], again[s]) for k in ("input", "output")),
                  "whole-puzzle determinism")
            for test, index in ((False, 0), (False, 1), (True, 0)):
                # Exercise the finite fallback directly rather than leaving it untested.
                fallback = _fallback_pair(np.random.default_rng(900 + int(test)),
                                          tuple(range(8)), test, index)
                fa = _validate_pair(fallback)
                _need(test or index == 0 or fa["source_sensitive"], "fallback source permission")
                _need(test or index != 0 or fa["same_depth_decoy"], "fallback word versus length")
            corrupt = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
            corrupt["test"][0]["output"] = corrupt["test"][0]["input"].copy()
            try:
                validate_generated_puzzle(corrupt)
            except AssertionError:
                pass
            else:
                raise AssertionError("identity answer was accepted")
    _need(len(digests) == 42, "seed diversity")
    _need(len(shapes) >= 25 and len(layouts) >= 100, "structural variation")
    _need(len(roles) >= 20 and len(stations) >= 6, "roles and counts must vary")
    _need({3, 4, 5, 6, 7, 8} <= words and {3, 5} <= sizes, "length and scale variation")
    _need({0, 1, 2} <= turns and max(turns) >= 3, "straight and multiply bent rails")
    _need(len(lengths) >= 12 and repeated >= 20, "spacing and repeated-word variation")
    for n, m in ((2, 1), (10, 4), (5, 3)):
        p = generate_puzzle(101 + n + m, n, m)
        _need(len(p["train"]) == n and len(p["test"]) == m, "explicit counts")
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")


if __name__ == "__main__":
    self_test()
    print(json.dumps({"self_test": "passed", "seeds": 42}, separators=(",", ":")))
