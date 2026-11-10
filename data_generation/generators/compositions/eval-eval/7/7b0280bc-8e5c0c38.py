"""Mirror-word railway: the shortest simple palindromic station route.

All conventions are visible: hollow square terminals, solid square stations,
and thin connectors. Reflection acts on station order, not canvas coordinates.
No parent code, files, random state, or training labels enter solve().
"""
from collections import Counter, deque
from itertools import combinations
import numpy as np


_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)), "bad grid colors")
    return a


def _key(grid):
    return grid.shape, np.ascontiguousarray(grid, dtype=np.uint8).tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    answer = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        cells, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in _D4:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    cells.add(q)
                    stack.append(q)
        answer.append(cells)
    return answer


def _block_kind(cells):
    r0 = min(r for r, c in cells)
    r1 = max(r for r, c in cells)
    c0 = min(c for r, c in cells)
    c1 = max(c for r, c in cells)
    size = r1 - r0 + 1
    if size not in (3, 4) or c1 - c0 + 1 != size:
        return None
    if len(cells) == size * size:
        return "solid", size
    border = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
              if r in (r0, r1) or c in (c0, c1)}
    return ("ring", size) if cells == border else None


def _adjacency(count, edges):
    adj = [[] for _ in range(count)]
    for e, (a, b) in enumerate(edges):
        adj[a].append((b, e))
        adj[b].append((a, e))
    return adj


def _paths(adj, start, target):
    """At most 24 stations and cycle rank four; simple paths only."""
    answer = []
    stack = [(start, [start], [])]
    expansions = 0
    while stack:
        a, nodes, edges = stack.pop()
        expansions += 1
        _require(expansions <= 100000, "network exceeds the supported path-search bound")
        if a == target:
            answer.append((nodes, edges))
            continue
        for b, edge in reversed(adj[a]):
            if b not in nodes:
                stack.append((b, nodes + [b], edges + [edge]))
    return sorted(answer, key=lambda p: (len(p[0]), p[0], p[1]))


def _read(grid):
    grid = _grid(grid)
    colors, counts = np.unique(grid, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    _require(int(counts.max()) > grid.size / 2, "background must be a strict majority")
    groups = {int(c): _components(grid == c) for c in colors if c != bg}
    kinds = {c: [_block_kind(x) for x in cc] for c, cc in groups.items()}
    line_colors = [c for c, kk in kinds.items() if any(k is None for k in kk)]
    _require(len(line_colors) == 1, "cannot identify the connector color")
    connector = line_colors[0]
    stations, labels, terminal_ids, sizes = [], [], [], []
    for color in sorted(groups):
        if color == connector:
            continue
        for cells, kind in zip(groups[color], kinds[color]):
            _require(kind is not None, "invalid station")
            if kind[0] == "ring":
                terminal_ids.append(len(stations))
            sizes.append(kind[1])
            stations.append(cells)
            labels.append(color)
    _require(6 <= len(stations) <= 24 and len(set(sizes)) == 1, "invalid station count or scale")
    _require(len(terminal_ids) == 2, "need two hollow terminals")
    _require(labels[terminal_ids[0]] == labels[terminal_ids[1]], "terminal colors disagree")
    terminal_color = labels[terminal_ids[0]]
    _require(all(labels[i] != terminal_color for i in range(len(labels)) if i not in terminal_ids),
             "terminal color is reserved for the two rings")
    station_map = {p: i for i, cells in enumerate(stations) for p in cells}
    edge_cells, edges = groups[connector], []
    for cells in edge_cells:
        touched = {station_map[q] for r, c in cells for dr, dc in _D4
                   if (q := (r + dr, c + dc)) in station_map}
        _require(len(touched) == 2, "each connector must touch exactly two stations")
        edges.append(tuple(sorted(touched)))
    _require(len(set(edges)) == len(edges), "parallel edges are excluded")
    adj = _adjacency(len(stations), edges)
    _require(all(1 <= len(a) <= 3 for a in adj), "station degrees must be 1..3")
    reached, queue = {0}, deque([0])
    while queue:
        for b, _ in adj[queue.popleft()]:
            if b not in reached:
                reached.add(b)
                queue.append(b)
    _require(len(reached) == len(stations), "network must be connected")
    rank = len(edges) - len(stations) + 1
    _require(1 <= rank <= 4, "cycle rank must be 1..4")
    routes = _paths(adj, *terminal_ids)
    return dict(bg=bg, connector=connector, stations=stations, labels=labels,
                terminals=terminal_ids, edges=edges, edge_cells=edge_cells,
                adj=adj, routes=routes, size=sizes[0], rank=rank)


def _word(data, route):
    return [data["labels"][i] for i in route[0][1:-1]]


def _palindromes(data):
    return [p for p in data["routes"] if (w := _word(data, p)) == w[::-1]]


def _palindrome_multiset(word):
    return sum(n % 2 for n in Counter(word).values()) <= 1


def _paint(grid, data, route):
    out = np.full_like(grid, data["bg"])
    for n in route[0]:
        for p in data["stations"][n]:
            out[p] = grid[p]
    for e in route[1]:
        for p in data["edge_cells"][e]:
            out[p] = grid[p]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Retain the unique minimum-station palindromic simple route."""
    grid = _grid(input_grid)
    data = _read(grid)
    eligible = _palindromes(data)
    _require(bool(eligible), "no palindromic route")
    _require(len(eligible) == 1 or len(eligible[0][0]) < len(eligible[1][0]),
             "shortest palindromic route is tied")
    return _paint(grid, data, eligible[0])


def _sample_graph(rng, target):
    nr = int(rng.integers(3, 5))
    nc = int(rng.integers(3 if target else 3, 6))
    vertices = [(r, c) for r in range(nr) for c in range(nc)]
    candidates = [(a, b) for a, b in combinations(vertices, 2)
                  if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1]
    active = {vertices[int(rng.integers(len(vertices)))]}
    edges, degree = [], {v: 0 for v in vertices}
    wanted = int(rng.integers(max(8, len(vertices) - 3), len(vertices) + 1))
    while len(active) < wanted:
        options = [(a, b) for a, b in candidates if (a in active) != (b in active)
                   and degree[a] < 3 and degree[b] < 3]
        if not options:
            return None
        a, b = options[int(rng.integers(len(options)))]
        edges.append((a, b))
        active.update((a, b))
        degree[a] += 1
        degree[b] += 1
    extras = [(a, b) for a, b in candidates if a in active and b in active and (a, b) not in edges]
    rng.shuffle(extras)
    desired = int(rng.integers(2, 5))
    added = 0
    for a, b in extras:
        if added < desired and degree[a] < 3 and degree[b] < 3:
            edges.append((a, b))
            degree[a] += 1
            degree[b] += 1
            added += 1
    if added < 2:
        return None
    visible = {v for v in active if degree[v] != 2 or rng.random() > .35}
    if len(visible) < 7:
        return None
    return nr, nc, active, edges, visible


def _compress(graph):
    nr, nc, active, raw_edges, visible = graph
    order = sorted(visible)
    ids = {v: i for i, v in enumerate(order)}
    neighbors = {v: [] for v in active}
    for a, b in raw_edges:
        neighbors[a].append(b)
        neighbors[b].append(a)
    edges, walks = [], []
    for a in order:
        for b in sorted(neighbors[a]):
            walk = [a, b]
            while walk[-1] not in visible:
                options = [v for v in neighbors[walk[-1]] if v != walk[-2]]
                _require(len(options) == 1, "invalid suppressed bend")
                walk.append(options[0])
            if ids[a] < ids[walk[-1]]:
                edges.append((ids[a], ids[walk[-1]]))
                walks.append(walk)
    _require(len(set(edges)) == len(edges), "parallel compressed paths")
    return order, edges, walks


def _coordinates(rng, n, size):
    base = n * size + (n - 1) * 2
    slack = int(rng.integers(0, min(5, 30 - base) + 1))
    extras = rng.multinomial(slack, np.full(n - 1, 1 / (n - 1)))
    return [0] + list(np.cumsum(size + 2 + extras).astype(int))


def _lengths(order, walks, rows, cols):
    return [sum(abs(rows[a[0]] - rows[b[0]]) + abs(cols[a[1]] - cols[b[1]])
                for a, b in zip(walk, walk[1:])) for walk in walks]


def _color_routes(rng, n, chosen, longer):
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for route in (chosen, longer):
        if route is None:
            continue
        word = route[0][1:-1]
        for a, b in zip(word, word[::-1]):
            parent[find(a)] = find(b)
    roots = sorted({find(i) for i in range(n)})
    palette_count = int(rng.integers(2, 4))
    colors = {root: int(rng.integers(palette_count)) for root in roots}
    return [colors[find(i)] for i in range(n)]


def _select(rng, order, edges, walks, rows, cols, parity, target, witness):
    adj = _adjacency(len(order), edges)
    lengths = _lengths(order, walks, rows, cols)
    endpoints = list(combinations(range(len(order)), 2))
    rng.shuffle(endpoints)
    for a, b in endpoints[:45]:
        routes = _paths(adj, a, b)
        if len(routes) < 3 or len(routes[0][0]) == len(routes[1][0]):
            continue
        selected = [p for p in routes if (len(p[0]) - 2) % 2 == parity
                    and (5 <= len(p[0]) - 2 <= 8 if target else 3 <= len(p[0]) - 2 <= 4)
                    and len(p[0]) > len(routes[0][0])]
        rng.shuffle(selected)
        for chosen in selected[:6]:
            losers = [p for p in routes if len(p[0]) > len(chosen[0])
                      and (not witness or sum(lengths[e] for e in p[1]) < sum(lengths[e] for e in chosen[1]))]
            if witness and not losers:
                continue
            rng.shuffle(losers)
            options = losers[:3] if witness else [None]
            for longer in options:
                for _ in range(4):
                    labels = _color_routes(rng, len(order), chosen, longer)
                    if len({labels[n] for n in chosen[0][1:-1]}) < 2:
                        continue
                    eligible = [p for p in routes if (w := [labels[n] for n in p[0][1:-1]]) == w[::-1]]
                    if eligible and eligible[0] == chosen and (len(eligible) == 1 or len(eligible[1][0]) > len(chosen[0])):
                        if not target and parity == 0 and not any(
                                len(p[0]) < len(chosen[0]) and _palindrome_multiset([labels[n] for n in p[0][1:-1]])
                                for p in routes):
                            continue
                        return (a, b), labels, chosen
    return None


def _fallback(rng, parity, target):
    """Three competing corridors; no output exception or hidden inference rule."""
    nr = nc = 5
    raw = []
    for row in (0, 2, 4):
        raw.extend([((row, c), (row, c + 1)) for c in range(4)])
    for col in (0, 4):
        raw.extend([((r, col), (r + 1, col)) for r in range(4)])
    active = {v for edge in raw for v in edge}
    top_walk = [(2, 0), (1, 0)] + [(0, c) for c in range(5)] + [(1, 4), (2, 4)]
    bottom_walk = [(2, 0), (3, 0)] + [(4, c) for c in range(5)] + [(3, 4), (2, 4)]
    selected_count = (6 if parity == 0 else 5) if target else (4 if parity == 0 else 3)
    longer_count = selected_count + 1
    chosen_nodes = sorted(map(int, rng.choice(7, selected_count, replace=False)))
    longer_nodes = sorted(map(int, rng.choice(7, longer_count, replace=False)))
    visible = {(2, 0), (2, 4), (2, 1), (2, 3)}
    if not target and parity == 0:
        visible.add((2, 2))
    visible.update(top_walk[j + 1] for j in chosen_nodes)
    visible.update(bottom_walk[j + 1] for j in longer_nodes)
    # A spare dead-end may branch off the middle route, never at a crossing.
    if rng.random() < .65:
        raw.append(((2, 1), (1, 1)))
        active.add((1, 1))
        visible.add((1, 1))
    graph = (nr, nc, active, raw, visible)
    order, edges, walks = _compress(graph)
    ids = {v: i for i, v in enumerate(order)}
    adj = _adjacency(len(order), edges)
    ends = (ids[(2, 0)], ids[(2, 4)])
    routes = _paths(adj, *ends)
    chosen = next(p for p in routes if len(p[0]) == selected_count + 2)
    longer = next(p for p in routes if len(p[0]) == longer_count + 2)
    labels = _color_routes(rng, len(order), chosen, longer)
    # Ensure the shortcut's two colors differ and the selected word is nonconstant.
    labels[ids[(2, 1)]], labels[ids[(2, 3)]] = 0, 1
    if (2, 2) in ids:
        labels[ids[(2, 2)]] = 0
    half = [int(rng.integers(3)) for _ in range((selected_count + 1) // 2)]
    half[0], half[1] = 0, 1
    word = half + (half[-2::-1] if selected_count % 2 else half[::-1])
    for n, color in zip(chosen[0][1:-1], word):
        labels[n] = color
    # Top is physically longer; the bottom contains more visible stations.
    size = 3
    rows = [0, 7, 14, 19, 24]
    cols = _coordinates(rng, nc, size)
    return graph, order, edges, walks, rows, cols, size, ends, labels, chosen


def _render(rng, order, walks, rows, cols, size, ends, labels, chosen):
    palette = list(map(int, rng.choice(10, 6, replace=False)))
    bg, connector, terminal, *ordinary = palette
    h, w = int(rows[-1] + size), int(cols[-1] + size)
    grid = np.full((h, w), bg, dtype=np.int8)
    keep = np.zeros((h, w), dtype=bool)
    center = (size - 1) // 2
    for e, walk in enumerate(walks):
        for a, b in zip(walk, walk[1:]):
            r0, c0 = rows[a[0]] + center, cols[a[1]] + center
            r1, c1 = rows[b[0]] + center, cols[b[1]] + center
            grid[min(r0, r1):max(r0, r1) + 1, min(c0, c1):max(c0, c1) + 1] = connector
            if e in chosen[1]:
                keep[min(r0, r1):max(r0, r1) + 1, min(c0, c1):max(c0, c1) + 1] = True
    for i, (r, c) in enumerate(order):
        top, left = rows[r], cols[c]
        color = terminal if i in ends else ordinary[labels[i]]
        grid[top:top + size, left:left + size] = color
        keep[top:top + size, left:left + size] = i in chosen[0]
        if i in ends:
            grid[top + 1:top + size - 1, left + 1:left + size - 1] = bg
    out = np.where(keep, grid, bg).astype(np.int8)
    rotation = int(rng.integers(4))
    grid, out = np.rot90(grid, rotation), np.rot90(out, rotation)
    if rng.random() < .5:
        grid = grid[:, ::-1]
        out = out[:, ::-1]
    hh, ww = grid.shape
    extra_h = int(rng.integers(0, min(4, 30 - hh) + 1))
    extra_w = int(rng.integers(0, min(4, 30 - ww) + 1))
    top = int(rng.integers(extra_h + 1))
    left = int(rng.integers(extra_w + 1))
    inp = np.full((hh + extra_h, ww + extra_w), bg, dtype=np.int8)
    output = inp.copy()
    inp[top:top + hh, left:left + ww] = grid
    output[top:top + hh, left:left + ww] = out
    return {"input": inp, "output": output}


def _make_pair(rng, parity, target, witness):
    for _ in range(35):
        graph = _sample_graph(rng, target)
        if graph is None:
            continue
        try:
            order, edges, walks = _compress(graph)
            size = int(rng.choice([3, 4]))
            rows = _coordinates(rng, graph[0], size)
            cols = _coordinates(rng, graph[1], size)
            choice = _select(rng, order, edges, walks, rows, cols, parity, target, witness)
            if choice is None:
                continue
            ends, labels, chosen = choice
            pair = _render(rng, order, walks, rows, cols, size, ends, labels, chosen)
            inp, out = pair["input"], pair["output"]
            data = _read(inp)
            _require(len(data["stations"]) == len(order) and len(data["edges"]) == len(edges), "rendered graph changed")
            _require(np.array_equal(solve(inp), out), "constructed answer differs from recovered route")
            if not target and parity == 0:
                _require(measure_pair(pair)["histogram_conflict"], "need an order-versus-counts witness")
            if witness:
                _require(measure_pair(pair)["geometric_conflict"], "rendered physical-distance witness lost")
            return pair
        except ValueError:
            continue
    graph, order, edges, walks, rows, cols, size, ends, labels, chosen = _fallback(rng, parity, target)
    return _render(rng, order, walks, rows, cols, size, ends, labels, chosen)


def _count(value, low, high, name):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)), name + " must be an integer")
    _require(low <= int(value) <= high, name + " out of range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    _require(num_train > num_test, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    parity_order = list(map(int, rng.permutation(2)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            parity = parity_order[i] if split == "train" and i < 2 else int(rng.integers(2))
            witness = split == "train" and i == 0
            for _ in range(10):
                pair = _make_pair(rng, parity, split == "test", witness)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input retry budget exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _geometric_length(data, route):
    # Raster path length plus the shortest center-to-port walk inside each block.
    total = 0.
    for e in route[1]:
        cells = data["edge_cells"][e]
        total += len(cells) - 1
        for n in data["edges"][e]:
            station = data["stations"][n]
            center = np.mean(list(station), axis=0)
            ports = [(r, c) for r, c in cells if any((r + dr, c + dc) in station for dr, dc in _D4)]
            total += min(abs(r - center[0]) + abs(c - center[1]) for r, c in ports)
    return total


def measure_pair(pair):
    data = _read(np.asarray(pair["input"]))
    eligible = _palindromes(data)
    chosen = eligible[0]
    return dict(shape=list(np.asarray(pair["input"]).shape), size=data["size"],
                stations=len(data["stations"]), cycle_rank=data["rank"],
                degrees=sorted(map(len, data["adj"])), routes=len(data["routes"]),
                palindromic_routes=len(eligible), selected=len(chosen[0]) - 2,
                raw_shortest=len(data["routes"][0][0]) - 2,
                word=_word(data, chosen), background=data["bg"],
                histogram_conflict=any(len(p[0]) < len(chosen[0]) and _palindrome_multiset(_word(data, p))
                                       for p in data["routes"]),
                geometric_conflict=any(len(p[0]) > len(chosen[0]) and
                    _geometric_length(data, p) < _geometric_length(data, chosen) for p in eligible),
                erased=int(np.count_nonzero(np.asarray(pair["input"]) != np.asarray(pair["output"]))))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _count(len(puzzle["train"]), 2, 10, "num_train")
    _count(len(puzzle["test"]), 1, 4, "num_test")
    _require(len(puzzle["train"]) > len(puzzle["test"]), "invalid split counts")
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "shape changed")
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            data = _read(inp)
            eligible = _palindromes(data)
            _require(bool(eligible) and (len(eligible) == 1 or len(eligible[0][0]) < len(eligible[1][0])), "ambiguous answer")
            chosen = eligible[0]
            _require(np.array_equal(out, _paint(inp, data, chosen)), "incorrect path mask")
            _require(np.array_equal(out, solve(inp.copy())), "input-only solve mismatch")
            _require(np.all((inp == out) | (out == data["bg"])), "only deletion is allowed")
            _require(len(data["routes"]) >= 3 and len(data["routes"][0][0]) < len(data["routes"][1][0]), "need a unique raw shortest route")
            _require(len(chosen[0]) > len(data["routes"][0][0]), "omitting symmetry must change the route")
            _require(len(set(_word(data, chosen))) >= 2, "selected word must be nonconstant")
            records[split].append(measure_pair(pair))
    _require({r["selected"] % 2 for r in records["train"]} == {0, 1}, "teach both reflection-center parities")
    _require(all(3 <= r["selected"] <= 4 for r in records["train"]), "training radius out of range")
    _require(all(5 <= r["selected"] <= 8 for r in records["test"]), "test must extend palindrome radius")
    _require(any(r["geometric_conflict"] and r["palindromic_routes"] >= 2 for r in records["train"]),
             "need a station-count versus physical-distance training witness")
    _require(any(r["histogram_conflict"] for r in records["train"]), "need a color-order witness")


def self_test():
    """Bounded, in-memory checks on 42 seeds, mutations, counts and symmetries."""
    stats, structures, fingerprints = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = measure_pair(pair)
                f["split"] = split
                stats.append(f)
                inp, out = pair["input"], pair["output"]
                # Geometry and literal colors cannot affect the route rule.
                permutation = np.random.default_rng(seed + 1000).permutation(10)
                assert np.array_equal(solve(permutation[inp]), permutation[out])
                assert np.array_equal(solve(np.rot90(inp)), np.rot90(out))
                roles = {}
                semantic = bytes(roles.setdefault(int(c), len(roles)) for c in inp.ravel())
                structures.add((inp.shape, semantic))
                # Check the word with explicit reflected indices, including centers.
                data = _read(inp)
                route = _palindromes(data)[0]
                word = _word(data, route)
                assert all(word[j] == word[len(word) - 1 - j] for j in range(len(word)))
                assert np.array_equal(_paint(inp, data, data["routes"][0]), out) is False
        if seed < 2:
            again = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle for p, q in zip(puzzle[s], again[s]) for k in p)
    assert len(fingerprints) == 42 and len(structures) >= 150
    assert len({tuple(f["shape"]) for f in stats}) >= 30
    assert {f["size"] for f in stats} == {3, 4}
    assert {f["background"] for f in stats} == set(range(10))
    assert len({f["stations"] for f in stats}) >= 6
    assert len({f["routes"] for f in stats}) >= 4
    assert any(1 in f["degrees"] for f in stats)
    assert any(f["cycle_rank"] >= 3 for f in stats)
    assert {f["selected"] for f in stats} >= {3, 4, 5, 6}
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(700 + nt * 10 + ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Equal symmetric routes have no arbitrary lexicographic or geometric tie-break.
    perimeter = {(r, c) for r in range(3) for c in range(3) if r in (0, 2) or c in (0, 2)}
    raw = [(a, b) for a, b in combinations(sorted(perimeter), 2)
           if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1]
    graph = (3, 3, perimeter, raw, perimeter)
    order, edges, walks = _compress(graph)
    ends = (order.index((1, 0)), order.index((1, 2)))
    routes = _paths(_adjacency(len(order), edges), *ends)
    for no_route in (False, True):
        labels = [0] * len(order)
        if no_route:
            for route in routes:
                for n, color in zip(route[0][1:-1], (0, 0, 1)):
                    labels[n] = color
        pair = _render(np.random.default_rng(4), order, walks, [0, 5, 10], [0, 5, 10],
                       3, ends, labels, routes[0])
        try:
            solve(pair["input"])
        except ValueError as error:
            assert ("no palindromic" if no_route else "tied") in str(error)
        else:
            raise AssertionError("ambiguous or absent palindromic route accepted")
    bad = generate_puzzle(991, 2, 1)
    inp, out = bad["test"][0]["input"], bad["test"][0]["output"]
    bg = _read(inp)["bg"]
    p = tuple(np.argwhere(out != bg)[0])
    out[p] = bg
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("partial-station deletion accepted")
    return dict(seeds=42, pairs=len(stats), distinct_structures=len(structures),
                shapes=len({tuple(f["shape"]) for f in stats}),
                station_counts=sorted({f["stations"] for f in stats}),
                route_counts=sorted({f["routes"] for f in stats}),
                selected_counts=sorted({f["selected"] for f in stats}))
