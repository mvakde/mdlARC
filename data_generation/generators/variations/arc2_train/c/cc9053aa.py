"""ARC rail routing with hazard clearance and a visible shortest, least-turn route."""

import heapq
import numpy as np

_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_SOURCE = (
    (("000000000", "088888889", "080008080", "088878780", "080808080", "088888880", "000000090"),
     ("000000000", "099999999", "090008080", "098878780", "090808080", "099999990", "000000090")),
    (("000000000", "088808880", "080878080", "980888089", "080000080", "088888880", "000000000"),
     ("000000000", "088808880", "080878080", "990888099", "090000090", "099999990", "000000000")),
    (("000090000", "088888880", "087000080", "088888880", "080080080", "088888880", "080000780", "088888880", "000090000"),
     ("000090000", "088899990", "087000090", "088899990", "080090080", "099998880", "090000780", "099998880", "000090000")),
    (("090000090", "088888880", "080878080", "088888880", "080878080", "088888880", "080808080", "088888880", "000000000"),
     ("090000090", "098888890", "090878090", "098888890", "090878090", "098888890", "090808090", "099999990", "000000000")),
)


def _count(name, value, lo, hi):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _counts(seed, rng, nt, ns):
    nt, ns = _count("num_train", nt, 2, 10), _count("num_test", ns, 1, 4)
    if nt is None and ns is None:
        ns = 1 + seed % 4
        nt = ns + 1 + (seed // 4) % (10 - ns)
    elif nt is None:
        nt = int(rng.integers(ns + 1, 11))
    elif ns is None:
        ns = int(rng.integers(1, min(4, nt - 1) + 1))
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    return nt, ns


def _nbr(p, h, w):
    r, c = p
    for dr, dc in _DIRS:
        q = (r + dr, c + dc)
        if 0 <= q[0] < h and 0 <= q[1] < w:
            yield q


def _edge_count(cells):
    return sum((r + 1, c) in cells for r, c in cells) + sum((r, c + 1) in cells for r, c in cells)


def _components(cells, h, w):
    left = set(cells)
    result = []
    while left:
        start = next(iter(left))
        seen = {start}
        todo = [start]
        left.remove(start)
        while todo:
            for q in _nbr(todo.pop(), h, w):
                if q in left:
                    left.remove(q)
                    seen.add(q)
                    todo.append(q)
        result.append(seen)
    return result


def _route(cells, start, goal, h, w):
    """Lexicographically shortest (cells, turns) route, with optimal count."""
    if start not in cells or goal not in cells:
        return None
    origin = (start, 4)
    dist = {origin: (1, 0)}
    parent = {origin: None}
    ways = {origin: 1}
    queue = [(1, 0, start[0], start[1], 4)]
    while queue:
        length, turns, r, c, direction = heapq.heappop(queue)
        state = ((r, c), direction)
        if dist[state] != (length, turns):
            continue
        for d, (dr, dc) in enumerate(_DIRS):
            q = (r + dr, c + dc)
            if q not in cells:
                continue
            nxt = (q, d)
            score = (length + 1, turns + int(direction != 4 and d != direction))
            if nxt not in dist or score < dist[nxt]:
                dist[nxt], parent[nxt], ways[nxt] = score, state, ways[state]
                heapq.heappush(queue, (score[0], score[1], q[0], q[1], d))
            elif score == dist[nxt]:
                ways[nxt] = min(2, ways[nxt] + ways[state])
    goals = [s for s in dist if s[0] == goal]
    if not goals:
        return None
    score = min(dist[s] for s in goals)
    best = [s for s in goals if dist[s] == score]
    count = min(2, sum(ways[s] for s in best))
    path = []
    cur = best[0]
    while cur is not None:
        path.append(cur[0])
        cur = parent[cur]
    path.reverse()
    return path, score, count


def _simple_paths_two(cells, a, b, h, w):
    # There is more than one simple route iff some edge of one route is
    # not a bridge. This avoids enumerating exponentially many loops.
    parent = {a: None}
    queue = [a]
    for p in queue:
        for q in _nbr(p, h, w):
            if q in cells and q not in parent:
                parent[q] = p
                queue.append(q)
    if b not in parent:
        return 0
    path = []
    p = b
    while p != a:
        path.append((parent[p], p))
        p = parent[p]
    for edge in path:
        blocked_edge = {edge, (edge[1], edge[0])}
        seen = {a}
        queue = [a]
        for p in queue:
            for q in _nbr(p, h, w):
                if q in cells and q not in seen and (p, q) not in blocked_edge:
                    seen.add(q)
                    queue.append(q)
        if b in seen:
            return 2
    return 1


def _shortest_paths_two(cells, start, goal, h, w):
    distance = {start: 0}
    ways = {start: 1}
    queue = [start]
    for p in queue:
        for q in _nbr(p, h, w):
            if q not in cells:
                continue
            if q not in distance:
                distance[q] = distance[p] + 1
                ways[q] = ways[p]
                queue.append(q)
            elif distance[q] == distance[p] + 1:
                ways[q] = min(2, ways[q] + ways[p])
    return ways.get(goal, 0)


def _axis(rng):
    mode = int(rng.choice(3, p=(0.45, 0.35, 0.20)))
    if mode == 0:
        count, gaps = int(rng.integers(3, 5)), (2, 3)
    elif mode == 1:
        count, gaps = int(rng.integers(3, 6)), (2, 3, 4)
    else:
        count, gaps = int(rng.integers(4, 7)), (3, 4, 5)
    for _ in range(20):
        coords = [1]
        for _ in range(count - 1):
            coords.append(coords[-1] + int(rng.choice(gaps)))
        pad = int(rng.integers(1, 3))
        size = coords[-1] + pad + 1
        if 7 <= size <= 30:
            return coords, size
    return [1, 3, 5], 7


def _logical_edges(nr, nc):
    result = []
    for r in range(nr):
        for c in range(nc):
            if r + 1 < nr:
                result.append(((r, c), (r + 1, c)))
            if c + 1 < nc:
                result.append(((r, c), (r, c + 1)))
    return result


def _mesh(rng, row_coords, col_coords, h, w):
    nr, nc = len(row_coords), len(col_coords)
    edges = _logical_edges(nr, nc)
    # Randomized Kruskal supplies a connected base. Independent chords set density.
    order = rng.permutation(len(edges))
    parent = list(range(nr * nc))
    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    chosen = set()
    for ix in order:
        a, b = edges[int(ix)]
        x, y = a[0] * nc + a[1], b[0] * nc + b[1]
        rx, ry = root(x), root(y)
        if rx != ry:
            parent[rx] = ry
            chosen.add(int(ix))
    density = float(rng.uniform(0.18, 1.0))
    for ix in range(len(edges)):
        if ix not in chosen and rng.random() < density:
            chosen.add(ix)
    if len(chosen) == nr * nc - 1:
        available = [i for i in range(len(edges)) if i not in chosen]
        chosen.add(int(rng.choice(available)))
    cells = set()
    for ix in chosen:
        (ar, ac), (br, bc) = edges[ix]
        r1, c1 = row_coords[ar], col_coords[ac]
        r2, c2 = row_coords[br], col_coords[bc]
        for r in range(min(r1, r2), max(r1, r2) + 1):
            for c in range(min(c1, c2), max(c1, c2) + 1):
                cells.add((r, c))
    return cells


def _terminal_options(rows, cols, h, w, side):
    if side == 0:
        return [((0, c), (1, c)) for c in cols]
    if side == 1:
        return [((r, w - 1), (r, w - 2)) for r in rows]
    if side == 2:
        return [((h - 1, c), (h - 2, c)) for c in cols]
    return [((r, 0), (r, 1)) for r in rows]


def _side_pair(rng, relation):
    a = int(rng.integers(4))
    if relation == "same":
        b = a
    elif relation == "opposite":
        b = (a + 2) % 4
    else:
        b = (a + int(rng.choice((1, 3)))) % 4
    return a, b


def _make_pair(rng, relation, hazard_count, colors, demand=None):
    for _ in range(240):
        rows, h = _axis(rng)
        cols, w = _axis(rng)
        network = _mesh(rng, rows, cols, h, w)
        side_a, side_b = _side_pair(rng, relation)
        choices_a = _terminal_options(rows, cols, h, w, side_a)
        choices_b = _terminal_options(rows, cols, h, w, side_b)
        ta, ea = choices_a[int(rng.integers(len(choices_a)))]
        tb, eb = choices_b[int(rng.integers(len(choices_b)))]
        if ta == tb or ea == eb:
            continue
        for terminal, entry in ((ta, ea), (tb, eb)):
            # Connect exterior entries to the logical mesh.
            r, c = terminal
            dr = 0 if entry[0] == r else (1 if entry[0] > r else -1)
            dc = 0 if entry[1] == c else (1 if entry[1] > c else -1)
            p = (r + dr, c + dc)
            while p != entry:
                network.add(p)
                p = (p[0] + dr, p[1] + dc)
            network.add(entry)
        if ta in network or tb in network:
            continue
        # Candidate hazards can replace a rail cell (including junctions) or
        # sit beside a rail. Interior only keeps exterior markers unambiguous.
        candidates = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)
                      if (r, c) not in (ea, eb) and 1 <= sum(q in network for q in _nbr((r, c), h, w)) <= 4]
        if not candidates:
            continue
        raw = _route(network, ea, eb, h, w)
        if raw is None:
            continue
        for _ in range(85):
            hazard = {candidates[int(rng.integers(len(candidates)))]}
            for _ in range(hazard_count - 1):
                if rng.random() < 0.35:
                    pool = [q for q in candidates if q not in hazard and any(abs(q[0]-z[0])+abs(q[1]-z[1]) == 2 for z in hazard)]
                else:
                    pool = []
                if not pool:
                    pool = [q for q in candidates if q not in hazard]
                if not pool:
                    break
                hazard.add(pool[int(rng.integers(len(pool)))])
            if len(hazard) != hazard_count:
                continue
            rail = network - hazard
            neighborhoods = [{q for q in _nbr(z, h, w) if q in rail} for z in hazard]
            if any(not 1 <= len(s) <= 4 for s in neighborhoods):
                continue
            blocked = set().union(*neighborhoods)
            available = rail - blocked
            if ea not in available or eb not in available:
                continue
            result = _route(available, ea, eb, h, w)
            if result is None or result[2] != 1 or len(result[0]) < 7:
                continue
            path, score, _ = result
            if score <= raw[1] or len(rail - set(path)) < 3:
                continue
            routes = _simple_paths_two(available, ea, eb, h, w)
            if demand == "unique" and routes != 1:
                continue
            if demand == "compete" and routes < 2:
                continue
            if demand == "tie" and _shortest_paths_two(available, ea, eb, h, w) < 2:
                continue
            if demand == "loop" and not any(_edge_count(comp) >= len(comp) for comp in _components(available, h, w)):
                continue
            grid = np.full((h, w), colors[0], dtype=np.int8)
            for r, c in rail:
                grid[r, c] = colors[1]
            for r, c in hazard:
                grid[r, c] = colors[2]
            grid[ta], grid[tb] = colors[3], colors[3]
            output = grid.copy()
            for r, c in path:
                output[r, c] = colors[3]
            return {"input": grid, "output": output}
    raise ValueError("could not construct a valid rail puzzle")


def _key(grid):
    return (grid.shape, grid.tobytes())


def _source_copy(pair):
    a, b = pair["input"], pair["output"]
    for source_a, source_b in _SOURCE:
        if a.shape != (len(source_a), len(source_a[0])):
            continue
        if tuple("".join(map(str, row)) for row in a) == source_a and tuple("".join(map(str, row)) for row in b) == source_b:
            return True
    return False


def _analyze(pair):
    a, b = pair["input"], pair["output"]
    assert a.shape == b.shape and 1 <= a.shape[0] <= 30 and 1 <= a.shape[1] <= 30
    for grid in (a, b):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer)
        assert np.all((0 <= grid) & (grid <= 9))
    changed = a != b
    assert np.any(changed)
    source, target = set(map(int, a[changed])), set(map(int, b[changed]))
    assert len(source) == len(target) == 1
    rail_color, terminal_color = next(iter(source)), next(iter(target))
    h, w = a.shape
    bg = int(a[0, 0])
    assert int(a[0, w - 1]) == int(a[h - 1, 0]) == int(a[h - 1, w - 1]) == bg
    colors = set(map(int, np.unique(a)))
    assert len(colors) == 4 and set(map(int, np.unique(b))) == colors
    hazard_color, = colors - {bg, rail_color, terminal_color}
    rail = {(int(r), int(c)) for r, c in np.argwhere(a == rail_color)}
    hazard = {(int(r), int(c)) for r, c in np.argwhere(a == hazard_color)}
    terminals = {(int(r), int(c)) for r, c in np.argwhere(a == terminal_color)}
    assert len(terminals) == 2 and 1 <= len(hazard) <= 3
    assert all(r in (0, h - 1) or c in (0, w - 1) for r, c in terminals)
    entries = []
    for t in sorted(terminals):
        adjacent = [q for q in _nbr(t, h, w) if q in rail]
        assert len(adjacent) == 1
        entries.append(adjacent[0])
    neighbors = [{q for q in _nbr(z, h, w) if q in rail} for z in hazard]
    assert all(1 <= len(s) <= 4 for s in neighbors)
    blocked = set().union(*neighbors)
    available = rail - blocked
    result = _route(available, entries[0], entries[1], h, w)
    assert result is not None and result[2] == 1
    path, score, _ = result
    assert set(path) == {(int(r), int(c)) for r, c in np.argwhere(changed)}
    expected = a.copy()
    for r, c in path:
        expected[r, c] = terminal_color
    assert np.array_equal(b, expected)
    assert not _source_copy(pair)
    def side(t):
        if t[0] == 0: return 0
        if t[1] == w - 1: return 1
        if t[0] == h - 1: return 2
        return 3
    sides = tuple(side(t) for t in sorted(terminals))
    relation = "same" if sides[0] == sides[1] else ("opposite" if (sides[0] - sides[1]) % 4 == 2 else "adjacent")
    comps = _components(available, h, w)
    return {"relation": relation, "sides": sides, "hazards": len(hazard), "neighbors": tuple(sorted(map(len, neighbors))),
            "overlap": sum(map(len, neighbors)) > len(blocked), "shape": (h, w), "path": len(path),
            "raw_cycles": _edge_count(rail) - len(rail) + len(_components(rail, h, w)),
            "available_cycles": sum(_edge_count(x) - len(x) + 1 for x in comps),
            "remnants": len(comps) - 1, "competing": _simple_paths_two(available, entries[0], entries[1], h, w) > 1,
            "shortest_tie": _shortest_paths_two(available, entries[0], entries[1], h, w) > 1,
            "roles": (bg, rail_color, hazard_color, terminal_color)}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            key = _key(pair["input"])
            assert key not in seen
            seen.add(key)
            analyses[split].append(_analyze(pair))
    # Both route color and hazard avoidance are explicit in every demonstration.
    assert all(x["path"] >= 7 for x in analyses["train"])
    if any(x["shortest_tie"] for x in analyses["test"]):
        assert any(x["shortest_tie"] for x in analyses["train"])
    return analyses


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    nt, ns = _counts(seed, rng, num_train, num_test)
    base_palette = tuple(map(int, rng.choice(10, size=4, replace=False)))
    recolor_pairs = bool(rng.integers(2))
    def palette():
        return tuple(map(int, rng.choice(10, size=4, replace=False))) if recolor_pairs else base_palette
    train, test, seen = [], [], set()
    # Relations and hazard counts have separate draws; the first two training
    # pairs span two relations when possible, with no count relation coupling.
    train_relations = list(rng.choice(("opposite", "adjacent", "same"), size=nt, p=(0.45, 0.45, 0.10)))
    if len(set(train_relations)) == 1:
        train_relations[1] = ("adjacent" if train_relations[0] == "opposite" else "opposite")
    train_hazards = [int(rng.integers(1, 4)) for _ in range(nt)]
    if len(set(train_hazards)) == 1:
        train_hazards[1] = 1 if train_hazards[0] != 1 else 2
    for i in range(nt):
        colors = palette()
        demand = "unique" if i == 0 else ("tie" if i == 1 else None)
        for _ in range(12):
            pair = _make_pair(rng, train_relations[i], train_hazards[i], colors, demand)
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                train.append(pair)
                break
        else:
            raise ValueError("could not create distinct training inputs")
    train_has_competition = any(_analyze(pair)["shortest_tie"] for pair in train)
    for _ in range(ns):
        colors = palette()
        relation = str(rng.choice(("opposite", "adjacent", "same"), p=(0.2, 0.2, 0.6)))
        hazards = int(rng.integers(1, 4))
        demand = None if train_has_competition else "unique"
        for _ in range(12):
            pair = _make_pair(rng, relation, hazards, colors, demand)
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                test.append(pair)
                break
        else:
            raise ValueError("could not create distinct test inputs")
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    analyses = [validate_generated_puzzle(p) for p in puzzles]
    assert len({(len(p["train"]), len(p["test"])) for p in puzzles}) >= 8
    all_pairs = [x for p in analyses for split in ("train", "test") for x in p[split]]
    assert {x["relation"] for x in all_pairs} == {"same", "adjacent", "opposite"}
    assert {x["hazards"] for x in all_pairs} == {1, 2, 3}
    assert any(x["overlap"] for x in all_pairs)
    assert any(x["available_cycles"] for x in all_pairs)
    assert any(x["remnants"] for x in all_pairs)
    assert any(max(x["neighbors"]) >= 4 for x in all_pairs)
    assert any(x["shape"][0] <= 12 and x["shape"][1] <= 12 for x in all_pairs)
    assert len({x["shape"] for x in all_pairs}) >= 25
    assert len({x["roles"] for x in all_pairs}) >= 20
    for nt, ns in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(2000 + nt + ns, nt, ns)
        assert len(p["train"]) == nt and len(p["test"]) == ns
    for kwargs in ({"num_train": 1}, {"num_test": 5}, {"num_train": 2, "num_test": 2}):
        try:
            generate_puzzle(1, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid pair count accepted")
    a, b = generate_puzzle(923, 6, 3), generate_puzzle(923, 6, 3)
    assert all(np.array_equal(x[k], y[k]) for s in ("train", "test") for x, y in zip(a[s], b[s]) for k in ("input", "output"))
