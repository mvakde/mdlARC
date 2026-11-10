"""Ghost rails: periodic latent connectivity and selective minimum-hop routing.

Only standard-library modules and NumPy are used. The solver reads its input;
the constructor independently paints the route in its sampled clean graph.
"""
from collections import deque
import heapq
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "invalid grid shape")
    _require(np.issubdtype(g.dtype, np.integer), "noninteger grid")
    _require(bool(np.all((g >= 0) & (g <= 9))), "invalid colors")
    return g.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
        result.append(np.array(cells, dtype=int))
    return result


def _square_size(cells):
    shape = cells.max(0) - cells.min(0) + 1
    return int(shape[0]) if shape[0] == shape[1] and len(cells) == int(np.prod(shape)) else 0


def _recover(h, v):
    """Require agreement among all supported majority-periodic explanations."""
    candidates = []
    cap = max(1, (h.size + v.size) // 8)
    for period in range(1, h.shape[1] // 3 + 1):
        repaired, errors, valid = [], 0, True
        for matrix in (h, v):
            clean = np.empty_like(matrix)
            for row in range(matrix.shape[0]):
                for phase in range(period):
                    samples = matrix[row, phase::period]
                    ones = int(samples.sum())
                    if 2 * ones == len(samples):
                        valid = False
                        break
                    bit = int(2 * ones > len(samples))
                    clean[row, phase::period] = bit
                    errors += min(ones, len(samples) - ones)
                if not valid:
                    break
            if not valid:
                break
            repaired.append(clean)
        if valid and errors <= cap:
            candidates.append((period, repaired[0], repaired[1], errors))
    _require(bool(candidates), "no supported periodic graph")
    first = candidates[0]
    _require(all(np.array_equal(first[1], c[1]) and np.array_equal(first[2], c[2])
                 for c in candidates), "ambiguous periodic graph")
    return first[1], first[2], first[0]


def _edge_slots(rows, cols, size, offsets=None):
    """All potential edges, including absent ones; order is H then V."""
    slots, edges, positions = [], [], []
    nr, nc = len(rows), len(cols)
    horizontal_offset, vertical_offset = offsets or (size // 2, size // 2)
    for r in range(nr):
        for c in range(nc - 1):
            slots.append((slice(rows[r] + horizontal_offset, rows[r] + horizontal_offset + 1),
                          slice(cols[c] + size, cols[c + 1])))
            edges.append((r * nc + c, r * nc + c + 1))
            positions.append((0, r, c))
    for r in range(nr - 1):
        for c in range(nc):
            slots.append((slice(rows[r] + size, rows[r + 1]),
                          slice(cols[c] + vertical_offset, cols[c] + vertical_offset + 1)))
            edges.append((r * nc + c, (r + 1) * nc + c))
            positions.append((1, r, c))
    return slots, edges, positions


def _adjacency(n, edges, bits):
    adj = [[] for _ in range(n)]
    for e, ((a, b), present) in enumerate(zip(edges, bits)):
        if present:
            adj[a].append((b, e))
            adj[b].append((a, e))
    return adj


def _bfs(adj, start):
    distances = [-1] * len(adj)
    ways = [0] * len(adj)
    previous = [None] * len(adj)
    distances[start], ways[start] = 0, 1
    queue = deque([start])
    while queue:
        a = queue.popleft()
        for b, e in adj[a]:
            if distances[b] < 0:
                distances[b] = distances[a] + 1
                previous[b] = a, e
                queue.append(b)
            if distances[b] == distances[a] + 1:
                ways[b] = min(2, ways[b] + ways[a])
    return distances, ways, previous


def _path(search, start, target):
    _, ways, previous = search
    if ways[target] != 1:
        return None
    nodes, edges = [target], []
    while nodes[-1] != start:
        a, e = previous[nodes[-1]]
        nodes.append(a)
        edges.append(e)
    return nodes[::-1], edges[::-1]


def _weighted_distances(adj, start, weights):
    best = [float("inf")] * len(adj)
    best[start] = 0
    todo = [(0, start)]
    while todo:
        cost, a = heapq.heappop(todo)
        if cost != best[a]:
            continue
        for b, e in adj[a]:
            alt = cost + weights[e]
            if alt < best[b]:
                best[b] = alt
                heapq.heappush(todo, (alt, b))
    return best


def _read_input(input_grid):
    g = _grid(input_grid)
    colors = [int(c) for c in np.unique(g)]
    _require(len(colors) == 4, "four input color roles required")
    bg = int(g[0, 0])
    _require(bool(np.all(g[0] == bg) and np.all(g[:, 0] == bg)), "missing background margins")
    by_color = {c: _components(g == c) for c in colors if c != bg}
    endpoints = [c for c, parts in by_color.items() if len(parts) == 2
                 and _square_size(parts[0]) in (2, 3)
                 and _square_size(parts[1]) == _square_size(parts[0])]
    _require(len(endpoints) == 1, "endpoint color ambiguous")
    ep = endpoints[0]
    size = _square_size(by_color[ep][0])
    ordinary = [c for c, parts in by_color.items() if c != ep and len(parts) >= 3
                and all(_square_size(p) == size for p in parts)]
    _require(len(ordinary) == 1, "ordinary station color ambiguous")
    st = ordinary[0]
    link = next(c for c in colors if c not in (bg, st, ep))
    blocks = by_color[ep] + by_color[st]
    tops = [tuple(p.min(0)) for p in blocks]
    rows = sorted({r for r, _ in tops})
    cols = sorted({c for _, c in tops})
    _require(len(rows) != len(cols), "rhythm axis ambiguous")
    transposed = len(rows) > len(cols)
    if transposed:
        g = g.T.copy()
        rows, cols = cols, rows
        tops = [(c, r) for r, c in tops]
    nr, nc = len(rows), len(cols)
    _require(3 <= nr <= 6 and 7 <= nc <= 10, "unsupported lattice counts")
    _require(size == 2 or nc == 7, "size-three lattice too wide")
    _require(set(tops) == {(r, c) for r in rows for c in cols}, "incomplete station lattice")
    _require(all(b - a > size for axis in (rows, cols) for a, b in zip(axis, axis[1:])),
             "station boxes must have a gap")
    # Reflections can exchange the two central lines of an even square.
    # Recover the global attachment offsets from visible links, never a seed.
    hoff, voff = set(), set()
    for top in rows:
        for left, right in zip(cols, cols[1:]):
            pixels = np.argwhere(g[top:top + size, left + size:right] == link)
            hoff.update(int(x) for x in pixels[:, 0])
    for top, bottom in zip(rows, rows[1:]):
        for left in cols:
            pixels = np.argwhere(g[top + size:bottom, left:left + size] == link)
            voff.update(int(x) for x in pixels[:, 1])
    _require(len(hoff) == len(voff) == 1, "inconsistent attachment offsets")
    offsets = next(iter(hoff)), next(iter(voff))
    _require(size == 2 or offsets == (1, 1), "noncentral link attachment")
    slots, edges, positions = _edge_slots(rows, cols, size, offsets)
    h = np.empty((nr, nc - 1), dtype=np.int8)
    v = np.empty((nr - 1, nc), dtype=np.int8)
    expected = np.full_like(g, bg)
    endpoint_ids = []
    for r, top in enumerate(rows):
        for c, left in enumerate(cols):
            box = (slice(top, top + size), slice(left, left + size))
            color = int(g[top, left])
            _require(color in (st, ep) and bool(np.all(g[box] == color)), "damaged station")
            expected[box] = color
            if color == ep:
                endpoint_ids.append(r * nc + c)
    for slot, (kind, r, c) in zip(slots, positions):
        values = np.unique(g[slot])
        _require(len(values) == 1 and int(values[0]) in (bg, link), "partially damaged link slot")
        (h if kind == 0 else v)[r, c] = int(values[0] == link)
        expected[slot] = values[0]
    _require(np.array_equal(g, expected), "pixels outside station/slot geometry")
    clean_h, clean_v, period = _recover(h, v)
    bits = np.concatenate((clean_h.ravel(), clean_v.ravel()))
    _require(set(int(x) for x in np.unique(bits)) == {0, 1}, "repeated vector must contain both states")
    raw_bits = np.concatenate((h.ravel(), v.ravel()))
    adj = _adjacency(nr * nc, edges, bits)
    search = _bfs(adj, endpoint_ids[0])
    _require(all(x >= 0 for x in search[0]), "disconnected intended graph")
    path = _path(search, *endpoint_ids)
    _require(path is not None, "shortest station route tied")
    return dict(grid=g, transposed=transposed, roles=(bg, st, ep, link), size=size,
                rows=rows, cols=cols, slots=slots, edges=edges, positions=positions,
                h=h, v=v, clean_h=clean_h, clean_v=clean_v, bits=bits, raw_bits=raw_bits,
                period=period, adjacency=adj, endpoints=endpoint_ids, path=path,
                weights=[int(g[s].size) for s in slots])


def _paint(grid, rows, cols, size, slots, path, color):
    result = grid.copy()
    nc = len(cols)
    for node in path[0][1:-1]:
        r, c = divmod(node, nc)
        result[rows[r]:rows[r] + size, cols[c]:cols[c] + size] = color
    for edge in path[1]:
        result[slots[edge]] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    d = _read_input(input_grid)
    result = _paint(d["grid"], d["rows"], d["cols"], d["size"], d["slots"],
                    d["path"], d["roles"][2])
    return result.T.copy() if d["transposed"] else result


def _axis(rng, count, size):
    budget = 29 - size * count - (count - 1)
    _require(budget >= 0, "layout exceeds budget")
    extra = int(rng.integers(budget + 1))
    gaps = np.ones(count - 1, dtype=int)
    if extra:
        gaps += rng.multinomial(extra, rng.dirichlet(np.full(count - 1, .45)))
    return [0] + list(np.cumsum(size + gaps).astype(int))


def _choose_path(rng, adj, edges, bits, weights, size, kind):
    choices = []
    absent = {tuple(edge): e for e, edge in enumerate(edges) if not bits[e]}
    target_length = int(rng.integers(2, 11))
    for a in range(len(adj)):
        search = _bfs(adj, a)
        if kind == "metric":
            pixel_best = _weighted_distances(adj, a, weights)
            center_best = _weighted_distances(adj, a, [w + size for w in weights])
        for b in range(a + 1, len(adj)):
            if search[1][b] != 1 or search[0][b] < 2:
                continue
            if kind == "topology" and (a, b) not in absent:
                continue
            path = _path(search, a, b)
            if kind == "metric":
                cost = sum(weights[e] for e in path[1])
                if cost <= pixel_best[b] or cost + size * len(path[1]) <= center_best[b]:
                    continue
            choices.append((abs(len(path[1]) - target_length), a, b, path, absent.get((a, b))))
    if not choices:
        return None
    best = min(c[0] for c in choices)
    pool = [c for c in choices if c[0] <= best + 1]
    return pool[int(rng.integers(len(pool)))][1:]


def _corrupt(rng, h, v, period, path_edges, ghost, positions):
    bits = np.concatenate((h.ravel(), v.ravel()))
    noisy = bits.copy()
    first = int(rng.choice(path_edges))  # Always erase a selected link.
    used = {tuple(positions[first][:2]) + (positions[first][2] % period,)}
    groups = [tuple(p[:2]) + (p[2] % period,) for p in positions]
    outside = [e for e in range(len(bits)) if e not in path_edges and groups[e] not in used]
    if ghost is not None:
        if groups[ghost] in used:
            # Try another selected edge rather than violating sparse phase noise.
            options = [e for e in path_edges if groups[e] != groups[ghost]]
            if not options:
                return None
            first = int(rng.choice(options))
            used = {groups[first]}
        second = ghost
    elif outside:
        second = int(rng.choice(outside))
    else:
        return None
    chosen = [first, second]
    used.add(groups[second])
    for _ in range(int(rng.integers(3))):
        options = [e for e in range(len(bits)) if groups[e] not in used]
        if not options:
            break
        e = int(rng.choice(options))
        chosen.append(e)
        used.add(groups[e])
    noisy[chosen] ^= 1
    cut = h.size
    nh, nv = noisy[:cut].reshape(h.shape), noisy[cut:].reshape(v.shape)
    try:
        rh, rv, _ = _recover(nh, nv)
    except ValueError:
        return None
    if not (np.array_equal(rh, h) and np.array_equal(rv, v)):
        return None
    return nh, nv


def _place(rng, grid, vertical):
    # Transform foreground first; background top/left are then supplied anew.
    if rng.random() < .5:
        grid = grid[::-1]
    if rng.random() < .5:
        grid = grid[:, ::-1]
    if vertical:
        grid = grid.T
    h, w = grid.shape
    hh, ww = int(rng.integers(h + 1, 31)), int(rng.integers(w + 1, 31))
    top, left = int(rng.integers(1, hh - h + 1)), int(rng.integers(1, ww - w + 1))
    result = np.zeros((hh, ww), dtype=np.int8)
    result[top:top + h, left:left + w] = grid
    return result


def _render_pair(rng, rows, cols, size, slots, path, endpoints, observed, vertical, roles):
    canvas = np.zeros((rows[-1] + size, cols[-1] + size), dtype=np.int8)
    for slot, bit in zip(slots, observed):
        canvas[slot] = 3 if bit else 0
    for r, top in enumerate(rows):
        for c, left in enumerate(cols):
            canvas[top:top + size, left:left + size] = 2 if r * len(cols) + c in endpoints else 1
    # Paint from the clean sampled graph, before any solver parsing.
    output = _paint(canvas, rows, cols, size, slots, path, 2)
    # Stack labels so transformation and margins are identical for both arrays.
    placed = _place(rng, canvas + 4 * output, vertical)
    lookup = np.asarray(roles, dtype=np.int8)
    return {"input": lookup[placed % 4], "output": lookup[placed // 4]}


def _fallback_pair(rng, nr, vertical, roles, kind):
    """Construct a valid member if bounded random template proposals run out.

    Alternating middle-row links and alternating downward links force a
    three-edge upper bypass, with a five-edge physically shorter lower bypass.
    Extra transverse rows simply extend the same connected periodic graph.
    """
    size, nc, period = 2, 7, 2
    rows = [0, 12] + [15 + 3 * r for r in range(nr - 2)]
    cols = [3 * c for c in range(nc)]
    h, v = np.ones((nr, nc - 1), dtype=np.int8), np.ones((nr - 1, nc), dtype=np.int8)
    h[1] = np.arange(nc - 1) % 2
    v[1] = np.arange(nc) % 2
    bits = np.concatenate((h.ravel(), v.ravel()))
    slots, edges, positions = _edge_slots(rows, cols, size)
    adj = _adjacency(nr * nc, edges, bits)
    weights = [(s[0].stop - s[0].start) * (s[1].stop - s[1].start) for s in slots]
    if kind == "topology":
        a, b = nc + 2, 2 * nc + 2
        path = _path(_bfs(adj, a), a, b)
        ghost = next(e for e, p in enumerate(positions) if p == (1, 1, 2))
        _require(path is not None, "fallback topology path tied")
    else:
        choice = _choose_path(rng, adj, edges, bits, weights, size, kind)
        _require(choice is not None, "constructive fallback lacks requested witness")
        a, b, path, ghost = choice
        # This intended-absent edge cannot be on any clean route.
        ghost = next(e for e, p in enumerate(positions) if p == (1, 1, 0))
    groups = [(k, r, c % period) for k, r, c in positions]
    first = next(e for e in path[1] if positions[e][:2] != (0, 1) and groups[e] != groups[ghost])
    observed = bits.copy()
    observed[[first, ghost]] ^= 1
    cut = h.size
    rh, rv, _ = _recover(observed[:cut].reshape(h.shape), observed[cut:].reshape(v.shape))
    _require(np.array_equal(rh, h) and np.array_equal(rv, v), "fallback recovery failed")
    pair = _render_pair(rng, rows, cols, size, slots, path, (a, b), observed, vertical, roles)
    d = _analyze_pair(pair)
    _require(kind != "topology" or _topology_witness(d), "fallback topology witness failed")
    _require(kind != "metric" or _metric_witness(d), "fallback metric witness failed")
    return pair


def _make_pair(rng, nr, vertical, roles, kind):
    for _ in range(250):
        size = int(rng.choice([2, 3], p=[.8, .2]))
        nc = 7 if size == 3 else int(rng.integers(7, 11))
        period = int(rng.choice(np.arange(1, (nc - 1) // 3 + 1),
                                p=([.12, .88] if nc < 10 else [.08, .55, .37])))
        rows, cols = _axis(rng, nr, size), _axis(rng, nc, size)
        slots, edges, positions = _edge_slots(rows, cols, size)
        ht = (rng.random((nr, period)) < rng.uniform(.55, .92)).astype(np.int8)
        vt = (rng.random((nr - 1, period)) < rng.uniform(.75, .98)).astype(np.int8)
        h, v = ht[:, np.arange(nc - 1) % period], vt[:, np.arange(nc) % period]
        bits = np.concatenate((h.ravel(), v.ravel()))
        if len(np.unique(bits)) != 2:
            continue
        try:
            rh, rv, rp = _recover(h, v)
        except ValueError:
            continue
        if rp != period or not (np.array_equal(rh, h) and np.array_equal(rv, v)):
            continue
        adj = _adjacency(nr * nc, edges, bits)
        if any(x < 0 for x in _bfs(adj, 0)[0]):
            continue
        weights = [(s[0].stop - s[0].start) * (s[1].stop - s[1].start) for s in slots]
        choice = _choose_path(rng, adj, edges, bits, weights, size, kind)
        if choice is None:
            continue
        a, b, path, ghost = choice
        for _ in range(12):
            corrupt = _corrupt(rng, h, v, period, path[1], ghost if kind == "topology" else None,
                               positions)
            if corrupt is not None:
                break
        else:
            continue
        nh, nv = corrupt
        observed = np.concatenate((nh.ravel(), nv.ravel()))
        pair = _render_pair(rng, rows, cols, size, slots, path, (a, b), observed, vertical, roles)
        d = _read_input(pair["input"])
        if kind == "topology" and not _topology_witness(d):
            continue
        if kind == "metric" and not _metric_witness(d):
            continue
        _analyze_pair(pair)
        return pair
    return _fallback_pair(rng, nr, vertical, roles, kind)


def _topology_witness(d):
    raw_adj = _adjacency(len(d["adjacency"]), d["edges"], d["raw_bits"])
    raw = _path(_bfs(raw_adj, d["endpoints"][0]), *d["endpoints"])
    return raw is not None and set(raw[1]) != set(d["path"][1])


def _metric_witness(d):
    a, b = d["endpoints"]
    cost = sum(d["weights"][e] for e in d["path"][1])
    return (cost > _weighted_distances(d["adjacency"], a, d["weights"])[b] and
            cost + d["size"] * len(d["path"][1]) >
            _weighted_distances(d["adjacency"], a, [w + d["size"] for w in d["weights"]])[b])


def _analyze_pair(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    _require(inp.shape == out.shape, "output shape changed")
    d = _read_input(inp)
    canonical_out = out.T if d["transposed"] else out
    expected = _paint(d["grid"], d["rows"], d["cols"], d["size"], d["slots"],
                      d["path"], d["roles"][2])
    _require(np.array_equal(canonical_out, expected), "wrong path or off-route edit")
    errors = set(np.flatnonzero(d["bits"] != d["raw_bits"]).tolist())
    selected = set(d["path"][1])
    _require(2 <= len(errors) <= 4, "corruption count outside generated range")
    error_groups = {(d["positions"][e][0], d["positions"][e][1],
                     d["positions"][e][2] % d["period"]) for e in errors}
    _require(len(error_groups) == len(errors), "multiple flips in one phase/channel group")
    _require(bool(errors - selected), "missing preserved off-route defect")
    _require(any(d["raw_bits"][e] == 0 for e in selected), "no selective missing-link completion")
    _require(len(d["path"][0]) >= 3, "generated route needs an ordinary station")
    for e in errors - selected:
        _require(np.array_equal(canonical_out[d["slots"][e]], d["grid"][d["slots"][e]]),
                 "off-route defect repaired")
    return d


def _key(grid):
    g = np.asarray(grid, dtype=np.int8)
    return g.shape, g.tobytes()


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, data = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            d = _analyze_pair(pair)
            key = _key(pair["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            data[split].append(d)
    train = data["train"]
    _require(any(_topology_witness(d) for d in train), "no training topology witness")
    _require(any(_metric_witness(d) for d in train), "no training station-count witness")
    _require({d["transposed"] for d in train} == {False, True}, "training needs both axes")
    _require({len(d["rows"]) for d in train[:2]} == {3, 4}, "first demonstrations need row variation")
    _require(all(3 <= len(d["rows"]) <= 5 for d in train), "invalid train transverse counts")
    _require(all(5 <= len(d["rows"]) <= 6 for d in data["test"]), "invalid test transverse counts")


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not lo <= int(value) <= hi:
            raise ValueError(f"{name} outside legal range")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    roles = tuple(int(c) for c in rng.choice(10, 4, replace=False))
    fresh_palette = bool(rng.integers(3))
    kinds = ["topology", "metric"]
    rng.shuffle(kinds)
    row_plan = [3, 4]
    rng.shuffle(row_plan)
    axes = [False, True]
    rng.shuffle(axes)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            kind = kinds[i] if split == "train" and i < 2 else "any"
            nr = row_plan[i] if split == "train" and i < 2 else int(rng.integers(3, 6) if split == "train" else rng.integers(5, 7))
            vertical = axes[i] if split == "train" and i < 2 else bool(rng.integers(2))
            palette = tuple(int(c) for c in rng.choice(10, 4, replace=False)) if fresh_palette else roles
            for _ in range(8):
                pair = _make_pair(rng, nr, vertical, palette, kind)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test():
    """Bounded checks of semantic witnesses, diversity and public interfaces."""
    signatures, shapes, structures, periods, sizes, counts, paths, palettes = (set() for _ in range(8))
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _analyze_pair(pair)
                assert np.array_equal(solve(pair["input"]), pair["output"])
                shapes.add(pair["input"].shape)
                structures.add((d["h"].shape, d["h"].tobytes(), d["v"].tobytes()))
                periods.add(d["period"])
                sizes.add(d["size"])
                counts.add(len(d["adjacency"]))
                paths.add(len(d["path"][1]))
                palettes.add(d["roles"])
        if seed < 3:
            again = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle for p, q in zip(puzzle[s], again[s])
                       for k in ("input", "output"))
            pair = puzzle["test"][0]
            bg = int(pair["input"][0, 0])
            foreground = np.argwhere(pair["input"] != bg)
            lo, hi = foreground.min(0), foreground.max(0) + 1
            cropped = {k: g[lo[0]:hi[0], lo[1]:hi[1]] for k, g in pair.items()}
            for transform in (lambda g: g.T, lambda g: g[::-1],
                              lambda g: g[:, ::-1], lambda g: np.rot90(g)):
                transformed = {k: np.pad(transform(g), ((1, 0), (1, 0)), constant_values=bg)
                               for k, g in cropped.items()}
                assert np.array_equal(solve(transformed["input"]), transformed["output"])
    assert len(signatures) == 42 and len(structures) >= 150
    assert len(shapes) >= 35 and len(counts) >= 10 and len(paths) >= 6
    assert periods == {1, 2, 3} and sizes == {2, 3} and len(palettes) >= 35
    # Force the constructive path, rather than hoping random proposals use it.
    for nr in range(3, 7):
        for kind in ("topology", "metric", "any"):
            for vertical in (False, True):
                p = _fallback_pair(np.random.default_rng(100 * nr + len(kind) + vertical),
                                   nr, vertical, (0, 1, 2, 3), kind)
                assert np.array_equal(solve(p["input"]), p["output"])
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            generate_puzzle(1000 + 10 * train + test, train, test)
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # BFS multiplicity must reject tied routes rather than use neighbor order.
    cycle = _adjacency(4, [(0, 1), (1, 2), (2, 3), (3, 0)], [1] * 4)
    assert _path(_bfs(cycle, 0), 0, 2) is None
    assert _path(_bfs(cycle, 0), 0, 1) == ([0, 1], [0])
    # Enumerate every simple four-node graph, including disconnected cases.
    # This checks the graph primitive; it is not a second puzzle solver.
    possible = [(a, b) for a in range(4) for b in range(a + 1, 4)]
    for mask in range(1 << len(possible)):
        adj = _adjacency(4, possible, [(mask >> e) & 1 for e in range(len(possible))])
        for start in range(4):
            search = _bfs(adj, start)
            for target in range(start + 1, 4):
                found, stack = [], [(start, [start], [])]
                while stack:
                    a, nodes, edges = stack.pop()
                    if a == target:
                        found.append((nodes, edges))
                    else:
                        stack.extend((b, nodes + [b], edges + [e]) for b, e in adj[a] if b not in nodes)
                found.sort(key=lambda p: len(p[1]))
                unique = bool(found) and (len(found) == 1 or len(found[0][1]) < len(found[1][1]))
                actual = _path(search, start, target)
                assert (actual is not None) == unique
                if unique:
                    assert actual == found[0]
    # Majority ties and incompatible period repairs must not be silently chosen.
    try:
        _recover(np.array([[0, 0, 0, 1, 1, 1]], dtype=np.int8),
                 np.array([[0, 0, 0, 1, 1, 1, 0]], dtype=np.int8))
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported periodic evidence accepted")
    # A sparse alternating channel has two supported but conflicting repairs.
    h = np.zeros((3, 8), dtype=np.int8)
    h[0] = [0, 0, 0, 1, 0, 1, 0, 1]
    try:
        _recover(h, np.zeros((2, 9), dtype=np.int8))
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting periodic repairs accepted")
    # A label perturbation must fail exact preservation/painting checks.
    puzzle = generate_puzzle(81, 2, 1)
    puzzle["test"][0]["output"][0, 0] = (int(puzzle["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(puzzle)
    except ValueError:
        pass
    else:
        raise AssertionError("altered output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 diversity seeds and all legal counts)")
