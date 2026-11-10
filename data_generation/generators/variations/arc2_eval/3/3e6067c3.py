"""Ordered, distinct-station routes with source-colored, patch-width gap bands.

All semantic conventions are fixed by the family. Palette policy and complexity
are puzzle-level sampling choices, never hidden information required to solve.
"""
from hashlib import sha256

import numpy as np


# Exact source pair contents (shape, input bytes, output bytes), not shape bans.
_SOURCE_PAIR_DIGESTS = {
    "9b8836487f3bc090b464ea5e1ae06fe2ae6ebdf08be7058ba26e84b4ac5123ee",
    "e25874a782388feef05359165a9d5a8fbfb65c0d9d028f68128f229adc7d0c6b",
    "4d074918eee8bd5019bcf9500287ce94ea9ba82729a4abf0edef1826ffaa9ce4",
    "528d284d945ab77e127403337a37855b93de7e0ae73d58b53d78d064c3276bc3",
    "5c3338b36761d4c443653e88cebeba39c3eec21712f0875c101112eb34a8b201",
}


def _check_optional_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def _resolve_counts(rng, num_train, num_test):
    nt = _check_optional_count(num_train, "num_train", 2, 10)
    nq = _check_optional_count(num_test, "num_test", 1, 4)
    if nt is not None and nq is not None:
        if nt <= nq:
            raise ValueError("num_train must exceed num_test")
    elif nt is None and nq is None:
        nq = int(rng.integers(1, 5))
        nt = int(rng.integers(max(2, nq + 1), 11))
    elif nt is None:
        nt = int(rng.integers(max(2, nq + 1), 11))
    else:
        nq = int(rng.integers(1, min(4, nt - 1) + 1))
    return nt, nq


def _grid_signature(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _pair_digest(pair):
    return sha256(repr(pair["input"].shape).encode()
                  + np.asarray(pair["input"], dtype=np.int8).tobytes()
                  + np.asarray(pair["output"], dtype=np.int8).tobytes()).hexdigest()


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for y, x in zip(*np.nonzero(mask)):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack, component = [(y, x)], []
        while stack:
            r, c = stack.pop()
            component.append((r, c))
            for yy, xx in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]
                        and mask[yy, xx] and not seen[yy, xx]):
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(component)
    return result


def _parse_input(grid):
    # The surrounding frame identifies background even in carrier-heavy scenes.
    background = int(grid[0, 0])
    assert all(np.all(edge == background) for edge in
               (grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    stations, tokens = [], []
    signature = None
    for component in _components(grid != background):
        if len(component) == 1:
            tokens.append(component[0])
            continue
        ys, xs = zip(*component)
        y, x = min(ys), min(xs)
        size = max(ys) - y + 1
        assert size >= 3 and max(xs) - x + 1 == size
        assert len(component) == size * size
        block = grid[y:y+size, x:x+size]
        carrier = int(block[0, 0])
        py, px = np.nonzero(block != carrier)
        assert len(py)
        offset = int(py.min())
        patch = int(py.max()) - offset + 1
        assert 0 < patch < size and 2 * offset + patch == size and offset > 0
        assert int(px.min()) == offset and int(px.max()) == offset + patch - 1
        assert len(py) == patch * patch
        label = int(block[offset, offset])
        assert label not in (background, carrier)
        assert np.all(block[offset:offset+patch, offset:offset+patch] == label)
        current = (size, patch, carrier)
        assert signature is None or current == signature
        signature = current
        stations.append((y, x, label))
    assert signature is not None and len(stations) >= 3
    size, patch, carrier = signature
    assert 3 <= len(tokens) <= 14
    tokens.sort(key=lambda t: t[1])
    key_y = tokens[0][0]
    assert all(y == key_y for y, x in tokens)
    assert all(b[1] - a[1] == 2 for a, b in zip(tokens, tokens[1:]))
    assert key_y >= max(s[0] + size for s in stations) + 1
    key = [int(grid[y, x]) for y, x in tokens]
    assert all(color not in (background, carrier) for color in key)
    assert len(set(key)) >= 2
    stations.sort()
    return dict(background=background, carrier=carrier, size=size, patch=patch,
                stations=stations, key=key, key_y=key_y, key_x=tokens[0][1])


def _band(a, b, size, patch):
    offset = (size - patch) // 2
    if a[0] == b[0] and abs(a[1] - b[1]) > size:
        return (slice(a[0]+offset, a[0]+offset+patch),
                slice(min(a[1], b[1])+size, max(a[1], b[1])))
    if a[1] == b[1] and abs(a[0] - b[0]) > size:
        return (slice(min(a[0], b[0])+size, max(a[0], b[0])),
                slice(a[1]+offset, a[1]+offset+patch))
    return None


def _visible_graph(cells):
    graph = [[] for _ in cells]
    for i, a in enumerate(cells):
        for j, b in enumerate(cells):
            if i == j:
                continue
            if a[0] == b[0]:
                lo, hi = sorted((a[1], b[1]))
                clear = not any(c[0] == a[0] and lo < c[1] < hi for c in cells)
            elif a[1] == b[1]:
                lo, hi = sorted((a[0], b[0]))
                clear = not any(c[1] == a[1] and lo < c[0] < hi for c in cells)
            else:
                continue
            if clear:
                graph[i].append(j)
    return graph


def _route_solutions(graph, labels, key, limit=2):
    solutions = []
    candidates = [[i for i, color in enumerate(labels) if color == k] for k in key]
    def visit(path, used):
        if len(solutions) >= limit:
            return
        if len(path) == len(key):
            solutions.append(tuple(path))
            return
        for i in candidates[len(path)]:
            if i not in used and (not path or i in graph[path[-1]]):
                visit(path + [i], used | {i})
    visit([], set())
    return solutions


def _solve_input(grid):
    info = _parse_input(grid)
    stations = info["stations"]
    # Unlike construction's lattice graph, decoding inspects full pixel bands.
    graph = [[] for _ in stations]
    for i, a in enumerate(stations):
        for j, b in enumerate(stations):
            if i == j:
                continue
            band = _band(a, b, info["size"], info["patch"])
            if band is not None and np.all(grid[band] == info["background"]):
                graph[i].append(j)
    solutions = _route_solutions(graph, [s[2] for s in stations], info["key"])
    assert len(solutions) == 1, "the key must identify exactly one distinct-station route"
    info["route"], info["graph"] = solutions[0], graph
    expected = grid.copy()
    painted = np.zeros(grid.shape, dtype=bool)
    directions = []
    for i, j in zip(info["route"], info["route"][1:]):
        a, b = stations[i], stations[j]
        band = _band(a, b, info["size"], info["patch"])
        assert not np.any(painted[band]), "connector bands overlap"
        painted[band] = True
        expected[band] = a[2]
        directions.append("D" if b[0] > a[0] else "U" if b[0] < a[0]
                          else "R" if b[1] > a[1] else "L")
    info["directions"] = directions
    return expected, info


def _crosses(a, b, c, d):
    if a[0] == b[0] and c[1] == d[1]:
        return min(a[1], b[1]) < c[1] < max(a[1], b[1]) and min(c[0], d[0]) < a[0] < max(c[0], d[0])
    if a[1] == b[1] and c[0] == d[0]:
        return _crosses(c, d, a, b)
    return False


def _sample_path(rng, cells, graph, length, kind, scope):
    budget = 2000
    def visit(path, axes):
        nonlocal budget
        budget -= 1
        if budget < 0:
            return None
        if len(path) == length:
            if kind == "both" and len(set(axes)) != 2:
                return None
            return path
        choices = list(rng.permutation(graph[path[-1]]))
        for j in choices:
            j = int(j)
            if j in path:
                continue
            a, b = cells[path[-1]], cells[j]
            axis = "h" if a[0] == b[0] else "v"
            if scope != "both" and axis != scope:
                continue
            if kind == "straight" and axes and axis != axes[0]:
                continue
            if kind == "elbow" and sum(x != y for x, y in zip(axes+[axis], (axes+[axis])[1:])) > 1:
                continue
            if any(_crosses(a, b, cells[u], cells[v]) for u, v in zip(path, path[1:])):
                continue
            found = visit(path+[j], axes+[axis])
            if found is not None:
                return found
        return None
    for start in rng.permutation(len(cells)):
        found = visit([int(start)], [])
        if found is not None:
            return found
        if budget < 0:
            break
    return None


def _sample_labels(rng, cells, graph, path, colors):
    length = len(path)
    mode = str(rng.choice(("unique", "mixed", "mixed", "start", "adjacent")))
    if mode == "unique" and length > 8:
        mode = "mixed"
    max_k = min(8, length)
    k = length if mode == "unique" else int(rng.integers(min(3, max_k), max_k+1))
    if mode in ("start", "adjacent"):
        k = min(k, length-1)
    pool = list(rng.permutation(colors)[:k])
    unused = [i for i in range(len(cells)) if i not in path]
    for attempt in range(100):
        # Broaden the pool on a hard graph, without privileging the start label.
        if attempt == 60 and mode == "mixed":
            k = max_k
            pool = list(rng.permutation(colors)[:k])
        key = pool + [int(rng.choice(pool)) for _ in range(length-k)]
        key = [int(x) for x in rng.permutation(key)]
        if mode == "start":
            key[int(rng.integers(1, length))] = key[0]
        elif mode == "adjacent":
            j = int(rng.integers(1, length))
            key[j] = key[j-1]
        if len(set(key)) < 2:
            continue
        labels = [0] * len(cells)
        for i, color in zip(path, key):
            labels[i] = color
        absent = [int(c) for c in colors if c not in key]
        for i in unused:
            labels[i] = int(rng.choice(absent if absent and rng.random() < .45 else key))
        solutions = _route_solutions(graph, labels, key)
        if len(solutions) == 1 and solutions[0] == tuple(path):
            return labels, key
    return None


def _axis_coordinates(rng, count, size, limit):
    gaps = np.ones(max(0, count-1), dtype=int)
    spare = limit - count*size - len(gaps)
    for i in rng.permutation(len(gaps)):
        extra = int(rng.integers(0, min(7, spare)+1))
        gaps[i] += extra
        spare -= extra
    return [0] + list(np.cumsum(size+gaps).astype(int))


def _draw_pair(rng, cells, path, labels, key, size, patch, roles, compact=False):
    cells = [(int(r), int(c)) for r, c in cells]
    row_ids = sorted({r for r, c in cells})
    col_ids = sorted({c for r, c in cells})
    compact = compact or rng.random() < .35
    nr, nc = max(row_ids)+1, max(col_ids)+1
    ys = _axis_coordinates(rng, nr, size, min(26, nr*size+nr-1) if compact else 26)
    xs = _axis_coordinates(rng, nc, size, min(28, nc*size+nc-1) if compact else 28)
    positions = [(ys[r], xs[c]) for r, c in cells]
    bh = max(y for y, x in positions)+size
    bw = max(x for y, x in positions)+size
    mh, mw = bh+4, max(bw+2, 2*len(key)+1)
    style = 0 if compact else int(rng.integers(0, 5))
    if style == 3:
        h = w = int(rng.integers(max(mh, mw), 31))
    elif style == 4:
        h = w = 30
    else:
        h = int(rng.integers(mh, min(30, mh+3)+1 if style == 0 else 31))
        w = int(rng.integers(mw, min(30, mw+3)+1 if style == 0 else 31))
    top = int(rng.integers(1, h-bh-2))
    left = int(rng.integers(1, w-bw))
    key_y = int(rng.integers(top+bh+1, h-1))
    key_x = int(rng.integers(1, w-(2*len(key)-1)))
    bg, carrier = roles
    grid = np.full((h, w), bg, dtype=np.int8)
    offset = (size-patch)//2
    positions = [(y+top, x+left) for y, x in positions]
    for (y, x), color in zip(positions, labels):
        grid[y:y+size, x:x+size] = carrier
        grid[y+offset:y+offset+patch, x+offset:x+offset+patch] = color
    grid[key_y, key_x:key_x+2*len(key):2] = key
    output = grid.copy()
    for a, b in zip(path, path[1:]):
        # Construction uses geometric endpoints; the validator decodes afresh.
        output[_band(positions[a], positions[b], size, patch)] = labels[a]
    return dict(input=grid, output=output)


def _sample_pair(rng, params, regime, index):
    assert regime in ("source_train", "target_test")
    train = regime == "source_train"
    anchor = train and index < 2
    if params["palette"] == "varied":
        roles = tuple(int(c) for c in rng.choice(10, 2, replace=False))
    elif params["palette"] == "swap" and not train:
        roles = params["roles"][::-1]
    else:
        roles = params["roles"]
    colors = [c for c in range(10) if c not in roles]
    scope = params["scope"]
    for attempt in range(48):
        size = params["demo_size"] if anchor else int(rng.choice((3, 3, 4, 4, 5, 5, 6, 7)))
        patches = list(range(1 if size % 2 else 2, size-1, 2))
        patch = params["demo_patches"][index] if anchor else int(rng.choice(patches))
        difficulty = params["difficulty"]
        if not train:
            difficulty = max(0, min(2, difficulty + params["shift"]))
        lo, hi = ((3, 5), (4, 10), (6, 14))[difficulty]
        if anchor:
            hi = min(hi, 9)
        length = int(rng.integers(lo, hi+1))
        unused = 0 if not params["unused"] or (anchor and index == 0) else int(rng.integers(0, 6))
        if anchor and index == 1 and params["unused"]:
            unused = int(rng.integers(1, 4))
        kind = str(rng.choice(("straight", "elbow", "free", "free", "free")))
        if anchor and index == 1 and scope == "both":
            kind = "both"
        if scope != "both":
            kind = "straight"
            length = min(length, (28+1)//(size+1) if scope == "h" else (26+1)//(size+1))
        n = length+unused
        options = [(r, c) for r in range(1, 28//(size+1)+1)
                   for c in range(1, 30//(size+1)+1)
                   if n <= r*c <= n+8 and r*size+r-1 <= 26 and c*size+c-1 <= 28
                   and (kind != "straight" or
                        (scope != "v" and c >= length) or (scope != "h" and r >= length))]
        if not options:
            continue
        weights = np.array([1/(1+r*c-n)**1.25 for r, c in options])
        rows, cols = options[int(rng.choice(len(options), p=weights/weights.sum()))]
        universe = [(r, c) for r in range(rows) for c in range(cols)]
        cells = sorted(universe[int(i)] for i in rng.choice(len(universe), n, replace=False))
        graph = _visible_graph(cells)
        path = _sample_path(rng, cells, graph, length, kind, scope)
        if path is None:
            continue
        assigned = _sample_labels(rng, cells, graph, path, colors)
        if assigned is None:
            continue
        labels, key = assigned
        return _draw_pair(rng, cells, path, labels, key, size, patch, roles)
    # A guaranteed simple construction after bounded search. It uses the same
    # rule and preserves required demonstrations; random search cannot fail a seed.
    size = params["demo_size"] if anchor else 3
    patch = params["demo_patches"][index] if anchor else 1
    cells = [(0, 0), (0, 1), (1, 1)] if scope == "both" else (
        [(0, i) for i in range(3)] if scope == "h" else [(i, 0) for i in range(3)])
    path = [0, 1, 2]
    if anchor and index == 1 and params["unused"]:
        cells.append((1, 0) if scope == "both" else (1, 1) if scope == "h" else (1, 1))
    labels = [int(c) for c in rng.permutation(colors)[:len(cells)]]
    return _draw_pair(rng, cells, path, labels, labels[:3], size, patch, roles)


def _validate_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert all(1 <= s <= 30 for s in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert 0 <= int(grid.min()) <= int(grid.max()) <= 9


def _validate_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    for grid in pair.values():
        _validate_grid(grid)
    assert pair["input"].shape == pair["output"].shape
    assert _pair_digest(pair) not in _SOURCE_PAIR_DIGESTS
    expected, info = _solve_input(pair["input"])
    assert np.array_equal(pair["output"], expected), "incorrect gap painting"
    assert not np.array_equal(pair["input"], expected)
    return info


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    infos, seen = {}, set()
    for split in ("train", "test"):
        infos[split] = []
        for pair in puzzle[split]:
            info = _validate_pair(pair)
            signature = _grid_signature(pair["input"])
            assert signature not in seen
            seen.add(signature)
            infos[split].append(info)
    train, test = infos["train"], infos["test"]
    # Different widths at a shared outer size rule out parity/outer-size lookup.
    assert any(a["size"] == b["size"] and a["patch"] != b["patch"] for a in train for b in train)
    train_axes = {"h" if d in "LR" else "v" for a in train for d in a["directions"]}
    test_axes = {"h" if d in "LR" else "v" for a in test for d in a["directions"]}
    assert test_axes <= train_axes
    # Literal colors need not carry across pairs: geometry defines every role.
    assert any(any(x != y for x, y in zip(a["key"], a["key"][1:])) for a in train)
    if any(len(a["stations"]) > len(a["key"]) for a in test):
        assert any(len(a["stations"]) > len(a["key"]) for a in train)
    assert any(len(a["stations"]) == len(a["key"]) for a in train)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    nt, nq = _resolve_counts(rng, num_train, num_test)
    demo_size = int(rng.choice((5, 6)))
    widths = (1, 3) if demo_size == 5 else (2, 4)
    params = dict(palette=str(rng.choice(("stable", "swap", "varied", "varied"))),
                  roles=tuple(int(x) for x in rng.choice(10, 2, replace=False)),
                  difficulty=int(rng.integers(0, 3)), shift=int(rng.choice((-1, 0, 0, 1))),
                  unused=bool(rng.random() < .85),
                  scope=str(rng.choice(("h", "v", "both"), p=(.08, .07, .85))),
                  demo_size=demo_size, demo_patches=tuple(int(x) for x in rng.permutation(widths)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, regime, count in (("train", "source_train", nt), ("test", "target_test", nq)):
        for index in range(count):
            pair = _sample_pair(rng, params, regime, index)
            # These 40 affine color permutations give distinct inputs because
            # every scene has at least four colors. At most 13 earlier inputs
            # and five source pairs are forbidden, so a variant always exists.
            if _grid_signature(pair["input"]) in seen or _pair_digest(pair) in _SOURCE_PAIR_DIGESTS:
                found = None
                for multiplier in (1, 3, 7, 9):
                    for shift in range(10):
                        remap = (multiplier*np.arange(10)+shift) % 10
                        candidate = {k: remap[v].astype(np.int8) for k, v in pair.items()}
                        if _grid_signature(candidate["input"]) not in seen and _pair_digest(candidate) not in _SOURCE_PAIR_DIGESTS:
                            found = candidate
                            break
                    if found is not None:
                        break
                assert found is not None
                pair = found
            seen.add(_grid_signature(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def measure_pair(pair):
    """Review features reconstructed from visible grids, not sampled parameters."""
    _, a = _solve_input(pair["input"])
    stations, route, key = a["stations"], a["route"], a["key"]
    ys = sorted({s[0] for s in stations})
    xs = sorted({s[1] for s in stations})
    gaps = {"h": [], "v": []}
    skipped = False
    for i, j in zip(route, route[1:]):
        u, v = stations[i], stations[j]
        horizontal = u[0] == v[0]
        axis, coord, levels = ("h", 1, xs) if horizontal else ("v", 0, ys)
        gaps[axis].append(abs(u[coord]-v[coord])-a["size"])
        skipped |= abs(levels.index(u[coord])-levels.index(v[coord])) > 1
    unused = [s[2] for i, s in enumerate(stations) if i not in route]
    labels = {s[2] for s in stations}
    branches, dead_depth = 0, 0
    # Count local candidates along the solution, and how long wrong choices last.
    def longest(prefix):
        if len(prefix) == len(key):
            return len(prefix)
        nxt = [j for j in a["graph"][prefix[-1]]
               if j not in prefix and stations[j][2] == key[len(prefix)]]
        return max([len(prefix)] + [longest(prefix+[j]) for j in nxt])
    for step, correct in enumerate(route):
        options = [i for i, s in enumerate(stations) if s[2] == key[step]
                   and i not in route[:step]
                   and (step == 0 or i in a["graph"][route[step-1]])]
        branches += len(options) > 1
        for wrong in options:
            if wrong != correct:
                dead_depth = max(dead_depth, longest(list(route[:step])+[wrong])-step-1)
    counts = np.bincount(pair["input"].ravel(), minlength=10)
    return dict(shape=pair["input"].shape, size=a["size"], patch=a["patch"],
                stations=len(stations), length=len(key), unused=len(unused),
                duplicate_unused=any(c in key for c in unused), absent_unused=any(c not in key for c in unused),
                rows=len(ys), cols=len(xs), sparse=len(stations) < len(ys)*len(xs),
                occupancy=len(stations)/(len(ys)*len(xs)), skipped=bool(skipped),
                horizontal_gaps=tuple(gaps["h"]), vertical_gaps=tuple(gaps["v"]),
                unequal_links=any(len(set(g)) > 1 for g in gaps.values()),
                row_gaps=tuple(y-x-a["size"] for x, y in zip(ys, ys[1:])),
                col_gaps=tuple(y-x-a["size"] for x, y in zip(xs, xs[1:])),
                key_clearance=a["key_y"]-max(s[0]+a["size"] for s in stations),
                margins=(min(ys), min(xs), pair["input"].shape[0]-a["key_y"]-1,
                         pair["input"].shape[1]-max(xs)-a["size"]),
                roles=(a["background"], a["carrier"]), label_count=len(labels),
                key_colors=len(set(key)), distinct=len(set(key)) == len(key),
                repeat_start=key[0] in key[1:], adjacent_equal=any(x == y for x, y in zip(key, key[1:])),
                branches=branches, dead_depth=dead_depth,
                turns=sum((x in "LR") != (y in "LR") for x, y in zip(a["directions"], a["directions"][1:])),
                directions=tuple(sorted(set(a["directions"]))),
                background_not_majority=int(counts[a["background"]]) < int(counts.max()),
                layout=tuple((y-min(ys), x-min(xs)) for y, x, color in stations),
                labels=tuple(sorted(labels)))


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    metrics, per_puzzle = [], []
    for p in puzzles:
        validate_generated_puzzle(p)
        m = [(s, measure_pair(pair)) for s in ("train", "test") for pair in p[s]]
        per_puzzle.append(m)
        metrics.extend(m)
    values = [m for s, m in metrics]
    assert len({m["shape"] for m in values}) >= 50
    assert len({m["layout"] for m in values}) >= 100
    assert len({m["roles"] for m in values}) >= 30
    assert {m["size"] for m in values} >= {3, 4, 5, 6, 7}
    assert {m["patch"] for m in values} >= {1, 2, 3, 4, 5}
    assert {m["label_count"] for m in values} >= {3, 4, 5, 6, 7, 8}
    assert min(m["length"] for m in values) == 3 and max(m["length"] for m in values) >= 12
    for split in ("train", "test"):
        group = [m for s, m in metrics if s == split]
        for flag in ("sparse", "skipped", "unequal_links", "distinct", "repeat_start", "adjacent_equal",
                     "duplicate_unused", "absent_unused", "background_not_majority"):
            assert any(m[flag] for m in group), (split, flag)
        assert any(m["unused"] == 0 for m in group)
        assert any(not m["sparse"] for m in group)
        assert any(m["sparse"] and m["unused"] == 0 for m in group)
        assert any(m["turns"] == 0 for m in group) and any(m["turns"] >= 2 for m in group)
    for axis in ("horizontal_gaps", "vertical_gaps"):
        assert max(g for m in values for g in m[axis]) >= 7
    assert max(m["dead_depth"] for m in values) >= 3
    assert any(m["shape"] == (30, 30) for m in values)
    assert len({m["key_clearance"] for m in values}) >= 4
    assert len({m["margins"] for m in values}) >= 30
    assert any(m["sparse"] and m["unequal_links"] and m["repeat_start"] and m["patch"] >= 3 for m in values)
    assert any(len({m["unused"] for s, m in metrics if m["length"] == n}) >= 4 for n in range(3, 15))
    assert sum(len({m["roles"] for s, m in group}) > 2 for group in per_puzzle) >= 10
    assert any(len({m["roles"] for s, m in group}) == 1 for group in per_puzzle)
    assert any(len({m["roles"] for s, m in group if s == "train"}) == 1
               and {m["roles"] for s, m in group if s == "test"}
               == {group[0][1]["roles"][::-1]} for group in per_puzzle)
    assert any(set(c for s, m in group if s == "train" for c in m["roles"])
               & set(c for s, m in group if s == "test" for c in m["labels"])
               for group in per_puzzle)
    assert any(set(c for s, m in group if s == "test" for c in m["labels"])
               - set(c for s, m in group if s == "train" for c in m["labels"])
               for group in per_puzzle)
    assert sum(len({m["stations"] for s, m in group}) > 1 for group in per_puzzle) >= 35
    assert sum(len({m["size"] for s, m in group}) > 1 for group in per_puzzle) >= 30
    assert sum(any(m["unequal_links"] for s, m in group) for group in per_puzzle) >= 25
    assert any(max(m["length"] for s, m in group if s == "test") > max(m["length"] for s, m in group if s == "train") for group in per_puzzle)
    assert any(max(m["length"] for s, m in group if s == "test") < max(m["length"] for s, m in group if s == "train") for group in per_puzzle)
    assert _puzzles_equal(puzzles[0], generate_puzzle(0))
    assert not _puzzles_equal(puzzles[0], puzzles[1])
    for i, (nt, nq) in enumerate(((2, 1), (10, 1), (10, 4), (7, 3),
                                   (None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(9000+i, nt, nq)
        assert nt is None or len(p["train"]) == nt
        assert nq is None or len(p["test"]) == nq
        assert _puzzles_equal(p, generate_puzzle(9000+i, nt, nq))
    for nt, nq in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4), (True, 1), (3.5, 1)):
        try:
            generate_puzzle(17, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal count request accepted")
    # Corruptions exercise output preservation and patch-width checking.
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        return
    raise AssertionError("a changed background frame was accepted")
