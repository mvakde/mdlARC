"""Color relays: a cupped source, a routed forest, and one lookup per contour.

Only standard library and NumPy are used. No files, parent code or generation
metadata are consulted by solve. Generation owns an explicit tree and labels;
validation reconstructs that tree from the returned input.
"""
from collections import deque
import hashlib
import numbers
import numpy as np

D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _components(cells, steps=D8):
    remaining = set(cells)
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        group = {p}
        todo = [p]
        for r, c in todo:
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    todo.append(q)
        result.append(group)
    return result


def _near(cells, shape, steps=D8):
    h, w = shape
    return {(r + dr, c + dc) for r, c in cells for dr, dc in steps
            if 0 <= r + dr < h and 0 <= c + dc < w} - set(cells)


def _move(cells, r, c):
    return {(y + r, x + c) for y, x in cells}


def _array(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a


def _key(a, bg):
    candidates = []
    h, w = a.shape
    for cells in _components(zip(*np.where(a != bg))):
        r0, r1 = min(r for r, c in cells), max(r for r, c in cells)
        c0, c1 = min(c for r, c in cells), max(c for r, c in cells)
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        if len(cells) != hh * ww:
            continue
        lanes = []
        if hh == 2 and 3 <= ww <= 6:
            if r0 == 0:
                lanes.append(("top", a[r0, c0:c1+1], a[r1, c0:c1+1]))
            if r1 == h - 1:
                lanes.append(("bottom", a[r1, c0:c1+1], a[r0, c0:c1+1]))
        if ww == 2 and 3 <= hh <= 6:
            if c0 == 0:
                lanes.append(("left", a[r0:r1+1, c0], a[r0:r1+1, c1]))
            if c1 == w - 1:
                lanes.append(("right", a[r0:r1+1, c1], a[r0:r1+1, c0]))
        for side, outer, inner in lanes:
            x, y = tuple(map(int, outer)), tuple(map(int, inner))
            if len(set(x)) == len(x) and set(x) == set(y) and all(p != q for p, q in zip(x, y)):
                mapping = dict(zip(x, y))
                # This family uses a single cycle, so every possible arriving
                # paint has a defined successor, without a terminal convention.
                walk, state = set(), x[0]
                while state not in walk:
                    walk.add(state)
                    state = mapping[state]
                if len(walk) == len(x):
                    candidates.append((cells, mapping, side))
    assert len(candidates) == 1, "expected one isolated directed edge key"
    return candidates[0]


def _analyze(input_grid):
    a = _array(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    assert np.count_nonzero(counts == counts.max()) == 1
    bg = int(counts.argmax())
    key_cells, mapping, side = _key(a, bg)
    paint = set(mapping)
    # Paint outside the key occurs only inside the visibly cupped sources.
    sources = [(p, int(a[p])) for p in zip(*np.where(np.isin(a, list(paint)))) if p not in key_cells]
    assert sources
    wire_colors = set()
    for p, color in sources:
        neighbors = {q for q in _near({p}, a.shape) if a[q] != bg}
        assert len(neighbors) == 5
        colors = {int(a[q]) for q in neighbors}
        assert len(colors) == 1 and not colors & (paint | {bg})
        wire_colors.update(colors)
    assert len(wire_colors) == 1
    wire = wire_colors.pop()
    other = set(map(int, np.unique(a))) - paint - {bg, wire}
    assert len(other) == 1, "one neutral contour color is required"
    rim = other.pop()
    rims = _components(zip(*np.where(a == rim)))
    wires = _components(zip(*np.where(a == wire)))
    loops = [{"rim": cells, "holes": []} for cells in rims]
    rim_id = {p: i for i, cells in enumerate(rims) for p in cells}
    # Background leaks orthogonally; diagonal ink is a valid wall.
    for region in _components(zip(*np.where(a == bg)), D4):
        if any(r in (0, a.shape[0]-1) or c in (0, a.shape[1]-1) for r, c in region):
            continue
        boundary = _near(region, a.shape, D4)
        owners = {rim_id.get(p, -1) for p in boundary}
        assert len(owners) == 1 and -1 not in owners, "non-contour enclosure"
        loops[owners.pop()]["holes"].append(region)
    assert loops and all(loop["holes"] for loop in loops)
    n = len(loops)
    graph = [set() for _ in range(n + len(wires))]
    wire_id = {p: i + n for i, cells in enumerate(wires) for p in cells}
    for i, loop in enumerate(loops):
        for j in {wire_id[p] for p in _near(loop["rim"], a.shape) if p in wire_id}:
            graph[i].add(j)
            graph[j].add(i)
    roots = {}
    for p, color in sources:
        ids = {wire_id[q] for q in _near({p}, a.shape) if q in wire_id}
        assert len(ids) == 1
        root = ids.pop()
        assert root not in roots, "two paint cups on one wire component"
        roots[root] = color
    out = a.copy()
    depths, fills, root_of, parents = {}, {}, {}, {}
    seen = set()
    for start in range(len(graph)):
        if start in seen:
            continue
        group = {start}
        todo = [start]
        for i in todo:
            for j in graph[i] - group:
                group.add(j)
                todo.append(j)
        seen.update(group)
        assert sum(len(graph[i]) for i in group) == 2 * (len(group) - 1), "route graph must be a tree"
        supplies = group & roots.keys()
        assert len(supplies) <= 1, "a route has competing sources"
        if not supplies:
            continue
        root = next(iter(supplies))
        queue = [(root, -1, roots[root], 0)]
        for i, parent, state, depth in queue:
            parents[i] = parent
            if i < n:
                state = mapping[state]
                depth += 1
                depths[i], fills[i], root_of[i] = depth, state, root
                for hole in loops[i]["holes"]:
                    for q in hole:
                        out[q] = state
            for j in sorted(graph[i] - {parent}):
                queue.append((j, i, state, depth))
    return out, {"background": bg, "wire": wire, "rim": rim, "mapping": mapping,
                 "side": side, "key_cells": key_cells, "loops": loops, "wires": wires,
                 "graph": graph, "roots": roots, "depths": depths, "fills": fills,
                 "root_of": root_of, "parents": parents}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer all roles and relay states solely from the input; do not mutate it."""
    return _analyze(input_grid)[0]


def _cavity_shape(rng, kind, small):
    h = int(rng.integers(1, 3 if small else 4))
    w = int(rng.integers(1, 3 if small else 5))
    if kind == "diamond":
        radius = 1 if small else int(rng.integers(1, 3))
        interior = {(r, c) for r in range(-radius+1, radius) for c in range(-radius+1, radius)
                    if abs(r) + abs(c) < radius}
    elif kind == "profile":
        h, w = (2, 2) if small else (3, 3)
        interior = {(r, c) for r in range(h) for c in range(w) if not (r == 0 and c == w - 1)}
    else:
        interior = {(r, c) for r in range(h) for c in range(w)}
    shell = {(r+dr, c+dc) for r, c in interior for dr, dc in D4} - interior
    if kind in ("rectangle", "multi"):
        shell |= {(r+dr, c+dc) for r, c in interior for dr, dc in D8} - interior
    if kind == "multi":
        # One physical contour with two cavities is still one relay event.
        interior = {(r, c) for r in range(1, 4) for c in (1, 3)}
        shell = {(r, c) for r in range(5) for c in range(5)} - interior
    for _ in range(int(rng.integers(4))):
        interior = {(-c, r) for r, c in interior}
        shell = {(-c, r) for r, c in shell}
    r0, c0 = min(r for r, c in shell), min(c for r, c in shell)
    return _move(interior, -r0, -c0), _move(shell, -r0, -c0)


def _mst(points):
    edges, used = [], {0}
    while len(used) < len(points):
        _, i, j = min((sum((points[i][d]-points[j][d])**2 for d in (0, 1)), i, j)
                      for i in sorted(used) for j in range(len(points)) if j not in used)
        edges.append((i, j))
        used.add(j)
    return edges


def _tree_depths(n, edges, root):
    graph = [set() for _ in range(n)]
    for i, j in edges:
        graph[i].add(j)
        graph[j].add(i)
    depths, parents, todo = {root: 1}, {root: -1}, [root]
    for i in todo:
        for j in sorted(graph[i]):
            if j not in depths:
                depths[j], parents[j] = depths[i] + 1, i
                todo.append(j)
    return depths, parents, graph


def _path(rng, starts, goals, allowed, shape):
    initial = sorted(starts & allowed)
    rng.shuffle(initial)
    queue = deque(initial)
    previous = {p: None for p in initial}
    directions = [D8[int(i)] for i in rng.permutation(8)]
    end = None
    while queue:
        p = queue.popleft()
        if p in goals:
            end = p
            break
        for dr, dc in directions:
            q = (p[0] + dr, p[1] + dc)
            if q in allowed and q not in previous:
                previous[q] = p
                queue.append(q)
    if end is None:
        return None
    path = set()
    while end is not None:
        path.add(end)
        end = previous[end]
    return path


def _construct(rng, lesson, test, turn, attempt):
    small = attempt >= 24
    h, w = (30, 30) if small else (int(rng.integers(22, 31)), int(rng.integers(22, 31)))
    if lesson == 0:
        main, second, inert = 3, 0, 1
    elif lesson == 1:
        main, second, inert = 4, 1, 0
    elif test:
        main, second, inert = int(rng.integers(4, 7)), int(rng.random() < .35), int(rng.random() < .5)
    else:
        main, second, inert = int(rng.integers(1, 5)), int(rng.random() < .35), int(rng.integers(0, 3))
    total = main + second + inert
    if not small and lesson < 0 and not test and total <= 2:
        h, w = int(rng.integers(14, 26)), int(rng.integers(12, 31))
    elif not small and lesson < 0 and not test and total == 3:
        h, w = int(rng.integers(18, 31)), int(rng.integers(18, 31))
    # Internal role identities are only construction labels; all are permuted.
    k = int(rng.integers(3, 7))
    states = list(range(3, 3+k))
    cycle = [states[int(i)] for i in rng.permutation(k)]
    mapping = dict(zip(cycle, cycle[1:] + cycle[:1]))
    grid = np.zeros((h, w), dtype=np.int8)
    key_start = int(rng.integers(w - k + 1))
    order = [states[int(i)] for i in rng.permutation(k)]
    grid[0, key_start:key_start+k] = order
    grid[1, key_start:key_start+k] = [mapping[c] for c in order]
    keycells = set(zip(*np.where(grid != 0)))
    occupied = set(keycells)
    regions = []
    kinds = ("rectangle", "diamond", "profile", "rectangle", "multi")
    for i in range(total):
        kind = "diamond" if lesson == 0 and i == 0 else "multi" if lesson == 1 and i == 0 else "profile" if lesson == 1 and i == 1 else str(rng.choice(kinds))
        if lesson == 0 and kind == "multi":
            kind = "rectangle"
        inner, shell = _cavity_shape(rng, kind, small)
        hh, ww = max(r for r, c in shell)+1, max(c for r, c in shell)+1
        placed = False
        for _ in range(70):
            rr, cc = int(rng.integers(4, h-hh)), int(rng.integers(1, w-ww))
            inside, edge = _move(inner, rr, cc), _move(shell, rr, cc)
            full = inside | edge
            if full & (occupied | _near(occupied, grid.shape)):
                continue
            regions.append({"inner": inside, "shell": edge, "center": (rr+hh/2, cc+ww/2)})
            occupied |= full
            placed = True
            break
        if not placed:
            return None
    edges = _mst([q["center"] for q in regions[:main]])
    candidates = []
    for root in range(main):
        depth, parents, graph = _tree_depths(main, edges, root)
        maximum = max(depth.values())
        if lesson == 0 and maximum != 3:
            continue
        if lesson == 1 and (root != 0 or maximum > 3 or len(graph[root]) < 2):
            continue
        if test and maximum < 4:
            continue
        if not test and lesson < 0 and maximum > 3:
            continue
        candidates.append((root, depth, parents))
    if not candidates:
        return None
    root, depths, parents = candidates[int(rng.integers(len(candidates)))]
    for q in regions:
        for p in q["shell"]:
            grid[p] = 2
    canvas = {(r, c) for r in range(3, h) for c in range(w)}
    all_contours = set().union(*(q["inner"] | q["shell"] for q in regions))
    wirecells = set()
    # Euclidean tree edges guide spatial embedding; visible ports alone define
    # connectivity to the solver. Edge order is randomized independently.
    for e in rng.permutation(len(edges)):
        i, j = edges[int(e)]
        foreign = set().union(*(q["inner"] | q["shell"] for z, q in enumerate(regions) if z not in (i, j)))
        blocked = all_contours | foreign | _near(foreign, grid.shape)
        blocked |= keycells | _near(keycells, grid.shape)
        blocked |= wirecells | _near(wirecells, grid.shape)
        allowed = canvas - blocked
        path = _path(rng, _near(regions[i]["shell"], grid.shape),
                     _near(regions[j]["shell"], grid.shape), allowed, grid.shape)
        if path is None:
            return None
        wirecells |= path
    root_specs = [(root, states[int(rng.integers(k))])]
    if second:
        alternatives = [s for s in states if s != root_specs[0][1]]
        root_specs.append((main, alternatives[int(rng.integers(len(alternatives)))]))
    source_cells = set()
    for target, state in root_specs:
        found = False
        # Each source is a 5-neighbor U cup, in an arbitrary orientation.
        for _ in range(100):
            cap = {(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)}
            src = {(0, 1)}
            for _ in range(int(rng.integers(4))):
                cap, src = {(-c, r) for r, c in cap}, {(-c, r) for r, c in src}
            r0, c0 = min(r for r, c in cap | src), min(c for r, c in cap | src)
            cap, src = _move(cap, -r0, -c0), _move(src, -r0, -c0)
            hh, ww = max(r for r, c in cap | src)+1, max(c for r, c in cap | src)+1
            rr, cc = int(rng.integers(3, h-hh+1)), int(rng.integers(w-ww+1))
            cap, src = _move(cap, rr, cc), _move(src, rr, cc)
            full = cap | src
            busy = all_contours | wirecells | keycells | source_cells
            if full & (busy | _near(busy, grid.shape)):
                continue
            foreign = set().union(*(q["inner"] | q["shell"] for z, q in enumerate(regions) if z != target))
            blocked = all_contours | _near(foreign, grid.shape) | keycells | _near(keycells, grid.shape)
            blocked |= wirecells | _near(wirecells, grid.shape) | source_cells | _near(source_cells, grid.shape)
            blocked |= src | (_near(src, grid.shape) - cap)
            allowed = (canvas - blocked) | cap
            path = _path(rng, cap, _near(regions[target]["shell"], grid.shape), allowed, grid.shape)
            if path is None:
                continue
            wirecells |= path | cap
            source_cells |= src
            for p in src:
                grid[p] = state
            found = True
            break
        if not found:
            return None
    for p in wirecells:
        grid[p] = 1
    output = grid.copy()
    expected_depths = dict(depths)
    expected_states = {}
    for i in sorted(range(main), key=lambda j: depths[j]):
        arriving = root_specs[0][1] if parents[i] == -1 else expected_states[parents[i]]
        expected_states[i] = mapping[arriving]
    if second:
        expected_states[main] = mapping[root_specs[1][1]]
        expected_depths[main] = 1
    for i, color in expected_states.items():
        for p in regions[i]["inner"]:
            output[p] = color
    palette = np.asarray(rng.permutation(10), dtype=np.int8)
    grid, output = palette[grid], palette[output]
    if rng.random() < .5:
        grid, output = np.fliplr(grid).copy(), np.fliplr(output).copy()
    grid, output = np.rot90(grid, turn).copy(), np.rot90(output, turn).copy()
    try:
        predicted, info = _analyze(grid)
    except AssertionError:
        return None
    if not np.array_equal(predicted, output) or len(info["loops"]) != total:
        return None
    if sorted(info["depths"].values()) != sorted(expected_depths.values()):
        return None
    return {"input": grid, "output": output}


def _valid_counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _fingerprint(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _valid_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    orientations = [int(i) for i in rng.permutation(4)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            lesson = index if split == "train" and index < 2 else -1
            turn = orientations[index % 2] if split == "train" else orientations[2 + index % 2]
            for attempt in range(96):
                pair = _construct(rng, lesson, split == "test", turn, attempt)
                if pair is not None and _fingerprint(pair["input"]) not in seen:
                    seen.add(_fingerprint(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded geometric construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    loops, graph, depths = info["loops"], info["graph"], info["depths"]
    branch = any(sum(info["parents"].get(j) == i for j in graph[i]) >= 2 for i in info["parents"])
    nonrect = any(len(hole) != (max(r for r, c in hole)-min(r for r, c in hole)+1) *
                  (max(c for r, c in hole)-min(c for r, c in hole)+1)
                  for loop in loops for hole in loop["holes"])
    # A diagonal wall or attachment must have a real effect on grouping.
    diagonal = any(len(_components(loop["rim"], D4)) > 1 for loop in loops)
    return {"loops": len(loops), "sources": len(info["roots"]), "max_depth": max(depths.values()),
            "branch": branch, "inactive": len(loops)-len(depths), "key_length": len(info["mapping"]),
            "multi": any(len(loop["holes"]) > 1 for loop in loops), "nonrect": nonrect,
            "internal_multi": any(len(loops[i]["holes"]) > 1 and
                                  any(info["parents"].get(j) == i for j in graph[i]) for i in depths),
            "diagonal": diagonal, "background": info["background"], "side": info["side"]}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _valid_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _array(pair["input"]), _array(pair["output"])
            assert a.shape == b.shape
            fingerprint = _fingerprint(a)
            assert fingerprint not in seen
            seen.add(fingerprint)
            predicted, info = _analyze(a)
            assert np.array_equal(predicted, b), "incorrect relay fill"
            assert np.any(a != b) and np.all(a[a != b] == info["background"])
            assert np.array_equal(a[a != info["background"]], b[a != info["background"]])
            for loop in info["loops"]:
                for hole in loop["holes"]:
                    assert len({int(b[p]) for p in hole}) == 1
            # Counterfactual dropping B: all supplied holes take source paint.
            assert any(info["fills"][i] != info["roots"][info["root_of"][i]] for i in info["fills"])
            # Counterfactual dropping route order: every contour receives just
            # the first successor of its root, irrespective of predecessor.
            ordered = any(info["fills"][i] != info["mapping"][info["roots"][info["root_of"][i]]]
                          for i in info["fills"])
            f = _features(info)
            f["order_witness"] = ordered
            features[split].append(f)
    train, test = features["train"], features["test"]
    assert any(f["inactive"] for f in train)
    assert any(f["order_witness"] and f["max_depth"] == 3 for f in train)
    assert any(f["branch"] and f["sources"] == 2 and f["internal_multi"] for f in train)
    assert any(f["diagonal"] for f in train)
    assert any(f["nonrect"] for f in train)
    assert all(f["max_depth"] <= 3 for f in train)
    assert all(f["max_depth"] >= 4 and f["order_witness"] for f in test)
    assert len({f["side"] for f in train}) >= 2
    assert {f["side"] for f in train}.isdisjoint({f["side"] for f in test})


def self_test():
    """Bounded 42-seed sweep, metamorphisms, count checks and error detection.

    No files are written, and only one puzzle at a time is retained.
    """
    # A hand-specified circuit: 3 -> 4 -> 5 -> 3. The second relay
    # branches, so both children receive 3, regardless of edge length.
    fixture = np.zeros((19, 27), dtype=np.int8)
    fixture[0, 8:11], fixture[1, 8:11] = [3, 5, 4], [4, 3, 5]
    centers = [(6, 6), (6, 14), (14, 14), (6, 22), (14, 23)]
    for r, c in centers:
        fixture[r-1:r+2, c-1:c+2] = 2
        fixture[r, c] = 0
    fixture[5, 0:3], fixture[6, 0:5] = [1, 3, 1], 1
    fixture[6, 8:13] = fixture[6, 16:21] = 1
    fixture[8:13, 14] = 1
    expected = fixture.copy()
    for (r, c), color in zip(centers, (4, 5, 3, 3, 0)):
        expected[r, c] = color
    assert np.array_equal(solve(fixture), expected)
    # Changing only the cup propagates a different sequence over the same
    # geometry; reversing the table also changes actual downstream answers.
    changed = fixture.copy()
    changed[5, 1] = 4
    predicted = solve(changed)
    assert [int(predicted[p]) for p in centers] == [5, 3, 4, 4, 0]
    changed = fixture.copy()
    changed[1, 8:11] = [5, 4, 3]
    predicted = solve(changed)
    assert [int(predicted[p]) for p in centers] == [5, 4, 3, 3, 0]
    hashes, shapes, structures, records = set(), set(), set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                digest.update(bytes(a.shape) + a.tobytes() + b.tobytes())
                shapes.add(a.shape)
                normalization, normalized = {}, []
                for c in a.flat:
                    normalized.append(normalization.setdefault(int(c), len(normalization)))
                structures.add(hashlib.sha256(bytes(a.shape) + bytes(normalized)).digest())
                records.append(_features(_analyze(a)[1]))
        hashes.add(digest.digest())
    assert len(hashes) == 42 and len(structures) >= 190 and len(shapes) >= 30
    assert {r["key_length"] for r in records} == {3, 4, 5, 6}
    assert {r["side"] for r in records} == {"top", "bottom", "left", "right"}
    assert {r["background"] for r in records} == set(range(10))
    assert max(r["max_depth"] for r in records) >= 5
    assert all(any(r[f] for r in records) for f in ("branch", "multi", "nonrect", "diagonal", "inactive"))
    a = generate_puzzle(0)
    b = generate_puzzle(0)
    for split in a:
        for p, q in zip(a[split], b[split]):
            assert all(np.array_equal(p[k], q[k]) for k in p)
    # Input-only geometric and palette invariance, with hand-derived relay
    # consequences checked by mutation below, not a hidden state oracle.
    pair = a["train"][0]
    for turn in range(4):
        for shift in (0, 3):
            inp = (np.rot90(pair["input"], turn) + shift) % 10
            expected = (np.rot90(pair["output"], turn) + shift) % 10
            before = inp.copy()
            assert np.array_equal(solve(inp), expected) and np.array_equal(inp, before)
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    pair = broken["test"][0]
    p = tuple(np.argwhere(pair["input"] != pair["output"])[0])
    pair["output"][p] = pair["input"][p]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("partial fill accepted")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(100 + nt*4 + ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
