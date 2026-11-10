"""Wired sheets: routed corner pairing and bridge-defined rectangle depth.

Only NumPy and the standard library are used. The solver parses the visible
glyphs and directional graph; generation uses independent explicit paths and
an independently sampled depth order. No parent generator is imported.
"""
from itertools import permutations
import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape):
        raise ValueError("grid dimensions must be 1..30")
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9)):
        raise ValueError("grid cells must be integer colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(cells):
    unseen, result = set(cells), []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        stack, found = [p], {p}
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    stack.append(q)
                    found.add(q)
        result.append(found)
    return result


def _overlap(a, b):
    return max(a[0], b[0]) <= min(a[2], b[2]) and max(a[1], b[1]) <= min(a[3], b[3])


def _compatible(a, b):
    r, c = a["elbow"]
    rr, cc = b["elbow"]
    dr, dc = a["inward"]
    return (b["inward"] == (-dr, -dc) and (rr-r)*dr >= 2 and (cc-c)*dc >= 2)


def _sheet(a, b, marker):
    assert _compatible(a, b), "cable endpoints are not opposite inward-facing corners"
    colors = {t["label"] for t in (a, b) if t["label"] is not None}
    assert len(colors) == 1, "a cable needs one consistent paint clue"
    r, c = a["elbow"]; rr, cc = b["elbow"]
    return dict(bounds=(min(r, rr), min(c, cc), max(r, rr), max(c, cc)),
                markers=a["cells"] | b["cells"], paint=colors.pop(), marker=marker,
                ends=(a["elbow"], b["elbow"]))


def _closure_and_order(n, edges, sheets):
    above = np.zeros((n, n), dtype=bool)
    for a, b in edges:
        above[a, b] = True
    for k in range(n):
        above |= above[:, k, None] & above[None, k, :]
    assert not np.any(np.diag(above)), "contradictory bridge depths"
    for i, a in enumerate(sheets):
        for j, b in enumerate(sheets[:i]):
            assert not _overlap(a["bounds"], b["bounds"]) or above[i, j] or above[j, i], \
                "overlapping sheets lack a visible depth comparison"
    # Incomparable sheets are disjoint, so this tie choice cannot affect pixels.
    remaining, order = set(range(n)), []
    while remaining:
        first = min(i for i in remaining if not any(above[j, i] for j in remaining))
        order.append(first)
        remaining.remove(first)
    return above, order


def _parse_roles(g, bg, marker, terminals, carrier, bridge):
    elbows = {t["elbow"]: t for t in terminals}
    dots = {t["dot"] for t in terminals}
    structural = {bg, marker, carrier, bridge}
    assert all(t["label"] not in structural for t in terminals if t["label"] is not None)
    visible = {tuple(map(int, x)) for x in np.argwhere(g == carrier)} | set(elbows)
    if bridge is not None:
        visible |= {tuple(map(int, x)) for x in np.argwhere(g == bridge)}
    assert all(int(g[p]) in structural or p in dots for p in np.ndindex(g.shape))
    graph = {}
    crossings, bridge_cells = {}, set()
    for p in visible:
        r, c = p
        ds = {d: (r+dr, c+dc) for d, (dr, dc) in enumerate(_DIRS)
              if (r+dr, c+dc) in visible}
        graph[p] = ds
        if p in elbows:
            assert len(ds) == 1, "terminal must have exactly one cable"
            dr, dc = _DIRS[next(iter(ds))]
            ir, ic = elbows[p]["inward"]
            assert dr*ir + dc*ic == -1, "cable must leave a corner outward"
        elif len(ds) == 4:
            assert bridge is not None and g[p] == carrier
            upper = [axis for axis, directions in enumerate(((0, 2), (1, 3)))
                     if all(g[ds[d]] == bridge for d in directions)]
            assert len(upper) == 1, "crossing needs exactly one marked upper strand"
            axis = upper[0]
            assert all(g[ds[d]] == carrier for d in ((1, 3) if axis == 0 else (0, 2)))
            crossings[p] = axis
            bridge_cells.update(ds[d] for d in ((0, 2) if axis == 0 else (1, 3)))
        else:
            assert len(ds) == 2, "cable is broken or branching"
    if bridge is not None:
        assert bridge_cells == {tuple(map(int, x)) for x in np.argwhere(g == bridge)}, \
            "stray bridge marks"
    assigned, routes, sheets, owners, used = set(), [], [], {}, set()
    for start in sorted(elbows):
        if start in assigned:
            continue
        p, previous, path, seen = start, None, [], set()
        while True:
            assert p not in seen, "self-intersecting or cyclic cable"
            seen.add(p); used.add(p); path.append(p)
            if p != start and p in elbows:
                assert p not in assigned
                assigned.update((start, p))
                break
            ds = graph[p]
            if previous is None:
                direction = next(iter(ds))
            else:
                incoming = next(d for d, q in ds.items() if q == previous)
                direction = (incoming+2) % 4 if len(ds) == 4 else next(d for d in ds if d != incoming)
                if len(ds) == 4:
                    owners[p, incoming % 2] = len(routes)
            previous, p = p, ds[direction]
        sheets.append(_sheet(elbows[start], elbows[p], marker))
        routes.append(path)
    assert used == visible and assigned == set(elbows), "unattached cable fragments"
    edges = []
    for p, upper_axis in crossings.items():
        assert (p, 0) in owners and (p, 1) in owners
        a, b = owners[p, upper_axis], owners[p, 1-upper_axis]
        assert a != b
        edges.append((a, b))
    above, order = _closure_and_order(len(sheets), edges, sheets)
    return dict(grid=g, background=bg, marker=marker, carrier=carrier, bridge=bridge,
                terminals=terminals, routes=routes, sheets=sheets, crossings=crossings,
                edges=edges, above=above, order=order)


def _parse(input_grid):
    g = _grid(input_grid)
    bg = int(g[0, 0])
    assert np.all(g[0] == bg) and np.all(g[-1] == bg) and np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg), \
        "input needs a blank outer border"
    candidates = []
    for marker in sorted(set(map(int, np.unique(g))) - {bg}):
        components = _components(map(tuple, np.argwhere(g == marker)))
        if not 4 <= len(components) <= 8 or len(components) % 2 or any(len(c) != 3 for c in components):
            continue
        terminals = []
        try:
            for cells in components:
                elbows = [p for p in cells if sum((p[0]+dr, p[1]+dc) in cells for dr, dc in _DIRS) == 2]
                assert len(elbows) == 1
                r, c = elbows[0]
                neighbors = [(rr-r, cc-c) for rr, cc in cells if (rr, cc) != (r, c)]
                assert len([x for x in neighbors if x[0]]) == len([x for x in neighbors if x[1]]) == 1
                dr, dc = sum(x[0] for x in neighbors), sum(x[1] for x in neighbors)
                dot = (r+dr, c+dc)
                label = int(g[dot])
                assert label != marker
                terminals.append(dict(elbow=(int(r), int(c)), inward=(int(dr), int(dc)),
                                      cells=set(cells), dot=dot, label=None if label == bg else label))
            terminal_mask = np.zeros(g.shape, bool)
            for t in terminals:
                for p in t["cells"] | {t["dot"]}:
                    terminal_mask[p] = True
            other = sorted(set(map(int, np.unique(g[~terminal_mask]))) - {bg})
            assert 1 <= len(other) <= 2
            possibilities = [(other[0], None)] if len(other) == 1 else [(other[0], other[1]), (other[1], other[0])]
            for carrier, bridge in possibilities:
                try:
                    candidates.append(_parse_roles(g, bg, marker, terminals, carrier, bridge))
                except (AssertionError, StopIteration, KeyError):
                    continue
        except (AssertionError, IndexError):
            continue
    assert len(candidates) == 1, f"input must have one glyph interpretation, found {len(candidates)}"
    return candidates[0]


def _predict(info, order=None):
    """Select the uppermost covering sheet per pixel; input-derived only."""
    g = info["grid"]
    out = np.full(g.shape, info["background"], dtype=np.int8)
    order = info["order"] if order is None else order
    for r, c in np.ndindex(g.shape):
        for i in order:
            s = info["sheets"][i]
            r0, c0, r1, c1 = s["bounds"]
            if r0 <= r <= r1 and c0 <= c <= c1:
                out[r, c] = info["marker"] if (r, c) in s["markers"] else s["paint"]
                break
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Consume the wiring blueprint and return the completed opaque sheets."""
    return _predict(_parse(input_grid))


def _compose(shape, bg, marker, sheets, order):
    """Construction oracle: stamp whole rectangles from back to front."""
    out = np.full(shape, bg, dtype=np.int8)
    for i in reversed(order):
        s = sheets[i]
        r0, c0, r1, c1 = s["bounds"]
        out[r0:r1+1, c0:c1+1] = s["paint"]
        for p in s["markers"]:
            out[p] = marker
    return out


def _hook(rng, rows, cols, common_diagonal=False):
    r0, r1 = sorted(map(int, rng.choice(range(1, rows-1), 2, replace=False)))
    c0, c1 = sorted(map(int, rng.choice(range(1, cols-1), 2, replace=False)))
    a, b = int(rng.integers(r0)), int(rng.integers(c1+1, cols))
    path = [(r, c0) for r in range(r0, a-1, -1)]
    path += [(a, c) for c in range(c0+1, b+1)]
    path += [(r, b) for r in range(a+1, r1+1)]
    path += [(r1, c) for c in range(b-1, c1-1, -1)]
    if not common_diagonal:
        if rng.integers(2): path = [(rows-1-r, c) for r, c in path]
        if rng.integers(2): path = [(r, cols-1-c) for r, c in path]
    return path


def _network(paths):
    visits, crossings, endpoints, bounds = {}, [], [], []
    for i, path in enumerate(paths):
        endpoints += [path[0], path[-1]]
        x, y = path[0], path[-1]
        bounds.append((min(x[0], y[0]), min(x[1], y[1]), max(x[0], y[0]), max(x[1], y[1])))
        for k, p in enumerate(path):
            axis = None
            if 0 < k < len(path)-1:
                a, b = path[k-1], path[k+1]
                if a[1] == b[1] and a[0] != b[0]: axis = 0
                if a[0] == b[0] and a[1] != b[1]: axis = 1
            if p in visits:
                j, old = visits[p]
                if axis is None or old is None or axis == old:
                    return None
                crossings.append((j, i, p, old))
                visits[p] = (i, None)
            else:
                visits[p] = (i, axis)
    if len(set(endpoints)) != len(endpoints):
        return None
    if any(abs(r-rr)+abs(c-cc) == 1 for k, (r, c) in enumerate(endpoints) for rr, cc in endpoints[:k]):
        return None
    return crossings, bounds


def _coordinates(rng, count, extra_cap=7):
    # Irregular segment lengths, with enough gap for the bridge glyphs.
    gaps = [3] * (count-1)
    before = int(rng.integers(1, 3))
    budget = 28 - before - sum(gaps)
    for _ in range(int(rng.integers(min(extra_cap, budget)+1))):
        gaps[int(rng.integers(len(gaps)))] += 1
    coords = [before]
    for gap in gaps:
        coords.append(coords[-1]+gap)
    after = int(rng.integers(1, min(3, 29-coords[-1])+1))
    return coords, coords[-1]+after+1


def _expand(path, rs, cs):
    result = [(rs[path[0][0]], cs[path[0][1]])]
    for a, b in zip(path, path[1:]):
        x, y = (rs[a[0]], cs[a[1]]), (rs[b[0]], cs[b[1]])
        dr, dc = int(np.sign(y[0]-x[0])), int(np.sign(y[1]-x[1]))
        p = x
        while p != y:
            p = (p[0]+dr, p[1]+dc)
            result.append(p)
    return result


def _geometric_alternative(info):
    """Nearest compatible corner matching, deliberately ignoring cables.

    Clues must agree, and every proposed sheet must have a clue. Return a
    counterfactual with the clue's original depth, to isolate corner pairing.
    """
    ts = info["terminals"]
    def match(remaining):
        if not remaining:
            yield []
            return
        a = remaining[0]
        for b in remaining[1:]:
            if not _compatible(ts[a], ts[b]): continue
            labels = {ts[i]["label"] for i in (a, b) if ts[i]["label"] is not None}
            if len(labels) != 1: continue
            for rest in match([i for i in remaining[1:] if i != b]):
                yield [(a, b)] + rest
    matches = list(match(list(range(len(ts)))))
    if len(matches) < 2:
        return None
    def cost(m):
        return (sum(sum((x-y)**2 for x, y in zip(ts[a]["elbow"], ts[b]["elbow"])) for a, b in m), m)
    chosen = min(matches, key=cost)
    actual = {frozenset(s["ends"]) for s in info["sheets"]}
    if {frozenset((ts[a]["elbow"], ts[b]["elbow"])) for a, b in chosen} == actual:
        return None
    alternative, ranks = [], []
    for a, b in chosen:
        s = _sheet(ts[a], ts[b], info["marker"])
        clue = ts[a] if ts[a]["label"] is not None else ts[b]
        original = next(i for i, old in enumerate(info["sheets"]) if clue["elbow"] in old["ends"])
        alternative.append(s)
        ranks.append(info["order"].index(original))
    order = sorted(range(len(alternative)), key=lambda i: (ranks[i], i))
    return _compose(info["grid"].shape, info["background"], info["marker"], alternative, order)


def _features(info, output):
    sheets, g = info["sheets"], info["grid"]
    reverse = _predict(info, list(reversed(info["order"])))
    alt = _geometric_alternative(info)
    hidden = sum(int(output[p]) != info["marker"] for s in sheets for p in s["markers"])
    top_markers = sum(int(output[p]) == info["marker"] for s in sheets for p in s["markers"])
    clue_modes = []
    ts = {t["elbow"]: t for t in info["terminals"]}
    for s in sheets:
        a, b = sorted(s["ends"])
        clue_modes.append((ts[a]["label"] is not None, ts[b]["label"] is not None))
    direct = set(info["edges"])
    transitive = sum(_overlap(a["bounds"], b["bounds"]) and
                     (i, j) not in direct and (j, i) not in direct
                     for i, a in enumerate(sheets) for j, b in enumerate(sheets[:i]))
    return dict(sheets=len(sheets), crossings=len(info["crossings"]), shape=g.shape,
                bounds=tuple(s["bounds"] for s in sheets), paints=tuple(s["paint"] for s in sheets),
                roles=(info["background"], info["marker"], info["carrier"], info["bridge"]),
                hidden_markers=hidden, retained_markers=top_markers,
                depth_essential=not np.array_equal(reverse, output),
                pairing_essential=alt is not None and not np.array_equal(alt, output),
                forward=any(a and not b for a, b in clue_modes),
                reverse=any(b and not a for a, b in clue_modes),
                both=any(a and b for a, b in clue_modes), transitive=transitive,
                diagonals=tuple(sorted({int((s["ends"][1][0]-s["ends"][0][0]) *
                    (s["ends"][1][1]-s["ends"][0][1]) > 0) for s in sheets})))


def _make_pair(rng, n, extent, lesson, require_transitive=False):
    rows = int(rng.integers(extent[0], extent[1]+1))
    cols = int(rng.integers(extent[0], extent[1]+1))
    if n == 4: rows = cols = 9
    paths = [_hook(rng, rows, cols, lesson == "pairing")]
    for i in range(1, n):
        for _ in range(180):
            path = _hook(rng, rows, cols, lesson == "pairing")
            network = _network(paths+[path])
            if network is not None and any(i in x[:2] for x in network[0]):
                paths.append(path)
                break
        else:
            return None
    crossings, bounds = _network(paths)
    # Try all small depth permutations: choose only visibly determined stacks.
    possible = []
    for order in permutations(range(n)):
        rank = {i: order.index(i) for i in range(n)}
        edges = [(i, j) if rank[i] < rank[j] else (j, i) for i, j, _, _ in crossings]
        try:
            above, _ = _closure_and_order(n, edges, [dict(bounds=b) for b in bounds])
        except AssertionError:
            continue
        direct = set(edges)
        transitive = any(_overlap(bounds[i], bounds[j]) and (i,j) not in direct and (j,i) not in direct
                         for i in range(n) for j in range(i))
        if not require_transitive or transitive:
            possible.append(order)
    if not possible: return None
    order = possible[int(rng.integers(len(possible)))]
    rank = {i: order.index(i) for i in range(n)}
    extra_cap = 2 if extent == (6,7) else 7
    rs, h = _coordinates(rng, rows, extra_cap); cs, w = _coordinates(rng, cols, extra_cap)
    paths_px = [_expand(p, rs, cs) for p in paths]
    palette = list(map(int, rng.permutation(10)))
    bg, marker, carrier, bridge = palette[:4]
    paints = list(map(int, rng.choice(palette[4:], n, replace=False)))
    if lesson is None and n >= 3 and rng.random() < .25:
        paints[int(rng.integers(n))] = paints[0]
    inp = np.full((h, w), bg, dtype=np.int8)
    for path in paths_px:
        for p in path: inp[p] = carrier
    for i, j, (r, c), axis_i in crossings:
        upper_axis = axis_i if rank[i] < rank[j] else 1-axis_i
        p = (rs[r], cs[c])
        for d in ((0, 2) if upper_axis == 0 else (1, 3)):
            dr, dc = _DIRS[d]
            inp[p[0]+dr, p[1]+dc] = bridge
    sheets = []
    for i, path in enumerate(paths_px):
        a, b = path[0], path[-1]
        dr, dc = int(np.sign(b[0]-a[0])), int(np.sign(b[1]-a[1]))
        ends = []
        if lesson == "pairing": modes = [True, False]
        elif lesson == "reverse": modes = [i == 0, True]
        else: modes = [(True, False), (False, True), (True, True)][int(rng.integers(3))]
        for endpoint, ir, ic, labeled in ((a, dr, dc, modes[0]), (b, -dr, -dc, modes[1])):
            r, c = endpoint
            cells = {(r,c), (r+ir,c), (r,c+ic)}
            for p in cells:
                if inp[p] not in (bg, carrier): return None
                inp[p] = marker
            dot = (r+ir, c+ic)
            if inp[dot] != bg: return None
            if labeled: inp[dot] = paints[i]
            ends.append(dict(elbow=endpoint, inward=(ir,ic), cells=cells,
                             label=paints[i] if labeled else None))
        sheets.append(_sheet(ends[0], ends[1], marker))
    out = _compose(inp.shape, bg, marker, sheets, order)
    # All spatial transforms preserve the local outward and bridge conventions.
    if rng.integers(2): inp, out = inp[::-1].copy(), out[::-1].copy()
    if rng.integers(2): inp, out = inp[:, ::-1].copy(), out[:, ::-1].copy()
    if rng.integers(2): inp, out = inp.T.copy(), out.T.copy()
    try:
        info = _parse(inp)
        assert np.array_equal(_predict(info), out)
        f = _features(info, out)
        if lesson is not None and not all(np.any(out == color) for color in paints): return None
        if lesson == "pairing" and not (f["pairing_essential"] and f["depth_essential"] and f["forward"]): return None
        if lesson == "reverse" and not (f["hidden_markers"] and f["depth_essential"] and f["both"] and f["reverse"]): return None
        if require_transitive and not f["transitive"]: return None
        if extent == (8,9) and max(max(b[2]-b[0]+1,b[3]-b[1]+1) for b in f["bounds"]) < 18:
            return None
    except AssertionError:
        return None
    return dict(input=inp, output=out)


def _count(name, value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if nt <= nq: raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(["count", "extent", "transitivity"]))
    lessons = list(map(int, rng.choice(nt, 2, replace=False)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            lesson = ("pairing" if index == lessons[0] else "reverse" if index == lessons[1] else None) if split == "train" else None
            transitive = split == "test" and regime == "transitivity"
            if lesson is not None: n = 2
            elif regime == "count": n = 2 if split == "train" else int(rng.choice([3,3,4]))
            elif transitive: n = int(rng.choice([3,3,4]))
            elif regime == "extent" and split == "train": n = int(rng.choice([2,2,3]))
            else: n = int(rng.choice([2,2,3,3,4]))
            extent = (6,7) if regime == "extent" and split == "train" else (8,9) if regime == "extent" else (7,9)
            for attempt in range(1200):
                pair = _make_pair(rng, n, extent, lesson, transitive)
                if pair is None: continue
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key); puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded spatial sampler could not satisfy its witnesses")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, evidence = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = (inp.shape, inp.tobytes())
            assert key not in seen; seen.add(key)
            info = _parse(inp)
            assert np.array_equal(_predict(info), out), "output disagrees with visible wiring"
            assert not np.array_equal(inp, out)
            # All wiring is consumed, and no sheet can create pixels beyond its box.
            union = np.zeros(inp.shape, bool)
            for s in info["sheets"]:
                r0,c0,r1,c1 = s["bounds"]; union[r0:r1+1,c0:c1+1] = True
            assert np.all(out[~union] == info["background"])
            assert not np.any((out == info["carrier"]) | (out == info["bridge"]))
            f = _features(info, out)
            assert f["retained_markers"] > 0
            assert any(np.any((out == s["paint"]) & (inp == info["background"])) for s in info["sheets"]), \
                "no visible rectangle completion"
            if split == "train":
                evidence.update(k for k in ("depth_essential", "pairing_essential", "hidden_markers", "forward", "reverse", "both") if f[k])
    assert {"depth_essential", "pairing_essential", "hidden_markers", "forward", "reverse", "both"} <= evidence, \
        "demonstrations lack an inherited-mechanism witness"


def self_test():
    records, structures, signatures = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple((p["input"].shape, p["input"].tobytes()) for s in puzzle for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                assert np.array_equal(solve(inp), out)
                info = _parse(inp); records.append(_features(info, out))
                roles = {}; normalized = tuple(roles.setdefault(int(v),len(roles)) for v in inp.flat)
                structures.add((inp.shape, normalized))
                # Equivariance is semantic: color names and axis direction carry no hidden phase.
                lut = np.arange(9,-1,-1,dtype=np.int8)
                assert np.array_equal(solve(lut[inp]), lut[out])
                assert np.array_equal(solve(inp.T), out.T)
    assert len(signatures) == 42 and len(structures) >= 180
    assert {f["sheets"] for f in records} == {2,3,4}
    assert len({f["shape"] for f in records}) >= 35
    assert len({f["bounds"] for f in records}) >= 150
    assert {f["roles"][0] for f in records} == set(range(10))
    assert any(f["transitive"] for f in records)
    assert any(len(f["diagonals"]) == 2 for f in records)
    assert any(len(set(f["paints"])) < f["sheets"] for f in records)
    a, b = generate_puzzle(918, 2, 1), generate_puzzle(918, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in a for x,y in zip(a[s],b[s]) for k in x)
    for nt in range(2,11):
        for nq in range(1,min(4,nt-1)+1):
            p = generate_puzzle(830+nt*10+nq,nt,nq)
            assert (len(p["train"]),len(p["test"])) == (nt,nq)
    for nt,nq in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(3.5,1),(4,None),(None,1)):
        try: generate_puzzle(0,nt,nq)
        except ValueError: pass
        else: raise AssertionError("illegal counts accepted")
    damaged = {s:[{k:v.copy() for k,v in p.items()} for p in a[s]] for s in a}
    damaged["test"][0]["output"][0,0] = (int(damaged["test"][0]["output"][0,0])+1)%10
    try: validate_generated_puzzle(damaged)
    except AssertionError: pass
    else: raise AssertionError("outside-sheet corruption accepted")
    return dict(seeds=42, pairs=len(records), structures=len(structures),
                shapes=len({f["shape"] for f in records}),
                sheet_counts=sorted({f["sheets"] for f in records}),
                transitive_pairs=sum(bool(f["transitive"]) for f in records))


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(),separators=(",",":")))
