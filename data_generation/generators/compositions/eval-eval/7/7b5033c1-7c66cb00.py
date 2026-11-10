"""Geodesic loading docks: independently route packets along a folded cable.

Only NumPy and the standard library are used.  Construction labels use the
sampled ordered cable; solve independently recovers its graph from input pixels.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np


FOUR = ((-1, 0), (0, 1), (1, 0), (0, -1))
AUDIT_SEEDS = (0, 980635612, 996678371, 301740191, 704563877, 366122450,
               724293429, 813538782, 261947452, 20051843, 700551654,
               839257563, 757330249, 428877819, 821080178, 365886727,
               17443398, 659936945, 711815652, 851500407, 348177527,
               614648856, 285453648, 300339265, 124955126, 991461462,
               529945338, 970713742, 271582512, 654068435, 912374640,
               285213675, 679268972, 556294887, 767484648, 312368025,
               37703223, 360015454, 591837398, 384713922, 173435485,
               718160914)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape),
             "grids must be two dimensional with dimensions 1..30")
    _require(np.issubdtype(g.dtype, np.integer) and np.all((0 <= g) & (g <= 9)),
             "grid cells must be integer colors 0..9")
    return g.astype(int, copy=False)


def _shape(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _rigid_shape(cells):
    """Ignore translation, rotation, and reflection when comparing masks."""
    variants = []
    for reflect in (-1, 1):
        points = [(r, reflect*c) for r, c in cells]
        for _ in range(4):
            variants.append(_shape(points))
            points = [(c, -r) for r, c in points]
    return min(variants)


def _key(grid):
    g = np.asarray(grid, dtype=np.int8)
    return g.shape, g.tobytes()


def _analyze(input_grid):
    """Recover crossbars, the induced cable, and packets without generation data."""
    g = _grid(input_grid)
    bg = int(g[0, 0])
    _require(np.all(g[0] == bg) and np.all(g[-1] == bg) and
             np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg),
             "a uniform background border is required")
    cells = {tuple(map(int, p)) for p in np.argwhere(g != bg)}
    neighbors = {p: [(p[0]+dr, p[1]+dc) for dr, dc in FOUR
                     if (p[0]+dr, p[1]+dc) in cells] for p in cells}
    arms, markers = set(), {}
    for p, ns in neighbors.items():
        if len(ns) != 4:
            continue
        leaves = [q for q in ns if len(neighbors[q]) == 1]
        _require(len(leaves) == 2 and
                 leaves[0][0]+leaves[1][0] == 2*p[0] and
                 leaves[0][1]+leaves[1][1] == 2*p[1],
                 "checkpoint needs exactly two opposite leaf tips")
        _require(g[leaves[0]] == g[leaves[1]] and g[leaves[0]] != g[p],
                 "crossbar tips must agree and differ from their center")
        markers[p] = (int(g[p]), int(g[leaves[0]]), tuple(leaves))
        arms.update(leaves)
    _require(markers, "no crossbar checkpoints")
    backbone = cells - arms
    adj = {p: [q for q in neighbors[p] if q in backbone] for p in backbone}
    ends = [p for p in backbone if len(adj[p]) == 1]
    _require(len(ends) == 2 and all(len(ns) in (1, 2) for ns in adj.values()),
             "removing checkpoint tips must leave one simple cable")
    _require(ends[0][0] != ends[1][0], "endpoint rows must differ")
    path, previous, current = [], None, min(ends, key=lambda p: p[0])
    visited = set()
    while current is not None:
        _require(current not in visited, "cycle in cable")
        path.append(current)
        visited.add(current)
        ns = [p for p in adj[current] if p != previous]
        _require(len(ns) <= 1, "branch in cable")
        previous, current = current, ns[0] if ns else None
    _require(visited == backbone, "disconnected foreground")
    values = [int(g[p]) for p in path]
    counts = Counter(values)
    track, frequency = counts.most_common(1)[0]
    _require(frequency * 2 > len(path), "neutral cable must have a strict majority")
    gates = {i: dict(route=markers[p][0], rail=markers[p][1], arms=markers[p][2])
             for i, p in enumerate(path) if p in markers}
    _require(all(b["route"] != track and b["rail"] != track for b in gates.values()),
             "checkpoint colors cannot equal neutral cable")
    route_colors = {b["route"] for b in gates.values()}
    packets, i = [], 0
    while i < len(path):
        if i in gates or values[i] == track:
            i += 1
            continue
        color, start = values[i], i
        _require(color in route_colors, "packet has no routing color")
        while i < len(path) and i not in gates and values[i] == color:
            i += 1
        source = tuple(range(start, i))
        _require(not any(j-1 in gates or j+1 in gates for j in source),
                 "source packets need a neutral cell beside every checkpoint")
        candidates = [j for j, b in gates.items() if j > source[-1] and b["route"] == color]
        _require(candidates, "packet has no downstream matching checkpoint")
        gate = min(candidates)
        target = tuple(range(gate-len(source), gate))
        _require(target[0] >= 0 and not set(target) & gates.keys(),
                 "packet does not fit before its checkpoint")
        _require(source[-1] < gate-1, "packet must visibly move forward")
        packets.append(dict(color=color, source=source, target=target, gate=gate))
    destinations = [p["gate"] for p in packets]
    _require(len(destinations) == len(set(destinations)) and set(destinations) == set(gates),
             "every checkpoint must receive exactly one packet")
    occupied = set()
    for packet in packets:
        _require(not occupied.intersection(packet["target"]), "destination collision")
        occupied.update(packet["target"])
    out = g.copy()
    for packet in packets:
        for j in packet["source"]:
            out[path[j]] = track
    for packet in packets:
        for j in packet["target"]:
            out[path[j]] = gates[packet["gate"]]["rail"]
    return dict(grid=g, background=bg, track=track, path=path,
                gates=gates, packets=packets, output=out)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Move each monochromatic packet to its next matching crossbar along the cable."""
    return _analyze(input_grid)["output"].copy()


def _coarse_walk(rng, edges, above):
    """Self-avoiding coarse walks permit close folds but never shared vertices."""
    for _ in range(220):
        p = tuple(map(int, rng.integers(0, 6, size=2)))
        walk, occupied, old = [p], {p}, None
        persistence = float(rng.uniform(.4, 5))
        for __ in range(edges):
            choices = [(walk[-1][0]+dr, walk[-1][1]+dc) for dr, dc in FOUR]
            choices = [q for q in choices if 0 <= q[0] < 6 and 0 <= q[1] < 6 and q not in occupied]
            if not choices:
                break
            weights = np.array([persistence if (q[0]-walk[-1][0], q[1]-walk[-1][1]) == old
                                else 1 for q in choices])
            q = choices[int(rng.choice(len(choices), p=weights/weights.sum()))]
            old = (q[0]-walk[-1][0], q[1]-walk[-1][1])
            walk.append(q)
            occupied.add(q)
        if len(walk) != edges+1:
            continue
        rotation, reflect = int(rng.integers(4)), int(rng.choice((-1, 1)))
        walk = [(r, reflect*c) for r, c in walk]
        for __ in range(rotation):
            walk = [(c, -r) for r, c in walk]
        if walk[0][0] == walk[-1][0]:
            continue
        if walk[0][0] > walk[-1][0]:
            walk.reverse()
        if above and min(r for r, c in walk) >= walk[0][0]:
            continue
        return walk
    # Finite reserve. The above-endpoint teaching case only requests 10..16 edges.
    if above:
        walk = ([(2, 1), (1, 1)] + [(0, c) for c in range(1, 6)] +
                [(r, 5) for r in range(1, 6)] + [(5, c) for c in range(4, -1, -1)])
        _require(edges < len(walk), "above-endpoint reserve is too short")
    else:
        walk = [(r, c) for r in range(6) for c in (range(6) if r % 2 == 0 else range(5, -1, -1))]
    return walk[:edges+1]


def _fine_path(rng, edges, above):
    coarse = _coarse_walk(rng, edges, above)
    rows, cols = sorted({r for r, c in coarse}), sorted({c for r, c in coarse})
    # Independent 4/5-cell intervals vary section lengths without changing topology.
    rmap, cmap = {rows[0]: 0}, {cols[0]: 0}
    for a, b in zip(rows, rows[1:]):
        rmap[b] = rmap[a] + int(rng.integers(4, 6))
    for a, b in zip(cols, cols[1:]):
        cmap[b] = cmap[a] + int(rng.integers(4, 6))
    nodes = [(rmap[r], cmap[c]) for r, c in coarse]
    path, possible = [nodes[0]], []
    for a, b in zip(nodes, nodes[1:]):
        dr, dc = int(np.sign(b[0]-a[0])), int(np.sign(b[1]-a[1]))
        length = abs(b[0]-a[0])+abs(b[1]-a[1])
        possible.append(len(path)-1+length//2)
        for _ in range(length):
            path.append((path[-1][0]+dr, path[-1][1]+dc))
    return path, possible


def _gate_positions(rng, possible, count, prefix, gap):
    options = [i for i in possible if i >= prefix]
    for _ in range(100):
        if len(options) < count:
            break
        picks = sorted(map(int, rng.choice(options, count, replace=False)))
        if all(b-a >= gap for a, b in zip(picks, picks[1:])):
            return picks
    picks = []
    for i in options:
        if not picks or i-picks[-1] >= gap:
            picks.append(i)
            if len(picks) == count:
                return picks
    return None


def _place_packets(rng, positions, routes, lengths, touching):
    occupied, packets = {}, []
    forbidden = {j+d for j in positions for d in (-1, 0, 1)}

    def add(g, start):
        source = tuple(range(start, start+lengths[g]))
        for j in source:
            occupied[j] = routes[g]
        packets.append(dict(gate=positions[g], source=source,
                            target=tuple(range(positions[g]-lengths[g], positions[g]))))

    if touching:
        order = list(map(int, rng.permutation(2)))
        total = lengths[0]+lengths[1]
        start = int(rng.integers(0, positions[0]-total))
        for k in order:
            add(k, start)
            start += lengths[k]
    for k, gate in enumerate(positions):
        if touching and k < 2:
            continue
        earlier = [positions[j] for j in range(k) if routes[j] == routes[k]]
        low = max(earlier)+2 if earlier else 0
        choices = []
        for start in range(low, gate-lengths[k]):
            source = set(range(start, start+lengths[k]))
            if source.intersection(forbidden) or source.intersection(occupied):
                continue
            if any(occupied.get(j+d) == routes[k] for j in source for d in (-1, 1)):
                continue
            choices.append(start)
        if not choices:
            return None
        add(k, int(rng.choice(choices)))
    return packets


def _render(rng, path, positions, routes, rails, packets, background, track):
    """Construction ground truth uses sampled indices, never solve or its parser."""
    tips = []
    for gate in positions:
        a, p, b = path[gate-1], path[gate], path[gate+1]
        _require(a[0]+b[0] == 2*p[0] and a[1]+b[1] == 2*p[1], "bent crossbar center")
        dr, dc = b[0]-p[0], b[1]-p[1]
        tips.append(((p[0]-dc, p[1]+dr), (p[0]+dc, p[1]-dr)))
    all_cells = path + [p for pair in tips for p in pair]
    r0, c0 = min(r for r, c in all_cells), min(c for r, c in all_cells)
    h = max(r for r, c in all_cells)-r0+3
    w = max(c for r, c in all_cells)-c0+3
    _require(h <= 30 and w <= 30, "construction exceeds canvas")
    extra_h, extra_w = int(rng.integers(0, min(3, 30-h)+1)), int(rng.integers(0, min(3, 30-w)+1))
    top, left = 1+int(rng.integers(extra_h+1)), 1+int(rng.integers(extra_w+1))
    shift = lambda p: (p[0]-r0+top, p[1]-c0+left)
    path = [shift(p) for p in path]
    tips = [[shift(p) for p in pair] for pair in tips]
    inp = np.full((h+extra_h, w+extra_w), background, dtype=int)
    for p in path:
        inp[p] = track
    for i, gate in enumerate(positions):
        inp[path[gate]] = routes[i]
        for p in tips[i]:
            inp[p] = rails[i]
    gate_to_index = {g: k for k, g in enumerate(positions)}
    for packet in packets:
        color = routes[gate_to_index[packet["gate"]]]
        for j in packet["source"]:
            inp[path[j]] = color
    out = inp.copy()
    for packet in packets:
        for j in packet["source"]:
            out[path[j]] = track
    for packet in packets:
        color = rails[gate_to_index[packet["gate"]]]
        for j in packet["target"]:
            out[path[j]] = color
    return {"input": inp, "output": out}


def _features(a):
    path, gates, packets = a["path"], a["gates"], a["packets"]
    source_owner = {j: k for k, p in enumerate(packets) for j in p["source"]}
    touching = any(j+1 in source_owner and source_owner[j+1] != k and
                   packets[k]["color"] != packets[source_owner[j+1]]["color"]
                   for j, k in source_owner.items())
    repeated = any(x["route"] == y["route"] and x["rail"] != y["rail"]
                   for i, x in gates.items() for j, y in gates.items() if i < j)
    warps = sum(_shape([path[j] for j in p["source"]]) !=
                _shape([path[j] for j in p["target"]]) for p in packets)
    deforms = sum(_rigid_shape([path[j] for j in p["source"]]) !=
                  _rigid_shape([path[j] for j in p["target"]]) for p in packets)
    steps = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:])]
    return dict(length=len(path), gates=len(gates), packets=len(packets),
                packet_lengths=tuple(len(p["source"]) for p in packets),
                max_packet=max(len(p["source"]) for p in packets),
                touching=touching, repeated=repeated, warps=warps, deforms=deforms,
                above=min(r for r, c in path) < path[0][0],
                turns=sum(a != b for a, b in zip(steps, steps[1:])),
                skips=sum(any(p["source"][-1] < g < p["gate"] for g in gates) for p in packets),
                route_reuse=any(b["rail"] in {x["route"] for x in gates.values()} for b in gates.values()),
                shape=a["grid"].shape, background=a["background"], track=a["track"],
                geometry=_shape(path))


def _sample_pair(rng, mode, split, lesson, background):
    test = split == "test"
    count = int(rng.integers(4, 6) if mode == "count" and test else
                rng.integers(2, 4) if lesson or mode != "length" else rng.integers(1, 5))
    maximum = 6 if mode == "packet" and test else 3 if mode == "packet" else 4
    lengths = list(map(int, rng.integers(4 if mode == "packet" and test else 1, maximum+1, size=count)))
    if lesson:
        lengths[0] = 3
        lengths[1] = int(rng.choice((1, 2)))
    if mode == "length":
        edges = int(rng.integers(23, 29) if test else rng.integers(11, 16))
    else:
        minimum = max(11, 3*count+1, (2*(sum(lengths)+count)+3)//4)
        edges = int(rng.integers(minimum, minimum+5))
    # Enough blank cable remains to identify neutral without a color dictionary.
    edges = max(edges, 3*count+1, (2*(sum(lengths)+count)+3)//4)
    for _ in range(100):
        path, possible = _fine_path(rng, edges, above=lesson == "repeat")
        prefix = max(maximum+4, sum(lengths[:2])+2 if lesson == "split" else 8)
        positions = _gate_positions(rng, possible, count, prefix, maximum+3)
        if positions is None:
            continue
        palette = list(map(int, rng.permutation([c for c in range(10) if c != background])))
        track = palette[0]
        route_pool = palette[1:1+int(rng.integers(2, 5))]
        routes = list(map(int, rng.choice(route_pool, size=count)))
        if lesson == "repeat":
            routes[1] = routes[0]
        if lesson == "split":
            routes[1] = int(rng.choice([c for c in route_pool if c != routes[0]]))
        rails = [int(rng.choice([c for c in range(10) if c not in (background, track, route)]))
                 for route in routes]
        if lesson == "repeat" and rails[0] == rails[1]:
            rails[1] = int(rng.choice([c for c in range(10) if c not in
                                      (background, track, routes[1], rails[0])]))
        packets = _place_packets(rng, positions, routes, lengths, lesson == "split")
        if packets is None:
            continue
        try:
            pair = _render(rng, path, positions, routes, rails, packets, background, track)
            a = _analyze(pair["input"])
        except ValueError:
            continue
        if lesson == "repeat" and not _features(a)["deforms"]:
            continue
        _require(np.array_equal(a["output"], pair["output"]), "construction/parser disagreement")
        return pair
    raise RuntimeError("bounded packet/cable construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, low, high, name in ((num_train, 2, 10, "num_train"), (num_test, 1, 4, "num_test")):
        _require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
                 and low <= value <= high, f"illegal {name}")
    _require(num_train > num_test, "num_train must exceed num_test")
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_))
             and seed >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    mode = str(rng.choice(("length", "count", "packet")))
    common_bg = int(rng.integers(10)) if rng.random() < .35 else None
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            lesson = ("repeat" if index == 0 else "split" if index == 1 else None) if split == "train" else None
            for _ in range(30):
                bg = int(rng.integers(10)) if common_bg is None else common_bg
                pair = _sample_pair(rng, mode, split, lesson, bg)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded uniqueness search exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4 and
             len(puzzle["train"]) > len(puzzle["test"]), "illegal example counts")
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "canvas dimensions must be preserved")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            a = _analyze(inp)
            _require(np.array_equal(out, a["output"]), "wrong docking output")
            path, gates = a["path"], a["gates"]
            source = {path[j] for p in a["packets"] for j in p["source"]}
            target = {path[j] for p in a["packets"] for j in p["target"]}
            changed = {tuple(map(int, p)) for p in np.argwhere(inp != out)}
            _require(changed and changed <= source | target, "changed a non-packet cable/marker pixel")
            _require(sum(out[p] != a["track"] for i, p in enumerate(path) if i not in gates)
                     == len(source) == len(target), "packet length was not conserved")
            features[split].append(_features(a))
    tr, te = features["train"], features["test"]
    _require(any(f["repeated"] for f in tr), "missing repeated-route teaching case")
    _require(any(f["touching"] and f["skips"] for f in tr), "missing independent multicolor routing lesson")
    _require(any(f["deforms"] for f in tr), "missing evidence for intrinsic rather than rigid movement")
    if any(f["above"] for f in te):
        _require(any(f["above"] for f in tr), "upper-endpoint distinction was not demonstrated")
    shifts = (min(f["length"] for f in te) > max(f["length"] for f in tr),
              min(f["gates"] for f in te) > max(f["gates"] for f in tr),
              min(n for f in te for n in f["packet_lengths"]) > max(f["max_packet"] for f in tr))
    _require(any(shifts), "missing measured train/test extrapolation")


def _fixture():
    """Known index-level answer: two equal-color packets, different destinations."""
    path = [(1, 2)]
    for (dr, dc), steps in (((0, 1), 6), ((1, 0), 4), ((0, -1), 6),
                             ((1, 0), 4), ((0, 1), 6), ((1, 0), 4)):
        for _ in range(steps):
            path.append((path[-1][0]+dr, path[-1][1]+dc))
    inp = np.zeros((15, 11), dtype=int)
    for p in path:
        inp[p] = 5
    for j in (0, 1, 2, 11, 12):
        inp[path[j]] = 1
    for gate, rail in ((8, 3), (18, 4)):
        r, c = path[gate]
        inp[r, c] = 1
        inp[r, c-1] = inp[r, c+1] = rail
    expected = inp.copy()
    for j in (0, 1, 2, 11, 12):
        expected[path[j]] = 5
    for j in (5, 6, 7):
        expected[path[j]] = 3
    for j in (16, 17):
        expected[path[j]] = 4
    return inp, expected


def self_test():
    """Bounded semantic checks and actual-array diversity measurements on 42 seeds."""
    inp, expected = _fixture()
    assert np.array_equal(solve(inp), expected)
    assert _features(_analyze(inp))["warps"] >= 1
    # Palette equivariance proves literal digit identities do not encode roles.
    permutation = np.array([8, 7, 0, 2, 6, 1, 9, 3, 5, 4])
    assert np.array_equal(solve(permutation[inp]), permutation[expected])
    # Horizontal reflection preserves the higher endpoint and the whole answer.
    assert np.array_equal(solve(np.fliplr(inp)), np.fliplr(expected))
    changed_rail = inp.copy()
    changed_rail[3, 7] = changed_rail[3, 9] = 2
    expected_rail = expected.copy()
    expected_rail[3, 7] = expected_rail[3, 9] = 2
    for p in ((1, 7), (1, 8), (2, 8)):
        expected_rail[p] = 2
    assert np.array_equal(solve(changed_rail), expected_rail)
    records, signatures = [], set()
    for seed in AUDIT_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            records.extend((split, _features(_analyze(p["input"]))) for p in puzzle[split])
    assert len(signatures) == 42
    assert len({f["shape"] for _, f in records}) >= 35
    assert len({f["geometry"] for _, f in records}) >= 150
    assert {f["background"] for _, f in records} == set(range(10))
    assert {f["track"] for _, f in records} == set(range(10))
    assert len({f["length"] for _, f in records}) >= 30
    assert len({f["turns"] for _, f in records}) >= 12
    assert {f["gates"] for _, f in records} >= {1, 2, 3, 4, 5}
    assert {n for _, f in records for n in f["packet_lengths"]} == set(range(1, 7))
    assert any(f["route_reuse"] for _, f in records)
    assert all(any(s == split and f["warps"] for s, f in records) for split in ("train", "test"))
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            p = generate_puzzle(4242, train, test)
            assert len(p["train"]) == train and len(p["test"]) == test
    first, again = generate_puzzle(19), generate_puzzle(19)
    assert all(np.array_equal(a[k], b[k]) for s in first for a, b in zip(first[s], again[s])
               for k in ("input", "output"))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(0)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted answer accepted")
    broken = inp.copy()
    broken[3, 7] = 0
    try:
        solve(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("one-sided crossbar accepted")
    return dict(seeds=42, pairs=len(records), shapes=len({f["shape"] for _, f in records}),
                geometries=len({f["geometry"] for _, f in records}),
                path_lengths=sorted({f["length"] for _, f in records}),
                gate_counts=sorted({f["gates"] for _, f in records}),
                packet_lengths=sorted({n for _, f in records for n in f["packet_lengths"]}),
                semantic_verification="construction and input-derived checks; no independent solver")


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
