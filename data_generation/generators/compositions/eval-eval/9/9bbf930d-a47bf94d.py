"""Paired-tip switchyards; only NumPy and the Python standard library.

Construction uses an abstract graph of junction ports. solve() reconstructs a
different, cell-level directional graph from its sole input argument.
"""
from collections import Counter
import json

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # N, E, S, W
PAIRINGS = (((0, 2), (1, 3)), ((0, 1), (2, 3)), ((0, 3), (1, 2)))
DIAMOND = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
CHECKER = ~DIAMOND


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _step(p, d, amount=1):
    dr, dc = DIRS[d]
    return p[0] + amount * dr, p[1] + amount * dc


def _grid(value):
    g = np.asarray(value)
    _need(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "grid dimensions")
    _need(np.issubdtype(g.dtype, np.integer), "integer cells required")
    _need(bool(np.all((g >= 0) & (g <= 9))), "ARC colors required")
    return g.astype(np.int8, copy=True)


def _inside(g, p):
    return 0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1]


def _signature(g):
    return g.shape, g.tobytes()


def _block_kind(block, bg):
    colors = set(map(int, np.unique(block))) - {bg}
    if not colors:
        return "blank", None
    _need(len(colors) == 1, "socket has multiple labels")
    label = colors.pop()
    if np.all(block == label):
        return "solid", label
    for name, mask in (("diamond", DIAMOND), ("checker", CHECKER)):
        if np.array_equal(block, np.where(mask, label, bg)):
            return name, label
    raise ValueError("socket is not blank, solid, diamond or checker")


def _trace_graph(graph):
    """Trace degree-two cell routes; edge keys give incoming directions."""
    _need(graph and all(len(edges) in (1, 2) for edges in graph.values()),
          "wire must have degree one or two")
    ends = sorted(p for p, edges in graph.items() if len(edges) == 1)
    _need(len(ends) >= 4 and len(ends) % 2 == 0, "socket endpoint count")
    seen, paired, routes = set(), set(), []
    for start in ends:
        if start in paired:
            continue
        p, incoming, cells = start, None, []
        while True:
            _need(p not in seen, "route revisits a cell or merges")
            seen.add(p)
            cells.append(p)
            if p != start and len(graph[p]) == 1:
                break
            choices = [d for d in graph[p] if d != incoming]
            _need(len(choices) == 1, "ambiguous wire continuation")
            p, incoming = graph[p][choices[0]]
        paired.update((start, p))
        routes.append((start, p, cells))
    _need(seen == set(graph), "unattached cycle or routing fragment")
    return routes


def _parse_carrier(g, bg, carrier):
    centers, tips, port_owner = [], set(), {}
    for raw in np.argwhere(g == carrier):
        p = tuple(map(int, raw))
        flank = [_step(p, d) for d in range(4)]
        ports = [_step(p, d, 2) for d in range(4)]
        if not all(_inside(g, q) for q in flank + ports):
            continue
        values = [int(g[q]) for q in flank]
        if (any(v in (bg, carrier) for v in values)
                or sorted(Counter(values).values()) != [2, 2]
                or not all(g[q] == carrier for q in ports)):
            continue
        _need(not tips.intersection(flank), "overlapping guide glyphs")
        _need(not set(ports).intersection(port_owner), "shared switch ports")
        tips.update(flank)
        for d, q in enumerate(ports):
            port_owner[q] = (p, d)
        pairings = []
        for color in sorted(set(values)):
            ds = [d for d, v in enumerate(values) if v == color]
            pairings.append(tuple(ds))
        mode = next(i for i, pairs in enumerate(PAIRINGS)
                    if {frozenset(x) for x in pairs} == {frozenset(x) for x in pairings})
        centers.append(dict(center=p, ports=ports, values=values, pairs=pairings, mode=mode))
    _need(centers, "at least one paired-tip junction required")
    center_cells = {n["center"] for n in centers}
    _need(center_cells.isdisjoint(tips | set(port_owner)), "overlapping junctions")
    visible = set(map(tuple, np.argwhere(g == carrier))) - center_cells
    graph = {p: {} for p in visible}
    for p in visible:
        for d in range(4):
            q = _step(p, d)
            if q in visible:
                graph[p][d] = (q, (d + 2) % 4)
    for node in centers:
        for a, b in node["pairs"]:
            p, q = node["ports"][a], node["ports"][b]
            da, db = (a + 2) % 4, (b + 2) % 4
            _need(da not in graph[p] and db not in graph[q], "occupied switch edge")
            graph[p][da], graph[q][db] = (q, db), (p, da)
    routes = _trace_graph(graph)
    sockets, footprints = {}, set()
    for endpoint in sorted(p for p in graph if len(graph[p]) == 1):
        inward = next(iter(graph[endpoint]))
        center = _step(endpoint, (inward + 2) % 4, 2)
        r, c = center
        # A complete blank perimeter makes role recognition unambiguous.
        _need(2 <= r < g.shape[0] - 2 and 2 <= c < g.shape[1] - 2,
              "socket footprint must lie inside the blank perimeter")
        cells = {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        _need(not footprints.intersection(cells), "socket footprints overlap")
        _need(cells.isdisjoint(visible | tips | center_cells), "socket overlaps routing")
        footprints.update(cells)
        kind, label = _block_kind(g[r - 1:r + 2, c - 1:c + 2], bg)
        sockets[endpoint] = dict(center=center, kind=kind, label=label)
    guides = {int(g[p]) for p in tips}
    labels = {s["label"] for s in sockets.values() if s["label"] is not None}
    _need(labels and labels.isdisjoint(guides | {bg, carrier}), "overlapping palette roles")
    allowed = visible | tips | center_cells | footprints
    _need(all(tuple(p) in allowed for p in np.argwhere(g != bg)), "unrecognized colored cells")
    info = dict(grid=g, background=bg, carrier=carrier, nodes=centers, graph=graph,
                routes=routes, sockets=sockets, footprint=footprints, guides=guides,
                labels=labels)
    _route_data(info)  # Reject absent/conflicting labels before accepting a role.
    return info


def _read(input_grid):
    g = _grid(input_grid)
    bg = int(g[0, 0])
    _need(np.all(g[0] == bg) and np.all(g[-1] == bg)
          and np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg), "blank perimeter required")
    candidates = []
    for carrier in sorted(set(map(int, np.unique(g))) - {bg}):
        try:
            candidates.append(_parse_carrier(g, bg, carrier))
        except ValueError:
            continue
    _need(len(candidates) == 1, "input must have one carrier interpretation")
    return candidates[0]


def _route_data(info):
    data = []
    for a, b, cells in info["routes"]:
        sa, sb = info["sockets"][a], info["sockets"][b]
        labels = {s["label"] for s in (sa, sb) if s["label"] is not None}
        _need(len(labels) == 1, "route has absent or conflicting label clues")
        ka, kb = sa["kind"], sb["kind"]
        pa = None if ka == "blank" else int(ka == "checker")
        pb = None if kb == "blank" else int(kb == "checker")
        if pa is None:
            pa = 1 - pb
        if pb is None:
            pb = 1 - pa
        _need(pa != pb, "route endpoint phases must be complementary")
        data.append((a, b, labels.pop(), pa, pb))
    return data


def _complete(info):
    out = info["grid"].copy()
    for a, b, label, pa, pb in _route_data(info):
        for endpoint, phase in ((a, pa), (b, pb)):
            r, c = info["sockets"][endpoint]["center"]
            out[r - 1:r + 2, c - 1:c + 2] = np.where(
                CHECKER if phase else DIAMOND, label, info["background"])
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer all wiring, labels and phases solely from the supplied cells."""
    return _complete(_read(input_grid))


def _straight_ablation(info):
    graph = {p: dict(edges) for p, edges in info["graph"].items()}
    for node in info["nodes"]:
        for d, p in enumerate(node["ports"]):
            del graph[p][(d + 2) % 4]
        for a, b in PAIRINGS[0]:
            p, q = node["ports"][a], node["ports"][b]
            da, db = (a + 2) % 4, (b + 2) % 4
            graph[p][da], graph[q][db] = (q, db), (p, da)
    other = dict(info, graph=graph, routes=_trace_graph(graph))
    try:
        answer = _complete(other)
    except ValueError:
        return True
    return not np.array_equal(answer, _complete(info))


def _nonnearest(info):
    centers = {p: s["center"] for p, s in info["sockets"].items()}
    for a, b, _ in info["routes"]:
        distance = lambda p: abs(centers[a][0] - centers[p][0]) + abs(centers[a][1] - centers[p][1])
        if distance(b) > min(distance(p) for p in centers if p != a):
            return True
    return False


def _abstract_routes(occupied, modes):
    """Construction oracle on lattice ports, with no rendered cells."""
    graph = {(r, c, d): [] for r, c in occupied for d in range(4)}
    for r, c in sorted(occupied):
        for a, b in PAIRINGS[modes[r, c]]:
            graph[r, c, a].append((r, c, b))
            graph[r, c, b].append((r, c, a))
        for d, (dr, dc) in enumerate(DIRS):
            q = (r + dr, c + dc)
            if q in occupied:
                graph[r, c, d].append((*q, (d + 2) % 4))
    endpoints = sorted(p for p, neighbors in graph.items() if len(neighbors) == 1)
    seen, paired, routes = set(), set(), []
    for start in endpoints:
        if start in paired:
            continue
        p, previous, path = start, None, []
        while True:
            _need(p not in seen, "abstract route cycle")
            seen.add(p)
            path.append(p)
            if p != start and len(graph[p]) == 1:
                break
            onward = [q for q in graph[p] if q != previous]
            _need(len(onward) == 1, "abstract continuation")
            previous, p = p, onward[0]
        paired.update((start, p))
        routes.append((start, p, path))
    _need(seen == set(graph), "abstract unattached loop")
    return routes


def _layout(rng, count, irregular):
    for _ in range(60):
        shapes = [(h, w) for h in range(1, 4) for w in range(1, 4)
                  if h * w >= count and (not irregular or h > 1 and w > 1 and h * w > count)]
        h, w = shapes[int(rng.integers(len(shapes)))]
        occupied = {(int(rng.integers(h)), int(rng.integers(w)))}
        while len(occupied) < count:
            frontier = sorted({(r + dr, c + dc) for r, c in occupied for dr, dc in DIRS
                               if 0 <= r + dr < h and 0 <= c + dc < w} - occupied)
            occupied.add(frontier[int(rng.integers(len(frontier)))])
        r0, c0 = min(r for r, c in occupied), min(c for r, c in occupied)
        occupied = {(r - r0, c - c0) for r, c in occupied}
        hh, ww = max(r for r, c in occupied) + 1, max(c for r, c in occupied) + 1
        if not irregular or hh > 1 and ww > 1 and hh * ww > count:
            return occupied
    # Connected prefix winds around the outside of a three-by-three lattice.
    path = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0), (1, 1)]
    return set(path[:count])


def _wiring(rng, occupied, lesson):
    order = sorted(occupied)
    for attempt in range(40):
        modes = {p: int(rng.integers(3)) for p in order}
        if lesson:
            modes[order[0]], modes[order[1]] = 0, 1 if lesson == 1 else 2
        elif all(m == 0 for m in modes.values()):
            modes[order[-1]] = int(rng.integers(1, 3))
        if attempt == 39:
            modes = {p: 0 for p in order}
            modes[order[-1]] = 1 if lesson != 2 else 2
        try:
            return modes, _abstract_routes(occupied, modes)
        except ValueError:
            continue
    raise ValueError("bounded wiring construction failed")


def _draw_geometry(rng, occupied, modes, palette, guide_count, compact):
    bg, carrier = palette[:2]
    guide_pool = palette[2:2 + guide_count]
    rows = [0]
    cols = [0]
    for _ in range(max(r for r, c in occupied)):
        rows.append(rows[-1] + (7 if compact else int(rng.integers(6, 10))))
    for _ in range(max(c for r, c in occupied)):
        cols.append(cols[-1] + (7 if compact else int(rng.integers(6, 10))))
    node_positions = {(r, c): (rows[r], cols[c]) for r, c in occupied}
    painted, socket_centers, footprints = {}, {}, set()

    def paint(p, color):
        _need(p not in painted or painted[p] == color, "geometry color collision")
        painted[p] = color

    for node in sorted(occupied):
        p = node_positions[node]
        paint(p, carrier)
        colors = list(map(int, rng.choice(guide_pool, 2, replace=False)))
        for color, pair in zip(colors, PAIRINGS[modes[node]]):
            for d in pair:
                paint(_step(p, d), color)
                paint(_step(p, d, 2), carrier)
    for r, c in sorted(occupied):
        p = node_positions[r, c]
        for d, (dr, dc) in enumerate(DIRS):
            other = (r + dr, c + dc)
            if other in occupied:
                if other < (r, c):
                    continue
                target = _step(node_positions[other], (d + 2) % 4, 2)
                q = _step(p, d, 2)
                paint(q, carrier)
                while q != target:
                    q = _step(q, d)
                    paint(q, carrier)
            else:
                lead = 2 if compact else int(rng.integers(2, 5))
                for amount in range(2, lead + 1):
                    paint(_step(p, d, amount), carrier)
                center = _step(p, d, lead + 2)
                socket_centers[r, c, d] = center
                cells = {(center[0] + rr, center[1] + cc)
                         for rr in (-1, 0, 1) for cc in (-1, 0, 1)}
                _need(not footprints.intersection(cells), "construction socket overlap")
                footprints.update(cells)
    _need(footprints.isdisjoint(painted), "socket/routing collision")
    extent = set(painted) | footprints
    minimum = tuple(min(p[i] for p in extent) for i in (0, 1))
    maximum = tuple(max(p[i] for p in extent) for i in (0, 1))
    margins = [1] * 4 if compact else list(map(int, rng.integers(1, 3, size=4)))
    h = maximum[0] - minimum[0] + 1 + margins[0] + margins[1]
    w = maximum[1] - minimum[1] + 1 + margins[2] + margins[3]
    _need(h <= 30 and w <= 30, "construction outside grid budget")
    offset = (margins[0] - minimum[0], margins[2] - minimum[1])
    move = lambda p: (p[0] + offset[0], p[1] + offset[1])
    g = np.full((h, w), bg, dtype=np.int8)
    for p, color in painted.items():
        g[move(p)] = color
    return g, {port: move(p) for port, p in socket_centers.items()}


def _render_clues(rng, skeleton, centers, routes, palette, guide_count, lesson):
    bg = palette[0]
    labels = palette[2 + guide_count:]
    count = len(routes)
    if lesson == 1:
        _need(count <= len(labels), "lesson needs distinct route labels")
        route_labels = list(map(int, rng.permutation(labels)[:count]))
    else:
        size = int(rng.integers(2, min(count, len(labels)) + 1))
        pool = list(map(int, rng.choice(labels, size, replace=False)))
        route_labels = pool + list(map(int, rng.choice(pool, count - size)))
        rng.shuffle(route_labels)
        if lesson == 2:
            route_labels[1] = route_labels[0]
            route_labels[2] = next(c for c in labels if c != route_labels[0])
    options = ["solid", "diamond", "checker", "both", "solid_both"]
    if lesson == 1:
        clues = ["solid", "diamond", "checker"] + ["both"] * (count - 3)
    elif lesson == 2:
        clues = ["both", "solid", "checker"] + ["diamond"] * (count - 3)
    else:
        mode = int(rng.integers(5))
        clues = (["solid"] * count if mode == 0 else ["checker"] * count if mode == 1
                 else ["diamond"] * count if mode == 2 else
                 ["solid_both"] * count if mode == 3 else
                 [options[int(rng.integers(len(options)))] for _ in routes])
    # At least one partner must be missing, so route tracing affects the work
    # requested by the example even when several other routes are completed.
    if all(c in ("both", "solid_both") for c in clues):
        clues[int(rng.integers(count))] = options[int(rng.integers(3))]
    inp, out = skeleton.copy(), skeleton.copy()
    for index, (a, b, _) in enumerate(routes):
        # The semantic diamond end is visibly identified, not tied to position.
        if rng.integers(2):
            a, b = b, a
        color, clue = route_labels[index], clues[index]
        diamond_kind = "solid" if clue in ("solid", "solid_both") else "diamond"
        for port, pattern, kind in (
            (a, DIAMOND, "blank" if clue == "checker" else diamond_kind),
            (b, CHECKER, "checker" if clue in ("checker", "both", "solid_both") else "blank"),
        ):
            r, c = centers[port]
            out[r - 1:r + 2, c - 1:c + 2] = np.where(pattern, color, bg)
            block = (np.full((3, 3), bg, dtype=np.int8) if kind == "blank" else
                     np.full((3, 3), color, dtype=np.int8) if kind == "solid" else
                     np.where(DIAMOND if kind == "diamond" else CHECKER, color, bg))
            inp[r - 1:r + 2, c - 1:c + 2] = block
    return {"input": inp, "output": out}


def _make_pair(rng, parameters, count, lesson, irregular):
    for attempt in range(80):
        occupied = _layout(rng, count, irregular)
        modes, routes = _wiring(rng, occupied, lesson)
        palette = (list(map(int, rng.permutation(10))) if parameters["mix_palette"]
                   else parameters["palette"])
        guide_count = parameters["guide_count"]
        try:
            skeleton, centers = _draw_geometry(rng, occupied, modes, palette, guide_count,
                                               compact=attempt >= 12)
            pair = _render_clues(rng, skeleton, centers, routes, palette, guide_count, lesson)
            if rng.integers(2):
                pair = {k: g[:, ::-1].copy() for k, g in pair.items()}
            rotation = int(rng.integers(4)) if parameters["mix_orientation"] else parameters["rotation"]
            pair = {k: np.rot90(g, rotation).copy() for k, g in pair.items()}
            info = _read(pair["input"])
            _need(np.array_equal(_complete(info), pair["output"]), "construction/reader disagree")
            if lesson == 1:
                _need(_straight_ablation(info) and _nonnearest(info), "lesson must distinguish shortcuts")
            return pair
        except ValueError:
            continue
    raise ValueError("bounded geometry/clue construction failed")


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                  ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not low <= value <= high:
            raise ValueError(f"{name} must lie in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    parameters = dict(palette=list(map(int, rng.permutation(10))),
                      guide_count=int(rng.integers(2, 5)),
                      mix_palette=bool(rng.integers(2)),
                      mix_orientation=bool(rng.integers(2)),
                      rotation=int(rng.integers(4)), shift=int(rng.integers(2)))
    puzzle, seen, demonstrated_modes = {"train": [], "test": []}, set(), set()
    for split, number in (("train", num_train), ("test", num_test)):
        for index in range(number):
            lesson = index + 1 if split == "train" and index < 2 else 0
            if lesson:
                nodes = int(rng.integers(2, 4))
            elif split == "train":
                nodes = int(rng.integers(1, 4 + parameters["shift"]))
            else:
                nodes = int(rng.integers(4, 8 if parameters["shift"] else 10))
            for _ in range(64):
                pair = _make_pair(rng, parameters, nodes, lesson,
                                  irregular=split == "test" and bool(parameters["shift"]))
                signature = _signature(pair["input"])
                actual_modes = {node["mode"] for node in _read(pair["input"])["nodes"]}
                if lesson == 2 and demonstrated_modes | actual_modes != {0, 1, 2}:
                    continue
                if signature not in seen:
                    break
            else:
                raise ValueError("bounded distinct-input construction failed")
            seen.add(signature)
            puzzle[split].append(pair)
            if split == "train":
                demonstrated_modes.update(actual_modes)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    features = {"mode" + str(n["mode"]) for n in info["nodes"]}
    labels = []
    for a, b, label, pa, pb in _route_data(info):
        labels.append(label)
        sa, sb = info["sockets"][a], info["sockets"][b]
        kinds = {sa["kind"], sb["kind"]}
        features.update(kinds)
        if "blank" not in kinds:
            features.add("both")
        if "checker" in kinds and "blank" in kinds:
            features.add("reverse")
        if ("solid" in kinds or "diamond" in kinds) and "blank" in kinds:
            features.add("forward")
    if len(labels) != len(set(labels)):
        features.add("repeat")
    if _nonnearest(info):
        features.add("nonnearest")
    if _straight_ablation(info):
        features.add("A_effect")
    return features


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "split lists required")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, evidence = set(), set()
    train_sizes, test_sizes = [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _need(inp.shape == out.shape, "dimensions must be preserved")
            signature = _signature(inp)
            _need(signature not in seen, "duplicate input")
            seen.add(signature)
            info = _read(inp)
            _need(any(s["kind"] == "blank" for s in info["sockets"].values()),
                  "generated examples must require label propagation")
            expected = _complete(info)
            _need(np.array_equal(out, expected), "output disagrees with input wiring/stencils")
            _need(not np.array_equal(inp, out), "generated example must require completion")
            outside = np.ones(inp.shape, dtype=bool)
            for p in info["footprint"]:
                outside[p] = False
            _need(np.array_equal(inp[outside], out[outside]), "nonterminal preservation")
            for a, b, label, pa, pb in _route_data(info):
                # One pair of complementary masks occupies exactly nine label cells.
                count = 0
                for endpoint in (a, b):
                    r, c = info["sockets"][endpoint]["center"]
                    count += int(np.count_nonzero(out[r - 1:r + 2, c - 1:c + 2] == label))
                _need(count == 9 and pa != pb, "complementary occupancy invariant")
            (train_sizes if split == "train" else test_sizes).append(len(info["nodes"]))
            if split == "train":
                evidence.update(_features(info))
                # Omitting propagation keeps every blank input socket blank.
                if any(s["kind"] == "blank" for s in info["sockets"].values()):
                    evidence.add("B_effect")
    required = {"mode0", "mode1", "mode2", "solid", "diamond", "checker",
                "forward", "reverse", "both", "repeat", "nonnearest", "A_effect", "B_effect"}
    _need(required <= evidence, "training lacks distinguishing lessons: " + str(sorted(required - evidence)))
    _need(min(test_sizes) >= 4 and max(train_sizes[:2]) <= 3, "network shift absent")


def inspect_pair(pair):
    """Metrics measured from actual input cells, with no sampling metadata."""
    info = _read(pair["input"])
    centers = sorted(n["center"] for n in info["nodes"])
    rows = sorted({r for r, c in centers})
    cols = sorted({c for r, c in centers})
    lengths, visits = [], []
    port_to_node = {p: n["center"] for n in info["nodes"] for p in n["ports"]}
    for a, b, cells in info["routes"]:
        lengths.append(int(sum(abs(p[0] - q[0]) + abs(p[1] - q[1])
                               for p, q in zip(cells, cells[1:]))))
        visits.append(len({port_to_node[p] for p in cells if p in port_to_node}))
    kinds = Counter(s["kind"] for s in info["sockets"].values())
    return dict(shape=list(info["grid"].shape), nodes=len(centers), routes=len(info["routes"]),
                modes=[n["mode"] for n in info["nodes"]],
                lattice_shape=[len(rows), len(cols)], concave=len(rows) * len(cols) > len(centers),
                pitches=[b - a for axis in (rows, cols) for a, b in zip(axis, axis[1:])],
                lengths=lengths, visits=visits, labels=len(info["labels"]), guides=len(info["guides"]),
                roles=[info["background"], info["carrier"]], clues=dict(kinds),
                repeated="repeat" in _features(info), A_effect=_straight_ablation(info))


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(p[k], q[k])
               for p, q in zip(a[s], b[s]) for k in ("input", "output")) for s in a)


def _switch_counterfactual():
    """Same geometry and clues; only guide positions change the answer."""
    rng = np.random.default_rng(901)
    palette = list(range(10))
    # Roles: background 0, carrier 1, guide 2/3, labels 4/5.
    grids = []
    for mode in (0, 1):
        g, centers = _draw_geometry(rng, {(0, 0)}, {(0, 0): mode}, palette, 2, True)
        for port, color in (((0, 0, 0), 4), ((0, 0, 3), 5)):
            r, c = centers[port]
            g[r - 1:r + 2, c - 1:c + 2] = color
        grids.append(g)
    a, b = grids
    _need(a.shape == b.shape and np.count_nonzero(a != b) == 2,
          "counterfactual must change only two guide tips")
    _need(not np.array_equal(solve(a), solve(b)), "guide pairing must affect output")
    return grids


def self_test():
    """Bounded 42-seed audit plus local regressions; returns compact metrics."""
    records, signatures, structures, shapes, palettes = [], set(), set(), set(), set()
    puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                before = inp.copy()
                _need(np.array_equal(solve(inp), out) and np.array_equal(inp, before),
                      "input-only solve or input mutation")
                signatures.add(_signature(inp))
                roles = {}
                normalized = tuple(tuple(roles.setdefault(int(c), len(roles)) for c in row) for row in inp)
                structures.add(normalized)
                shapes.add(inp.shape)
                palettes.add(tuple(map(int, np.unique(inp))))
                records.append(dict(seed=seed, split=split, **inspect_pair(pair)))
    _need(len(signatures) == 210 and len(structures) >= 150, "structural diversity collapsed")
    _need(len(shapes) >= 35 and len(palettes) >= 15, "shape/palette diversity collapsed")
    _need({r["nodes"] for r in records} >= set(range(1, 10)), "junction count coverage")
    _need({m for r in records for m in r["modes"]} == {0, 1, 2}, "switch modes")
    _need({r["guides"] for r in records} >= {2, 3, 4}, "guide inventory variation")
    _need(any(r["concave"] for r in records), "irregular topology absent")
    _need(max(v for r in records for v in r["visits"]) >= 4, "long switch composition absent")
    _need(len({p for r in records for p in r["pitches"]}) >= 4, "wire spacings collapsed")
    _need(len({r["routes"] for r in records}) >= 5, "route count variation")
    _need(any(len(set(r["lengths"])) > 1 for r in records), "unequal routes absent")
    _need(any(r["repeated"] for r in records if r["split"] == "test"), "repeated query labels absent")
    _need(sum(len({tuple(inspect_pair(p)["roles"]) for p in puzzle["train"] + puzzle["test"]}) > 1
              for puzzle in puzzles) >= 10, "visible roles never change within puzzles")
    for counts in ((2, 1), (10, 1), (10, 4), (5, 4)):
        a = generate_puzzle(185, *counts)
        _need(_same(a, generate_puzzle(185, *counts)), "whole-puzzle determinism")
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    _switch_counterfactual()
    # Grammar is invariant to whole-grid rotations, reflections and palette permutations.
    pair = puzzles[0]["train"][0]
    lut = np.random.default_rng(822).permutation(10).astype(np.int8)
    for rotation in range(4):
        for reflected in (False, True):
            inp = np.rot90(pair["input"], rotation)
            out = np.rot90(pair["output"], rotation)
            if reflected:
                inp, out = inp[:, ::-1], out[:, ::-1]
            _need(np.array_equal(solve(lut[inp]), lut[out]), "geometry/palette equivariance")
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("nonterminal corruption accepted")
    bad = puzzles[0]["train"][1]["input"].copy()
    info = _read(bad)
    a, b, label, pa, pb = next(data for data in _route_data(info)
                              if all(info["sockets"][p]["kind"] != "blank" for p in data[:2]))
    r, c = info["sockets"][b]["center"]
    other_label = next(v for v in range(10) if v not in info["guides"]
                       and v not in (info["background"], info["carrier"], label))
    bad[r - 1:r + 2, c - 1:c + 2] = np.where(CHECKER if pb else DIAMOND,
                                            other_label, info["background"])
    try:
        solve(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting endpoint clues accepted")
    summary = dict(seeds=42, pairs=len(records), distinct_inputs=len(signatures),
                   normalized_structures=len(structures), shapes=len(shapes), palettes=len(palettes),
                   node_counts=sorted({r["nodes"] for r in records}),
                   route_counts=sorted({r["routes"] for r in records}),
                   guide_counts=sorted({r["guides"] for r in records}),
                   concave_pairs=sum(r["concave"] for r in records),
                   A_sensitive_pairs=sum(r["A_effect"] for r in records),
                   max_junctions_on_route=max(v for r in records for v in r["visits"]),
                   semantic_verification="generator-owned checks only")
    return summary


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
