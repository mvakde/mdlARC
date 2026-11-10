"""Compass programs on damaged rails: an input-only ARC task family.

Symbols are distributed over the drawing. A triangle supplies a preferred exit
and the next stroke's color; repaired connectivity supplies the available exits.
No parent generator, learned dictionary, seed, or stored puzzle enters solve().
"""

from functools import lru_cache
from itertools import combinations
import numpy as np


# Clockwise order: up, right, down, left. Filled triangles point at their tip.
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
UP = np.array([[0, 1, 0], [1, 1, 1], [1, 1, 1]], dtype=bool)
ARROWS = tuple(np.rot90(UP, -d) for d in range(4))


def _integer(name, value, lo, hi=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < lo or (hi is not None and value > hi):
        raise ValueError(f"{name} is out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _components(mask):
    cells = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while cells:
        start = min(cells)
        cells.remove(start)
        stack, comp = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = r + dr, c + dc
                    if p in cells:
                        cells.remove(p)
                        stack.append(p)
                        comp.append(p)
        result.append(comp)
    return result


def _at(mask, p):
    r, c = p
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])


def _orthogonal_support(mask):
    keep = np.zeros_like(mask)
    links = mask[:-1] & mask[1:]
    keep[:-1] |= links
    keep[1:] |= links
    links = mask[:, :-1] & mask[:, 1:]
    keep[:, :-1] |= links
    keep[:, 1:] |= links
    return keep


def _repair_rails(mask):
    """Use original evidence simultaneously; never bootstrap new repairs.

    This family's retained rail portions have length >=2. Off-axis evidence and
    clutter have no orthogonal neighbors, so their removal is unambiguous.
    """
    keep = _orthogonal_support(mask)
    clean = keep.copy()
    repairs = []
    for dr, dc in ((0, 1), (1, 0)):
        for r, c in np.argwhere(keep):
            r, c = int(r), int(c)
            end = r + 4 * dr, c + 4 * dc
            gap = [(r + k * dr, c + k * dc) for k in (1, 2, 3)]
            if not _at(keep, end) or any(_at(mask, p) for p in gap):
                continue
            if not (_at(keep, (r - dr, c - dc))
                    or _at(keep, (r + 5 * dr, c + 5 * dc))):
                continue
            matches = []
            for endpoint in (0, 4):
                ar, ac = r + endpoint * dr, c + endpoint * dc
                forward = 1 if endpoint == 0 else -1
                for side in (-1, 1):
                    dy, dx = forward * dr + side * dc, forward * dc + side * dr
                    spur = [(ar + k * dy, ac + k * dx) for k in (1, 2)]
                    if (all(_at(mask, p) and not _at(keep, p) for p in spur)
                            and not _at(mask, (ar + 3 * dy, ac + 3 * dx))):
                        matches.append(tuple(spur))
            if matches:
                if len(matches) != 1:
                    raise ValueError("ambiguous gap evidence")
                for p in gap:
                    clean[p] = True
                repairs.append((tuple(gap), matches[0]))
    return clean, repairs


def _longest_run(mask):
    best = 0
    for m in (mask, mask.T):
        for row in m:
            n = 0
            for cell in row:
                n = n + 1 if cell else 0
                best = max(best, n)
    return best


def _direction(a, b):
    dy, dx = b[0] - a[0], b[1] - a[1]
    if dy and dx or not (dy or dx):
        raise ValueError("rail endpoints must be aligned")
    return DIRS.index((int(np.sign(dy)), int(np.sign(dx))))


def _segment(a, b):
    dr, dc = DIRS[_direction(a, b)]
    return [(a[0] + k * dr, a[1] + k * dc)
            for k in range(abs(b[0] - a[0]) + abs(b[1] - a[1]) + 1)]


def _choose(preferred, exits, clockwise=True):
    for offset in range(4):
        d = (preferred + (offset if clockwise else -offset)) % 4
        if d in exits:
            return exits[d]
    raise ValueError("no forward exit")


def _read_scene(inp, mend=True):
    colors, counts = np.unique(inp, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    if np.count_nonzero(inp == bg) <= inp.size // 2:
        raise ValueError("background must be a strict majority")
    wires = [int(c) for c in colors if c != bg and _longest_run(inp == c) >= 4]
    if len(wires) != 1:
        raise ValueError("rail color is not uniquely visible")
    wire = wires[0]
    rest = [(int(c), int(n)) for c, n in zip(colors, counts) if c not in (bg, wire)]
    starts = [c for c, n in rest if n == 1]
    ends = [c for c, n in rest if 2 <= n <= 4]
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("start and terminal roles are not uniquely visible")
    start_color, terminal_color = starts[0], ends[0]
    nodes = {}
    start = tuple(map(int, np.argwhere(inp == start_color)[0]))
    nodes[start] = ("start", None, start_color)
    for p in np.argwhere(inp == terminal_color):
        nodes[tuple(map(int, p))] = ("terminal", None, terminal_color)
    for color, _ in rest:
        if color in (start_color, terminal_color):
            continue
        for comp in _components(inp == color):
            rs, cs = zip(*comp)
            top, left = min(rs), min(cs)
            if max(rs) - top != 2 or max(cs) - left != 2 or len(comp) != 7:
                raise ValueError("invalid triangle")
            mask = inp[top:top + 3, left:left + 3] == color
            found = [d for d, arrow in enumerate(ARROWS) if np.array_equal(mask, arrow)]
            if len(found) != 1:
                raise ValueError("unknown triangle")
            nodes[(top + 1, left + 1)] = ("arrow", found[0], color)
    clean, repairs = _repair_rails(inp == wire)
    if not mend:
        clean, repairs = _orthogonal_support(inp == wire), []
    graph = {p: {} for p in nodes}
    potential = {p: {} for p in nodes}
    for p in nodes:
        for d in range(4):
            aligned = [q for q in nodes if q != p and
                       (q[0] == p[0] or q[1] == p[1]) and _direction(p, q) == d]
            if not aligned:
                continue
            q = min(aligned, key=lambda t: abs(t[0] - p[0]) + abs(t[1] - p[1]))
            cells = _segment(p, q)[1:-1]
            rail = [x for x in cells if not any(
                nodes[y][0] == "arrow" and max(abs(x[0] - y[0]), abs(x[1] - y[1])) <= 1
                for y in (p, q))]
            if rail and all(_at(clean, x) for x in rail):
                graph[p][d] = q
            # A visible broken rail distinguishes rejection from absent geometry.
            if rail and sum(_at(clean, x) for x in rail) >= len(rail) - 3:
                potential[p][d] = q
    return {"background": bg, "wire": wire, "start_color": start_color,
            "terminal_color": terminal_color, "start": start, "nodes": nodes,
            "graph": graph, "potential": potential, "clean": clean, "repairs": repairs}


def _walk(nodes, graph, start, ignore_arrows=False, clockwise=True):
    if len(graph[start]) != 1:
        raise ValueError("start must have exactly one connected exit")
    current = next(iter(graph[start].values()))
    route, previous, trace = [start, current], start, []
    for _ in range(len(nodes)):
        kind, preferred, color = nodes[current]
        if kind == "terminal":
            return route, trace
        if kind != "arrow":
            raise ValueError("route reaches its start again")
        exits = {d: q for d, q in graph[current].items() if q != previous}
        command = _direction(previous, current) if ignore_arrows else preferred
        nxt = _choose(command, exits, clockwise)
        trace.append((current, previous, nxt, preferred, color, exits))
        if nxt in route:
            raise ValueError("program forms a cycle")
        route.append(nxt)
        previous, current = current, nxt
    raise ValueError("program does not terminate")


def _paint(shape, bg, wire, nodes, route):
    out = np.full(shape, bg, dtype=np.int8)
    out[route[0]] = nodes[route[0]][2]
    for i, (a, b) in enumerate(zip(route, route[1:])):
        color = wire if i == 0 else nodes[a][2]
        for p in _segment(a, b)[1:]:
            out[p] = color
    out[route[-1]] = nodes[route[-1]][2]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    inp = _grid(input_grid)
    scene = _read_scene(inp)
    route, _ = _walk(scene["nodes"], scene["graph"], scene["start"])
    return _paint(inp.shape, scene["background"], scene["wire"], scene["nodes"], route)


@lru_cache(maxsize=3)
def _catalog(rows, cols):
    """Finite enumeration of small rail trees and instructive leaf-to-leaf routes.

    Records contain topology only, never a solved grid or a palette. The solver
    does not call this construction helper.
    """
    points = tuple((r, c) for r in range(rows) for c in range(cols))
    edges = tuple((p, (p[0] + dr, p[1] + dc)) for p in points
                  for dr, dc in ((0, 1), (1, 0)) if (p[0] + dr, p[1] + dc) in points)
    records = []
    for tree in combinations(edges, len(points) - 1):
        graph = {p: {} for p in points}
        for a, b in tree:
            graph[a][_direction(a, b)] = b
            graph[b][_direction(b, a)] = a
        reached, todo = set(), [points[0]]
        while todo:
            p = todo.pop()
            if p not in reached:
                reached.add(p)
                todo.extend(graph[p].values())
        if len(reached) != len(points):
            continue
        leaves = [p for p in points if len(graph[p]) == 1]
        if not 3 <= len(leaves) <= 5:
            continue
        blocked = [e for e in edges if e not in tree]
        for start in leaves:
            for end in leaves:
                if start == end:
                    continue
                todo = [(start, [start])]
                while todo:
                    p, route = todo.pop()
                    if p == end:
                        break
                    todo.extend((q, route + [q]) for q in graph[p].values() if q not in route)
                length = len(route) - 2
                if length not in (3, 4, 5, 6):
                    continue
                a_witnesses, b_witnesses = [], []
                for i in range(1, len(route) - 1):
                    p, prev, nxt = route[i], route[i - 1], route[i + 1]
                    exits = {d: q for d, q in graph[p].items() if q != prev}
                    if len(exits) >= 2 and _choose(_direction(prev, p), exits) != nxt:
                        a_witnesses.append(i)
                    for a, b in blocked:
                        if p in (a, b):
                            q = b if p == a else a
                            d = _direction(p, q)
                            if _choose(d, exits) == nxt:
                                b_witnesses.append((i, d))
                # Separate the direct branch decision from the blocked preference.
                witnesses = [(a, b, d) for a in a_witnesses for b, d in b_witnesses if a != b]
                if witnesses:
                    records.append((tree, tuple(route), tuple(leaves), tuple(witnesses), length))
    return points, edges, tuple(records)


def _canonical_pair(rng, roles, regime, strong_priority=False):
    dims = (3, 3) if strong_priority or regime == "test" or rng.random() < .72 else (2, 3)
    points, edges, records = _catalog(*dims)
    pool = [x for x in records if x[4] in ((5, 6) if regime == "test" else (3, 4))]
    if strong_priority:
        pool = [x for x in pool if any(sum(x[1][b] in e for e in x[0]) >= 3
                                      for a, b, d in x[3])]
    record = pool[int(rng.integers(len(pool)))]
    tree, path, leaves, witnesses, _ = record
    if strong_priority:
        witnesses = [w for w in witnesses if sum(path[w[1]] in e for e in tree) >= 3]
    a_step, b_step, blocked_direction = witnesses[int(rng.integers(len(witnesses)))]
    ys, xs = [2], [2]
    for _ in range(dims[0] - 1):
        ys.append(ys[-1] + int(rng.integers(10, 13)))
    for _ in range(dims[1] - 1):
        xs.append(xs[-1] + int(rng.integers(10, 13)))
    position = {p: (ys[p[0]], xs[p[1]]) for p in points}
    bg, wire, start_color, end_color, inks = roles
    shape = ys[-1] + 3, xs[-1] + 3
    inp = np.full(shape, bg, np.int8)
    nodes, graph = {}, {position[p]: {} for p in points}
    for a, b in tree:
        pa, pb = position[a], position[b]
        graph[pa][_direction(pa, pb)] = pb
        graph[pb][_direction(pb, pa)] = pa
    for p in points:
        center = position[p]
        if p == path[0]:
            nodes[center] = ("start", None, start_color)
        elif p in leaves:
            nodes[center] = ("terminal", None, end_color)
        else:
            preferred = int(rng.integers(4))
            if p in path:
                i = path.index(p)
                prev, nxt = position[path[i - 1]], position[path[i + 1]]
                exits = {d: q for d, q in graph[center].items() if q != prev}
                possible = [d for d in range(4) if _choose(d, exits) == nxt]
                preferred = int(rng.choice(possible))
                if i == a_step:
                    preferred = _direction(center, nxt)
                if i == b_step:
                    preferred = blocked_direction
            color = int(inks[len(nodes) % len(inks)])
            nodes[center] = ("arrow", preferred, color)
    rail_cells = {}
    for edge in edges:
        a, b = (position[p] for p in edge)
        cells = _segment(a, b)[1:-1]
        cells = [p for p in cells if not any(nodes[q][0] == "arrow" and
                 max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= 1 for q in (a, b))]
        rail_cells[edge] = cells
        for p in cells:
            inp[p] = wire
    for p, (kind, preferred, color) in nodes.items():
        if kind == "arrow":
            block = inp[p[0] - 1:p[0] + 2, p[1] - 1:p[1] + 2]
            block[ARROWS[preferred]] = color
        else:
            inp[p] = color
    desired = inp == wire
    # Break the preferred outgoing rail at the A witness. Protect the entire
    # alternative route: an interpreter that omits mending gets a concrete wrong
    # terminal and colored path, rather than merely failing on a disconnected rail.
    forced_key = frozenset((path[a_step], path[a_step + 1]))
    forced = next(e for e in tree if frozenset(e) == forced_key)
    unmended_graph = {p: dict(exits) for p, exits in graph.items()}
    pa, pb = position[path[a_step]], position[path[a_step + 1]]
    del unmended_graph[pa][_direction(pa, pb)]
    del unmended_graph[pb][_direction(pb, pa)]
    alternate, _ = _walk(nodes, unmended_graph, position[path[0]])
    protected = {frozenset((a, b)) for a, b in zip(alternate, alternate[1:])}
    damaged = {forced}
    for e in tree:
        if frozenset((position[e[0]], position[e[1]])) not in protected and rng.random() < .42:
            damaged.add(e)
    blocked = [e for e in edges if e not in tree]
    for edge in edges:
        if edge not in damaged and edge not in blocked:
            continue
        cells = rail_cells[edge]
        offset = int(rng.integers(2, len(cells) - 4))
        gap = cells[offset:offset + 3]
        for p in gap:
            inp[p] = bg
            if edge in blocked:
                desired[p] = False
        if edge in damaged:
            anchor_index = offset - 1 if rng.integers(2) else offset + 3
            anchor = cells[anchor_index]
            toward = 1 if anchor_index < offset else -1
            dr, dc = DIRS[_direction(cells[0], cells[-1])]
            side = int(rng.choice((-1, 1)))
            for k in (1, 2):
                inp[anchor[0] + k * (toward * dr + side * dc),
                    anchor[1] + k * (toward * dc + side * dr)] = wire
    # A near miss extrapolates to a missing cell instead of a retained anchor.
    for edge in rng.permutation(len(blocked)):
        cells = rail_cells[blocked[int(edge)]]
        missing = [i for i, p in enumerate(cells) if inp[p] == bg]
        dr, dc = DIRS[_direction(cells[0], cells[-1])]
        ghost = cells[missing[0]]
        for side in rng.permutation((-1, 1)):
            trial = inp.copy()
            for k in (1, 2):
                trial[ghost[0] + k * (dr + int(side) * dc),
                      ghost[1] + k * (dc + int(side) * dr)] = wire
            if np.array_equal(_repair_rails(trial == wire)[0], desired):
                inp = trial
                break
        else:
            continue
        break
    # Additional detached diagonal fragments occupy genuinely empty space.
    for _ in range(int(rng.integers(3))):
        length = int(rng.choice((2, 3)))
        candidates = []
        for r in range(shape[0]):
            for c in range(shape[1]):
                for dc in (-1, 1):
                    cells = [(r + k, c + k * dc) for k in range(length)]
                    if all(0 <= y < shape[0] and 0 <= x < shape[1] and
                           np.all(inp[max(0, y - 1):y + 2, max(0, x - 1):x + 2] == bg)
                           for y, x in cells):
                        candidates.append(cells)
        if candidates:
            for p in candidates[int(rng.integers(len(candidates)))]:
                inp[p] = wire
    if not np.array_equal(_repair_rails(inp == wire)[0], desired):
        raise AssertionError("construction introduced an accidental rail interaction")
    route = [position[p] for p in path]
    out = _paint(shape, bg, wire, nodes, route)
    # Tight crop followed by variable margins; every occupied input cell survives.
    rr, cc = np.nonzero(inp != bg)
    box = np.s_[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    inp, out = inp[box], out[box]
    pads = []
    for size in inp.shape:
        total = int(rng.integers(0, min(5, 30 - size) + 1))
        first = int(rng.integers(total + 1))
        pads.append((first, total - first))
    inp, out = np.pad(inp, pads, constant_values=bg), np.pad(out, pads, constant_values=bg)
    turns = int(rng.integers(4))
    return {"input": np.rot90(inp, turns).copy(), "output": np.rot90(out, turns).copy()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer("seed", seed, 0)
    nt = _integer("num_train", num_train, 2, 10)
    nq = _integer("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    fixed_roles = tuple(map(int, rng.permutation(10)))
    varied = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            palette = tuple(map(int, rng.permutation(10))) if varied else fixed_roles
            ink_count = 3 if split == "train" and index == 0 else int(rng.integers(2, 6))
            roles = (*palette[:4], tuple(palette[4:4 + ink_count]))
            pair = _canonical_pair(rng, roles, split, split == "train" and index == 0)
            # Ninety distinct observable background/rail assignments exceed the
            # maximum fourteen inputs, so this finite fallback cannot exhaust.
            if _key(pair["input"]) in seen:
                original_bg, original_wire = roles[:2]
                original_order = [c for c in range(10) if c not in (original_bg, original_wire)]
                found = None
                for bg in range(10):
                    for wire in range(10):
                        if wire == bg:
                            continue
                        palette_order = [c for c in range(10) if c not in (bg, wire)]
                        lut = np.zeros(10, np.int8)
                        lut[original_bg], lut[original_wire] = bg, wire
                        lut[original_order] = palette_order
                        candidate = {k: lut[v] for k, v in pair.items()}
                        if _key(candidate["input"]) not in seen:
                            found = candidate
                            break
                    if found is not None:
                        pair = found
                        break
                else:
                    raise AssertionError("exhausted distinct input construction")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    scene = _read_scene(inp)
    nodes, graph = scene["nodes"], scene["graph"]
    # Connected rails form a tree; deliberate gaps do not add graph edges.
    reached, todo = set(), [scene["start"]]
    while todo:
        p = todo.pop()
        if p not in reached:
            reached.add(p)
            todo.extend(graph[p].values())
    assert reached == set(nodes)
    assert sum(map(len, graph.values())) == 2 * (len(nodes) - 1)
    assert all(len(graph[p]) == 1 for p, node in nodes.items() if node[0] != "arrow")
    route, trace = _walk(nodes, graph, scene["start"])
    expected = solve(inp)
    assert out.shape == inp.shape and np.array_equal(out, expected), "incorrect full output"
    assert 3 <= len(trace) <= 6
    assert 6 <= len(nodes) <= 9
    assert len(scene["repairs"]) >= 1
    # Account for every surviving rail pixel, including deliberately broken rails.
    # This rejects stray intact linework and noncontiguous three-cell damage.
    accounted = np.zeros(inp.shape, bool)
    seen_edges = set()
    for p in nodes:
        for q in scene["potential"][p].values():
            edge = frozenset((p, q))
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            cells = [x for x in _segment(p, q)[1:-1] if not any(
                nodes[y][0] == "arrow" and max(abs(x[0] - y[0]), abs(x[1] - y[1])) <= 1
                for y in (p, q))]
            missing = [i for i, x in enumerate(cells) if not scene["clean"][x]]
            assert not missing or (len(missing) == 3 and missing == list(range(missing[0], missing[0] + 3)))
            if missing:
                assert missing[0] >= 2 and len(cells) - missing[-1] - 1 >= 2
            for x in cells:
                if scene["clean"][x]:
                    accounted[x] = True
    assert np.array_equal(accounted, scene["clean"]), "unaccounted rail geometry"
    removed = (inp == scene["wire"]) & ~scene["clean"]
    for comp in _components(removed):
        comp = sorted(comp)
        assert len(comp) in (2, 3)
        assert all(comp[i + 1][0] - comp[i][0] == 1 for i in range(len(comp) - 1))
        shifts = {comp[i + 1][1] - comp[i][1] for i in range(len(comp) - 1)}
        assert len(shifts) == 1 and abs(next(iter(shifts))) == 1
    assert not (np.logical_and.reduce((out[:-1, :-1] != scene["background"],
                                       out[1:, :-1] != scene["background"],
                                       out[:-1, 1:] != scene["background"],
                                       out[1:, 1:] != scene["background"]))).any()
    selected = set(p for a, b in zip(route, route[1:]) for p in _segment(a, b))
    assert any(set(gap) <= selected for gap, _ in scene["repairs"])
    a_witness = any(len(exits) >= 2 and preferred == _direction(p, nxt) and
                    _choose(_direction(prev, p), exits) != nxt
                    for p, prev, nxt, preferred, _, exits in trace)
    b_witness = any(preferred in scene["potential"][p] and preferred not in graph[p]
                    for p, _, _, preferred, _, _ in trace)
    priority_witness = any(preferred in scene["potential"][p] and preferred not in graph[p]
                           and len(exits) >= 2
                           for p, _, _, preferred, _, exits in trace)
    assert a_witness and b_witness, "missing interaction demonstrations"
    if priority_witness:
        reverse_route, _ = _walk(nodes, graph, scene["start"], clockwise=False)
        assert reverse_route[-1] != route[-1], "counterclockwise priority was not distinguished"
    blind_route, _ = _walk(nodes, graph, scene["start"], ignore_arrows=True)
    assert blind_route != route, "straight-first shortcut was not distinguished"
    unmended = _read_scene(inp, mend=False)
    wrong_route, _ = _walk(unmended["nodes"], unmended["graph"], unmended["start"])
    assert wrong_route[-1] != route[-1], "omitting B did not select a different terminal"
    assert not np.array_equal(_paint(inp.shape, scene["background"], scene["wire"], nodes,
                                    wrong_route), out)
    return {"shape": inp.shape, "roles": (scene["background"], scene["wire"],
            scene["start_color"], scene["terminal_color"]), "nodes": len(nodes),
            "commands": len(trace), "repairs": len(scene["repairs"]),
            "clutter": len(_components(removed)) - len(scene["repairs"]),
            "priority_witness": priority_witness,
            "blocked": sum(len(scene["potential"][p]) - len(graph[p]) for p in nodes) // 2,
            "degrees": tuple(sorted(map(len, graph.values()))),
            "arrow_directions": tuple(sorted({n[1] for n in nodes.values() if n[0] == "arrow"})),
            "stroke_directions": tuple(_direction(a, b) for a, b in zip(route, route[1:]))}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    infos, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp = _grid(pair["input"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            infos[split].append(_analyze(pair))
    assert all(x["commands"] in (3, 4) for x in infos["train"])
    assert all(x["commands"] in (5, 6) for x in infos["test"])
    assert infos["train"][0]["priority_witness"], "clockwise versus counterclockwise is not exposed"
    assert min(x["commands"] for x in infos["test"]) > max(x["commands"] for x in infos["train"])


def self_test():
    puzzles, infos = [], []
    for seed in range(42):
        p = generate_puzzle(seed)
        puzzles.append(p)
        infos.extend(_analyze(pair) for s in ("train", "test") for pair in p[s])
    signatures = {tuple(_key(pair["input"]) for s in ("train", "test") for pair in p[s])
                  for p in puzzles}
    assert len(signatures) == 42
    assert len({i["shape"] for i in infos}) >= 40
    assert {i["nodes"] for i in infos} == {6, 9}
    assert {i["commands"] for i in infos} == {3, 4, 5, 6}
    assert len({i["degrees"] for i in infos}) >= 4
    assert len({i["roles"] for i in infos}) >= 30
    assert max(i["repairs"] for i in infos) >= 4
    assert {d for i in infos for d in i["arrow_directions"]} == set(range(4))
    assert {d for i in infos for d in i["stroke_directions"]} == set(range(4))
    # Determinism, count boundaries, invalid argument types, and a complete-label
    # regression: dropping a necessary route pixel must fail full validation.
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        a, b = generate_puzzle(923, nt, nq), generate_puzzle(923, nt, nq)
        assert all(np.array_equal(x[k], y[k]) for s in ("train", "test")
                   for x, y in zip(a[s], b[s]) for k in ("input", "output"))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
         for s in ("train", "test")}
    pair = p["test"][0]
    scene = _read_scene(pair["input"])
    route, trace = _walk(scene["nodes"], scene["graph"], scene["start"])
    gap = next(gap for gap, _ in scene["repairs"]
               if set(gap) <= {x for a, b in zip(route, route[1:]) for x in _segment(a, b)})
    pair["output"][gap[0]] = scene["background"]
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("omitted route repair accepted")
    # Change one branch command while retaining exactly the same damaged rails.
    pair = puzzles[0]["train"][0]
    scene = _read_scene(pair["input"])
    route, trace = _walk(scene["nodes"], scene["graph"], scene["start"])
    p, prev, nxt, preferred, color, exits = next(t for t in trace if len(t[5]) >= 2)
    other = next(d for d, q in exits.items() if q != nxt)
    altered = pair["input"].copy()
    block = altered[p[0] - 1:p[0] + 2, p[1] - 1:p[1] + 2]
    block[:] = scene["background"]
    block[ARROWS[other]] = color
    assert not np.array_equal(solve(altered), pair["output"])
    # A B-only change: restore a blocked preferred rail's three missing cells.
    p, _, _, preferred, _, _ = next(t for t in trace if t[3] in scene["potential"][t[0]]
                                  and t[3] not in scene["graph"][t[0]])
    altered = pair["input"].copy()
    q = scene["potential"][p][preferred]
    for x in _segment(p, q)[1:-1]:
        if altered[x] == scene["background"] and max(abs(x[0] - p[0]), abs(x[1] - p[1])) > 1 \
                and max(abs(x[0] - q[0]), abs(x[1] - q[1])) > 1:
            altered[x] = scene["wire"]
    # Adding an edge can create a cycle outside the family. A changed result OR
    # explicit cycle detection refutes interpreting every break as a connection.
    try:
        prediction = solve(altered)
    except ValueError:
        pass
    else:
        assert not np.array_equal(prediction, pair["output"])


if __name__ == "__main__":
    self_test()
