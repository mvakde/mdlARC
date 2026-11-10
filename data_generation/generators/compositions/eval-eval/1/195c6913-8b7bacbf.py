"""Clocked reservoir networks: wire travel and enclosed area advance a cycle.

Only standard-library modules and NumPy are required. No parent generator is
imported, and solve has no access to generation parameters or examples.
"""
from collections import deque
import json

import numpy as np


D4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0, 818357883, 552711415, 661484107, 408253045, 510887407,
              131174872, 406599260, 399969203, 776604620, 570888910,
              411998001, 798059147, 608753636, 74032126, 649682769,
              155317667, 918295392, 80185497, 716886140, 673900335,
              652254820, 539394682, 3550586, 618115567, 136168818,
              847697884, 808148056, 548273467, 933330203, 787891985,
              50573496, 434662190, 577127652, 903358226, 311230184,
              851483033, 68330968, 893370694, 201332408, 421328479, 59981878)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    _require(_integer(num_train) and _integer(num_test), "counts must be integers")
    _require(2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test,
             "require 2..10 training, 1..4 test, and training > test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must have integer cells")
    _require(bool(np.all((a >= 0) & (a <= 9))), "bad colors")
    return a.astype(np.int8, copy=True)


def _neighbors(p):
    r, c = p
    return ((r-1, c), (r, c+1), (r+1, c), (r, c-1))


def _rim(cells):
    return {q for p in cells for q in _neighbors(p)} - set(cells)


def _components(cells):
    left = set(cells)
    groups = []
    while left:
        root = min(left)
        left.remove(root)
        group = {root}
        todo = [root]
        for p in todo:
            for q in _neighbors(p):
                if q in left:
                    left.remove(q)
                    group.add(q)
                    todo.append(q)
        groups.append(group)
    return groups


def _cells(a, color):
    return {tuple(map(int, p)) for p in np.argwhere(a == color)}


def _graph(rooms, wires, sources):
    """A room is one node; each wire pixel and each source is its own node.

    Shell/shell contact is deliberately absent from this graph. Several shell
    pixels touching the SAME wire pixel still form just one port/edge.
    """
    nodes = [("room", i) for i in range(len(rooms))]
    nodes += [("wire", p) for p in sorted(wires)]
    nodes += [("source", p) for p in sorted(sources)]
    graph = {node: set() for node in nodes}
    for p in wires:
        u = ("wire", p)
        for q in _neighbors(p):
            v = ("wire", q) if q in wires else ("source", q)
            if v in graph:
                graph[u].add(v)
                graph[v].add(u)
    for i, room in enumerate(rooms):
        for p in _rim(room["shell"]) & wires:
            u, v = ("room", i), ("wire", p)
            graph[u].add(v)
            graph[v].add(u)
    return graph


def _interpret(input_grid):
    a = _grid(input_grid)
    # Exterior background occupies more than half the input, an explicit
    # family constraint rather than a general ARC background heuristic.
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    _require(counts[bg] > a.size // 2, "background must be a strict majority")
    swatches = []
    for color in map(int, np.unique(a)):
        if color == bg:
            continue
        for comp in _components(_cells(a, color)):
            rows = sorted({r for r, c in comp})
            cols = sorted({c for r, c in comp})
            if len(comp) == 4 and len(rows) == len(cols) == 2:
                if rows[1] == rows[0] + 1 and cols[1] == cols[0] + 1:
                    swatches.append(dict(cells=comp, row=rows[0], col=cols[0], color=color))
    _require(2 <= len(swatches) <= 6, "need two to six legend entries")
    swatches.sort(key=lambda s: s["col"])
    _require(len({s["row"] for s in swatches}) == 1 and swatches[0]["row"] <= 2,
             "legend must be horizontal at the top")
    _require(all(t["col"] - s["col"] >= 3 for s, t in zip(swatches, swatches[1:])),
             "legend swatches must be separated")
    word = tuple(s["color"] for s in swatches)
    _require(len(set(word)) >= 2, "cycle needs at least two distinct colors")
    annotations = set().union(*(s["cells"] for s in swatches))
    for p in _rim(annotations):
        if 0 <= p[0] < a.shape[0] and 0 <= p[1] < a.shape[1]:
            _require(a[p] == bg, "swatches must have background surroundings")
    sources = _cells(a, word[0]) - annotations
    _require(1 <= len(sources) <= 3, "need one to three first-color singleton sources")
    for color in set(word) - {word[0]}:
        _require(_cells(a, color) <= annotations, "other program colors occur only in legend")
    _require(all(not (_neighbors(p)[i] in sources) for p in sources for i in range(4)),
             "sources must be isolated singletons")
    structural = set(map(int, np.unique(a))) - set(word) - {bg}
    _require(len(structural) == 2, "need one wire and one shell color")
    blank = a.copy()
    for p in annotations:
        blank[p] = bg
    cavities = [g for g in _components(_cells(blank, bg))
                if not any(r in (0, a.shape[0]-1) or c in (0, a.shape[1]-1) for r, c in g)]
    rooms = []
    for inner in cavities:
        shell = _rim(inner)
        colors = {int(blank[p]) for p in shell}
        _require(len(colors) == 1 and colors <= structural, "cavity needs a monochrome shell")
        rooms.append(dict(inner=inner, shell=shell, color=colors.pop()))
    _require(rooms, "need at least one enclosed reservoir")
    shell_colors = {r["color"] for r in rooms}
    _require(len(shell_colors) == 1, "all shells share a role color")
    shell_color = shell_colors.pop()
    wire_color = (structural - {shell_color}).pop()
    all_shell = set().union(*(r["shell"] for r in rooms))
    _require(_cells(a, shell_color) == all_shell, "shell cells must bound known cavities")
    _require(sum(len(r["inner"]) for r in rooms) == len(set().union(*(r["inner"] for r in rooms))),
             "cavities must be disjoint")
    wires = _cells(a, wire_color)
    graph = _graph(rooms, wires, sources)
    for p in sources:
        _require(len(graph[("source", p)]) == 1, "source must cap exactly one wire cell")
        _require(not (_rim({p}) & all_shell), "source cannot touch a shell")
    # A forest makes propagation and all branch phases unique.
    seen = set()
    groups = []
    for root in graph:
        if root in seen:
            continue
        group = {root}
        todo = [root]
        for u in todo:
            for v in graph[u] - group:
                group.add(v)
                todo.append(v)
        seen |= group
        _require(sum(len(graph[u]) for u in group) == 2 * (len(group) - 1), "network must be a tree")
        _require(sum(u[0] == "source" for u in group) <= 1, "a component cannot have two sources")
        groups.append(group)
    return dict(grid=a, bg=bg, word=word, annotations=annotations, rooms=rooms,
                wires=wires, sources=sources, graph=graph, groups=groups,
                wire_color=wire_color, shell_color=shell_color)


def _propagate(graph, rooms, sources, word, mode="correct"):
    """Return room colors and input-derived event records.

    Wrong modes are explicit ablations used only to audit demonstrations.
    The correct transition needs neither canvas orientation nor object order.
    """
    if mode == "deduplicate":
        word = tuple(dict.fromkeys(word))
    elif mode == "reverse":
        word = tuple(reversed(word))
    fills, events = {}, []
    global_offset = 0
    for source in sorted(sources):
        root = ("source", source)
        todo = [(root, None, global_offset if mode == "global_restart" else 0)]
        local_events = []
        serial_clock = 0
        visited = set()
        while todo:
            u, parent, incoming = todo.pop()
            _require(u not in visited, "propagation encountered a cycle")
            visited.add(u)
            kind, value = u
            time = serial_clock if mode == "serial_branches" else incoming
            if kind == "wire":
                time += 0 if mode == "no_wire" else 1
                if mode == "bend_reset" and parent is not None and parent[0] == "wire":
                    children = [v for v in graph[u] if v != parent and v[0] == "wire"]
                    before = (value[0] - parent[1][0], value[1] - parent[1][1])
                    if children and any((v[1][0]-value[0], v[1][1]-value[1]) != before for v in children):
                        time = 1
            elif kind == "room":
                fills[value] = word[0] if mode == "source_only" else word[time % len(word)]
                event = dict(room=value, arrival=time, area=len(rooms[value]["inner"]),
                             degree=len(graph[u]), color=fills[value], source=source)
                events.append(event)
                local_events.append(event)
                if mode == "no_capacity":
                    pass
                elif mode == "one_per_room":
                    time += 1
                elif mode == "perimeter":
                    time += len(rooms[value]["shell"])
                else:
                    time += len(rooms[value]["inner"])
            children = sorted(graph[u] - ({parent} if parent is not None else set()))
            serial_clock = time
            for v in reversed(children):
                todo.append((v, u, time))
        # Deliberately bad shared clock: second source inherits completed work.
        if local_events:
            global_offset += max(e["arrival"] + e["area"] for e in local_events)
    if mode == "fill_all":
        for i in range(len(rooms)):
            fills.setdefault(i, word[0])
    return fills, events


def _output(info, mode="correct"):
    out = info["grid"].copy()
    for p in info["annotations"]:
        out[p] = info["bg"]
    fills, events = _propagate(info["graph"], info["rooms"], info["sources"], info["word"], mode)
    for i, color in fills.items():
        for p in info["rooms"][i]["inner"]:
            out[p] = color
    return out, events


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the input, collapse cavities to capacitated nodes, run the clock."""
    return _output(_interpret(input_grid))[0]


ABLATIONS = ("deduplicate", "reverse", "no_wire", "no_capacity", "one_per_room",
             "perimeter", "bend_reset", "source_only", "fill_all", "global_restart",
             "serial_branches")


def _evidence(info):
    correct, events = _output(info)
    evidence = {mode for mode in ABLATIONS if not np.array_equal(correct, _output(info, mode)[0])}
    if any(e["arrival"] >= 2 * len(info["word"]) for e in events):
        evidence.add("wraparound")
    if any(e["degree"] >= 3 for e in events):
        evidence.add("branch")
    return evidence


def _randint(rng, low, high):
    return int(rng.integers(low, high + 1))


def _shape(rng, maximum, large=False):
    """Connected small polyomino cavities, including asymmetric capacities."""
    target = _randint(rng, max(1, maximum // 2) if large else 1, maximum)
    kind = _randint(rng, 0, 3)
    if kind == 0:
        h = _randint(rng, 1, min(4, target))
        w = max(1, min(5, target // h))
        inner = {(r, c) for r in range(h) for c in range(w)}
    elif kind == 1:
        inner = {(0, c) for c in range(target)} if target <= 7 else {(r, c) for r in range(2) for c in range(target // 2)}
    else:
        inner = {(0, 0)}
        for _ in range(target - 1):
            options = sorted(q for q in _rim(inner)
                             if -2 <= q[0] <= 2 and -2 <= q[1] <= 3)
            if not options:
                break
            q = options[_randint(rng, 0, len(options)-1)]
            # Avoid enclosed gaps within the cavity itself.
            trial = inner | {q}
            r0, r1 = min(r for r,c in trial), max(r for r,c in trial)
            c0, c1 = min(c for r,c in trial), max(c for r,c in trial)
            holes = {(r,c) for r in range(r0,r1+1) for c in range(c0,c1+1)} - trial
            if all(any(r in (r0,r1) or c in (c0,c1) for r,c in comp) for comp in _components(holes)):
                inner = trial
    shell = _rim(inner)
    # A minimal shell sometimes accidentally encloses a second pocket; reject
    # that geometry rather than supplying an unannounced extra reservoir.
    r0, r1 = min(r for r,c in shell), max(r for r,c in shell)
    c0, c1 = min(c for r,c in shell), max(c for r,c in shell)
    free = {(r,c) for r in range(r0-1,r1+2) for c in range(c0-1,c1+2)} - shell
    pockets = [g for g in _components(free) if not any(r in (r0-1,r1+1) or c in (c0-1,c1+1) for r,c in g)]
    if pockets != [inner]:
        inner = {(0,c) for c in range(min(target, 7))}
        shell = _rim(inner)
    r0, c0 = min(r for r,c in shell), min(c for r,c in shell)
    return { (r-r0,c-c0) for r,c in inner }, { (r-r0,c-c0) for r,c in shell }


def _route(rng, canvas, rooms, wires, sources, anchor_nodes, target):
    """Route a new induced path with exactly one existing-network attachment."""
    shell_map = {}
    blocked = set(sources)
    for i, room in enumerate(rooms):
        blocked |= room["inner"] | room["shell"]
        for p in room["shell"]:
            shell_map[p] = i
    blocked |= wires

    def contacts(p):
        result = set()
        for q in _neighbors(p):
            if q in wires:
                result.add(("wire", q))
            if q in sources:
                result.add(("source", q))
            if q in shell_map:
                result.add(("room", shell_map[q]))
        return result

    free = canvas - blocked
    contact_map = {p: contacts(p) for p in free}
    ordinary = {p for p in free if not contact_map[p]}
    starts = [p for p in sorted(free) if len(contact_map[p]) == 1 and contact_map[p] <= anchor_nodes]
    goals = {p for p in free if contact_map[p] == {("room", target)}}
    if not starts or not goals:
        return None
    rng.shuffle(starts)
    directions = [D4[int(i)] for i in rng.permutation(4)]
    allowed = ordinary | goals
    # Multi-source BFS: the first reached goal gives a shortest, hence induced,
    # wire path. Each start is an alternative port, not a simultaneous branch.
    queue = deque(starts)
    previous = {p: None for p in starts}
    end = None
    while queue:
        p = queue.popleft()
        if p in goals:
            end = p
            break
        for dr, dc in directions:
            q = (p[0] + dr, p[1] + dc)
            if q in previous or q not in allowed:
                continue
            previous[q] = p
            queue.append(q)
    if end is None:
        return None
    path = []
    while end is not None:
        path.append(end)
        end = previous[end]
    return set(path)


def _word(rng, length, repeated=False):
    d = _randint(rng, 2, min(5, length - 1 if repeated else length))
    for _ in range(64):
        word = list(range(d)) + [int(v) for v in rng.integers(0, d, length-d)]
        rng.shuffle(word)
        unique = tuple(dict.fromkeys(word))
        if not repeated or (word != word[::-1] and
                            any(x != unique[i % d] for i,x in enumerate(word))):
            return word
    return [0] * (length-d+1) + list(range(1,d))


def _compress_empty_rows(inp, out, bg):
    """Keep at most three wholly empty separator rows; preserve all objects.

    A wire cannot cross a wholly background row. Keeping three rows prevents
    new contacts when empty spatial padding is removed.
    """
    keep = []
    r = 0
    while r < inp.shape[0]:
        if not (np.all(inp[r] == bg) and np.all(out[r] == bg)):
            keep.append(r)
            r += 1
            continue
        end = r
        while end < inp.shape[0] and np.all(inp[end] == bg) and np.all(out[end] == bg):
            end += 1
        keep.extend(range(r, min(end, r + 3)))
        r = end
    a, b = inp[keep].copy(), out[keep].copy()
    return (a,b) if np.count_nonzero(a == bg) > a.size // 2 else (inp,out)


def _make_pair(rng, lesson, split, regime):
    # Lessons constrain evidence/topology, never concrete positions or shapes.
    if lesson == "serial":
        owned_counts, detached, topology = [3], 1, "serial"
        length, maximum = _randint(rng, 3, 5), 8
    elif lesson == "branch":
        owned_counts, detached, topology = [3, 1], 1, "branch"
        length, maximum = _randint(rng, 3, 5), 8
    else:
        nsource = int(rng.choice([1, 2, 3], p=[.58, .36, .06]))
        owned_counts = [_randint(rng, 1, 3) for _ in range(nsource)]
        while sum(owned_counts) > 6:
            owned_counts[int(np.argmax(owned_counts))] -= 1
        detached = _randint(rng, 0, 2)
        topology = str(rng.choice(["serial", "branch", "mixed"]))
        length, maximum = _randint(rng, 2, 5), 8
    if split == "test":
        if regime == "program":
            length = 6
        elif regime == "capacity":
            maximum = 16
        else:
            owned_counts = [4, 1] if rng.random() < .6 else [5]
            topology = "mixed"
    repeated = lesson is not None
    symbols = _word(rng, length, repeated)
    palette = list(map(int, rng.permutation(10)))
    bg, wire_color, shell_color = palette[:3]
    word = tuple(palette[3+i] for i in symbols)
    nroom = sum(owned_counts) + detached
    rejected = {}
    def reject(reason):
        rejected[reason] = rejected.get(reason, 0) + 1
    for attempt in range(240):
        if attempt and attempt % 12 == 0:
            symbols = _word(rng, length, repeated)
            word = tuple(palette[3+i] for i in symbols)
        dense = nroom >= 5 or maximum > 8
        h = _randint(rng, 23 if dense else 18, 30)
        w = _randint(rng, max(23 if dense else 18, 3*length+1), 30)
        legend_row = _randint(rng, 0, 2)
        legend_left = _randint(rng, 0, w - (3*length-1))
        canvas = {(r,c) for r in range(legend_row+5, h) for c in range(w)}
        rooms = []
        occupied = set()
        for i in range(nroom):
            inner, shell = _shape(rng, maximum, split == "test" and regime == "capacity")
            rh, rw = max(r for r,c in shell)+1, max(c for r,c in shell)+1
            placed = False
            for _ in range(100):
                rr = _randint(rng, legend_row+5, h-rh-1)
                cc = _randint(rng, 1, w-rw-1)
                ins = {(r+rr,c+cc) for r,c in inner}
                edge = {(r+rr,c+cc) for r,c in shell}
                full = ins | edge
                # A two-cell halo makes unrelated contours visually separate.
                halo = full | _rim(full)
                if halo & occupied:
                    continue
                rooms.append(dict(inner=ins, shell=edge, color=shell_color))
                occupied |= halo
                placed = True
                break
            if not placed:
                break
        if len(rooms) != nroom:
            reject("packing")
            continue
        wires, sources = set(), set()
        owners = {}
        cursor, okay = 0, True
        for owner, count in enumerate(owned_counts):
            free = sorted(canvas - occupied - wires - _rim(wires) - sources - _rim(sources))
            # Start away from all contours; an ordinary terminal singleton is
            # the fixed source convention of this new family.
            excluded = occupied | wires | sources
            candidates = [p for p in free if all(q not in excluded for q in _neighbors(p))]
            if not candidates:
                okay = False
                break
            source = candidates[_randint(rng, 0, len(candidates)-1)]
            sources.add(source)
            source_node = ("source", source)
            component_nodes = {source_node}
            root_room = cursor
            prior_room = None
            for j in range(count):
                target = cursor + j
                if j == 0:
                    anchors = {source_node}
                elif topology == "serial":
                    anchors = {("room", prior_room)}
                elif topology == "branch":
                    anchors = {("room", root_room)}
                else:
                    anchors = component_nodes - {source_node}
                path = _route(rng, canvas, rooms, wires, sources, anchors, target)
                if not path:
                    okay = False
                    break
                wires |= path
                component_nodes |= {("wire", p) for p in path} | {("room", target)}
                owners[target] = source
                prior_room = target
            cursor += count
            if not okay:
                break
        if not okay:
            reject("routing")
            continue
        # Optional open tails on inactive reservoirs distinguish supply from
        # mere attachment to a string. No tail is a paint source.
        for target in range(cursor, nroom):
            if rng.random() < .65:
                own = rooms[target]["inner"] | rooms[target]["shell"]
                unrelated = occupied - (own | _rim(own))
                available = sorted((_rim(rooms[target]["shell"]) & canvas) - own - unrelated
                                   - wires - _rim(wires) - sources - _rim(sources))
                if available:
                    wires.add(available[_randint(rng,0,len(available)-1)])
        inp = np.full((h,w), bg, dtype=np.int8)
        for room in rooms:
            for p in room["shell"]:
                inp[p] = shell_color
        for p in wires:
            inp[p] = wire_color
        for p in sources:
            inp[p] = word[0]
        # Scene symmetries vary network orientation without changing the
        # consistently horizontal, left-to-right legend convention.
        body = inp[legend_row+4:, :].copy()
        flip = _randint(rng, 0, 3)
        if flip & 1:
            body = body[:, ::-1]
        if flip & 2:
            body = body[::-1, :]
        inp[legend_row+4:, :] = body
        for i,color in enumerate(word):
            c = legend_left + 3*i
            inp[legend_row:legend_row+2, c:c+2] = color
        # Construct labels from placed geometry before inspecting the input.
        graph = _graph(rooms, wires, sources)
        try:
            fills, _ = _propagate(graph, rooms, sources, word)
        except ValueError:
            reject("cycle")
            continue
        if set(fills) != set(owners):
            reject("ownership")
            continue
        out = np.full((h,w), bg, dtype=np.int8)
        out[legend_row+4:, :] = inp[legend_row+4:, :]
        unflipped = np.full((h,w), bg, dtype=np.int8)
        for room in rooms:
            for p in room["shell"]:
                unflipped[p] = shell_color
        for p in wires:
            unflipped[p] = wire_color
        for p in sources:
            unflipped[p] = word[0]
        for i,color in fills.items():
            for p in rooms[i]["inner"]:
                unflipped[p] = color
        body = unflipped[legend_row+4:, :]
        if flip & 1:
            body = body[:, ::-1]
        if flip & 2:
            body = body[::-1, :]
        out[legend_row+4:, :] = body
        inp, out = _compress_empty_rows(inp, out, bg)
        try:
            info = _interpret(inp)
            if not np.array_equal(out, _output(info)[0]):
                reject("labels")
                continue
            if split == "test" and regime == "capacity" and not any(len(r["inner"]) > 8 for r in info["rooms"]):
                continue
            evidence = _evidence(info)
            serial_required = {"deduplicate", "reverse", "no_wire", "no_capacity", "one_per_room",
                               "perimeter", "bend_reset", "source_only", "fill_all", "wraparound"}
            if lesson == "serial" and not serial_required <= evidence:
                reject("serial:" + str(sorted(serial_required - evidence)))
                continue
            if lesson == "branch" and not {"global_restart", "branch", "serial_branches"} <= evidence:
                reject("branch:" + str(sorted({"global_restart", "branch", "serial_branches"} - evidence)))
                continue
        except ValueError as error:
            reject(str(error))
            continue
        return dict(input=inp, output=out)
    raise RuntimeError("could not pack reservoir network: " + str((lesson, split, word, rejected)))


def _key(a):
    a = np.asarray(a, dtype=np.uint8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    _require(_integer(seed) and int(seed) >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(["program", "capacity", "topology"]))
    lesson_indices = list(map(int, rng.choice(nt, size=2, replace=False)))
    lessons = {lesson_indices[0]: "serial", lesson_indices[1]: "branch"}
    result, used = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ne)):
        for index in range(count):
            for _ in range(12):
                pair = _make_pair(rng, lessons.get(index) if split == "train" else None,
                                  split, regime)
                key = _key(pair["input"])
                if key not in used:
                    break
            else:
                raise RuntimeError("could not make a distinct input")
            used.add(key)
            result[split].append(pair)
    validate_generated_puzzle(result)
    return result


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, evidence = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "shape changes are forbidden")
            _require(_key(inp) not in seen, "duplicate inputs")
            seen.add(_key(inp))
            info = _interpret(inp)
            expected, events = _output(info)
            _require(bool(events), "a source must supply a reservoir")
            _require(np.array_equal(out, expected), "incorrect clocked fill or preservation")
            _require(all(e["color"] in info["word"] for e in events), "fill outside program")
            changed = set(map(tuple, np.argwhere(inp != out)))
            inner_cells = set().union(*(r["inner"] for r in info["rooms"]))
            _require(changed <= inner_cells | info["annotations"], "non-target pixels changed")
            _require(all(out[p] == inp[p] for p in info["wires"] | info["sources"]), "route/source changed")
            _require(all(out[p] == info["bg"] for p in info["annotations"]), "legend not erased")
            if split == "train":
                evidence |= _evidence(info)
    _require(set(ABLATIONS) | {"wraparound", "branch"} <= evidence,
             "training does not distinguish required alternatives: " + str((set(ABLATIONS) | {"wraparound", "branch"}) - evidence))


def self_test():
    """Bounded audit over the 42 supplied seeds, counts, mutations and geometry."""
    puzzles = [generate_puzzle(seed, 2, 1) for seed in HOST_SEEDS]
    infos = [_interpret(q["input"]) for p in puzzles for s in p for q in p[s]]
    _require(len({_key(p["train"][0]["input"]) for p in puzzles}) == 42, "seed diversity")
    _require(len({i["grid"].shape for i in infos}) >= 35, "dimension diversity")
    _require({len(i["sources"]) for i in infos} >= {1, 2}, "source-count diversity")
    _require({len(i["word"]) for i in infos} >= {2, 3, 4, 5, 6}, "cycle-length diversity")
    _require(len({i["bg"] for i in infos}) == 10, "background diversity")
    areas = {len(r["inner"]) for i in infos for r in i["rooms"]}
    _require(len(areas) >= 10 and max(areas) >= 12, "capacity diversity")
    shapes = set()
    for i in infos:
        for room in i["rooms"]:
            r0, c0 = min(r for r,c in room["inner"]), min(c for r,c in room["inner"])
            shapes.add(tuple(sorted((r-r0,c-c0) for r,c in room["inner"])))
    _require(len(shapes) >= 35, "cavity-shape diversity")
    _require(any(any(u[0] == "wire" and len(v) >= 3 for u,v in i["graph"].items()) for i in infos),
             "wire forks absent")
    # The host verifier exhausts all legal count combinations; self_test keeps
    # only count boundaries so its extra seed/semantic audit remains bounded.
    for nt,ne in ((2,1),(10,1),(10,4)):
        p = generate_puzzle(92000+nt*10+ne,nt,ne)
        _require(len(p["train"]) == nt and len(p["test"]) == ne, "count mismatch")
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(3,1.5),(None,1)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    repeat = generate_puzzle(HOST_SEEDS[0], 2, 1)
    _require(all(_key(a[k]) == _key(b[k]) for s in repeat for a,b in zip(repeat[s],puzzles[0][s])
                 for k in ("input","output")), "nondeterminism")
    for mode in ABLATIONS:
        bad = {s: [{k:v.copy() for k,v in q.items()} for q in puzzles[0][s]] for s in puzzles[0]}
        witness = next(q for q in bad["train"] if mode in _evidence(_interpret(q["input"])))
        witness["output"] = _output(_interpret(witness["input"]),mode)[0]
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            continue
        raise AssertionError("incorrect model accepted: " + mode)
    return {"seeds":42, "pairs":len(infos), "dimensions":len({i["grid"].shape for i in infos}),
            "cavity_shapes":len(shapes), "cavity_areas":sorted(areas),
            "alternatives_rejected":list(ABLATIONS)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
