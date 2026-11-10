"""Paint chambers: simultaneous exports through geometrically polarized ports.

Standard library and NumPy only. No parent imports, external files, hidden
conventions, generated traces, or disk writes are needed by this module.
"""
from collections import deque
import hashlib

import numpy as np


D4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))
AUDIT_SEEDS = (0, 359339640, 593200491, 584882627, 524886621, 520364311,
               890987498, 536723672, 608486678, 347104456, 392765319,
               292852895, 478894338, 994986703, 113514480, 602814147,
               735261600, 654833265, 984655414, 538510218, 572527702,
               572909865, 767714241, 577110403, 366224382, 792177478,
               261061975, 243673755, 587529579, 5817452, 264355718,
               858393362, 994500755, 534246809, 283821728, 130104326,
               365773656, 273887892, 70509400, 849889341, 370406991,
               910350384)


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _need(bool(np.all((a >= 0) & (a <= 9))), "invalid grid colors")
    return a


def _add(p, d, k=1):
    return p[0] + k*d[0], p[1] + k*d[1]


def _neighbors(cells, steps=D4):
    return {_add(p, d) for p in cells for d in steps} - set(cells)


def _components(cells):
    left = set(cells)
    groups = []
    while left:
        start = min(left)
        left.remove(start)
        group, todo = {start}, [start]
        for p in todo:
            for d in D4:
                q = _add(p, d)
                if q in left:
                    left.remove(q)
                    group.add(q)
                    todo.append(q)
        groups.append(group)
    return groups


def _points(mask):
    return {(int(r), int(c)) for r, c in np.argwhere(mask)}


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _rooms(a, bg):
    """The shell role is the unique color exactly outlining closed chambers.

    Inspect bounded components of each candidate shell color, then flood their
    bounding boxes. This avoids scanning the whole exterior for every color.
    """
    answers = []
    for color in sorted(int(c) for c in np.unique(a) if c != bg):
        rooms = []
        for shell in _components(_points(a == color)):
            r0, r1 = min(r for r, c in shell), max(r for r, c in shell)
            c0, c1 = min(c for r, c in shell), max(c for r, c in shell)
            if (r1-r0-1)*(c1-c0-1) < 9 or min(r1-r0, c1-c0) < 2:
                break
            empty = {(r, c) for r in range(r0, r1+1) for c in range(c0, c1+1)} - shell
            outside = {p for p in empty if p[0] in (r0, r1) or p[1] in (c0, c1)}
            todo = list(outside)
            for p in todo:
                for d in D4:
                    q = _add(p, d)
                    if q in empty and q not in outside:
                        outside.add(q)
                        todo.append(q)
            inner = empty - outside
            if (len(inner) < 9 or len(_components(inner)) != 1 or
                    _neighbors(inner, D8) != shell or any(a[p] == color for p in inner)):
                break
            rooms.append(dict(inner=inner, shell=shell))
        else:
            if rooms:
                answers.append((color, rooms))
    _need(len(answers) == 1, "chamber shell is not uniquely identifiable")
    return answers[0]


def _ports(room):
    """Return straight ports with enough interior context for a readable T."""
    if "ports" in room:
        return room["ports"]
    inner, shell = room["inner"], room["shell"]
    result = []
    for center in sorted(inner):
        for d in D4:
            perp = (-d[1], d[0])
            arms = (_add(center, perp), _add(center, perp, -1))
            tip, wall, port = (_add(center, d, k) for k in (1, 2, 3))
            write = {center, tip, *arms}
            flanks = {_add(tip, perp), _add(tip, perp, -1)}
            if (write | flanks) <= inner and wall in shell and port not in inner | shell:
                # An exterior port contacts exactly one wall pixel.
                if _neighbors({port}) & shell == {wall}:
                    result.append(dict(center=center, d=d, arms=arms, tip=tip,
                                       wall=wall, port=port, write=write))
    room["ports"] = result
    return result


def _analyze(input_grid):
    a = _grid(input_grid)
    border = np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))
    _need(len(np.unique(border)) == 1, "background border must be uniform")
    bg = int(border[0])
    shell, rooms = _rooms(a, bg)
    occupied = set().union(*(r["inner"] | r["shell"] for r in rooms))
    exterior = _points(a != bg) - occupied
    routes = []
    for color in sorted({int(a[p]) for p in exterior}):
        for cells in _components(p for p in exterior if a[p] == color):
            attachments = {}
            for i, room in enumerate(rooms):
                contact = cells & _neighbors(room["shell"])
                if contact:
                    _need(len(contact) == 1, "wire runs along a chamber wall")
                    port = next(iter(contact))
                    _need(any(g["port"] == port for g in _ports(room)), "not a straight chamber port")
                    attachments[i] = port
            _need(len(attachments) >= 2, "a wire must attach to at least two chambers")
            routes.append(dict(cells=cells, color=color, attachments=attachments))
    _need(bool(routes), "no transport wires")
    # Marker ink is inferred from complete T geometry, never from a literal ID.
    interpretations = []
    for ink in sorted(int(c) for c in np.unique(a) if c not in (bg, shell)):
        if any(a[p] == ink for p in exterior):
            continue
        decoded, arrows, used_ink = [], [], set()
        valid = True
        for i, room in enumerate(rooms):
            colors = {int(a[p]) for p in room["inner"]}
            if len(colors) == 1:
                fill = next(iter(colors))
                if fill == ink:
                    valid = False
                    break
                decoded.append(dict(room, fill=fill, arrow=None))
                continue
            choices = []
            for g in _ports(room):
                if not all(a[p] == ink for p in (*g["arms"], g["tip"])):
                    continue
                rest = room["inner"] - g["write"]
                fills = {int(a[p]) for p in rest}
                if len(fills) != 1 or ink in fills:
                    continue
                fill = next(iter(fills))
                matches = [j for j, route in enumerate(routes)
                           if route["attachments"].get(i) == g["port"]]
                if len(matches) != 1:
                    continue
                explicit = int(a[g["center"]]) != ink
                payload = int(a[g["center"]]) if explicit else fill
                if payload in (shell, ink) or payload in {r["color"] for r in routes}:
                    continue
                choices.append(dict(g, room=i, route=matches[0], payload=payload,
                                    explicit=explicit, fill=fill))
            if len(choices) != 1:
                valid = False
                break
            arrow = choices[0]
            arrows.append(arrow)
            used_ink.update(p for p in arrow["write"] if a[p] == ink)
            decoded.append(dict(room, fill=arrow["fill"], arrow=arrow))
        if valid and arrows and used_ink == _points(a == ink):
            interpretations.append((ink, decoded, arrows))
    _need(len(interpretations) == 1, "outlet interpretation is not unique")
    ink, rooms, arrows = interpretations[0]
    route_colors = {r["color"] for r in routes}
    _need(all(r["fill"] not in route_colors | {shell, ink} for r in rooms), "ambiguous fill role")
    assignments = {}
    for j, route in enumerate(routes):
        sources = [v for v in arrows if v["route"] == j]
        _need(len(sources) <= 1, "conflicting supplies on one wire")
        source = sources[0] if sources else None
        route["source"] = source
        if source is None:
            continue
        for i in route["attachments"]:
            if i == source["room"]:
                continue
            _need(i not in assignments or assignments[i] == source["payload"], "conflicting chamber inlets")
            assignments[i] = source["payload"]
    out = a.astype(np.int8, copy=True)
    for i, room in enumerate(rooms):
        fill = assignments.get(i, room["fill"])
        for p in room["inner"]:
            out[p] = fill
    return out, dict(bg=bg, shell=shell, ink=ink, rooms=rooms, routes=routes,
                     arrows=arrows, assignments=assignments)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer from this input alone, without mutation."""
    return _analyze(input_grid)[0]


def _pick(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]


def _shape(rng, compact):
    h, w = (int(rng.integers(3, 5 if compact else 6)) for _ in range(2))
    kind = int(rng.integers(4))
    cells = {(r, c) for r in range(h) for c in range(w)}
    if kind == 1 and w >= 4:
        cells -= {(0, c) for c in range(int(rng.integers(1, w-2)))}
    elif kind == 2 and h >= 4 and w >= 4:
        cells -= {(r, 0) for r in range(int(rng.integers(1, h-1)))}
    elif kind == 3 and h >= 4 and w >= 4:
        cells -= {(0, 0), (h-1, w-1)}
    if rng.random() < .5:
        cells = {(c, r) for r, c in cells}
    shell = _neighbors(cells, D8)
    lowr, lowc = min(r for r, c in shell), min(c for r, c in shell)
    move = lambda points: {(r-lowr, c-lowc) for r, c in points}
    return dict(inner=move(cells), shell=move(shell))


def _place(rng, n, shape, compact, cycle):
    rooms = []
    h, w = shape
    blocked = np.zeros(shape, dtype=bool)
    for _ in range(n):
        local = _shape(rng, compact)
        sh = max(r for r, c in local["shell"]) + 1
        sw = max(c for r, c in local["shell"]) + 1
        windows = np.lib.stride_tricks.sliding_window_view(blocked, (sh, sw))
        legal = ~windows.any(axis=(-1, -2))
        legal[:2] = False
        legal[-2:] = False
        legal[:, :2] = False
        legal[:, -2:] = False
        origins = np.argwhere(legal)
        if not len(origins):
            return None
        dr, dc = (int(v) for v in origins[int(rng.integers(len(origins)))])
        room = {k: {(r+dr, c+dc) for r, c in points} for k, points in local.items()}
        if not _ports(room):
            return None
        rooms.append(room)
        for r, c in room["inner"] | room["shell"]:
            blocked[max(0, r-2):min(h, r+3), max(0, c-2):min(w, c+3)] = True
    # A nearest-neighbor path gives a changing spatial scaffold for the graph;
    # room placement itself is continuous pixel-coordinate sampling.
    centers = [np.mean(list(r["inner"]), axis=0) for r in rooms]
    if cycle:
        middle = np.mean(centers, axis=0)
        order = sorted(range(n), key=lambda i: np.arctan2(*(centers[i]-middle)))
        offset = int(rng.integers(n))
        order = order[offset:] + order[:offset]
        if rng.random() < .5:
            order.reverse()
        return [rooms[i] for i in order]
    order = [int(rng.integers(n))]
    while len(order) < n:
        remaining = [i for i in range(n) if i not in order]
        weights = {i: float(np.linalg.norm(centers[i]-centers[order[-1]])) *
                   float(rng.uniform(.8, 1.2)) for i in remaining}
        order.append(min(remaining, key=weights.get))
    return [rooms[i] for i in order]


def _path(rng, starts, goal, allowed):
    starts = sorted(starts)
    rng.shuffle(starts)
    todo = deque(starts)
    previous = {p: None for p in starts}
    steps = [D4[int(i)] for i in rng.permutation(4)]
    while todo:
        p = todo.popleft()
        if p == goal:
            path = []
            while p is not None:
                path.append(p)
                p = previous[p]
            return set(path)
        for d in steps:
            q = _add(p, d)
            if q in allowed and q not in previous:
                previous[q] = p
                todo.append(q)
    return None


def _wire(rng, rooms, members, source, shape, existing):
    chamber_cells = set().union(*(r["inner"] | r["shell"] for r in rooms))
    canvas = {(r, c) for r in range(1, shape[0]-1) for c in range(1, shape[1]-1)}
    allowed = canvas - chamber_cells - _neighbors(chamber_cells)
    wire_blocked = existing | _neighbors(existing, D8)
    allowed -= wire_blocked
    options = {}
    for i in members:
        options[i] = [g for g in _ports(rooms[i])
                      if g["port"] in canvas and g["port"] not in wire_blocked
                      and _add(g["port"], g["d"]) in allowed]
        if not options[i]:
            return None
    for _ in range(18):
        ports = {i: _pick(rng, options[i]) for i in members}
        qs = {g["port"] for g in ports.values()}
        if len(qs) != len(members):
            continue
        local_allowed = allowed | qs
        first = source if source is not None else members[0]
        tree = {ports[first]["port"]}
        todo = [i for i in members if i != first]
        rng.shuffle(todo)
        for i in todo:
            path = _path(rng, tree, ports[i]["port"], local_allowed)
            if path is None:
                break
            tree.update(path)
        else:
            return tree, ports
    return None


def _topology(rng, n, lesson, cycle):
    # Each record is (all attached rooms, exporting room or None, explicit).
    if lesson == 0:
        return [([0, 1], 0, False), ([1, 2], 1, False), ([2, 3], None, False)]
    if lesson == 1:
        return [([0, 1, 2], 1, True), ([2, 3], 2, False)]
    if cycle:
        # Consecutive links may share one branching supply. The final link
        # closes a cycle of exporting/receiving chambers without requiring
        # a separate crowded wire for every sink.
        routes, start = [], 0
        while start < n:
            length = min(n-start, 1 if n == 3 else int(rng.integers(1, 3)))
            members = [(start+i) % n for i in range(length+1)]
            routes.append((members, start, bool(rng.random() < .3)))
            start += length
        return routes
    routes, start = [], 0
    while start < n-1:
        length = min(n-1-start, int(_pick(rng, (1, 1, 2, 2, 3))))
        members = list(range(start, start+length+1))
        source = start
        if not routes and length >= 2 and rng.random() < .5:
            source = start+1
        if routes and rng.random() < .25:
            source = None
        routes.append((members, source, bool(source is not None and rng.random() < .4)))
        start += length
    return routes


def _build(rng, roles, n, lesson, cycle, attempt):
    bg, shell, ink = roles
    low = 22 if n <= 4 else (26 if n <= 6 else 29)
    shape = tuple(int(rng.integers(low, 31)) for _ in range(2))
    rooms = _place(rng, n, shape, compact=n >= 6 or attempt >= 20, cycle=cycle)
    if rooms is None:
        return None
    topology = _topology(rng, n, lesson, cycle)
    available = [int(v) for v in rng.permutation(10) if v not in roles]
    nw = min(len(topology), int(rng.integers(2, 4)))
    wires, paints = available[:nw], available[nw:]
    fills = [int(_pick(rng, paints + [bg])) for _ in rooms]
    if lesson == 0:
        fills = [paints[0], paints[1], paints[2], paints[2]]
    elif lesson == 1:
        fills = [paints[1], paints[0], bg, paints[1]]
    elif cycle and n == 3:
        fills = paints[:3]
    a = np.full(shape, bg, dtype=np.int8)
    for room, color in zip(rooms, fills):
        for p in room["inner"]:
            a[p] = color
        for p in room["shell"]:
            a[p] = shell
    out = a.copy()
    existing, assignments, arrow_writes = set(), {}, set()
    for j, (members, source, explicit) in enumerate(topology):
        result = _wire(rng, rooms, members, source, shape, existing)
        if result is None:
            return None
        cells, ports = result
        existing.update(cells)
        for p in cells:
            a[p] = out[p] = wires[j % nw]
        if source is None:
            continue
        g = ports[source]
        if g["write"] & arrow_writes:
            return None
        arrow_writes.update(g["write"])
        choices = [c for c in paints + [bg] if c != fills[source]]
        payload = int(_pick(rng, choices)) if explicit else fills[source]
        if lesson == 1 and j == 0:
            payload = paints[3]
        for p in g["write"]:
            a[p] = ink
        if explicit:
            a[g["center"]] = payload
        for i in members:
            if i == source:
                continue
            if i in assignments and assignments[i] != payload:
                return None
            assignments[i] = payload
    for i, color in assignments.items():
        for p in rooms[i]["inner"]:
            out[p] = color
    rotation = int(rng.integers(4))
    a, out = np.rot90(a, rotation).copy(), np.rot90(out, rotation).copy()
    if rng.random() < .5:
        a, out = np.fliplr(a).copy(), np.fliplr(out).copy()
    # Construction uses sampled chambers and graph edges. The reader receives
    # only the finished input and must recover both independently.
    try:
        prediction, info = _analyze(a)
    except AssertionError:
        return None
    if not np.array_equal(prediction, out) or len(info["rooms"]) != n:
        return None
    if len(info["routes"]) != len(topology):
        return None
    return {"input": a, "output": out}


def _features(info):
    rooms, routes = info["rooms"], info["routes"]
    assigned = info["assignments"]
    final = [assigned.get(i, r["fill"]) for i, r in enumerate(rooms)]
    return dict(
        chambers=len(rooms), wires=len(routes), outlets=len(info["arrows"]),
        branch=any(r["source"] is not None and len(r["attachments"]) >= 3 for r in routes),
        explicit=any(v["explicit"] and v["payload"] != v["fill"] for v in info["arrows"]),
        snapshot=any(not v["explicit"] and final[v["room"]] != v["fill"]
                     for v in info["arrows"]),
        inactive=any(r["source"] is None and any(i not in assigned for i in r["attachments"]) for r in routes),
        locality=any(rooms[i]["fill"] == rooms[j]["fill"] and final[i] != final[j]
                     for i in range(len(rooms)) for j in range(i)),
        empty_target=any(rooms[i]["fill"] == info["bg"] and c != info["bg"] for i, c in assigned.items()),
        filled_target=any(rooms[i]["fill"] not in (info["bg"], c) for i, c in assigned.items()),
        erase=any(c == info["bg"] and rooms[i]["fill"] != c for i, c in assigned.items()),
        nonrectangle=any(len(r["inner"]) !=
                         (max(p[0] for p in r["inner"])-min(p[0] for p in r["inner"])+1) *
                         (max(p[1] for p in r["inner"])-min(p[1] for p in r["inner"])+1) for r in rooms),
        directions={v["d"] for v in info["arrows"]},
        wire_lengths=[len(r["cells"]) for r in routes],
        cavity_areas=[len(r["inner"]) for r in rooms])


def _counterexamples(input_grid, analysis=None):
    """Concrete wrong-rule arrays for the two minimal curriculum lessons."""
    a = np.asarray(input_grid)
    correct, info = _analyze(a) if analysis is None else analysis
    clean = a.copy()
    for room in info["rooms"]:
        for p in room["inner"]:
            clean[p] = room["fill"]
    wrong = {"omit_routes": clean.copy()}
    for mode in ("junction_only", "ignore_explicit", "cascade", "wire_paint"):
        result = clean.copy()
        for route in info["routes"]:
            source = route["source"]
            if source is None:
                continue
            if mode == "junction_only":
                color = int(a[source["center"]])
            elif mode == "ignore_explicit":
                color = source["fill"]
            elif mode == "wire_paint":
                color = route["color"]
            else:
                color = source["payload"] if source["explicit"] else info["assignments"].get(source["room"], source["fill"])
            for i in route["attachments"]:
                if i != source["room"]:
                    for p in info["rooms"][i]["inner"]:
                        result[p] = color
        wrong[mode] = result
    return {name: int(np.count_nonzero(result != correct)) for name, result in wrong.items()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test) or
            not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError("require integer counts 2..10 / 1..4 with train > test")
    rng = np.random.default_rng(seed)
    roles = tuple(int(c) for c in rng.permutation(10)[:3])
    puzzle, seen = {"train": [], "test": []}, set()
    largest_train = 4
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            lesson = index if split == "train" and index < 2 else -1
            cycle = split == "train" and index == 2
            if split == "test" and num_train >= 3:
                cycle = bool(rng.random() < .35)
            n = 4 if lesson >= 0 else int(rng.integers(3, 6) if split == "train" else rng.integers(5, 8))
            if split == "test":
                n = max(n, largest_train+1)
            if cycle and split == "train":
                n = 3
            for attempt in range(100):
                pair = _build(rng, roles, n, lesson, cycle, attempt)
                if pair is not None and _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    if split == "train":
                        largest_train = max(largest_train, n)
                    break
            else:
                raise RuntimeError("bounded chamber routing exhausted")
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "split schema")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _need(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal pair counts")
    seen, roles, features, contrasts = set(), set(), {"train": [], "test": []}, []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _need(inp.shape == out.shape, "shape changed")
            _need(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            predicted, info = _analyze(inp)
            _need(np.array_equal(predicted, out), "incorrect chamber transfer")
            interior = set().union(*(r["inner"] for r in info["rooms"]))
            _need(all(inp[p] == out[p] for p in np.ndindex(inp.shape) if p not in interior), "structure changed")
            _need(not np.array_equal(inp, out), "identity example")
            _need(info["ink"] not in out, "annotation survived")
            roles.add((info["bg"], info["shell"], info["ink"]))
            features[split].append(_features(info))
            if split == "train" and len(contrasts) < 2:
                contrasts.append(_counterexamples(inp, (predicted, info)))
    _need(len(roles) == 1, "puzzle-wide structural roles changed")
    for name in ("snapshot", "locality", "inactive", "explicit", "branch", "filled_target", "empty_target", "erase"):
        _need(any(f[name] for f in features["train"]), "missing discriminating demonstration: " + name)
    for name in ("omit_routes", "junction_only", "ignore_explicit", "cascade", "wire_paint"):
        _need(any(c[name] > 0 for c in contrasts), "shortcut survives the two lessons: " + name)
    _need(min(f["chambers"] for f in features["test"]) >
          max(f["chambers"] for f in features["train"]), "missing graph-size shift")


def _normalized(a):
    rename, result = {}, []
    for c in np.asarray(a).flat:
        c = int(c)
        result.append(rename.setdefault(c, len(rename)))
    return a.shape, bytes(result)


def _semantic_checks():
    """Metamorphic checks and a hand-built, consistent two-inlet chamber."""
    puzzle = generate_puzzle(0, 2, 1)
    rng = np.random.default_rng(271828)
    checks = 0
    for pair in puzzle["train"]:
        inp, out = pair["input"], pair["output"]
        perm = rng.permutation(10)
        for turn in range(4):
            transformed = np.rot90(perm[inp], turn)
            expected = np.rot90(perm[out], turn)
            _need(np.array_equal(solve(transformed), expected), "color/rotation equivariance failed")
            _need(np.array_equal(solve(np.fliplr(transformed)), np.fliplr(expected)), "reflection failed")
            checks += 2
        _, info = _analyze(inp)
        source = info["arrows"][0]
        allowed = [c for c in range(10) if c not in
                   {info["shell"], info["ink"], source["payload"]} |
                   {r["color"] for r in info["routes"]}]
        color = allowed[0]
        changed, expected = inp.copy(), out.copy()
        changed[source["center"]] = color
        route = info["routes"][source["route"]]
        for i in route["attachments"]:
            if i != source["room"]:
                for p in info["rooms"][i]["inner"]:
                    expected[p] = color
        _need(np.array_equal(solve(changed), expected), "literal payload intervention failed")
        checks += 1
    rooms = []
    for top, left, height, width in ((3, 3, 3, 3), (3, 21, 3, 3), (22, 11, 4, 7)):
        inner = {(r, c) for r in range(top, top+height) for c in range(left, left+width)}
        rooms.append(dict(inner=inner, shell=_neighbors(inner, D8)))
    inp = np.zeros((30, 30), dtype=np.int8)
    for room, color in zip(rooms, (6, 8, 9)):
        for p in room["shell"]:
            inp[p] = 5
        for p in room["inner"]:
            inp[p] = color
    expected = inp.copy()
    for p in rooms[2]["inner"]:
        expected[p] = 7
    existing, outlet_centers = set(), []
    for i, wire_color in ((0, 2), (1, 3)):
        result = _wire(rng, rooms, [i, 2], i, inp.shape, existing)
        _need(result is not None, "semantic fixture routing failed")
        cells, ports = result
        existing.update(cells)
        for p in cells:
            inp[p] = expected[p] = wire_color
        for p in ports[i]["write"]:
            inp[p] = 1
        inp[ports[i]["center"]] = 7
        outlet_centers.append(ports[i]["center"])
    _need(np.array_equal(solve(inp), expected), "consistent multiple inlets failed")
    wrong = inp.copy()
    wrong[outlet_centers[0]] = 6
    try:
        solve(wrong)
    except AssertionError:
        pass
    else:
        raise AssertionError("inconsistent multiple inlets were silently resolved")
    return dict(metamorphic_checks=checks, consistent_inlets=True, conflicting_inlets_rejected=True,
                larger_unsampled_interior_area=28,
                curriculum_counterexamples=[_counterexamples(p["input"]) for p in puzzle["train"]])


def self_test():
    """Bounded 42-seed audit. Retain statistics/hashes, never puzzle corpora.

    At most two puzzles coexist during replacement (50,400 raw int8 grid
    bytes), plus small structural signatures and transient reconstructions.
    No files are made and no full 42-seed corpus is retained.
    """
    signatures, structures, shapes, palettes = set(), set(), set(), set()
    totals = {k: 0 for k in ("snapshot", "locality", "inactive", "explicit", "branch",
                             "empty_target", "filled_target", "erase", "nonrectangle")}
    directions, areas, lengths, counts = set(), set(), set(), set()
    seed_zero = None
    for seed in AUDIT_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        if seed == 0:
            seed_zero = _signature(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                before = inp.copy()
                prediction, info = _analyze(inp)
                _need(np.array_equal(inp, before), "solver mutates its input")
                _need(np.array_equal(prediction, out), "input-only prediction failed")
                f = _features(info)
                for name in totals:
                    totals[name] += bool(f[name])
                directions.update(f["directions"])
                counts.add(f["chambers"])
                areas.update(f["cavity_areas"])
                lengths.update(f["wire_lengths"])
                shapes.add(inp.shape)
                palettes.add((info["bg"], info["shell"], info["ink"]))
                structures.add(hashlib.sha256(repr(_normalized(inp)).encode()).hexdigest())
        # A changed answer must be detected even if it remains within 0..9.
        bad = {s: list(puzzle[s]) for s in puzzle}
        changed = dict(bad["test"][0])
        changed["output"] = np.array(changed["output"], copy=True)
        p = next(iter(_analyze(changed["input"])[1]["rooms"][0]["inner"]))
        changed["output"][p] = (int(changed["output"][p])+1) % 10
        bad["test"][0] = changed
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("wrong interior label accepted")
    _need(len(signatures) == 42 and len(structures) >= 200, "collapsed structural variation")
    _need(len(shapes) >= 20 and len(palettes) >= 30, "collapsed dimensions or roles")
    _need(directions == set(D4) and {3, 4, 5, 6, 7} <= counts, "missing geometry/count variation")
    _need(len(areas) >= 8 and len(lengths) >= 12, "collapsed cavity or route geometry")
    _need(all(value >= 3 for value in totals.values()), "missing family mechanism")
    _need(_signature(generate_puzzle(0)) == seed_zero, "seed does not reproduce whole puzzle")
    # All 30 legal explicit count combinations, plus types and range failures.
    legal_calls = 0
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(901+nt*5+ne, nt, ne)
            _need(len(p["train"]) == nt and len(p["test"]) == ne, "ignored explicit counts")
            legal_calls += 1
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return dict(seeds=42, pairs=210, distinct_puzzles=len(signatures),
                structures=len(structures), shapes=len(shapes), palettes=len(palettes),
                chamber_counts=sorted(counts), cavity_areas=sorted(areas),
                wire_lengths=sorted(lengths), features=totals, legal_count_calls=legal_calls,
                semantic_checks=_semantic_checks(),
                semantic_verification="generator-owned reconstruction and counterexamples only")
