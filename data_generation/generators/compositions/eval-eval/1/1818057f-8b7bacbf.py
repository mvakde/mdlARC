"""Insulated plus crossings route visible paint to hollow chambers.

All semantic information is in the input.  Generation builds a sparse circuit,
chooses terminal supplies, and attaches independently sampled chamber shapes.
Only standard Python and NumPy are used.
"""
import json

import numpy as np

D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer grid with dimensions 1..30 and colors 0..9")
    return a.astype(np.int64, copy=True)


def _components(cells, steps=D4):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group = {start}
        todo = [start]
        for r, c in todo:
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    todo.append(q)
        result.append(group)
    return result


def _neighbors(cells, steps=D4):
    return {(r + dr, c + dc) for r, c in cells for dr, dc in steps} - set(cells)


def _circuit(cells, mode="plus"):
    """Expand a crossing into horizontal and vertical states, without an edge
    between them.  Long arms and other nearby wire pixels are ordinary graph
    cells.  `plain` and `junctions` exist only for counterfactual checks.
    """
    cells = set(cells)
    centers = {p for p in cells if all((p[0] + dr, p[1] + dc) in cells
                                      for dr, dc in D4)}
    if mode == "plus":
        split = centers
    elif mode == "plain":
        split = set()
    elif mode == "junctions":
        split = {p for p in cells if sum((p[0] + dr, p[1] + dc) in cells
                                         for dr, dc in D4) >= 3}
    else:
        raise ValueError("Unknown circuit mode")

    def state(p, direction):
        axis = "v" if direction[0] else "h"
        return (p[0], p[1], axis if p in split else "o")

    graph = {}
    for p in sorted(cells):
        ports = [d for d in D4 if (p[0] + d[0], p[1] + d[1]) in cells]
        nodes = {state(p, d) for d in ports} or {(p[0], p[1], "o")}
        for node in nodes:
            graph.setdefault(node, set())
        for dr, dc in ports:
            q = (p[0] + dr, p[1] + dc)
            left, right = state(p, (dr, dc)), state(q, (-dr, -dc))
            graph.setdefault(left, set()).add(right)
            graph.setdefault(right, set()).add(left)
    labels, groups = {}, []
    for node in sorted(graph):
        if node in labels:
            continue
        label = len(groups)
        labels[node] = label
        todo = [node]
        for current in todo:
            for nxt in sorted(graph[current]):
                if nxt not in labels:
                    labels[nxt] = label
                    todo.append(nxt)
        groups.append(set(todo))
    return dict(centers=centers, split=split, labels=labels, groups=groups,
                state=state, graph=graph)


def _analyze(input_grid, mode="plus", conflicts=False):
    a = _grid(input_grid)
    values, counts = np.unique(a, return_counts=True)
    bg = int(values[int(np.argmax(counts))])
    if int(counts.max()) * 2 <= a.size:
        raise ValueError("The background must occupy more than half the input")
    colors = set(map(int, values)) - {bg}
    color_groups = {c: _components(map(tuple, np.argwhere(a == c)), D8)
                    for c in colors}
    paint_colors = {c for c, groups in color_groups.items()
                    if all(len(g) == 1 for g in groups)}
    structural = colors - paint_colors
    if len(structural) != 2 or not paint_colors:
        raise ValueError("Need two structural colors and isolated paint terminals")
    sources, wire_colors = [], set()
    for color in sorted(paint_colors):
        for group in color_groups[color]:
            p = next(iter(group))
            adjacent = [q for q in _neighbors({p})
                        if 0 <= q[0] < a.shape[0] and 0 <= q[1] < a.shape[1]
                        and a[q] != bg]
            if len(adjacent) != 1 or int(a[adjacent[0]]) not in structural:
                raise ValueError("Each paint terminal needs exactly one orthogonal wire neighbor")
            wire_colors.add(int(a[adjacent[0]]))
            sources.append((p, color, adjacent[0]))
    if len(wire_colors) != 1:
        raise ValueError("All routes use one visibly identifiable wire color")
    wire = wire_colors.pop()
    wall = (structural - {wire}).pop()
    if any(len(g) < 4 for g in color_groups[wall]):
        raise ValueError("Walls must be recognizable contours, not singleton markers")
    cells = set(map(tuple, np.argwhere(a == wire)))
    circuit = _circuit(cells, mode)
    supplies = {}
    for p, color, q in sources:
        direction = (p[0] - q[0], p[1] - q[1])
        node = circuit["state"](q, direction)
        label = circuit["labels"][node]
        supplies.setdefault(label, set()).add(color)
    if not conflicts and any(len(s) > 1 for s in supplies.values()):
        raise ValueError("Conflicting paint sources share a conducting route")

    loops = []
    for interior in _components(map(tuple, np.argwhere(a == bg))):
        if any(r in (0, a.shape[0] - 1) or c in (0, a.shape[1] - 1)
               for r, c in interior):
            continue
        shell = _neighbors(interior)
        if not all(int(a[q]) == wall for q in shell):
            # A wire cycle or a mixed-color enclosure is not a wall chamber.
            continue
        ports = _neighbors(shell) & cells
        if len(ports) > 1:
            raise ValueError("A chamber must have at most one wire port")
        label = None
        if ports:
            q = next(iter(ports))
            # A wire cell next to a wall cannot itself be a complete plus.
            label = circuit["labels"][(q[0], q[1], "o")]
        paints = supplies.get(label, set())
        paint = min(paints) if paints else None
        loops.append(dict(interior=interior, shell=shell, ports=ports,
                          label=label, paint=paint))
    if not loops:
        raise ValueError("Need at least one hollow wall chamber")
    answer = a.copy()
    for loop in loops:
        if loop["paint"] is not None:
            for p in loop["interior"]:
                answer[p] = loop["paint"]
    return answer, dict(bg=bg, wire=wire, wall=wall, sources=sources,
                        loops=loops, circuit=circuit, supplies=supplies)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive roles, insulated routes, source ownership and holes from input only."""
    return _analyze(input_grid)[0]


def _ri(rng, low, high):
    return int(rng.integers(low, high + 1))


def _choice(rng, seq):
    return seq[_ri(rng, 0, len(seq) - 1)]


def _axis(rng, count, fallback=False):
    gaps = [5] * (count - 1) if fallback else [_ri(rng, 4, 6) for _ in range(count - 1)]
    while sum(gaps) > 19:
        i = int(np.argmax(gaps))
        gaps[i] -= 1
    start = 5 if fallback else _ri(rng, 4, 25 - sum(gaps))
    return [start] + [start + int(x) for x in np.cumsum(gaps)]


def _skeleton(rng, kind, fallback=False):
    nr = 3 if fallback else _ri(rng, 3, 5)
    nc = (4 if kind == "repeat" else 3) if fallback else _ri(rng, 4 if kind == "repeat" else 3, 5)
    rows, cols = _axis(rng, nr, fallback), _axis(rng, nc, fallback)
    all_edges = []
    for i in range(nr):
        for j in range(nc):
            if i + 1 < nr:
                all_edges.append(((i, j), (i + 1, j)))
            if j + 1 < nc:
                all_edges.append(((i, j), (i, j + 1)))
    density = 0 if fallback else float(rng.uniform(.08, .28))
    edges = {e for e in all_edges if rng.random() < density}
    i = 1 if fallback else _ri(rng, 1, nr - 2)
    j = 1 if fallback else _ri(rng, 1, nc - (3 if kind == "repeat" else 2))
    hubs = [(i, j), (i, j + 1)] if kind == "repeat" else [(i, j)]
    for hub in hubs:
        for dr, dc in D4:
            nxt = (hub[0] + dr, hub[1] + dc)
            edges.add(tuple(sorted((hub, nxt))))
    if kind == "branch":
        direction = (0, -1) if fallback else _choice(rng, D4)
        nxt = (i + direction[0], j + direction[1])
        edges.discard(tuple(sorted(((i, j), nxt))))
    cells = set()
    for (r, c), (y, x) in sorted(edges):
        r, c, y, x = rows[r], cols[c], rows[y], cols[x]
        if r == y:
            cells.update((r, z) for z in range(min(c, x), max(c, x) + 1))
        else:
            cells.update((z, c) for z in range(min(r, y), max(r, y) + 1))
    return cells, [(rows[r], cols[c]) for r, c in hubs]


def _chamber(rng, simple=False):
    """Four-connected row profiles: boxes, diamonds, steps, slopes, waists."""
    kind = "box" if simple else _choice(rng, ("box", "box", "diamond", "step", "slope", "waist", "long"))
    if simple:
        inner = {(0, 0)}
    elif kind == "diamond":
        radius = _ri(rng, 0, 2)
        inner = {(r, c) for r in range(-radius, radius + 1)
                 for c in range(-radius, radius + 1) if abs(r) + abs(c) <= radius}
    else:
        h, w = _ri(rng, 1, 4), _ri(rng, 1, 5)
        if kind == "long":
            h, w = 1, _ri(rng, 3, 6)
        inner, left = set(), 0
        for r in range(h):
            if kind in ("step", "slope") and r:
                left += _choice(rng, (0, 1))
            width = w
            if kind == "waist" and 0 < r < h - 1:
                width = max(1, w - 1)
            if kind == "step":
                width = max(2, w + _ri(rng, -1, 1))
            inner.update((r, c) for c in range(left, left + width))
            if r and not {c for rr, c in inner if rr == r - 1} & {c for rr, c in inner if rr == r}:
                inner.add((r, left - 1))
    turns = _ri(rng, 0, 3)
    for _ in range(turns):
        inner = {(-c, r) for r, c in inner}
    shell = _neighbors(inner)
    return inner, shell


def _place_chamber(rng, a, bg, wall, wire_cells, reserved, leaf=None,
                   outward=None, simple=False):
    shape = a.shape
    occupied = set(map(tuple, np.argwhere(a != bg)))
    for attempt in range(32):
        inner, shell = _chamber(rng, simple or attempt >= 20)
        if leaf is None:
            anchor = (_ri(rng, 1, 28), _ri(rng, 1, 28))
            origin = min(shell)
            dr, dc = anchor[0] - origin[0], anchor[1] - origin[1]
        else:
            # Attach only at the face pointing back towards the wire leaf.
            axis = 0 if outward[0] else 1
            extreme = (min if outward[axis] > 0 else max)(p[axis] for p in shell)
            face = sorted(p for p in shell if p[axis] == extreme)
            port = _choice(rng, face)
            anchor = (leaf[0] + outward[0], leaf[1] + outward[1])
            dr, dc = anchor[0] - port[0], anchor[1] - port[1]
        inner = {(r + dr, c + dc) for r, c in inner}
        shell = {(r + dr, c + dc) for r, c in shell}
        footprint = inner | shell
        if not all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in footprint):
            continue
        if footprint & (occupied | reserved):
            continue
        if _neighbors(footprint, D8) & (occupied - wire_cells):
            continue
        ports = _neighbors(shell) & wire_cells
        if ports != (set() if leaf is None else {leaf}):
            continue
        if _neighbors(inner) & wire_cells:
            continue
        for p in shell:
            a[p] = wall
        return inner, shell
    return None


def _scene(rng, kind, roles, fallback=False):
    bg, wire, wall = roles
    cells, hubs = _skeleton(rng, kind, fallback)
    circuit = _circuit(cells)
    leaves = {p for p in cells if len(_neighbors({p}) & cells) == 1}
    by_label = {}
    for p in sorted(leaves):
        label = circuit["labels"][(p[0], p[1], "o")]
        by_label.setdefault(label, []).append(p)
    if kind == "branch":
        hub = hubs[0]
        label = circuit["labels"][(hub[0], hub[1], "o")]
        if len(by_label.get(label, [])) < 3:
            return None
        active = [label]
    else:
        hub = hubs[0]
        h_label = circuit["labels"][(hub[0], hub[1], "h")]
        v_label = circuit["labels"][(hub[0], hub[1], "v")]
        if h_label == v_label or min(len(by_label.get(x, [])) for x in (h_label, v_label)) < 2:
            return None
        active = [h_label, v_label]
        if kind == "repeat":
            second = hubs[1]
            if circuit["labels"][(second[0], second[1], "h")] != h_label:
                return None
            extra = circuit["labels"][(second[0], second[1], "v")]
            if extra not in active and len(by_label.get(extra, [])) >= 2 and (fallback or rng.random() < .6):
                active.append(extra)
        elif not fallback and rng.random() < .3:
            options = [label for label, tips in by_label.items()
                       if label not in active and len(tips) >= 2]
            if options:
                active.append(_choice(rng, sorted(options)))

    paints = list(map(int, rng.choice([c for c in range(10) if c not in roles],
                                      size=len(active), replace=False)))
    if len(paints) == 3 and not fallback and rng.random() < .3:
        paints[2] = _choice(rng, paints[:2])
    a = np.full((30, 30), bg, dtype=np.int64)
    for p in cells:
        a[p] = wire
    source_points, targets, used = [], [], set()
    for label, paint in zip(active, paints):
        tips = list(by_label[label])
        rng.shuffle(tips)
        source = tips.pop()
        a[source] = paint
        source_points.append(source)
        used.add(source)
        count = 2 if kind == "branch" else _ri(rng, 1, min(3, len(tips)))
        for tip in tips[:count]:
            targets.append((tip, paint))
            used.add(tip)
    wire_cells = cells - set(source_points)
    # Terminal reservations keep chamber footprints away from other markers
    # and future ports, but impose no semantic role on layout positions.
    reserved = leaves - used
    intended = []
    rng.shuffle(targets)
    for tip, paint in targets:
        neighbor = next(iter(_neighbors({tip}) & wire_cells))
        outward = (tip[0] - neighbor[0], tip[1] - neighbor[1])
        placed = _place_chamber(rng, a, bg, wall, wire_cells, reserved,
                               tip, outward, fallback)
        if placed is None:
            return None
        intended.append((placed[0], paint))
    # Optional unmarked circuit ends get attached hollow targets.
    inactive = [p for label, tips in by_label.items() if label not in active for p in tips]
    rng.shuffle(inactive)
    for tip in inactive[:0 if fallback else _ri(rng, 0, min(2, len(inactive)))]:
        neighbor = next(iter(_neighbors({tip}) & wire_cells))
        outward = (tip[0] - neighbor[0], tip[1] - neighbor[1])
        placed = _place_chamber(rng, a, bg, wall, wire_cells, reserved - {tip},
                               tip, outward, fallback)
        if placed is not None:
            intended.append((placed[0], None))
    # Every branch lesson includes a detached chamber, teaching that wall color
    # and proximity alone do not constitute a supply.
    detached = 1 if kind == "branch" else (0 if fallback else _ri(rng, 0, 2))
    for _ in range(detached):
        placed = _place_chamber(rng, a, bg, wall, wire_cells, reserved,
                               simple=fallback)
        if placed is None:
            if kind == "branch":
                return None
            continue
        intended.append((placed[0], None))
    b = a.copy()
    for interior, paint in intended:
        if paint is not None:
            for p in interior:
                b[p] = paint
    occupied = np.argwhere(a != bg)
    lo, hi = occupied.min(axis=0), occupied.max(axis=0)
    margins = rng.integers(0, 4, size=4)
    r0, c0 = max(0, int(lo[0]) - int(margins[0])), max(0, int(lo[1]) - int(margins[1]))
    r1, c1 = min(30, int(hi[0]) + 1 + int(margins[2])), min(30, int(hi[1]) + 1 + int(margins[3]))
    if not fallback and rng.random() < .12:
        r0 = c0 = 0
        r1 = c1 = 30
    a, b = a[r0:r1, c0:c1].copy(), b[r0:r1, c0:c1].copy()
    rotation = _ri(rng, 0, 3)
    a, b = np.rot90(a, rotation).copy(), np.rot90(b, rotation).copy()
    if rng.random() < .5:
        a, b = np.fliplr(a).copy(), np.fliplr(b).copy()
    try:
        predicted, info = _analyze(a)
        if not np.array_equal(predicted, b) or len(info["loops"]) != len(intended):
            return None
        if kind == "branch" and np.array_equal(_analyze(a, "junctions", True)[0], b):
            return None
        if kind != "branch" and not any(len(s) > 1 for s in _analyze(a, "plain", True)[1]["supplies"].values()):
            return None
    except (ValueError, KeyError):
        return None
    return {"input": a, "output": b}


def _key(a):
    return a.shape, np.asarray(a, dtype=np.uint8).tobytes()


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer from {low} through {high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    preferred = tuple(map(int, rng.choice(10, 3, replace=False)))
    stable_palette = rng.random() < .3
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            kind = ("cross", "branch")[index] if split == "train" and index < 2 else (
                "repeat" if split == "test" else _choice(rng, ("cross", "branch", "repeat")))
            roles = preferred if stable_palette or rng.random() < .5 else tuple(map(int, rng.choice(10, 3, replace=False)))
            for attempt in range(120):
                pair = _scene(rng, kind, roles, fallback=attempt >= 80)
                if pair is not None and _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Bounded circuit construction could not find a distinct scene")
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _nearest(a, info):
    """A deliberately wrong alternative: ignore routing and use nearest source."""
    answer = a.copy()
    for loop in info["loops"]:
        for p in loop["interior"]:
            source = min(info["sources"], key=lambda s: (abs(s[0][0] - p[0]) + abs(s[0][1] - p[1]), s[1], s[0]))
            answer[p] = source[1]
    return answer


def _features(pair):
    a = _grid(pair["input"])
    out, info = _analyze(a)
    circuit = info["circuit"]
    center_counts = [sum((r, c) in circuit["centers"] for r, c, axis in group)
                     for group in circuit["groups"]]
    plain_answer, raw = _analyze(a, "plain", True)
    wire_cells = set(map(tuple, np.argwhere(a == info["wire"])))
    shape_keys = []
    for loop in info["loops"]:
        r0 = min(r for r, c in loop["interior"])
        c0 = min(c for r, c in loop["interior"])
        shape_keys.append(tuple(sorted((r - r0, c - c0) for r, c in loop["interior"])))
    return dict(shape=a.shape, roles=(info["bg"], info["wire"], info["wall"]),
                sources=len(info["sources"]), paints=len({s[1] for s in info["sources"]}),
                centers=len(circuit["centers"]), loops=len(info["loops"]),
                blank=sum(loop["paint"] is None for loop in info["loops"]),
                detached=sum(not loop["ports"] for loop in info["loops"]),
                chamber_shapes=shape_keys, areas=[len(loop["interior"]) for loop in info["loops"]],
                wire_cells=int(np.count_nonzero(a == info["wire"])),
                branches=sum(len(_neighbors({p}) & wire_cells) == 3 for p in wire_cells),
                mixed_raw=any(len(s) > 1 for s in raw["supplies"].values()),
                plus_witness=not np.array_equal(plain_answer, out),
                turn_witness=not np.array_equal(_analyze(a, "junctions", True)[0], out),
                nearest_witness=not np.array_equal(_nearest(a, info), out),
                productive_cross_depth=max([center_counts[loop["label"]] for loop in info["loops"]
                                            if loop["paint"] is not None] or [0]))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            predicted, info = _analyze(a)
            assert np.array_equal(predicted, b), "Paint ownership or cavity fill mismatch"
            allowed = set().union(*(loop["interior"] for loop in info["loops"]
                                    if loop["paint"] is not None))
            changed = set(map(tuple, np.argwhere(a != b)))
            assert changed == allowed, "Only complete supplied chamber interiors change"
            assert all(int(a[p]) == info["bg"] for p in changed)
            assert set(map(int, np.unique(b))) <= set(map(int, np.unique(a)))
            for loop in info["loops"]:
                if loop["paint"] is None:
                    assert all(b[p] == a[p] for p in loop["interior"])
                else:
                    assert all(b[p] == loop["paint"] for p in loop["interior"])
            features[split].append(_features(pair))
    train = features["train"]
    assert any(f["mixed_raw"] and f["plus_witness"] for f in train), "Training must expose insulated ownership"
    assert any(f["turn_witness"] for f in train), "Training must show a T junction conducting around a turn"
    assert any(f["blank"] and f["detached"] for f in train), "Training must distinguish hollow shape from supply"
    assert any(f["nearest_witness"] for f in train), "Training must defeat geometric proximity"
    for f in features["test"]:
        assert f["centers"] >= 2 and f["productive_cross_depth"] >= 2
        assert f["mixed_raw"] and f["plus_witness"], "Test must retain a meaningful insulation decision"


def _fingerprint(puzzle):
    return tuple((_key(pair["input"]), _key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def _normalized(a):
    roles, cells = {}, []
    for value in a.flat:
        value = int(value)
        cells.append(roles.setdefault(value, len(roles)))
    return a.shape, tuple(cells)


def self_test():
    """Bounded checks of 42 seeds, legal counts, semantic witnesses and mutations.
    Returns measured coverage; no independent solver is claimed.
    """
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    fs = [_features(pair) for puzzle in puzzles for split in ("train", "test")
          for pair in puzzle[split]]
    assert len({_fingerprint(p) for p in puzzles}) == 42
    assert _fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0))
    structures = {_normalized(pair["input"]) for puzzle in puzzles
                  for split in ("train", "test") for pair in puzzle[split]}
    assert len(structures) >= 160
    assert len({f["shape"] for f in fs}) >= 45
    assert len({f["roles"] for f in fs}) >= 20
    assert len({shape for f in fs for shape in f["chamber_shapes"]}) >= 25
    assert len({f["wire_cells"] for f in fs}) >= 40
    assert {1, 2, 3} <= {f["sources"] for f in fs}
    assert max(f["loops"] for f in fs) >= 5
    assert max(f["centers"] for f in fs) >= 3
    assert max(area for f in fs for area in f["areas"]) >= 12
    assert all(len({_normalized(pair["input"]) for split in ("train", "test")
                    for pair in puzzle[split]}) >= 3 for puzzle in puzzles)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(21000 + 10 * nt + ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")
    # Local state topology, independent of the scene builder.
    plus = {(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)}
    assert len(_circuit(plus)["groups"]) == 2
    assert len(_circuit(plus - {(0, -1)})["groups"]) == 1
    adjacent = plus | {(0, 2), (-1, 1), (1, 1)}
    assert _circuit(adjacent)["centers"] == {(0, 0), (0, 1)}
    # The vertical tracks reconnect through their neighboring arm pixels;
    # splitting both centers does not erase these ordinary visible edges.
    assert len(_circuit(adjacent)["groups"]) == 2
    shared_arm = plus | {(0, 2), (0, 3), (-1, 2), (1, 2)}
    assert len(_circuit(shared_arm)["groups"]) == 3
    # Input mutation is forbidden; reflected circuits must give reflected fills.
    for pair in puzzles[0]["train"] + puzzles[0]["test"]:
        a = pair["input"].copy()
        original = a.copy()
        assert np.array_equal(solve(a), pair["output"]) and np.array_equal(a, original)
        for k in range(4):
            assert np.array_equal(solve(np.rot90(a, k)), np.rot90(pair["output"], k))
        assert np.array_equal(solve(np.fliplr(a)), np.fliplr(pair["output"]))
    bad = {s: [{k: np.array(a, copy=True) for k, a in pair.items()} for pair in puzzles[0][s]]
           for s in ("train", "test")}
    p = bad["test"][0]
    cell = tuple(np.argwhere(p["input"] != p["output"])[0])
    p["output"][cell] = p["input"][cell]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Validator accepted a partially filled chamber")
    return dict(seeds=42, pairs=len(fs), distinct_puzzles=42,
                shapes=len({f["shape"] for f in fs}), structures=len(structures),
                role_palettes=len({f["roles"] for f in fs}),
                chamber_shapes=len({shape for f in fs for shape in f["chamber_shapes"]}),
                source_counts=sorted({f["sources"] for f in fs}),
                crossing_counts=sorted({f["centers"] for f in fs}),
                chamber_counts=sorted({f["loops"] for f in fs}),
                cavity_areas=sorted({area for f in fs for area in f["areas"]}),
                semantic_verification="generator-owned checks only")


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
