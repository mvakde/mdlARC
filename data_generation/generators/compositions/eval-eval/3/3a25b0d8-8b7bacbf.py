"""Wired floorplan printing: routes broadcast compartment-addressed palettes.

Only the standard library and NumPy are used. Generation labels construct an
answer; solve reconstructs sources, networks, and room matches from pixels.
"""
from collections import deque
import itertools
import numbers

import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _components(cells, steps=D4):
    remaining = set(cells)
    result = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        part, queue = {root}, [root]
        for r, c in queue:
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    queue.append(q)
        result.append(part)
    return result


def _cells(mask):
    return set(map(tuple, np.argwhere(mask)))


def _near(cells, shape, steps=D8):
    h, w = shape
    return {(r + dr, c + dc) for r, c in cells for dr, dc in steps
            if 0 <= r + dr < h and 0 <= c + dc < w} - set(cells)


def _bbox(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells) + 1, max(c for r, c in cells) + 1)


def _center(cells):
    return np.mean(np.array(sorted(cells)), axis=0)


def _enclosed(walls):
    """Four-connected cavities of one eight-connected framework component."""
    r, c, R, C = _bbox(walls)
    shape = (R - r + 2, C - c + 2)
    solid = {(y - r + 1, x - c + 1) for y, x in walls}
    empty = {(y, x) for y in range(shape[0]) for x in range(shape[1])} - solid
    return [{(y + r - 1, x + c - 1) for y, x in p}
            for p in _components(empty)
            if all(0 < y < shape[0] - 1 and 0 < x < shape[1] - 1 for y, x in p)]


def _adjacency(rooms, walls):
    """Directed shared-wall graph; bits mean up, down, left, right.

    Areas, room discovery indices, absolute coordinates, and colors are not
    correspondence keys. An edge requires a wall pixel between two rooms.
    """
    lookup = {p: i for i, room in enumerate(rooms) for p in room}
    graph = [[0] * len(rooms) for _ in rooms]
    for i, room in enumerate(rooms):
        for r, c in room:
            for bit, (dr, dc) in enumerate(D4):
                q, mid = (r + 2 * dr, c + 2 * dc), (r + dr, c + dc)
                j = lookup.get(q)
                if j is not None and j != i and mid in walls:
                    graph[i][j] |= 1 << bit
    return tuple(map(tuple, graph))


def _matches(a, b):
    """All orientation-preserving isomorphisms of the shared-wall graphs."""
    if len(a) != len(b):
        return []
    n = len(a)
    signatures_a = [tuple(sorted(row)) for row in a]
    signatures_b = [tuple(sorted(row)) for row in b]
    candidates = [[j for j in range(n) if signatures_a[i] == signatures_b[j]]
                  for i in range(n)]
    order = sorted(range(n), key=lambda i: (len(candidates[i]), i))
    found = []

    def visit(k, mapping, used):
        if k == n:
            found.append(tuple(mapping[i] for i in range(n)))
            return
        i = order[k]
        for j in candidates[i]:
            if j in used:
                continue
            if all(a[i][u] == b[j][v] and a[u][i] == b[v][j]
                   for u, v in mapping.items()):
                mapping[i] = j
                visit(k + 1, mapping, used | {j})
                del mapping[i]

    visit(0, {}, set())
    return found


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _decode(input_grid):
    a = _grid(input_grid)
    # The unique modal border color is a fixed, observable family convention.
    border = np.concatenate((a[0], a[-1], a[1:-1, 0], a[1:-1, -1]))
    counts = np.bincount(border, minlength=10)
    if np.sum(counts == counts.max()) != 1:
        raise ValueError("Ambiguous background")
    bg = int(counts.argmax())
    candidates = []
    for wall in sorted(set(map(int, a.flat)) - {bg}):
        objects = []
        valid = True
        for walls in _components(_cells(a == wall), D8):
            rooms = _enclosed(walls)
            if not 3 <= len(rooms) <= 6:
                valid = False
                break
            colors = [{int(a[p]) for p in room} for room in rooms]
            if any(len(c) != 1 for c in colors):
                valid = False
                break
            colors = [next(iter(c)) for c in colors]
            blank = all(c == bg for c in colors)
            donor = bg not in colors and wall not in colors and len(set(colors)) >= 2
            if not (blank or donor):
                valid = False
                break
            objects.append(dict(walls=walls, rooms=rooms, colors=colors,
                                donor=donor, graph=_adjacency(rooms, walls)))
        if (valid and 2 <= len(objects) <= 6
                and len({len(o['rooms']) for o in objects}) == 1
                and any(o['donor'] for o in objects)
                and any(not o['donor'] for o in objects)
                and all(_matches(objects[0]['graph'], o['graph']) for o in objects)):
            candidates.append((wall, objects))
    if len(candidates) != 1:
        raise ValueError("Expected one recognizable framework color")
    wall, objects = candidates[0]
    payload = {c for o in objects if o['donor'] for c in o['colors']}
    wire_colors = sorted(set(map(int, a.flat)) - {bg, wall} - payload)
    wires = [(part, color) for color in wire_colors
             for part in _components(_cells(a == color), D8)]
    wall_cells = set().union(*(o['walls'] for o in objects))
    room_cells = set().union(*(p for o in objects for p in o['rooms']))
    if _cells((a != bg) & (a != wall)) - room_cells != set().union(
            *(p for p, color in wires)):
        raise ValueError("Payload colors may only occur inside donor rooms")
    if not wires or len(wire_colors) > 3:
        raise ValueError("Expected one to three wire colors")
    n = len(objects)
    graph = [set() for _ in range(n + len(wires))]
    for j, (wire, color) in enumerate(wires, n):
        neighbors = _near(wire, a.shape)
        for i, obj in enumerate(objects):
            if obj['walls'] & neighbors:
                graph[i].add(j)
                graph[j].add(i)
    # Wire color continuity and framework relay edges define supply. Neither
    # differently colored wire contact nor mere frame proximity is an edge.
    owners, groups = {}, []
    remaining = set(range(len(graph)))
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        group, queue = {root}, [root]
        for node in queue:
            for other in sorted(graph[node] & remaining):
                remaining.remove(other)
                group.add(other)
                queue.append(other)
        sources = [i for i in sorted(group) if i < n and objects[i]['donor']]
        if len(sources) > 1:
            raise ValueError("A supply network cannot contain two donors")
        for i in group:
            if i < n and not objects[i]['donor']:
                owners[i] = sources[0] if sources else None
        groups.append(group)
    output = a.copy()
    mappings = {}
    for i, source in owners.items():
        if source is None:
            continue
        donor, target = objects[source], objects[i]
        possibilities = _matches(donor['graph'], target['graph'])
        if not possibilities:
            raise ValueError("Incompatible supplied framework")
        assignments = []
        for mapping in possibilities:
            colors = [None] * len(mapping)
            for d, t in enumerate(mapping):
                colors[t] = donor['colors'][d]
            assignments.append(tuple(colors))
        if len(set(assignments)) != 1:
            raise ValueError("Ambiguous compartment correspondence")
        mappings[i] = possibilities[0]
        for room, color in zip(target['rooms'], assignments[0]):
            for p in room:
                output[p] = color
    return output, dict(bg=bg, wall=wall, objects=objects, wires=wires,
                        graph=graph, owners=owners, mappings=mappings, groups=groups)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the full output entirely from the visible input."""
    return _decode(input_grid)[0]


def _ri(rng, low, high):
    return int(rng.integers(low, high + 1))


def _pick(rng, values):
    return values[_ri(rng, 0, len(values) - 1)]


def _plan(rng, count):
    shapes = [(2, 2), (2, 3), (3, 2), (3, 3)]
    shapes = [s for s in shapes if count <= s[0] * s[1] <= count + 3]
    h, w = _pick(rng, shapes)
    plan = np.arange(h * w).reshape(h, w)
    # Missing corners make nonrectangular framework footprints, without adding
    # spurious enclosed exterior regions.
    for r, c in rng.permutation([(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]):
        if np.sum(plan >= 0) > count and rng.random() < .45:
            plan[r, c] = -1
    while len(set(plan.flat) - {-1}) > count:
        edges = [(int(plan[r, c]), int(plan[r + dr, c + dc]))
                 for r in range(h) for c in range(w) for dr, dc in ((1, 0), (0, 1))
                 if r + dr < h and c + dc < w and plan[r, c] >= 0
                 and plan[r + dr, c + dc] >= 0
                 and plan[r, c] != plan[r + dr, c + dc]]
        if not edges:
            return _plan(rng, count)
        first, second = _pick(rng, edges)
        plan[plan == second] = first
    old = sorted(set(plan.flat) - {-1})
    original = plan.copy()
    for label, value in enumerate(old):
        plan[original == value] = label
    return plan


def _shell(labels, square_corners=False):
    cells = _cells(labels >= 0)
    shell = _near(cells, labels.shape, D8)
    if not square_corners:
        orthogonal = _near(cells, labels.shape, D4)
        shell = {p for p in shell if p in orthogonal or sum(
            (p[0] + dr, p[1] + dc) in cells
            for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1))) >= 2}
    return shell


def _label_graph(labels):
    rooms = [_cells(labels == i) for i in range(int(labels.max()) + 1)]
    return _adjacency(rooms, _shell(labels))


def _geometry(rng, plan, compact=False, trim=None, fallback=False):
    h, w = plan.shape
    heights = rng.integers(1 if compact else 2, 3 if compact else 4, size=h)
    widths = rng.integers(1 if compact else 2, 3 if compact else 4, size=w)
    if fallback:
        widths[:] = 1
        heights[:] = 2
    ys, xs = np.r_[0, np.cumsum(heights + 1)], np.r_[0, np.cumsum(widths + 1)]
    labels = np.full((int(ys[-1]) + 1, int(xs[-1]) + 1), -1, dtype=int)
    for r in range(h):
        for c in range(w):
            v = plan[r, c]
            if v < 0:
                continue
            labels[ys[r] + 1:ys[r + 1], xs[c] + 1:xs[c + 1]] = v
            if r and plan[r - 1, c] == v:
                labels[ys[r], xs[c] + 1:xs[c + 1]] = v
            if c and plan[r, c - 1] == v:
                labels[ys[r] + 1:ys[r + 1], xs[c]] = v
            if r and c and np.all(plan[r - 1:r + 1, c - 1:c + 1] == v):
                labels[ys[r], xs[c]] = v
    base_graph = _label_graph(labels)
    if trim is not None:
        row = min(r for r, c in _cells(labels == trim))
        trial = labels.copy()
        trial[row, trial[row] == trim] = -1
        if (not np.any(trial == trim) or len(_components(_cells(trial == trim))) != 1
                or _label_graph(trial) != base_graph):
            raise ValueError("Requested trim erases a shared-wall landmark")
        labels = trial
    # Independent local deformation keeps the directed room adjacency intact.
    for _ in range(_ri(rng, 0, 6)):
        v = _ri(rng, 0, int(plan.max()))
        cells = _cells(labels == v)
        if len(cells) <= 2:
            continue
        r, c, R, C = _bbox(cells)
        p = _pick(rng, [(r, c), (r, C - 1), (R - 1, c), (R - 1, C - 1)])
        if p not in cells:
            continue
        trial = labels.copy()
        trial[p] = -1
        if len(_components(cells - {p})) == 1 and _label_graph(trial) == base_graph:
            labels = trial
    walls = _shell(labels, rng.random() < .3)
    body = _cells(labels >= 0) | walls
    r, c, R, C = _bbox(body)
    labels = labels[r:R, c:C].copy()
    walls = {(y - r, x - c) for y, x in walls}
    # Negative cells are exterior; -2 denotes the framework.
    for p in walls:
        labels[p] = -2
    if len(_components(walls, D8)) != 1:
        raise ValueError("Disconnected framework")
    rooms = _enclosed(walls)
    if len(rooms) != int(plan.max()) + 1 or set(map(frozenset, rooms)) != {
            frozenset(_cells(labels == i)) for i in range(int(plan.max()) + 1)}:
        raise ValueError("Unexpected cavity after deformation")
    return labels


def _trim_pair(plan):
    boxes = [_bbox(_cells(plan == i)) for i in range(int(plan.max()) + 1)]
    pairs = [(i, j) for i, a in enumerate(boxes) for j, b in enumerate(boxes)
             if i != j and a[0] == b[0] and a[3] <= b[1]]
    if not pairs:
        raise ValueError("No neighboring row-start landmarks for the lesson")
    return pairs[0]


def _place(rng, labels, shape, fallback=False):
    """Free masked placement, including overlapping exterior bounding boxes."""
    occupied = set()
    placements = [None] * len(labels)
    h, w = shape
    if fallback:
        slots = [(1 + 14 * r, 1 + 10 * c) for r in range(2) for c in range(3)]
        rng.shuffle(slots)
    order = sorted(range(len(labels)), key=lambda i: -np.sum(labels[i] != -1))
    for i in order:
        footprint = _cells(labels[i] != -1)
        lh, lw = labels[i].shape
        blocked = occupied | _near(occupied, shape)
        for _ in range(180):
            if fallback:
                r, c = slots[i]
            else:
                if h - lh - 1 < 1 or w - lw - 1 < 1:
                    raise ValueError("Packing failed")
                r, c = _ri(rng, 1, h - lh - 1), _ri(rng, 1, w - lw - 1)
            cells = {(y + r, x + c) for y, x in footprint}
            if max(y for y, x in cells) >= h - 1 or max(x for y, x in cells) >= w - 1:
                raise ValueError("Slot is too small")
            if cells & blocked:
                continue
            placements[i] = r, c
            occupied |= cells
            break
        else:
            raise ValueError("Packing failed")
    return placements


def _path(rng, starts, goals, allowed):
    starts = sorted(starts & allowed)
    rng.shuffle(starts)
    queue = deque(starts)
    previous = {p: None for p in starts}
    directions = [D8[int(i)] for i in rng.permutation(8)]
    end = None
    while queue:
        p = queue.popleft()
        if p in goals:
            end = p
            break
        for dr, dc in directions:
            q = p[0] + dr, p[1] + dc
            if q in allowed and q not in previous:
                previous[q] = p
                queue.append(q)
    if end is None:
        raise ValueError("Routing failed")
    path = set()
    while end is not None:
        path.add(end)
        end = previous[end]
    return path


def _route(rng, bodies, walls, placements, sources, owners, shape, inactive,
           force_contact=False, force_relay=False):
    canvas = {(r, c) for r in range(1, shape[0] - 1) for c in range(1, shape[1] - 1)}
    allbody = set().union(*bodies)
    allwall = set().union(*walls)
    ports = [_near(wall, shape) - allbody - _near(allwall - wall, shape)
             for wall in walls]
    allowed_base = canvas - allbody - _near(allwall, shape)
    wire_sets = []
    for source in range(sources):
        targets = [i for i, owner in owners.items() if owner == source]
        rng.shuffle(targets)
        participants = [source] + targets
        allowed = allowed_base | set().union(*(ports[i] for i in participants))
        used = set().union(*wire_sets) if wire_sets else set()
        allowed -= used
        tree, previous_obj = set(), source
        connected = [source]
        serial = force_relay or rng.random() < .5
        for target in targets:
            starts = ports[previous_obj] if serial else tree | set().union(
                *(ports[i] for i in connected))
            connection_allowed = allowed
            if force_relay and tree:
                # Leave a receiver from another side, with a genuinely separate
                # wire component. Only the framework node joins both fragments.
                connection_allowed = allowed - tree - _near(tree, shape)
                starts &= connection_allowed
            path = _path(rng, starts, ports[target], connection_allowed)
            tree |= path
            previous_obj = target
            connected.append(target)
        wire_sets.append(tree)
    if inactive:
        targets = [i for i, owner in owners.items() if owner is None]
        allowed = allowed_base | set().union(*(ports[i] for i in targets))
        allowed -= set().union(*wire_sets)
        target = targets[0]
        # The open string has no multicolored reservoir and transmits no paint.
        candidates = allowed_base & allowed
        if not candidates:
            raise ValueError("No inactive tail space")
        port = _pick(rng, sorted(ports[target] & allowed))
        goals = {p for p in candidates if max(abs(p[0] - port[0]), abs(p[1] - port[1])) >= 4}
        tail = _path(rng, {port}, goals, allowed)
        for other in targets[1:]:
            tail |= _path(rng, tail | ports[target], ports[other], allowed)
        wire_sets.append(tail)
    if force_contact and len(wire_sets) >= 2:
        # Add a harmless branch to exhibit color-preserving wire contact.
        if not wire_sets[0] & _near(wire_sets[1], shape):
            for fragment in _components(wire_sets[0], D8):
                others = wire_sets[0] - fragment
                allowed = (allowed_base - set().union(*wire_sets[1:])
                           - others - _near(others, shape))
                try:
                    branch = _path(rng, fragment, _near(wire_sets[1], shape) & allowed,
                                   allowed | fragment)
                except ValueError:
                    continue
                wire_sets[0] |= branch
                break
            else:
                raise ValueError("Cannot demonstrate contact while preserving relay")
    if any(not wire for wire in wire_sets):
        raise ValueError("Missing productive route")
    return wire_sets


def _example(rng, room_count, sources, target_counts, detached, lesson=None,
             force_contact=False, attempt=0):
    fallback = attempt >= 65
    compact = attempt >= 12
    if fallback:
        # A grid partition is only a bounded packing fallback, not the family
        # vocabulary. All examples retain the same room/route semantics.
        plan = np.arange(6).reshape(2, 3)
        if room_count == 3:
            plan = np.array([[0, 1], [2, 2]])
        elif room_count == 4:
            plan = np.arange(4).reshape(2, 2)
        elif room_count == 5:
            plan[plan == 5] = 4
    else:
        plan = _plan(rng, room_count)
    trim_left, trim_right = _trim_pair(plan) if lesson == 'scan' else (None, None)
    # Keep room heights at least two for forced opposing trims; fallback widths
    # are one to fit three columns of objects in a 30-cell canvas.
    count = sources + sum(target_counts) + detached
    labels = []
    for i in range(count):
        trim = trim_left if lesson == 'scan' and i < sources else (
            trim_right if lesson == 'scan' and i == sources else None)
        label = _geometry(rng, plan, compact and trim is None, trim, fallback)
        if fallback and label.shape[1] > 8:
            raise ValueError("Compact fallback width exceeded")
        labels.append(label)
    if fallback:
        shape = (30, 30)
    else:
        area = sum(np.count_nonzero(g != -1) for g in labels)
        low = max(18, int(np.ceil(np.sqrt(area * 2.2))))
        low = min(low, 28)
        shape = (_ri(rng, low, 30), _ri(rng, low, 30))
    places = _place(rng, labels, shape, fallback)
    bodies, walls = [], []
    for g, (r, c) in zip(labels, places):
        bodies.append({(y + r, x + c) for y, x in _cells(g != -1)})
        walls.append({(y + r, x + c) for y, x in _cells(g == -2)})
    targets = list(range(sources, count - detached))
    owners = {i: None for i in range(count - detached, count)}
    assignments = [s for s, total in enumerate(target_counts) for _ in range(total)]
    rng.shuffle(assignments)
    if lesson == 'nearest':
        alternatives = sorted(set(itertools.permutations(assignments)))
        scores = []
        for assignment in alternatives:
            score = sum(np.sum((_center(bodies[t]) - _center(bodies[s])) ** 2) >
                        min(np.sum((_center(bodies[t]) - _center(bodies[q])) ** 2)
                            for q in range(sources)) + 1e-6
                        for t, s in zip(targets, assignment))
            scores.append(score)
        best = max(scores)
        if not best:
            raise ValueError("Nearest-donor contrast is not visible")
        assignments = alternatives[scores.index(best)]
    owners.update(zip(targets, assignments))
    palette = list(map(int, rng.permutation(10)))
    bg, wall = palette[:2]
    inactive = bool(detached and (lesson == 'scan' or rng.random() < .6))
    nwires = sources + int(inactive)
    wire_colors = palette[2:2 + nwires]
    pool = palette[2 + nwires:]
    richness = _ri(rng, 2, min(len(pool), room_count))
    pattern = pool[:richness] + [_pick(rng, pool[:richness]) for _ in range(room_count - richness)]
    rng.shuffle(pattern)
    # A cyclic color substitution, not a room permutation: every corresponding
    # room differs between the two visible competing donors.
    patterns = [pattern]
    if sources == 2:
        color_cycle = {pool[i]: pool[(i + 1) % richness] for i in range(richness)}
        patterns.append([color_cycle[c] for c in pattern])
    a = np.full(shape, bg, dtype=np.int8)
    for i, (g, (r, c)) in enumerate(zip(labels, places)):
        region = a[r:r + g.shape[0], c:c + g.shape[1]]
        region[g == -2] = wall
        if i < sources:
            for v, color in enumerate(patterns[i]):
                region[g == v] = color
    wire_sets = _route(rng, bodies, walls, places, sources, owners, shape, inactive,
                       force_contact, lesson == 'scan')
    for wire, color in zip(wire_sets, wire_colors):
        for p in wire:
            a[p] = color
    expected = a.copy()
    for i, owner in owners.items():
        if owner is None:
            continue
        g, (r, c) = labels[i], places[i]
        region = expected[r:r + g.shape[0], c:c + g.shape[1]]
        for v, color in enumerate(patterns[owner]):
            region[g == v] = color
    # Rotate/reflect the entire scene together. Matching remains in the common
    # visible orientation, not a puzzle-specific inferred rotation convention.
    rotation, reflected = _ri(rng, 0, 3), rng.random() < .5
    a, expected = np.rot90(a, rotation), np.rot90(expected, rotation)
    if reflected:
        a, expected = a[:, ::-1], expected[:, ::-1]
    a, expected = a.copy(), expected.copy()
    prediction, info = _decode(a)
    if not np.array_equal(prediction, expected):
        raise ValueError("Visible reconstruction disagrees with construction")
    pair = dict(input=a, output=expected)
    features = _measure(pair, info)
    if lesson == 'scan' and not features['scan_shortcut_wrong']:
        raise ValueError("Scan-order contrast not exposed")
    if lesson == 'scan' and not features['relay_required']:
        raise ValueError("Receiver relay not exposed")
    if lesson == 'nearest' and not features['nearest_shortcut_wrong']:
        raise ValueError("Nearest-donor contrast not exposed")
    if force_contact and not features['wire_contact']:
        raise ValueError("Wire-color contact not exposed")
    return pair


def _integer(value):
    return isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_))


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError("Counts must satisfy 2..10 train, 1..4 test, train > test")
    rng = np.random.default_rng(int(seed))
    shift = _pick(rng, ['rooms', 'fanout'])
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, total in [('train', num_train), ('test', num_test)]:
        for index in range(total):
            if split == 'train' and index == 0:
                sources, counts, detached, lesson = 2, [1, 1], 0, 'nearest'
            elif split == 'train' and index == 1:
                sources, counts, detached, lesson = 1, [2], 1, 'scan'
            elif split == 'test' and shift == 'fanout':
                sources, counts, detached, lesson = 1, [3], _ri(rng, 0, 1), None
            else:
                sources = _ri(rng, 1, 2)
                counts = [_ri(rng, 1, 2) for _ in range(sources)]
                detached = _ri(rng, 0, min(1, 6 - sources - sum(counts)))
                lesson = None
            room_count = _ri(rng, 3, 4) if shift == 'rooms' and split == 'train' else (
                _ri(rng, 5, 6) if shift == 'rooms' else _ri(rng, 3, 6))
            pair = None
            last_error = None
            for attempt in range(120):
                try:
                    candidate = _example(rng, room_count, sources, counts, detached,
                                         lesson, split == 'train' and index < 2, attempt)
                    if _key(candidate['input']) in seen:
                        continue
                    pair = candidate
                    break
                except (ValueError, IndexError) as error:
                    last_error = str(error)
            if pair is None:
                raise RuntimeError(f"Bounded construction exhausted: {last_error}")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    return puzzle


def _measure(pair, info=None):
    a = np.asarray(pair['input'])
    if info is None:
        _, info = _decode(a)
    objects, owners = info['objects'], info['owners']
    sources = [i for i, obj in enumerate(objects) if obj['donor']]
    supplied = [i for i, source in owners.items() if source is not None]
    nearest_wrong = scan_wrong = False
    signs = set()
    for i in supplied:
        source = owners[i]
        d, t = objects[source], objects[i]
        distances = {s: np.sum((_center(objects[s]['walls']) - _center(t['walls'])) ** 2)
                     for s in sources}
        if distances[source] > min(distances.values()) + 1e-6:
            nearest_wrong = True
        mapping = info['mappings'][i]
        assigned = [None] * len(mapping)
        for j, k in enumerate(mapping):
            assigned[k] = d['colors'][j]
            signs.add(int(np.sign(len(t['rooms'][k]) - len(d['rooms'][j]))))
        scan_wrong |= assigned != d['colors']
    wire_contact = any(color != other_color and cells & _near(other, a.shape)
                       for i, (cells, color) in enumerate(info['wires'])
                       for other, other_color in info['wires'][:i])
    return dict(rooms=len(objects[0]['rooms']), donors=len(sources),
                supplied=len(supplied), unsupplied=sum(s is None for s in owners.values()),
                max_fanout=max(sum(s == source for s in owners.values()) for source in sources),
                nearest_shortcut_wrong=bool(nearest_wrong), scan_shortcut_wrong=bool(scan_wrong),
                wire_contact=bool(wire_contact),
                relay_targets=sum(len(info['graph'][i]) >= 2 for i in supplied),
                relay_required=sum(not (info['graph'][i] & info['graph'][owners[i]])
                                   for i in supplied),
                branch_wires=sum(len(info['graph'][j]) >= 3 for j in range(len(objects), len(info['graph']))),
                wire_colors=len({c for cells, c in info['wires']}),
                diagonal_ports=sum(not (cells & _near(objects[i]['walls'], a.shape, D4))
                                   for j, (cells, color) in enumerate(info['wires'], len(objects))
                                   for i in info['graph'][j]),
                room_area_signs=sorted(signs), background=info['bg'], wall=info['wall'],
                shapes=[list(a.shape)],
                nonrectangular_rooms=sum(len(room) != (_bbox(room)[2] - _bbox(room)[0]) *
                                        (_bbox(room)[3] - _bbox(room)[1])
                                        for obj in objects for room in obj['rooms']),
                floorplan=objects[0]['graph'],
                changed_cells=int(np.count_nonzero(a != np.asarray(pair['output']))))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, measurements = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            prediction, info = _decode(a)
            assert np.array_equal(prediction, b), "Input-derived answer mismatch"
            mask = np.zeros(a.shape, dtype=bool)
            for i, source in info['owners'].items():
                if source is None:
                    continue
                obj = info['objects'][i]
                fills = set()
                for room in obj['rooms']:
                    values = {int(b[p]) for p in room}
                    assert len(values) == 1
                    fills |= values
                    for p in room:
                        mask[p] = True
                assert len(fills) >= 2, "A routed payload must remain multicolored"
            assert np.any(mask) and np.array_equal(a != b, mask)
            assert np.all(a[mask] == info['bg'])
            assert np.array_equal(a[~mask], b[~mask]), "Walls, wires, donors, exterior must remain"
            measurements[split].append(_measure(pair, info))
    # These semantic contrasts survive the minimum two-demonstration request.
    train = measurements['train']
    assert any(m['nearest_shortcut_wrong'] and m['donors'] == 2 for m in train)
    assert any(m['scan_shortcut_wrong'] and m['max_fanout'] >= 2
               and m['unsupplied'] and m['relay_required'] for m in train)
    assert all(m['wire_contact'] for m in train[:2])
    assert len({_key(pair['input']) for pair in puzzle['train']}) >= 2
    tests = measurements['test']
    assert (all(m['rooms'] > max(t['rooms'] for t in train) for m in tests)
            or all(m['max_fanout'] > max(t['max_fanout'] for t in train) for m in tests))


def self_test():
    """Bounded 42-seed semantic/diversity audit, counts, and metamorphic checks."""
    metrics, structures, shapes, puzzles = [], set(), set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = pair['input']
                metrics.append(_measure(pair))
                shapes.add(a.shape)
                roles, normalized = {}, []
                for color in a.flat:
                    normalized.append(roles.setdefault(int(color), len(roles)))
                structures.add((a.shape, tuple(normalized)))
    assert len(structures) >= 150 and len(shapes) >= 25
    assert {m['rooms'] for m in metrics} == {3, 4, 5, 6}
    assert {m['donors'] for m in metrics} == {1, 2}
    assert {1, 2, 3} <= {m['max_fanout'] for m in metrics}
    assert len({m['floorplan'] for m in metrics}) >= 15
    assert len({m['background'] for m in metrics}) == 10
    assert len({m['wall'] for m in metrics}) == 10
    assert sum(m['nonrectangular_rooms'] > 0 for m in metrics) >= 100
    assert sum(m['relay_targets'] > 0 for m in metrics) >= 5
    assert sum(m['relay_required'] > 0 for m in metrics) >= 42
    assert sum(m['branch_wires'] > 0 for m in metrics) >= 5
    assert sum(m['diagonal_ports'] > 0 for m in metrics) >= 5
    assert any(-1 in m['room_area_signs'] and 1 in m['room_area_signs'] for m in metrics)
    again = generate_puzzle(0)
    assert all(_key(p[k]) == _key(q[k]) for s in ('train', 'test')
               for p, q in zip(puzzles[0][s], again[s]) for k in ('input', 'output'))
    for nt, ne in [(2, 1), (10, 1), (10, 4), (5, 3)]:
        validate_generated_puzzle(generate_puzzle(119, nt, ne))
    for nt, ne in [(1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.0, 1), (4, 1.5), (None, 1)]:
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    pair = puzzles[0]['train'][0]
    # Color identities and common orientation are equivariant, including a
    # permutation that changes the background's literal identity.
    recolor = np.random.default_rng(507).permutation(10)
    assert np.array_equal(solve(recolor[pair['input']]), recolor[pair['output']])
    for rotation in range(4):
        assert np.array_equal(solve(np.rot90(pair['input'], rotation)),
                              np.rot90(pair['output'], rotation))
    assert np.array_equal(solve(pair['input'][:, ::-1]), pair['output'][:, ::-1])
    # A single room is one address in the broadcast payload. Changing it must
    # affect exactly that address at the reservoir and all its receivers.
    _, info = _decode(pair['input'])
    source = next(i for i, obj in enumerate(info['objects']) if obj['donor'])
    new_color = next(c for c in range(10) if c not in pair['input'])
    changed_input, changed_expected = pair['input'].copy(), pair['output'].copy()
    for p in info['objects'][source]['rooms'][0]:
        changed_input[p] = changed_expected[p] = new_color
    for i, owner in info['owners'].items():
        if owner == source:
            room = info['objects'][i]['rooms'][info['mappings'][i][0]]
            for p in room:
                changed_expected[p] = new_color
    assert np.array_equal(solve(changed_input), changed_expected)
    assert not np.array_equal(changed_expected, pair['output'])
    # Cutting ALL wire pixels disables every receiver without recoloring donors.
    cut = pair['input'].copy()
    for cells, color in info['wires']:
        for p in cells:
            cut[p] = info['bg']
    # Decode requires at least one wire; leave an isolated open three-cell tail.
    cut[0, 1:4] = info['wires'][0][1]
    assert np.array_equal(solve(cut), cut)
    assert not np.array_equal(cut, pair['output'])
    # Ignoring receiver nodes must miss a downstream target in the relay lesson.
    relay_pair = puzzles[0]['train'][1]
    correct, relay_info = _decode(relay_pair['input'])
    wire_only = relay_pair['input'].copy()
    for target, donor in relay_info['owners'].items():
        if donor is not None and relay_info['graph'][target] & relay_info['graph'][donor]:
            for room in relay_info['objects'][target]['rooms']:
                for p in room:
                    wire_only[p] = correct[p]
    assert not np.array_equal(wire_only, correct)
    # Validator must reject a partial fill and a preserved-cell corruption.
    for changed in (True, False):
        bad = {s: [{k: np.asarray(v).copy() for k, v in q.items()} for q in puzzles[0][s]]
               for s in ('train', 'test')}
        p = bad['test'][0]
        cell = tuple(np.argwhere((p['input'] != p['output']) == changed)[0])
        p['output'][cell] = p['input'][cell] if changed else (int(p['output'][cell]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("Corrupted output accepted")
    return dict(seeds=42, pairs=len(metrics), distinct_structures=len(structures),
                grid_shapes=len(shapes), floorplan_graphs=len({m['floorplan'] for m in metrics}),
                room_counts=sorted({m['rooms'] for m in metrics}),
                nearest_contrasts=sum(m['nearest_shortcut_wrong'] for m in metrics),
                scan_contrasts=sum(m['scan_shortcut_wrong'] for m in metrics),
                relays=sum(m['relay_targets'] > 0 for m in metrics),
                essential_relays=sum(m['relay_required'] > 0 for m in metrics),
                branches=sum(m['branch_wires'] > 0 for m in metrics),
                diagonal_ports=sum(m['diagonal_ports'] > 0 for m in metrics))


if __name__ == '__main__':
    print(self_test())
