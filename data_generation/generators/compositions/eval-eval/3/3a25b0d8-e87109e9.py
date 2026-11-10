"""Selective compartment transfer driven by a keyed, relative-turn room walk.

Only standard library and NumPy. No parent module, saved example, seed, or
construction label is accessed by solve(). Directions are N,E,S,W = 0,1,2,3.
"""
from __future__ import annotations

from collections import Counter
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(np.all((a >= 0) & (a <= 9)), "colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    todo = set(map(tuple, np.argwhere(mask)))
    parts = []
    steps = DIRS + (((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else ())
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, points = [start], []
        while stack:
            r, c = stack.pop()
            points.append((r, c))
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in todo:
                    todo.remove(p)
                    stack.append(p)
        parts.append(np.array(points, dtype=int))
    return parts


def _bbox(points):
    return tuple(map(int, points.min(0))) + tuple(map(int, points.max(0) + 1))


def _rooms(barriers):
    free = np.pad(~barriers, 1, constant_values=True)
    limit = np.array(free.shape) - 1
    return [p - 1 for p in _components(free)
            if np.all(p > 0) and np.all(p < limit)]


def _relations(rooms):
    boxes = [_bbox(p) for p in rooms]
    return [[tuple(-1 if a[k + 2] <= b[k] else
                   1 if b[k + 2] <= a[k] else 0 for k in (0, 1))
             for b in boxes] for a in boxes]


def _matching(donor_rooms, target_rooms, colors):
    """Exact complete spatial-graph isomorphism, with color agreement on ties."""
    a, b = _relations(donor_rooms), _relations(target_rooms)
    n = len(a)
    _require(n == len(b) and n >= 2, "equal compartment counts required")
    profiles_a = [Counter(row) for row in a]
    profiles_b = [Counter(row) for row in b]
    candidates = [[j for j in range(n) if profiles_a[i] == profiles_b[j]] for i in range(n)]
    answers, visits = [], 0

    def search(mapping, used):
        nonlocal visits
        visits += 1
        _require(visits <= 20000, "spatial correspondence is insufficiently distinct")
        if len(mapping) == n:
            assigned = [None] * n
            for i, j in mapping.items():
                assigned[j] = colors[i]
            if answers:
                _require(assigned == answers[0][0], "ambiguous donor colors")
            else:
                answers.append((assigned, dict(mapping)))
            return
        options = {i: [j for j in candidates[i] if j not in used and all(
            a[i][k] == b[j][v] and a[k][i] == b[v][j] for k, v in mapping.items())]
                   for i in range(n) if i not in mapping}
        i = min(options, key=lambda x: (len(options[x]), x))
        for j in options[i]:
            mapping[i] = j
            search(mapping, used | {j})
            del mapping[i]

    search({}, set())
    _require(bool(answers), "no orientation-preserving spatial correspondence")
    return answers[0]


def _header_info(grid):
    wall = int(grid[0, 0])
    _require(np.all(grid[0] == wall), "missing key top frame")
    bottoms = np.flatnonzero(np.all(grid[1:] == wall, axis=1)) + 1
    _require(len(bottoms) > 0, "missing key bottom frame")
    bottom = int(bottoms[0])
    _require(2 <= bottom < len(grid) - 1, "key height")
    interior = grid[1:bottom]
    _require(np.all(interior == interior[0]), "key bars must be vertical")
    row = interior[0]
    _require(row[0] == row[-1] == wall, "key side frames")
    runs, c = [], 0
    while c < len(row):
        if row[c] == wall:
            c += 1
        else:
            a = c
            while c < len(row) and row[c] != wall:
                c += 1
            runs.append((a, c))
    _require(len(runs) >= 2 and all(b - a >= 3 for a, b in runs), "key panels")
    bg = int(row[runs[0][0] + 1])
    turns = {}
    for a, b in runs:
        _require(np.all(row[a + 1:b - 1] == bg), "key empty interiors")
        left, right = int(row[a]), int(row[b - 1])
        _require((left == bg) != (right == bg), "one side bar per panel")
        color, turn = (right, 1) if left == bg else (left, -1)
        _require(color not in turns and color not in (bg, wall), "unique key color")
        turns[color] = turn
    return bottom + 1, bg, wall, turns


def _ports(target, rooms, wall, gate):
    """A marked wall cell joins opposite rooms, or one room to the exterior."""
    labels = np.full(target.shape, -1, dtype=int)
    labels[(target == wall) | (target == gate)] = -2
    for i, p in enumerate(rooms):
        labels[tuple(p.T)] = i
    padded = np.pad(labels, 1, constant_values=-1)
    edges = {i: {} for i in range(len(rooms))}
    occupied, entrances, records = set(), [], []
    for rr, cc in np.argwhere(target == gate):
        r, c = int(rr), int(cc)
        neighbors = [int(padded[r + 1 + dr, c + 1 + dc]) for dr, dc in DIRS]
        axes = [d for d in (0, 1) if neighbors[d] >= -1 and neighbors[d + 2] >= -1
                and neighbors[(d + 1) % 4] == neighbors[(d + 3) % 4] == -2
                and (neighbors[d] >= 0 or neighbors[d + 2] >= 0)]
        _require(len(axes) == 1, "gate must cross exactly one wall face")
        d = axes[0]
        a, b = neighbors[d], neighbors[d + 2]
        if a >= 0 and b >= 0:
            _require(a != b, "gate must join different rooms")
            uses = ((b, d), (a, (d + 2) % 4))
            for use in uses:
                _require(use not in occupied, "multiple doors on one room side")
                occupied.add(use)
            edges[b][d] = a
            edges[a][(d + 2) % 4] = b
            records.append((r, c, b, d, a))
        else:
            room, heading = (a, d) if a >= 0 else (b, (d + 2) % 4)
            side = room, (heading + 2) % 4
            _require(side not in occupied, "entrance conflicts with an internal door")
            occupied.add(side)
            entrances.append((room, heading))
    _require(len(entrances) == 1, "exactly one exterior entrance")
    return edges, entrances[0], records


def _decode(input_grid):
    grid = _grid(input_grid)
    crop, bg, wall, turns = _header_info(grid)
    field = grid[crop:]
    other = set(map(int, np.unique(field))) - {bg, wall} - set(turns)
    _require(len(other) == 1, "unique unkeyed door color")
    gate = other.pop()
    barrier = (field == wall) | (field == gate)
    parts = _components(barrier, diagonal=True)
    _require(len(parts) == 2, "two separate frameworks required")
    donor_info, target_info = [], []
    covered = barrier.copy()
    for part in parts:
        r, c, R, C = _bbox(part)
        obj = np.full((R - r, C - c), bg, dtype=np.int8)
        obj[part[:, 0] - r, part[:, 1] - c] = field[tuple(part.T)]
        rooms = _rooms((obj == wall) | (obj == gate))
        _require(len(rooms) >= 2, "framework must enclose compartments")
        raw = field[r:R, c:C]
        values = []
        for p in rooms:
            colors = np.unique(raw[tuple(p.T)])
            _require(len(colors) == 1, "room interiors must be uniform")
            values.append(int(colors[0]))
            obj[tuple(p.T)] = colors[0]
            covered[p[:, 0] + r, p[:, 1] + c] = True
        if np.any(obj == gate):
            _require(set(values) == {bg}, "target rooms must be blank")
            target_info.append((obj, rooms, (r, c, R, C)))
        else:
            _require(set(values) <= set(turns) and len(set(values)) >= 2,
                     "donor room colors must be keyed and multicolored")
            donor_info.append((obj, rooms, values, (r, c, R, C)))
    _require(len(donor_info) == len(target_info) == 1, "donor and target roles")
    _require(np.all((field == bg) | covered), "unrelated foreground outside the frameworks")
    donor, dr, colors, db = donor_info[0]
    target, tr, tb = target_info[0]
    assigned, matching = _matching(dr, tr, colors)
    edges, start, records = _ports(target, tr, wall, gate)
    return dict(bg=bg, wall=wall, gate=gate, turns=turns, donor=donor, target=target,
                donor_rooms=dr, rooms=tr, colors=assigned, donor_colors=colors,
                matching=matching, edges=edges, start=start, ports=records,
                donor_box=db, target_box=tb, header_height=crop)


def _walk(colors, turns, edges, start, glyph_map=None, straight=False):
    room, heading = start
    seen, visited, events = set(), set(), []
    while (room, heading) not in seen:
        seen.add((room, heading))
        visited.add(room)
        turn = turns[colors[room]]
        if glyph_map is not None:
            turn = glyph_map[turn]
        outgoing = heading if straight else (heading + turn) % 4
        next_room = edges[room].get(outgoing)
        events.append((room, heading, outgoing, next_room, turn))
        if next_room is None:
            return visited, events, "closed"
        room, heading = next_room, outgoing
    return visited, events, "repeat"


def _answer(info, colors=None, glyph_map=None, straight=False):
    colors = info['colors'] if colors is None else colors
    visited, events, end = _walk(colors, info['turns'], info['edges'], info['start'],
                                 glyph_map, straight)
    output = info['target'].copy()
    for i in visited:
        output[tuple(info['rooms'][i].T)] = colors[i]
    return output, visited, events, end


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read only the supplied input, match compartments, and activate the walk."""
    return _answer(_decode(input_grid))[0]


def _plan(rng, large=False):
    shapes = [(2, 3), (3, 2), (3, 3), (2, 4), (4, 2), (3, 4), (4, 3), (4, 4)]
    if large:
        shapes = [(3, 3), (3, 4), (4, 3), (4, 4)]
    rows, cols = shapes[int(rng.integers(len(shapes)))]
    plan = np.arange(rows * cols).reshape(rows, cols)
    for r, c in rng.permutation(np.array([(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)])):
        if np.count_nonzero(plan >= 0) > 6 and rng.random() < .3:
            plan[r, c] = -1
    count = int(rng.integers(6 if large else 4, min(12, np.count_nonzero(plan >= 0)) + 1))
    while len(set(plan.flat) - {-1}) > count:
        edges = [(int(plan[r, c]), int(plan[r + dr, c + dc]))
                 for r in range(rows) for c in range(cols) for dr, dc in ((1, 0), (0, 1))
                 if r + dr < rows and c + dc < cols and plan[r, c] >= 0
                 and plan[r + dr, c + dc] >= 0 and plan[r, c] != plan[r + dr, c + dc]]
        a, b = edges[int(rng.integers(len(edges)))]
        plan[plan == b] = a
    for i, old in enumerate(sorted(set(plan.flat) - {-1})):
        plan[plan == old] = i
    return plan


def _render(rng, plan, widths, heights, trim=True):
    ys = np.r_[0, np.cumsum(heights + 1)]
    xs = np.r_[0, np.cumsum(widths + 1)]
    labels = np.full((int(ys[-1]) + 1, int(xs[-1]) + 1), -1, dtype=int)
    for r, c in np.argwhere(plan >= 0):
        label = plan[r, c]
        labels[ys[r] + 1:ys[r + 1], xs[c] + 1:xs[c + 1]] = label
        if r and plan[r - 1, c] == label:
            labels[ys[r], xs[c] + 1:xs[c + 1]] = label
        if c and plan[r, c - 1] == label:
            labels[ys[r] + 1:ys[r + 1], xs[c]] = label
        if r and c and np.all(plan[r - 1:r + 1, c - 1:c + 1] == label):
            labels[ys[r], xs[c]] = label
    # Eight-neighbor walls enclose each room even around stepped contours.
    foreground = labels >= 0
    padded = np.pad(foreground, 1)
    near = np.zeros(labels.shape, dtype=bool)
    for dr in range(3):
        for dc in range(3):
            near |= padded[dr:dr + labels.shape[0], dc:dc + labels.shape[1]]
    labels[(labels < 0) & near] = -2
    if trim:
        for label in range(int(plan.max()) + 1):
            p = np.argwhere(labels == label)
            if len(p) < 3 or rng.random() > .45:
                continue
            a, b, A, B = _bbox(p)
            side = int(rng.integers(4))
            cut = (p[:, 0] == a if side == 0 else p[:, 1] == B - 1 if side == 1
                   else p[:, 0] == A - 1 if side == 2 else p[:, 1] == b)
            if rng.random() < .5:
                # A small corner notch, with no texture to stretch.
                chosen = p[int(rng.integers(len(p)))]
                cut = np.all(p == chosen, axis=1)
            candidate = labels == label
            candidate[tuple(p[cut].T)] = False
            if candidate.any() and len(_components(candidate)) == 1:
                labels[tuple(p[cut].T)] = -2
    a, b, A, B = _bbox(np.argwhere(labels != -1))
    return labels[a:A, b:B]


def _gate_candidates(labels):
    padded = np.pad(labels, 1, constant_values=-1)
    internal, external = [], []
    for rr, cc in np.argwhere(labels == -2):
        r, c = int(rr), int(cc)
        ns = [int(padded[r + 1 + dr, c + 1 + dc]) for dr, dc in DIRS]
        for d in (0, 1):
            a, b = ns[d], ns[d + 2]
            if ns[(d + 1) % 4] != -2 or ns[(d + 3) % 4] != -2:
                continue
            if a >= 0 and b >= 0 and a != b:
                internal.append((r, c, b, d, a))
            elif a >= 0 and b == -1:
                external.append((r, c, a, d))
            elif b >= 0 and a == -1:
                external.append((r, c, b, (d + 2) % 4))
    return internal, external


def _paint(labels, bg, wall, colors=None):
    a = np.full(labels.shape, bg, dtype=np.int8)
    a[labels == -2] = wall
    if colors is not None:
        for i, color in enumerate(colors):
            a[labels == i] = color
    return a


def _make_header(rng, width, height, bg, wall, turns):
    colors = list(map(int, rng.permutation(list(turns))))
    # Panels share a one-cell wall. Each has at least three interior columns.
    spans = np.full(len(colors), 4, dtype=int)
    spans += rng.multinomial(width - 1 - int(spans.sum()), np.full(len(colors), 1 / len(colors)))
    key = np.full((height, width), bg, dtype=np.int8)
    key[[0, -1]] = wall
    c = 0
    for color, span in zip(colors, spans):
        end = c + int(span)
        key[:, c] = key[:, end] = wall
        key[1:-1, c + 1 if turns[color] == -1 else end - 1] = color
        c = end
    return key


def _embed(rng, donor, target, bg, wall, turns):
    hh = int(rng.integers(3, 6))
    dh, dw = donor.shape
    th, tw = target.shape
    layouts = []
    if dw + tw + 2 <= 30 and max(dh, th) + hh + 1 <= 30:
        layouts.append('side')
    if dh + th + 2 + hh + 1 <= 30 and max(dw, tw) <= 30:
        layouts.append('stack')
    _require(bool(layouts), "frameworks do not fit")
    mode = layouts[int(rng.integers(len(layouts)))]
    first, second = (donor, target) if rng.random() < .5 else (target, donor)
    h1, w1 = first.shape
    h2, w2 = second.shape
    if mode == 'side':
        min_h, min_w = max(h1, h2), w1 + w2 + 2
    else:
        min_h, min_w = h1 + h2 + 2, max(w1, w2)
    min_w = max(min_w, 4 * len(turns) + 1)
    _require(min_w <= 30, "key does not fit")
    h = min_h + int(rng.integers(min(4, 30 - hh - 1 - min_h) + 1))
    w = min_w + int(rng.integers(min(4, 30 - min_w) + 1))
    field = np.full((h + 1, w), bg, dtype=np.int8)
    if mode == 'side':
        c1 = int(rng.integers(w - w1 - w2 - 2 + 1))
        c2 = c1 + w1 + 2
        r1, r2 = 1 + int(rng.integers(h - h1 + 1)), 1 + int(rng.integers(h - h2 + 1))
    else:
        r1 = 1 + int(rng.integers(h - h1 - h2 - 2 + 1))
        r2 = r1 + h1 + 2
        c1, c2 = int(rng.integers(w - w1 + 1)), int(rng.integers(w - w2 + 1))
    for obj, r, c in ((first, r1, c1), (second, r2, c2)):
        field[r:r + obj.shape[0], c:c + obj.shape[1]] = obj
    return np.vstack((_make_header(rng, w, hh, bg, wall, turns), field))


def _candidate(rng, large=False, lesson=False, allowed_headings=None, lesson_heading=1):
    if lesson:
        plan = np.arange(9).reshape(3, 3)
        d = _render(rng, plan, rng.integers(1, 3, 3), np.array([3, 2, 1]), trim=False)
        t = _render(rng, plan, rng.integers(1, 4, 3), np.array([3, 1, 2]), trim=False)
        # Stagger one room: donor discovery order is not target correspondence.
        p = np.argwhere(t == (2 if lesson_heading == 3 else 0))
        t[tuple(p[p[:, 0] == p[:, 0].min()].T)] = -2
    else:
        plan = _plan(rng, large)
        d = _render(rng, plan, rng.integers(1, 3, plan.shape[1]), rng.integers(1, 4, plan.shape[0]))
        t = _render(rng, plan, rng.integers(1, 4, plan.shape[1]), rng.integers(1, 4, plan.shape[0]))
    n = int(plan.max()) + 1
    palette = list(map(int, rng.permutation(10)))
    bg, wall, gate = palette[:3]
    richness = 3 if lesson else int(rng.integers(2, min(5, n) + 1))
    keys = palette[3:3 + richness]
    turns = {color: int(rng.choice([-1, 1])) for color in keys}
    turns[keys[0]], turns[keys[1]] = -1, 1
    if lesson:
        turns[keys[2]] = 1
    colors = keys + [int(rng.choice(keys)) for _ in range(n - richness)]
    rng.shuffle(colors)
    if lesson:
        # Entrance E; route 0,3,4,1,2,5. Both turns occur from two headings.
        colors = [keys[1], keys[2], keys[1], keys[0], keys[0], keys[0],
                  int(rng.choice(keys)), int(rng.choice(keys)), int(rng.choice(keys))]
        if lesson_heading == 3:
            d, t = d[:, ::-1].copy(), t[:, ::-1].copy()
            turns = {color: -turn for color, turn in turns.items()}
        elif lesson_heading in (0, 2):
            rotation = 1 if lesson_heading == 0 else 3
            d, t = np.rot90(d, rotation).copy(), np.rot90(t, rotation).copy()
    internal, external = _gate_candidates(t)
    edges = {i: {} for i in range(n)}
    chosen = []
    if lesson:
        # Two unused links attach unvisited rooms to the reached component.
        route_edges = [(0, 2, 3), (3, 1, 4), (4, 0, 1), (1, 1, 2), (2, 2, 5),
                       (5, 2, 8), (8, 3, 7)]
        for a, direction, b in route_edges:
            if lesson_heading == 3:
                direction = (-direction) % 4
            elif lesson_heading in (0, 2):
                direction = (direction - (1 if lesson_heading == 0 else 3)) % 4
            options = [x for x in internal if (x[2], x[3], x[4]) == (a, direction, b)
                       or (x[4], (x[3] + 2) % 4, x[2]) == (a, direction, b)]
            _require(bool(options), "lesson gate unavailable")
            chosen.append(options[int(rng.integers(len(options)))])
    else:
        for i in rng.permutation(len(internal)):
            record = internal[int(i)]
            _, _, a, direction, b = record
            if direction not in edges[a] and (direction + 2) % 4 not in edges[b] and rng.random() < .7:
                chosen.append(record)
                edges[a][direction] = b
                edges[b][(direction + 2) % 4] = a
    for _, _, a, direction, b in chosen:
        edges[a][direction] = b
        edges[b][(direction + 2) % 4] = a
    choices = [x for x in external if (x[3] + 2) % 4 not in edges[x[2]]
               and (allowed_headings is None or x[3] in allowed_headings)
               and (not lesson or x[2:] == (0, lesson_heading))]
    _require(bool(choices), "no entrance fits")
    entrance = choices[int(rng.integers(len(choices)))]
    visited, events, end = _walk(colors, turns, edges, entrance[2:])
    _require(len(visited) >= 3 and len(visited) < n and len({colors[i] for i in visited}) >= 2,
             "route must expose selective multicolor transfer")
    _require(end == 'closed', "route must terminate")
    donor, target = _paint(d, bg, wall, colors), _paint(t, bg, wall)
    for r, c, *_ in chosen + [entrance]:
        target[r, c] = gate
    expected = target.copy()
    for i in visited:
        expected[t == i] = colors[i]
    inp = _embed(rng, donor, target, bg, wall, turns)
    info = _decode(inp)
    _require(np.array_equal(_answer(info)[0], expected), "visible matching disagrees with construction")
    _require(not np.array_equal(_answer(info, straight=True)[0], expected), "turning must matter")
    return {'input': inp, 'output': expected}


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _teaching(pairs):
    headings = {-1: set(), 1: set()}
    fits = []
    color_sides, decoded = {}, []
    area_fail, scan_fail, connectivity_fail = False, False, False
    for pair in pairs:
        info = _decode(pair['input'])
        decoded.append(info)
        _, _, events, _ = _answer(info)
        for room, incoming, _, _, turn in events:
            headings[turn].add(incoming)
            color_sides.setdefault(info['colors'][room], set()).add(turn)
        n = len(info['rooms'])
        area_colors = [None] * n
        order_d = sorted(range(n), key=lambda i: (len(info['donor_rooms'][i]), i))
        order_t = sorted(range(n), key=lambda i: (len(info['rooms'][i]), i))
        for i, j in zip(order_d, order_t):
            area_colors[j] = info['donor_colors'][i]
        area_fail |= not np.array_equal(_answer(info, area_colors)[0], pair['output'])
        scan_fail |= not np.array_equal(_answer(info, info['donor_colors'])[0], pair['output'])
        reached, stack = {info['start'][0]}, [info['start'][0]]
        while stack:
            for neighbor in info['edges'][stack.pop()].values():
                if neighbor not in reached:
                    reached.add(neighbor)
                    stack.append(neighbor)
        connectivity_fail |= reached != _answer(info)[1]
    for left in (-1, 1):
        for right in (-1, 1):
            mapping = {-1: left, 1: right}
            if all(np.array_equal(_answer(info, glyph_map=mapping)[0], p['output'])
                   for info, p in zip(decoded, pairs)):
                fits.append(mapping)
    return (all(len(s) >= 2 for s in headings.values()) and fits == [{-1: -1, 1: 1}]
            and area_fail and scan_fail and connectivity_fail
            and any(len(sides) == 2 for sides in color_sides.values()))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, low, high, name in ((num_train, 2, 10, 'num_train'), (num_test, 1, 4, 'num_test')):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    # A held-out entrance side tests relative heading, not a new instruction.
    held_heading = int(rng.integers(4))
    train_headings = set(range(4)) - {held_heading}
    larger_test = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(count):
            pair = None
            for attempt in range(120):
                try:
                    lesson = ((split == 'train' and index == 1 and attempt >= 70)
                              or attempt >= 110)
                    heading = (1 if held_heading != 1 else 3) if split == 'train' else held_heading
                    candidate = _candidate(rng, large=split == 'test' and larger_test,
                                           lesson=lesson,
                                           allowed_headings=train_headings if split == 'train' else {held_heading},
                                           lesson_heading=heading)
                    if _signature(candidate['input']) in seen:
                        continue
                    if split == 'train' and index == 1 and not _teaching(puzzle['train'] + [candidate]):
                        continue
                    pair = candidate
                    break
                except AssertionError:
                    continue
            _require(pair is not None, f'bounded generation exhausted for {split}/{index}')
            puzzle[split].append(pair)
            seen.add(_signature(pair['input']))
    rng.shuffle(puzzle['train'])
    return puzzle


def _measure(pair):
    info = _decode(pair['input'])
    output, visited, events, end = _answer(info)
    expected = _grid(pair['output'])
    _require(np.array_equal(output, expected), "incorrect output")
    _require(4 <= len(info['rooms']) <= 12, "family room count")
    _require(end == 'closed', "nonterminating route")
    _require(3 <= len(visited) < len(info['rooms']), "nonselective room walk")
    _require(len({info['colors'][i] for i in visited}) >= 2, "monochrome transfer")
    interior = np.zeros(output.shape, dtype=bool)
    active = np.zeros(output.shape, dtype=bool)
    for i, p in enumerate(info['rooms']):
        interior[tuple(p.T)] = True
        if i in visited:
            active[tuple(p.T)] = True
            _require(np.all(expected[tuple(p.T)] == info['colors'][i]), "room color")
        else:
            _require(np.all(expected[tuple(p.T)] == info['bg']), "unvisited room was filled")
    _require(np.array_equal(expected[~active], info['target'][~active]), "walls, doors or exterior changed")
    _require(np.array_equal(expected != info['target'], active), "changes must be whole activated interiors")
    all_filled = info['target'].copy()
    for i, p in enumerate(info['rooms']):
        all_filled[tuple(p.T)] = info['colors'][i]
    _require(not np.array_equal(all_filled, expected), "omitting routing must change the answer")
    _require(not np.array_equal(_answer(info, straight=True)[0], expected), "turning must affect answer")
    db, tb = info['donor_box'], info['target_box']
    layout = 'side' if min(db[2], tb[2]) > max(db[0], tb[0]) else 'stack'
    signs = [int(np.sign(len(info['rooms'][j]) - len(info['donor_rooms'][i])))
             for i, j in info['matching'].items()]
    nonrect = sum(len(p) != (_bbox(p)[2] - _bbox(p)[0]) * (_bbox(p)[3] - _bbox(p)[1])
                  for p in info['rooms'] + info['donor_rooms'])
    return dict(rooms=len(info['rooms']), visited=len(visited), events=len(events),
                gates=len(info['ports']), entrance_heading=info['start'][1],
                revisits=len(events) > len(visited), layout=layout, nonrect=nonrect,
                richness=len(set(info['colors'])), roles=(info['bg'], info['wall'], info['gate']),
                area_signs=tuple(sorted(set(signs))),
                scan_changed=any(i != j for i, j in info['matching'].items()),
                target_shape=output.shape, input_shape=np.asarray(pair['input']).shape,
                graph=tuple(sorted((a, d, b) for a, links in info['edges'].items() for d, b in links.items())),
                structure=(output.shape, (info['target'] == info['wall']).tobytes(),
                           (info['target'] == info['gate']).tobytes()))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    _require(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
             and len(puzzle['train']) > len(puzzle['test']), "pair counts")
    seen = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair keys")
            inp = _grid(pair['input'])
            _grid(pair['output'])
            signature = _signature(inp)
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            _measure(pair)
    _require(_teaching(puzzle['train']), "training must distinguish relative turns, area and scan shortcuts")
    train_sides = {_decode(p['input'])['start'][1] for p in puzzle['train']}
    test_sides = {_decode(p['input'])['start'][1] for p in puzzle['test']}
    _require(len(test_sides) == 1 and not (train_sides & test_sides), "held-out entrance shift")


def self_test():
    """Bounded 42-seed semantic, variation, determinism and corruption checks."""
    features, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_signature(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        features.extend(_measure(p) for s in ('train', 'test') for p in puzzle[s])
    _require(len(signatures) == 42, "seed diversity")
    for key, minimum in (('rooms', 6), ('gates', 6), ('visited', 4), ('target_shape', 30),
                         ('input_shape', 30), ('graph', 60), ('roles', 35), ('structure', 100)):
        _require(len({f[key] for f in features}) >= minimum, 'variation: ' + key)
    _require({f['entrance_heading'] for f in features} == set(range(4)), "entrance orientations")
    _require({f['layout'] for f in features} == {'side', 'stack'}, "layout variation")
    _require(sum(f['nonrect'] > 0 for f in features) >= 30, "merged or notched rooms")
    _require(sum(f['scan_changed'] for f in features) >= 30, "spatial matching evidence")
    _require(sum(-1 in f['area_signs'] and 1 in f['area_signs'] for f in features) >= 20,
             "simultaneous growth and shrinkage")
    a, b = generate_puzzle(903, 2, 1), generate_puzzle(903, 2, 1)
    _require(all(_signature(p[k]) == _signature(q[k]) for s in a for p, q in zip(a[s], b[s])
                 for k in ('input', 'output')), "determinism")
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(904, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(905, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    damaged = generate_puzzle(906, 2, 1)
    damaged['test'][0]['output'] = damaged['test'][0]['output'].copy()
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds)')
