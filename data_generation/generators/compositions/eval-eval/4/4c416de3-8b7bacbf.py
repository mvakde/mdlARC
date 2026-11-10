"""Stencil sockets: repair a circuit, then irrigate frames from powered corners.

Only NumPy and the standard library are used. The parser recovers every role,
socket, route, source and corner supply from the input, independently of the
random construction plan. Frame outlines are insulators, not graph edges.
"""
from collections import deque
import numpy as np

D4 = ((-1, 0), (0, -1), (0, 1), (1, 0))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
DOMAIN = frozenset((u, v) for u in range(-3, 0) for v in range(-3, 0))
PORTS = tuple([(-4, v) for v in range(-3, 0)] +
              [(u, -4) for u in range(-3, 0)])


def _components(cells, steps=D8):
    left = set(cells)
    result = []
    while left:
        root = min(left)
        left.remove(root)
        group, queue = {root}, [root]
        for r, c in queue:
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in left:
                    left.remove(q)
                    group.add(q)
                    queue.append(q)
        result.append(group)
    return result


def _near(cells):
    return {(r + dr, c + dc) for r, c in cells for dr, dc in D8}


def _outline(rect):
    t, l, b, r = rect
    return ({(t, c) for c in range(l, r + 1)} |
            {(b, c) for c in range(l, r + 1)} |
            {(y, l) for y in range(t, b + 1)} |
            {(y, r) for y in range(t, b + 1)})


def _box(rect):
    t, l, b, r = rect
    return {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)}


def _corners(rect):
    t, l, b, r = rect
    return ((t, l, 0), (t, r, 1), (b, l, 2), (b, r, 3))


def _map(corner, pattern):
    r, c, q = corner
    dy, dx = DIRS[q]
    return {(r + dy * u, c + dx * v) for u, v in pattern}


def _touching_ports(pattern):
    active, idle = [], []
    for p in PORTS:
        (active if _near({p}) & pattern else idle).append(p)
    return active, idle


def _as_grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape) or
            not np.issubdtype(a.dtype, np.integer) or
            not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _read_input(value):
    a = _as_grid(value)
    bg = int(np.bincount(a.ravel(), minlength=10).argmax())
    # Intact outlines make frames recognizable without decoration metadata.
    candidates = []
    for color in sorted(set(map(int, np.unique(a))) - {bg}):
        rects = []
        for cells in _components(map(tuple, np.argwhere(a == color)), D4):
            t, b = min(y for y, x in cells), max(y for y, x in cells)
            l, r = min(x for y, x in cells), max(x for y, x in cells)
            rect = (t, l, b, r)
            if b - t < 4 or r - l < 4 or cells != _outline(rect):
                break
            rects.append(rect)
        else:
            if rects:
                candidates.append((color, sorted(rects)))
    if len(candidates) != 1:
        raise ValueError("Frame color must have a unique rectangular interpretation")
    frame, rects = candidates[0]
    sources = []
    h, w = a.shape
    for y, x in np.argwhere((a != bg) & (a != frame)):
        p = (int(y), int(x))
        edge = {q for q in _near({p}) if 0 <= q[0] < h and 0 <= q[1] < w and a[q] != bg}
        values = {int(a[q]) for q in edge}
        if len(edge) == 5 and len(values) == 1 and int(a[p]) not in values:
            # Five neighbours must be exactly one of the four U-shaped caps.
            offsets = {(r - p[0], c - p[1]) for r, c in edge}
            base = {(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1)}
            for _ in range(4):
                if offsets == base:
                    sources.append((p, int(a[p]), next(iter(values))))
                    break
                base = {(-c, r) for r, c in base}
    wire_colors = {wire for _, _, wire in sources}
    if not sources or len(wire_colors) != 1:
        raise ValueError("Sources must identify one common wire role")
    wire = next(iter(wire_colors))
    paints = {paint for _, paint, _ in sources}
    if paints & {bg, frame, wire}:
        raise ValueError("Paint and structural roles must be distinct")
    remaining = set(map(int, np.unique(a))) - {bg, frame, wire} - paints
    if len(remaining) != 1:
        raise ValueError("Expected a single socket role distinct from paint")
    socket = remaining.pop()
    all_corners = [c for rect in rects for c in _corners(rect)]
    assigned = {c: set() for c in all_corners}
    for r, c in np.argwhere(a == socket):
        possible = [corner for corner in all_corners if (int(r), int(c)) in _map(corner, DOMAIN)]
        if len(possible) != 1:
            raise ValueError("Socket clue must belong to one exterior corner window")
        assigned[possible[0]].add((int(r), int(c)))
    occurrences = []
    for corner, cells in assigned.items():
        if cells:
            y, x, q = corner
            dy, dx = DIRS[q]
            clue = frozenset(((r - y) * dy, (c - x) * dx) for r, c in cells)
            occurrences.append((corner, clue))
    maximal = {p for _, p in occurrences if all(other <= p for _, other in occurrences)}
    if len(occurrences) < 2 or len(maximal) != 1:
        raise ValueError("An intact socket must contain every normalized partial clue")
    pattern = maximal.pop()
    if (not pattern <= DOMAIN or (-1, -1) not in pattern or
            not 3 <= len(pattern) <= 7 or len(_components(pattern)) != 1):
        raise ValueError("Invalid complete socket stencil")
    if not _touching_ports(pattern)[0] or not _touching_ports(pattern)[1]:
        raise ValueError("Sockets must offer both conducting and disconnected port positions")
    full_sets = {corner: _map(corner, pattern) for corner, _ in occurrences}
    full = set().union(*full_sets.values())
    if len(full) != sum(map(len, full_sets.values())):
        raise ValueError("Socket copies overlap")
    inside = set().union(*(_box(rect) - _outline(rect) for rect in rects))
    if any(a[p] != bg for p in inside):
        raise ValueError("Input frame interiors must be empty")
    if any(not (0 <= y < h and 0 <= x < w) for y, x in full):
        raise ValueError("Complete sockets must be visible in bounds")
    if any(a[p] not in (bg, socket) for p in full):
        raise ValueError("A completed socket would overwrite another role")
    # Paint markers are cups, not conductive cells. Outlines do not relay.
    for paint in paints:
        if set(map(tuple, np.argwhere(a == paint))) != {p for p, color, _ in sources if color == paint}:
            raise ValueError("Paint appears outside its source cups")
    wire_cells = set(map(tuple, np.argwhere(a == wire)))
    forbidden = set().union(*(_box(rect) for rect in rects))
    if wire_cells & forbidden or _near(wire_cells) & set().union(*map(_outline, rects)):
        raise ValueError("Wires may contact frames only through exterior sockets")
    info = dict(background=bg, frame=frame, socket=socket, wire=wire,
                rectangles=rects, occurrences=occurrences, stencil=pattern,
                full_sets=full_sets, wire_cells=wire_cells, sources=sources)
    supplies = _supplies(info, full)
    out = _paint_frames(a, info, supplies)
    for p in full:
        out[p] = socket
    info['supplies'] = supplies
    return out, info


def _circuits(info, socket_cells):
    circuits = _components(info['wire_cells'] | set(socket_cells))
    index = {p: i for i, cells in enumerate(circuits) for p in cells}
    colors = {}
    for p, paint, _ in info['sources']:
        touching = {index[q] for q in _near({p}) if q in index}
        if len(touching) != 1:
            raise ValueError("A cup must supply exactly one circuit")
        circuit = touching.pop()
        if circuit in colors and colors[circuit] != paint:
            raise ValueError("A conductive circuit cannot mix different sources")
        colors[circuit] = paint
    return index, colors


def _supplies(info, socket_cells):
    index, colors = _circuits(info, socket_cells)
    result = {}
    # The near-corner socket cell is the frame inlet. The restored stencil is
    # connected, so this cell speaks for its complete socket, not the outline.
    for corner, _ in info['occurrences']:
        inlet = next(iter(_map(corner, {(-1, -1)})))
        if inlet in index and index[inlet] in colors:
            result[corner] = colors[index[inlet]]
    return result


def _geometric_witnesses(inp, info):
    """Measure contact and relay evidence from visible input geometry."""
    full = set().union(*info['full_sets'].values())
    index, colors = _circuits(info, full)
    wire_parts = _components(info['wire_cells'])
    wire_index = {p: i for i, cells in enumerate(wire_parts) for p in cells}
    source_parts = {wire_index[q] for p, _, _ in info['sources']
                    for q in _near({p}) if q in wire_index}
    relay_sockets = 0
    idle_approaches = 0
    for corner, cells in info['full_sets'].items():
        fragments = {wire_index[p] for p in _near(cells) if p in wire_index}
        if len(fragments) >= 2 and fragments - source_parts:
            relay_sockets += 1
        if corner in info['supplies']:
            continue
        for p in _map(corner, PORTS):
            if (0 <= p[0] < inp.shape[0] and 0 <= p[1] < inp.shape[1]
                    and p in info['wire_cells'] and index[p] in colors
                    and not _near({p}) & cells):
                idle_approaches += 1
    return relay_sockets, idle_approaches


def _paint_frames(inp, info, supplies):
    out = inp.copy()
    for t, l, b, r in info['rectangles']:
        active = [(c, supplies[c]) for c in _corners((t, l, b, r)) if c in supplies]
        if not active:
            continue
        for y in range(t + 1, b):
            for x in range(l + 1, r):
                distances = [(abs(y - c[0]) + abs(x - c[1]), paint) for c, paint in active]
                nearest = min(d for d, _ in distances)
                winners = {paint for d, paint in distances if d == nearest}
                if len(winners) == 1:
                    out[y, x] = winners.pop()
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the reflected stencil and circuit supplies from input alone."""
    return _read_input(input_grid)[0]


def _ri(rng, lo, hi):
    return int(rng.integers(lo, hi + 1))


def _choose(rng, items):
    return items[_ri(rng, 0, len(items) - 1)]


def _sample_stencil(rng):
    for _ in range(100):
        pattern = {(-1, -1)}
        size = _ri(rng, 3, 7)
        while len(pattern) < size:
            choices = sorted((DOMAIN - pattern) & _near(pattern))
            pattern.add(_choose(rng, choices))
        true, false = _touching_ports(pattern)
        # Two well-separated contacts permit a socket to bridge wire pieces.
        bridge = [(p, q) for p in true for q in true if max(abs(p[0]-q[0]), abs(p[1]-q[1])) >= 3]
        if true and false and bridge and pattern != {(v, u) for u, v in pattern}:
            return frozenset(pattern)
    return frozenset({(-1, -1), (-2, -2), (-3, -3), (-3, -2)})


def _place_frames(rng, n, lesson):
    if n == 1:
        h, w = _ri(rng, 20, 30), _ri(rng, 20, 30)
    else:
        h, w = _ri(rng, 27, 30), _ri(rng, 27, 30)
    rects = []
    for _ in range(n):
        for attempt in range(180):
            if n >= 3:
                sh, sw = _ri(rng, 5, 6), _ri(rng, 5, 6)
            elif n == 2:
                sh, sw = _ri(rng, 5, 9), _ri(rng, 5, 9)
                if lesson == 1 and not rects:
                    sh = _choose(rng, [5, 7, 9])
                    sw = _choose(rng, [k for k in (5, 7, 9) if k != sh])
            else:
                sh, sw = _ri(rng, 7, min(17, h-8)), _ri(rng, 7, min(17, w-8))
                if lesson == 0:
                    sh += 1 - sh % 2
                    sw += 1 - sw % 2
                    if sh > h - 8:
                        sh -= 2
                    if sw > w - 8:
                        sw -= 2
            t, l = _ri(rng, 4, h-sh-4), _ri(rng, 4, w-sw-4)
            rect = (t, l, t+sh-1, l+sw-1)
            if all(rect[2]+8 < other[0] or other[2]+8 < rect[0] or
                   rect[3]+8 < other[1] or other[3]+8 < rect[1] for other in rects):
                rects.append(rect)
                break
        else:
            return None
    return (h, w), rects


def _path(rng, starts, goals, allowed):
    starts = sorted(starts)
    rng.shuffle(starts)
    queue = deque(starts)
    prev = {p: None for p in starts}
    directions = [D8[int(i)] for i in rng.permutation(8)]
    end = None
    while queue:
        p = queue.popleft()
        if p in goals:
            end = p
            break
        for dr, dc in directions:
            q = (p[0]+dr, p[1]+dc)
            if q in allowed and q not in prev:
                prev[q] = p
                queue.append(q)
    if end is None:
        return None
    result = set()
    while end is not None:
        result.add(end)
        end = prev[end]
    return result


def _cup(rng, canvas, blocked):
    src, wire = {(0, 1)}, {(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)}
    for _ in range(_ri(rng, 0, 3)):
        src = {(-c, r) for r, c in src}
        wire = {(-c, r) for r, c in wire}
    for _ in range(90):
        p = _choose(rng, sorted(canvas))
        s = {(r+p[0], c+p[1]) for r, c in src}
        w = {(r+p[0], c+p[1]) for r, c in wire}
        cells = s | w
        if cells <= canvas and not (cells | _near(cells)) & blocked:
            return s, w
    return None


def _draw_scene(rng, pattern, roles, lesson, test):
    n = 1 if lesson == 0 else (2 if lesson == 1 else _choose(rng, [1, 1, 2, 2, 3]))
    if test:
        n = _choose(rng, [2, 2, 3])
    placement = _place_frames(rng, n, lesson)
    if placement is None:
        return None
    shape, rects = placement
    canvas = {(r, c) for r in range(shape[0]) for c in range(shape[1])}
    corners = [c for rect in rects for c in _corners(rect)]
    rng.shuffle(corners)
    # Lesson zero compares two adjacent paints, with two unsupplied sockets.
    if lesson == 0:
        q = _ri(rng, 0, 3)
        marked = list(_corners(rects[0]))
        targets = [marked[q], marked[q ^ 1]]
        inactive = [c for c in marked if c not in targets]
        reference = inactive[0]
        owners = {targets[0]: 0, targets[1]: 1}
        source_count = 2
        bridge = None
    elif lesson == 1:
        first = _choose(rng, list(_corners(rects[0])))
        second = _choose(rng, list(_corners(rects[1])))
        opposite = _corners(rects[0])[first[2] ^ 3]
        reference = _choose(rng, [c for c in corners if c not in (first, second, opposite)])
        marked = [first, second, opposite, reference]
        owners = {first: 0, second: 0, opposite: 1}
        source_count = 2
        bridge = (first, second)
    else:
        source_count = _ri(rng, 1, 3 if test else 2)
        count = _ri(rng, max(source_count + 1, 3), min(len(corners), 8))
        marked = corners[:count]
        reference = _choose(rng, marked)
        targets = marked[:source_count]
        owners = {c: i for i, c in enumerate(targets)}
        for c in marked[source_count:]:
            if c != reference and rng.random() < .45:
                owners[c] = _ri(rng, 0, source_count-1)
        inactive = [c for c in marked if c not in owners]
        if not inactive:
            owners.pop(marked[-1])
        bridge = None
        if n >= 2 and len(owners) >= 2 and rng.random() < .5:
            options = [(a, b) for a in owners for b in owners if a != b and owners[a] == owners[b]]
            if options:
                bridge = _choose(rng, options)
    inactive = [c for c in marked if c not in owners]
    if not inactive:
        return None
    # A full reference can be supplied in ordinary pairs; there is no active
    # reference convention. Lesson references deliberately remain unsupplied.
    true_ports, false_ports = _touching_ports(pattern)
    full_sets = {c: _map(c, pattern) for c in marked}
    full = set().union(*full_sets.values())
    windows = set().union(*(_map(c, DOMAIN) for c in corners))
    outlines = set().union(*map(_outline, rects))
    forbidden = set().union(*map(_box, rects)) | _near(outlines) | windows | _near(full)
    bg, frame, socket, wire = roles[:4]
    paints = list(roles[4:4+source_count])
    # A few circuits share a paint value; equal-distance equal colors coalesce.
    if lesson == -1 and source_count > 1 and rng.random() < .25:
        paints[-1] = paints[0]
    wires, sources = [], []
    for owner in range(source_count):
        others = set().union(*wires, *sources) if wires else set()
        cup = _cup(rng, canvas, forbidden | others | _near(others))
        if cup is None:
            return None
        src, cells = cup
        wires.append(cells)
        sources.append(src)
    all_source = set().union(*sources)
    goal_lists = [[] for _ in range(source_count)]
    bridge_spec = None
    for corner, owner in owners.items():
        if bridge and corner == bridge[1]:
            continue
        if bridge and corner == bridge[0]:
            options = [(p, q) for p in true_ports for q in true_ports
                       if max(abs(p[0]-q[0]), abs(p[1]-q[1])) >= 3]
            p, q = _choose(rng, options)
            target = next(iter(_map(bridge[1], {_choose(rng, true_ports)})))
            bridge_spec = (owner, next(iter(_map(corner, {q}))), target)
        else:
            p = _choose(rng, true_ports)
        goal_lists[owner].append(next(iter(_map(corner, {p}))))
    # A supplied wire ending near an idle socket is a required geometric
    # counterexample to "any nearby route powers this corner".
    idle_corner = _choose(rng, inactive)
    idle_goal = next(iter(_map(idle_corner, {_choose(rng, false_ports)})))
    goal_lists[_ri(rng, 0, source_count-1)].append(idle_goal)
    all_goals = set(p for goals in goal_lists for p in goals)
    if bridge_spec:
        all_goals.update(bridge_spec[1:])
    # Route circuits independently. Their bounds may overlap, but their pixels
    # do not cross or touch. A socket may reconnect separate wire components.
    for owner in rng.permutation(source_count):
        owner = int(owner)
        own = wires[owner]
        others = set().union(*(s for i, s in enumerate(wires) if i != owner))
        blocked = forbidden | all_source | _near(all_source) | others | _near(others)
        blocked |= all_goals - set(goal_lists[owner])
        allowed = (canvas - blocked) | own
        for goal in goal_lists[owner]:
            path = _path(rng, own, {goal}, allowed | {goal})
            if path is None or (path - own) & _near(others):
                return None
            own.update(path)
        wires[owner] = own
    fragments = []
    if bridge_spec:
        owner, start, goal = bridge_spec
        occupied = set().union(*wires)
        blocked = forbidden | all_source | _near(all_source) | occupied | _near(occupied)
        allowed = (canvas - blocked) | {start, goal}
        path = _path(rng, {start}, {goal}, allowed)
        if path is None or path & _near(occupied):
            return None
        fragments.append(path)
    a = np.full(shape, bg, dtype=np.int8)
    for rect in rects:
        for p in _outline(rect):
            a[p] = frame
    for cells in wires + fragments:
        for p in cells:
            a[p] = wire
    for cells, paint in zip(sources, paints):
        for p in cells:
            a[p] = paint
    for corner in marked:
        if corner == reference or (lesson not in (0, 1) and rng.random() < .15):
            retained = pattern
        elif lesson in (0, 1):
            retained = {(-1, -1)}
        else:
            points = sorted(pattern)
            rng.shuffle(points)
            retained = set(points[:_ri(rng, 1, len(pattern)-1)])
        for p in _map(corner, retained):
            if a[p] != bg:
                return None
            a[p] = socket
    expected_info = dict(rectangles=rects)
    supplies = {c: paints[owner] for c, owner in owners.items()}
    b = _paint_frames(a, expected_info, supplies)
    for p in full:
        if a[p] not in (bg, socket):
            return None
        b[p] = socket
    if int(np.bincount(a.ravel(), minlength=10).argmax()) != bg:
        return None
    try:
        prediction, info = _read_input(a)
        if not np.array_equal(prediction, b) or info['supplies'] != supplies:
            return None
        if not np.any(a != b):
            return None
        if lesson in (0, 1):
            unrepaired = _supplies(info, set(map(tuple, np.argwhere(a == socket))))
            cavity_ablation = _paint_frames(a, info, unrepaired)
            if not any(cavity_ablation[p] != b[p] for rect in rects for p in _box(rect)-_outline(rect)):
                return None
        if lesson == 0:
            t, l, bot, r = rects[0]
            if not np.any(b[t+1:bot, l+1:r] == bg):
                return None
    except ValueError:
        return None
    return {'input': a, 'output': b}


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed) % (1 << 64))
    stable_roles = tuple(map(int, rng.permutation(10))) if rng.random() < .5 else None
    puzzle, seen, training_stencils = {'train': [], 'test': []}, set(), set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for i in range(count):
            lesson = i if split == 'train' and i < 2 else -1
            pattern = _sample_stencil(rng)
            if split == 'test':
                for _ in range(100):
                    if pattern not in training_stencils:
                        break
                    pattern = _sample_stencil(rng)
            roles = stable_roles or tuple(map(int, rng.permutation(10)))
            for attempt in range(700):
                if attempt and attempt % 100 == 0:
                    # Resample a geometric socket when its terminal positions
                    # repeatedly frustrate packing; keep the same lesson.
                    pattern = _sample_stencil(rng)
                if split == 'test' and pattern in training_stencils:
                    pattern = _sample_stencil(rng)
                    continue
                pair = _draw_scene(rng, pattern, roles, lesson, split == 'test')
                if pair is not None and _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError(f'Bounded construction exhausted at seed {seed}, {split} {i}')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
            if split == 'train':
                training_stencils.add(pattern)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _as_grid(pair['input']), _as_grid(pair['output'])
            assert a.shape == b.shape and np.any(a != b)
            assert _key(a) not in seen
            seen.add(_key(a))
            predicted, info = _read_input(a)
            assert np.array_equal(predicted, b), 'Input-derived prediction differs'
            sockets = set().union(*info['full_sets'].values())
            interior = set().union(*(_box(rect)-_outline(rect) for rect in info['rectangles']))
            changed = set(map(tuple, np.argwhere(a != b)))
            assert changed <= sockets | interior
            assert all(a[p] == info['background'] for p in changed)
            assert all(b[p] == info['socket'] for p in sockets)
            assert all(a[p] == b[p] for rect in info['rectangles'] for p in _outline(rect))
            assert all(a[p] == b[p] for p in info['wire_cells'])
            assert all(a[p] == b[p] for p, _, _ in info['sources'])
            assert info['supplies']
            assert any(c not in info['supplies'] for c, _ in info['occurrences'])
            assert _geometric_witnesses(a, info)[1] >= 1, 'Missing supplied but disconnected port witness'
            infos[split].append(info)
    # Minimum lessons explicitly witness the two inherited mechanisms.
    for pair, info in zip(puzzle['train'][:2], infos['train'][:2]):
        a, b = np.asarray(pair['input']), np.asarray(pair['output'])
        raw = _supplies(info, set(map(tuple, np.argwhere(a == info['socket']))))
        ignored_a = _paint_frames(a, info, raw)
        interiors = set().union(*(_box(rect)-_outline(rect) for rect in info['rectangles']))
        assert any(ignored_a[p] != b[p] for p in interiors), 'Repair must change irrigation'
        # Ignoring sources leaves every cavity empty, changing a productive cell.
        assert any(b[p] != info['background'] for p in interiors)
    first = infos['train'][0]
    assert len(set(first['supplies'].values())) == 2
    assert len(first['rectangles']) == 1
    assert len(infos['train'][1]['rectangles']) == 2
    reference = next(c for c, p in first['occurrences'] if p == first['stencil'])
    rotated_pattern = ({(v, u) for u, v in first['stencil']} if reference[2] in (1, 2)
                       else first['stencil'])
    assert any(_map(c, {(v, u) for u, v in rotated_pattern} if c[2] in (1, 2) else rotated_pattern)
               != first['full_sets'][c] for c, _ in first['occurrences']), 'Training must distinguish rotation'
    assert _geometric_witnesses(np.asarray(puzzle['train'][1]['input']), infos['train'][1])[0] >= 1
    second_pair, second = puzzle['train'][1], infos['train'][1]
    alternative = _euclidean_fill(np.asarray(second_pair['input']), second)
    interior = set().union(*(_box(rect)-_outline(rect) for rect in second['rectangles']))
    assert any(alternative[p] != np.asarray(second_pair['output'])[p] for p in interior), 'Training must teach Manhattan distance'
    assert all(i['stencil'] not in {j['stencil'] for j in infos['train']} for i in infos['test'])


def _signature(puzzle):
    return tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in puzzle[s])


def _euclidean_fill(inp, info):
    """Countermodel used only to check that training teaches the distance."""
    out = inp.copy()
    for rect in info['rectangles']:
        active = [(c, info['supplies'][c]) for c in _corners(rect) if c in info['supplies']]
        if not active:
            continue
        for y, x in _box(rect) - _outline(rect):
            distances = [((y-c[0])**2+(x-c[1])**2, paint) for c, paint in active]
            best = min(d for d, _ in distances)
            colors = {paint for d, paint in distances if d == best}
            if len(colors) == 1:
                out[y, x] = colors.pop()
    return out


def _regression_test():
    # Explicit geometry, not an RNG-generated output: agreeing sources fill
    # the bisector, whereas different paints leave the center column blank.
    pattern = frozenset({(-1, -1), (-2, -2), (-3, -3), (-3, -2)})
    rect = (9, 9, 19, 19)
    corners = _corners(rect)
    for right_paint in (2, 4):
        a = np.zeros((30, 30), dtype=np.int8)
        for p in _outline(rect):
            a[p] = 7
        for col, paint in ((6, 2), (22, right_paint)):
            a[1, col] = paint
            for p in ((1, col-1), (1, col+1), (2, col-1), (2, col), (2, col+1)):
                a[p] = 1
            a[3:6, col] = 1
        for i, corner in enumerate(corners[:3]):
            for p in _map(corner, pattern if i == 0 else {(-1, -1)}):
                a[p] = 3
        expected = a.copy()
        for corner in corners[:3]:
            for p in _map(corner, pattern):
                expected[p] = 3
        expected[10:19, 10:14] = 2
        expected[10:19, 15:19] = right_paint
        expected[10:19, 14] = 2 if right_paint == 2 else 0
        assert np.array_equal(solve(a), expected)
        # Complete rotation/reflection of the entire scene remains valid.
        for rotation in range(4):
            assert np.array_equal(solve(np.rot90(a, rotation)), np.rot90(expected, rotation))
        assert np.array_equal(solve(np.fliplr(a)), np.fliplr(expected))
        # A transposed completion is the wrong adjacent-corner convention.
        transposed = _map(corners[1], {(v, u) for u, v in pattern})
        assert transposed != _map(corners[1], pattern)


def self_test():
    """Bounded structural, semantic-ablation and 42-seed diversity checks."""
    _regression_test()
    signatures, shapes, patterns, frame_shapes, counts, palettes = set(), set(), set(), set(), set(), set()
    offset, multiple_reference, repeated_paint, competition, relays = 0, 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a, b = pair['input'], pair['output']
                snapshot = a.copy()
                predicted, info = _read_input(a)
                assert np.array_equal(predicted, b) and np.array_equal(snapshot, a)
                shapes.add(a.shape)
                patterns.add(info['stencil'])
                counts.add(len(info['rectangles']))
                palettes.add(tuple(info[k] for k in ('background', 'frame', 'socket', 'wire')))
                frame_shapes.update((bot-t+1, r-l+1) for t, l, bot, r in info['rectangles'])
                offset += any((-1, -1) not in p for _, p in info['occurrences'])
                multiple_reference += sum(p == info['stencil'] for _, p in info['occurrences']) > 1
                paints = [paint for _, paint, _ in info['sources']]
                repeated_paint += len(set(paints)) < len(paints)
                competition += any(len({info['supplies'][c] for c in _corners(rect) if c in info['supplies']}) > 1
                                   for rect in info['rectangles'])
                relays += _geometric_witnesses(a, info)[0] > 0
    assert len(signatures) == 42
    assert len(shapes) >= 30 and len(frame_shapes) >= 30 and len(patterns) >= 20
    assert {1, 2, 3} <= counts and len(palettes) >= 20
    assert offset and multiple_reference and repeated_paint and competition >= 42
    assert relays >= 42
    assert _signature(generate_puzzle(0)) == _signature(generate_puzzle(0))
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(1234, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError('Illegal count accepted')
    good = generate_puzzle(0, 2, 1)
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in good[s]] for s in good}
    p = bad['test'][0]
    r, c = np.argwhere(p['input'] != p['output'])[0]
    p['output'][r, c] = p['input'][r, c]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Wrong socket or irrigation pixel accepted')
