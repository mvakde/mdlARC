"""One-way delivery maps: direction-keyed reachability of a rigid square.

The trail and tip of the isolated key visibly identify cargo and dock colors.
All reasoning, including role decoding, is input-only. No parent code is imported.
"""
from collections import deque
import numpy as np


CARDINAL = ((-1, 0), (0, 1), (1, 0), (0, -1))
COMPASS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if (r, c) != (0, 0))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, queue = [start], [start]
        for r, c in queue:
            for dr, dc in CARDINAL:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    queue.append(p)
                    cells.append(p)
        result.append(cells)
    return result


def _square(cells):
    _require(bool(cells), "absent square")
    points = np.asarray(cells)
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    k = int(hi[0] - lo[0])
    _require(k == int(hi[1] - lo[1]) and len(cells) == k * k, "marker must be a solid square")
    return tuple(map(int, lo)), k


def _decode_key(a, background):
    candidates = []
    for cells in _components(a != background):
        p = np.asarray(cells)
        lo, hi = p.min(axis=0), p.max(axis=0) + 1
        h, w = hi - lo
        if h < 4 or w < 4 or len(cells) != h * w:
            continue
        r, c = map(int, lo)
        sub = a[r:int(hi[0]), c:int(hi[1])]
        wall = int(sub[0, 0])
        if not (np.all(sub[0] == wall) and np.all(sub[-1] == wall)
                and np.all(sub[:, 0] == wall) and np.all(sub[:, -1] == wall)):
            continue
        others = set(map(int, np.unique(sub))) - {wall}
        if len(others) != 2:
            continue
        interpretations = []
        for tip in sorted(others):
            trail = next(iter(others - {tip}))
            tips = set(map(tuple, np.argwhere(sub == tip)))
            trails = set(map(tuple, np.argwhere(sub == trail)))
            if len(tips) < 2:
                continue
            for dr, dc in COMPASS:
                used, lengths = set(), []
                for tr, tc in sorted(tips):
                    stroke = []
                    tr, tc = tr - dr, tc - dc
                    while (tr, tc) in trails:
                        stroke.append((tr, tc))
                        tr, tc = tr - dr, tc - dc
                    if len(stroke) < 2 or used.intersection(stroke):
                        break
                    used.update(stroke)
                    lengths.append(len(stroke))
                if len(lengths) == len(tips) and used == trails:
                    interpretations.append((trail, tip, (dr, dc), tuple(lengths)))
        if len(interpretations) == 1:
            trail, tip, direction, lengths = interpretations[0]
            candidates.append(dict(wall=wall, mover=trail, goal=tip, direction=direction,
                                   key_box=(r, int(hi[0]), c, int(hi[1])),
                                   key_lengths=lengths))
    _require(len(candidates) == 1, "input needs exactly one uniquely decoded key")
    return candidates[0]


def _reach(valid, start, moves, sources=None):
    _require(bool(valid[start]), "invalid starting footprint")
    seen = np.zeros(valid.shape, dtype=bool)
    points = [start] if sources is None else sources
    for p in points:
        _require(bool(valid[p]), "invalid reachability source")
        seen[p] = True
    queue = deque(points)
    while queue:
        r, c = queue.popleft()
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < valid.shape[0] and 0 <= nc < valid.shape[1]:
                if valid[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
    return seen


def _placements(allowed, k):
    return np.lib.stride_tricks.sliding_window_view(allowed, (k, k)).all(axis=(-1, -2))


def _point_docks(allowed, start, k, moves):
    # Dropping the footprint allows any source pixel to deliver to any dock
    # pixel. This ablation is independent of the square's anchor convention.
    r, c = start
    sources = [(r+i, c+j) for i in range(k) for j in range(k)]
    points = _reach(allowed, start, moves, sources=sources)
    return np.lib.stride_tricks.sliding_window_view(points, (k, k)).any(axis=(-1, -2))


def _read(a):
    background = int(a[0, 0])
    _require(np.all(a[0] == background) and np.all(a[-1] == background)
             and np.all(a[:, 0] == background) and np.all(a[:, -1] == background),
             "canvas must have an exterior background frame")
    info = _decode_key(a, background)
    _require(set(map(int, np.unique(a))) == {background, info['wall'], info['mover'], info['goal']},
             "exactly four distinct color roles required")
    key = np.zeros(a.shape, bool)
    r0, r1, c0, c1 = info['key_box']
    key[r0:r1, c0:c1] = True
    start, k = _square(list(map(tuple, np.argwhere((a == info['mover']) & ~key))))
    goals = []
    for cells in _components((a == info['goal']) & ~key):
        position, size = _square(cells)
        _require(size == k, "all docks must match the cargo footprint")
        goals.append(position)
    _require(bool(goals), "missing docks")
    allowed = (a != info['wall']) & ~key
    component = _reach(allowed, start, CARDINAL)
    _require(not (component[0].any() or component[-1].any()
                  or component[:, 0].any() or component[:, -1].any()), "cargo is not enclosed")
    _require(all(component[p] for p in goals), "docks must share one cell-connected network")
    dr, dc = info['direction']
    _require(dr != 0 and dc != 0, "this family uses diagonal quadrant keys")
    moves = ((dr, 0), (0, dc))
    valid = _placements(allowed, k)
    reachable = _reach(valid, start, moves)
    unrestricted = _reach(valid, start, CARDINAL)
    point_reachable = _point_docks(allowed, start, k, moves)
    reached = [p for p in goals if reachable[p]]
    _require(bool(reached), "at least one dock must be deliverable")
    info.update(background=background, key=key, start=start, k=k, goals=goals,
                allowed=allowed, component=component, moves=moves, valid=valid,
                reachable=reachable, reached=reached,
                direction_decoys=[p for p in goals if unrestricted[p] and not reachable[p]],
                size_decoys=[p for p in goals if point_reachable[p] and not reachable[p]])
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Erase the source and activate every dock reachable by the keyed square."""
    a = _grid(input_grid)
    info = _read(a)
    out = a.copy()
    r, c = info['start']
    k = info['k']
    out[r:r+k, c:c+k] = info['background']
    for r, c in info['reached']:
        out[r:r+k, c:c+k] = info['mover']
    return out


def _make_key(rng, wall, mover, goal):
    count = int(rng.choice([2, 2, 3]))
    length = int(rng.choice([2, 2, 3, 4]))
    points = []
    offset = 0
    for _ in range(count):
        stagger = int(rng.integers(2))
        for j in range(length + 1):
            points.append((j + stagger, offset + j + stagger, goal if j == length else mover))
        offset += int(rng.integers(2, 4))
    h = max(p[0] for p in points) + 3
    w = max(p[1] for p in points) + 3
    key = np.full((h, w), wall, np.int8)
    for r, c, color in points:
        key[r+1, c+1] = color
    return key


def _maze(rng, k, nr, nc, fork=False):
    # Unequal rooms on a spaced lattice; passages have their own widths/offsets.
    q = k + int(rng.integers(0, 3))
    row_gaps = rng.integers(1, 4, size=nr-1)
    col_gaps = rng.integers(1, 4, size=nc-1)
    rows = np.concatenate(([1], 1 + np.cumsum(q + row_gaps)))
    cols = np.concatenate(([1], 1 + np.cumsum(q + col_gaps)))
    h, w = int(rows[-1] + q + 1), int(cols[-1] + q + 1)
    if max(h, w) > 27:
        return None
    mask = np.zeros((h, w), bool)
    rooms = {}
    for r in range(nr):
        for c in range(nc):
            rh, rw = map(int, rng.integers(k, q+1, size=2))
            rr, cc = int(rows[r]), int(cols[c])
            mask[rr:rr+rh, cc:cc+rw] = True
            rooms[(r, c)] = (rr, cc, rh, rw)
    sr = int(rng.integers(nr-1))
    sc = 1 if fork else int(rng.integers(1, nc-1))
    source = (sr, sc)
    east, south, west, southeast = (sr, sc+1), (sr+1, sc), (sr, sc-1), (sr+1, sc+1)

    def edge(a, b):
        return tuple(sorted((a, b)))

    # This small motif supplies learnability witnesses. The rest is a random
    # connected graph, with extra edges creating forks and alternate routes.
    forced = {edge(source, east): 'wide', edge(source, west): 'wide',
              edge(source, south): 'narrow', edge(east, southeast): 'wide'}
    farther, merge = east, southeast
    if fork:
        farther, merge = (sr, sc+2), (sr+1, sc+2)
        forced.update({edge(east, farther): 'wide', edge(farther, merge): 'narrow',
                       edge(southeast, merge): 'wide'})
    edges = set(forced)
    nodes = list(rooms)
    connected = {source}
    while len(connected) < len(nodes):
        frontier = [(a, b) for a in sorted(connected) for dr, dc in CARDINAL
                    if (b := (a[0]+dr, a[1]+dc)) in rooms and b not in connected]
        a, b = frontier[int(rng.integers(len(frontier)))]
        edges.add(edge(a, b))
        connected.add(b)
    for a in nodes:
        for dr, dc in ((1, 0), (0, 1)):
            b = a[0]+dr, a[1]+dc
            if b in rooms and rng.random() < .35:
                edges.add(edge(a, b))
    for a, b in sorted(edges):
        ar, ac, ah, aw = rooms[a]
        br, bc, bh, bw = rooms[b]
        horizontal = a[0] == b[0]
        maximum = min(ah, bh) if horizontal else min(aw, bw)
        kind = forced.get((a, b))
        low, high = (k, maximum) if kind == 'wide' else (1, k-1) if kind == 'narrow' else (1, maximum)
        width = int(rng.integers(low, high+1))
        offset = 0 if kind is not None else int(rng.integers(maximum-width+1))
        if horizontal:
            mask[ar+offset:ar+offset+width, ac+aw:bc] = True
        else:
            mask[ar+ah:br, ac+offset:ac+offset+width] = True
    start = rooms[source][:2]
    anchors = {n: (r+h-k, c+w-k) for n, (r, c, h, w) in rooms.items() if n != source}
    valid = _placements(mask, k)
    reach = _reach(valid, start, ((1, 0), (0, 1)))
    point = _point_docks(mask, start, k, ((1, 0), (0, 1)))
    unrestricted = _reach(valid, start, CARDINAL)
    good = [n for n in anchors if reach[anchors[n]]]
    size_bad = [n for n in anchors if point[anchors[n]] and not reach[anchors[n]]]
    dir_bad = [n for n in anchors if unrestricted[anchors[n]] and not reach[anchors[n]]]
    if east not in good or southeast not in good or west not in dir_bad or south not in size_bad:
        return None
    if fork and (farther not in good or merge not in good):
        return None
    return mask, start, anchors, (east, southeast, west, south, farther, merge), good


def _assemble(rng, maze, key, palette):
    background, wall, mover, goal = palette
    mh, mw = maze.shape
    kh, kw = key.shape
    gap = int(rng.integers(1, 3))
    options = []
    if mh + kh + gap + 2 <= 30 and max(mw, kw) + 2 <= 30:
        options.extend(['above', 'below'])
    if mw + kw + gap + 2 <= 30 and max(mh, kh) + 2 <= 30:
        options.extend(['left', 'right'])
    if not options:
        return None
    side = options[int(rng.integers(len(options)))]
    if side in ('above', 'below'):
        h, w = mh + kh + gap + 2, max(mw, kw) + 2
        mr, kr = (kh+gap+1, 1) if side == 'above' else (1, mh+gap+1)
        mc, kc = int(rng.integers(1, w-mw)), int(rng.integers(1, w-kw))
    else:
        h, w = max(mh, kh) + 2, mw + kw + gap + 2
        mc, kc = (kw+gap+1, 1) if side == 'left' else (1, mw+gap+1)
        mr, kr = int(rng.integers(1, h-mh)), int(rng.integers(1, h-kh))
    a = np.full((h, w), background, np.int8)
    a[mr:mr+mh, mc:mc+mw] = maze
    a[kr:kr+kh, kc:kc+kw] = key
    # Independently pad both axes when room remains, then rotate/reflect.
    padding = []
    for dimension in a.shape:
        extra = int(rng.integers(min(3, 30-dimension)+1))
        before = int(rng.integers(extra+1))
        padding.append((before, extra-before))
    return np.pad(a, padding, constant_values=background)


def _pair(rng, k, nr, nc, palette, turns, reflect, replication, used):
    for _ in range(180):
        fork = replication and nc >= 4
        layout = _maze(rng, k, nr, nc, fork=fork)
        if layout is None:
            continue
        mask, start, anchors, witnesses, good = layout
        east, southeast, west, south, farther, merge = witnesses
        selected = {farther, southeast, merge, west, south} if replication else {
            (east, southeast)[int(rng.integers(2))], west, south}
        extras = [n for n in anchors if n not in selected]
        extra_count = int(rng.integers(min(3 if nr*nc <= 9 else 5, len(extras))+1))
        if extra_count:
            selected.update(extras[int(i)] for i in rng.choice(len(extras), extra_count, replace=False))
        bg, wall, mover, goal = palette
        maze = np.where(mask, bg, wall).astype(np.int8)
        r, c = start
        maze[r:r+k, c:c+k] = mover
        for node in sorted(selected):
            r, c = anchors[node]
            maze[r:r+k, c:c+k] = goal
        key = _make_key(rng, wall, mover, goal)
        a = _assemble(rng, maze, key, palette)
        if a is None:
            continue
        a = np.rot90(a, turns)
        if reflect:
            a = np.fliplr(a)
        a = a.copy()
        fingerprint = (a.shape, a.tobytes())
        if fingerprint in used:
            continue
        info = _read(a)
        _require(info['size_decoys'] and info['direction_decoys'], "lost ablation witness")
        _require(not replication or len(info['reached']) >= 2, "lost replication witness")
        if replication and not any(abs(p[0]-info['start'][0]) != abs(p[1]-info['start'][1])
                                   for p in info['reached']):
            continue
        used.add(fingerprint)
        return {'input': a, 'output': solve(a)}
    raise RuntimeError('bounded construction exhausted')


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(name + ' must be an integer')
        if not low <= int(value) <= high:
            raise ValueError(name + ' outside allowed range')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    variable_palette = bool(rng.integers(2))
    first_turn = int(rng.integers(4))
    specs = [(int(rng.integers(2, 4)), 2, 4, first_turn, False, True),
             (int(rng.integers(2, 4)), 3, 3, (first_turn+2) % 4, False, False)]
    for _ in range(nt-2):
        replication = bool(rng.integers(2))
        nr, nc = (2, 4) if replication else (3, 3)
        specs.append((int(rng.integers(2, 4)), nr, nc, int(rng.integers(4)), bool(rng.integers(2)),
                      replication))
    specs = [specs[int(i)] for i in rng.permutation(nt)]
    for _ in range(nq):
        nr, nc = ((3, 4), (4, 3))[int(rng.integers(2))]
        specs.append((int(rng.integers(2, 5)), nr, nc, int(rng.integers(4)), bool(rng.integers(2)),
                      bool(rng.integers(2))))
    puzzle, used = {'train': [], 'test': []}, set()
    for index, (k, nr, nc, turns, reflect, replication) in enumerate(specs):
        local = tuple(map(int, rng.choice(10, 4, replace=False))) if variable_palette else palette
        if variable_palette and index == 1 and local == previous:
            local = local[1:] + local[:1]
        pair = _pair(rng, k, nr, nc, local, turns, reflect, replication, used)
        puzzle['train' if index < nt else 'test'].append(pair)
        previous = local
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair schema")
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape, "changed dimensions")
            signature = (a.shape, a.tobytes())
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            info = _read(a)
            _require(np.array_equal(solve(a), b), "incorrect delivery or altered scenery")
            k = info['k']
            _require(not np.array_equal(a, b), "no delivery")
            _require(info['direction_decoys'] and info['size_decoys'], "pair lacks both parent witnesses")
            _require(np.array_equal(a[info['key']], b[info['key']]), "key changed")
            _require(np.array_equal(a[a == info['wall']], b[a == info['wall']]), "wall changed")
            cargo = (b == info['mover']) & ~info['key']
            _require(int(cargo.sum()) == len(info['reached']) * k*k, "incorrect copied area")
            for p in info['goals']:
                r, c = p
                color = info['mover'] if p in info['reached'] else info['goal']
                _require(np.all(b[r:r+k, c:c+k] == color), "dock partly changed")
            if split == 'train':
                training.append(info)
    _require(len({v['direction'] for v in training}) >= 2, "missing direction contrast")
    _require(any(len(v['reached']) >= 2 for v in training), "missing replication demonstration")
    def has_fork(v):
        dr, dc = v['direction']
        return any((p[0]-q[0])*dr * ((p[1]-q[1])*dc) < 0
                   for p in v['reached'] for q in v['reached'])
    _require(any(has_fork(v) for v in training), "missing incomparable branch destinations")
    # A bend is evidenced by a reached dock off the arrow's straight diagonal ray.
    _require(any(any(abs(p[0]-v['start'][0]) != abs(p[1]-v['start'][1]) for p in v['reached'])
                 for v in training), "missing non-ray delivery witness")


def _fingerprint(puzzle):
    return tuple((np.asarray(p[k]).shape, np.asarray(p[k]).tobytes())
                 for split in ('train', 'test') for p in puzzle[split] for k in ('input', 'output'))


def self_test():
    """Bounded semantic regressions, negative mutations, and 42-seed diversity."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_fingerprint(p) for p in puzzles}) == 42, "seed diversity collapsed")
    _require(_fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0)), "nondeterminism")
    shapes, structures, sizes, directions, key_shapes, dock_counts, copy_counts = set(), set(), set(), set(), set(), set(), set()
    palettes, free_areas = set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = pair['input']
                v = _read(a)
                normalized = np.zeros(a.shape, np.int8)
                for i, role in enumerate(('background', 'wall', 'mover', 'goal')):
                    normalized[a == v[role]] = i
                structures.add((a.shape, normalized.tobytes()))
                shapes.add(a.shape)
                sizes.add(v['k'])
                directions.add(v['direction'])
                palettes.add(tuple(v[r] for r in ('background', 'wall', 'mover', 'goal')))
                r0, r1, c0, c1 = v['key_box']
                key_shapes.add((r1-r0, c1-c0))
                dock_counts.add(len(v['goals']))
                copy_counts.add(len(v['reached']))
                free_areas.add(int(v['component'].sum()))
    _require(len(structures) >= 200 and len(shapes) >= 35, "geometry diversity collapsed")
    _require(sizes == {2, 3, 4} and len(directions) == 4, "scale/orientation coverage")
    _require(len(key_shapes) >= 10 and len(palettes) >= 30, "key/palette variation")
    _require(len(dock_counts) >= 4 and len(copy_counts) >= 3 and len(free_areas) >= 70, "structural counts collapsed")
    # Explicit geometry with directly stipulated destinations, including a
    # passable narrowing, a too-small throat, a backward dock and a fork.
    a = np.zeros((20, 25), np.int8)
    a[7:19, 1:24] = 5
    a[8:11, 6:10] = 0       # source chamber
    a[8:11, 10:13] = 0      # three-wide passage, passes k=2
    a[8:12, 13:17] = 0      # east chamber
    a[12:14, 14:16] = 0     # two-wide elbow
    a[14:18, 13:17] = 0     # southeast chamber
    a[11:14, 6:7] = 0       # one-wide throat
    a[14:18, 6:10] = 0      # south chamber
    a[8:10, 3:6] = 0        # backward passage
    a[8:12, 2:4] = 0        # west chamber
    a[8:10, 6:8] = 3
    docks = [(10, 2), (10, 15), (16, 15), (16, 8)]
    for r, c in docks:
        a[r:r+2, c:c+2] = 2
    key = np.full((5, 7), 5, np.int8)
    for off in (0, 2):
        key[1, 1+off] = key[2, 2+off] = 3
        key[3, 3+off] = 2
    a[1:6, 1:8] = key
    expected = a.copy()
    expected[8:10, 6:8] = 0
    expected[10:12, 15:17] = expected[16:18, 15:17] = 3
    _require(np.array_equal(solve(a), expected), "handcrafted dock endpoints")
    for turns in range(4):
        _require(np.array_equal(solve(np.rot90(a, turns)), np.rot90(expected, turns)), "rotation equivariance")
    # Opening the narrow gate changes delivery; reflecting the arrow changes
    # delivery. These are direct causal tests of both inherited mechanisms.
    wider = a.copy()
    wider[11:14, 7] = 0
    widened = solve(wider)
    _require(np.all(widened[16:18, 8:10] == 3), "width intervention did not enable south dock")
    reversed_key = a.copy()
    reversed_key[1:6, 1:8] = np.fliplr(key)
    reverse_expected = a.copy()
    reverse_expected[1:6, 1:8] = np.fliplr(key)
    reverse_expected[8:10, 6:8] = 0
    reverse_expected[10:12, 2:4] = 3
    _require(np.array_equal(solve(reversed_key), reverse_expected), "key intervention did not switch docks")
    # Corrupt supplied output, while retaining a legal schema.
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'] = bad['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('accepted identity corruption')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(814, nt, nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, "ignored explicit counts")
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError('accepted illegal counts')
    return {'seeds': 42, 'structures': len(structures), 'shapes': len(shapes),
            'palettes': len(palettes), 'key_shapes': len(key_shapes),
            'dock_counts': sorted(dock_counts), 'copy_counts': sorted(copy_counts)}


if __name__ == '__main__':
    print(self_test())
