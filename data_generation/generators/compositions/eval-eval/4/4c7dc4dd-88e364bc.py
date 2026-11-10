"""Color-clock billiards: a local analogy advances at each reflected wall impact.

Only the standard library and NumPy are used. No parent modules or files are
needed to generate or solve a puzzle. See task_spec.md for the full language.
"""
from __future__ import annotations

from numbers import Integral
import numpy as np


DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
              (0, 1), (1, -1), (1, 0), (1, 1))
CARDINAL = tuple(d for d in DIRECTIONS if 0 in d)
DIAGONAL = tuple(d for d in DIRECTIONS if 0 not in d)


def _require(value, message):
    if not value:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= s <= 30 for s in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger cells")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=True)


def _fingerprint(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        # A stable traversal keeps all seed-dependent choices reproducible.
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in CARDINAL:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
                    cells.add(p)
        result.append(cells)
    return result


def _bbox(cells):
    rows, cols = zip(*cells)
    return int(min(rows)), int(min(cols)), int(max(rows)) + 1, int(max(cols)) + 1


def _decode_key(a, bg):
    candidates = []
    for cells in _components(a != bg):
        r, c, bottom, right = _bbox(cells)
        h, w = bottom - r, right - c
        if min(h, w) < 4 or len(cells) != h * w:
            continue
        key = a[r:bottom, c:right]
        carrier = int(key[0, 0])
        border = np.r_[key[0], key[-1], key[:, 0], key[:, -1]]
        others = set(map(int, np.unique(key))) - {carrier}
        if len(others) != 2 or np.any(border != carrier):
            continue
        interpretations = []
        for tip in sorted(others):
            trail = next(iter(others - {tip}))
            tips = set(map(tuple, np.argwhere(key == tip)))
            trails = set(map(tuple, np.argwhere(key == trail)))
            if len(tips) < 2:
                continue
            for dr, dc in DIRECTIONS:
                used, lengths = set(), []
                for tr, tc in sorted(tips):
                    stroke = []
                    p = tr - dr, tc - dc
                    while p in trails:
                        stroke.append(p)
                        p = p[0] - dr, p[1] - dc
                    if len(stroke) < 2 or used.intersection(stroke):
                        break
                    used.update(stroke)
                    lengths.append(len(stroke))
                if len(lengths) == len(tips) and used == trails:
                    interpretations.append((trail, tip, (dr, dc)))
        if len(interpretations) == 1:
            trail, terminal, direction = interpretations[0]
            candidates.append(dict(box=(r, c, bottom, right), carrier=carrier,
                                   trail=trail, terminal=terminal, direction=direction))
    _require(len(candidates) == 1, "expected one unambiguous direction key")
    return candidates[0]


def _reference_panels(a, bg, key):
    """Read monochrome hollow frames and their one-cell polarity tabs."""
    candidates = []
    excluded = {bg, key['carrier'], key['trail'], key['terminal']}
    for frame in sorted(set(map(int, np.unique(a))) - excluded):
        panels = []
        for cells in _components(a == frame):
            r, c, bottom, right = _bbox(cells)
            h, w = bottom-r, right-c
            if min(h, w) < 4 or len(cells) != 2*h + 2*w - 4:
                continue
            sub = a[r:bottom, c:right]
            if not (np.all(sub[0] == frame) and np.all(sub[-1] == frame)
                    and np.all(sub[:, 0] == frame) and np.all(sub[:, -1] == frame)):
                continue
            outside = ([(r-1, x) for x in range(c+1, right-1)]
                       + [(bottom, x) for x in range(c+1, right-1)]
                       + [(y, c-1) for y in range(r+1, bottom-1)]
                       + [(y, right) for y in range(r+1, bottom-1)])
            tabs = [(p, int(a[p])) for p in outside
                    if 0 <= p[0] < a.shape[0] and 0 <= p[1] < a.shape[1]
                    and a[p] != bg]
            if len(tabs) != 1 or tabs[0][1] not in (key['trail'], key['terminal']):
                continue
            panels.append(dict(box=(r, c, bottom, right), tab=tabs[0],
                               inside=sub[1:-1, 1:-1].copy()))
        if len(panels) == 2 and panels[0]['inside'].shape == panels[1]['inside'].shape:
            if {p['tab'][1] for p in panels} == {key['trail'], key['terminal']}:
                candidates.append((frame, panels))
    _require(len(candidates) == 1, "reference pair is ambiguous")
    frame, panels = candidates[0]
    source = next(p for p in panels if p['tab'][1] == key['trail'])
    partner = next(p for p in panels if p['tab'][1] == key['terminal'])
    x, y = source['inside'], partner['inside']
    _require(np.array_equal(x == bg, y == bg), "reference geometry changed")
    colors = sorted(set(map(int, np.unique(x))) - {bg})
    _require(3 <= len(colors) <= 5, "cycle must have 3..5 colors")
    _require(set(map(int, np.unique(y))) - {bg} == set(colors), "reference palette changed")
    _require(key['terminal'] in colors, "terminal is absent from the clock")
    _require(not set(colors) & {bg, frame, key['carrier'], key['trail']}, "role overlap")
    mapping = {}
    for color in colors:
        values = np.unique(y[x == color])
        _require(len(values) == 1, "reference recoloring is not role-consistent")
        mapping[color] = int(values[0])
    _require(set(mapping.values()) == set(colors), "not a permutation")
    current, visited = colors[0], []
    while current not in visited:
        visited.append(current)
        current = mapping[current]
    _require(current == colors[0] and len(visited) == len(colors), "not a single cycle")
    return dict(frame=frame, source=source, partner=partner, mapping=mapping, colors=colors)


def _read(a):
    bg = int(a[0, 0])
    _require(bool(np.all(np.r_[a[0], a[-1], a[:, 0], a[:, -1]] == bg)),
             "canvas needs a background margin")
    key = _decode_key(a, bg)
    reference = _reference_panels(a, bg, key)
    wall = a == key['carrier']
    r, c, bottom, right = key['box']
    wall[r:bottom, c:right] = False
    _require(bool(np.any(wall)), "missing chamber")
    room_box = _bbox(np.argwhere(wall))
    r, c, bottom, right = room_box
    room = a[r:bottom, c:right].copy()
    local_wall = room == key['carrier']
    allowed = {bg, key['carrier']} | set(reference['colors'])
    _require(set(map(int, np.unique(room))) <= allowed, "unexplained chamber color")
    regions, starts = [], []
    for cells in _components(~local_wall):
        enclosed = all(0 < y < room.shape[0]-1 and 0 < x < room.shape[1]-1 for y, x in cells)
        tokens = [p for p in sorted(cells) if int(room[p]) in reference['colors']]
        _require(not tokens or enclosed, "token is outside a closed chamber")
        if enclosed:
            index = len(regions)
            regions.append(cells)
            starts += [(p, int(room[p]), index) for p in tokens]
    _require(starts and all(color != key['terminal'] for _, color, _ in starts),
             "tokens must begin before the terminal state")
    # Explain every non-background pixel, including reference tabs.
    explained = wall.copy()
    r0, c0, b0, d0 = key['box']
    explained[r0:b0, c0:d0] = True
    for panel in (reference['source'], reference['partner']):
        pr, pc, pb, pd = panel['box']
        _require(not np.any(explained[pr:pb, pc:pd]), "overlapping objects")
        explained[pr:pb, pc:pd] = True
        explained[panel['tab'][0]] = True
    for (y, x), _, _ in starts:
        explained[room_box[0]+y, room_box[1]+x] = True
    _require(bool(np.all(explained | (a == bg))), "unexplained input cells")
    return dict(background=bg, key=key, reference=reference, room=room,
                box=room_box, wall=local_wall, regions=regions, starts=starts)


def _steps(mapping, color, terminal):
    count = 0
    while color != terminal:
        color = mapping[color]
        count += 1
        _require(count <= len(mapping), "clock cannot reach terminal")
    return count


def _region_flight(free, start, direction, count, corner=True):
    """Input-reading physics; free-space membership defines all boundaries."""
    position = start
    dr, dc = direction
    legs, witnesses, kinds = [], [], []
    for _ in range(count):
        path = [position]
        while True:
            r, c = position
            next_cell = r+dr, c+dc
            side_r = (r+dr, c) not in free if dr else False
            side_c = (r, c+dc) not in free if dc else False
            if next_cell not in free or (corner and dr and dc and (side_r or side_c)):
                break
            position = next_cell
            path.append(position)
        witnesses.append(bool(dr and dc and next_cell in free and (side_r or side_c)))
        kinds.append((bool(side_r), bool(side_c)))
        legs.append(path)
        if not dr:
            dc = -dc
        elif not dc:
            dr = -dr
        elif side_r or side_c:
            if side_r:
                dr = -dr
            if side_c:
                dc = -dc
        else:
            # An isolated diagonal wall cell acts as a point corner.
            dr, dc = -dr, -dc
    return dict(end=position, legs=legs, corner=any(witnesses), kinds=kinds,
                cells=set(p for path in legs for p in path))


def _predict(info, *, mapping=None, direction=None, first_only=False, corner=True):
    out = info['room'].copy()
    bg, terminal = info['background'], info['key']['terminal']
    mapping = info['reference']['mapping'] if mapping is None else mapping
    direction = info['key']['direction'] if direction is None else direction
    flights = []
    for p, color, region in info['starts']:
        out[p] = bg
        count = 1 if first_only else _steps(mapping, color, terminal)
        flights.append(_region_flight(info['regions'][region], p, direction, count, corner))
    for flight in flights:
        out[flight['end']] = terminal
    return out, flights


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the local color clock, then reconstruct the chamber's final state."""
    return _predict(_read(_grid(input_grid)))[0]


def _make_key(rng, direction, carrier, trail, terminal):
    dr, dc = direction
    length = int(rng.integers(2, 4))
    # Parallel but optionally staggered strokes; no shape dictionary is used.
    points = []
    for i in range(2):
        stagger = int(rng.integers(2))
        base = (0, 2*i) if dr else (2*i, 0)
        for j in range(length+1):
            points.append((base[0]+(j+stagger)*dr, base[1]+(j+stagger)*dc,
                           terminal if j == length else trail))
    r0, c0 = min(p[0] for p in points), min(p[1] for p in points)
    h = max(p[0] for p in points)-r0+3
    w = max(p[1] for p in points)-c0+3
    a = np.full((h, w), carrier, np.int8)
    for r, c, color in points:
        a[r-r0+1, c-c0+1] = color
    return a


def _make_reference(rng, colors, mapping, bg, frame, trail, terminal):
    h, w = int(rng.integers(3, 5)), int(rng.integers(3, 6))
    mask = rng.random((h, w)) < float(rng.uniform(.55, .9))
    order = rng.permutation(h*w)
    mask.flat[order[:len(colors)+1]] = True
    mask.flat[order[-1]] = False
    if np.count_nonzero(mask) < len(colors):
        mask.flat[order[:len(colors)]] = True
    positions = np.argwhere(mask)
    motif = np.full((h, w), bg, np.int8)
    ink = list(colors) + list(map(int, rng.choice(colors, len(positions)-len(colors))))
    rng.shuffle(ink)
    for p, color in zip(positions, ink):
        motif[tuple(p)] = color
    partner = np.full_like(motif, bg)
    for color, successor in mapping.items():
        partner[motif == color] = successor
    blocks = []
    for interior, tab in ((motif, trail), (partner, terminal)):
        # The first row is tab space; the frame itself begins at row one.
        block = np.full((h+3, w+2), bg, np.int8)
        block[1, :] = block[-1, :] = frame
        block[1:, 0] = block[1:, -1] = frame
        block[2:-1, 1:-1] = interior
        block[0, 1 + w//2] = tab
        blocks.append(block)
    return blocks


def _wall_shape(rng, h, w, complexity, witness, direction, fallback=False):
    footprint = np.ones((h, w), bool)
    kind = 0 if fallback else int(rng.integers(4))
    if kind == 1:
        for corner in rng.choice(4, int(rng.integers(1, 4)), replace=False):
            dh, dw = int(rng.integers(1, h//3)), int(rng.integers(1, w//3))
            rs = slice(0, dh) if corner < 2 else slice(h-dh, h)
            cs = slice(0, dw) if corner % 2 == 0 else slice(w-dw, w)
            footprint[rs, cs] = False
    elif kind == 2:
        rr, cc = np.indices((h, w))
        cut = int(rng.integers(1, max(2, min(h, w)//3)))
        distance = np.minimum(rr, h-1-rr) + np.minimum(cc, w-1-cc)
        footprint &= distance >= cut
    elif kind == 3:
        left, right = 0, 0
        for r in range(h):
            left = int(np.clip(left + rng.integers(-1, 2), 0, max(1, w//4)))
            right = int(np.clip(right + rng.integers(-1, 2), 0, max(1, w//4)))
            footprint[r, :left] = False
            if right:
                footprint[r, w-right:] = False
    padded = np.pad(footprint, 1)
    interior = footprint.copy()
    for dr, dc in CARDINAL:
        interior &= padded[1+dr:1+dr+h, 1+dc:1+dc+w]
    wall = footprint & ~interior
    if witness:
        col, end = w//2, h//2
        rows = slice(0, end+1) if direction[0] > 0 else slice(h-end-1, h)
        wall[rows, col] |= footprint[rows, col]
    for _ in range(complexity if not fallback else 0):
        if rng.integers(2):
            row = int(rng.integers(2, h-2))
            cut = int(rng.integers(2, w-2))
            cols = slice(0, cut) if rng.integers(2) else slice(cut, w)
            wall[row, cols] |= footprint[row, cols]
        else:
            col = int(rng.integers(2, w-2))
            cut = int(rng.integers(2, h-2))
            rows = slice(0, cut) if rng.integers(2) else slice(cut, h)
            wall[rows, col] |= footprint[rows, col]
    return wall, footprint


def _mask_flight(wall, start, direction, count):
    """Construction path uses the wall array, independently of scene parsing."""
    r, c = start
    vy, vx = direction
    paths, corners, kinds = [], [], []
    h, w = wall.shape
    def blocked(y, x):
        return not (0 <= y < h and 0 <= x < w) or bool(wall[y, x])
    for _ in range(count):
        path = [(r, c)]
        for _ in range(h+w+1):
            vertical = bool(vy) and blocked(r+vy, c)
            horizontal = bool(vx) and blocked(r, c+vx)
            front = blocked(r+vy, c+vx)
            if front or (vy and vx and (vertical or horizontal)):
                corners.append(bool(vy and vx and not front and (vertical or horizontal)))
                kinds.append((bool(vertical), bool(horizontal)))
                break
            r, c = r+vy, c+vx
            path.append((r, c))
        else:
            raise AssertionError("ray failed to terminate")
        paths.append(path)
        if vy == 0:
            vx *= -1
        elif vx == 0:
            vy *= -1
        elif vertical or horizontal:
            vy *= -1 if vertical else 1
            vx *= -1 if horizontal else 1
        else:
            vy, vx = -vy, -vx
    return dict(end=(r, c), legs=paths, corner=any(corners), kinds=kinds,
                cells=set(p for path in paths for p in path))


def _make_room(rng, mapping, terminal, direction, witness, test):
    # Walk backward through the visible clock to obtain initial states.
    inverse = {v:k for k, v in mapping.items()}
    state, by_count = terminal, {}
    for n in range(1, len(mapping)):
        state = inverse[state]
        by_count[n] = state
    for attempt in range(70):
        fallback = attempt >= 55
        h = 13 if fallback else int(rng.integers(10 if test else 9, 19 if test else 16))
        w = 15 if fallback else int(rng.integers(11 if test else 9, 20 if test else 18))
        wall, footprint = _wall_shape(rng, h, w, int(rng.integers(0, 3 if test else 2)),
                                     witness, direction, fallback)
        free = footprint & ~wall
        regions = [s for s in _components(free)
                   if all(0 < r < h-1 and 0 < c < w-1 for r, c in s)]
        candidates = {n:[] for n in by_count}
        for region in regions:
            for p in sorted(region):
                for n in by_count:
                    flight = _mask_flight(wall, p, direction, n)
                    if any(len(path) < 2 for path in flight['legs']) or not flight['cells'] <= region:
                        continue
                    # Nonterminal diagonal point contacts would introduce a
                    # special reflection case without a dedicated demonstration.
                    if direction in DIAGONAL and (False, False) in flight['kinds'][:-1]:
                        continue
                    if n > 1 and flight['end'] == flight['legs'][0][-1]:
                        continue
                    candidates[n].append((p, by_count[n], n, flight, region))
        ones = candidates[1]
        if witness:
            ones = [x for x in ones if x[3]['corner']
                    and (x[3]['end'][0]+direction[0], x[3]['end'][1]+direction[1]) in x[4]]
        multi = [x for n in candidates if n > 1 for x in candidates[n]]
        if witness:
            multi = [x for x in multi if
                     sum(x[3]['kinds'][0]) == 1]
        if not ones or not multi:
            continue
        selected = None
        for i in rng.permutation(len(ones))[:60]:
            one = ones[int(i)]
            options = [x for x in multi if one[3]['cells'].isdisjoint(x[3]['cells'])]
            if options:
                selected = [one, options[int(rng.integers(len(options)))]]
                break
        if selected is None:
            continue
        used = selected[0][3]['cells'] | selected[1][3]['cells']
        desired = int(rng.integers(2, 5 if test else 4))
        rest = [x for xs in candidates.values() for x in xs]
        for i in rng.permutation(len(rest)):
            if len(selected) >= desired:
                break
            x = rest[int(i)]
            if used.isdisjoint(x[3]['cells']):
                selected.append(x)
                used |= x[3]['cells']
        return wall, selected
    raise AssertionError("bounded chamber construction failed")


def _pack(rng, objects, bg):
    """Pack actual independent rectangles, then rotate the whole scene."""
    room = objects[0]
    # A guaranteed shelf arrangement bounds the random-packing fallback.
    min_h = room.shape[0] + max(a.shape[0] for a in objects[1:]) + 3
    min_w = max(room.shape[1]+2, sum(a.shape[1] for a in objects[1:])+4)
    height = int(rng.integers(min_h, 31))
    width = int(rng.integers(min_w, 31))
    placed = None
    for _ in range(35):
        occupied = np.zeros((height, width), bool)
        proposal = []
        for i in [0] + list(map(int, rng.permutation([1, 2, 3]))):
            a = objects[i]
            h, w = a.shape
            choices = []
            for r in range(1, height-h):
                for c in range(1, width-w):
                    if not occupied[r-1:r+h+1, c-1:c+w+1].any():
                        choices.append((r, c))
            if not choices:
                break
            r, c = choices[int(rng.integers(len(choices)))]
            occupied[r:r+h, c:c+w] = True
            proposal.append((i, r, c))
        if len(proposal) == 4:
            placed = proposal
            break
    if placed is None:
        placed = []
        order = list(map(int, rng.permutation([1, 2, 3])))
        c, top_h = 1, max(a.shape[0] for a in objects[1:])
        for i in order:
            placed.append((i, 1, c))
            c += objects[i].shape[1] + 1
        placed.append((0, top_h+2, int(rng.integers(1, width-room.shape[1]))))
    a = np.full((height, width), bg, np.int8)
    for i, r, c in placed:
        h, w = objects[i].shape
        a[r:r+h, c:c+w] = objects[i]
    return a


def _make_pair(rng, direction, witness=False, test=False):
    colors = list(map(int, rng.permutation(10)))
    bg, carrier, frame, trail = colors[:4]
    size = int(rng.choice([4, 5])) if test else int(rng.choice([3, 3, 4]))
    clock = colors[4:4+size]
    rng.shuffle(clock)
    mapping = dict(zip(clock, clock[1:]+clock[:1]))
    terminal = int(rng.choice(clock))
    key = _make_key(rng, direction, carrier, trail, terminal)
    refs = _make_reference(rng, clock, mapping, bg, frame, trail, terminal)
    wall, tokens = _make_room(rng, mapping, terminal, direction, witness, test)
    room = np.full(wall.shape, bg, np.int8)
    room[wall] = carrier
    out = room.copy()
    for p, color, _, _, _ in tokens:
        room[p] = color
    for _, _, _, flight, _ in tokens:
        out[flight['end']] = terminal
    # Crop wall-only empty margins before packing and returning the answer.
    r, c, bottom, right = _bbox(np.argwhere(wall))
    room, out = room[r:bottom, c:right], out[r:bottom, c:right]
    a = _pack(rng, [room, key, *refs], bg)
    turns = int(rng.integers(4))
    a, out = np.rot90(a, turns).copy(), np.rot90(out, turns).copy()
    if rng.integers(2):
        a, out = np.fliplr(a).copy(), np.fliplr(out).copy()
    return {'input':a, 'output':out}


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " is outside its legal range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt = _count(num_train, 2, 10, 'num_train')
    nq = _count(num_test, 1, 4, 'num_test')
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    plans = [(DIAGONAL[int(rng.integers(4))], True),
             (CARDINAL[int(rng.integers(4))], False)]
    plans += [(DIRECTIONS[int(rng.integers(8))], False) for _ in range(nt-2)]
    rng.shuffle(plans)
    puzzle, seen = {'train':[], 'test':[]}, set()
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            direction, witness = plans[i] if split == 'train' else (DIRECTIONS[int(rng.integers(8))], False)
            for _ in range(12):
                pair = _make_pair(rng, direction, witness, split == 'test')
                fp = _fingerprint(pair['input'])
                if fp not in seen:
                    puzzle[split].append(pair)
                    seen.add(fp)
                    break
            else:
                raise AssertionError("could not construct distinct inputs")
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _require(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
             and len(puzzle['train']) > len(puzzle['test']), "invalid counts")
    seen, corner_train, one_axis_train, directions = set(), False, False, set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            a, out = _grid(pair['input']), _grid(pair['output'])
            fp = _fingerprint(a)
            _require(fp not in seen, "duplicate input")
            seen.add(fp)
            info = _read(a)
            predicted, flights = _predict(info)
            _require(np.array_equal(predicted, out), "incorrect final chamber")
            terminal, bg = info['key']['terminal'], info['background']
            _require(out.shape == info['room'].shape, "wrong crop")
            _require(np.array_equal(out == info['key']['carrier'], info['wall']), "walls changed")
            _require(np.count_nonzero(out == terminal) == len(info['starts']), "token count changed")
            _require(set(map(int, np.unique(out))) <= {bg, terminal, info['key']['carrier']}, "unexplained output ink")
            used = set()
            for flight in flights:
                _require(used.isdisjoint(flight['cells']), "different token trajectories interact")
                used |= flight['cells']
                _require(all(len(path) >= 2 for path in flight['legs']), "a flight does not move")
                if info['key']['direction'] in DIAGONAL:
                    _require((False, False) not in flight['kinds'][:-1], "undemonstrated point reflection")
            counts = {_steps(info['reference']['mapping'], color, terminal) for _, color, _ in info['starts']}
            _require(1 in counts and max(counts) >= 2, "missing one/multiple-impact comparison")
            first, _ = _predict(info, first_only=True)
            _require(not np.array_equal(first, out), "first-contact shortcut succeeds")
            stationary = info['room'].copy()
            for p, _, _ in info['starts']:
                stationary[p] = terminal
            _require(not np.array_equal(stationary, out), "motion can be omitted")
            if split == 'train':
                directions.add(info['key']['direction'])
                no_corner, _ = _predict(info, corner=False)
                corner_train |= any(f['corner'] for f in flights) and not np.array_equal(no_corner, out)
                if info['key']['direction'] in DIAGONAL:
                    one_axis_train |= any(len(f['legs']) >= 2 and sum(f['kinds'][0]) == 1 for f in flights)
    _require(any(d in CARDINAL for d in directions) and any(d in DIAGONAL for d in directions),
             "training must show cardinal and diagonal motion")
    _require(corner_train, "missing observable conservative-corner demonstration")
    _require(one_axis_train, "missing one-axis reflection demonstration")


def _diagnostics(puzzle):
    """Counterfactual full inputs: keep the target fixed and change the analogy."""
    changed = False
    for pair in puzzle['train'] + puzzle['test']:
        a = pair['input']
        info = _read(a)
        ref = info['reference']
        inverse = {v:k for k, v in ref['mapping'].items()}
        twin = a.copy()
        r, c, b, d = ref['partner']['box']
        source = ref['source']['inside']
        partner = np.full_like(source, info['background'])
        for color, successor in inverse.items():
            partner[source == color] = successor
        twin[r+1:b-1, c+1:d-1] = partner
        expected, flights = _predict(info, mapping=inverse)
        _require(np.array_equal(solve(twin), expected), "counterfactual reference not read")
        counts = {_steps(inverse, color, info['key']['terminal']) for _, color, _ in info['starts']}
        eligible = 1 in counts and max(counts) >= 2
        used = set()
        for flight in flights:
            eligible &= used.isdisjoint(flight['cells']) and all(len(p) >= 2 for p in flight['legs'])
            if info['key']['direction'] in DIAGONAL:
                eligible &= (False, False) not in flight['kinds'][:-1]
            used |= flight['cells']
        first, _ = _predict(info, first_only=True)
        eligible &= not np.array_equal(first, expected)
        if eligible and not np.array_equal(expected, pair['output']):
            changed = True
    _require(changed, "no reference-dependent answer witness")


def self_test():
    """Bounded 42-seed semantic, counterfactual, equivariance and diversity checks."""
    shapes, room_shapes, structures, clocks, counts, states, directions = set(), set(), set(), set(), set(), set(), set()
    irregular = baffles = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _diagnostics(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a, out = pair['input'], pair['output']
                info = _read(a)
                shapes.add(a.shape)
                room_shapes.add(out.shape)
                clocks.add(tuple(sorted(info['reference']['mapping'].items())))
                counts.add(len(info['starts']))
                states.add(len(info['reference']['mapping']))
                directions.add(info['key']['direction'])
                normalized = {info['background']:0}
                structures.add((a.shape, tuple(normalized.setdefault(int(x), len(normalized)) for x in a.flat)))
                wall = info['wall']
                irregular += not (wall[0].all() and wall[-1].all() and wall[:, 0].all() and wall[:, -1].all())
                free = np.zeros(wall.shape, bool)
                for region in info['regions']:
                    for p in region:
                        free[p] = True
                p = np.pad(free, 1)
                baffles += bool(np.any(wall & ((p[:-2, 1:-1] & p[2:, 1:-1])
                                               | (p[1:-1, :-2] & p[1:-1, 2:]))))
        # One fresh equivariance check per seed covers all family directions.
        pair = puzzle['test'][0]
        _require(np.array_equal(solve(np.rot90(pair['input'])), np.rot90(pair['output'])), "rotation failure")
        _require(np.array_equal(solve(np.fliplr(pair['input'])), np.fliplr(pair['output'])), "reflection failure")
    _require(len(shapes) >= 25 and len(room_shapes) >= 35, "insufficient geometric variation")
    _require(len(structures) == 210 and len(clocks) >= 100, "insufficient structural/clock variation")
    _require(states == {3, 4, 5} and counts >= {2, 3}, "missing state/count range")
    _require(directions == set(DIRECTIONS), "missing compass direction")
    _require(irregular >= 25 and baffles >= 25, "insufficient wall variation")
    first = generate_puzzle(0)
    again = generate_puzzle(0)
    _require(all(_fingerprint(x[k]) == _fingerprint(y[k]) for s in first
                 for x, y in zip(first[s], again[s]) for k in ('input', 'output')), "nondeterminism")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            pz = generate_puzzle(1000+10*nt+nq, nt, nq)
            _require(len(pz['train']) == nt and len(pz['test']) == nq, "count mismatch")
            validate_generated_puzzle(pz)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s:[{k:v.copy() for k, v in pair.items()} for pair in first[s]] for s in first}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect output accepted")


if __name__ == '__main__':
    self_test()
