"""Topology-controlled courier: input-local turns steer a rigid, recolored arrow."""
import numpy as np

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape):
        raise ValueError('grid dimensions must be 1..30')
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9)):
        raise ValueError('grid cells must be integer colors 0..9')
    return a.astype(np.int8, copy=True)


def _components(cells, diagonal=False):
    unseen, result = set(cells), []
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        todo, comp = [p], {p}
        while todo:
            r, c = todo.pop()
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    comp.add(q)
                    todo.append(q)
        result.append(comp)
    return result


def _bitmap(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    h = max(r for r, c in cells) - r0 + 1
    w = max(c for r, c in cells) - c0 + 1
    a = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        a[r-r0, c-c0] = True
    return a, (r0, c0)


def _holes(cells):
    """Four-connected cavities of the complete object, with a true exterior."""
    mask, origin = _bitmap(cells)
    pad = np.pad(mask, 1)
    regions = _components(map(tuple, np.argwhere(~pad)))
    return [{(r + origin[0]-1, c + origin[1]-1) for r, c in region}
            for region in regions
            if all(0 < r < pad.shape[0]-1 and 0 < c < pad.shape[1]-1
                   for r, c in region)]


def _rot(p, turns):
    r, c = p
    for _ in range(turns % 4):
        r, c = c, -r
    return r, c


def _arrow(cells):
    """Infer the unique projecting tip and transverse symmetry, without a bank."""
    matches = []
    for tip in sorted(cells):
        for d in range(4):
            mask = {_rot((r-tip[0], c-tip[1]), 1-d) for r, c in cells}
            if any(c > 0 for r, c in mask):
                continue
            if {p for p in mask if p[1] == 0} != {(0, 0)}:
                continue
            if max(abs(r) for r, c in mask) == 0:
                continue
            if {(-r, c) for r, c in mask} == mask:
                matches.append((tuple(sorted(mask)), tip, d))
    if len(matches) != 1:
        raise ValueError('arrow must have one symmetric projecting tip')
    return matches[0]


def _footprint(mask, tip, d):
    return {(tip[0]+r, tip[1]+c) for r, c in (_rot(p, d-1) for p in mask)}


def _inside(cells, shape):
    return all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in cells)


def _distance(p, q, d):
    dr, dc = DIRS[d]
    r, c = q[0]-p[0], q[1]-p[1]
    if (dr == 0 and r == 0 and c*dc > 0) or (dc == 0 and c == 0 and r*dr > 0):
        return abs(r) + abs(c)
    return None


def _decode_regions(grid, panel, work, bg, sep):
    pc = set(int(x) for x in np.unique(grid[panel])) - {bg}
    wc = set(int(x) for x in np.unique(grid[work])) - {bg}
    shared = pc & wc
    if len(shared) != 1 or len(wc) != 3 or not 3 <= len(pc) <= 5:
        raise ValueError('roles do not distinguish key from field')
    ink = next(iter(shared))
    if sep in pc | wc:
        raise ValueError('separator role must be exclusive')
    reference = set(map(tuple, np.argwhere(panel & (grid == ink))))
    mover = set(map(tuple, np.argwhere(work & (grid == ink))))
    if len(_components(reference, True)) != 1 or len(_components(mover, True)) != 1:
        raise ValueError('reference and courier must each be one arrow')
    _, _, ref_direction = _arrow(reference)
    mask, tip, direction = _arrow(mover)
    commands, entries = {}, []
    for color in sorted(pc - {ink}):
        cells = set(map(tuple, np.argwhere(panel & (grid == color))))
        if len(_components(cells)) != 1:
            raise ValueError('each command glyph must be connected')
        _, _, d = _arrow(cells)
        holes = _holes(cells)
        if any(grid[r, c] != bg for region in holes for r, c in region):
            raise ValueError('legend cavities must contain background')
        k = len(holes)
        if k in commands:
            raise ValueError('duplicate topology class in legend')
        commands[k] = ((d-ref_direction) % 4, color)
        entries.append((k, d, color, cells))
    # The peg color has exactly one cell in each union component. This role is
    # geometrically recoverable, rather than determined by a literal color ID.
    other = sorted(wc - {ink})
    station_cells = set(map(tuple, np.argwhere(work & np.isin(grid, other))))
    comps = _components(station_cells)
    candidates = [color for color in other
                  if all(sum(int(grid[p]) == color for p in comp) == 1 for comp in comps)]
    if len(candidates) != 1 or len(comps) < 2:
        raise ValueError('stations need a unique singleton docking color')
    peg = candidates[0]
    neutral = next(c for c in other if c != peg)
    stations = {}
    for comp in comps:
        p = next(p for p in comp if int(grid[p]) == peg)
        holes = _holes(comp)  # the peg is part of the wall, never a hole
        if len(holes) not in commands:
            raise ValueError('every station class must be listed')
        if any(grid[r, c] != bg for region in holes for r, c in region):
            raise ValueError('station cavities must contain only background')
        stations[p] = dict(cells=comp, holes=holes, count=len(holes))
    return dict(bg=bg, sep=sep, ink=ink, neutral=neutral, peg=peg,
                panel=panel, work=work, mask=mask, tip=tip, direction=direction,
                reference=ref_direction, commands=commands, entries=entries,
                stations=stations)


def _parse(grid):
    values, counts = np.unique(grid, return_counts=True)
    bg = int(values[np.argmax(counts)])
    if max(counts) <= grid.size // 2:
        raise ValueError('background must occupy more than half the input')
    interpretations = []
    for axis in (0, 1):
        for line in range(1, grid.shape[axis]-1):
            bar = grid[line, :] if axis == 0 else grid[:, line]
            if int(bar[0]) == bg or not np.all(bar == bar[0]):
                continue
            low = np.zeros(grid.shape, dtype=bool)
            high = np.zeros(grid.shape, dtype=bool)
            if axis == 0:
                low[:line] = True
                high[line+1:] = True
            else:
                low[:, :line] = True
                high[:, line+1:] = True
            for panel, work in ((low, high), (high, low)):
                try:
                    interpretations.append(_decode_regions(grid, panel, work, bg, int(bar[0])))
                except ValueError:
                    continue
    if len(interpretations) != 1:
        raise ValueError('input needs exactly one legend/field interpretation')
    return interpretations[0]


def _trace(a, ignore_topology=False, farthest=False, ignore_reference=False):
    p, d, color = a['tip'], a['direction'], a['ink']
    remaining, route, directions, colors = set(a['stations']), [], [], []
    while True:
        ahead = [(dist, q) for q in remaining
                 if (dist := _distance(p, q, d)) is not None]
        if not ahead:
            break
        _, p = max(ahead) if farthest else min(ahead)
        remaining.remove(p)
        k = a['stations'][p]['count']
        turn, color = a['commands'][k]
        if ignore_reference:
            turn = (turn+a['reference']) % 4
        if not ignore_topology:
            d = (d+turn) % 4
        route.append(p)
        directions.append(d)
        colors.append(color)
    return dict(route=route, directions=directions, colors=colors,
                tip=p, direction=d, color=color, remaining=remaining)


def _finish(grid, a, trace):
    output = grid.copy()
    output[a['work']] = a['bg']
    cells = _footprint(a['mask'], trace['tip'], trace['direction'])
    if not _inside(cells, grid.shape) or any(not a['work'][p] for p in cells):
        raise ValueError('terminal arrow must fit wholly inside the field')
    for p in cells:
        output[p] = trace['color']
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    a = _parse(grid)
    return _finish(grid, a, _trace(a))


def _mover(span=1, extension=1, solid=True, tails=0):
    cells = {(0, -i) for i in range(extension)}
    for depth in range(1, span+1):
        rows = range(-depth, depth+1) if solid else (-depth, depth)
        cells.update((r, -extension+1-depth) for r in rows)
    for length in range(1, tails+1):
        cells.update((r, -extension+1-span-length) for r in (-span, span))
    return tuple(sorted(cells))


def _glyph(k):
    w = max(3, 2*k+1)
    cells = {(r, c-w) for r in (-1, 0, 1) for c in range(w)}
    cells -= {(0, c-w) for c in range(1, 2*k, 2)}
    cells.add((0, 0))
    return tuple(sorted(cells))


def _station(rng, k, rich=False):
    """Procedural objects; contour edits are accepted only if topology survives."""
    if k == 0:
        h, w = int(rng.integers(2, 5)), int(rng.integers(2, 6))
        mask = np.ones((h, w), dtype=bool)
        if rng.random() < .5:
            mask[:-1, 1:-1] = False  # an open U rather than an artificial hole
    else:
        rows = 2 if k == 3 and rng.random() < .55 else 1
        cols = (k+rows-1)//rows
        hs = [int(rng.integers(1, 4 if rich else 3)) for _ in range(rows)]
        ws = [int(rng.integers(1, 4 if rich else 3)) for _ in range(cols)]
        # Limit footprints while still allowing unequal and L-shaped cavities.
        if 2+sum(hs)+rows-1 > 7:
            hs = [1]*rows
        if 2+sum(ws)+cols-1 > 7:
            ws = [1]*cols
        mask = np.ones((2+sum(hs)+rows-1, 2+sum(ws)+cols-1), dtype=bool)
        rstarts, cstarts = [], []
        pos = 1
        for n in hs:
            rstarts.append(pos)
            pos += n+1
        pos = 1
        for n in ws:
            cstarts.append(pos)
            pos += n+1
        for i in range(k):
            rr, cc = i//cols, i % cols
            r, c, h, w = rstarts[rr], cstarts[cc], hs[rr], ws[cc]
            mask[r:r+h, c:c+w] = False
            if rich and h >= 2 and w >= 2 and rng.random() < .6:
                mask[r+h-1, c+w-1] = True  # bent cavity, still one region
    cells = set(map(tuple, np.argwhere(mask)))
    for _ in range(3 if rich else 1):
        boundary = [p for p in sorted(cells) if any((p[0]+dr, p[1]+dc) not in cells
                                                   for dr, dc in DIRS)]
        p = boundary[int(rng.integers(len(boundary)))]
        candidate = cells - {p}
        if len(candidate) >= 4 and len(_components(candidate)) == 1 and len(_holes(candidate)) == k:
            cells = candidate
    turns = int(rng.integers(4))
    cells = {_rot(p, turns) for p in cells}
    choices = sorted(cells)
    # A peg is on an actual foreground cell; its location is not a geometric
    # center convention. Favor central locations but retain edge anchors.
    if rng.random() < .65:
        mr = (min(r for r, c in cells)+max(r for r, c in cells))/2
        mc = (min(c for r, c in cells)+max(c for r, c in cells))/2
        best = min(abs(r-mr)+abs(c-mc) for r, c in cells)
        choices = [(r, c) for r, c in choices if abs(r-mr)+abs(c-mc) == best]
    peg = choices[int(rng.integers(len(choices)))]
    offsets = {(r-peg[0], c-peg[1]) for r, c in cells}
    assert len(_components(offsets)) == 1 and len(_holes(offsets)) == k
    return offsets


def _separate(cells, occupied):
    return all((r+dr, c+dc) not in occupied for r, c in cells
               for dr, dc in ((0, 0),)+DIRS)


def _header(rng, classes, turns, roles, reference=None, min_width=18):
    bg, neutral, peg, sep, ink, *colors = roles
    ref = int(rng.integers(4)) if reference is None else reference
    entries = [(ink, _footprint(_mover(), (0, 0), ref))]
    order = [int(x) for x in rng.permutation(len(classes))]
    for i in order:
        entries.append((colors[i], _footprint(_glyph(classes[i]), (0, 0), (ref+turns[i]) % 4)))
    widths = [max(c for r, c in s)-min(c for r, c in s)+1 for _, s in entries]
    heights = [max(r for r, c in s)-min(r for r, c in s)+1 for _, s in entries]
    width = int(rng.integers(max(min_width, sum(widths)+len(widths)+1), 31))
    ph = max(heights)+2
    header = np.full((ph+1, width), bg, dtype=np.int8)
    header[-1] = sep
    slack = width-(sum(widths)+len(widths)-1)
    c0 = int(rng.integers(1, slack))
    for (color, cells), w, h in zip(entries, widths, heights):
        r0 = int(rng.integers(1, ph-h))
        minr, minc = min(r for r, c in cells), min(c for r, c in cells)
        for r, c in cells:
            header[r0+r-minr, c0+c-minc] = color
        c0 += w+1
    return header, ref


def _permits(q, points, departures):
    # Adding a future station must not steal an earlier nearest arrival.
    for p, nxt, d in zip(points[:-1], points[1:], departures[:-1]):
        dist = _distance(p, q, d)
        if dist is not None and dist <= _distance(p, nxt, d):
            return False
    return True


def _grow(rng, shape, mask, commands, n, rich):
    for _ in range(50):
        d = int(rng.integers(4))
        p = (int(rng.integers(2, shape[0]-2)), int(rng.integers(2, shape[1]-2)))
        initial = _footprint(mask, p, d)
        if not _inside(initial, shape):
            continue
        points, departures, stations = [p], [d], []
        occupied, footprints = set(initial), set(initial)
        for i in range(n):
            options = []
            for k in rng.permutation(sorted(commands)):
                k = int(k)
                turn, color = commands[k]
                if i == 0 and turn not in (1, 3):
                    continue
                offsets = _station(rng, k, rich)
                for dist in range(4, max(shape)):
                    dr, dc = DIRS[d]
                    q = (p[0]+dr*dist, p[1]+dc*dist)
                    cells = {(q[0]+r, q[1]+c) for r, c in offsets}
                    facing = (d+turn) % 4
                    arrow = _footprint(mask, q, facing)
                    if not _inside(cells | arrow, shape) or not _separate(cells, occupied):
                        continue
                    if cells & footprints or not _permits(q, points, departures):
                        continue
                    # Future pegs cannot hide under a past visible courier.
                    if q in footprints:
                        continue
                    options.append((q, cells, k, facing, arrow))
            if not options:
                break
            q, cells, k, facing, arrow = options[int(rng.integers(len(options)))]
            stations.append((q, cells, k))
            points.append(q)
            departures.append(facing)
            occupied |= cells
            footprints |= arrow
            p, d = q, facing
        if len(stations) == n:
            return points, departures, stations, occupied, footprints
    return None


def _add_decoy(rng, shape, state, commands, rich, kind):
    points, ds, stations, occupied, footprints = state
    candidates = []
    if kind == 'far':
        origins = [(points[0], ds[0])]
    elif kind == 'branch':
        origins = [(points[i], d) for i in range(1, len(points))
                   for d in range(4) if d != ds[i]]
    else:
        origins = []
        for _ in range(60):
            candidates.append((int(rng.integers(shape[0])), int(rng.integers(shape[1]))))
    for p, d in origins:
        for dist in range(4, max(shape)):
            dr, dc = DIRS[d]
            candidates.append((p[0]+dr*dist, p[1]+dc*dist))
    rng.shuffle(candidates)
    for q in candidates:
        if q in points or _distance(points[-1], q, ds[-1]) is not None:
            continue
        if not _permits(q, points, ds):
            continue
        for k in rng.permutation(sorted(commands)):
            k = int(k)
            offsets = _station(rng, k, rich)
            cells = {(q[0]+r, q[1]+c) for r, c in offsets}
            if _inside(cells, shape) and _separate(cells, occupied) and not cells & footprints:
                stations.append((q, cells, k))
                occupied |= cells
                return True
    return False


def _render(header, shape, roles, mask, state, commands, rotation):
    bg, neutral, peg, sep, ink, *colors = roles
    points, ds, stations, _, _ = state
    work = np.full(shape, bg, dtype=np.int8)
    for p, cells, k in stations:
        for q in cells:
            work[q] = neutral
        work[p] = peg
    for p in _footprint(mask, points[0], ds[0]):
        work[p] = ink
    inp = np.concatenate((header, work), axis=0)
    out = inp.copy()
    out[header.shape[0]:] = bg
    terminal_color = commands[stations[len(points)-2][2]][1]
    for r, c in _footprint(mask, points[-1], ds[-1]):
        out[r+header.shape[0], c] = terminal_color
    return {'input': np.rot90(inp, rotation).copy(), 'output': np.rot90(out, rotation).copy()}


def _fallback(rng, roles, mask, n, rotation, ordinal, avoid_terminal=None):
    # Guaranteed spacious construction, also used directly in bounded tests.
    labels = [0, 2, 1] if avoid_terminal == 1 else [0, 1, 2]
    classes, turns = labels, [1, 3, 1]
    reference = int(rng.choice([d for d in range(4) if d != rotation % 4]))
    header, _ = _header(rng, classes, turns, roles, reference, min_width=24)
    shape = (30-header.shape[0], header.shape[1])
    assert shape[0] >= 20 and shape[1] >= 23
    top = 4 + (ordinal//3) % 2
    mid = 11 + ordinal % 2
    x = 10 + ordinal//6
    end = shape[1]-5
    points = [(top, 4+ordinal % 3), (top, x), (mid, x)]
    ds = [1, 2, 1]
    ks = [0, 1]
    if n >= 3:
        points.append((mid, end))
        ds.append(2)
        ks.append(2)
    if n >= 4:
        points.append((mid+6, end))
        ds.append(1 if n == 4 else 3)
        ks.append(1 if n == 4 else 0)
    if n >= 5:
        points.append((mid+6, x))
        ds.append(2)
        ks.append(1)
    ks = [labels[k] for k in ks]
    base = [np.ones((3, 3), dtype=bool),
            np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
            np.array([[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]], dtype=bool)]
    anchors = [(1, 1), (1, 0), (1, 2)]
    stations = []
    for q, k in zip(points[1:], ks):
        offsets = {(int(r)-anchors[k][0], int(c)-anchors[k][1]) for r, c in np.argwhere(base[k])}
        stations.append((q, {(q[0]+r, q[1]+c) for r, c in offsets}, k))
    decoy = (top, end)
    stations.append((decoy, {(decoy[0]+r, decoy[1]+c) for r in (-1, 0, 1) for c in (-1, 0, 1)}, 0))
    occupied = set().union(*(s[1] for s in stations))
    state = points, ds, stations, occupied, set()
    commands = {k: (t, roles[5+i]) for i, (k, t) in enumerate(zip(classes, turns))}
    return _render(header, shape, roles, mask, state, commands, rotation)


def _make_pair(rng, roles, mask, n, rich, rotation, ordinal, teach_far=False, avoid_terminal=None):
    for _ in range(24):
        size = int(rng.integers(2, 5))
        classes = [int(x) for x in rng.choice(4, size, replace=False)]
        # A leg turn is always visible; other commands vary independently.
        turns = [1, 3] + [int(x) for x in rng.choice([0, 2], size-2, replace=False)]
        rng.shuffle(turns)
        commands = {k: (t, roles[5+i]) for i, (k, t) in enumerate(zip(classes, turns))}
        reference = int(rng.choice([d for d in range(4) if d != rotation % 4])) if teach_far else None
        header, _ = _header(rng, classes, turns, roles, reference)
        field_min = 12 if n == 2 else 15 if n == 3 else 17
        height = int(rng.integers(max(20, header.shape[0]+field_min), 31))
        shape = height-header.shape[0], header.shape[1]
        state = _grow(rng, shape, mask, commands, n, rich)
        if state is None:
            continue
        if teach_far and not _add_decoy(rng, shape, state, commands, rich, 'far'):
            continue
        if not _add_decoy(rng, shape, state, commands, rich, 'branch'):
            continue
        for _ in range(int(rng.integers(3))):
            _add_decoy(rng, shape, state, commands, rich, 'free')
        pair = _render(header, shape, roles, mask, state, commands, rotation)
        try:
            a = _parse(pair['input'])
        except ValueError:
            continue
        tr = _trace(a)
        if len(tr['route']) != n or not np.array_equal(_finish(pair['input'], a, tr), pair['output']):
            continue
        if a['stations'][tr['route'][-1]]['count'] == avoid_terminal:
            continue
        if teach_far and _trace(a, farthest=True)['tip'] == tr['tip']:
            continue
        if teach_far:
            if len({a['stations'][p]['count'] for p in tr['route']}) < 2:
                continue
            wrong = _trace(a, ignore_reference=True)
            if (wrong['tip'], wrong['direction'], wrong['color']) == (tr['tip'], tr['direction'], tr['color']):
                continue
        return pair
    return _fallback(rng, roles, mask, n, rotation, ordinal, avoid_terminal)


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _key(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    # Distribution choices, never a latent instruction for solve.
    stable = bool(rng.random() < .25)
    roles = [int(x) for x in rng.permutation(10)]
    base_rotation = int(rng.integers(4))
    varied_side = bool(rng.random() < .55)
    puzzle, seen, first_terminal = {'train': [], 'test': []}, set(), None
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            if not stable:
                roles = [int(x) for x in rng.permutation(10)]
            if split == 'train' and i == 0:
                mask, n = _mover(1, int(rng.integers(1, 3)), True), 3
            elif split == 'train' and i == 1:
                mask, n = _mover(2, 1, True), 2
            else:
                mask = _mover(int(rng.integers(1, 3)), int(rng.integers(1, 3)),
                              bool(rng.integers(2)), int(rng.integers(2)))
                n = int(rng.integers(2, 4))
            if split == 'test':
                n = int(rng.integers(4, 6))
            rich = bool(rng.random() < (.75 if split == 'test' else .35))
            rotation = int(rng.integers(4)) if varied_side else base_rotation
            for attempt in range(12):
                pair = _make_pair(rng, roles, mask, n, rich, rotation, len(seen),
                                  teach_far=split == 'train' and i == 0,
                                  avoid_terminal=first_terminal if split == 'train' and i == 1 else None)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                # Twenty distinct bijections of the visible roles versus at
                # most thirteen earlier inputs guarantee an unused rendering.
                base = _fallback(rng, roles, mask, n, rotation, len(seen),
                                 first_terminal if split == 'train' and i == 1 else None)
                found = False
                for sign in (1, -1):
                    for offset in range(10):
                        permutation = ((offset+sign*np.arange(10)) % 10).astype(np.int8)
                        candidate = {k: permutation[v] for k, v in base.items()}
                        key = _key(candidate['input'])
                        if key not in seen:
                            pair, found = candidate, True
                            break
                    if found:
                        break
                assert found
            seen.add(key)
            puzzle[split].append(pair)
            if split == 'train' and i == 0:
                a = _parse(pair['input'])
                first_terminal = a['stations'][_trace(a)['route'][-1]]['count']
    validate_generated_puzzle(puzzle)
    return puzzle


def measure_pair(pair):
    grid = _grid(pair['input'])
    a = _parse(grid)
    tr = _trace(a)
    no_turn = _trace(a, ignore_topology=True)
    far = _trace(a, farthest=True)
    wrong_ref = _trace(a, ignore_reference=True)
    ds = [a['direction']] + tr['directions']
    points = [a['tip']] + tr['route']
    return dict(shape=grid.shape, mask=a['mask'], palette=(a['bg'], a['neutral'], a['peg'], a['sep'], a['ink']),
                commands=tuple(sorted((k, t) for k, (t, c) in a['commands'].items())),
                reference=a['reference'], visited=len(tr['route']), stations=len(a['stations']),
                unvisited=len(tr['remaining']), initial=a['direction'], final=tr['direction'],
                turns=tuple((b-a) % 4 for a, b in zip(ds, ds[1:])),
                distances=tuple(abs(p[0]-q[0])+abs(p[1]-q[1]) for p, q in zip(points, points[1:])),
                station_counts=tuple(sorted(s['count'] for s in a['stations'].values())),
                visited_classes=tuple(a['stations'][p]['count'] for p in tr['route']),
                terminal_class=a['stations'][tr['route'][-1]]['count'],
                hole_areas=tuple(sorted(len(h) for s in a['stations'].values() for h in s['holes'])),
                no_topology_diff=(no_turn['tip'], no_turn['direction'], no_turn['color']) != (tr['tip'], tr['direction'], tr['color']),
                farthest_diff=(far['tip'], far['direction'], far['color']) != (tr['tip'], tr['direction'], tr['color']),
                reference_diff=(wrong_ref['tip'], wrong_ref['direction'], wrong_ref['color']) != (tr['tip'], tr['direction'], tr['color']),
                route=tuple((p[0]-points[0][0], p[1]-points[0][1]) for p in points))


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, train, tests = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            grid, output = _grid(pair['input']), _grid(pair['output'])
            assert grid.shape == output.shape
            key = _key(grid)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            a = _parse(grid)
            tr = _trace(a)
            assert len(tr['route']) >= 2, 'journey must visit at least two stations'
            assert np.array_equal(_finish(grid, a, tr), output), 'incorrect terminal courier'
            assert np.array_equal(grid[~a['work']], output[~a['work']]), 'legend/separator altered'
            outcells = set(map(tuple, np.argwhere(a['work'] & (output != a['bg']))))
            assert outcells == _footprint(a['mask'], tr['tip'], tr['direction']), 'mask must survive rigidly'
            assert all(output[p] == tr['color'] for p in outcells)
            # Actual intermediate arrows must stay in the field and must not
            # cover a station that is still waiting to be visited.
            waiting = set(a['stations'])
            for p, d in zip([a['tip']]+tr['route'], [a['direction']]+tr['directions']):
                waiting.discard(p)
                footprint = _footprint(a['mask'], p, d)
                assert _inside(footprint, grid.shape) and all(a['work'][q] for q in footprint)
                assert not any(footprint & a['stations'][q]['cells'] for q in waiting)
            if split == 'train':
                train.append(measure_pair(pair))
            else:
                tests.append(measure_pair(pair))
    assert len({m['mask'] for m in train}) >= 2, 'demonstrate actual mask preservation'
    assert any(m['no_topology_diff'] for m in train), 'topology must affect answer'
    assert any(m['farthest_diff'] for m in train), 'teach nearest, not farthest'
    assert any(m['reference_diff'] for m in train), 'teach visibly relative turns'
    assert any(m['unvisited'] for m in train), 'teach stopping despite off-ray stations'
    assert len({m['visited'] for m in train}) >= 2, 'vary itinerary length in demonstrations'
    assert any(len(set(m['visited_classes'])) >= 2 for m in train), 'teach multiple topological commands'
    assert len({m['terminal_class'] for m in train}) >= 2, 'demonstrate more than one terminal recoloring class'
    assert min(m['visited'] for m in tests) > max(m['visited'] for m in train), 'tests extend demonstrated journeys'


def self_test():
    # Topology fixtures independent of construction: diagonal escape is closed;
    # an orthogonal opening and an open notch are not cavities.
    diagonal = {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)}
    assert len(_holes(diagonal)) == 1
    assert len(_holes(diagonal-{(0, 1)})) == 0
    assert len(_holes({(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)})) == 0
    # Hand-specified semantic fixture. Its expected final pixels do not come
    # from _trace, _finish, or a sampled route. The reference points right;
    # zero/two holes turn right, one hole turns left.
    fixture = np.zeros((30, 26), dtype=np.int8)
    fixture[8] = 9
    for p in ((1, 1), (2, 1), (3, 1), (2, 2)):
        fixture[p] = 6
    fixture[1:4, 7:10] = 2
    fixture[4, 8] = 2
    fixture[2:5, 13:16] = 3
    fixture[3, 14] = 0
    fixture[1, 14] = 3
    fixture[1:6, 20:23] = 4
    fixture[2, 21] = fixture[4, 21] = 0
    fixture[6, 21] = 4
    for p in ((12, 3), (13, 3), (14, 3), (13, 4)):
        fixture[p] = 6
    fixture[12:15, 9:12] = 5
    fixture[13, 10] = 8
    fixture[19:22, 10:13] = 5
    fixture[20, 11] = 0
    fixture[20, 10] = 8
    fixture[19:22, 19:24] = 5
    fixture[20, 20] = fixture[20, 22] = 0
    fixture[20, 21] = 8
    fixture[12:15, 20:23] = 5
    fixture[13, 21] = 8  # farther initial-ray decoy
    expected = fixture.copy()
    expected[9:] = 0
    for p in ((19, 20), (19, 21), (19, 22), (20, 21)):
        expected[p] = 4
    assert np.array_equal(solve(fixture), expected)
    changed = fixture.copy()
    changed[20, 11] = 5  # close the second station's cavity, changing its turn
    changed_expected = fixture.copy()
    changed_expected[9:] = 0
    for p in ((19, 11), (20, 11), (21, 11), (20, 10)):
        changed_expected[p] = 2
    assert np.array_equal(solve(changed), changed_expected)
    duplicated = fixture.copy()
    duplicated[3, 14] = 3  # two zero-hole legend glyphs are ambiguous
    try:
        solve(duplicated)
    except ValueError:
        pass
    else:
        raise AssertionError('duplicate topology class accepted')
    for span in (1, 2):
        for solid in (False, True):
            for n in range(2, 6):
                pair = _fallback(np.random.default_rng(7), list(range(10)),
                                 _mover(span, 2, solid, 1), n, n % 4, 13)
                a = _parse(pair['input'])
                assert len(_trace(a)['route']) == n
                assert np.array_equal(solve(pair['input']), pair['output'])
    metrics, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in puzzle.values() for p in s))
        metrics.extend(measure_pair(p) for s in puzzle.values() for p in s)
    assert len(signatures) == 42
    assert len({m['shape'] for m in metrics}) >= 20
    assert len({m['mask'] for m in metrics}) >= 6
    assert len({m['route'] for m in metrics}) >= 80
    assert len({m['commands'] for m in metrics}) >= 15
    assert len({m['palette'] for m in metrics}) >= 25
    assert {m['initial'] for m in metrics} == set(range(4))
    assert {m['final'] for m in metrics} == set(range(4))
    assert {m['reference'] for m in metrics} == set(range(4))
    assert max(m['visited'] for m in metrics) >= 4
    assert max(m['stations'] for m in metrics) >= 6
    assert any(len(set(m['distances'])) > 1 for m in metrics)
    assert any(0 in m['turns'] for m in metrics) and any(2 in m['turns'] for m in metrics)
    assert any(1 in m['turns'] and 3 in m['turns'] for m in metrics)
    assert max(area for m in metrics for area in m['hole_areas']) >= 4
    assert any(30 in m['shape'] for m in metrics)
    # All explicit legal counts are cheap enough to test with one seed.
    for n in range(2, 11):
        for t in range(1, min(4, n-1)+1):
            validate_generated_puzzle(generate_puzzle(91, n, t))
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p, q = generate_puzzle(7), generate_puzzle(7)
    assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    # Global rotations and palette permutations keep commands relative and
    # require no generator state, including reference arrows that face left.
    pair = p['train'][0]
    permutation = np.array([7, 3, 9, 1, 5, 0, 8, 2, 6, 4], dtype=np.int8)
    for turns in range(4):
        inp, out = np.rot90(pair['input'], turns), np.rot90(pair['output'], turns)
        assert np.array_equal(solve(inp), out)
        assert np.array_equal(solve(permutation[inp]), permutation[out])
    # Reject corrupted output and a duplicated legend class.
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return {'seeds': 42, 'pairs': len(metrics), 'shapes': len({m['shape'] for m in metrics}),
            'routes': len({m['route'] for m in metrics}), 'masks': len({m['mask'] for m in metrics}),
            'command_maps': len({m['commands'] for m in metrics})}


if __name__ == '__main__':
    print(self_test())
