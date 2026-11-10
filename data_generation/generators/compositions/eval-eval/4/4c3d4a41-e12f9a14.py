"""Comb-controlled ray organ: clipping a tape opens and colors its cap ports.

All geometry and roles are read from the input. No parent module is imported.
The constructor places stacks by a destination formula; solve pushes pixels.
Ray encounters are spatial, with full-vector addition and major-axis-first
rasterization. Authored checks are not independent semantic certification.
"""

from __future__ import annotations

from math import gcd
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "invalid colors")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _orient(a, rotation, reflection=False):
    return np.rot90(np.fliplr(a) if reflection else a, rotation).copy()


def _shell(f):
    a = np.full(f['shape'], f['bg'], dtype=np.int8)
    s, floor = f['s'], f['floor']
    a[floor, f['cols'][0]-1:f['cols'][-1]+2] = s
    for c, roof, ports in zip(f['cols'], f['roofs'], f['ports']):
        a[roof+1:floor, c-1] = a[roof+1:floor, c+1] = s
        a[roof, c-1:c+2] = s
        for direction in ports:
            a[roof, c+direction] = f['bg']
    return a


def _draw_key(a, f, base):
    a[base, f['cols'][0]:f['cols'][-1]+1] = f['s']
    for c, h in zip(f['cols'], f['heights']):
        a[base-h:base, c] = f['s']


def _input(f):
    a = _shell(f)
    for c, roof, tape in zip(f['cols'], f['roofs'], f['tapes']):
        a[roof+1:roof+1+len(tape), c] = tape
    _draw_key(a, f, f['key_base'])
    return a


def _placed(f):
    """Direct construction formula: bottom-align, then discard roof overflow."""
    a = _shell(f)
    base = f['floor']-1
    _draw_key(a, f, base)
    for c, roof, h, tape in zip(f['cols'], f['roofs'], f['heights'], f['tapes']):
        start = base-h-len(tape)
        for offset, color in enumerate(tape):
            r = start+offset
            if r > roof:
                a[r, c] = color
    return a


def _read_canonical(a, bg, s, floor, left, right, key_base):
    cols = tuple(int(c) for c in np.flatnonzero(a[key_base-1, left+1:right] == s)+left+1)
    _require(2 <= len(cols) <= 6 and cols[0] == left+1 and cols[-1] == right-1,
             "key must match the rail endpoints")
    pitch = np.diff(cols)
    _require(np.all(pitch == pitch[0]) and pitch[0] >= 4, "unmatched or touching shafts")
    _require(np.all(a[key_base, left+1:right] == s), "broken key base")
    heights = []
    for c in cols:
        r = key_base-1
        while r >= 0 and a[r, c] == s:
            r -= 1
        heights.append(key_base-r-1)
    _require(min(key_base-h for h in heights) > floor+1, "external key needs a blank separating row")
    roofs, ports, tapes = [], [], []
    base = floor-1
    for c, h in zip(cols, heights):
        readings = []
        for roof in range(1, floor-3):
            cap = a[roof, c-1:c+2]
            if not np.all((cap == bg) | (cap == s)) or not np.any(cap == bg):
                continue
            if not (np.all(a[roof+1:floor, c-1] == s) and
                    np.all(a[roof+1:floor, c+1] == s)):
                continue
            segment = a[roof+1:base, c]
            occupied = np.flatnonzero(segment != bg)
            if not len(occupied) or not np.array_equal(occupied, np.arange(len(occupied))):
                continue
            seq = tuple(int(v) for v in segment[:len(occupied)])
            prefix = next((i for i, color in enumerate(seq) if color != s), len(seq))
            if prefix == len(seq) or s in seq[prefix:]:
                continue
            capacity = base-roof-h-1
            if not (capacity >= 1 and len(seq) >= capacity):
                continue
            readings.append((roof, tuple(i-1 for i, color in enumerate(cap) if color == bg), seq))
        _require(len(readings) == 1, "shaft must have one cap/contact interpretation")
        roof, port, tape = readings[0]
        roofs.append(roof)
        ports.append(port)
        tapes.append(tape)
    f = dict(shape=a.shape, bg=bg, s=s, floor=floor, cols=cols,
             roofs=tuple(roofs), ports=tuple(ports), heights=tuple(heights),
             tapes=tuple(tapes), key_base=key_base)
    _require(np.array_equal(_input(f), a), "unaccounted input cells")
    return f


def _parse(a):
    counts = np.bincount(a.ravel(), minlength=10)
    _require(np.count_nonzero(counts == counts.max()) == 1, "background must be uniquely modal")
    bg = int(np.argmax(counts))
    candidates = []
    # A horizontal reflection is already legal canonical geometry; only four
    # rotations are needed to put the external key below the common rail.
    for rotation in range(4):
        b = np.rot90(a, rotation)
        runs = []
        for r, row in enumerate(b):
            boundaries = [0, *(np.flatnonzero(row[1:] != row[:-1])+1).tolist(), b.shape[1]]
            for l, stop in zip(boundaries, boundaries[1:]):
                if stop-l >= 7 and int(row[l]) != bg:
                    runs.append((r, l, stop-1, int(row[l])))
        for floor, left, right, s in runs:
            for key_base in range(floor+2, b.shape[0]):
                if not np.all(b[key_base, left+1:right] == s):
                    continue
                try:
                    f = _read_canonical(b, bg, s, floor, left, right, key_base)
                except ValueError:
                    continue
                f['rotation'] = rotation
                candidates.append(f)
    _require(len(candidates) == 1, "organ must have one geometric interpretation")
    return candidates[0]


def _push_pixels(a, f):
    """Insert actual key cells; move each tape one row at a time to contact."""
    out = a.copy()
    bg, s, floor = f['bg'], f['s'], f['floor']
    external = np.argwhere(a[floor+1:] != bg)
    external[:, 0] += floor+1
    for r, c in external:
        out[r, c] = bg
    for c, roof in zip(f['cols'], f['roofs']):
        out[roof+1:floor-1, c] = bg
    displacement = floor-1-f['key_base']
    for r, c in external:
        out[r+displacement, c] = s
    for c, roof, h, tape in zip(f['cols'], f['roofs'], f['heights'], f['tapes']):
        cells = [(roof+1+i, color) for i, color in enumerate(tape)]
        tooth_top = floor-1-h
        while cells[-1][0] >= tooth_top:
            cells = [(r-1, color) for r, color in cells if r-1 > roof]
            _require(bool(cells), "empty surviving tape")
        _require(cells[-1][0] == tooth_top-1, "unsupported pre-contact gap")
        for r, color in cells:
            out[r, c] = color
    return out


def _walk(origin, heading, shape):
    scale = max(abs(v) for v in heading)
    result = []
    for step in range(3*max(shape)+1):
        point = tuple(origin[i]+(1 if heading[i] >= 0 else -1)*
                      (step*abs(heading[i])//scale) for i in range(2))
        if not (0 <= point[0] < shape[0] and 0 <= point[1] < shape[1]):
            break
        result.append(point)
    return result


def _encounter(aa, bb):
    hits = []
    for ia, a in enumerate(aa):
        lookup = {p: i for i, p in enumerate(a)}
        mids = {(p[0]+q[0], p[1]+q[1]): i
                for i, (p, q) in enumerate(zip(a, a[1:]))
                if p[0] != q[0] and p[1] != q[1]}
        for ib, b in enumerate(bb):
            for j, p in enumerate(b):
                if p in lookup:
                    i = lookup[p]
                    hits.append((i-1, j-1, ia, ib, 'cell', i, j))
            for j, (p, q) in enumerate(zip(b, b[1:])):
                if p[0] == q[0] or p[1] == q[1]:
                    continue
                midpoint = (p[0]+q[0], p[1]+q[1])
                if midpoint in mids:
                    i = mids[midpoint]
                    if {p, q} != {a[i], a[i+1]}:
                        hits.append((i, j, ia, ib, 'between', i+.5, j+.5))
    return min(hits, key=lambda x: (sum(x[:2]), max(x[:2]), x[2], x[3])) if hits else None


def _emit(mechanical, f, *, original_leads=False):
    groups, colors = [], []
    active = []
    for lane, (c, roof, ports) in enumerate(zip(f['cols'], f['roofs'], f['ports'])):
        color = f['tapes'][lane][0] if original_leads else int(mechanical[roof+1, c])
        active.append(color != f['s'])
        if color == f['s']:
            continue
        for direction in ports:
            identity = len(colors)
            colors.append(color)
            groups.append(dict(v=(-1, direction), members=[(identity, (roof, c+direction))],
                               depth=0, lanes={lane}))
    out = mechanical.copy()
    occupied, diagonals, last = {}, set(), {}
    events = []
    travelled = [0]*len(colors)

    def paint(identity, path):
        for i, p in enumerate(path):
            _require(mechanical[p] == f['bg'], "ray hits the organ")
            _require(p not in occupied or (i == 0 and occupied[p] == identity and last[identity] == p),
                     "ray overlap or self-intersection")
            occupied[p] = identity
            last[identity] = p
            out[p] = colors[identity]
        for p, q in zip(path, path[1:]):
            if p[0] != q[0] and p[1] != q[1]:
                midpoint = (p[0]+q[0], p[1]+q[1])
                slope = 1 if (q[0]-p[0])*(q[1]-p[1]) > 0 else -1
                _require((midpoint, -slope) not in diagonals, "unresolved ray crossing")
                diagonals.add((midpoint, slope))

    for _ in range(len(colors)+1):
        paths = [[_walk(p, g['v'], mechanical.shape) for _, p in g['members']] for g in groups]
        meetings = []
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                hit = _encounter(paths[i], paths[j])
                if hit is not None:
                    _require(groups[i]['v'] != groups[j]['v'], "parallel overlap")
                    meetings.append((i, j, hit))
        if not meetings:
            for g, forecasts in zip(groups, paths):
                for (identity, _), path in zip(g['members'], forecasts):
                    paint(identity, path)
            break
        nearest = {}
        for i, j, hit in meetings:
            nearest[i] = min(nearest.get(i, 1000), hit[0])
            nearest[j] = min(nearest.get(j, 1000), hit[1])
        ready = [(i, j, hit) for i, j, hit in meetings
                 if hit[0] == nearest[i] and hit[1] == nearest[j]]
        _require(bool(ready), "cyclic spatial encounter order")
        i, j, hit = min(ready, key=lambda e: (max(e[2][:2]), sum(e[2][:2]), e[0], e[1]))
        _require(min(hit[:2]) >= 0, "encounter at a port or old bend")
        for k, l, other in meetings:
            if (k, l) != (i, j):
                _require(not ((k in (i, j) and other[0] == nearest[k]) or
                              (l in (i, j) and other[1] == nearest[l])), "ambiguous multiway encounter")
        u, v = groups[i]['v'], groups[j]['v']
        _require(max(map(abs, u)) == max(map(abs, v)) == 1,
                 "established staircase re-encounters are outside this family")
        raw = (u[0]+v[0], u[1]+v[1])
        divisor = gcd(abs(raw[0]), abs(raw[1]))
        _require(divisor != 0, "canceling headings")
        heading = (raw[0]//divisor, raw[1]//divisor)
        kind = 'opposing' if u[1]*v[1] == -1 else 'staircase'
        members = []
        distances = []
        for index, cut in ((i, hit[0]), (j, hit[1])):
            for (identity, _), path in zip(groups[index]['members'], paths[index]):
                _require(cut < len(path)-1, "partially clipped incoming bundle")
                paint(identity, path[:cut+1])
                members.append((identity, path[cut]))
                travelled[identity] += cut
            representative = hit[2] if index == i else hit[3]
            identity = groups[index]['members'][representative][0]
            distances.append(travelled[identity]+(1 if hit[4] == 'cell' else .5))
        depth = 1+max(groups[i]['depth'], groups[j]['depth'])
        lanes = groups[i]['lanes'] | groups[j]['lanes']
        incoming_path = paths[i][hit[2]]
        if hit[4] == 'cell':
            position = incoming_path[hit[0]+1]
        else:
            p, q = incoming_path[hit[0]], incoming_path[hit[0]+1]
            position = ((p[0]+q[0])/2, (p[1]+q[1])/2)
        events.append(dict(kind=kind, intersection=hit[4], unequal=distances[0] != distances[1],
                           depth=depth, heading=heading, lanes=tuple(sorted(lanes)),
                           strands=len(members), colors=tuple(colors[x] for x, _ in members),
                           distances=tuple(distances), position=position, incoming=(u, v)))
        groups = [g for k, g in enumerate(groups) if k not in (i, j)]
        groups.append(dict(v=heading, members=members, depth=depth, lanes=lanes))
    else:
        raise ValueError("ray merger did not terminate")
    return out, dict(active=tuple(active), events=events, ray_cells=len(occupied), strands=len(colors))


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    f = _parse(a)
    canonical = np.rot90(a, f['rotation']).copy()
    mechanical = _push_pixels(canonical, f)
    out, _ = _emit(mechanical, f)
    return np.rot90(out, -f['rotation']).copy()


def _tags(info):
    result = set()
    for e in info['events']:
        result.add((e['kind'], e['intersection']))
        if e['unequal']:
            result.add(('unequal', e['kind'], e['intersection']))
        if e['depth'] > 1:
            result.add(('chain', e['kind']))
    return result


def _base_geometry(rng, n, kind, pitch=None, roof_delta=None):
    allowed = [p for p in (4, 5, 6) if (n-1)*p+5 <= 30]
    pitch = int(rng.choice(allowed)) if pitch is None else pitch
    width_min = (n-1)*pitch+3
    slack = min(4, 30-width_min)
    pad_left = int(rng.integers(1, min(3, slack)+1)) if slack else 0
    first = pad_left+1
    cols = tuple(first+i*pitch for i in range(n))
    roof0 = int(rng.integers(max(7, pitch+4 if kind == 'staircase' else 7), 12))
    roofs = [roof0+int(rng.integers(2)) for _ in cols]
    primary = int(rng.integers(n-1))
    if roof_delta is not None:
        roofs[primary], roofs[primary+1] = roof0, roof0+roof_delta
    floor = max(roofs)+int(rng.integers(6, 9))
    width = min(30, cols[-1]+2+int(rng.integers(1, 4)))
    ports = [(-1,) if i < primary else (1,) for i in range(n)]
    ports[primary] = (-1, 0, 1)
    ports[primary+1] = (-1,) if kind == 'opposing' else (0,)
    return dict(bg=0, s=5, cols=cols, roofs=tuple(roofs), floor=floor,
                ports=tuple(ports), shape=(30, width)), primary


def _finish_geometry(rng, f, max_height=None):
    max_height = max(f['heights']) if max_height is None else max_height
    low = f['floor']+max_height+2
    _require(low <= 29, "key does not fit")
    key_base = int(rng.integers(low, min(29, low+2)+1))
    height = int(rng.integers(key_base+1, min(30, key_base+3)+1))
    return dict(f, key_base=key_base, shape=(height, f['shape'][1]))


def _controlled(rng, n, kind, nonflat):
    """Two identical organs: a taller tooth opens a previously closed valve."""
    for _ in range(100):
        f, primary = _base_geometry(rng, n, kind)
        heights, tapes = [], []
        for i, roof in enumerate(f['roofs']):
            span = f['floor']-1-roof
            gap = int(rng.integers(1, min(2, span-4)+1))
            length = span-1-gap
            if i == primary:
                shift = int(rng.integers(1, min(2, length-2)+1))
                h = gap+shift
                seq = rng.integers(1, 4, length).tolist()
                seq[0], seq[shift], seq[-1] = 1, 2, 3
            elif i == primary+1:
                h = gap
                seq = [5]+rng.integers(1, 4, length-1).tolist()
                seq[1] = 3
            else:
                h = int(rng.integers(gap, min(4, span-3)+1))
                shift = h-gap
                prefix = int(rng.integers(0, length))
                seq = [5]*prefix+rng.integers(1, 4, length-prefix).tolist()
            heights.append(h)
            tapes.append(tuple(seq))
        high = list(heights)
        high[primary+1] += 1
        if nonflat and (len(set(heights)) == 1 or len(set(high)) == 1):
            continue
        try:
            low = _finish_geometry(rng, dict(f, heights=tuple(heights), tapes=tuple(tapes)), max(high))
            high = dict(low, heights=tuple(high))
            _, before = _emit(_placed(low), low)
            _, after = _emit(_placed(high), high)
            _require(not before['events'] and len(after['events']) == 1, "unclear controlled contrast")
            _require(after['events'][0]['kind'] == kind, "controlled encounter changed kind")
            return low, high
        except ValueError:
            continue
    raise ValueError("bounded controlled construction exhausted")


def _random_scene(rng, n, kind, *, flat=False, nonflat=False, busy=True, safe_geometry=None):
    f, primary = _base_geometry(rng, n, kind,
                                pitch=safe_geometry[0] if safe_geometry else None,
                                roof_delta=safe_geometry[1] if safe_geometry else None)
    cap = min(5, min(f['floor']-1-r-3 for r in f['roofs']), 27-f['floor'])
    _require(cap >= 2, "insufficient tape capacity")
    mode = str(rng.choice(('random', 'levels', 'plateau', 'rising', 'falling')))
    heights = rng.integers(1, cap+1, n).tolist()
    if flat:
        heights = [int(rng.integers(1, cap+1))]*n
    elif mode == 'levels':
        low, high = sorted(rng.choice(np.arange(1, cap+1), 2, replace=False).tolist())
        heights = rng.choice((low, high), n).tolist()
    elif mode == 'plateau' and n >= 3:
        high = int(rng.integers(2, cap+1))
        heights = rng.integers(1, high, n).tolist()
        index = int(rng.integers(n-1))
        heights[index:index+2] = [high, high]
    elif mode in ('rising', 'falling'):
        heights.sort(reverse=mode == 'falling')
    _require(not nonflat or len(set(heights)) > 1, "training profile holdout")
    tapes = []
    for i, (roof, h) in enumerate(zip(f['roofs'], heights)):
        capacity = f['floor']-1-roof-h-1
        shift = int(rng.integers(h+1))
        length = capacity+shift
        active = i in (primary, primary+1) or rng.random() < .7
        prefix = int(rng.integers(min(shift, length-1)+1)) if active else int(rng.integers(shift+1, length))
        seq = [5]*prefix+rng.integers(1, 5, length-prefix).tolist()
        if i == primary:
            seq[shift] = 2
        if i == primary+1:
            seq[shift] = 3
        tapes.append(tuple(seq))
    ports = list(f['ports'])
    if busy and rng.random() < .75:
        for i in range(n):
            if rng.random() < .5:
                count = int(rng.integers(1, 4))
                chosen = set(int(x) for x in rng.choice((-1, 0, 1), count, replace=False))
                if i == primary:
                    chosen.add(1)
                if i == primary+1:
                    chosen.add(-1 if kind == 'opposing' else 0)
                ports[i] = tuple(sorted(chosen))
    f = _finish_geometry(rng, dict(f, ports=tuple(ports), heights=tuple(heights), tapes=tuple(tapes)))
    _, info = _emit(_placed(f), f)
    _require(any(e['kind'] == kind for e in info['events']), "requested collision not visible")
    return f, info


def _recolor(rng, a, b, fixed_roles=None):
    roles = tuple(int(v) for v in rng.choice(10, 2, replace=False)) if fixed_roles is None else fixed_roles
    palette = [c for c in range(10) if c not in roles]
    rng.shuffle(palette)
    lookup = np.arange(10, dtype=np.int8)
    lookup[0], lookup[5] = roles
    for old, new in zip((1, 2, 3, 4), palette):
        lookup[old] = new
    return lookup[a], lookup[b], roles


def _pair(f, rng, rotation, reflection, roles=None):
    a = _input(f)
    b, _ = _emit(_placed(f), f)
    a, b, roles = _recolor(rng, a, b, roles)
    return {'input': _orient(a, rotation, reflection), 'output': _orient(b, rotation, reflection)}, roles


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def _inspect(pair):
    a = _grid(pair['input'])
    f = _parse(a)
    canonical = np.rot90(a, f['rotation']).copy()
    mechanical = _push_pixels(canonical, f)
    out, info = _emit(mechanical, f)
    info.update(f=f, mechanical=mechanical, canonical=canonical,
                output=np.rot90(out, -f['rotation']).copy())
    return info


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    count_shift = bool(rng.random() < .7)
    first_kind = str(rng.choice(('opposing', 'staircase')))
    n = int(rng.choice((2, 3))) if count_shift else int(rng.choice((3, 4)))
    low, high = _controlled(rng, n, first_kind, not count_shift)
    rotation, reflection = int(rng.integers(4)), bool(rng.integers(2))
    # The controlled pair has exactly the same colors as well as hardware.
    coloring_seed = int(rng.integers(2**32))
    p0, _ = _pair(low, np.random.default_rng(coloring_seed), rotation, reflection)
    p1, _ = _pair(high, np.random.default_rng(coloring_seed), rotation, reflection)
    puzzle = {'train': [p0, p1], 'test': []}
    seen = {_key(p0['input']), _key(p1['input'])}
    taught = _tags(_inspect(p1))
    known = {first_kind}
    for index in range(2, nt):
        kind = ('staircase' if first_kind == 'opposing' else 'opposing') if index == 2 else str(rng.choice(tuple(sorted(known))))
        for attempt in range(120):
            try:
                n = int(rng.choice((2, 3))) if count_shift else int(rng.integers(3, 6))
                f, _ = _random_scene(rng, n, kind, nonflat=not count_shift, busy=attempt < 80)
                pair, _ = _pair(f, rng, int(rng.integers(4)), bool(rng.integers(2)))
                if _key(pair['input']) in seen:
                    continue
                info = _inspect(pair)
                puzzle['train'].append(pair)
                seen.add(_key(pair['input']))
                taught |= _tags(info)
                known.add(kind)
                break
            except ValueError:
                continue
        else:
            raise ValueError("bounded training construction exhausted")
    templates = []
    for pair in puzzle['train']:
        info = _inspect(pair)
        f = info['f']
        for e in info['events']:
            if e['depth'] == 1 and len(e['lanes']) == 2:
                i, j = e['lanes']
                if j == i+1:
                    templates.append((e['kind'], f['cols'][1]-f['cols'][0], f['roofs'][j]-f['roofs'][i]))
    for _ in range(ne):
        for attempt in range(160):
            kind = str(rng.choice(tuple(sorted(known))))
            try:
                n = int(rng.integers(4, 7)) if count_shift else int(rng.integers(2, 7))
                safe = None
                if attempt >= 80:
                    kind, pitch, delta = templates[int(rng.integers(len(templates)))]
                    safe = pitch, delta
                    if (n-1)*pitch+5 > 30:
                        n = 4 if count_shift else 3
                f, _ = _random_scene(rng, n, kind, flat=not count_shift,
                                     busy=attempt < 80, safe_geometry=safe)
                pair, _ = _pair(f, rng, int(rng.integers(4)), bool(rng.integers(2)))
                if _key(pair['input']) in seen:
                    continue
                info = _inspect(pair)
                if not _tags(info) <= taught:
                    continue
                puzzle['test'].append(pair)
                seen.add(_key(pair['input']))
                break
            except ValueError:
                continue
        else:
            raise ValueError("bounded test construction exhausted")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _hardware(f):
    return tuple((k, f[k]) for k in ('shape', 'bg', 's', 'floor', 'cols', 'roofs', 'ports', 'tapes', 'key_base'))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, taught, needed, training = set(), set(), set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape and _key(a) not in seen, "shape change or duplicate input")
            seen.add(_key(a))
            info = _inspect(pair)
            f, mechanical = info['f'], info['mechanical']
            _require(np.array_equal(b, info['output']), "incorrect answer")
            _require(np.array_equal(mechanical, _placed(f)), "pixel pushing differs from destination formula")
            _require(info['ray_cells'] >= 3, "no meaningful emission")
            _require(np.all(np.rot90(b, f['rotation'])[mechanical != f['bg']] == mechanical[mechanical != f['bg']]),
                     "ray changed mechanical foreground")
            # Each open shaft samples precisely the first un-clipped tape cell.
            for c, r, h, seq, active in zip(f['cols'], f['roofs'], f['heights'], f['tapes'], info['active']):
                capacity = f['floor']-r-h-2
                shift = len(seq)-capacity
                _require(tuple(int(v) for v in mechanical[r+1:r+1+capacity, c]) == seq[shift:],
                         "sequence not preserved under clipping")
                _require(active == (seq[shift] != f['s']), "incorrect valve state")
            if split == 'train':
                taught |= _tags(info)
                training.append(info)
            else:
                needed |= _tags(info)
                _require(bool(info['events']), "test must exercise ray interaction")
    _require(needed <= taught, "test uses an untaught encounter type")
    _require(any(i['events'] for i in training), "training lacks collisions")
    _require(any(len(set(e['colors'])) > 1 for i in training for e in i['events']), "collision identity not visible")
    _require(any(not all(i['active']) for i in training), "no closed valve demonstration")
    controlled = [(a, b) for i, a in enumerate(training) for b in training[i+1:]
                  if _hardware(a['f']) == _hardware(b['f']) and a['active'] != b['active']]
    _require(bool(controlled), "missing hardware-identical activation contrast")
    _require(any(len(set(a['f']['heights'])) > 1 for a in training), "no lane-specific profile")
    # Ablations concern emitted pixels, not merely the always-moving key.
    sensitive = False
    for info in training:
        try:
            alternative, _ = _emit(info['mechanical'], info['f'], original_leads=True)
            canonical_out = np.rot90(info['output'], info['f']['rotation'])
            sensitive |= not np.array_equal(alternative, canonical_out)
        except ValueError:
            sensitive = True  # Unmoved sources may also imply invalid crossings.
    _require(sensitive, "ignoring tape displacement leaves all emissions unchanged")
    _require(any(not np.array_equal(i['mechanical'], np.rot90(i['output'], i['f']['rotation'])) for i in training),
             "omitting ray emission does not change an answer")


def _regressions():
    f = dict(shape=(23, 15), bg=0, s=5, cols=(3, 9), roofs=(8, 9), floor=16,
             ports=((1,), (-1,)), heights=(2, 2),
             tapes=((1, 1, 2, 4, 4, 4), (5, 3, 4, 4)), key_base=21)
    mechanical = _placed(f)
    expected = mechanical.copy()
    expected[8, 4] = 2
    expected[9, 8] = expected[8, 7] = 3
    expected[:8, 5] = 2
    expected[:8, 6] = 3
    _require(np.array_equal(solve(_input(f)), expected), "unequal half-cell bend regression")
    _require(tuple(mechanical[9:13, 3]) == (2, 4, 4, 4), "ordered multicolor clipping regression")
    closed = dict(f, heights=(2, 1))
    out = solve(_input(closed))
    _require(out[10, 9] == 5 and out[9, 8] == 0, "stationary closed valve regression")
    # Cardinal + diagonal: keep the 2:1 slope and start with the major step.
    g = dict(f, shape=(26, 16), roofs=(10, 10), floor=18, key_base=23,
             ports=((1,), (0,)), tapes=((1, 1, 2, 4, 4, 4), (1, 1, 3, 4, 4, 4)))
    expected = _placed(g)
    for r in range(6, 11):
        expected[r, 14-r] = 2
        expected[r, 9] = 3
    for r, c in ((5, 8), (4, 9), (3, 9), (2, 10), (1, 10), (0, 11)):
        expected[r, c], expected[r, c+1] = 2, 3
    _require(np.array_equal(solve(_input(g)), expected), "major-axis-first staircase regression")
    for fixture, expected in ((f, solve(_input(f))), (g, expected)):
        for rotation in range(4):
            for reflection in (False, True):
                mapping = np.array([(3*i+rotation+int(reflection)) % 10 for i in range(10)], dtype=np.int8)
                a = _orient(mapping[_input(fixture)], rotation, reflection)
                b = _orient(mapping[expected], rotation, reflection)
                _require(np.array_equal(solve(a), b), "rotation/recoloring changed semantics")


def self_test():
    _regressions()
    shapes, signatures, normalized, lane_counts, pitches, profiles, roofs, colors = (set() for _ in range(8))
    tags, events, masks = set(), [], set()
    stationary = clipped = closed = multicolor = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for split in puzzle.values() for p in split))
        for split in puzzle.values():
            for pair in split:
                info = _inspect(pair)
                f = info['f']
                _require(np.array_equal(solve(pair['input']), pair['output']), "input-only solve failed")
                shapes.add(pair['input'].shape)
                lane_counts.add(len(f['cols']))
                pitches.add(f['cols'][1]-f['cols'][0])
                profiles.add(f['heights'])
                roofs.add(tuple(r-min(f['roofs']) for r in f['roofs']))
                colors.add((f['bg'], f['s']))
                masks.add(info['active'])
                tags |= _tags(info)
                events += info['events']
                rolemap = {}
                normalized.add((pair['input'].shape, tuple(rolemap.setdefault(int(v), len(rolemap)) for v in pair['input'].ravel())))
                closed += sum(not v for v in info['active'])
                for r, h, seq in zip(f['roofs'], f['heights'], f['tapes']):
                    shift = len(seq)-(f['floor']-r-h-2)
                    stationary += shift == 0
                    clipped += shift > 0
                    multicolor += len(set(v for v in seq if v != f['s'])) > 1
    _require(len(signatures) == 42 and len(normalized) >= 180, "structural diversity collapsed")
    _require(lane_counts == set(range(2, 7)) and pitches == {4, 5, 6}, "geometry diversity collapsed")
    _require(len(shapes) >= 35 and len(profiles) >= 90 and len(roofs) >= 30, "shape/profile/cap diversity collapsed")
    _require(len(colors) >= 45 and len(masks) >= 25, "palette/activation diversity collapsed")
    _require(stationary >= 42 and clipped >= 150 and closed >= 42 and multicolor >= 150, "mechanical evidence collapsed")
    _require({('opposing', 'cell'), ('opposing', 'between'), ('staircase', 'cell')} <= tags,
             "encounter geometry missing")
    _require(any(e['unequal'] for e in events), "stepped caps have no spatial effect")
    _require(any(e['depth'] > 1 for e in events), "no subsequent bundle encounters")
    repeat_a, repeat_b = generate_puzzle(123, 10, 4), generate_puzzle(123, 10, 4)
    _require(all(np.array_equal(a[k], b[k]) for split in repeat_a
                 for a, b in zip(repeat_a[split], repeat_b[split]) for k in ('input', 'output')),
             "non-deterministic generation")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(77+nt*10+ne, nt, ne))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    corrupt = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in split]
               for s, split in repeat_a.items()}
    corrupt['test'][0]['output'][0, 0] = (int(corrupt['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
