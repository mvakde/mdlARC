"""Symmetry couriers: transport empty guide frames and illustrate their twins.

Only standard Python and NumPy. The public solver reads just the visible input.
Generation uses array operations; decoding uses signed coordinate maps.
"""

from itertools import combinations
import numpy as np


SIDES = ('T', 'R', 'B', 'L')
NORMAL = {'T': (-1, 0), 'R': (0, 1), 'B': (1, 0), 'L': (0, -1)}
KINDS = {'H': np.array([[-1, 0], [0, 1]]),
         'V': np.array([[1, 0], [0, -1]]),
         'P': np.array([[-1, 0], [0, -1]])}
D4 = ('id', 'ccw', 'half', 'cw', 'lr', 'ud', 'diag', 'anti')
MATRICES = (np.eye(2, dtype=int), np.array([[0, -1], [1, 0]]),
            KINDS['P'], np.array([[0, 1], [-1, 0]]),
            KINDS['V'], KINDS['H'], np.array([[0, 1], [1, 0]]),
            np.array([[0, -1], [-1, 0]]))


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), 'Invalid grid dimensions')
    _need(np.issubdtype(a.dtype, np.integer), 'Grid must contain integers')
    _need(np.all((0 <= a) & (a <= 9)), 'Grid colors out of range')
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _pick(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    cells.append(p)
                    stack.append(p)
        result.append(sorted(cells))
    return result


def _bbox(cells):
    a = np.asarray(list(cells), dtype=int)
    return (int(a[:, 0].min()), int(a[:, 1].min()),
            int(a[:, 0].max()), int(a[:, 1].max()))


def _rectangle(bounds):
    r, c, h, w = bounds
    return {(r+i, c+j) for i in range(h) for j in range(w)}


def _bars(bounds, guides):
    r, c, h, w = bounds
    cells = {}
    for side, color in guides.items():
        points = {'T': [(r-1, c+j) for j in range(w)],
                  'B': [(r+h, c+j) for j in range(w)],
                  'L': [(r+i, c-1) for i in range(h)],
                  'R': [(r+i, c+w) for i in range(h)]}[side]
        cells.update({p: color for p in points})
    return cells


def _frame(bounds, guides):
    bars = _bars(bounds, guides)
    b = _bbox(_rectangle(bounds) | set(bars))
    footprint = {(r, c) for r in range(b[0], b[2]+1)
                 for c in range(b[1], b[3]+1)}
    return dict(bounds=tuple(bounds), guides=dict(guides), bars=bars,
                box=b, footprint=footprint)


def _adjacent(corner):
    return SIDES[corner], SIDES[(corner+1) % 4]


def _corner(guides):
    return next(i for i in range(4) if set(_adjacent(i)) == set(guides))


def _side_map(matrix):
    return {s: next(t for t, n in NORMAL.items()
                    if tuple(matrix @ NORMAL[s]) == n) for s in SIDES}


def _frame_matrix(source, target):
    _need(set(source.values()) == set(target.values()) and len(source) == len(target) == 2,
          'Guide identities do not match')
    m = np.zeros((2, 2), dtype=int)
    for side, color in source.items():
        dest = next(s for s, v in target.items() if v == color)
        m += np.outer(NORMAL[dest], NORMAL[side])
    _need(any(np.array_equal(m, x) for x in MATRICES), 'Not a unique dihedral frame match')
    return m


def _orient(payload, matrix):
    coordinates = np.indices(payload.shape).reshape(2, -1)
    moved = matrix @ coordinates
    moved -= moved.min(axis=1, keepdims=True)
    result = np.empty(tuple(moved.max(axis=1)+1), dtype=np.int8)
    result[moved[0], moved[1]] = payload.ravel()
    return result


def _array_transform(payload, name):
    if name == 'id':
        return payload.copy()
    if name == 'ccw':
        return np.rot90(payload, 1).copy()
    if name == 'half':
        return np.rot90(payload, 2).copy()
    if name == 'cw':
        return np.rot90(payload, 3).copy()
    if name == 'lr':
        return np.fliplr(payload).copy()
    if name == 'ud':
        return np.flipud(payload).copy()
    if name == 'diag':
        return payload.T.copy()
    if name == 'anti':
        return np.rot90(payload.T, 2).copy()
    raise ValueError('Unknown transform')


def _name(matrix):
    return next(name for name, m in zip(D4, MATRICES) if np.array_equal(matrix, m))


def _symmetric(p, kind, marker):
    r, c = p
    a, b = marker[0]
    return (2*a-r if kind in ('H', 'P') else r,
            2*b-c if kind in ('V', 'P') else c)


def _marker_kind(cells):
    if len(cells) == 1:
        return 'P'
    r0, c0, r1, c1 = _bbox(cells)
    if r0 == r1 and len(cells) == c1-c0+1:
        return 'H'
    if c0 == c1 and len(cells) == r1-r0+1:
        return 'V'
    raise ValueError('Hinge must be a straight line or singleton')


def _eligible(frame, cells, kind):
    r0, c0, r1, c1 = frame['box']
    a, b, aa, bb = _bbox(cells)
    if kind == 'H':
        return a in (r0-1, r1+1) and b <= c1 and bb >= c0
    if kind == 'V':
        return b in (c0-1, c1+1) and a <= r1 and aa >= r0
    return a in (r0-1, r1+1) and b in (c0-1, c1+1)


def _candidate_frame(horizontal, vertical):
    hc, hp = horizontal
    vc, vp = vertical
    row = hp[0][0]
    start, end = hp[0][1], hp[-1][1]
    col = vp[0][1]
    top, bottom = vp[0][0], vp[-1][0]
    hs = 'T' if top == row+1 else 'B' if bottom == row-1 else None
    vs = 'L' if col == start-1 else 'R' if col == end+1 else None
    if hs is None or vs is None:
        return None
    return _frame((top, start, len(vp), len(hp)), {hs: hc, vs: vc})


def _decode(input_grid):
    inp = _grid(input_grid)
    bars = {}
    for color in sorted(set(map(int, np.unique(inp))) - {0, 5}):
        components = _components(inp == color)
        eligible = []
        for comp in components:
            a, b, aa, bb = _bbox(comp)
            if len(comp) >= 3 and a == aa and len(comp) == bb-b+1:
                eligible.append(('H', comp))
            elif len(comp) >= 3 and b == bb and len(comp) == aa-a+1:
                eligible.append(('V', comp))
        if sum(len(p) for _, p in eligible) == np.count_nonzero(inp == color):
            bars[color] = eligible
    decompositions = []
    for colors in combinations(sorted(bars), 2):
        horizontal, vertical = [], []
        for color in colors:
            for kind, points in bars[color]:
                (horizontal if kind == 'H' else vertical).append((color, points))
        frames = []
        for h in horizontal:
            for v in vertical:
                if h[0] == v[0]:
                    continue
                frame = _candidate_frame(h, v)
                if frame is not None:
                    frames.append(frame)
        if len(frames) < 2:
            continue
        all_bars = set().union(*(set(f['bars']) for f in frames))
        guide_cells = set(map(tuple, np.argwhere(np.isin(inp, colors)).tolist()))
        if all_bars != guide_cells or sum(len(f['bars']) for f in frames) != len(all_bars):
            continue
        footprints = set()
        valid = True
        populated = []
        for f in frames:
            r, c, h, w = f['bounds']
            if min(r, c) < 0 or r+h > inp.shape[0] or c+w > inp.shape[1]:
                valid = False
                break
            if footprints & f['footprint']:
                valid = False
                break
            footprints.update(f['footprint'])
            content = inp[r:r+h, c:c+w]
            if 5 in content or set(map(int, np.unique(content))) & set(colors):
                valid = False
                break
            if np.any(content):
                populated.append(f)
        if not valid or len(populated) != 1:
            continue
        master = populated[0]
        other_ink = set(map(tuple, np.argwhere((inp != 0) & (inp != 5) & ~np.isin(inp, colors)).tolist()))
        if not other_ink <= _rectangle(master['bounds']):
            continue
        decompositions.append((master, [f for f in frames if f is not master], footprints))
    _need(len(decompositions) == 1, 'Input needs one unambiguous master/receiver guide system')
    master, receivers, source_footprint = decompositions[0]
    markers = _components(inp == 5)
    _need(len(markers) == len(receivers), 'Each receiver needs exactly one hinge')
    r, c, h, w = master['bounds']
    payload = inp[r:r+h, c:c+w].copy()
    used, destination_footprint, groups = set(), set(), []
    for cells in markers:
        kind = _marker_kind(cells)
        choices = [i for i, f in enumerate(receivers) if _eligible(f, cells, kind)]
        _need(len(choices) == 1 and choices[0] not in used, 'Ambiguous hinge ownership')
        index = choices[0]
        used.add(index)
        frame = receivers[index]
        moved = {_symmetric(p, kind, cells) for p in frame['footprint']}
        _need(not moved & source_footprint and not moved & destination_footprint,
              'Frame destinations overlap sources or each other')
        _need(all(0 <= x < inp.shape[0] and 0 <= y < inp.shape[1] and inp[x, y] == 0
                  for x, y in moved), 'Destination must be clear and in bounds')
        destination_footprint.update(moved)
        moved_aperture = {_symmetric(p, kind, cells) for p in _rectangle(frame['bounds'])}
        a, b, aa, bb = _bbox(moved_aperture)
        destination_bounds = (a, b, aa-a+1, bb-b+1)
        side_mapping = _side_map(KINDS[kind])
        guides = {side_mapping[s]: color for s, color in frame['guides'].items()}
        matrix = _frame_matrix(master['guides'], guides)
        oriented = _orient(payload, matrix)
        _need(oriented.shape == destination_bounds[2:], 'Guide correspondence gives wrong aperture shape')
        groups.append(dict(frame=frame, marker=cells, kind=kind, matrix=matrix,
                           destination=_frame(destination_bounds, guides), payload=oriented))
    _need(len(used) == len(receivers), 'Unassigned receiver')
    return dict(input=inp, master=master, payload=payload, groups=groups,
                source_footprint=source_footprint, destination_footprint=destination_footprint)


def _prediction(info, alternative=None):
    out = info['input'].copy()
    for g in info['groups']:
        dest = g['destination']
        oriented = g['payload']
        if alternative == 'omit_A':
            dest = g['frame']
            oriented = _orient(info['payload'], _frame_matrix(info['master']['guides'], dest['guides']))
        elif alternative == 'omit_B':
            oriented = _orient(info['payload'], KINDS[g['kind']])
            if oriented.shape != dest['bounds'][2:]:
                return None
        elif alternative == 'translation':
            oriented = _orient(info['payload'], _frame_matrix(info['master']['guides'], g['frame']['guides']))
            dest = _frame(dest['bounds'], g['frame']['guides'])
        r, c, h, w = dest['bounds']
        if any(not (0 <= a < out.shape[0] and 0 <= b < out.shape[1])
               for a, b in dest['bars']):
            return None
        out[r:r+h, c:c+w] = oriented
        for p, color in dest['bars'].items():
            out[p] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve without seed, training pairs, saved examples, or generator state."""
    return _prediction(_decode(input_grid))


def _palette(rng):
    values = [int(x) for x in rng.permutation([1, 2, 3, 4, 6, 7, 8, 9])]
    return values[:2], values[2:2+int(rng.integers(1, 5))]


def _asymmetric(payload):
    return len({_key(_array_transform(payload, t)) for t in D4}) == 8


def _motif(rng, shape, colors, blank=True, witness=False):
    h, w = shape
    # A mixture of sparse and filled stamps; no protected motif corner.
    for _ in range(100):
        a = np.zeros(shape, dtype=np.int8)
        if rng.random() < .35:
            a[:] = colors[0]
        style = int(rng.integers(5))
        row, col = int(rng.integers(h)), int(rng.integers(w))
        for _ in range(int(rng.integers(3, max(4, h*w)))):
            color = int(_pick(rng, colors))
            a[row, col] = color
            if style == 0:
                dr, dc = _pick(rng, ((1, 0), (-1, 0), (0, 1), (0, -1)))
            elif style == 1:
                dr, dc = _pick(rng, ((1, 1), (-1, 1), (1, -1), (-1, -1)))
            elif style == 2:
                dr, dc = (0, 1) if rng.random() < .7 else (1, 0)
            elif style == 3:
                row, col = int(rng.integers(h)), int(rng.integers(w))
                continue
            else:
                row, col = _pick(rng, [(0, j) for j in range(w)] + [(i, w-1) for i in range(h)]
                                      + [(h-1, j) for j in range(w)])
                continue
            row, col = (row+dr) % h, (col+dc) % w
        if blank:
            edge = int(rng.integers(4))
            thickness = 2 if min(shape) >= 5 and rng.random() < .3 else 1
            if edge == 0:
                a[:thickness] = 0
            elif edge == 1:
                a[:, -thickness:] = 0
            elif edge == 2:
                a[-thickness:] = 0
            else:
                a[:, :thickness] = 0
        if np.any(a) and (not witness or _asymmetric(a)):
            return a
    # Guaranteed asymmetric, without requiring a special literal color.
    a = np.zeros(shape, dtype=np.int8)
    a[1, 0] = a[1, 1] = a[2, 0] = colors[0]
    a[2, 1] = colors[-1]
    if not _asymmetric(a):
        a[2, 1] = 0
    _need(_asymmetric(a), 'Failed asymmetric motif construction')
    return a


def _receiver_prototype(rng, shape, guides, kind):
    frame = _frame((0, 0, *shape), guides)
    r0, c0, r1, c1 = frame['box']
    if kind == 'H':
        row = int(_pick(rng, (r0-1, r1+1)))
        length = int(rng.integers(2, c1-c0+4))
        start = int(rng.integers(c0-length+1, c1+1))
        marker = [(row, c) for c in range(start, start+length)]
    elif kind == 'V':
        col = int(_pick(rng, (c0-1, c1+1)))
        length = int(rng.integers(2, r1-r0+4))
        start = int(rng.integers(r0-length+1, r1+1))
        marker = [(r, col) for r in range(start, start+length)]
    else:
        marker = [(int(_pick(rng, (r0-1, r1+1))), int(_pick(rng, (c0-1, c1+1))))]
    dest = {_symmetric(p, kind, marker) for p in frame['footprint']}
    footprint = frame['footprint'] | dest | set(marker)
    return dict(frame=frame, marker=marker, kind=kind, footprint=footprint)


def _shift_frame(frame, dr, dc):
    r, c, h, w = frame['bounds']
    return _frame((r+dr, c+dc, h, w), frame['guides'])


def _pack(rng, prototypes, shape):
    H, W = shape
    blocked = np.zeros(shape, dtype=bool)
    placed = [None] * len(prototypes)
    order = sorted(range(len(prototypes)), key=lambda i: -len(prototypes[i]['footprint']))
    prefer_edge = rng.random() < .35
    for i in order:
        proto = prototypes[i]
        a, b, aa, bb = _bbox(proto['footprint'])
        ph, pw = aa-a+1, bb-b+1
        if ph > H or pw > W:
            return None
        origins = [(r, c) for r in range(H-ph+1) for c in range(W-pw+1)]
        rng.shuffle(origins)
        if prefer_edge:
            origins.sort(key=lambda p: min(p[0], p[1], H-ph-p[0], W-pw-p[1]))
        footprint = np.asarray(sorted(proto['footprint']))
        position = None
        for row, col in origins:
            rr, cc = footprint[:, 0]+row-a, footprint[:, 1]+col-b
            if not blocked[rr, cc].any():
                position = row-a, col-b
                break
        if position is None:
            return None
        dr, dc = position
        q = dict(frame=_shift_frame(proto['frame'], dr, dc))
        if 'kind' in proto:
            q.update(kind=proto['kind'], marker=[(r+dr, c+dc) for r, c in proto['marker']])
        placed[i] = q
        for r, c in proto['footprint']:
            rr, cc = r+dr, c+dc
            blocked[max(0, rr-1):min(H, rr+2), max(0, cc-1):min(W, cc+2)] = True
    return placed


def _render_generated(inp, payload, master, receivers):
    """Generation-side answer: orient into the SOURCE, then map its coordinates.

    This operates on local construction records, not the solver's decoded frame.
    """
    out = inp.copy()
    for g in receivers:
        frame = g['frame']
        matrix = _frame_matrix(master['guides'], frame['guides'])
        local = _array_transform(payload, _name(matrix))
        r, c, h, w = frame['bounds']
        _need(local.shape == (h, w), 'Construction extent mismatch')
        for i in range(h):
            for j in range(w):
                p = _symmetric((r+i, c+j), g['kind'], g['marker'])
                out[p] = local[i, j]
        for p, color in frame['bars'].items():
            out[_symmetric(p, g['kind'], g['marker'])] = color
    return out


def _make_scene(rng, shape, payload, master_guides, plans, large=False):
    prototypes = []
    master = _frame((0, 0, *shape), master_guides)
    prototypes.append(dict(frame=master, footprint=master['footprint']))
    for kind, guides in plans:
        m = _frame_matrix(master_guides, guides)
        local_shape = _array_transform(payload, _name(m)).shape
        prototypes.append(_receiver_prototype(rng, local_shape, guides, kind))
    for attempt in range(14):
        if attempt >= 8:
            canvas = (30, 30)
        else:
            low = 20 if large or len(plans) >= 2 else 13
            canvas = int(rng.integers(low, 31)), int(rng.integers(low, 31))
        placed = _pack(rng, prototypes, canvas)
        if placed is None:
            continue
        master = placed[0]['frame']
        receivers = placed[1:]
        inp = np.zeros(canvas, dtype=np.int8)
        r, c, h, w = master['bounds']
        inp[r:r+h, c:c+w] = payload
        for g in placed:
            for p, color in g['frame']['bars'].items():
                inp[p] = color
            for p in g.get('marker', []):
                inp[p] = 5
        out = _render_generated(inp, payload, master, receivers)
        try:
            info = _decode(inp)
        except ValueError:
            continue
        if np.array_equal(_prediction(info), out):
            return {'input': inp, 'output': out}, (master, receivers)
    return None


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    value = int(value)
    _need(low <= value <= high, name + ' out of range')
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    nt, ne = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    _need(nt > ne, 'num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    guide_colors, motif_colors = _palette(rng)
    stable = rng.random() < .4
    held = int(rng.integers(4))
    corners = [i for i in range(4) if i != held]
    line = str(_pick(rng, ('H', 'V')))
    active = [line, 'P']
    third = 'V' if line == 'H' else 'H'
    add_third = nt >= 3 and rng.random() < .5
    if add_third:
        active.append(third)
    master_corner = int(rng.integers(4))
    master_guides = dict(zip(_adjacent(master_corner), guide_colors))
    # A visible quarter-turn followed by an exact color-swap reflection contrast.
    choices = []
    for corner in corners:
        for reverse in (False, True):
            guides = dict(zip(_adjacent(corner), guide_colors[::-1] if reverse else guide_colors))
            total = KINDS[line] @ _frame_matrix(master_guides, guides)
            if _name(total) in ('ccw', 'cw'):
                choices.append(guides)
    line_guides = _pick(rng, choices)
    point_guides = dict(zip(_adjacent(int(_pick(rng, corners))),
                            guide_colors[::-1] if rng.random() < .5 else guide_colors))
    first = None
    for _ in range(32):
        n = int(rng.integers(3, 5))
        payload = _motif(rng, (n, n), motif_colors, blank=True, witness=True)
        first = _make_scene(rng, payload.shape, payload, master_guides,
                            [(line, line_guides), ('P', point_guides)], large=True)
        if first is not None:
            break
    _need(first is not None, 'Could not pack teaching scene')
    first_pair, (master, receivers) = first
    second_input = first_pair['input'].copy()
    changed = dict(receivers[0])
    old = receivers[0]['frame']
    values = list(old['guides'].values())
    reversed_guides = {s: values[1-i] for i, s in enumerate(old['guides'])}
    changed['frame'] = _frame(old['bounds'], reversed_guides)
    for p, color in changed['frame']['bars'].items():
        second_input[p] = color
    second_output = _render_generated(second_input, payload, master, [changed, receivers[1]])
    puzzle = {'train': [first_pair, {'input': second_input, 'output': second_output}], 'test': []}
    seen = {_key(first_pair['input']), _key(second_input)}
    for split, count in (('train', nt-2), ('test', ne)):
        for index in range(count):
            accepted = None
            for attempt in range(48):
                gc, mc = (guide_colors, motif_colors) if stable else _palette(rng)
                side = int(rng.integers(4))
                mg = dict(zip(_adjacent(side), gc))
                # Sample counts and sizes independently; high counts use smaller
                # prototypes only on late packing retries, never alter the rule.
                number = int(rng.integers(1, 4))
                cap = 6 if number == 1 else 5 if number == 2 else 4
                if attempt >= 24:
                    cap = 3
                h, w = int(rng.integers(3, cap+1)), int(rng.integers(3, cap+1))
                if rng.random() < .3:
                    w = h
                motif = _motif(rng, (h, w), mc, blank=rng.random() < .65,
                                witness=split == 'test' or rng.random() < .8)
                plans = []
                for j in range(number):
                    kind = third if split == 'train' and index == 0 and add_third and j == 0 else str(_pick(rng, active))
                    corner = held if split == 'test' and j == 0 else int(_pick(rng, corners))
                    guides = dict(zip(_adjacent(corner), gc[::-1] if rng.random() < .5 else gc))
                    if split == 'test' and j == 0 and np.array_equal(_frame_matrix(mg, guides), MATRICES[0]):
                        guides = {s: gc[1-gc.index(v)] for s, v in guides.items()}
                    plans.append((kind, guides))
                made = _make_scene(rng, motif.shape, motif, mg, plans)
                if made is not None and _key(made[0]['input']) not in seen:
                    accepted = made[0]
                    break
            _need(accepted is not None, 'Bounded scene generation exhausted')
            seen.add(_key(accepted['input']))
            puzzle[split].append(accepted)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Invalid puzzle keys')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'Splits must be lists')
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, 'Invalid counts')
    seen, decoded = set(), {'train': [], 'test': []}
    witnesses = {'omit_A': False, 'omit_B': False, 'translation': False}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Invalid pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(inp.shape == out.shape, 'Canvas dimensions must be preserved')
            _need(_key(inp) not in seen, 'Duplicate input')
            seen.add(_key(inp))
            info = _decode(inp)
            _need(np.array_equal(out, _prediction(info)), 'Incorrect twin guides or illustration')
            _need(np.array_equal(inp[inp != 0], out[inp != 0]), 'Original colored cells changed')
            _need(all(out[p] == inp[p] for p in info['source_footprint']), 'Original apertures changed')
            allowed = np.zeros(inp.shape, dtype=bool)
            for p in info['destination_footprint']:
                allowed[p] = True
            _need(np.array_equal(inp[~allowed], out[~allowed]), 'Change outside destinations')
            for g in info['groups']:
                r, c, h, w = g['destination']['bounds']
                _need(np.array_equal(np.sort(out[r:r+h, c:c+w], axis=None),
                                     np.sort(info['payload'], axis=None)), 'Motif colors/cells were lost')
            if split == 'train':
                for shortcut in witnesses:
                    alternative = _prediction(info, shortcut)
                    witnesses[shortcut] |= alternative is None or not np.array_equal(alternative, out)
            else:
                _need(_asymmetric(info['payload']), 'Test stamp hides orientation')
                alternative = _prediction(info, 'omit_B')
                _need(alternative is None or not np.array_equal(alternative, out),
                      'Test allows the hinge-only shortcut')
            decoded[split].append(info)
    _need(all(witnesses.values()), 'Training hides one parent contribution')
    train_kinds = {g['kind'] for x in decoded['train'] for g in x['groups']}
    _need({g['kind'] for x in decoded['test'] for g in x['groups']} <= train_kinds,
          'Test hinge operation not demonstrated')
    train_corners = {_corner(g['frame']['guides']) for x in decoded['train'] for g in x['groups']}
    _need(all(any(_corner(g['frame']['guides']) not in train_corners for g in x['groups'])
              for x in decoded['test']), 'No receiver-corner shift')
    # Locate an exact controlled contrast, not merely two different random images.
    contrast = False
    for i, a in enumerate(decoded['train']):
        for b in decoded['train'][i+1:]:
            if a['input'].shape != b['input'].shape or not np.array_equal(a['payload'], b['payload']):
                continue
            if a['master']['bounds'] != b['master']['bounds'] or a['master']['guides'] != b['master']['guides']:
                continue
            for ga in a['groups']:
                for gb in b['groups']:
                    if ga['kind'] != gb['kind'] or ga['marker'] != gb['marker'] or ga['frame']['bounds'] != gb['frame']['bounds']:
                        continue
                    sa, sb = ga['frame']['guides'], gb['frame']['guides']
                    if set(sa) != set(sb) or not all(sa[s] != sb[s] for s in sa):
                        continue
                    allowed = set(ga['frame']['bars'])
                    differences = set(map(tuple, np.argwhere(a['input'] != b['input']).tolist()))
                    if differences != allowed or not _asymmetric(a['payload']):
                        continue
                    names = {_name(ga['matrix']), _name(gb['matrix'])}
                    if names & {'ccw', 'cw'} and names & {'lr', 'ud', 'diag', 'anti'}:
                        contrast = True
    _need(contrast, 'Training lacks the controlled quarter-turn/reflection guide swap')
    return None


def self_test():
    """Bounded 42-seed semantic/diversity audit, counts and mutation checks."""
    shapes, stamp_shapes, counts, colors, matrices = set(), set(), set(), set(), set()
    quadrants, line_sides, master_corners, receiver_corners = set(), set(), set(), set()
    motifs, point_count, overhangs, blank_edges, edge_contact = set(), 0, 0, 0, 0
    puzzles = []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        if seed < 3:
            q = generate_puzzle(seed)
            assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
        for split in ('train', 'test'):
            for pair in p[split]:
                info = _decode(pair['input'])
                assert np.array_equal(solve(pair['input']), pair['output'])
                shapes.add(pair['input'].shape)
                stamp_shapes.add(info['payload'].shape)
                counts.add(len(info['groups']))
                colors.add(tuple(sorted(info['master']['guides'].values())))
                master_corners.add(_corner(info['master']['guides']))
                # Normalize motif colors but retain zero and geometry.
                mapping = {0: 0}
                motifs.add((info['payload'].shape, tuple(mapping.setdefault(int(v), len(mapping)) for v in info['payload'].flat)))
                blank_edges += any(np.all(line == 0) for line in
                                   (info['payload'][0], info['payload'][-1],
                                    info['payload'][:, 0], info['payload'][:, -1]))
                edge_contact += bool(np.any(pair['output'][0]) or np.any(pair['output'][-1])
                                     or np.any(pair['output'][:, 0]) or np.any(pair['output'][:, -1]))
                for g in info['groups']:
                    matrices.add(_name(g['matrix']))
                    receiver_corners.add(_corner(g['frame']['guides']))
                    a, b = g['marker'][0]
                    r0, c0, r1, c1 = g['frame']['box']
                    if g['kind'] == 'P':
                        point_count += 1
                        quadrants.add((int(np.sign(a-r0)), int(np.sign(b-c0))))
                    else:
                        line_sides.add((g['kind'], int(np.sign((a-r0) if g['kind'] == 'H' else (b-c0)))))
                        mr0, mc0, mr1, mc1 = _bbox(g['marker'])
                        overhangs += (mc0 > c0 or mc1 < c1) if g['kind'] == 'H' else (mr0 > r0 or mr1 < r1)
                    # Cross-check coordinate solver against generator-side array
                    # operations without relying on expected test labels.
                    source_matrix = _frame_matrix(info['master']['guides'], g['frame']['guides'])
                    source_stamp = _array_transform(info['payload'], _name(source_matrix))
                    expected = _array_transform(source_stamp, {'H': 'ud', 'V': 'lr', 'P': 'half'}[g['kind']])
                    assert np.array_equal(expected, g['payload'])
    assert len(shapes) >= 40 and len(stamp_shapes) >= 10
    assert counts == {1, 2, 3} and matrices == set(D4)
    assert len(colors) >= 15 and len(motifs) >= 80
    assert master_corners == receiver_corners == set(range(4))
    assert quadrants == {(-1, -1), (-1, 1), (1, -1), (1, 1)}
    assert line_sides == {('H', -1), ('H', 1), ('V', -1), ('V', 1)}
    assert point_count >= 60 and overhangs >= 30 and blank_edges >= 100 and edge_contact >= 10
    # Hand-built rectangular fixtures are not sampled by the generator. The
    # second exchanges axes, and both include a whole blank master edge.
    stamp = np.array([[0, 1, 1, 0], [1, 4, 0, 1], [0, 0, 0, 0]], dtype=np.int8)
    for exchange_axes in (False, True):
        inp = np.zeros((18, 22), dtype=np.int8)
        inp[1:4, 1:5] = stamp
        inp[0, 1:5] = 2
        inp[1:4, 0] = 3
        if exchange_axes:
            inp[5, 12:15] = 3
            inp[6:10, 15] = 2
            inp[10, 12:14] = 5
            expected = inp.copy()
            expected[11:15, 12:15] = np.rot90(stamp.T, 2)
            expected[11:15, 15] = 2
            expected[15, 12:15] = 3
        else:
            inp[5, 12:16] = 2
            inp[6:9, 11] = 3
            inp[9, 12:14] = 5
            expected = inp.copy()
            expected[10:13, 12:16] = np.flipud(stamp)
            expected[13, 12:16] = 2
            expected[10:13, 11] = 3
        assert np.array_equal(solve(inp), expected)
        assert np.array_equal(solve(inp.tolist()), expected)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(4242+17*nt+ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    original = puzzles[0]
    for mutation in ('cell', 'crop', 'guide', 'omit_A', 'omit_B', 'translation'):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in original[s]] for s in original}
        pair = bad['train'][0]
        info = _decode(pair['input'])
        if mutation == 'cell':
            p = next(iter(info['groups'][0]['destination']['bars']))
            pair['output'][p] = (int(pair['output'][p])+1) % 10
        elif mutation == 'crop':
            pair['output'] = pair['output'][1:]
        elif mutation == 'guide':
            p = next(iter(info['groups'][0]['frame']['bars']))
            pair['input'][p] = 0
        else:
            pair['output'] = _prediction(info, mutation)
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Accepted mutation ' + mutation)
    return {'seeds': 42, 'pairs': 210, 'hand_built_rectangles': 2, 'canvas_shapes': len(shapes),
            'master_shapes': len(stamp_shapes), 'motifs': len(motifs),
            'guide_palettes': len(colors), 'orientations': sorted(matrices),
            'short_axis_overhangs': int(overhangs), 'blank_edge_stamps': int(blank_edges),
            'edge_contact_pairs': int(edge_contact)}


if __name__ == '__main__':
    print(self_test())
