"""Shared boundary-port evidence, with a different symmetry echo at each frame.

All solving state is reconstructed from the supplied grid. Border colors identify
correspondence; the color and geometry inside each frame identify its marker.
Only Python's standard library and NumPy are used.
"""
from __future__ import annotations

import numpy as np


HOST_SEEDS = (0, 262178156, 43747376, 46656102, 976726101, 896982360,
              885731912, 103981353, 523368230, 875814604, 130899602,
              563390667, 826470434, 367037194, 797568160, 236854735,
              708057830, 227818498, 410832175, 406601926, 741223658,
              739377697, 500138369, 246572800, 978696777, 521108071,
              746402477, 318047563, 317487639, 250154899, 890877323,
              893498671, 296833541, 430950312, 803738038, 791126324,
              175362426, 226792523, 819664781, 627336889, 129115048,
              937432241)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             'Grid must be two-dimensional with dimensions 1..30')
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             'Grid must contain integer colors 0..9')
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _orient(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, pending = [], [start]
        while pending:
            r, c = pending.pop()
            cells.append((r, c))
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    pending.append(q)
        result.append(cells)
    return result


def _border_mask(h, w):
    mask = np.zeros((h, w), dtype=bool)
    mask[[0, -1], :] = True
    mask[:, [0, -1]] = True
    return mask


def _ray(emitter, side, slot, shape):
    """Transform a long-edge launch port and its outward unit vector."""
    labels = emitter['labels']
    length = max(labels.shape)
    row, inner = (0, 1) if side == 0 else (4, 3)
    a = np.argwhere(labels == row*length + slot)[0]
    b = np.argwhere(labels == inner*length + slot)[0]
    dr, dc = map(int, a-b)
    r, c = emitter['r']+int(a[0])+dr, emitter['c']+int(a[1])+dc
    path = []
    while 0 <= r < shape[0] and 0 <= c < shape[1]:
        path.append((r, c))
        r, c = r+dr, c+dc
    return tuple(path)


def _echo(port, kind, pivot):
    side, slot = port
    return ((1-side if kind in ('H', 'P') else side),
            (2*pivot-slot if kind in ('V', 'P') else slot))


def _closed_code(code, kind, pivot, length):
    result = dict(code)
    for port, color in code.items():
        partner = _echo(port, kind, pivot)
        _require(0 <= partner[1] < length, 'Symmetric port leaves the frame')
        _require(partner not in result or result[partner] == color,
                 'Symmetry assigns conflicting colors to one port')
        result[partner] = color
    return result


def _order_witness(code, kind, pivot):
    """Does the colored echo differ from translation to its destination interval?"""
    echoed = {_echo(p, kind, pivot): c for p, c in code.items()}
    ds = min(p[0] for p in echoed)-min(p[0] for p in code)
    dj = min(p[1] for p in echoed)-min(p[1] for p in code)
    translated = {(s+ds, j+dj): c for (s, j), c in code.items()}
    return echoed != translated


def _draw(grid, emitters, codes, background, strict=True):
    result = grid.copy()
    boxes = set().union(*(e['box'] for e in emitters))
    painted, ray_lengths = {}, []
    for emitter, code in zip(emitters, codes):
        for port, color in code.items():
            path = _ray(emitter, *port, grid.shape)
            _require(len(path) >= 2, 'Every emitted ray needs at least two cells')
            ray_lengths.append(len(path))
            for p in path:
                _require(p not in boxes, 'Ray crosses a frame')
                _require(not strict or p not in painted, 'Rays intersect')
                _require(int(grid[p]) in (background, color), 'Ray overwrites input')
                painted[p] = color
                result[p] = color
    return result, ray_lengths


def _read(input_grid):
    grid = _grid(input_grid)
    values, counts = np.unique(grid, return_counts=True)
    _require(sum(counts == counts.max()) == 1, 'Background is ambiguous')
    background = int(values[counts.argmax()])
    emitters = []
    # A frame is identified by its complete non-background perimeter. Detached
    # interior markers and boundary clue clusters need not belong to that component.
    for component in _components(grid != background):
        rs, cs = zip(*component)
        r, c, h, w = min(rs), min(cs), max(rs)-min(rs)+1, max(cs)-min(cs)+1
        if min(h, w) != 5 or not 6 <= max(h, w) <= 11:
            continue
        a = grid[r:r+h, c:c+w]
        mask = _border_mask(h, w)
        if not np.all(a[mask] != background):
            continue
        emitters.append({'r': r, 'c': c, 'arr': a.copy(),
                         'box': {(rr, cc) for rr in range(r, r+h)
                                 for cc in range(c, c+w)}})
    _require(2 <= len(emitters) <= 4, 'Expected two to four barcode frames')
    boxes = set()
    for e in emitters:
        _require(not boxes.intersection(e['box']), 'Overlapping frames')
        boxes.update(e['box'])
    first = emitters[0]['arr']
    if first.shape[0] != 5:
        first = np.rot90(first)
    length = first.shape[1]
    template = first.copy()
    template[1:-1, 1:-1] = -1
    forms = [_orient(template, t) for t in range(8)]
    _require(len({_key(a) for a in forms}) == 8, 'Border does not identify orientation')
    structural = set(map(int, template[template >= 0]))
    _require(2 <= len(structural) <= 3, 'Border needs two or three structural colors')
    marker_colors, transforms = set(), []
    for e in emitters:
        border = e['arr'].copy()
        border[1:-1, 1:-1] = -1
        matches = [t for t, a in enumerate(forms) if np.array_equal(a, border)]
        _require(len(matches) == 1, 'Frame does not uniquely match the common border')
        t = matches[0]
        transforms.append(t)
        e['labels'] = _orient(np.arange(5*length).reshape(5, length), t)
        interior = []
        for rr in range(1, e['arr'].shape[0]-1):
            for cc in range(1, e['arr'].shape[1]-1):
                color = int(e['arr'][rr, cc])
                if color != background:
                    label = int(e['labels'][rr, cc])
                    interior.append((label//length, label % length, color))
        _require(interior and len({p[2] for p in interior}) == 1,
                 'Each interior contains one monochrome marker')
        marker = interior[0][2]
        _require(marker not in structural and marker != background,
                 'Marker must differ from border and background')
        marker_colors.add(marker)
        coords = sorted((r, c) for r, c, _ in interior)
        rows, cols = {r for r, _ in coords}, {c for _, c in coords}
        if len(coords) == 1:
            kind, pivot = 'P', coords[0][1]
            _require(rows == {2}, 'Point center must lie on the middle row')
        elif rows == {2}:
            kind, pivot = 'H', 0
            _require(sorted(cols) == list(range(min(cols), max(cols)+1)),
                     'Parallel marker is not contiguous')
        elif len(cols) == 1:
            kind, pivot = 'V', coords[0][1]
            _require(2 in rows and sorted(rows) == list(range(min(rows), max(rows)+1)),
                     'Transverse marker must be contiguous and reach the middle row')
        else:
            raise ValueError('Marker must be a line or a point')
        _require(kind == 'H' or 1 <= pivot <= length-2, 'Invalid pivot')
        e.update(kind=kind, pivot=pivot, marker=marker, transform=t)
    _require(len(marker_colors) == 1, 'All markers must share a color')
    marker = marker_colors.pop()
    clues = [(int(r), int(c), int(grid[r, c])) for r, c in np.argwhere(grid != background)
             if (int(r), int(c)) not in boxes]
    _require(clues, 'There must be boundary evidence')
    corridors = {}
    for index, e in enumerate(emitters):
        for side in (0, 1):
            for slot in range(length):
                path = _ray(e, side, slot, grid.shape)
                if len(path) >= 2 and not boxes.intersection(path):
                    if all(int(grid[p]) == background for p in path[:-1]):
                        corridors.setdefault(path[-1], []).append((index, (side, slot)))
    code, allocations = {}, [dict() for _ in emitters]
    for r, c, color in clues:
        _require(r in (0, grid.shape[0]-1) or c in (0, grid.shape[1]-1),
                 'Exterior cells must be boundary clues')
        _require(color not in structural | {background, marker}, 'Clue role conflict')
        matches = corridors.get((r, c), [])
        _require(len(matches) == 1, 'Boundary clue has ambiguous ownership')
        index, port = matches[0]
        _require(port not in code or code[port] == color, 'Inconsistent shared observations')
        code[port] = color
        allocations[index][port] = color
    _require(1 <= len(code) <= 4, 'Expected one to four observed ports')
    codes = [_closed_code(code, e['kind'], e['pivot'], length) for e in emitters]
    output, ray_lengths = _draw(grid, emitters, codes, background)
    _require(np.array_equal(grid[grid != background], output[grid != background]),
             'Input foreground is not preserved')
    _require(np.any(output != grid), 'Identity scene')
    no_b, _ = _draw(grid, emitters, [code]*len(emitters), background)
    no_a, _ = _draw(grid, emitters,
                    [_closed_code(local, e['kind'], e['pivot'], length)
                     for local, e in zip(allocations, emitters)], background)
    info = {'background': background, 'marker': marker, 'length': length,
            'structural': tuple(sorted(structural)), 'code': code,
            'emitters': emitters, 'allocations': allocations, 'clues': clues,
            'codes': codes, 'ray_lengths': ray_lengths,
            'pooled': not any(len(a) == len(code) for a in allocations),
            'unhinted': any(not a for a in allocations),
            'a_essential': not np.array_equal(no_a, output),
            'b_essential': not np.array_equal(no_b, output),
            'b_essential_kinds': tuple(sorted({e['kind'] for e, full in zip(emitters, codes)
                                              if full != code})),
            'order_witness_kinds': tuple(sorted({e['kind'] for e in emitters
                                               if _order_witness(code, e['kind'], e['pivot'])})),
            'kinds': tuple(sorted({e['kind'] for e in emitters})),
            'transforms': tuple(transforms)}
    return output, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the exact completion using only this input's pixels."""
    return _read(input_grid)[0]


def _make_border(rng, length, colors):
    mask = _border_mask(5, length)
    for _ in range(40):
        a = np.full((5, length), -1, dtype=np.int8)
        a[mask] = rng.choice(colors, int(mask.sum()))
        style = int(rng.integers(3))
        if style == 1:
            a[0] = colors[0]
            a[4] = colors[1]
            a[4, :int(rng.integers(1, length-1))] = colors[-1]
            a[1:4, 0] = colors[1]
            a[1:4, -1] = colors[0]
        elif style == 2:
            a[0] = colors[0]
            a[4] = colors[1]
            a[0, -2:] = colors[1]
            a[1:4, 0] = colors[0]
            a[1:4, -1] = colors[-1]
        for color in colors:
            if color not in a:
                a[0, int(rng.integers(length))] = color
        if len({_key(_orient(a, t)) for t in range(8)}) == 8:
            return a
    # An asymmetric cap guarantees eight forms even if all random trials fail.
    a = np.full((5, length), -1, dtype=np.int8)
    a[mask] = colors[0]
    a[4] = colors[1]
    a[0, -1] = colors[1]
    if len(colors) == 3:
        a[2, -1] = colors[2]
    return a


def _pivot_choices(code, kind, length):
    choices = []
    for pivot in range(1, length-1):
        try:
            full = _closed_code(code, kind, pivot, length)
        except ValueError:
            continue
        # Rich witnesses must make the missing partner genuinely missing.
        if len(full) > len(code):
            choices.append(pivot)
    return choices


def _prototype(border, kind, pivot, transform, marker, background, rng):
    a = border.copy()
    a[a < 0] = background
    length = a.shape[1]
    if kind == 'P':
        a[2, pivot] = marker
    elif kind == 'V':
        start = int(rng.integers(1, 3))
        stop = 4 if start == 2 else int(rng.integers(3, 5))
        a[start:stop, pivot] = marker
    else:
        count = int(rng.integers(2, length-1))
        start = int(rng.integers(1, length-count))
        a[2, start:start+count] = marker
    return {'arr': _orient(a, transform),
            'labels': _orient(np.arange(a.size).reshape(a.shape), transform),
            'kind': kind, 'pivot': pivot, 'transform': transform}


def _place(rng, prototypes, codes, shape):
    """Free placement reserves full boxes and the actual final ray corridors."""
    for _ in range(5):
        placed, occupied, paths = [], set(), set()
        for proto, code in zip(prototypes, codes):
            h, w = proto['arr'].shape
            if h > shape[0] or w > shape[1]:
                break
            options = [(r, c) for r in range(shape[0]-h+1)
                       for c in range(shape[1]-w+1)]
            rng.shuffle(options)
            for r, c in options[:200]:
                e = dict(proto, r=r, c=c)
                box = {(rr, cc) for rr in range(r, r+h) for cc in range(c, c+w)}
                if box.intersection(occupied | paths):
                    continue
                if any((rr+dr, cc+dc) in occupied for rr, cc in box
                       for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))):
                    continue
                rays = [_ray(e, *p, shape) for p in code]
                if any(len(path) < 2 for path in rays):
                    continue
                cells = set(q for path in rays for q in path)
                if cells.intersection(occupied | paths):
                    continue
                e['box'] = box
                placed.append(e)
                occupied.update(box)
                paths.update(cells)
                break
            else:
                break
        if len(placed) == len(prototypes):
            return placed
    return None


def _choose_clues(rng, emitters, code, shape, allocation_mode):
    boxes = set().union(*(e['box'] for e in emitters))
    endpoints = {}
    length = max(emitters[0]['arr'].shape)
    for index, e in enumerate(emitters):
        for side in (0, 1):
            for slot in range(length):
                path = _ray(e, side, slot, shape)
                if len(path) >= 2 and not boxes.intersection(path):
                    endpoints.setdefault(path[-1], []).append((index, (side, slot)))
    options = {}
    for port in code:
        options[port] = []
        for index, e in enumerate(emitters):
            p = _ray(e, *port, shape)[-1]
            if endpoints.get(p) == [(index, port)]:
                if not any((p[0]+dr, p[1]+dc) in boxes
                           for dr in (-1, 0, 1) for dc in (-1, 0, 1)):
                    options[port].append((index, p))
        if not options[port]:
            return None
    complete = [i for i in range(len(emitters))
                if all(any(index == i for index, _ in opts) for opts in options.values())]
    if allocation_mode == 'full':
        if not complete:
            return None
        owner = int(rng.choice(complete))
        chosen = [(port, owner, next(p for i, p in options[port] if i == owner))
                  for port in code]
    else:
        chosen = []
        ports = list(code)
        rng.shuffle(ports)
        for port in ports:
            opts = options[port]
            if len(chosen) == len(ports)-1 and allocation_mode == 'split':
                used = {i for _, i, _ in chosen}
                opts = [(i, p) for i, p in opts if i not in used]
            if not opts:
                return None
            i, p = opts[int(rng.integers(len(opts)))]
            chosen.append((port, i, p))
    if allocation_mode == 'spread' and rng.random() < .35:
        extras = [(port, i, p) for port, opts in options.items() for i, p in opts
                  if (port, i, p) not in chosen]
        rng.shuffle(extras)
        chosen.extend(extras[:int(rng.integers(1, min(3, len(extras))+1))] if extras else [])
    return chosen


def _make_pair(rng, marker, mode, previous_background=None):
    # Parameter rejection is geometric: long frames and many two-sided rays
    # are naturally incompatible with some small canvases. No layout bank is used.
    for attempt in range(180):
        length = int(rng.integers(6, 12))
        palette = [int(c) for c in rng.permutation([c for c in range(10) if c != marker])]
        if palette[0] == previous_background:
            palette[0], palette[-1] = palette[-1], palette[0]
        background = palette[0]
        nstruct = int(rng.integers(2, 4))
        border = _make_border(rng, length, palette[1:1+nstruct])
        if mode in ('parallel_witness', 'transverse_witness'):
            count = int(rng.choice([2, 3, 4], p=[.55, .35, .10]))
            primary = 'H' if mode == 'parallel_witness' else 'V'
            kinds = [primary, 'P'] + [str(rng.choice([primary, 'P'])) for _ in range(count-2)]
            nports = 2
            slots = sorted(map(int, rng.choice(length, nports, replace=False)))
            side = int(rng.integers(2))
            ports = [(side, j) for j in slots]
            colors = palette[1+nstruct:3+nstruct]
        else:
            count = int(rng.choice([2, 3, 4], p=[.25, .55, .20]))
            if mode == 'mixed_test':
                count = max(3, count)
                kinds = ['H', 'V', 'P'] + [str(rng.choice(['H', 'V', 'P'])) for _ in range(count-3)]
            else:
                pool = ['H', 'P'] if rng.integers(2) else ['V', 'P']
                kinds = [str(rng.choice(pool)) for _ in range(count)]
            nports = int(rng.integers(1, 5))
            slots = list(map(int, rng.choice(length, nports, replace=False)))
            same_side = int(rng.integers(2))
            ports = [(int(rng.integers(2)) if rng.random() < .3 else same_side, j) for j in slots]
            color_pool = palette[1+nstruct:1+nstruct+int(rng.integers(1, min(3, nports)+1))]
            colors = [int(rng.choice(color_pool)) for _ in ports]
        code = dict(zip(ports, colors))
        valid_pivots = {kind: _pivot_choices(code, kind, length) for kind in ('V', 'P')}
        if any(not valid_pivots[kind] for kind in kinds if kind != 'H'):
            continue
        pivots = [0 if kind == 'H' else int(rng.choice(valid_pivots[kind])) for kind in kinds]
        codes = [_closed_code(code, kind, pivot, length) for kind, pivot in zip(kinds, pivots)]
        # Parallel, opposite and longitudinally reversed copies are common;
        # perpendicular mixtures are attempted when their actual corridors fit.
        base = int(rng.integers(8))
        transforms = [base ^ int(rng.choice([0, 4])) for _ in kinds]
        if mode == 'parallel_witness':
            transforms[:2] = [base, base ^ 4]
        elif rng.random() < .35 and attempt < 100:
            transforms = list(map(int, rng.integers(8, size=count)))
        else:
            transforms = [((base//4)*4 + (base+int(rng.choice([0, 2]))) % 4)
                          ^ int(rng.choice([0, 4])) for _ in kinds]
        prototypes = [_prototype(border, k, p, t, marker, background, rng)
                      for k, p, t in zip(kinds, pivots, transforms)]
        shape = tuple(map(int, rng.integers(13, 31, size=2)))
        if rng.random() < .2:
            shape = (shape[0], shape[0])
        placed = _place(rng, prototypes, codes, shape)
        if placed is None:
            continue
        allocation = ('split' if mode == 'parallel_witness' else
                      'full' if mode == 'transverse_witness' else
                      str(rng.choice(['full', 'split', 'spread'])))
        if len(code) == 1 and allocation == 'split':
            allocation = 'full'
        clues = _choose_clues(rng, placed, code, shape, allocation)
        if clues is None:
            continue
        grid = np.full(shape, background, dtype=np.int8)
        for e in placed:
            h, w = e['arr'].shape
            grid[e['r']:e['r']+h, e['c']:e['c']+w] = e['arr']
        for port, _, pos in clues:
            grid[pos] = code[port]
        try:
            # Construction uses its own coordinates for the target. The decoder
            # independently reconstructs frames, markers, clues and correspondence.
            output, _ = _draw(grid, placed, codes, background)
            recovered, info = _read(grid)
            if not np.array_equal(output, recovered):
                continue
            if mode == 'parallel_witness' and not (info['pooled'] and info['a_essential'] and info['b_essential']):
                continue
            if mode == 'transverse_witness' and not (info['unhinted'] and info['a_essential'] and info['b_essential']):
                continue
        except ValueError:
            continue
        return {'input': grid, 'output': output}
    return _fallback_pair(rng, marker, mode, previous_background)


def _fallback_pair(rng, marker, mode, previous_background=None):
    """A parametric spacious construction guarantees the required witnesses.

    Disjoint longitudinal spans make both-sided corridors safe. This is used
    only after free placement exhausts its bound; it is not a stored example.
    """
    count = int(rng.integers(3, 5) if mode == 'mixed_test' else rng.integers(2, 5))
    length = int(rng.integers(6, min(11, (31-count)//count)+1))
    palette = list(map(int, rng.permutation([c for c in range(10) if c != marker])))
    if palette[0] == previous_background:
        palette[0], palette[-1] = palette[-1], palette[0]
    background = palette[0]
    border = _make_border(rng, length, palette[1:4])
    code = {(0, 1): palette[4], (0, 2): palette[5]}
    if mode == 'parallel_witness':
        kinds = ['H', 'P'] + [str(rng.choice(['H', 'P'])) for _ in range(count-2)]
    elif mode == 'transverse_witness':
        kinds = ['V', 'P'] + [str(rng.choice(['V', 'P'])) for _ in range(count-2)]
    elif mode == 'mixed_test':
        kinds = ['H', 'V', 'P'] + [str(rng.choice(['H', 'V', 'P'])) for _ in range(count-3)]
    else:
        pool = ['H', 'P'] if rng.integers(2) else ['V', 'P']
        kinds = [str(rng.choice(pool)) for _ in range(count)]
    height = int(rng.integers(13, 31))
    minimum_width = count*length + count-1
    width = int(rng.integers(minimum_width, 31))
    # Distribute all extra horizontal room over exterior and interior gaps.
    extra = rng.multinomial(width-minimum_width, np.full(count+1, 1/(count+1)))
    col, emitters, codes = int(extra[0]), [], []
    grid = np.full((height, width), background, dtype=np.int8)
    for i, kind in enumerate(kinds):
        pivot = length//2 if kind != 'H' else 0
        transform = 0 if i == 0 else 4 if i == 1 else int(rng.choice([0, 4]))
        proto = _prototype(border, kind, pivot, transform, marker, background, rng)
        row = int(rng.integers(2, height-6))
        e = dict(proto, r=row, c=col,
                 box={(r, c) for r in range(row, row+5) for c in range(col, col+length)})
        emitters.append(e)
        codes.append(_closed_code(code, kind, pivot, length))
        grid[row:row+5, col:col+length] = proto['arr']
        col += length+1+int(extra[i+1])
    allocation = 'split' if mode == 'parallel_witness' else 'full'
    clues = _choose_clues(rng, emitters, code, grid.shape, allocation)
    assert clues is not None
    for port, _, p in clues:
        grid[p] = code[port]
    output, _ = _draw(grid, emitters, codes, background)
    turn = int(rng.integers(8))
    pair = {'input': _orient(grid, turn), 'output': _orient(output, turn)}
    assert np.array_equal(solve(pair['input']), pair['output'])
    return pair


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    _require(num_train > num_test, 'num_train must exceed num_test')
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_))
             and seed >= 0, 'Seed must be a non-negative integer')
    rng = np.random.default_rng(seed)
    marker = int(rng.integers(10))
    puzzle, seen, previous_background = {'train': [], 'test': []}, set(), None
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            mode = ('parallel_witness' if i == 0 else 'transverse_witness' if i == 1 else 'extra') if split == 'train' else ('mixed_test' if i == 0 else 'extra')
            for _ in range(8):
                pair = _make_pair(rng, marker, mode, previous_background)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('Could not construct distinct inputs')
            seen.add(_key(pair['input']))
            previous_background = _read(pair['input'])[1]['background']
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Schema, reconstruction, preservation, evidence and compositional shift."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'Splits must be lists')
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, 'Invalid example counts')
    seen, markers, train_info, test_info = set(), set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Invalid pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'Output dimensions changed')
            _require(_key(inp) not in seen, 'Duplicate input')
            seen.add(_key(inp))
            expected, info = _read(inp)
            _require(np.array_equal(expected, out), 'Incorrect port echo completion')
            markers.add(info['marker'])
            (train_info if split == 'train' else test_info).append(info)
    _require(len(markers) == 1, 'Generated puzzle changes marker color')
    _require(any(f['pooled'] and f['a_essential'] and f['b_essential'] for f in train_info),
             'Training lacks distributed pooling and symmetry evidence')
    _require(any(f['unhinted'] and f['a_essential'] for f in train_info),
             'Training lacks transfer to an unhinted frame')
    for kind in ('H', 'V', 'P'):
        _require(any(kind in f['b_essential_kinds'] for f in train_info),
                 'Training lacks a required marker operation')
    for kind in ('V', 'P'):
        _require(any(kind in f['order_witness_kinds'] for f in train_info),
                 'Training conceals reflection or half-turn behind translation')
    _require(any(len({t >= 4 for t in f['transforms']}) == 2
                 and f['code'] != {(s, f['length']-1-j): c for (s, j), c in f['code'].items()}
                 for f in train_info), 'Training lacks observable reversed correspondence')
    _require(any(f['kinds'] == ('H', 'P', 'V') for f in test_info),
             'Missing mixed-operation test recombination')


def self_test():
    """Bounded 42-seed suite, evidence ablations, equivariance and corruption."""
    puzzles, features = [], []
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        puzzles.append(p)
        for split in ('train', 'test'):
            for pair in p[split]:
                expected, f = _read(pair['input'])
                assert np.array_equal(expected, pair['output'])
                assert f['b_essential'] or len(f['code']) > 1
                features.append((pair, f))
    assert len({tuple(_key(x['input']) for x in p['train']+p['test']) for p in puzzles}) == 42
    assert len({x['input'].shape for x, _ in features}) >= 35
    assert len({f['length'] for _, f in features}) >= 5
    assert {len(f['emitters']) for _, f in features} == {2, 3, 4}
    assert {len(f['code']) for _, f in features} == {1, 2, 3, 4}
    assert len({f['marker'] for _, f in features}) >= 8
    assert len({f['background'] for _, f in features}) >= 8
    assert any(len({t % 2 for t in f['transforms']}) == 2 for _, f in features)
    assert any(len(f['clues']) > len(f['code']) for _, f in features)
    assert any(len(set(f['code'].values())) < len(f['code']) for _, f in features)
    assert any(len({p[0] for p in f['code']}) == 2 for _, f in features)
    assert any(any(abs(a[1]-b[1]) == 1 for a in f['code'] for b in f['code'] if a != b)
               for _, f in features)
    assert any(len({e['pivot'] for e in f['emitters'] if e['kind'] != 'H'}) > 1
               for _, f in features)
    # All dihedral motions and role permutations must commute with solving.
    pair = puzzles[0]['train'][0]
    recolor = np.array([7, 3, 9, 1, 0, 8, 2, 6, 5, 4], dtype=np.int8)
    for t in range(8):
        inp, out = (_orient(recolor[pair[k]], t) for k in ('input', 'output'))
        assert np.array_equal(solve(inp), out)
    # A small procedural fixture isolates marker geometry while keeping border,
    # evidence, layout and colors identical. No parent pair is used as a fixture.
    fixture_rng = np.random.default_rng(195)
    border = _make_border(fixture_rng, 9, [6, 7])
    fixture = np.zeros((26, 28), dtype=np.int8)
    for kind, r, c in [('H', 4, 2), ('P', 14, 16)]:
        proto = _prototype(border, kind, 4, 0, 5, 0, fixture_rng)
        fixture[r:r+5, c:c+9] = proto['arr']
    fixture[0, 3], fixture[0, 19] = 2, 3
    reference, f = _read(fixture)
    assert f['a_essential'] and f['b_essential'] and f['pooled']
    point_version = fixture.copy()
    point_version[5:8, 3:10] = 0
    point_version[6, 6] = 5
    point_output = solve(point_version)
    assert not np.array_equal(reference, point_output)
    vertical_version = point_version.copy()
    vertical_version[5:8, 6] = 5
    assert not np.array_equal(solve(vertical_version), point_output)
    # Permuting/reallocating clue observations need not alter the pooled answer;
    # deleting one nonredundant observed color must alter the answer.
    for pair, f in features:
        if f['pooled'] and len(f['code']) == 2 and len(f['clues']) == 2:
            changed = pair['input'].copy()
            r, c, _ = f['clues'][0]
            changed[r, c] = f['background']
            assert not np.array_equal(solve(changed), pair['output'])
            break
    else:
        raise AssertionError('Missing nonredundant pooling mutation witness')
    bad = {s: [{k: v.copy() for k, v in x.items()} for x in puzzles[0][s]] for s in ('train', 'test')}
    changed = np.argwhere(bad['test'][0]['input'] != bad['test'][0]['output'])[0]
    bad['test'][0]['output'][tuple(changed)] = bad['test'][0]['input'][tuple(changed)]
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('Accepted corrupted output')
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(7001, a, b)
            assert (len(p['train']), len(p['test'])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('Accepted illegal counts')
    # Exercise the guaranteed route directly instead of waiting for rare retries.
    fallback_rng = np.random.default_rng(20261007)
    for mode in ('parallel_witness', 'transverse_witness', 'mixed_test', 'extra'):
        for _ in range(4):
            pair = _fallback_pair(fallback_rng, 5, mode)
            out, f = _read(pair['input'])
            assert np.array_equal(out, pair['output']) and f['a_essential'] and f['b_essential']
            if mode == 'parallel_witness':
                assert f['pooled']
            if mode == 'transverse_witness':
                assert f['unhinted'] and {'V', 'P'} <= set(f['order_witness_kinds'])
    q = generate_puzzle(0)
    assert all(np.array_equal(x[k], y[k]) for s in q for x, y in zip(q[s], puzzles[0][s])
               for k in ('input', 'output'))
