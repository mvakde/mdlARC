"""Circuit-routed unary addition, located by a miniature color-sequence clue.

Only the input is used by solve. Generation draws labels from logical circuits;
validation recovers circuits, geometry, counts and routing from returned pixels.
"""
from collections import defaultdict
import numpy as np

SOURCE, DESTINATION, PROCESSED = 2, 3, 5
FREE_COLORS = (0, 1, 4, 6, 7, 8, 9)
AUDIT_SEEDS = (0, 851587744, 661063565, 730062837, 225352489, 476484748,
    442050020, 272506954, 40052476, 352128144, 477276083, 387879168,
    562333646, 184571210, 320559018, 892698713, 820293441, 863513833,
    658883958, 346041095, 688787285, 115945795, 784476863, 762635138,
    93855471, 80138327, 276116868, 907772072, 703634156, 625034882,
    568747814, 714897794, 260871733, 174993333, 229294270, 202996483,
    713852937, 138085855, 140797204, 283083824, 708205759, 995641230)


def _labels(mask):
    """Four-neighbor logical connectivity; colors never affect connectivity."""
    label = np.full(mask.shape, -1, dtype=int)
    number = 0
    for y, x in np.argwhere(mask):
        y, x = int(y), int(x)
        if label[y, x] >= 0:
            continue
        label[y, x] = number
        todo = [(y, x)]
        while todo:
            r, c = todo.pop()
            for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if (0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1]
                        and mask[rr, cc] and label[rr, cc] < 0):
                    label[rr, cc] = number
                    todo.append((rr, cc))
        number += 1
    return label


def _pixel_components(mask):
    labels = _labels(mask)
    return [np.argwhere(labels == i) for i in range(int(labels.max()) + 1)]


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError('expected an integer ARC grid with sides 1..30 and colors 0..9')
    return a.astype(np.int8, copy=True)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _matches(plan, sequence, background, occupancy_only=False):
    length = len(sequence)
    target = np.asarray(sequence)
    if occupancy_only:
        target = target != background
    result = []
    for r in range(plan.shape[0]):
        for c in range(plan.shape[1] - length + 1):
            actual = plan[r, c:c+length]
            if occupancy_only:
                actual = actual != background
            if np.array_equal(actual, target):
                result.append((r, c))
    return result


def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _routes(parsed, match=None):
    """Return destination x -> contributing source stacks, at a proposed match."""
    r, c = parsed['match'] if match is None else match
    by_component = defaultdict(list)
    for stack in parsed['stacks']:
        k = (stack['x'] - parsed['left']) // parsed['pitch']
        component = int(parsed['labels'][r, c+k])
        _require(component >= 0, 'socket must be occupied')
        if stack['role']:
            by_component[component].append(stack)
    routes = {}
    for members in by_component.values():
        dst = [s for s in members if s['role'] == DESTINATION]
        src = [s for s in members if s['role'] == SOURCE]
        _require(len(dst) == 1 and bool(src),
                 'each controlled circuit needs exactly one destination and at least one source')
        routes[dst[0]['x']] = src
    _require(bool(routes), 'no controlled circuits')
    return routes


def _parse_input(value):
    a = _grid(value)
    histogram = np.bincount(a.ravel(), minlength=10)
    bg = int(histogram.argmax())
    _require(histogram[bg] > np.partition(histogram, -2)[-2], 'background must be unique')
    _require(bg in FREE_COLORS and PROCESSED not in a, 'reserved role colors in payload/background')
    swatches, tokens, bars = [], [], []
    for cells in _pixel_components(a != bg):
        y, x = map(int, cells.min(axis=0))
        end_y, end_x = map(int, cells.max(axis=0) + 1)
        h, w = end_y-y, end_x-x
        color = int(a[y, x])
        _require(np.all(a[y:end_y, x:end_x] == color), 'objects must be solid')
        if len(cells) == 1:
            _require(color in FREE_COLORS, 'miniature uses only payload colors')
            swatches.append((y, x))
        elif h == 1 and w >= 2:
            _require(color in (SOURCE, DESTINATION), 'unknown control bar')
            bars.append((y, x, w, color))
        else:
            _require(h == w and 2 <= h <= 4 and len(cells) == h*w,
                     'tokens must be separated solid squares, side 2..4')
            _require(color in FREE_COLORS, 'token uses a reserved control color')
            tokens.append((y, x, h, color))
    _require(len(swatches) >= 3 and tokens and bars, 'missing diagram, bank or controls')
    points = np.asarray(swatches)
    my, mx = map(int, points.min(axis=0))
    ey, ex = map(int, points.max(axis=0) + 1)
    _require(np.all((points - (my, mx)) % 2 == 0), 'miniature swatch pitch must be two')
    _require(any(np.diff(np.unique(points[:, 0])) == 2)
             and any(np.diff(np.unique(points[:, 1])) == 2), 'miniature needs visible pitch evidence')
    plan = a[my:ey:2, mx:ex:2].copy()
    _require(2 <= plan.shape[0] <= 10 and 2 <= plan.shape[1] <= 10,
             'miniature dimensions outside family')
    sampled = np.zeros((ey-my, ex-mx), bool)
    sampled[::2, ::2] = True
    _require(np.all(a[my:ey, mx:ex][~sampled] == bg)
             and np.count_nonzero(plan != bg) == len(swatches), 'bad miniature separators')
    side = tokens[0][2]
    _require(all(t[2] == side for t in tokens), 'one token size per scene')
    groups = defaultdict(list)
    for y, x, _, color in tokens:
        groups[x].append((y, color))
    pitches, bottom_rows, stacks = set(), set(), []
    for x, group in sorted(groups.items()):
        group.sort()
        ys = [g[0] for g in group]
        colors = {g[1] for g in group}
        _require(len(colors) == 1, 'each stack is monochrome')
        pitches.update(b-a for a, b in zip(ys, ys[1:]))
        bottom_rows.add(ys[-1])
        stacks.append(dict(x=x, ys=ys, height=len(ys), color=colors.pop(), role=0))
    _require(len(pitches) == 1, 'one visible local stacking pitch is required')
    pitch = pitches.pop()
    _require(pitch > side and len(bottom_rows) == 1, 'tokens need gaps and a common baseline')
    bottom = bottom_rows.pop()
    left, right = stacks[0]['x'], stacks[-1]['x']
    _require(len(stacks) >= 2 and all((s['x']-left) % pitch == 0 for s in stacks),
             'horizontal socket spacing must use the vertical pitch')
    length = (right-left)//pitch + 1
    _require(length <= plan.shape[1], 'bank fragment is wider than the miniature')
    sequence = [bg] * length
    for stack in stacks:
        sequence[(stack['x']-left)//pitch] = stack['color']
    controls = {}
    rows = set()
    for y, x, width, role in bars:
        _require(width == side and x in groups and x not in controls, 'bar must align with one stack')
        controls[x] = role
        rows.add(y)
    _require(len(rows) == 1, 'controls need a common row')
    control_y = rows.pop()
    _require(control_y >= bottom+side+1, 'a blank gap must separate bars from tokens')
    for stack in stacks:
        stack['role'] = controls.get(stack['x'], 0)
    matches = _matches(plan, sequence, bg)
    _require(len(matches) == 1, 'literal socket sequence must have exactly one miniature match')
    p = dict(grid=a, background=bg, plan=plan, miniature=(my, mx, ey, ex),
             labels=_labels(plan != bg), stacks=stacks, side=side, pitch=pitch,
             left=left, bottom=bottom, control_y=control_y, sequence=sequence,
             match=matches[0], controls=controls)
    p['routes'] = _routes(p)
    _check_target_space(p, p['routes'])
    return p


def _check_target_space(parsed, routes):
    a, side, pitch = parsed['grid'], parsed['side'], parsed['pitch']
    top = min(s['ys'][0] for s in parsed['stacks'])
    for dst in parsed['stacks']:
        if dst['x'] not in routes:
            continue
        count = dst['height'] + sum(s['height'] for s in routes[dst['x']])
        y = parsed['bottom'] - (count-1)*pitch
        _require(y >= 0, 'complete sum would leave the canvas')
        top = min(top, y)
        for k in range(dst['height'], count):
            yy = parsed['bottom'] - k*pitch
            _require(np.all(a[yy:yy+side, dst['x']:dst['x']+side] == parsed['background']),
                     'new destination tokens would collide with existing pixels')
    bank = (top, parsed['left'], parsed['control_y']+1,
            parsed['stacks'][-1]['x']+side)
    _require(not _overlap(bank, parsed['miniature']), 'bank and miniature rectangles overlap')


def _render_solution(parsed, routes=None):
    routes = parsed['routes'] if routes is None else routes
    _check_target_space(parsed, routes)
    out = parsed['grid'].copy()
    side, pitch, bg = parsed['side'], parsed['pitch'], parsed['background']
    for stack in parsed['stacks']:
        x = stack['x']
        if stack['role']:
            out[parsed['control_y'], x:x+side] = bg
        if stack['role'] == SOURCE:
            for y in stack['ys']:
                out[y:y+side, x:x+side] = PROCESSED
        if x in routes:
            final = stack['height'] + sum(s['height'] for s in routes[x])
            for k in range(stack['height'], final):
                y = parsed['bottom'] - k*pitch
                out[y:y+side, x:x+side] = stack['color']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Match sockets, trace logical circuits, then sum their marked source counts."""
    return _render_solution(_parse_input(input_grid))


def _check_counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError('counts must be integers: train 2..10, test 1..4')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _geometry_options(mask, length, minimum_capacity):
    mh, mw = (2*n-1 for n in mask.shape)
    options = []
    for side in range(2, 5):
        for gutter in range(1, 4):
            pitch = side+gutter
            bw = (length-1)*pitch+side
            if bw > 30:
                continue
            for capacity in range(minimum_capacity, 10):
                bh = (capacity-1)*pitch+side+2  # one blank row, then controls
                if bh <= 30 and (bh+mh+1 <= 30 or bw+mw+1 <= 30):
                    options.append((side, pitch, capacity))
    return options


def _partition(rng, total, number, unequal=False):
    if number == 1:
        return [total]
    cuts = sorted(rng.choice(np.arange(1, total), number-1, replace=False).tolist())
    values = np.diff([0]+cuts+[total]).tolist()
    if unequal and len(set(values)) == 1:
        values[0] -= 1
        values[1] += 1
    rng.shuffle(values)
    return values


def _random_model(rng, kind):
    """Sample occupancy/topology before counts or colors, with bounded rejection."""
    if kind != 'detour' and rng.random() < .18:
        return _separated_model(rng, kind)
    for attempt in range(700):
        if kind == 'detour':
            h, w = int(rng.integers(3, 7)), int(rng.integers(5, 10))
            r, c, length = int(rng.choice([0, h-1])), 0, w
        elif kind == 'test':
            h, w = int(rng.integers(3, 7)), int(rng.integers(7, 11))
            r = int(rng.integers(1, h-1))
            length = int(rng.integers(4, w-1))
            c = int(rng.integers(1, w-length))
        else:
            h, w = int(rng.integers(2, 7)), int(rng.integers(4, 10))
            r = int(rng.integers(h))
            length = int(rng.integers(3, w+1))
            c = int(rng.integers(w-length+1))
        mask = rng.random((h, w)) < float(rng.uniform(.38, .68))
        mask[r, c] = mask[r, c+length-1] = True
        if any(not edge.any() for edge in (mask[0], mask[-1], mask[:, 0], mask[:, -1])):
            continue
        occupied = np.argwhere(mask)
        if any(not np.any(np.diff(np.unique(occupied[:, d])) == 1) for d in (0, 1)):
            continue
        lab = _labels(mask)
        groups = defaultdict(list)
        for j in range(c, c+length):
            if mask[r, j]:
                groups[int(lab[r, j])].append(j)
        useful = [g for g in groups.values() if len(g) >= (4 if kind == 'detour' else 2)]
        if not useful:
            continue
        side_options = _geometry_options(mask, length, 4 if kind == 'detour' else 3)
        if not side_options:
            continue
        if kind == 'detour':
            bridges = [(g, d, s) for g in useful for d in g for s in g
                       if d != s and not np.all(mask[r, min(d, s):max(d, s)+1])]
            if not bridges:
                continue
            g, dst, far_source = bridges[int(rng.integers(len(bridges)))]
            other = [j for j in g if j not in (dst, far_source)]
            rng.shuffle(other)
            sources = [far_source, other[0]]
            unmarked_witness = other[1]
            selected = [(g, dst, sources)]
        else:
            rng.shuffle(useful)
            active = int(rng.integers(1, min(3, len(useful))+1))
            selected = []
            for g in useful[:active]:
                order = rng.permutation(g).tolist()
                n = int(rng.integers(1, min(3, len(g)-1)+1))
                selected.append((g, order[0], order[1:n+1]))
            unmarked_witness = None
        side, pitch, capacity = side_options[int(rng.integers(len(side_options)))]
        roles = {}
        counts = {j: int(rng.integers(1, capacity+1)) for j in range(c, c+length) if mask[r, j]}
        acceptable = True
        for index, (_, dst, sources) in enumerate(selected):
            n = len(sources)
            choices = [(s, d) for s in range(n, capacity) for d in range(1, capacity-s+1)
                       if (kind != 'detour' or (s > n and s != d))
                       and (index > 0 or kind == 'detour' or s > n or d > 1)]
            if not choices:
                acceptable = False
                break
            total, d = choices[int(rng.integers(len(choices)))]
            heights = _partition(rng, total, n, unequal=kind == 'detour')
            counts[dst], roles[dst] = d, DESTINATION
            for j, height in zip(sources, heights):
                counts[j], roles[j] = height, SOURCE
        if not acceptable:
            continue
        return dict(mask=mask, match=(r, c), length=length, roles=roles, counts=counts,
                    side=side, pitch=pitch, reserve=capacity,
                    same_color=(selected[0][2][0], unmarked_witness) if unmarked_witness is not None else None)
    # A constructive bounded fallback retains the requested teaching/shift.
    if kind == 'detour':
        mask = np.array([[1, 1, 0, 1, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]], bool)
        return dict(mask=mask, match=(0, 0), length=5, roles={0:3, 1:2, 3:2},
                    counts={0:1, 1:2, 3:1, 4:2}, side=2, pitch=3, reserve=4, same_color=(1, 4))
    mask = np.array([[1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 0, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1]], bool)
    return dict(mask=mask, match=(1, 1), length=5, roles={1:3, 2:2, 4:2},
                counts={1:1, 2:2, 4:1, 5:2}, side=2, pitch=3, reserve=4, same_color=None)


def _separated_model(rng, kind):
    """Several small circuits: the same conservation rule scales to three."""
    number = int(rng.choice([2, 3]))
    widths = [2]*number
    gaps = rng.integers(1, 3, size=number-1).tolist()
    length = sum(widths)+sum(gaps)
    # Three pairs with two one-cell gaps fit an interior length-eight fragment.
    if length > 8:
        gaps = [1]*(number-1)
        length = sum(widths)+sum(gaps)
    padding = 2 if kind == 'test' else int(rng.integers(0, 3))
    left = 1 if kind == 'test' else int(rng.integers(padding+1))
    h = int(rng.integers(3, 6))
    r = int(rng.integers(1, h-1))
    mask = np.zeros((h, length+padding), bool)
    groups, x = [], left
    for i, width in enumerate(widths):
        group = list(range(x, x+width))
        lo = 0 if i == 0 else x
        hi = mask.shape[1] if i == number-1 else x+width
        mask[:, lo:hi] = rng.random((h, hi-lo)) < .65
        mask[r, group] = True
        mask[0, group[0]] = mask[-1, group[-1]] = True
        groups.append(group)
        x += width+(gaps[i] if i < number-1 else 0)
    mask[r, :left] = False
    mask[r, left+length:] = False
    mask[0, 0] = mask[-1, -1] = True
    if not np.any(np.diff(np.unique(np.argwhere(mask)[:, 0])) == 1):
        mask[r-1, groups[0][0]] = True
    options = _geometry_options(mask, length, 3)
    side, pitch, capacity = options[int(rng.integers(len(options)))]
    roles, counts = {}, {}
    for index, group in enumerate(groups):
        rng.shuffle(group)
        dst, src = group
        choices = [(s, d) for s in range(1, capacity) for d in range(1, capacity-s+1)
                   if index > 0 or s > 1 or d > 1]
        s, d = choices[int(rng.integers(len(choices)))]
        roles.update({dst: DESTINATION, src: SOURCE})
        counts.update({dst: d, src: s})
    return dict(mask=mask, match=(r, left), length=length, roles=roles, counts=counts,
                side=side, pitch=pitch, reserve=capacity, same_color=None)


def _matching_witness(rng):
    """Two occupancy-identical socket rows, with genuinely different routing.

    One board pairs adjacent sockets; the other pairs interleaved sockets using
    insulated upper/lower detours. A blank row isolates the two candidate boards.
    Both hypothetical routings fit the bank, so color matching is necessary.
    """
    length = int(rng.integers(7, 10))
    gap_extra = rng.multinomial(length-7, np.ones(3)/3)
    ports = [0]
    for extra in gap_extra:
        ports.append(ports[-1]+2+int(extra))
    padding = int(rng.integers(1, 11-length))
    pad_left = int(rng.integers(padding+1))
    width = length+padding
    ports = [j+pad_left for j in ports]
    a, b, c, d = ports
    mask = np.zeros((8, width), bool)
    mask[0, ports] = True
    mask[1, :b+1] = True
    mask[1, c:] = True
    mask[3, :c+1] = True
    mask[4, [a, c]] = True
    mask[5, ports] = True
    mask[6, [b, d]] = True
    mask[7, b:] = True
    row = 5  # Interior crossed wiring is the explanatory match; boundary wiring is the decoy.
    if rng.integers(2):
        mask = mask[::-1].copy()
        row = 7-row
    if rng.integers(2):
        mask = mask[:, ::-1].copy()
        ports = sorted(width-1-j for j in ports)
        pad_left = width-pad_left-length
    lab = _labels(mask)
    groups = defaultdict(list)
    for j in ports:
        groups[int(lab[row, j])].append(j)
    # End sockets are destinations, inner sockets sources in either circuit board.
    roles = {ports[0]: DESTINATION, ports[-1]: DESTINATION,
             ports[1]: SOURCE, ports[2]: SOURCE}
    source_heights = [1, 2]
    destination_heights = [1, 2]
    rng.shuffle(source_heights)
    rng.shuffle(destination_heights)
    counts = dict(zip((ports[1], ports[2], ports[0], ports[-1]),
                      source_heights+destination_heights))
    return dict(mask=mask, match=(row, pad_left), length=length,
                roles=roles, counts=counts, side=2, pitch=3, reserve=4, same_color=None)


def _colorize(rng, model):
    bg = int(rng.choice(FREE_COLORS))
    pool = [v for v in FREE_COLORS if v != bg]
    palette = rng.choice(pool, int(rng.integers(2, len(pool)+1)), replace=False)
    r, c = model['match']
    for _ in range(40):
        plan = np.full(model['mask'].shape, bg, np.int8)
        plan[model['mask']] = rng.choice(palette, int(model['mask'].sum()))
        if model['same_color']:
            src, unmarked = model['same_color']
            plan[r, unmarked] = plan[r, src]
        seq = plan[r, c:c+model['length']]
        if len(_matches(plan, seq, bg)) == 1:
            break
    else:
        # A destination-only color makes the sequence unique without changing roles.
        dst = next(j for j, role in model['roles'].items() if role == DESTINATION)
        unique = pool[-1]
        replacement = next(v for v in pool if v != unique)
        plan[plan == unique] = replacement
        plan[r, dst] = unique
    model = dict(model)
    model.update(plan=plan, background=bg)
    return model


def _place(rng, model):
    mh, mw = (2*n-1 for n in model['mask'].shape)
    side, pitch, reserve = model['side'], model['pitch'], model['reserve']
    bw = (model['length']-1)*pitch+side
    gap_control_options = [g for g in (1, 2, 3)
        if ((reserve-1)*pitch+side+g+1+mh+1 <= 30
            or (bw+mw+1 <= 30 and (reserve-1)*pitch+side+g+1 <= 30))]
    control_gap = int(rng.choice(gap_control_options))
    bh = (reserve-1)*pitch+side+control_gap+1
    axes = [axis for axis in (0, 1)
            if (bh+mh+1 <= 30 if axis == 0 else bw+mw+1 <= 30)]
    axis = int(rng.choice(axes))
    dims_bank, dims_mini = (bh, bw), (mh, mw)
    spare = 30-dims_bank[axis]-dims_mini[axis]
    gap = int(rng.integers(1, min(3, spare)+1))
    required = [max(bh, mh), max(bw, mw)]
    required[axis] = dims_bank[axis]+dims_mini[axis]+gap
    mode = float(rng.random())
    if mode < .13:
        shape = (30, 30)
    elif mode < .28:
        n = int(rng.integers(max(required), 31))
        shape = (n, n)
    else:
        shape = tuple(int(rng.integers(n, min(30, n+4)+1)) for n in required)
    bank, mini = [0, 0], [0, 0]
    extra = shape[axis]-required[axis]
    margins = rng.multinomial(extra, np.ones(3)/3)
    if rng.integers(2):
        bank[axis] = int(margins[0])
        mini[axis] = bank[axis]+dims_bank[axis]+gap+int(margins[1])
    else:
        mini[axis] = int(margins[0])
        bank[axis] = mini[axis]+dims_mini[axis]+gap+int(margins[1])
    other = 1-axis
    bank[other] = int(rng.integers(shape[other]-dims_bank[other]+1))
    mini[other] = int(rng.integers(shape[other]-dims_mini[other]+1))
    return shape, tuple(bank), tuple(mini), control_gap


def _draw_model(rng, model):
    shape, bank, mini, control_gap = _place(rng, model)
    bg, side, pitch = model['background'], model['side'], model['pitch']
    plan, (r, c) = model['plan'], model['match']
    inp = np.full(shape, bg, np.int8)
    mh, mw = plan.shape
    inp[mini[0]:mini[0]+2*mh:2, mini[1]:mini[1]+2*mw:2] = plan
    bottom = bank[0]+(model['reserve']-1)*pitch
    control_y = bottom+side+control_gap
    labels = _labels(model['mask'])
    contributions = defaultdict(int)
    for j, role in model['roles'].items():
        if role == SOURCE:
            contributions[int(labels[r, j])] += model['counts'][j]
    for j, height in model['counts'].items():
        x = bank[1]+(j-c)*pitch
        for k in range(height):
            y = bottom-k*pitch
            inp[y:y+side, x:x+side] = plan[r, j]
        if j in model['roles']:
            inp[control_y, x:x+side] = model['roles'][j]
    out = inp.copy()
    for j, height in model['counts'].items():
        role = model['roles'].get(j, 0)
        x = bank[1]+(j-c)*pitch
        if role:
            out[control_y, x:x+side] = bg
        if role == SOURCE:
            for k in range(height):
                y = bottom-k*pitch
                out[y:y+side, x:x+side] = PROCESSED
        elif role == DESTINATION:
            total = contributions[int(labels[r, j])]
            for k in range(height, height+total):
                y = bottom-k*pitch
                out[y:y+side, x:x+side] = plan[r, j]
    return {'input': inp, 'output': out}


def _key(a):
    return tuple(a.shape), np.asarray(a, np.int8).tobytes()


def _unique(pair, seen):
    if _key(pair['input']) not in seen:
        return pair
    # Forty-two distinct injective recolorings of two visible colors exceed
    # the fourteen-pair maximum. Reserved source/destination/processed stay fixed.
    bg = int(np.bincount(pair['input'].ravel(), minlength=10).argmax())
    color = next(int(v) for v in np.unique(pair['input']) if v in FREE_COLORS and v != bg)
    domain = [bg, color]+[v for v in FREE_COLORS if v not in (bg, color)]
    for first in FREE_COLORS:
        for second in FREE_COLORS:
            if first == second:
                continue
            targets = [first, second]+[v for v in FREE_COLORS if v not in (first, second)]
            lookup = np.arange(10, dtype=np.int8)
            lookup[domain] = targets
            candidate = {k: lookup[v] for k, v in pair.items()}
            if _key(candidate['input']) not in seen:
                return candidate
    raise RuntimeError('finite uniqueness construction exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ns = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    for index in range(nt+ns):
        kind = 'detour' if index == 0 else 'witness' if index == 1 else 'extra' if index < nt else 'test'
        model = _matching_witness(rng) if kind == 'witness' else _random_model(rng, kind)
        pair = _unique(_draw_model(rng, _colorize(rng, model)), seen)
        seen.add(_key(pair['input']))
        puzzle['train' if index < nt else 'test'].append(pair)
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _color_matching_witness(parsed):
    for match in _matches(parsed['plan'], parsed['sequence'], parsed['background'], True):
        if match == parsed['match']:
            continue
        try:
            other = _routes(parsed, match)
            answer = _render_solution(parsed, other)
        except ValueError:
            continue
        if not np.array_equal(answer, _render_solution(parsed)):
            return True
    return False


def _features(parsed):
    stacks, routes = parsed['stacks'], parsed['routes']
    circuits = []
    for dst in stacks:
        if dst['x'] in routes:
            heights = tuple(s['height'] for s in routes[dst['x']])
            circuits.append((heights, dst['height']))
    unmarked = [s for s in stacks if not s['role']]
    sources = [s for s in stacks if s['role'] == SOURCE]
    r, c = parsed['match']
    detour = False
    for dst in stacks:
        for src in routes.get(dst['x'], []):
            j = c+(src['x']-parsed['left'])//parsed['pitch']
            k = c+(dst['x']-parsed['left'])//parsed['pitch']
            detour |= bool(np.any(parsed['plan'][r, min(j,k):max(j,k)+1] == parsed['background']))
    return dict(canvas=tuple(parsed['grid'].shape), diagram=tuple(parsed['plan'].shape),
                side=parsed['side'], pitch=parsed['pitch'], match=parsed['match'],
                length=len(parsed['sequence']), holes=parsed['sequence'].count(parsed['background']),
                sockets=len(stacks), circuits=len(circuits), operands=circuits,
                background=parsed['background'], miniature=parsed['miniature'][:2],
                bank=(parsed['bottom'], parsed['left']),
                detour=bool(detour), unmarked=len(unmarked),
                repeated_source_unmarked=any(s['color'] == u['color'] for s in sources for u in unmarked),
                interior_columns=c > 0 and c+len(parsed['sequence']) < parsed['plan'].shape[1],
                interior_row=0 < r < parsed['plan'].shape[0]-1)


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'bad puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'bad pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'output shape changed')
            key = _key(inp)
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            parsed = _parse_input(inp)
            expected = _render_solution(parsed)
            _require(np.array_equal(out, expected), 'wrong routing, arithmetic or preservation')
            _require(not np.array_equal(inp, out), 'identity pair')
            if split == 'train':
                training.append(parsed)
    fs = [_features(p) for p in training]
    operands = [operand for f in fs for operand in f['operands']]
    _require(any(len(s) >= 2 and len(set(s)) >= 2 for s, d in operands),
             'training must teach unequal multiple-source aggregation')
    _require(any(f['circuits'] >= 2 for f in fs), 'training must teach independent circuit totals')
    _require(any(f['detour'] for f in fs), 'training must teach connections outside the socket row')
    _require(any(f['repeated_source_unmarked'] for f in fs), 'training must teach spatial marker roles')
    _require(any(_color_matching_witness(p) for p in training), 'training must require literal sequence matching')
    _require(len({sum(s) for s, d in operands}) > 1 and len({d for s, d in operands}) > 1,
             'source totals and destination counts must vary independently')
    s0, d0 = sum(operands[0][0]), operands[0][1]
    _require(any(sum(s)*d0 != s0*d for s, d in operands), 'a destination multiplier fits training')
    _require(any(any(abs(s['x']-dst['x']) > min(abs(s['x']-d['x'])
                 for d in p['stacks'] if d['role'] == DESTINATION)
                 for s in p['routes'][dst['x']])
                 for p in training for dst in p['stacks'] if dst['role'] == DESTINATION),
             'training must distinguish wiring from nearest-destination routing')
    rivals = (lambda s, d: 2*d, lambda s, d: d+len(s), lambda s, d: d+max(s),
              lambda s, d: sum(s), lambda s, d: d*sum(s))
    for rival in rivals:
        _require(any(rival(s, d) != d+sum(s) for s, d in operands), 'arithmetic shortcut fits training')


def _signature(puzzle):
    return tuple((_key(np.asarray(pair['input'])), _key(np.asarray(pair['output'])))
                 for split in ('train', 'test') for pair in puzzle[split])


def self_test():
    """Bounded audit of 42 declared seeds, all counts, ablations and mutations."""
    puzzles = [generate_puzzle(seed) for seed in AUDIT_SEEDS]
    _require(len({_signature(p) for p in puzzles}) == 42, 'seed diversity failure')
    _require(_signature(puzzles[0]) == _signature(generate_puzzle(0)), 'replay differs')
    records = []
    for seed, puzzle in zip(AUDIT_SEEDS, puzzles):
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = np.asarray(pair['input'])
                old = a.copy()
                predicted = solve(a)
                _require(np.array_equal(a, old), 'solve mutates input')
                _require(np.array_equal(predicted, pair['output']), 'input-only solve differs')
                parsed = _parse_input(a)
                f = _features(parsed)
                f.update(seed=seed, split=split)
                records.append(f)
                # Source-count ablation: count bars rather than tokens.
                if any(sum(s['height'] for s in src) > len(src) for src in parsed['routes'].values()):
                    fake = dict(parsed)
                    fake['routes'] = {x: [dict(s, height=1) for s in src]
                                      for x, src in parsed['routes'].items()}
                    _require(not np.array_equal(_render_solution(fake), predicted), 'count ablation had no effect')
                # Blueprint ablation: route every source to every destination.
                if len(parsed['routes']) > 1:
                    all_sources = [s for s in parsed['stacks'] if s['role'] == SOURCE]
                    _require(all(sum(s['height'] for s in src) < sum(s['height'] for s in all_sources)
                                 for src in parsed['routes'].values()), 'routing ablation had no effect')
    for nt in range(2, 11):
        for ns in range(1, min(4, nt-1)+1):
            p = generate_puzzle(71+nt*5+ns, nt, ns)
            _require(len(p['train']) == nt and len(p['test']) == ns, 'explicit count failure')
    for nt, ns in ((1,1), (11,1), (4,0), (6,5), (2,2), (3,4),
                   (True,1), (4,False), (2.5,1), (None,1), (4,None), ('4',1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    for defect in ('addition', 'highlight', 'control', 'miniature', 'unmarked'):
        broken = {split: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][split]]
                  for split in ('train', 'test')}
        changed = False
        for pair in broken['train']:
            parsed = _parse_input(pair['input'])
            side = parsed['side']
            if defect == 'addition':
                dst = next(s for s in parsed['stacks'] if s['role'] == DESTINATION)
                y, x = dst['ys'][0]-parsed['pitch'], dst['x']
            elif defect == 'highlight':
                source = next(s for s in parsed['stacks'] if s['role'] == SOURCE)
                y, x = source['ys'][0], source['x']
            elif defect == 'control':
                y, x = parsed['control_y'], next(iter(parsed['controls']))
            elif defect == 'miniature':
                y, x = parsed['miniature'][:2]
            else:
                unmarked = [s for s in parsed['stacks'] if not s['role']]
                if not unmarked:
                    continue
                y, x = unmarked[0]['ys'][0], unmarked[0]['x']
            pair['output'][y, x] = (int(pair['output'][y, x])+1) % 10
            changed = True
            break
        _require(changed, 'missing mutation witness')
        try:
            validate_generated_puzzle(broken)
        except ValueError:
            pass
        else:
            raise AssertionError('bad '+defect+' accepted')
    # Reject input ambiguity and invalid circuit ownership, beyond output errors.
    for pair in puzzles[0]['train']:
        parsed = _parse_input(pair['input'])
        if not _color_matching_witness(parsed):
            continue
        alternatives = _matches(parsed['plan'], parsed['sequence'], parsed['background'], True)
        rr, cc = next(m for m in alternatives if m != parsed['match'])
        ambiguous = parsed['grid'].copy()
        my, mx = parsed['miniature'][:2]
        for j, color in enumerate(parsed['sequence']):
            ambiguous[my+2*rr, mx+2*(cc+j)] = color
        try:
            solve(ambiguous)
        except ValueError:
            pass
        else:
            raise AssertionError('ambiguous literal socket match accepted')
        break
    for pair in puzzles[0]['train']:
        parsed = _parse_input(pair['input'])
        unmarked = [s for s in parsed['stacks'] if not s['role']]
        if not unmarked:
            continue
        malformed = parsed['grid'].copy()
        x = unmarked[0]['x']
        malformed[parsed['control_y'], x:x+parsed['side']] = DESTINATION
        try:
            solve(malformed)
        except ValueError:
            pass
        else:
            raise AssertionError('a second or sourceless destination was accepted')
        break
    summary = {}
    for split in ('train', 'test'):
        group = [r for r in records if r['split'] == split]
        summary[split] = {name: len({r[name] for r in group}) for name in
            ('canvas', 'diagram', 'side', 'pitch', 'match', 'length', 'background', 'bank', 'miniature')}
        _require(summary[split]['canvas'] >= 15 and summary[split]['diagram'] >= 8
                 and summary[split]['match'] >= 8 and summary[split]['bank'] >= 20,
                 'insufficient structural variation')
    _require({2,3,4} <= {r['side'] for r in records}, 'token sizes did not vary')
    _require(len({r['pitch'] for r in records}) >= 4, 'lattice pitches did not vary')
    _require(len({r['background'] for r in records}) == 7, 'background variation missing')
    _require(any(r['circuits'] >= 3 for r in records), 'three-circuit case missing')
    _require(any(r['canvas'] == (30,30) for r in records), 'full canvas case missing')
    _require(all(r['interior_columns'] and r['interior_row'] for r in records if r['split'] == 'test'),
             'interior-fragment query shift missing')
    summary['distinct_puzzles'] = 42
    summary['pairs'] = len(records)
    summary['within_puzzle'] = {name: sum(len({r[name] for r in records if r['seed'] == seed}) > 1
                              for seed in AUDIT_SEEDS) for name in ('canvas','diagram','pitch','sockets','match')}
    return summary
