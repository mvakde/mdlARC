"""Dock and draw: arrival-ordered glyphs enabled and sized by leading cavities.

Only standard Python and NumPy are used. solve reads no examples or RNG state.
"""
from itertools import product
import numpy as np


_MASKS = (
    np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
)
# Right, left, down in the seed-relative canonical scene.
_DIRECTIONS = ((0, 1), (0, -1), (1, 0))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, lo, hi=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < lo or (hi is not None and value > hi):
        raise ValueError(f"{name} is out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    answer = []
    for r, c in zip(*np.nonzero(mask)):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        cells, pending = [], [(r, c)]
        while pending:
            y, x = pending.pop()
            cells.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    pending.append((yy, xx))
        answer.append(cells)
    return answer


def _cavities(mask):
    """Flood-fill actual background; glyph ink also counts as occupied."""
    h, w = mask.shape
    holes = []
    for cells in _components(~mask):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells):
            continue
        rows = {r for r, c in cells}
        contact = {r for r, c in cells if c == w - 2 and mask[r, w - 1]}
        holes.append(dict(cells=cells, top=min(rows), left=min(c for r, c in cells),
                          height=len(rows), active=rows == contact,
                          partial=bool(contact) and rows != contact))
    return sorted(holes, key=lambda hole: (hole['top'], hole['left']))


def _scene(input_grid):
    a = _grid(input_grid)
    candidates = []
    for turn in range(4):
        g = np.rot90(a, turn)
        h, w = g.shape
        for col in range(1, w - 1):
            rail = int(g[0, col])
            if not np.all(g[:, col] == rail) or np.count_nonzero(g == rail) != h:
                continue
            pane = g[:, col + 1:]
            values, counts = np.unique(pane, return_counts=True)
            bg = int(values[np.argmax(counts)])
            marks = np.argwhere(pane != bg)
            if (bg != rail and marks.shape == (1, 2) and marks[0, 0] == 0
                    and pane[tuple(marks[0])] != rail and np.any(g[:, :col] != bg)):
                candidates.append((g, turn, bg, rail, col, int(marks[0, 1])))
    _require(len(candidates) == 1, "need one exclusive rail and seeded pane")
    g, turn, bg, rail, col, start = candidates[0]
    return dict(grid=g, turn=turn, bg=bg, rail=rail, col=col, start=start,
                marker=int(g[0, col + 1 + start]), width=g.shape[1] - col - 1)


def _devices(info):
    g, bg, col = info['grid'], info['bg'], info['col']
    objects = []
    for cells in _components(g[:, :col] != bg):
        rr, cc = np.array(cells).T
        r0, r1, c0, c1 = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())
        local = g[r0:r1 + 1, c0:c1 + 1]
        occupied = np.zeros_like(local, dtype=bool)
        occupied[rr - r0, cc - c0] = True
        colors, counts = np.unique(g[rr, cc], return_counts=True)
        _require(len(colors) == 2, "device needs body and display ink")
        body = int(colors[np.argmax(counts)])
        ink = int(colors[np.argmin(counts)])
        _require(counts.max() > counts.sum() / 2, "body needs a strict majority")
        _require(body not in (bg, info['rail'], info['marker']) and
                 ink not in (bg, info['rail'], info['marker']), "reserved color in device")
        _require(local.shape[0] >= 6 and local.shape[1] >= 5, "display cap too small")
        _require(np.all(occupied[:4, :3]), "solid four-row display backing required")
        tag = local[:3, :3] == ink
        matches = [k for k, mask in enumerate(_MASKS) if np.array_equal(tag, mask)]
        _require(len(matches) == 1 and np.count_nonzero(local == ink) == tag.sum(),
                 "unrecognized or extra display ink")
        holes = _cavities(occupied)
        _require(not any(h['partial'] for h in holes), "partial-contact cavities excluded")
        objects.append(dict(cells=cells, row=r0, bottom=r1, left=c0, right=c1,
                            shift=col - 1 - c1, body=body, ink=ink, command=matches[0],
                            mask=occupied, holes=holes))
    objects.sort(key=lambda o: o['row'])
    _require(objects and objects[0]['row'] >= 1, "blank top work margin required")
    _require(all(b['row'] > a['bottom'] + 1 for a, b in zip(objects, objects[1:])),
             "device bands must be separated")
    return objects


def _row_runs(rows):
    runs = []
    for r in sorted(set(rows)):
        if not runs or r != runs[-1][-1] + 1:
            runs.append([r])
        else:
            runs[-1].append(r)
    return runs


def _events(objects, ordering='arrival', gating='active', merge=False,
            width_operand=False, distance_operand=False, first_only=False):
    if ordering == 'arrival':
        ordered = sorted(objects, key=lambda o: (o['shift'], o['row']))
    elif ordering == 'reverse_ties':
        ordered = sorted(objects, key=lambda o: (o['shift'], -o['row']))
    elif ordering == 'vertical':
        ordered = sorted(objects, key=lambda o: o['row'])
    else:
        raise ValueError('unknown diagnostic ordering')
    events = []
    for obj in ordered:
        if gating == 'column':
            # Deliberately incorrect shortcut from B's review: enclosure omitted.
            heights = [len(run) for run in _row_runs(
                r for r in range(4, obj['mask'].shape[0]) if not obj['mask'][r, -2])]
        else:
            selected = [h for h in obj['holes'] if gating == 'all_holes' or h['active']]
            heights = [(max(c for r, c in h['cells']) - h['left'] + 1
                        if width_operand else h['height']) for h in selected]
        if distance_operand:
            heights = [obj['shift']] * len(heights)
        if first_only:
            heights = heights[:1]
        if merge and heights:
            heights = [sum(heights)]
        events.extend((obj['command'], height + 1, obj['ink']) for height in heights)
    return events


def _draw_pane(info, events, chained=True, constant=False):
    pane = info['grid'][:, info['col'] + 1:].copy()
    row, col = 0, info['start']
    for command, size, ink in events:
        if not chained:
            row, col = 0, info['start']
        if constant:
            size = 2
        dr, dc = _DIRECTIONS[command]
        rr = row + 1 + dr * np.arange(size)
        cc = col + dc * np.arange(size)
        _require(rr.max() < pane.shape[0] and cc.min() >= 0 and cc.max() < pane.shape[1],
                 'stroke would leave pane')
        if chained:
            _require(np.all(pane[rr, cc] == info['bg']), 'stroke intersection')
        pane[rr, cc] = ink
        row, col = int(rr[-1]), int(cc[-1])
    return pane


def _render(info, objects, events, chained=True, constant=False):
    g = info['grid']
    out = g.copy()
    out[:, :info['col']] = info['bg']
    for obj in objects:
        rr, cc = np.array(obj['cells']).T
        out[rr, cc + obj['shift']] = g[rr, cc]
    out[:, info['col'] + 1:] = _draw_pane(info, events, chained, constant)
    return np.rot90(out, -info['turn']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive roles, docking priorities, cavities and commands from this input."""
    info = _scene(input_grid)
    objects = _devices(info)
    return _render(info, objects, _events(objects))


def _different_variant(pair, parsed=None, **kwargs):
    """Authored ablation, not an independent reference solver."""
    if parsed is None:
        info = _scene(pair['input'])
        objects = _devices(info)
    else:
        info, objects = parsed
    chained = kwargs.pop('chained', True)
    constant = kwargs.pop('constant', False)
    try:
        candidate = _render(info, objects, _events(objects, **kwargs), chained, constant)
    except AssertionError:
        return True  # This alternative does not even fit this demonstrated canvas.
    return not np.array_equal(candidate, pair['output'])


def _device_mask(rng, kind, heights, width, width_contrast=False):
    """A solid display cap above separately cut, procedurally sized pockets."""
    h = 4 + sum(heights) + len(heights)
    mask = np.ones((h, width), dtype=bool)
    row = 4
    for index, height in enumerate(heights):
        is_decoy = kind == 'mixed' and index % 2 == 1
        end = width - 3 if is_decoy else width - 2
        left = int(rng.integers(1, end + 1))
        for r in range(row, row + height):
            back = int(rng.integers(1, end + 1)) if rng.random() < .45 else left
            if width_contrast and index == 0:
                back = int(rng.integers(1, end))  # first height-one cavity is wider than one
            mask[r, back:end + 1] = False
        if kind == 'open':
            door = int(rng.integers(3))
            if door == 0:
                mask[row:row + height, :end + 1] = False
            elif door == 1:
                mask[row:row + height, end + 1:] = False
            else:
                # Open to the bottom. Open devices have exactly one pocket.
                mask[row:, end] = False
        row += height + 1
    if kind == 'solid':
        mask[:] = True
    if kind == 'recessed':
        extension = int(rng.integers(1, 3))
        extended = np.zeros((h, width + extension), dtype=bool)
        extended[:, :width] = mask
        extended[0, width:] = True  # a thin tail defines the global leading extreme
        mask = extended
    _require(len(_components(mask)) == 1, 'builder made disconnected device')
    return mask


def _specs(rng, regime):
    if regime == 'anchor_program':
        return [('active', [1, 2], 0), ('active', [3], 1), ('active', [2], 2)]
    if regime == 'anchor_topology':
        return [('active', [1], 1), ('active', [1], 2),
                ('open', [1], int(rng.integers(3))),
                ('recessed', [1], int(rng.integers(3)))]
    if regime == 'test':
        total = int(rng.integers(5, 8))
        count = int(rng.integers(1, 4))
        # Positive composition of event count; counts do not label command types.
        counts = np.ones(count, dtype=int)
        for _ in range(total - count):
            counts[int(rng.integers(count))] += 1
        specs = [('active', [1] * int(n), int(rng.integers(3))) for n in counts]
        base = sum(4 + 2 * len(x[1]) for x in specs) + count  # includes top margin
        if base + 7 <= 30 and rng.random() < .4:
            kind = str(rng.choice(['open', 'recessed', 'solid']))
            specs.append((kind, [1], int(rng.integers(3))))
    else:
        count = int(rng.integers(1, 5))
        specs = []
        active_budget = 4
        for _ in range(count):
            kind = str(rng.choice(['active', 'active', 'mixed', 'open', 'recessed', 'solid']))
            if not active_budget and kind in ('active', 'mixed'):
                kind = 'open'
            n = 1
            if count <= 3 and kind == 'active' and active_budget >= 2 and rng.random() < .35:
                n = 2
            if kind == 'mixed' and count <= 3:
                n = 2
            if kind in ('active', 'mixed'):
                active_budget -= n if kind == 'active' else 1
            specs.append((kind, [1] * n, int(rng.integers(3))))
    # Allocate height increases only where the complete input fits. Horizontal
    # excursion is checked separately after arrival order and directions exist.
    spare = 30 - (sum(4 + sum(hs) + len(hs) for _, hs, _ in specs) + len(specs))
    indices = [(i, j) for i, (_, hs, _) in enumerate(specs) for j in range(len(hs))]
    rng.shuffle(indices)
    for i, j in indices:
        maximum = min(2, spare)
        delta = int(rng.integers(maximum + 1))
        specs[i][1][j] += delta
        spare -= delta
    return specs


def _make_pair(rng, regime, roles, inks, turn):
    bg, rail, marker = roles
    specs = _specs(rng, regime)
    items = []
    body_colors = [v for v in range(10) if v not in roles and v not in inks]
    for kind, heights, command in specs:
        width = int(rng.integers(5, 9))
        mask = _device_mask(rng, kind, heights, width,
                            width_contrast=regime == 'anchor_program' and command == 0)
        if regime == 'anchor_program':
            ink = inks[command]
        elif regime == 'anchor_topology' and kind == 'active':
            ink = inks[3 - command]  # left/down exchange inks between anchors
        else:
            ink = int(rng.choice(inks))
        items.append(dict(mask=mask, body=int(rng.choice(body_colors)), ink=int(ink),
                          command=command, kind=kind))
    rng.shuffle(items)
    widest = max(x['mask'].shape[1] for x in items)
    work = widest + int(rng.integers(1, min(3, 13 - widest) + 1))
    for obj in items:
        obj['shift'] = int(rng.integers(work - obj['mask'].shape[1] + 1))
    if regime == 'anchor_program':
        # Three distinct displays: an arrival inversion and two simultaneous arrivals.
        for obj, shift in zip(items, (1, 0, 1)):
            obj['shift'] = shift
    elif regime == 'anchor_topology':
        items[0]['shift'] = 0
        items[-1]['shift'] = 1
    # Program geometry is derived from masks, not kind labels.
    def program():
        seq = []
        for _, obj in sorted(enumerate(items), key=lambda x: (x[1]['shift'], x[0])):
            seq.extend((obj['command'], hole['height'] + 1, obj['ink'])
                       for hole in _cavities(obj['mask']) if hole['active'])
        return seq
    events = program()
    def extent(seq):
        row, col, low, high = 0, 0, 0, 0
        for command, size, _ in seq:
            dr, dc = _DIRECTIONS[command]
            row += 1 + dr * (size - 1)
            col += dc * (size - 1)
            low, high = min(low, col), max(high, col)
        return row + 1, low, high
    path_h, lo, hi = extent(events)
    # Bounded command reassignment for random scenes with a large excursion.
    # Anchor directions never change. All-down is always a geometric fallback.
    if hi - lo + 1 > 29 - work:
        _require(regime not in ('anchor_program', 'anchor_topology'), 'anchor too wide')
        choices = list(product(range(3), repeat=len(items)))
        rng.shuffle(choices)
        for commands in choices:
            for obj, command in zip(items, commands):
                obj['command'] = command
            events = program()
            path_h, lo, hi = extent(events)
            if hi - lo + 1 <= 29 - work:
                break
        else:
            raise AssertionError('no fitting direction assignment')
    min_pane = max(4, hi - lo + 1)
    pane_w = int(rng.integers(min_pane, 30 - work))
    start = int(rng.integers(-lo, pane_w - hi))
    base_h = sum(x['mask'].shape[0] for x in items) + len(items)
    minimum_h = max(base_h, path_h)
    height = int(rng.integers(minimum_h, 31))
    extras = rng.multinomial(height - base_h, np.full(len(items) + 1, 1 / (len(items) + 1)))
    g = np.full((height, work + 1 + pane_w), bg, dtype=np.int8)
    g[:, work] = rail
    g[0, work + 1 + start] = marker
    row = 1 + int(extras[0])
    for index, obj in enumerate(items):
        mask = obj['mask']
        left = work - mask.shape[1] - obj['shift']
        block = g[row:row + mask.shape[0], left:left + mask.shape[1]]
        block[mask] = obj['body']
        display = block[:3, :3]
        display[_MASKS[obj['command']]] = obj['ink']
        row += mask.shape[0] + 1 + int(extras[index + 1])
    g = np.rot90(g, turn).copy()
    return {'input': g, 'output': solve(g)}


def _unique(pair, roles, seen):
    # A seven-color odd-cycle dihedral orbit gives 14 distinct candidates;
    # at most 13 earlier inputs exist. Reserved colors remain fixed.
    colors = [v for v in range(10) if v not in roles]
    for sign in (1, -1):
        for offset in range(7):
            lookup = np.arange(10, dtype=np.int8)
            for i, color in enumerate(colors):
                lookup[color] = colors[(sign * i + offset) % 7]
            candidate = {k: lookup[v] for k, v in pair.items()}
            if _key(candidate['input']) not in seen:
                return candidate
    raise AssertionError('uniqueness orbit exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    palette = [int(v) for v in rng.permutation(10)]
    roles, inks = tuple(palette[:3]), tuple(palette[3:6])
    base_turn = int(rng.integers(4))
    mixed = bool(rng.integers(2))
    turns = [base_turn] * num_train
    if mixed:
        turns[1] = (base_turn + int(rng.integers(1, 4))) % 4
        for i in range(2, num_train):
            turns[i] = int(rng.integers(4))
    regimes = ['anchor_program', 'anchor_topology'] + ['random'] * (num_train - 2)
    seen = set()
    puzzle = {'train': [], 'test': []}
    # Give anchors uniqueness priority, preserving their shared-color contrast.
    for regime, turn in zip(regimes, turns):
        pair = _make_pair(rng, regime, roles, inks, turn)
        pair = _unique(pair, roles, seen)
        seen.add(_key(pair['input']))
        puzzle['train'].append(pair)
    rng.shuffle(puzzle['train'])
    for _ in range(num_test):
        pair = _make_pair(rng, 'test', roles, inks, int(rng.choice(turns)))
        pair = _unique(pair, roles, seen)
        seen.add(_key(pair['input']))
        puzzle['test'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'split schema')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'pair counts')
    seen, roles, counts = set(), set(), {'train': [], 'test': []}
    training_devices, training_parsed = [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'canvas dimensions changed')
            key = _key(inp)
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            info = _scene(inp)
            devices = _devices(info)
            roles.add((info['bg'], info['rail'], info['marker']))
            events = _events(devices)
            counts[split].append(len(events))
            _require(np.array_equal(solve(inp.copy()), out), 'output violates input-derived rule')
            canonical_out = np.rot90(out, info['turn'])
            _require(np.array_equal(canonical_out[:, info['col']], info['grid'][:, info['col']]),
                     'rail changed')
            _require(np.array_equal(canonical_out[0, info['col'] + 1:],
                                   info['grid'][0, info['col'] + 1:]), 'seed row changed')
            # Independently check exact foreground pixel/color preservation under docking.
            expected = np.full((inp.shape if info['turn'] % 2 == 0 else inp.shape[::-1]),
                               info['bg'], dtype=int)[:, :info['col']]
            for obj in devices:
                for r, c in obj['cells']:
                    expected[r, c + obj['shift']] = int(info['grid'][r, c])
            _require(np.array_equal(expected, canonical_out[:, :info['col']]), 'rigid docking failed')
            _require(set(map(int, np.unique(out))) <= set(map(int, np.unique(inp))),
                     'unexpected output-only color')
            if split == 'train':
                training_devices.extend(devices)
                training_parsed.append((pair, info, devices))
    _require(len(roles) == 1, 'semantic palette changed within puzzle')
    _require(max(counts['train']) <= 4 and min(counts['test']) >= 5,
             'longer-program test shift absent')
    _require({o['command'] for o in training_devices if any(h['active'] for h in o['holes'])}
             == {0, 1, 2}, 'all directions need positive demonstrations')
    heights_by_command = {k: {h['height'] for o in training_devices if o['command'] == k
                             for h in o['holes'] if h['active']} for k in range(3)}
    _require(all(len(heights) >= 2 for heights in heights_by_command.values()),
             'each glyph must demonstrate a variable cavity operand')
    _require(set.union(*heights_by_command.values()) >= {1, 2, 3},
             'training must cover the entire generated cavity-height range')
    by_ink = {}
    for obj in training_devices:
        if any(h['active'] for h in obj['holes']):
            by_ink.setdefault(obj['ink'], set()).add(obj['command'])
    _require(any(len(directions) > 1 for directions in by_ink.values()), 'color-only decoding remains possible')
    variants = [dict(ordering='vertical'), dict(ordering='reverse_ties'),
                dict(gating='all_holes'), dict(gating='column'), dict(constant=True),
                dict(merge=True), dict(chained=False), dict(width_operand=True),
                dict(distance_operand=True), dict(first_only=True)]
    for variant in variants:
        _require(any(_different_variant(pair, parsed=(info, devices), **variant)
                     for pair, info, devices in training_parsed),
                 f'training does not reject shortcut {variant}')


def puzzle_features(puzzle):
    """Measurements recovered from returned pixels, not builder configuration."""
    result = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info = _scene(pair['input'])
            objects = _devices(info)
            events = _events(objects)
            result.append(dict(split=split, shape=tuple(np.asarray(pair['input']).shape),
                               orientation=info['turn'], devices=len(objects), events=len(events),
                               work=info['col'], pane=info['width'], start=info['start'],
                               distances=tuple(o['shift'] for o in objects),
                               commands=tuple(e[0] for e in events), lengths=tuple(e[1] for e in events),
                               masks=tuple(_key(o['mask']) for o in objects),
                               hole_counts=tuple(len(o['holes']) for o in objects),
                               cavity_shapes=tuple((h['height'], max(c for r, c in h['cells']) - h['left'] + 1)
                                                   for o in objects for h in o['holes']),
                               silent=not events))
    return result


def _puzzle_key(p):
    return tuple(tuple((_key(pair['input']), _key(pair['output'])) for pair in p[s])
                 for s in ('train', 'test'))


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    features = [f for p in puzzles for f in puzzle_features(p)]
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, 'seed diversity collapsed')
    _require(len({f['shape'] for f in features}) >= 35, 'canvas geometry collapsed')
    _require({f['orientation'] for f in features} == set(range(4)), 'missing orientation')
    _require({f['devices'] for f in features} == {1, 2, 3, 4}, 'device count variation missing')
    _require({n for f in features if f['split'] == 'test' for n in [f['events']]} == {5, 6, 7},
             'test program count variation missing')
    _require(len({m for f in features for m in f['masks']}) >= 90, 'device shapes collapsed')
    _require({n for f in features for n in f['lengths']} >= {2, 3, 4}, 'stroke sizes collapsed')
    _require(any(f['silent'] for f in features), 'silent scene absent')
    _require(any(0 in f['distances'] and max(f['distances']) > 0 for f in features), 'zero/positive mixture absent')
    _require(len({f['start'] for f in features}) >= 5, 'seed positions collapsed')
    _require(len({f['pane'] for f in features}) >= 8, 'pane widths collapsed')
    p = generate_puzzle(123456, 7, 3)
    _require(_puzzle_key(p) == _puzzle_key(generate_puzzle(123456, 7, 3)), 'not deterministic')
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            q = generate_puzzle(0, a, b)
            _require(len(q['train']) == a and len(q['test']) == b, 'explicit count ignored')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    # Equivariance exercises solve with transforms never sent to the builder.
    for pair in puzzles[0]['train'] + puzzles[0]['test']:
        for turn in range(4):
            _require(np.array_equal(solve(np.rot90(pair['input'], turn).copy()),
                                    np.rot90(pair['output'], turn)), 'rotation rule failed')
    # Corruption checks target both a label and visible instruction geometry.
    for damage in ('label', 'display'):
        q = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in puzzles[0][s]]
             for s in ('train', 'test')}
        pair = q['test'][0]
        if damage == 'label':
            pair['output'][0, 0] = (int(pair['output'][0, 0]) + 1) % 10
        else:
            info = _scene(pair['input'])
            obj = _devices(info)[0]
            canonical = info['grid'].copy()
            block = canonical[obj['row']:obj['row'] + 3, obj['left']:obj['left'] + 3]
            block[block == obj['ink']] = obj['body']
            pair['input'] = np.rot90(canonical, -info['turn']).copy()
        try:
            validate_generated_puzzle(q)
        except AssertionError:
            continue
        raise AssertionError('semantic corruption accepted')
    return {'seeds': 42, 'pairs': len(features), 'shapes': len({f['shape'] for f in features}),
            'device_masks': len({m for f in features for m in f['masks']}),
            'device_counts': sorted({f['devices'] for f in features}),
            'test_command_counts': sorted({f['events'] for f in features if f['split'] == 'test'})}
