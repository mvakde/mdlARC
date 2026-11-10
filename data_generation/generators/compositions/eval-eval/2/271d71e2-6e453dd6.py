"""Synchronous aperture engines: topology selects the counters that pay for motion.

Fixed visible conventions: 0 background, 1 spent, 2 remaining; walls/rail 3..9.
Only the standard library and NumPy are used. No parent files are loaded.
"""

import numpy as np


def _need(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _need(np.all((a >= 0) & (a <= 9)), "ARC color range")
    return a


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " out of range")
    return int(value)


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    groups = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        stack, cells = [(sr, sc)], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for y, x in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= y < h and 0 <= x < w and mask[y, x] and not seen[y, x]:
                    seen[y, x] = True
                    stack.append((y, x))
        groups.append(cells)
    return groups


def _holes(wall):
    h, w = wall.shape
    return [p for p in _components(~wall)
            if all(r not in (0, h - 1) and c not in (0, w - 1) for r, c in p)]


def _order(cells):
    # Rightward travel is upward travel rotated clockwise: front columns first.
    return sorted(cells, key=lambda p: (-p[1], p[0]))


def _remaining(block, cells):
    return sum(int(block[p]) == 2 for p in cells)


def _scene(grid):
    """Return a uniquely determined rightward view, without color frequencies."""
    candidates = []
    for turn in range(4):
        g = np.rot90(grid, turn)
        h, w = g.shape
        for c in range(1, w - 1):
            color = int(g[0, c])
            if color < 3 or not np.all(g[:, c] == color):
                continue
            if np.count_nonzero(g == color) != h:
                continue
            if np.any(g[:, c + 1:] != 0) or not np.any(g[:, :c] != 0):
                continue
            candidates.append((g, turn, c, color))
    _need(len(candidates) == 1, "unique full rail and blank projection side required")
    return candidates[0]


def _read_engine(g, cells, rail):
    rr, cc = zip(*cells)
    t, b, l, r = min(rr), max(rr), min(cc), max(cc)
    block = g[t:b + 1, l:r + 1].copy()
    colors = set(int(g[p]) for p in cells if int(g[p]) >= 3)
    _need(len(colors) == 1, "one wall color per engine")
    color = colors.pop()
    wall = block == color
    _need(len(_components(wall)) == 1, "wall must be connected")
    wy, wx = np.nonzero(wall)
    _need((int(wy.min()), int(wy.max()), int(wx.min()), int(wx.max())) ==
          (0, block.shape[0] - 1, 0, block.shape[1] - 1), "wall defines engine bounds")
    active, decoy, closed_cells = [], [], set()
    for hole in _holes(wall):
        ordered = _order(hole)
        values = [int(block[p]) for p in ordered]
        used = values.count(1)
        _need(values == [1] * used + [2] * (len(values) - used), "closed chamber prefix")
        closed_cells.update(hole)
        contact = {y for y, x in hole if x == block.shape[1] - 2 and wall[y, x + 1]}
        if contact:
            _need(contact == {y for y, x in hole}, "partial leading contact excluded")
            active.append(ordered)
        else:
            decoy.append(ordered)
    active.sort(key=lambda room: min(room))
    decoy.sort(key=lambda room: min(room))
    open_mask = (block == 1) | (block == 2)
    for p in closed_cells:
        open_mask[p] = False
    opened = [_order(p) for p in _components(open_mask)]
    return dict(t=t, l=l, block=block, color=color, active=active, decoy=decoy,
                opened=opened, gap=rail - 1 - r)


def _read(g, rail):
    objects = [_read_engine(g, cells, rail) for cells in _components(g[:, :rail] != 0)]
    _need(bool(objects), "no engines")
    previous = -2
    for obj in objects:
        _need(obj['t'] > previous + 1, "separated transverse engine bands required")
        previous = obj['t'] + obj['block'].shape[0] - 1
        _need(obj['gap'] >= 0, "engine crosses rail")
    _need(np.any(np.all(g[:, :rail] == 0, axis=1)), "blank transverse line required")
    return objects


def _movement(obj, rooms=None):
    rooms = obj['active'] if rooms is None else rooms
    return min([obj['gap']] + [_remaining(obj['block'], p) for p in rooms]) if rooms else 0


def _render(g, rail, objects, policy='normal', unit_steps=False):
    """Render explicit local plans; policy variants are counterexample checks only."""
    out = g.copy()
    out[:, :rail] = 0
    for obj in objects:
        block = obj['block'].copy()
        rooms = obj['active']
        if policy == 'all_closed':
            rooms = rooms + obj['decoy']
        elif policy == 'all_pockets':
            rooms = rooms + obj['decoy'] + obj['opened']
        elif policy == 'edge_only':
            # Deliberately wrong: select counter pockets by local wall contact
            # without determining whether the pocket reaches the exterior.
            wall = block == obj['color']
            rooms = [p for p in _components((block == 1) | (block == 2))
                     if any(c == block.shape[1] - 2 and wall[r, c + 1] for r, c in p)]
        supplies = [_remaining(block, p) for p in rooms]
        d = min([obj['gap']] + supplies) if supplies else 0
        if policy == 'unlimited' and rooms:
            d = obj['gap']
        elif policy == 'sum' and supplies:
            d = min(obj['gap'], sum(supplies))
        elif policy == 'max' and supplies:
            d = min(obj['gap'], max(supplies))
        consumed = rooms
        if policy == 'limiter_only' and supplies:
            consumed = [p for p, supply in zip(rooms, supplies) if supply == min(supplies)]
        if unit_steps:
            # Tick each reservoir separately, rediscovering the next fuel pixel.
            for _ in range(d):
                for room in consumed:
                    pending = [p for p in _order(room) if block[p] == 2]
                    if pending:
                        block[pending[0]] = 1
        else:
            for room in consumed:
                pending = [p for p in _order(room) if block[p] == 2]
                for p in pending[:d]:
                    block[p] = 1
        t, l = obj['t'], obj['l'] + d
        h, w = block.shape
        out[t:t + h, l:l + w] = block
        if obj['active'] and (d == obj['gap'] or policy == 'always_signal'):
            projected = obj['active']
            if policy == 'project_all':
                projected = projected + obj['decoy'] + obj['opened']
            for row in {r for room in projected for r, c in room}:
                out[t + row, rail + 1:] = obj['color']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from pixels alone; accepts all structures in the written family."""
    g, turn, rail, color = _scene(_grid(input_grid))
    return np.rot90(_render(g, rail, _read(g, rail)), -turn).copy()


def _ri(rng, low, high):
    return int(rng.integers(low, high + 1))


def _intervals(rng, height, count, minimum=1):
    spare = height - (minimum * count + count + 1)
    _need(spare >= 0, "room intervals do not fit")
    bins = rng.multinomial(spare, np.full(2 * count + 1, 1 / (2 * count + 1)))
    intervals, row = [], 1 + int(bins[0])
    for i in range(count):
        length = minimum + int(bins[2 * i + 1])
        intervals.append(range(row, row + length))
        row += length + 1 + int(bins[2 * i + 2])
    return intervals


def _prefix(block, cells, remaining):
    order = _order(cells)
    for index, p in enumerate(order):
        block[p] = 1 if index < len(order) - remaining else 2


def _make_engine(rng, kind, budget, max_active, color):
    """Build chambers from coordinates, not the solver's enclosure classifier."""
    if kind == 'solid':
        h, w = _ri(rng, 2, min(10, budget)), _ri(rng, 2, 11)
        block = np.full((h, w), color, dtype=int)
        if rng.random() < .6:
            for row in range(1, h - 1):
                block[row, :_ri(rng, 0, w - 1)] = 0
        return dict(block=block, color=color, active=[], decoy=[], opened=[], gap=_ri(rng, 0, 8))
    demo = kind in ('dock', 'bottleneck')
    if kind == 'bottleneck':
        n = 2
    elif kind == 'multi':
        n = _ri(rng, 3, 4)
    else:
        n = _ri(rng, 1, min(max_active, (budget - 1) // 2))
    minimum = 2 if demo else 1
    # The docking lesson has an extra leading pocket with an exterior doorway.
    total_front = n + int(kind == 'dock')
    min_h = max(7 if demo else 3, (minimum + 1) * total_front + 1)
    h = _ri(rng, min_h, min(budget, 18, min_h + 6))
    left = kind == 'dock' or rng.random() < .55
    w = _ri(rng, 8 if left and demo else 7 if left else 5 if demo else 3, 13)
    divider = _ri(rng, 2, w - 4) if left else 0
    block = np.full((h, w), color, dtype=int)
    fronts, decoy, opened = [], [], []
    for rows in _intervals(rng, h, total_front, minimum):
        end = w - 2
        start_max = end - 1 if demo else end
        start = _ri(rng, divider + 1, start_max)
        coords = []
        stepped = rng.random() < .5
        for row in rows:
            edge = _ri(rng, divider + 1, start_max) if stepped else start
            coords.extend((row, col) for col in range(edge, end + 1))
        fronts.append(_order(coords))
    if kind == 'dock':
        pocket = set(fronts.pop(0))
        pocket.update((r, w - 2) for r in range(max(r for r, c in pocket) + 1))
        opened.append(_order(pocket))
    if left:
        left_count = 2 if kind == 'dock' else _ri(rng, 1, min(3, (h - 1) // 2))
        for i, rows in enumerate(_intervals(rng, h, left_count)):
            is_open = i == 1 if kind == 'dock' else rng.random() < .45
            start = 0 if is_open else _ri(rng, 1, divider - 1)
            coords = [(row, col) for row in rows for col in range(start, divider)]
            (opened if is_open else decoy).append(_order(coords))
    if kind == 'open':
        # A doorway on the top reaches the pocket behind the intact front wall.
        # Thus local leading-wall contact alone is insufficient for activation.
        pocket = {p for room in fronts for p in room}
        pocket.update((r, w - 2) for r in range(max(r for r, c in pocket) + 1))
        opened.append(_order(pocket))
        fronts = []
    if kind == 'recessed':
        # A thin forward tab changes the global extreme without changing the rooms.
        ext = _ri(rng, 1, 3)
        block = np.pad(block, ((0, 0), (0, ext)), constant_values=0)
        block[int(rng.choice((0, h - 1))), w:] = color
        decoy.extend(fronts)
        fronts = []
    caps = [len(room) for room in fronts]
    if kind == 'dock':
        gap = _ri(rng, 1, min(4, min(caps) - 2))
        supplies = [_ri(rng, gap + 1, cap - 1) for cap in caps]
    elif kind == 'bottleneck':
        small = _ri(rng, 1, min(3, min(caps) - 2))
        low = _ri(rng, 0, 1)
        supplies = [_ri(rng, small + 1, cap - 1) for cap in caps]
        supplies[low] = small
        gap = _ri(rng, small + 1, min(8, max(supplies)))
    elif fronts:
        regime = str(rng.choice(('surplus', 'fuel', 'exact') if kind == 'multi' else
                                ('surplus', 'fuel', 'exact', 'zero', 'aligned', 'random')))
        if regime == 'surplus' and min(caps) == 1:
            regime = 'exact'
        if regime == 'surplus':
            gap = _ri(rng, 1, min(5, min(caps) - 1))
            supplies = [_ri(rng, gap + 1, cap) for cap in caps]
        elif regime in ('fuel', 'exact'):
            base = _ri(rng, 1, min(5, min(caps)))
            supplies = [_ri(rng, base, cap) for cap in caps]
            supplies[_ri(rng, 0, n - 1)] = base
            gap = base if regime == 'exact' else base + _ri(rng, 1, 3)
        else:
            gap = 0 if regime == 'aligned' else _ri(rng, 1, 8)
            supplies = [_ri(rng, 0, cap) for cap in caps]
            if regime == 'zero':
                supplies[_ri(rng, 0, n - 1)] = 0
    else:
        gap, supplies = _ri(rng, 0, 8), []
    for room, supply in zip(fronts, supplies):
        _prefix(block, room, supply)
    for room in decoy + opened:
        _prefix(block, room, 0 if kind == 'dock' else _ri(rng, 0, len(room)))
    # Safely roughen the rear outline while preserving the intended closed rooms.
    target_holes = {frozenset(p) for p in fronts + decoy}
    for _ in range(_ri(rng, 0, 5)):
        trial = block.copy()
        row = _ri(rng, 1, h - 2)
        if trial[row, 0] != color:
            continue
        trial[row, 0] = 0
        wall = trial == color
        if len(_components(wall)) != 1 or len(_components(trial != 0)) != 1:
            continue
        if {frozenset(p) for p in _holes(wall)} == target_holes:
            block = trial
    return dict(block=block, color=color, active=fronts, decoy=decoy, opened=opened, gap=gap)


def _make_pair(rng, lesson, test, config, turn):
    desired = _ri(rng, 1, 5)
    rail_color = config['rail'] if config['shared'] else int(rng.choice(config['palette']))
    wall_pool = [c for c in config['palette'] if c != rail_color]
    common_color = config['wall'] if config['shared'] else int(rng.choice(wall_pool))
    objects = []
    for i in range(desired):
        remaining_slots = desired - i - 1
        occupied = sum(o['block'].shape[0] + 1 for o in objects)
        budget = min(18, 29 - occupied - 4 * remaining_slots)
        if i == 0:
            kind = lesson if lesson else 'multi' if test else str(rng.choice(
                ('active', 'active', 'solid', 'recessed', 'open')))
        else:
            kind = str(rng.choice(('active', 'active', 'solid', 'recessed', 'open')))
        if budget < (9 if kind == 'multi' else 7 if kind in ('dock', 'bottleneck') else 3):
            break
        color = common_color if config['shared'] or rng.random() < .5 else int(rng.choice(wall_pool))
        obj = _make_engine(rng, kind, budget, 4 if test else 2, color)
        objects.append(obj)
    _need(bool(objects), "construction needs an engine")
    base = sum(o['block'].shape[0] for o in objects) + len(objects) - 1
    height = _ri(rng, base + 1, 30)
    extras = rng.multinomial(height - base, np.full(len(objects) + 1, 1 / (len(objects) + 1)))
    # Guarantee a completely blank transverse line even when there is one engine.
    if not (extras[0] or extras[-1]) and len(objects) == 1:
        extras[0], extras[1] = 1, extras[1] - 1
    required_width = max(o['block'].shape[1] + o['gap'] for o in objects)
    rail = _ri(rng, required_width, min(27, required_width + 4))
    width = _ri(rng, rail + 2, 30)
    g = np.zeros((height, width), dtype=int)
    g[:, rail] = rail_color
    row = int(extras[0])
    for i, obj in enumerate(objects):
        h, w = obj['block'].shape
        obj['t'], obj['l'] = row, rail - w - obj['gap']
        g[row:row + h, obj['l']:obj['l'] + w] = obj['block']
        row += h + 1 + int(extras[i + 1])
    # Output uses construction coordinates; solve uses pixel-derived enclosure.
    out = _render(g, rail, objects)
    return {'input': np.rot90(g, turn).copy(), 'output': np.rot90(out, turn).copy()}


def _signature(grid):
    a = np.asarray(grid)
    return a.shape, a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(np.arange(3, 10))))
    config = dict(palette=palette, rail=palette[0], wall=palette[1], shared=rng.random() < .35)
    nturns = 1 if rng.random() < .35 else _ri(rng, 2, min(4, num_train))
    pool = list(map(int, rng.choice(4, nturns, replace=False)))
    lessons = ['dock', 'bottleneck'] + [None] * (num_train - 2)
    rng.shuffle(lessons)
    # Each orientation has an informative moving counter demonstration. Optional
    # examples may be silent, but cannot be the only evidence for a query direction.
    train_turns = [None] * num_train
    lesson_indices = [lessons.index('dock'), lessons.index('bottleneck')]
    other_indices = [i for i in range(num_train) if i not in lesson_indices]
    rng.shuffle(other_indices)
    for i, index in enumerate(lesson_indices + other_indices):
        train_turns[index] = pool[i % len(pool)]
        if 2 <= i < len(pool):
            lessons[index] = 'dock'
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            turn = train_turns[index] if split == 'train' else int(rng.choice(pool))
            lesson = lessons[index] if split == 'train' else None
            for _ in range(64):
                pair = _make_pair(rng, lesson, split == 'test', config, turn)
                key = _signature(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded distinct-input construction failed')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "split lists required")
    a, b = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= a <= 10 and 1 <= b <= 4 and a > b, "pair counts")
    seen, training, query, orientations = set(), [], [], {'train': set(), 'test': set()}
    taught_orientations = set()
    failed = {p: False for p in ('unlimited', 'all_closed', 'all_pockets', 'edge_only',
                                 'sum', 'max', 'limiter_only', 'always_signal', 'project_all')}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(inp.shape == out.shape, "shape preservation")
            key = _signature(inp)
            _need(key not in seen, "distinct inputs required")
            seen.add(key)
            g, turn, rail, color = _scene(inp)
            objects = _read(g, rail)
            orientations[split].add(turn)
            expected = np.rot90(_render(g, rail, objects), -turn)
            _need(np.array_equal(out, expected), "incorrect output")
            _need(np.array_equal(_render(g, rail, objects, unit_steps=True),
                                 _render(g, rail, objects)), "synchronous unit-step mismatch")
            (training if split == 'train' else query).extend(objects)
            if split == 'train':
                if any(_movement(o) > 0 and any(0 < _remaining(o['block'], p) < len(p)
                                              for p in o['active']) for o in objects):
                    taught_orientations.add(turn)
                for policy in failed:
                    failed[policy] |= not np.array_equal(_render(g, rail, objects, policy),
                                                         np.rot90(out, turn))
            else:
                _need(any(len(o['active']) >= 3 and _movement(o) > 0 for o in objects),
                      "query must compose at least three active reservoirs")
    _need(all(failed.values()), "training admits a documented shortcut: " + str(failed))
    _need(all(len(o['active']) <= 2 for o in training), "training reservoir count")
    _need(orientations['test'] <= orientations['train'], "untaught orientation")
    _need(orientations['test'] <= taught_orientations, "query orientation lacks moving counter lesson")
    _need(any(o['active'] and o['gap'] > 0 and _movement(o) == o['gap'] and
              min(_remaining(o['block'], p) for p in o['active']) > o['gap'] and
              any(_remaining(o['block'], p) == 0 for p in o['decoy']) and
              any(_remaining(o['block'], p) == 0 and
                  any(c == o['block'].shape[1] - 2 and o['block'][r, c + 1] == o['color']
                      for r, c in p) for p in o['opened'])
              for o in training), "missing docking/enclosure/extreme lesson")
    _need(any(len(o['active']) >= 2 and 0 < _movement(o) < o['gap'] and
              max(_remaining(o['block'], p) for p in o['active']) >= o['gap']
              for o in training), "missing synchronous bottleneck lesson")


def _features(puzzle):
    records = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            g, turn, rail, color = _scene(pair['input'])
            objects = _read(g, rail)
            records.append(dict(shape=g.shape, turn=turn, count=len(objects), band=g.shape[1] - rail - 1,
                                rail=rail, split=split, objects=objects))
    return records


def self_test():
    """Bounded semantic, boundary, corruption and 42-seed diversity checks."""
    puzzles, records, signatures, normalized = [], [], set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        puzzles.append(p)
        records.extend(_features(p))
        signatures.add(tuple(_signature(x['input']) for s in p.values() for x in s))
        for s in p.values():
            for pair in s:
                a = pair['input']
                mapping = {}
                normalized.add((a.shape, tuple(mapping.setdefault(int(c), len(mapping)) for c in a.flat)))
                _need(np.array_equal(solve(a.copy()), pair['output']), 'input-only solution')
    _need(len(signatures) == 42, 'cross-seed puzzle diversity')
    _need(len(normalized) >= 180, 'structure diversity beyond recoloring')
    _need(len({r['shape'] for r in records}) >= 45, 'canvas diversity')
    _need({r['turn'] for r in records} == set(range(4)), 'all travel directions')
    _need(len({r['count'] for r in records}) >= 4, 'engine count diversity')
    _need(any(r['band'] == 1 for r in records), 'one-cell projection band')
    objects = [o for r in records for o in r['objects']]
    _need({1, 2, 3, 4} <= {len(o['active']) for o in objects}, 'reservoir count diversity')
    _need(any(not o['active'] for o in objects), 'inactive engines')
    _need(any(o['gap'] == 0 and o['active'] for o in objects), 'already docked engine')
    _need(any(o['active'] and _movement(o) == 0 and o['gap'] for o in objects), 'empty bottleneck')
    _need(any(o['active'] and o['gap'] > 0 and
              min(_remaining(o['block'], p) for p in o['active']) == o['gap'] for o in objects),
          'exact-arrival tie')
    _need(any(len(p) >= 12 for o in objects for p in o['active']), 'larger reservoirs')
    _need(len({o['block'].shape for o in objects}) >= 45, 'engine geometry diversity')
    _need(len({o['color'] for o in objects}) == 7, 'wall palettes')
    _need(len({int(np.unique(np.rot90(p['input'], _scene(p['input'])[1])[:, _scene(p['input'])[2]])[0])
               for puz in puzzles for s in puz.values() for p in s}) == 7, 'rail palettes')
    # Every explicit count, including large requests, has the same lessons and rule.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(12000 + a * 10 + b, a, b)
            _need(len(p['train']) == a and len(p['test']) == b, 'count handling')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    _need(all(_signature(x[k]) == _signature(y[k])
              for s in ('train', 'test') for x, y in zip(puzzles[17][s], generate_puzzle(17)[s])
              for k in ('input', 'output')), 'determinism')
    _boundary_tests()
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('incorrect query label accepted')


def _boundary_tests():
    # Explicit small fixture with a stepped leading chamber, a recessed chamber,
    # an extreme-adjacent but open pocket, and exact-arrival signaling.
    g = np.zeros((9, 14), dtype=int)
    g[:, 10] = 8
    block = np.full((7, 7), 4, dtype=int)
    active = [(1, 5), (2, 5), (2, 4)]
    decoy = [(4, 2), (5, 2)]
    opened = [(4, 5), (5, 5), (6, 5)]
    for p in active:
        block[p] = 2
    for p in decoy + opened:
        block[p] = 1
    g[1:8, 0:7] = block  # gap 3; the open pocket reaches the bottom edge.
    expected = np.zeros_like(g)
    expected[:, 10] = 8
    moved = block.copy()
    for p in active:
        moved[p] = 1
    expected[1:8, 3:10] = moved
    expected[2:4, 11:] = 4
    for k in range(4):
        _need(np.array_equal(solve(np.rot90(g, k)), np.rot90(expected, k)), 'rotated exact-arrival fixture')
    # Empty active chamber, even beside full other chambers, must stop the engine.
    dry = g.copy()
    for r, c in active:
        dry[1 + r, c] = 1
    _need(np.array_equal(solve(dry), dry), 'empty reservoir locks motion and signaling')
    # At zero gap even this exhausted chamber emits; ignored reservoirs stay intact.
    aligned = np.zeros_like(g)
    aligned[:, 10] = 8
    aligned[1:8, 3:10] = dry[1:8, :7]
    answer = aligned.copy()
    answer[2:4, 11:] = 4
    _need(np.array_equal(solve(aligned), answer), 'already docked exhausted signaling')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds, all legal counts, boundary and ablation checks)')
