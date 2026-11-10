"""Paired diagonal deliveries: endpoint rays address enclosed blank chambers.

Only the visible input is used by solve. No parent module or data is imported.
"""
import json
import numpy as np


DIRECTIONS = ((-1, -1), (-1, 1), (1, 1), (1, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=False)


def _inside(p, shape):
    return all(0 <= x < n for x, n in zip(p, shape))


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        todo, cells = [start], {start}
        while todo:
            r, c = todo.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    cells.add(q)
                    todo.append(q)
        result.append(cells)
    return result


def _diamond(center, radius, interior=False):
    r, c = center
    return {(r + dr, c + dc) for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1)
            if (abs(dr) + abs(dc) < radius if interior
                else abs(dr) + abs(dc) == radius)}


def _cap(center, radius, direction):
    r, c = center
    vr, vc = direction
    return [(r + vr * (k + 1), c + vc * (radius - k + 1))
            for k in range(radius + 1)]


def _base(shape, background):
    return np.asarray(background, dtype=np.int8)[np.indices(shape).sum(0) % 2]


def _analyze(grid):
    """Infer palette, filled diamonds, caps and enclosed regions from pixels."""
    grid = _grid(grid)
    parity = np.indices(grid.shape).sum(0) % 2
    background = []
    for p in (0, 1):
        values = grid[parity == p]
        counts = np.bincount(values, minlength=10)
        color = int(counts.argmax())
        assert counts[color] > len(values) // 2, "checkerboard is not recoverable"
        background.append(color)
    assert background[0] != background[1]
    base = _base(grid.shape, background)
    donors = []
    # A colored chamber distinguishes a donor from a blank diamond-shaped room.
    for radius in (2, 3, 4):
        h, w = grid.shape
        if min(h, w) < 2 * radius + 1:
            continue
        shape = (h - 2 * radius, w - 2 * radius)
        offsets = sorted(_diamond((0, 0), radius))
        dr, dc = offsets[0]
        colors = grid[radius + dr:radius + dr + shape[0],
                      radius + dc:radius + dc + shape[1]]
        ok = ~np.isin(colors, background)
        for dr, dc in offsets[1:]:
            ok &= grid[radius + dr:radius + dr + shape[0],
                       radius + dc:radius + dc + shape[1]] == colors
        for rr, cc in np.argwhere(ok):
            center = (int(rr) + radius, int(cc) + radius)
            cells = _diamond(center, radius, True)
            payloads = {int(grid[q]) for q in cells}
            wall = int(colors[rr, cc])
            if len(payloads) != 1:
                continue
            payload = next(iter(payloads))
            if payload in (*background, wall):
                continue
            caps = []
            for direction in DIRECTIONS:
                cap = _cap(center, radius, direction)
                if all(_inside(q, grid.shape) and grid[q] == payload for q in cap):
                    caps.append(direction)
            if caps:
                donors.append(dict(center=center, radius=radius, color=payload,
                                   wall=wall, caps=caps, cells=cells))
    assert donors, "no filled diamond donors"
    walls = {d['wall'] for d in donors}
    assert len(walls) == 1, "framework color must be shared within an example"
    wall = next(iter(walls))
    h, w = grid.shape
    rooms = [p for p in _components(grid != wall)
             if all(r not in (0, h - 1) and c not in (0, w - 1) for r, c in p)]
    donor_rooms, blank_rooms = [], []
    for room in rooms:
        matches = [i for i, d in enumerate(donors) if room == d['cells']]
        if matches:
            assert len(matches) == 1
            donor_rooms.append(matches[0])
        else:
            assert all(grid[q] == base[q] for q in room), "receiver is not blank"
            blank_rooms.append(room)
    assert sorted(donor_rooms) == list(range(len(donors)))
    assert blank_rooms, "no blank receivers"
    expected_walls = set().union(*(_diamond(d['center'], d['radius']) for d in donors),
                                 *(_ring(room) for room in blank_rooms))
    assert expected_walls == set(map(tuple, np.argwhere(grid == wall))), 'unexplained framework'
    covered = set(map(tuple, np.argwhere(grid == wall)))
    for d in donors:
        assert not covered.intersection(d['cells'])
        covered.update(d['cells'])
        for direction in d['caps']:
            cap = _cap(d['center'], d['radius'], direction)
            assert not covered.intersection(cap), "overlapping donor caps"
            covered.update(cap)
    assert covered == set(map(tuple, np.argwhere(grid != base))), "unexplained foreground"
    # A wall contact can identify a room even at a square corner. Ambiguous
    # shared-wall contacts never count as a delivery.
    owners = {}
    for i, room in enumerate(blank_rooms):
        for r, c in room:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = r + dr, c + dc
                    if _inside(q, grid.shape) and grid[q] == wall:
                        owners.setdefault(q, set()).add(i)
    return dict(background=tuple(background), base=base, wall=wall,
                donors=donors, rooms=blank_rooms, owners=owners)


def _trace(grid, info):
    claims, deliveries, routes = {}, {}, []
    for donor_index, donor in enumerate(info['donors']):
        for direction in donor['caps']:
            cap = _cap(donor['center'], donor['radius'], direction)
            paths, contacts, owners = [], [], []
            for endpoint in (cap[0], cap[-1]):
                path, q = [], endpoint
                while True:
                    q = (q[0] + direction[0], q[1] + direction[1])
                    if not _inside(q, grid.shape):
                        contact = None
                        break
                    if grid[q] == info['wall']:
                        contact = q
                        break
                    assert grid[q] == info['base'][q], "a ray hits colored foreground"
                    assert q not in claims, "rays may not overlap"
                    claims[q] = donor['color']
                    path.append(q)
                paths.append(path)
                contacts.append(contact)
                candidates = info['owners'].get(contact, set())
                owners.append(next(iter(candidates)) if len(candidates) == 1 else None)
            destination = owners[0] if owners[0] is not None and owners[0] == owners[1] else None
            if destination is not None:
                color = donor['color']
                assert destination not in deliveries or deliveries[destination] == color, "conflicting deliveries"
                deliveries[destination] = color
            routes.append(dict(donor=donor_index, direction=direction, paths=paths,
                               contacts=contacts, owners=owners, destination=destination))
    assert claims and deliveries, "each generated input must show rays and transfer"
    return claims, deliveries, routes


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Continue cap ends; transfer a chamber color on a paired room contact."""
    grid = _grid(input_grid)
    info = _analyze(grid)
    claims, deliveries, _ = _trace(grid, info)
    out = grid.copy()
    for q, color in claims.items():
        out[q] = color
    for room, color in deliveries.items():
        for q in info['rooms'][room]:
            out[q] = color
    return out


def _ring(room):
    return {(r + dr, c + dc) for r, c in room
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)} - room


def _room_shape(rng, center, radius, irregular):
    # Independent target geometry: tiny pockets, rectangles, L shapes and steps.
    h, w = map(int, rng.integers(1, 7, size=2))
    if h + w < 2 * radius - 1:
        h, w = max(h, radius), max(w, radius)
    top, left = center[0] - h // 2, center[1] - w // 2
    room = {(top + r, left + c) for r in range(h) for c in range(w)}
    if irregular and h > 1 and w > 1:
        for _ in range(int(rng.integers(1, 4))):
            corner = (top + int(rng.choice([0, h - 1])),
                      left + int(rng.choice([0, w - 1])))
            candidate = room - {corner}
            if candidate and len(_components_from_cells(candidate)) == 1:
                room = candidate
        if rng.random() < .45:
            side = int(rng.integers(4))
            if side == 0:
                room.add((top - 1, left + w // 2))
            elif side == 1:
                room.add((top + h, left + w // 2))
            elif side == 2:
                room.add((top + h // 2, left - 1))
            else:
                room.add((top + h // 2, left + w))
    return room


def _components_from_cells(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    mask = np.zeros((max(r for r, c in cells) - r0 + 1,
                     max(c for r, c in cells) - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return _components(mask)


def _render(shape, background, wall, donors, rooms):
    grid = _base(shape, background)
    occupied = set()
    for d in donors:
        outline = _diamond(d['center'], d['radius'])
        inside = _diamond(d['center'], d['radius'], True)
        caps = {q for v in d['caps'] for q in _cap(d['center'], d['radius'], v)}
        footprint = outline | inside | caps
        assert all(_inside(q, shape) for q in footprint)
        assert not footprint & occupied
        occupied |= footprint
        for q in outline:
            grid[q] = wall
        for q in inside | caps:
            grid[q] = d['color']
    all_interiors, all_walls = set(), set()
    for room in rooms:
        ring = _ring(room)
        assert all(_inside(q, shape) for q in room | ring)
        assert not (room | ring) & occupied
        assert not room & (all_interiors | all_walls)
        assert not ring & all_interiors
        all_interiors |= room
        all_walls |= ring
    for q in all_walls:
        grid[q] = wall
    return grid


def _sample_scene(rng, radius, main_directions, multicolor, negative):
    for attempt in range(180):
        # Later retries broaden space without changing any semantic rule.
        count = 2 if multicolor else int(rng.choice([1, 1, 2, 3]))
        low = (2 * radius + 7 if count == 1 and len(main_directions) == 1 else
               max(16, 2 * radius + 10) if count == 1 else 22)
        shape = tuple(map(int, rng.integers(low, 31, size=2))) if attempt < 90 else (30, 30)
        palette = list(map(int, rng.permutation(10)))
        background, wall = tuple(palette[:2]), palette[2]
        payloads = palette[3:3 + count]
        if not multicolor and count > 1 and rng.random() < .35:
            payloads = [int(rng.choice(payloads)) for _ in range(count)]
        donors, rooms, requested = [], [], []
        try:
            for i in range(count):
                rad = radius if i == 0 else int(rng.integers(2, radius + 1))
                margin = rad + 2
                center = tuple(int(rng.integers(margin, n - margin)) for n in shape)
                caps = main_directions if i == 0 else [DIRECTIONS[int(rng.integers(4))]]
                donor = dict(center=center, radius=rad, color=payloads[i], caps=caps)
                donors.append(donor)
                for j, direction in enumerate(caps):
                    mode = ('one' if negative and i == 0 and j == 1 else
                            'success' if i > 0 or j == 0 else str(rng.choice(['success', 'one', 'split', 'miss'])))
                    if mode == 'one' and rng.random() < .5:
                        mode = 'split'
                    distance = int(rng.integers(rad + 2, 11))
                    vr, vc = direction
                    if mode == 'success':
                        center_t = (center[0] + vr * distance, center[1] + vc * distance)
                        room = _room_shape(rng, center_t, rad, True)
                        rooms.append(room)
                    elif mode in ('one', 'split'):
                        # Put one tiny pocket on one rail, optionally a second
                        # pocket on the other. The pockets can share framework.
                        centers = [(center[0] + vr * distance, center[1] + vc * (distance + rad))]
                        if mode == 'split':
                            centers.append((center[0] + vr * (distance + rad), center[1] + vc * distance))
                        rooms.extend([{q} for q in centers])
                    requested.append((i, direction, mode))
            grid = _render(shape, background, wall, donors, rooms)
            info = _analyze(grid)
            claims, deliveries, routes = _trace(grid, info)
            # Find donors by visible center, not extraction order.
            for i, direction, mode in requested:
                routes_here = [v for v in routes if info['donors'][v['donor']]['center'] == donors[i]['center']
                               and v['direction'] == direction]
                assert len(routes_here) == 1
                route = routes_here[0]
                if mode == 'success':
                    assert route['destination'] is not None
                elif mode == 'one':
                    assert route['destination'] is None and sum(x is not None for x in route['owners']) == 1
                elif mode == 'split':
                    assert all(x is not None for x in route['owners']) and route['owners'][0] != route['owners'][1]
                else:
                    assert route['destination'] is None
                assert min(map(len, route['paths'])) >= 1
            if multicolor:
                assert len(set(deliveries.values())) >= 2
            if negative:
                assert len(deliveries) < len(info['rooms'])
            return dict(input=grid, output=solve(grid))
        except AssertionError:
            continue
    return _fallback(rng, radius, main_directions, multicolor, negative)


def _fallback(rng, radius, main_directions, multicolor, negative):
    """A fitting radial layout; used only after bounded free placement fails."""
    palette = list(map(int, rng.permutation(10)))
    donors = [dict(center=(14, 14), radius=radius, color=palette[3], caps=main_directions)]
    rooms = []
    for j, (vr, vc) in enumerate(main_directions):
        if negative and j == 1:
            rooms.append({(14 + vr * 8, 14 + vc * (8 + radius))})
            rooms.append({(14 + vr * (8 + radius), 14 + vc * 8)})
        else:
            center = (14 + vr * 8, 14 + vc * 8)
            rooms.append({(center[0] + dr, center[1] + dc)
                          for dr in range(-2, 3) for dc in range(-2, 3)})
    if multicolor:
        # Witness zero uses NW/SE, leaving this corner route available.
        assert main_directions == [DIRECTIONS[0], DIRECTIONS[2]]
        donors.append(dict(center=(5, 24), radius=2, color=palette[4], caps=[DIRECTIONS[0]]))
        rooms.append({(1, 20)})
    grid = _render((30, 30), tuple(palette[:2]), palette[2], donors, rooms)
    return dict(input=grid, output=solve(grid))


def _counts(rng, num_train, num_test):
    for value, name, low, high in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError(f'{name} must be an integer in {low}..{high}, or None')
    if num_train is None:
        num_train = int(rng.integers(max(2, (num_test or 1) + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    num_train, num_test = _counts(rng, num_train, num_test)
    radius = int(rng.integers(2, 5))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            if split == 'train':
                axis = index % 2
                directions = [DIRECTIONS[axis], DIRECTIONS[axis + 2]]
                if index >= 2 and rng.random() < .35:
                    directions = [DIRECTIONS[int(rng.integers(4))]]
            else:
                start = int(rng.integers(4))
                length = int(rng.choice([2, 2, 2, 3, 4]))
                directions = [DIRECTIONS[(start + k) % 4] for k in range(length)]
            multicolor = split == 'train' and index == 0
            negative = multicolor or (len(directions) > 1 and rng.random() < .5)
            for _ in range(24):
                pair = _sample_scene(rng, radius, directions, multicolor, negative)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise AssertionError('unique-input construction exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    return puzzle


def _measure_pair(pair):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    info = _analyze(inp)
    claims, deliveries, routes = _trace(inp, info)
    assert inp.shape == out.shape
    foreground = inp != info['base']
    assert np.array_equal(inp[foreground], out[foreground]), 'input framework/caps/donors changed'
    changed = set(map(tuple, np.argwhere(inp != out)))
    filled = set().union(*(info['rooms'][i] for i in deliveries))
    assert not filled.intersection(claims), 'ray enters a closed room'
    assert changed == set(claims) | filled, 'change outside ray/receiver support'
    for q, color in claims.items():
        assert out[q] == color
    for i, room in enumerate(info['rooms']):
        if i in deliveries:
            assert all(out[q] == deliveries[i] for q in room), 'nonuniform transfer'
        else:
            assert all(out[q] == inp[q] for q in room), 'unaddressed room filled'
    assert np.array_equal(out, solve(inp))
    failed = [v for v in routes if v['destination'] is None]
    area_signs = {int(np.sign(len(info['rooms'][v['destination']])
                             - len(info['donors'][v['donor']]['cells'])))
                  for v in routes if v['destination'] is not None}
    # Evidence against whole-cap continuation: a middle lane remains untouched
    # outside receivers, at a location where a whole-cap ray would paint it.
    hollow = 0
    for route in routes:
        d = info['donors'][route['donor']]
        vr, vc = route['direction']
        for r, c in _cap(d['center'], d['radius'], route['direction'])[1:-1]:
            for t in (1, 2):
                q = r + vr * t, c + vc * t
                if (_inside(q, inp.shape) and inp[q] == info['base'][q]
                        and out[q] == inp[q] and q not in filled):
                    hollow += 1
    return dict(donors=len(info['donors']), rooms=len(info['rooms']), delivered=len(deliveries),
                colors=len(set(deliveries.values())), radius=max(d['radius'] for d in info['donors']),
                directions={v['direction'] for v in routes}, hollow=hollow,
                failures=len(failed), split_failures=sum(all(x is not None for x in v['owners'])
                    and v['owners'][0] != v['owners'][1] for v in failed),
                one_failures=sum(sum(x is not None for x in v['owners']) == 1 for v in failed),
                lengths=[len(p) for v in routes for p in v['paths']],
                filled=len(filled), rays=len(claims),
                area_signs=area_signs,
                cap_counts=[len(d['caps']) for d in info['donors']],
                irregular=sum(len(room) != (max(r for r, c in room) - min(r for r, c in room) + 1)
                              * (max(c for r, c in room) - min(c for r, c in room) + 1)
                              for room in info['rooms']),
                structure=(inp.shape, (inp == info['wall']).tobytes(),
                           tuple((d['center'], d['radius'], tuple(d['caps'])) for d in info['donors'])),
                background=info['background'], wall=info['wall'])


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, metrics = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            metrics[split].append(_measure_pair(dict(input=inp, output=out)))
    training = metrics['train']
    assert set().union(*(m['directions'] for m in training)) == set(DIRECTIONS)
    assert any(m['colors'] >= 2 for m in training), 'no object-local color evidence'
    assert any(m['failures'] and m['delivered'] < m['rooms'] for m in training), 'no paired-contact counterexample'
    assert sum(m['hollow'] for m in training) >= 2, 'no endpoint-only witness'
    assert min(m['filled'] for m in training) > 0, 'parent A unused'
    assert min(m['rays'] for m in training) > 0, 'parent B unused'
    assert max(m['radius'] for m in training) >= max(m['radius'] for m in metrics['test'])
    return None


def self_test():
    # A small hand-specified case separates paired delivery from any-hit filling,
    # whole-cap propagation and unlimited rays. The two NW receiver pockets
    # share a framework corner, but remain distinct enclosed compartments.
    donor = dict(center=(9, 9), radius=2, color=3,
                 caps=[DIRECTIONS[0], DIRECTIONS[2]])
    recipient = {(r, c) for r in range(16, 19) for c in range(16, 19)}
    fixture = _render((23, 23), (0, 1), 2, [donor], [{(3, 1)}, {(1, 3)}, recipient])
    expected = fixture.copy()
    for q in recipient:
        expected[q] = 3
    for q in ((7, 5), (6, 4), (5, 3), (5, 7), (4, 6), (3, 5),
              (11, 13), (12, 14), (13, 15), (14, 16),
              (13, 11), (14, 12), (15, 13), (16, 14)):
        expected[q] = 3
    assert np.array_equal(solve(fixture), expected)
    assert expected[3, 1] == fixture[3, 1] and expected[1, 3] == fixture[1, 3]
    assert expected[12, 12] == fixture[12, 12]  # the cap's middle lane
    assert expected[18, 20] == fixture[18, 20]  # beyond the first wall
    assert all(expected[q] == fixture[q] for q in _ring(recipient))
    for k in range(4):
        assert np.array_equal(solve(np.rot90(fixture, k)), np.rot90(expected, k))
    assert np.array_equal(solve(fixture[:, ::-1]), expected[:, ::-1])
    edge_donor = dict(center=(3, 3), radius=2, color=3,
                      caps=[DIRECTIONS[0], DIRECTIONS[2]])
    edge = _render((16, 16), (0, 1), 2, [edge_donor], [{(11, 11)}])
    edge_info = _analyze(edge)
    edge_routes = _trace(edge, edge_info)[2]
    outward = next(v for v in edge_routes if v['direction'] == DIRECTIONS[0])
    assert outward['paths'] == [[], []] and outward['contacts'] == [None, None]
    assert solve(edge)[0, 0] == edge[0, 0] == 0
    shared_donors = [dict(center=(3, 3), radius=2, color=3, caps=[DIRECTIONS[2]]),
                     dict(center=(19, 19), radius=2, color=3, caps=[DIRECTIONS[0]])]
    shared = _render((23, 23), (0, 1), 2, shared_donors, [{(11, 11)}])
    shared_info = _analyze(shared)
    _, shared_deliveries, shared_routes = _trace(shared, shared_info)
    assert len(shared_info['donors']) == 2 and len(shared_deliveries) == 1
    assert all(v['destination'] is not None for v in shared_routes)
    assert solve(shared)[11, 11] == 3
    shared_donors[1]['color'] = 4
    conflict = _render((23, 23), (0, 1), 2, shared_donors, [{(11, 11)}])
    try:
        solve(conflict)
    except AssertionError:
        pass
    else:
        raise AssertionError('conflicting receiver colors accepted')
    metrics, structures, palettes, shapes = [], set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                m = _measure_pair(pair)
                metrics.append(m)
                structures.add(m['structure'])
                palettes.add((m['background'], m['wall']))
                shapes.add(pair['input'].shape)
    assert len(structures) >= 80 and len(palettes) >= 50 and len(shapes) >= 20
    assert {m['radius'] for m in metrics} == {2, 3, 4}
    assert {m['donors'] for m in metrics} == {1, 2, 3}
    assert max(m['rooms'] for m in metrics) >= 4
    assert {n for m in metrics for n in m['cap_counts']} == {1, 2, 3, 4}
    assert set().union(*(m['area_signs'] for m in metrics)) == {-1, 0, 1}
    assert sum(m['irregular'] for m in metrics) >= 10
    assert sum(m['split_failures'] for m in metrics) >= 5
    assert sum(m['one_failures'] for m in metrics) >= 5
    assert max(max(m['lengths']) for m in metrics) >= 12
    assert any(m['filled'] > m['rays'] for m in metrics)
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(701 + a + b, a, b))
    for a, b in ((None, None), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(801, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p, q = generate_puzzle(928, 2, 1), generate_puzzle(928, 2, 1)
    assert all(_key(x[k]) == _key(y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x)
    # Palette equivariance is a semantic check of input-derived roles.
    permutation = np.random.default_rng(929).permutation(10)
    for pair in p['train'] + p['test']:
        assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
    bad = {s: [{k: g.copy() for k, g in pair.items()} for pair in p[s]] for s in p}
    pair = bad['test'][0]
    point = tuple(np.argwhere(pair['input'] != pair['output'])[0])
    pair['output'][point] = pair['input'][point]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('missing transfer/ray cell was accepted')
    return {'seeds': 42, 'pairs': len(metrics), 'structures': len(structures),
            'palettes': len(palettes), 'shapes': len(shapes)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
