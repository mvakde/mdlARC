"""Rim pulses: unary fuel in hollow arrows drives geodesic outer-rim growth.

All prediction conventions are visible in the input. No parent code is loaded.
"""
from collections import deque
import numpy as np


N4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _flood(mask, starts, steps=N4):
    h, w = mask.shape
    reached = np.zeros_like(mask, dtype=bool)
    queue = deque()
    for r, c in starts:
        if 0 <= r < h and 0 <= c < w and mask[r, c] and not reached[r, c]:
            reached[r, c] = True
            queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in steps:
            y, x = r + dr, c + dc
            if 0 <= y < h and 0 <= x < w and mask[y, x] and not reached[y, x]:
                reached[y, x] = True
                queue.append((y, x))
    return reached


def _components(mask, steps=N4):
    remaining = mask.copy()
    result = []
    while remaining.any():
        start = tuple(int(v) for v in np.argwhere(remaining)[0])
        part = _flood(remaining, [start], steps)
        result.append(part)
        remaining[part] = False
    return result


def _rim(region):
    """Four-connected outside complement; eight-neighbor inner contact."""
    h, w = region.shape
    allowed = np.pad(~region, 1, constant_values=True)
    outside = _flood(allowed, [(0, 0)])
    contact = np.zeros_like(region)
    for dr, dc in N8:
        contact |= outside[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return region & contact, outside[1:-1, 1:-1]


def _wave(rim, impact, quantity):
    """Round one paints the impact. Every subsequent round advances one edge."""
    h, w = rim.shape
    distances = {impact: 0}
    queue = deque([impact])
    while queue:
        r, c = queue.popleft()
        distance = distances[(r, c)]
        if distance + 1 >= quantity:
            continue
        for dr, dc in N8:
            y, x = r + dr, c + dc
            if 0 <= y < h and 0 <= x < w and rim[y, x] and (y, x) not in distances:
                distances[(y, x)] = distance + 1
                queue.append((y, x))
    return set(distances)


def _canonical(point, direction):
    r, c = point
    return ((r, c), (-c, r), (-r, -c), (c, -r))[direction]


def _world(point, direction):
    y, x = point
    return ((y, x), (x, -y), (-y, -x), (-x, y))[direction]


def _read_frame(cells, direction):
    local = {_canonical(p, direction) for p in cells}
    y0, y1 = min(y for y, _ in local), max(y for y, _ in local)
    if y1 - y0 < 2:
        return None
    rows = [sorted(x for y, x in local if y == level) for level in range(y0, y1 + 1)]
    if not rows[0] or len(rows[0]) != 1:
        return None
    tip_x = rows[0][0]
    left = right = tip_x
    split = False
    cavity = set()
    for level, row in enumerate(rows[1:], y0 + 1):
        if not split and row == [tip_x]:
            continue
        if len(row) != 2:
            return None
        lo, hi = row
        if lo > left or hi < right or hi - lo < 2:
            return None
        if left - lo > 1 or hi - right > 1:
            return None
        split = True
        left, right = lo, hi
        cavity.update(_world((level, x), direction) for x in range(lo + 1, hi))
    if not split or not left < tip_x < right:
        return None
    return _world((y0, tip_x), direction), cavity


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not np.issubdtype(grid.dtype, np.integer)
            or any(not 1 <= size <= 30 for size in grid.shape)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return grid


def _read(grid):
    """Recover palette, arrows, compartments, impacts, and fuel from input alone."""
    grid = _grid(grid)
    values, counts = np.unique(grid, return_counts=True)
    if len(values) != 4 or counts.max() * 2 <= grid.size:
        raise ValueError("four visible roles and a majority background are required")
    background = int(values[np.argmax(counts)])
    edge_colors = set(int(v) for v in np.concatenate(
        (grid[0], grid[-1], grid[:, 0], grid[:, -1]))) - {background}
    if len(edge_colors) != 1:
        raise ValueError("walls must be the unique nonbackground role on the canvas edge")
    wall = edge_colors.pop()
    free = grid != wall
    regions = _components(free)
    region_data = []
    for region in regions:
        rim, outside = _rim(region)
        region_data.append((region, rim, outside))
    candidates = []
    for color in (int(v) for v in values if v not in (background, wall)):
        for component in _components(grid == color, N8):
            cells = {tuple(int(v) for v in p) for p in np.argwhere(component)}
            if len(cells) < 5:
                continue
            for direction in range(4):
                found = _read_frame(cells, direction)
                if found is None:
                    continue
                tip, cavity = found
                if any(not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]) for r, c in cavity):
                    continue
                payload_colors = {int(grid[p]) for p in cavity} - {background}
                if len(payload_colors) != 1 or payload_colors & {color, wall}:
                    continue
                payload_color = payload_colors.pop()
                payload = {p for p in cavity if grid[p] == payload_color}
                if len(payload) >= len(cavity):
                    continue
                candidates.append(dict(frame=cells, cavity=cavity, tip=tip,
                    direction=direction, frame_color=color, payload_color=payload_color,
                    payload=payload, quantity=len(payload)))
    # An incidental pointed fuel cluster still belongs to its enclosing arrow.
    objects = [obj for obj in candidates if not any(
        obj is not other and obj['frame'] <= other['cavity'] for other in candidates)]
    if not objects or len({frozenset(obj['frame']) for obj in objects}) != len(objects):
        raise ValueError("missing or ambiguously oriented hollow arrows")
    frame_colors = {obj['frame_color'] for obj in objects}
    payload_colors = {obj['payload_color'] for obj in objects}
    if len(frame_colors) != 1 or len(payload_colors) != 1:
        raise ValueError("arrows in one scene must share frame and payload roles")
    occupied = set()
    for obj in objects:
        if occupied & (obj['frame'] | obj['cavity']):
            raise ValueError("overlapping arrows")
        occupied.update(obj['frame'] | obj['cavity'])
        dr, dc = N4[obj['direction']]
        tip = obj['tip']
        seed = (tip[0] + dr, tip[1] + dc)
        if not (0 <= seed[0] < grid.shape[0] and 0 <= seed[1] < grid.shape[1]) or grid[seed] != background:
            raise ValueError("an arrow needs a background cell immediately beyond its tip")
        region, rim, outside = next(data for data in region_data if data[0][seed])
        if any(not region[p] or rim[p] for p in obj['frame'] | obj['cavity']):
            raise ValueError("the complete arrow must lie inside the compartment, clear of its rim")
        r, c = seed
        holes_crossed = 0
        while 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
            if rim[r, c]:
                break
            if outside[r, c]:
                raise ValueError("launch ray left its compartment before reaching its rim")
            holes_crossed += int(grid[r, c] == wall)
            r, c = r + dr, c + dc
        else:
            raise ValueError("no exterior-rim impact")
        impact = (r, c)
        wave = _wave(rim, impact, obj['quantity'])
        obj.update(seed=seed, region=region, rim=rim, outside=outside,
                   impact=impact, wave=wave, holes_crossed=holes_crossed)
    frame_color = next(iter(frame_colors))
    all_frames = set().union(*(obj['frame'] for obj in objects))
    if all_frames != {tuple(int(v) for v in p) for p in np.argwhere(grid == frame_color)}:
        raise ValueError("every frame-colored cell must belong to a hollow arrow")
    return background, wall, sorted(objects, key=lambda obj: min(obj['frame'])), regions


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    background, _, objects, _ = _read(grid)
    output = grid.copy()
    for obj in objects:
        for cell in obj['payload']:
            output[cell] = background
    for obj in objects:
        for cell in obj['wave']:
            output[cell] = obj['payload_color']
    return output


def _profile(rng, anchor=False):
    """Thin, sometimes asymmetric open frames with variable tails and stems."""
    left = right = 0
    a = 2 if anchor else int(rng.integers(1, 4))
    b = 2 if anchor else int(rng.integers(1, 4))
    tail = int(rng.integers(3, 6)) if anchor else int(rng.integers(1, 5))
    stem = 0 if anchor else int(rng.integers(0, 2))
    sides = [(0, 0)] * (stem + 1)
    while left < a or right < b:
        dl = left < a and (left == right == 0 or anchor or right == b or rng.random() < .8)
        dr = right < b and (left == right == 0 or anchor or left == a or rng.random() < .8)
        if not dl and not dr:
            dl = left < a
            dr = not dl
        left, right = left + int(dl), right + int(dr)
        sides.append((-left, right))
    sides.extend([(-a, b)] * tail)
    frame, cavity = set(), set()
    for y, (lo, hi) in enumerate(sides):
        frame.update({(y, lo + a), (y, hi + a)})
        cavity.update((y, x + a) for x in range(lo + 1, hi))
    return dict(frame=frame, cavity=cavity, tip=(0, a))


def _orient(profile, direction):
    points = profile['frame'] | profile['cavity']
    world = {p: _world(p, direction) for p in points}
    y0 = min(y for y, _ in world.values())
    x0 = min(x for _, x in world.values())
    def convert(p):
        y, x = world[p]
        return y - y0, x - x0
    frame = {convert(p) for p in profile['frame']}
    return dict(frame=frame, cavity={convert(p) for p in profile['cavity']},
        tip=convert(profile['tip']), direction=direction,
        height=max(y for y, _ in frame) + 1, width=max(x for _, x in frame) + 1)


def _fuel(rng, cavity, quantity, pattern):
    cells = sorted(cavity)
    start = cells[int(rng.integers(len(cells)))]
    selected = {start}
    while len(selected) < quantity:
        choices = sorted(cavity - selected)
        if pattern == 'compact':
            choices = [p for p in choices if any(
                abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1 for q in selected)]
        elif pattern == 'scattered':
            distances = [min(max(abs(p[0] - q[0]), abs(p[1] - q[1])) for q in selected) for p in choices]
            choices = [p for p, d in zip(choices, distances) if d == max(distances)]
        selected.add(choices[int(rng.integers(len(choices)))])
    return selected


def _walls(rng, rooms, witness=False, fallback=False):
    h = 28 if fallback else int(rng.integers(23 if witness else 17, 31))
    w = 28 if fallback else int(rng.integers(26 if witness else 20, 31))
    wall = np.zeros((h, w), dtype=bool)
    if rooms >= 2:
        center = 9 if fallback else int(rng.integers(9, w - 9))
        columns = np.full(h, center, dtype=int)
        if not fallback and rng.random() < .7:
            cuts = sorted(set([0, h] + [int(v) for v in rng.integers(2, h - 2, size=int(rng.integers(1, 5)))]))
            for start, stop in zip(cuts, cuts[1:]):
                columns[start:stop] = int(np.clip(center + rng.integers(-2, 3), 8, w - 9))
        for r, col in enumerate(columns):
            wall[r, col] = True
            if r:
                wall[r, min(col, columns[r - 1]):max(col, columns[r - 1]) + 1] = True
        if rooms >= 3:
            cut = int(rng.integers(8, h - 8))
            wall[cut, int(columns[cut]):] = True
    # Edge intrusions are independent of dividers; none are closed rooms.
    if not fallback:
        for _ in range(int(rng.integers(1, 5))):
            side = int(rng.integers(4))
            depth, extent = int(rng.integers(1, 4)), int(rng.integers(1, 5))
            if side in (0, 2):
                c = int(rng.integers(1, w - extent))
                rows = slice(0, depth) if side == 0 else slice(h - depth, h)
                wall[rows, c:c + extent] = True
            else:
                r = int(rng.integers(1, h - extent))
                cols = slice(0, depth) if side == 3 else slice(w - depth, w)
                wall[r:r + extent, cols] = True
    else:
        wall[0, -3:] = True
        wall[:3, 5:7] = True
    regions = _components(~wall)
    if len(regions) != rooms or any(not (r[0].any() or r[-1].any() or r[:, 0].any() or r[:, -1].any()) for r in regions):
        return None
    # Islands have a two-cell free halo, so they neither partition nor pinch a rim.
    if fallback:
        wall[10:12, 4:6] = True
        wall[10:12, 18:20] = True
    else:
        for region in regions:
            for _ in range(int(rng.integers(1 if witness else 0, 4))):
                shape_h, shape_w = int(rng.integers(1, 4)), int(rng.integers(1, 4))
                choices = [(r, c) for r in range(3, h - shape_h - 2)
                           for c in range(3, w - shape_w - 2)
                           if np.all(region[r-2:r+shape_h+2, c-2:c+shape_w+2])
                           and not np.any(wall[r-2:r+shape_h+2, c-2:c+shape_w+2])]
                if choices:
                    r, c = choices[int(rng.integers(len(choices)))]
                    wall[r:r + shape_h, c:c + shape_w] = True
                    if rng.random() < .4 and shape_h > 1 and shape_w > 1:
                        wall[r + 1:r + shape_h, c + 1:c + shape_w] = False
    return wall


def _place(rng, wall, objects, shape, quantity, pattern, selection=None,
           require_hole=False, require_turn=False, require_topology=False):
    h, w = wall.shape
    sh, sw = shape['height'], shape['width']
    regions = _components(~wall)
    regions.sort(key=lambda region: int(region.sum()))
    if selection == 'small':
        regions = regions[:1]
    elif selection == 'large':
        regions = regions[-1:]
    used = set().union(*(obj['box'] for obj in objects)) if objects else set()
    prior_waves = set().union(*(obj['wave'] for obj in objects)) if objects else set()
    candidates = [(r, c) for r in range(1, h - sh) for c in range(1, w - sw)
                  if not np.any(wall[r-1:r+sh+1, c-1:c+sw+1])]
    rng.shuffle(candidates)
    for region in regions:
        rim, outside = _rim(region)
        for r, c in candidates:
            if not np.all(region[r:r+sh, c:c+sw]) or np.any(rim[r:r+sh, c:c+sw]):
                continue
            box = {(r+y, c+x) for y in range(-1, sh+1) for x in range(-1, sw+1)}
            if box & used:
                continue
            tip = (r + shape['tip'][0], c + shape['tip'][1])
            dr, dc = N4[shape['direction']]
            y, x = tip[0] + dr, tip[1] + dc
            holes = 0
            while 0 <= y < h and 0 <= x < w and not rim[y, x]:
                if outside[y, x]:
                    break
                holes += int(wall[y, x])
                y, x = y + dr, x + dc
            if not (0 <= y < h and 0 <= x < w and rim[y, x]):
                continue
            wave = _wave(rim, (y, x), quantity)
            turned = len({p[0] for p in wave}) > 1 and len({p[1] for p in wave}) > 1
            if (require_hole and not holes or require_turn and not turned
                    or len(wave) >= int(rim.sum()) or wave & prior_waves):
                continue
            if require_topology:
                padded = np.pad(outside, 1, constant_values=True)
                orth = np.zeros_like(rim)
                for dy, dx in N4:
                    orth |= padded[1+dy:1+dy+h, 1+dx:1+dx+w]
                if not any(not orth[p] for p in wave):
                    continue
                rr, cc = np.indices(rim.shape)
                geometric = {tuple(int(v) for v in p) for p in np.argwhere(
                    rim & (np.maximum(abs(rr-y), abs(cc-x)) < quantity))}
                if geometric == wave:
                    continue
            fuel = _fuel(rng, shape['cavity'], quantity, pattern)
            return dict(frame={(r+y, c+x) for y, x in shape['frame']},
                payload={(r+y, c+x) for y, x in fuel}, box=box, tip=tip,
                wave=wave, quantity=quantity, direction=shape['direction'])
    return None


def _scene(rng, palette, profile, low, high, witness, shift, split, rotation, reflect,
           fallback=False):
    rooms = 2 if witness else (3 if split == 'test' and shift == 'topology' else int(rng.integers(1, 3)))
    wall = _walls(rng, rooms, bool(witness), fallback)
    if wall is None:
        return None
    specs = []
    if witness == 1:
        specs = [(profile, 0, low, 'compact', 'small', True, False)]
    elif witness == 2:
        specs = [(profile, 0, low, 'scattered', 'large', False, False),
                 (profile, 0, high, 'compact', 'small', False, True)]
    else:
        count = int(rng.integers(1, 4))
        for index in range(count):
            shape = _profile(rng, anchor=split == 'test' and shift == 'reach' and index == 0)
            capacity = len(shape['cavity'])
            q = min(capacity - 1, int(rng.integers(1, high + 1)))
            if split == 'test' and shift == 'reach' and index == 0:
                q = min(capacity - 1, high + int(rng.integers(1, 4)))
            direction = int(rng.integers(1, 4)) if split == 'test' and shift == 'rotation' else (0 if shift == 'rotation' else int(rng.integers(4)))
            specs.append((shape, direction, q, str(rng.choice(['compact', 'scattered', 'random'])), None, False, index == 0 and split == 'test'))
    objects = []
    for index, (shape, direction, q, pattern, selection, hole, turn) in enumerate(specs):
        if q < 1:
            continue
        obj = _place(rng, wall, objects, _orient(shape, direction), q, pattern,
                     selection, hole, turn, require_topology=witness == 2 and index == 1)
        if obj is None:
            if witness or index == 0:
                return None
            continue
        objects.append(obj)
    background, structure, frame, payload = palette
    inp = np.full(wall.shape, background, dtype=np.int8)
    inp[wall] = structure
    out = inp.copy()
    for obj in objects:
        for cell in obj['frame']:
            inp[cell] = out[cell] = frame
        for cell in obj['payload']:
            inp[cell] = payload
        for cell in obj['wave']:
            out[cell] = payload
    # Exterior fuel marks teach local ownership; they are inert, not new sources.
    all_rims = np.zeros_like(wall)
    for region in _components(~wall):
        all_rims |= _rim(region)[0]
    used = set().union(*(obj['box'] for obj in objects))
    candidates = [(r, c) for r in range(2, wall.shape[0]-2) for c in range(2, wall.shape[1]-2)
                  if not wall[r, c] and not all_rims[r, c] and (r, c) not in used]
    rng.shuffle(candidates)
    for cell in candidates[:int(rng.integers(1 if witness else 0, 4))]:
        inp[cell] = out[cell] = payload
    inp, out = np.rot90(inp, rotation), np.rot90(out, rotation)
    if reflect:
        inp, out = np.fliplr(inp), np.fliplr(out)
    try:
        predicted = solve(inp)
    except ValueError:
        return None
    if not np.array_equal(predicted, out):
        return None
    return {'input': inp.copy(), 'output': out.copy()}


def _key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = tuple(int(v) for v in rng.choice(10, 4, replace=False))
    varying_palette = bool(rng.integers(2))
    profile = _profile(rng, anchor=True)
    low = int(rng.integers(2, 4))
    high = low + int(rng.integers(3, 5))
    shift = str(rng.choice(['reach', 'topology', 'rotation']))
    base_rotation, base_reflect = int(rng.integers(4)), bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            witness = index + 1 if split == 'train' and index < 2 else 0
            local_palette = palette if witness or not varying_palette else tuple(int(v) for v in rng.choice(10, 4, replace=False))
            rotation = base_rotation if witness or shift == 'rotation' else int(rng.integers(4))
            reflect = base_reflect if witness else bool(rng.integers(2))
            for attempt in range(80):
                pair = _scene(rng, local_palette, profile, low, high, witness,
                    shift, split, rotation, reflect, fallback=witness and attempt >= 35)
                if pair is not None and _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('bounded scene construction exhausted')
            puzzle[split].append(pair)
            seen.add(_key(pair['input']))
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(grid):
    bg, wall, objects, regions = _read(grid)
    result = []
    for obj in objects:
        local = {_canonical(p, obj['direction']) for p in obj['frame']}
        y0, x0 = min(y for y, _ in local), min(x for _, x in local)
        signature = tuple(sorted((y-y0, x-x0) for y, x in local))
        payload_mask = np.zeros_like(grid, dtype=bool)
        for p in obj['payload']:
            payload_mask[p] = True
        area = int(obj['region'].sum())
        areas = [int(region.sum()) for region in regions]
        h, w = grid.shape
        padded = np.pad(obj['outside'], 1, constant_values=True)
        orth = np.zeros_like(obj['rim'])
        for dr, dc in N4:
            orth |= padded[1+dr:1+dr+h, 1+dc:1+dc+w]
        rr, cc = np.indices(grid.shape)
        geometric = {tuple(int(v) for v in p) for p in np.argwhere(obj['rim'] & (
            np.maximum(abs(rr-obj['impact'][0]), abs(cc-obj['impact'][1])) < obj['quantity']))}
        result.append(dict(quantity=obj['quantity'], signature=signature,
            components=len(_components(payload_mask, N8)), direction=obj['direction'],
            palette=(bg, wall, obj['frame_color'], obj['payload_color']),
            selection='only' if len(areas) == 1 else 'small' if area == min(areas) and area < max(areas) else 'large' if area == max(areas) and area > min(areas) else 'middle',
            holes=obj['holes_crossed'], turned=len({p[0] for p in obj['wave']}) > 1 and len({p[1] for p in obj['wave']}) > 1,
            diagonal=any(not orth[p] for p in obj['wave']),
            geodesic=geometric != obj['wave'],
            rim_size=int(obj['rim'].sum()), wave_size=len(obj['wave']),
            island_cells=int(np.count_nonzero((grid == wall) & ~obj['outside'])),
            rooms=len(regions), arrows=len(objects), impact=obj['impact']))
    return result


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            bg, wall, objects, regions = _read(inp)
            assert 1 <= len(objects) <= 3
            assert all(region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any() for region in regions)
            assert np.array_equal(solve(inp), out) and not np.array_equal(inp, out)
            assert np.array_equal(out[inp == wall], inp[inp == wall])
            allowed_edits = set().union(*(obj['payload'] | obj['wave'] for obj in objects))
            assert {tuple(int(v) for v in p) for p in np.argwhere(inp != out)} <= allowed_edits
            for obj in objects:
                assert all(out[p] == obj['frame_color'] for p in obj['frame'])
                assert all(out[p] == bg for p in obj['payload'])
                assert all(out[p] == obj['payload_color'] for p in obj['wave'])
                assert all(obj['region'][p] and obj['rim'][p] for p in obj['wave'])
            if split == 'train':
                training.extend(_features(inp))
    # Pedagogical checks concern visible examples, not generation labels.
    assert len({f['quantity'] for f in training if f['wave_size'] < f['rim_size']}) >= 2
    matched = {}
    arrangements = {}
    for f in training:
        matched.setdefault((f['signature'], f['direction'], f['palette']), set()).add(f['quantity'])
        arrangements.setdefault((f['signature'], f['quantity'], f['palette']), set()).add(f['components'])
    assert any(len(qs) > 1 for qs in matched.values()), 'fuel count is confounded with geometry or direction'
    assert any(len(cs) > 1 for cs in arrangements.values()), 'no cell-count versus component-count witness'
    assert any(f['quantity'] > f['components'] for f in training)
    assert {'small', 'large'} <= {f['selection'] for f in training}
    assert any(f['holes'] for f in training), 'launch never distinguishes islands from outside walls'
    assert any(f['turned'] for f in training), 'training admits straight exports'
    assert any(f['diagonal'] for f in training), 'corner-inclusive exterior contact is not exposed'
    assert any(f['geodesic'] for f in training), 'training admits coordinate-distance growth'


def self_test() -> None:
    """Bounded checks: 42 seeds, semantic mutations, symmetries, and interfaces."""
    signatures, shapes, profiles, quantities, directions, palettes = set(), set(), set(), set(), set(), set()
    room_counts, arrow_counts, impacts, island_sizes = set(), set(), set(), set()
    sample = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple((_key(pair['input']), _key(pair['output'])) for split in ('train', 'test') for pair in puzzle[split]))
        local_shapes = set()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input']
                shapes.add(inp.shape)
                local_shapes.add(inp.shape)
                for f in _features(inp):
                    profiles.add(f['signature'])
                    quantities.add(f['quantity'])
                    directions.add(f['direction'])
                    palettes.add(f['palette'])
                    room_counts.add(f['rooms'])
                    arrow_counts.add(f['arrows'])
                    impacts.add((f['impact'][0] >= inp.shape[0]/2, f['impact'][1] >= inp.shape[1]/2))
                    island_sizes.add(f['island_cells'])
        assert len(local_shapes) >= 2
        if seed == 0:
            sample = puzzle
    assert len(signatures) == 42 and len(shapes) >= 40 and len(profiles) >= 25
    assert max(max(shape) for shape in shapes) == 30
    assert {1, 2, 3} <= room_counts and {1, 2, 3} <= arrow_counts
    assert directions == {0, 1, 2, 3} and len(impacts) == 4
    assert max(quantities) >= 8 and len(palettes) >= 30 and len(island_sizes) >= 10
    assert all({palette[i] for palette in palettes} == set(range(10)) for i in range(4))
    # D4 covariance, palette covariance, and non-mutation on actual examples.
    for pair in sample['train'] + sample['test']:
        before = pair['input'].copy()
        for rotation in range(4):
            for reflect in (False, True):
                inp = np.rot90(pair['input'], rotation)
                expected = np.rot90(pair['output'], rotation)
                if reflect:
                    inp, expected = np.fliplr(inp), np.fliplr(expected)
                assert np.array_equal(solve(inp), expected)
                assert np.array_equal(solve((inp + 3) % 10), (expected + 3) % 10)
        assert np.array_equal(before, pair['input'])
    # Explicit overlap and q=1 fixtures with hand-written target coordinates.
    overlap = np.zeros((22, 24), dtype=np.int8)
    overlap[:, 0] = 1
    overlap_target = overlap.copy()
    singleton = overlap.copy()
    singleton_target = overlap.copy()
    for r0, c0 in ((5, 4), (13, 8)):
        frame = {(0, 1)} | {(y, x) for y in range(1, 7) for x in (0, 2)}
        for y, x in frame:
            for grid in (overlap, overlap_target, singleton, singleton_target):
                grid[r0+y, c0+x] = 2
        for y in range(1, 5):
            overlap[r0+y, c0+1] = 3
        singleton[r0+1, c0+1] = 3
        singleton_target[0, c0+1] = 3
    overlap_target[0, 2:13] = 3
    assert np.array_equal(solve(overlap), overlap_target)
    assert np.array_equal(solve(singleton), singleton_target)
    # A large legal hollow can hold enough fuel to saturate a rectangular rim.
    saturated = np.zeros((30, 30), dtype=np.int8)
    saturated[:, 0] = 1
    saturation_target = saturated.copy()
    cavity = []
    for y in range(22):
        radius = min(y, 9)
        for x in {9-radius, 9+radius}:
            saturated[y+4, x+5] = saturation_target[y+4, x+5] = 2
        cavity.extend((y+4, x+5) for x in range(10-radius, 9+radius))
    for cell in cavity[:60]:
        saturated[cell] = 3
    saturation_target[0, 1:] = saturation_target[-1, 1:] = 3
    saturation_target[:, 1] = saturation_target[:, -1] = 3
    assert np.array_equal(solve(saturated), saturation_target)
    # Moving equal fuel inside a hollow changes no output; increasing it
    # changes the wave. Removing an intercepted island does not change impact.
    for pair in sample['train']:
        inp = pair['input']
        bg, wall, objects, _ = _read(inp)
        for obj in objects:
            mutated = inp.copy()
            for p in obj['payload']:
                mutated[p] = bg
            alternatives = sorted(obj['cavity'])[-obj['quantity']:]
            for p in alternatives:
                mutated[p] = obj['payload_color']
            assert np.array_equal(solve(mutated), pair['output'])
            empty = sorted(obj['cavity'] - obj['payload'])
            if empty and len(obj['wave']) < int(obj['rim'].sum()):
                more = inp.copy()
                more[empty[0]] = obj['payload_color']
                # Keep the hollow nonempty.
                if len(empty) > 1:
                    assert not np.array_equal(solve(more), pair['output'])
            if obj['holes_crossed']:
                removed = inp.copy()
                removed[(inp == wall) & ~obj['outside']] = bg
                new_objects = _read(removed)[2]
                counterpart = next(o for o in new_objects if o['tip'] == obj['tip'])
                assert counterpart['impact'] == obj['impact'] and counterpart['wave'] == obj['wave']
    for n in range(2, 11):
        for m in range(1, min(4, n-1) + 1):
            generate_puzzle(100+n*10+m, n, m)
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal example counts accepted')
    repeated = generate_puzzle(0)
    for split in ('train', 'test'):
        for a, b in zip(sample[split], repeated[split]):
            assert all(np.array_equal(a[k], b[k]) for k in ('input', 'output'))
    corrupt = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in sample[s]] for s in sample}
    corrupt['test'][0]['output'][0, 0] = (int(corrupt['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds and bounded semantic/interface checks')
