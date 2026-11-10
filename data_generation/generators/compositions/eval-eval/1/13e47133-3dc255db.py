"""Contour sieve: unary payloads move through matching virtual contour bands.

Only standard Python and NumPy are used. Labels are painted from sampled scene
data; solve reconstructs frames, keys, rooms and bands from the input alone.
"""
from collections import deque
import json
import numpy as np


DIR4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("grids must be integer arrays of size 1..30 with colors 0..9")
    return a.astype(int, copy=True)


def _components(cells, steps=DIR4):
    remaining = set(cells)
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, queue = {start}, [start]
        for r, c in queue:
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    group.add(p)
                    queue.append(p)
        groups.append(group)
    return groups


def _rooms(walls):
    masks = []
    for cells in _components(map(tuple, np.argwhere(~walls))):
        mask = np.zeros(walls.shape, dtype=bool)
        for p in cells:
            mask[p] = True
        masks.append(mask)
    return masks


def _depth(mask, diagonal=True):
    """Input reader uses simultaneous erosion, with the exterior outside."""
    active = mask.copy()
    depth = np.full(mask.shape, -1, dtype=int)
    h, w = mask.shape
    d = 0
    while active.any():
        pad = np.pad(active, 1)
        inner = active.copy()
        for dr, dc in DIR8 if diagonal else DIR4:
            inner &= pad[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        depth[active & ~inner] = d
        active = inner
        d += 1
    return depth


def _construction_depth(mask):
    """Label construction uses a boundary-seeded eight-neighbor BFS."""
    h, w = mask.shape
    result = np.full(mask.shape, -1, dtype=int)
    queue = deque()
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if any(not (0 <= r + dr < h and 0 <= c + dc < w
                    and mask[r + dr, c + dc]) for dr, dc in DIR8):
            result[r, c] = 0
            queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR8:
            nr, nc = r + dr, c + dc
            if (0 <= nr < h and 0 <= nc < w and mask[nr, nc]
                    and result[nr, nc] < 0):
                result[nr, nc] = result[r, c] + 1
                queue.append((nr, nc))
    return result


def _inside(mask, p):
    r, c = p
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[p])


def _key_paths(mask, depth):
    paths = []
    for r, c in np.argwhere(mask & (depth == 0)):
        r, c = int(r), int(c)
        for dr in (-1, 1):
            for dc in (-1, 1):
                if (_inside(mask, (r - dr, c)) or _inside(mask, (r, c - dc))
                        or not _inside(mask, (r + dr, c))
                        or not _inside(mask, (r, c + dc))):
                    continue
                cells = []
                for d in range(15):
                    p = r + d * dr, c + d * dc
                    if not _inside(mask, p) or depth[p] != d:
                        break
                    cells.append(p)
                if cells:
                    paths.append(tuple(cells))
    return paths


def _local(p, direction):
    # y increases from the tip towards the opening; x is transverse.
    r, c = p
    return ((r, c), (-c, r), (-r, -c), (c, -r))[direction]


def _world(p, direction):
    y, x = p
    return ((y, x), (x, -y), (-y, -x), (-x, y))[direction]


def _read_outline(cells, direction):
    local = {_local(p, direction) for p in cells}
    y0, y1 = min(y for y, x in local), max(y for y, x in local)
    if y1 - y0 < 3:
        return None
    rows = [sorted(x for y, x in local if y == level) for level in range(y0, y1 + 1)]
    if len(rows[0]) != 1 or any(not row for row in rows):
        return None
    tip_x = rows[0][0]
    lo = hi = tip_x
    split = False
    cavity = set()
    for offset, row in enumerate(rows[1:], 1):
        runs = []
        for x in row:
            if not runs or x != runs[-1][-1] + 1:
                runs.append([x])
            else:
                runs[-1].append(x)
        if row[0] > lo or row[-1] < hi:
            return None
        lo, hi = row[0], row[-1]
        if len(runs) == 1 and not split:
            continue
        if len(runs) != 2 or runs[1][0] - runs[0][-1] < 2:
            return None
        split = True
        cavity.update(_world((y0 + offset, x), direction)
                      for x in range(runs[0][-1] + 1, runs[1][0]))
    if not cavity or not lo < tip_x < hi:
        return None
    return _world((y0, tip_x), direction), cavity


def _axis(mask, tip, direction):
    dr, dc = DIR4[direction]
    r, c = tip
    result = []
    while True:
        r, c = r + dr, c + dc
        if not _inside(mask, (r, c)):
            return tuple(result)
        result.append((r, c))


def _eligible(region, obj, depth=None, cycle=True):
    depth = region['depth'] if depth is None else depth
    period = region['period']
    return tuple(p for p in _axis(region['mask'], obj['tip'], obj['direction'])
                 if (cycle or depth[p] < len(period))
                 and period[int(depth[p]) % len(period)] == obj['payload_color'])


def _read(grid):
    """Recover every convention from visible geometry and cells."""
    h, w = grid.shape
    border = np.concatenate((grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    if len(np.unique(border)) != 1 or h < 5 or w < 5:
        raise ValueError('a single-color enclosing wall is required')
    wall = int(border[0])
    walls = grid == wall
    counts = np.bincount(grid[~walls], minlength=10)
    background = int(np.argmax(counts))
    if counts[background] <= int((~walls).sum()) / 2:
        raise ValueError('neutral must occupy more than half the open cells')
    regions = [dict(mask=m, depth=_depth(m), objects=[]) for m in _rooms(walls)]
    found = []
    for color in sorted(set(map(int, np.unique(grid))) - {wall, background}):
        for cells in _components(map(tuple, np.argwhere(grid == color)), DIR8):
            if len(cells) < 5:
                continue
            for direction in range(4):
                geom = _read_outline(cells, direction)
                if geom is None:
                    continue
                tip, cavity = geom
                if any(not (0 <= p[0] < h and 0 <= p[1] < w) for p in cavity):
                    continue
                values = {int(grid[p]) for p in cavity} - {background}
                if len(values) != 1 or color in values or wall in values:
                    continue
                payload_color = next(iter(values))
                payload = {p for p in cavity if grid[p] == payload_color}
                rs, cs = zip(*cells)
                bbox = {(r, c) for r in range(min(rs), max(rs) + 1)
                        for c in range(min(cs), max(cs) + 1)}
                owners = [region for region in regions if all(region['mask'][p] for p in bbox)]
                if len(owners) != 1:
                    continue
                found.append(dict(frame=cells, frame_color=color, cavity=cavity,
                                  payload=payload, payload_color=payload_color,
                                  q=len(payload), tip=tip, direction=direction,
                                  bbox=bbox, region=owners[0]))
    # A monochrome payload cluster cannot own the enclosing frame.
    found = [o for o in found if not any(o is not other and o['frame'] <= other['cavity']
                                         for other in found)]
    if not found or len({frozenset(o['frame']) for o in found}) != len(found):
        raise ValueError('missing or directionally ambiguous emitter')
    if len({o['frame_color'] for o in found}) != 1:
        raise ValueError('all emitter outlines share one dedicated color')
    occupied = set()
    for obj in found:
        if not occupied.isdisjoint(obj['bbox']):
            raise ValueError('emitter boxes overlap')
        occupied.update(obj['bbox'])
        obj['region']['objects'].append(obj)
    frame_color = found[0]['frame_color']
    for region in regions:
        excluded = set().union(*(o['frame'] | o['payload'] for o in region['objects']))
        cues = {tuple(map(int, p)) for p in np.argwhere(region['mask'] & (grid != background))} - excluded
        if not cues:
            region.update(period=(background,), key=set())
        else:
            length = max(int(region['depth'][p]) for p in cues) + 1
            if length > 4:
                raise ValueError('key is too long or contains unrelated cells')
            readings = []
            for path in _key_paths(region['mask'], region['depth']):
                if len(path) >= length and cues <= set(path[:length]):
                    readings.append((tuple(int(grid[p]) for p in path[:length]), set(path[:length])))
            if not readings or len({x[0] for x in readings}) != 1:
                raise ValueError('ambiguous contour key')
            period, key = readings[0]
            if period[-1] == background or frame_color in period or wall in period:
                raise ValueError('key has invalid role colors')
            region.update(period=period, key=key)
        for obj in region['objects']:
            if obj['payload_color'] not in region['period']:
                raise ValueError('payload color is absent from its room key')
            axis = _axis(region['mask'], obj['tip'], obj['direction'])
            if any(grid[p] != background for p in axis):
                raise ValueError('scan corridor must be neutral in the input')
            obj['axis'] = axis
            obj['eligible'] = _eligible(region, obj)
    # No crossing scan axes: transport belongs unambiguously to each emitter.
    scanned = set()
    for obj in found:
        if not scanned.isdisjoint(obj['axis']):
            raise ValueError('scan corridors cross')
        scanned.update(obj['axis'])
    return dict(wall=wall, background=background, walls=walls, regions=regions, objects=found)


def _paint(grid, analysis, variant=None):
    out = grid.copy()
    for obj in analysis['objects']:
        for p in obj['payload']:
            out[p] = analysis['background']
        region = obj['region']
        positions = obj['eligible']
        q = obj['q']
        if variant == 'ungated':
            positions = obj['axis']
        elif variant == 'bbox':
            rr, cc = np.nonzero(region['mask'])
            r0, r1, c0, c1 = rr.min(), rr.max(), cc.min(), cc.max()
            rows, cols = np.indices(grid.shape)
            depth = np.minimum.reduce((rows - r0, r1 - rows, cols - c0, c1 - cols))
            positions = _eligible(region, obj, depth)
        elif variant == 'manhattan':
            positions = _eligible(region, obj, _depth(region['mask'], False))
        elif variant == 'no_cycle':
            positions = _eligible(region, obj, cycle=False)
        elif variant == 'components':
            q = len(_components(obj['payload'], DIR8))
        elif variant == 'constant':
            q = 2
        elif variant == 'unlimited':
            q = len(positions)
        for p in positions[:q]:
            out[p] = obj['payload_color']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    return _paint(grid, _read(grid))


def _outline(rng):
    """Unequal, independently delayed side walks and a variable open tail."""
    a, b = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    stem, tail = int(rng.integers(0, 3)), int(rng.integers(2, 6))
    spans = [(0, 0)] * (stem + 1)
    l = r = 0
    while l < a or r < b:
        dl = l < a and (r == b or rng.random() < .8)
        dr = r < b and (l == a or rng.random() < .8)
        if not dl and not dr:
            dl = l < a
            dr = not dl
        l, r = l + int(dl), r + int(dr)
        spans.append((-l, r))
    spans.extend([(-a, b)] * tail)
    thick = rng.random() < .3
    frame, cavity = set(), set()
    for y, (lo, hi) in enumerate(spans):
        t = 2 if thick and y <= len(spans) - tail - 1 else 1
        left, right = lo + t - 1, hi - t + 1
        if left >= right:
            frame.update((y, x + a) for x in range(lo, hi + 1))
        else:
            frame.update((y, x + a) for x in range(lo, left + 1))
            frame.update((y, x + a) for x in range(right, hi + 1))
            cavity.update((y, x + a) for x in range(left + 1, right))
    return dict(frame=frame, cavity=cavity, tip=(0, a))


def _orient(profile, direction):
    cells = profile['frame'] | profile['cavity'] | {profile['tip']}
    transformed = {p: _world(p, direction) for p in cells}
    mr, mc = min(p[0] for p in transformed.values()), min(p[1] for p in transformed.values())
    def f(p):
        q = transformed[p]
        return q[0] - mr, q[1] - mc
    frame = {f(p) for p in profile['frame']}
    return dict(frame=frame, cavity={f(p) for p in profile['cavity']}, tip=f(profile['tip']),
                height=max(p[0] for p in frame) + 1, width=max(p[1] for p in frame) + 1)


def _wall_layout(rng, anchor=False, testing=False):
    h = int(rng.integers(24 if anchor or testing else 14, 31))
    w = int(rng.integers(22 if anchor or testing else 14, 31))
    walls = np.zeros((h, w), dtype=bool)
    walls[[0, -1], :] = True
    walls[:, [0, -1]] = True
    rooms = [(1, h - 2, 1, w - 2)]
    target = 1 if anchor else int(rng.integers(1, 4 if testing else 3))
    while len(rooms) < target:
        choices = [(i, axis) for i, (r0, r1, c0, c1) in enumerate(rooms)
                   for axis, size in ((0, r1 - r0 + 1), (1, c1 - c0 + 1)) if size >= 17]
        if not choices:
            break
        i, axis = choices[int(rng.integers(len(choices)))]
        r0, r1, c0, c1 = rooms.pop(i)
        if axis == 0:
            cut = int(rng.integers(r0 + 7, r1 - 6))
            walls[cut, c0 - 1:c1 + 2] = True
            rooms.extend(((r0, cut - 1, c0, c1), (cut + 1, r1, c0, c1)))
        else:
            cut = int(rng.integers(c0 + 7, c1 - 6))
            walls[r0 - 1:r1 + 2, cut] = True
            rooms.extend(((r0, r1, c0, cut - 1), (r0, r1, cut + 1, c1)))
    # Wall-attached peninsulas make true concavities without a layout bank.
    for _ in range(int(rng.integers(1 if anchor else 0, 4 if testing else 3))):
        r0, r1, c0, c1 = rooms[int(rng.integers(len(rooms)))]
        side = int(rng.integers(4))
        if side in (0, 2):
            c = int(rng.integers(c0 + 2, c1 - 1))
            length = int(rng.integers(2, min(8, r1 - r0 - 2) + 1))
            start = r0 - 1 if side == 0 else r1 + 1
            sign = 1 if side == 0 else -1
            for k in range(length + 1):
                walls[start + sign * k, c] = True
            if testing and rng.random() < .5:
                end = start + sign * length
                walls[end, c:min(c1, c + int(rng.integers(2, 5))) + 1] = True
        else:
            r = int(rng.integers(r0 + 2, r1 - 1))
            length = int(rng.integers(2, min(8, c1 - c0 - 2) + 1))
            start = c1 + 1 if side == 1 else c0 - 1
            sign = -1 if side == 1 else 1
            for k in range(length + 1):
                walls[r, start + sign * k] = True
            if testing and rng.random() < .5:
                end = start + sign * length
                walls[r:min(r1, r + int(rng.integers(2, 5))) + 1, end] = True
    return walls


def _payload(rng, cavity, q, compact=False):
    ordered = sorted(cavity)
    selected = {ordered[int(rng.integers(len(ordered)))]}
    while len(selected) < q:
        remaining = sorted(cavity - selected)
        distances = [min(max(abs(p[0] - x[0]), abs(p[1] - x[1])) for x in selected)
                     for p in remaining]
        goal = min(distances) if compact else max(distances)
        choices = [p for p, d in zip(remaining, distances) if d == goal]
        selected.add(choices[int(rng.integers(len(choices)))])
    return selected


def _place(rng, regions, profile, occupied, scanned, frame_color, mode, anchor=False):
    candidates = []
    for direction in range(4):
        shape = _orient(profile, direction)
        h, w = regions[0]['mask'].shape
        if len(shape['cavity']) < (5 if mode == 'large' else 4):
            continue
        for r in range(h - shape['height'] + 1):
            for c in range(w - shape['width'] + 1):
                candidates.append((direction, shape, r, c))
    for idx in rng.permutation(len(candidates)):
        direction, shape, r, c = candidates[int(idx)]
        box = {(r + y, c + x) for y in range(shape['height']) for x in range(shape['width'])}
        if not box.isdisjoint(occupied | scanned):
            continue
        owner = next((z for z in regions if all(z['mask'][p] for p in box)), None)
        if owner is None or not box.isdisjoint(owner['key']):
            continue
        tip = r + shape['tip'][0], c + shape['tip'][1]
        axis = _axis(owner['mask'], tip, direction)
        if not axis or not set(axis).isdisjoint(occupied | scanned | owner['key']):
            continue
        colors = sorted(set(owner['period']) - {owner['background']})
        for color in rng.permutation(colors):
            obj = dict(tip=tip, direction=direction, payload_color=int(color), bbox=box,
                       frame_color=frame_color, region=owner, axis=axis,
                       frame={(r + y, c + x) for y, x in shape['frame']},
                       cavity={(r + y, c + x) for y, x in shape['cavity']})
            obj['eligible'] = _eligible(owner, obj)
            available = len(obj['eligible'])
            capacity = len(obj['cavity']) - 1
            if anchor:
                if available < 4 or tuple(axis[:3]) == obj['eligible'][:3]:
                    continue
                if not any(owner['depth'][p] >= len(owner['period']) for p in obj['eligible'][:3]):
                    continue
                # Compare spatial hypotheses on the actual probe, not elsewhere.
                rr, cc = np.nonzero(owner['mask'])
                rows, cols = np.indices(owner['mask'].shape)
                boxdepth = np.minimum.reduce((rows - rr.min(), rr.max() - rows,
                                              cols - cc.min(), cc.max() - cols))
                if _eligible(owner, obj, boxdepth)[:3] == obj['eligible'][:3]:
                    continue
                if _eligible(owner, obj, _depth(owner['mask'], False))[:3] == obj['eligible'][:3]:
                    continue
                q = 3
            elif mode == 'large':
                if available < 5:
                    continue
                q = int(rng.integers(4, min(8, capacity, available - 1) + 1))
                if obj['eligible'][:q] == tuple(axis[:q]):
                    continue
            elif mode == 'clip':
                if not 1 <= available < capacity:
                    continue
                q = int(rng.integers(available + 1, min(capacity, available + 4) + 1))
            else:
                if available < 2:
                    continue
                q = int(rng.integers(1, min(capacity, available - 1, 6) + 1))
            obj['q'] = q
            return obj
    return None


def _render_scene(rng, walls, regions, objects, background, wall, anchor=False):
    inp = np.full(walls.shape, background, dtype=int)
    inp[walls] = wall
    for region in regions:
        for p, color in zip(region['path'], region['period']):
            inp[p] = color
    for obj in objects:
        obj['payload'] = _payload(rng, obj['cavity'], obj['q'])
        for p in obj['frame']:
            inp[p] = obj['frame_color']
        for p in obj['payload']:
            inp[p] = obj['payload_color']
    # Output uses construction records, not the input reader or solve.
    out = inp.copy()
    for obj in objects:
        for p in obj['payload']:
            out[p] = background
        for p in obj['eligible'][:obj['q']]:
            out[p] = obj['payload_color']
    if not anchor:
        return {'input': inp, 'output': out}
    # The matched low/high demonstrations have the same geometry and key.
    obj = objects[0]
    low_in, low_out = inp.copy(), out.copy()
    for p in obj['payload']:
        low_in[p] = background
    low_payload = _payload(rng, obj['cavity'], 2, compact=True)
    for p in low_payload:
        low_in[p] = obj['payload_color']
    low_out[obj['eligible'][2]] = background
    return [{'input': low_in, 'output': low_out}, {'input': inp, 'output': out}]


def _scene(rng, mode='free', anchor=False, testing=False, simple_keys=False):
    for attempt in range(100):
        background, wall, frame_color = map(int, rng.choice(10, 3, replace=False))
        palette = [c for c in range(10) if c not in (background, wall, frame_color)]
        walls = _wall_layout(rng, anchor, testing)
        if np.any(walls[:-1, :-1] & walls[1:, :-1] & walls[:-1, 1:] & walls[1:, 1:]):
            continue
        regions = []
        for mask in _rooms(walls):
            depth = _construction_depth(mask)
            options = _key_paths(mask, depth)
            length = 3 if anchor or simple_keys else int(rng.integers(1, 5))
            options = [p for p in options if len(p) >= length]
            if not options:
                break
            path = options[int(rng.integers(len(options)))][:length]
            period = list(map(int, rng.choice(palette, length, replace=False)))
            if not anchor and not simple_keys and length >= 2 and rng.random() < .3:
                period[int(rng.integers(length - 1))] = background
            regions.append(dict(mask=mask, depth=depth, period=tuple(period), path=path,
                                key=set(path), background=background))
        else:
            profile = _outline(rng)
            primary = _place(rng, regions, profile, set(), set(), frame_color, mode, anchor)
            if primary is None:
                continue
            objects = [primary]
            target = 1 if anchor else int(rng.integers(2 if testing or mode == 'clip' else 1,
                                                     5 if testing else 4))
            for _ in range(target - 1):
                occupied = set().union(*(o['bbox'] for o in objects))
                # Keep monochrome outlines separate even diagonally.
                occupied = {(r + dr, c + dc) for r, c in occupied for dr, dc in ((0, 0),) + DIR8}
                scanned = set().union(*(set(o['axis']) for o in objects))
                extra_mode = 'clip' if testing and not simple_keys and rng.random() < .3 else 'free'
                obj = _place(rng, regions, _outline(rng), occupied, scanned, frame_color, extra_mode)
                if obj is not None:
                    objects.append(obj)
            if testing and len(objects) < 2:
                continue
            pair = _render_scene(rng, walls, regions, objects, background, wall, anchor)
            pairs = pair if anchor else [pair]
            try:
                for p in pairs:
                    a = _read(p['input'])
                    assert len(a['objects']) == len(objects)
                    assert np.array_equal(_paint(p['input'], a), p['output'])
                if anchor:
                    # Low and high witnesses jointly expose every central shortcut.
                    for variant in ('ungated', 'bbox', 'manhattan', 'no_cycle',
                                    'components', 'constant', 'unlimited'):
                        assert any(not np.array_equal(_paint(p['input'], _read(p['input']), variant), p['output'])
                                   for p in pairs)
            except (ValueError, AssertionError):
                continue
            return pair
    raise RuntimeError('bounded scene search exhausted')


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def _signature(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    anchors = _scene(rng, anchor=True)
    if rng.integers(2):
        anchors.reverse()
    puzzle = {'train': anchors, 'test': []}
    seen = {_signature(p['input']) for p in anchors}
    for split, count in (('train', num_train - 2), ('test', num_test)):
        for index in range(count):
            mode = ('clip' if index == 0 else 'free') if split == 'train' else (
                'large' if index == 0 else 'clip' if num_train >= 3 and rng.random() < .35 else 'free')
            for _ in range(10):
                pair = _scene(rng, mode, testing=split == 'test', simple_keys=num_train == 2)
                key = _signature(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('distinct input search exhausted')
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    n, m = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= n <= 10 and 1 <= m <= 4 and n > m
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = _signature(inp)
            assert key not in seen
            seen.add(key)
            a = _read(inp)
            assert 1 <= len(a['objects']) <= 4
            assert len(_components(map(tuple, np.argwhere(a['walls'])))) == 1
            walls_mask = a['walls']
            assert not np.any(walls_mask[:-1, :-1] & walls_mask[1:, :-1]
                              & walls_mask[:-1, 1:] & walls_mask[1:, 1:])
            assert not np.array_equal(inp, out)
            assert np.array_equal(_paint(inp, a), out)
            assert np.array_equal(inp[a['walls']], out[a['walls']])
            assert set(map(int, np.unique(out))) <= set(map(int, np.unique(inp)))
            for obj in a['objects']:
                assert 1 <= obj['q'] < len(obj['cavity'])
                assert all(out[p] == a['background'] for p in obj['payload'])
                assert all(out[p] == obj['frame_color'] for p in obj['frame'])
                selected = obj['eligible'][:obj['q']]
                assert sum(out[p] == obj['payload_color'] for p in obj['axis']) == len(selected)
                assert all(out[p] == a['background'] for p in set(obj['axis']) - set(selected))
                assert all(out[p] == inp[p] for p in obj['region']['key'])
            analyses[split].append((inp, out, a))
    full_counts, matched = set(), {}
    component_witness = False
    for inp, out, a in analyses['train']:
        base = inp.copy()
        for obj in a['objects']:
            for p in obj['payload']:
                base[p] = a['background']
            if len(obj['eligible']) > obj['q']:
                full_counts.add(obj['q'])
                matched.setdefault((_signature(base), obj['tip'], obj['direction']), set()).add(obj['q'])
                component_witness |= len(_components(obj['payload'], DIR8)) < obj['q']
    assert {2, 3} <= full_counts
    assert any({2, 3} <= values for values in matched.values())
    assert component_witness
    for variant in ('ungated', 'bbox', 'manhattan', 'no_cycle', 'components', 'constant', 'unlimited'):
        assert any(not np.array_equal(_paint(inp, a, variant), out)
                   for inp, out, a in analyses['train']), f'training permits shortcut {variant}'
    first_test = analyses['test'][0][2]
    assert len(first_test['objects']) >= 2
    assert any(o['q'] >= 4 and len(o['eligible']) > o['q'] for o in first_test['objects'])
    test_in, test_out, test_analysis = analyses['test'][0]
    assert not np.array_equal(_paint(test_in, test_analysis, 'ungated'), test_out)
    if n >= 3:
        assert any(0 < len(o['eligible']) < o['q'] for _, _, a in analyses['train'] for o in a['objects'])


def self_test():
    """Bounded structural and semantic evidence over 42 reproducible seeds."""
    shapes, walls, profiles, counts, directions, periods, roles, quantities = (set() for _ in range(8))
    fingerprints = set()
    neutral_keys, clipped, multi_room = 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(tuple(_signature(p['input']) for s in puzzle for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input']
                a = _read(inp)
                shapes.add(inp.shape)
                walls.add(_signature(a['walls'].astype(int)))
                roles.add((a['background'], a['wall'], a['objects'][0]['frame_color']))
                counts.add(len(a['objects']))
                multi_room += len(a['regions']) > 1
                for region in a['regions']:
                    assert np.array_equal(region['depth'], _construction_depth(region['mask']))
                    periods.add(len(region['period']))
                    neutral_keys += a['background'] in region['period'] and len(region['period']) > 1
                for obj in a['objects']:
                    directions.add(obj['direction'])
                    quantities.add(obj['q'])
                    clipped += len(obj['eligible']) < obj['q']
                    local = [_local(p, obj['direction']) for p in obj['frame']]
                    y0, x0 = min(y for y, x in local), min(x for y, x in local)
                    profiles.add(tuple(sorted((y - y0, x - x0) for y, x in local)))
    assert len(fingerprints) == 42
    assert len(shapes) >= 35 and len(walls) >= 90 and len(profiles) >= 60
    assert directions == {0, 1, 2, 3} and periods == {1, 2, 3, 4}
    assert {1, 2, 3} <= counts and len(roles) >= 60
    assert min(quantities) == 1 and max(quantities) >= 7
    assert neutral_keys >= 10 and clipped >= 42 and multi_room >= 35
    # Determinism, rotations and arbitrary palette permutations are input-only.
    first = generate_puzzle(71, 2, 1)
    again = generate_puzzle(71, 2, 1)
    for s in first:
        for p, q in zip(first[s], again[s]):
            assert np.array_equal(p['input'], q['input']) and np.array_equal(p['output'], q['output'])
            permutation = np.array([3, 8, 0, 6, 2, 9, 4, 1, 7, 5])
            for turns in range(4):
                inp = permutation[np.rot90(p['input'], turns)]
                out = permutation[np.rot90(p['output'], turns)]
                assert np.array_equal(solve(inp), out)
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            generate_puzzle(900 + n * 10 + m, n, m)
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (3, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    broken = {s: [{'input': p['input'].copy(), 'output': p['output'].copy()} for p in first[s]] for s in first}
    broken['test'][0]['output'][0, 0] = (broken['test'][0]['output'][0, 0] + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('validator accepted a changed wall')
    return dict(seeds=42, distinct_puzzles=len(fingerprints), shapes=len(shapes),
                wall_geometries=len(walls), frame_profiles=len(profiles),
                object_counts=sorted(counts), directions=sorted(directions),
                period_lengths=sorted(periods), role_assignments=len(roles),
                quantities=sorted(quantities), neutral_keys=neutral_keys,
                clipped_emitters=clipped, multi_room_pairs=multi_room)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
