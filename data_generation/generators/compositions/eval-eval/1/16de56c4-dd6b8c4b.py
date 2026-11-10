"""Cadenced harvesting on wall-deformed distance shells.

Only input-visible syntax affects solving. No file I/O or parent imports.
Saved diagnostics belong to the caller; this module keeps no trace or cache.
"""

from collections import deque

import numpy as np


_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    _require(_integer(num_train) and _integer(num_test), "Counts must be integers.")
    _require(2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test,
             "Require 2..10 train, 1..4 test, and more train than test.")
    return int(num_train), int(num_test)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape), "Invalid grid shape.")
    _require(np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9)),
             "Grid cells must be integer colors 0..9.")
    return grid


def _counter_mask(shape, pos):
    result = np.zeros(shape, dtype=bool)
    r, c = pos
    result[r:r+3, c:c+3] = True
    return result


def _distances(wall, pos):
    distance = np.full(wall.shape, -1, dtype=np.int32)
    r, c = pos
    queue = deque((r+i, c+j) for i in range(3) for j in range(3))
    for cell in queue:
        distance[cell] = 0
    h, w = wall.shape
    while queue:
        rr, cc = queue.popleft()
        for dr, dc in _STEPS:
            nr, nc = rr+dr, cc+dc
            if 0 <= nr < h and 0 <= nc < w and not wall[nr, nc] and distance[nr, nc] < 0:
                distance[nr, nc] = distance[rr, cc]+1
                queue.append((nr, nc))
    return distance


def _wave_distances(wall, pos):
    """Array wavefront audit, separate from the queue traversal."""
    seen = _counter_mask(wall.shape, pos)
    frontier = seen.copy()
    result = np.full(wall.shape, -1, dtype=np.int32)
    result[seen] = 0
    for depth in range(1, wall.size+1):
        spread = np.zeros(wall.shape, dtype=bool)
        spread[1:] |= frontier[:-1]
        spread[:-1] |= frontier[1:]
        spread[:, 1:] |= frontier[:, :-1]
        spread[:, :-1] |= frontier[:, 1:]
        frontier = spread & ~seen & ~wall
        if not frontier.any():
            break
        seen |= frontier
        result[frontier] = depth
    return result


def _manhattan(shape, pos):
    rr, cc = np.indices(shape)
    r, c = pos
    return np.maximum(r-rr, 0)+np.maximum(rr-r-2, 0)+np.maximum(c-cc, 0)+np.maximum(cc-c-2, 0)


def _parse(input_grid):
    grid = _grid(input_grid)
    wall_color = int(grid[0, 0])
    _require(all(np.all(edge == wall_color) for edge in (grid[0], grid[-1], grid[:, 0], grid[:, -1])),
             "Need a uniform closed outer wall frame.")
    colors, counts = np.unique(grid, return_counts=True)
    freq = {int(c): int(n) for c, n in zip(colors, counts)}
    candidates = []
    for slot, count in freq.items():
        if count != 8 or slot == wall_color:
            continue
        rr, cc = np.where(grid == slot)
        r, c = int(rr.min()), int(cc.min())
        if int(rr.max()) != r+2 or int(cc.max()) != c+2:
            continue
        patch = grid[r:r+3, c:c+3]
        reference = int(patch[1, 1])
        perimeter = np.ones((3, 3), dtype=bool)
        perimeter[1, 1] = False
        if np.all(patch[perimeter] == slot) and reference not in (slot, wall_color) and freq.get(reference) == 3:
            candidates.append(((r, c), slot, reference))
    _require(len(candidates) == 1, "Need one marked collector linked to two external beacons.")
    pos, slot, reference = candidates[0]
    counter = _counter_mask(grid.shape, pos)
    remaining = set(freq)-{wall_color, slot, reference}
    _require(bool(remaining), "No background.")
    background = max(remaining, key=lambda color: freq[color])
    _require(freq[background]*2 > grid.size-freq[wall_color], "Background must dominate non-wall cells.")
    remaining.remove(background)
    payloads = [c for c in remaining if freq[c] >= 2]
    _require(len(payloads) == 1, "Need one repeated payload hue outside the collector.")
    payload = payloads[0]
    extras = sorted(remaining-{payload})
    _require(len(extras) <= 2 and all(freq[c] == 1 for c in extras), "Extras must be singleton hues.")
    wall = grid == wall_color
    distance = _distances(wall, pos)
    beacons = [int(x) for x in np.flatnonzero((grid == reference) & ~counter)]
    radii = sorted(int(distance.flat[x]) for x in beacons)
    _require(len(radii) == 2 and 0 < radii[0] < radii[1], "Beacon distances must be distinct and positive.")
    a, b = radii
    period = b-a
    terminals, decoys = [], []
    for color in extras:
        cell = int(np.flatnonzero(grid == color)[0])
        radius = int(distance.flat[cell])
        if radius > 0 and (radius-a) % period == 0:
            _require(radius < a or radius > b, "On-phase singleton must lie beyond the ticks.")
            terminals.append((cell, color, radius))
        else:
            decoys.append(cell)
    _require(len(terminals) <= 1, "Need at most one terminal.")
    terminal = terminals[0] if terminals else None
    low, high = (1, int(distance.max())) if terminal is None else (min(a, terminal[2]), max(b, terminal[2]))
    return dict(grid=grid, pos=pos, slot=slot, reference=reference, background=background,
                payload=payload, wall_color=wall_color, wall=wall, counter=counter,
                distance=distance, beacons=beacons, a=a, b=b, period=period,
                terminal=terminal, decoys=decoys, low=low, high=high,
                paint=payload if terminal is None else terminal[1])


def _eligible(info, distance=None, ignore_phase=False, ignore_span=False):
    d = info['distance'] if distance is None else distance
    mask = (info['grid'] == info['payload']) & (d > 0)
    if not ignore_phase:
        mask &= (d-info['a']) % info['period'] == 0
    if not ignore_span:
        mask &= (d >= info['low']) & (d <= info['high'])
    return mask


def _select(eligible, distance):
    levels = np.sort(distance[eligible])
    if len(levels) > 9:
        _require(levels[8] < levels[9], "Ambiguous ninth/tenth distance tie.")
    if not len(levels):
        return np.zeros(eligible.shape, dtype=bool)
    return eligible & (distance <= levels[min(8, len(levels)-1)])


def _write_output(grid, pos, selected, background, paint):
    output = grid.copy()
    output[selected] = background
    r, c = pos
    for index in range(int(np.count_nonzero(selected))):
        output[r+index//3, c+index%3] = paint
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct every convention from the input alone."""
    info = _parse(input_grid)
    selected = _select(_eligible(info), info['distance'])
    return _write_output(info['grid'], info['pos'], selected, info['background'], info['paint'])


def _direct_prediction(info):
    """Audit a competing rule that uses direct radius for ruler and payload."""
    d = _manhattan(info['grid'].shape, info['pos'])
    radii = sorted(int(d.flat[x]) for x in info['beacons'])
    if radii[0] == radii[1]:
        return None  # This alternative cannot even calibrate the visible ruler.
    a, b = radii
    terminals = []
    extras = info['decoys']+([] if info['terminal'] is None else [info['terminal'][0]])
    for cell in extras:
        radius = int(d.flat[cell])
        if (radius-a) % (b-a) == 0:
            if a <= radius <= b:
                return None
            terminals.append((radius, int(info['grid'].flat[cell])))
    if len(terminals) > 1:
        return None
    alt = dict(info, distance=d, a=a, b=b, period=b-a)
    alt['low'], alt['high'] = (1, int(d.max())) if not terminals else (min(a, terminals[0][0]), max(b, terminals[0][0]))
    alt['paint'] = info['payload'] if not terminals else terminals[0][1]
    try:
        selected = _select(_eligible(alt), d)
    except ValueError:
        return None
    return _write_output(info['grid'], info['pos'], selected, info['background'], alt['paint'])


def _rectangle(wall, r0, c0, r1, c1):
    wall[r0, c0:c1+1] = True
    wall[r1, c0:c1+1] = True
    wall[r0:r1+1, c0] = True
    wall[r0:r1+1, c1] = True


def _pocket(wall, pos, rng):
    h, w = wall.shape
    counter = _counter_mask(wall.shape, pos)
    corners = [(1, 1), (1, w-5), (h-5, 1), (h-5, w-5)]
    for k in rng.permutation(4):
        r, c = corners[int(k)]
        if not counter[max(0, r-1):r+5, max(0, c-1):c+5].any():
            _rectangle(wall, r, c, r+3, c+3)
            wall[r+1:r+3, c+1:c+3] = False
            return


def _geometry(rng, witness, test, fallback=False):
    if witness == 'capacity':
        h, w = (22, 24) if fallback else (int(rng.integers(16, 25)), int(rng.integers(18, 25)))
        r = int(rng.integers(3, 6))
        c = int(rng.integers(8, w-5))
        pos = (r, c)
        wall = np.zeros((h, w), dtype=bool)
        wall[r-2, c-2:c+5] = True
        wall[r-2:h-3, c-2] = True
        wall[r-2:h-3, c+4] = True
        _rectangle(wall, 1, 1, 4, 4)
    else:
        low, high = (21, 30) if test else (13, 25)
        h, w = (24, 25) if fallback else (int(rng.integers(low, high+1)), int(rng.integers(low, high+1)))
        pos = (int(rng.integers(2, h-4)), int(rng.integers(2, w-4)))
        wall = np.zeros((h, w), dtype=bool)
        modes = ['open', 'segments', 'partitions', 'rings', 'corridors']
        weights = [.18, .25, .23, .18, .16] if not test else [.12, .19, .27, .16, .26]
        kind = 'open' if fallback else str(rng.choice(modes, p=weights))
        if kind == 'segments':
            for _ in range(int(rng.integers(2, 7))):
                rr, cc = int(rng.integers(2, h-3)), int(rng.integers(2, w-3))
                r1, c1 = int(rng.integers(rr+1, h-1)), int(rng.integers(cc+1, w-1))
                wall[rr, cc:c1+1] = True
                if rng.random() < .7:
                    wall[rr:r1+1, c1] = True
        elif kind == 'partitions':
            rooms = [(1, 1, h-2, w-2)]
            for _ in range(int(rng.integers(3, 8))):
                if not rooms:
                    break
                index = int(rng.integers(len(rooms)))
                r0, c0, r1, c1 = rooms.pop(index)
                horizontal = r1-r0 > 5 and (c1-c0 < 6 or rng.random() < .5)
                if horizontal:
                    rr = int(rng.integers(r0+2, r1-1))
                    wall[rr, c0:c1+1] = True
                    if rng.random() < .85:
                        cc = int(rng.integers(c0, c1+1))
                        wall[rr, cc:min(c1+1, cc+int(rng.integers(1, 4)))] = False
                    rooms.extend([(r0, c0, rr-1, c1), (rr+1, c0, r1, c1)])
                elif c1-c0 > 5:
                    cc = int(rng.integers(c0+2, c1-1))
                    wall[r0:r1+1, cc] = True
                    if rng.random() < .85:
                        rr = int(rng.integers(r0, r1+1))
                        wall[rr:min(r1+1, rr+int(rng.integers(1, 4))), cc] = False
                    rooms.extend([(r0, c0, r1, cc-1), (r0, cc+1, r1, c1)])
        elif kind == 'rings':
            r, c = pos
            _rectangle(wall, r-2, c-2, r+4, c+4)
            for _ in range(int(rng.integers(1, 5))):
                side = int(rng.integers(4))
                offset = int(rng.integers(-1, 4))
                if side < 2:
                    wall[r-2 if side == 0 else r+4, c+offset] = False
                else:
                    wall[r+offset, c-2 if side == 2 else c+4] = False
        elif kind == 'corridors':
            spacing = int(rng.integers(3, 6))
            for index, cc in enumerate(range(spacing, w-2, spacing)):
                if index % 2:
                    wall[3:h-1, cc] = True
                else:
                    wall[1:h-3, cc] = True
        if rng.random() < .45:
            _pocket(wall, pos, rng)
    wall[0] = wall[-1] = True
    wall[:, 0] = wall[:, -1] = True
    r, c = pos
    wall[r:r+3, c:c+3] = False
    counter = _counter_mask(wall.shape, pos)
    turns = int(rng.integers(4))
    wall, counter = np.rot90(wall, turns).copy(), np.rot90(counter, turns).copy()
    if rng.random() < .5:
        wall, counter = np.fliplr(wall).copy(), np.fliplr(counter).copy()
    rr, cc = np.where(counter)
    return wall, (int(rr.min()), int(cc.min()))


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _pick_cells(rng, mask, count, forced=(), style='dispersed'):
    available = [int(x) for x in np.flatnonzero(mask)]
    chosen = list(dict.fromkeys(int(x) for x in forced))
    if count > len(available) or len(chosen) > count or not set(chosen) <= set(available):
        return None
    used = set(chosen)
    pool = [x for x in available if x not in used]
    if len(chosen) == count:
        return chosen
    if style == 'dispersed':
        return chosen+[int(x) for x in rng.choice(pool, count-len(chosen), replace=False)]
    anchor = chosen[0] if chosen else int(_choose(rng, pool))
    ar, ac = divmod(anchor, mask.shape[1])
    cells = np.array(pool, dtype=int)
    rr, cc = cells//mask.shape[1], cells % mask.shape[1]
    if style == 'compact':
        scores = (rr-ar)**2+(cc-ac)**2
    elif rng.random() < .5:
        scores = np.abs(rr-ar)*mask.shape[1]+np.abs(cc-ac)
    else:
        scores = np.abs(cc-ac)*mask.shape[0]+np.abs(rr-ar)
    order = np.lexsort((rng.random(len(pool)), scores))
    return chosen+[int(x) for x in cells[order[:count-len(chosen)]]]


def _capacity_supply(rng, eligible, distance, metric, a, b, load, witness, style):
    radii = np.unique(distance[eligible])
    for q in rng.permutation(radii):
        near = eligible & (distance <= q)
        far = eligible & (distance > q)
        if near.sum() < 9 or far.sum() < load-9:
            continue
        chosen, retained = [], []
        if witness:
            lower = near & (distance < a)
            if not lower.any():
                continue
            xpool = near & (distance > b) & (metric > int(metric[far].min()))
            if not xpool.any():
                continue
            x = int(_choose(rng, np.flatnonzero(xpool)))
            ypool = far & (metric < int(metric.flat[x]))
            y = int(_choose(rng, np.flatnonzero(ypool)))
            chosen = [x, int(_choose(rng, np.flatnonzero(lower)))]
            retained = [y]
        selected = _pick_cells(rng, near, 9, chosen, style)
        excess = _pick_cells(rng, far, load-9, retained, style)
        return selected, excess
    return None


def _build_pair(rng, palette, witness, test, fixed_period=None):
    for attempt in range(48):
        wall, pos = _geometry(rng, witness, test, fallback=attempt >= 36)
        distance = _distances(wall, pos)
        counter = _counter_mask(wall.shape, pos)
        active = (distance > 0) & ~wall
        inactive = (distance < 0) & ~wall
        maxd = int(distance.max())
        if maxd < 8 or (witness == 'capacity' and not inactive.any()):
            continue
        metric = _manhattan(wall.shape, pos)
        for _ in range(28):
            terminal_mode = witness == 'terminal' or (witness is None and rng.random() < .55)
            period = fixed_period if fixed_period is not None else int(rng.integers(2 if witness == 'capacity' else 1, 7))
            if maxd <= 4*period+1:
                continue
            direction = int(rng.choice([-1, 1]))
            if witness == 'capacity':
                a = int(rng.integers(period+1, min(3*period+4, maxd//2)+1))
            elif terminal_mode:
                minimum = period+1 if direction > 0 else 2*period+1
                maximum = maxd-2*period if direction > 0 else maxd-period
                if maximum < minimum:
                    continue
                a = int(rng.integers(minimum, maximum+1))
            else:
                a = int(rng.integers(1, maxd-period+1))
            b = a+period
            t = None
            if terminal_mode:
                choices = list(range(b+period, maxd+1, period)) if direction > 0 else list(range(a-period, period, -period))
                if not choices:
                    continue
                t = int(_choose(rng, choices))
            low, high = (1, maxd) if t is None else (min(a, t), max(b, t))
            reserved = counter.copy()
            ticks = []
            for radius in ([a, b] if t is None else [a, b, t]):
                options = np.flatnonzero(active & (distance == radius) & ~reserved)
                if not len(options):
                    break
                cell = int(_choose(rng, options))
                ticks.append(cell)
                reserved.flat[cell] = True
            if len(ticks) != (2 if t is None else 3):
                continue
            phase = active & ((distance-a) % period == 0)
            eligible = phase & (distance >= low) & (distance <= high) & ~reserved
            style = str(rng.choice(['compact', 'line', 'dispersed']))
            load = int(rng.integers(10, 23)) if witness == 'capacity' else int(rng.choice([2, 4])) if witness == 'terminal' else (0 if rng.random() < .07 else int(rng.integers(1, 23)))
            if load > 9:
                plan = _capacity_supply(rng, eligible, distance, metric, a, b, load, witness == 'capacity', style)
                if plan is None:
                    continue
                selected, excess = plan
            else:
                selected = _pick_cells(rng, eligible, load, style=style)
                if selected is None:
                    continue
                excess = []
            payload = selected+excess
            occupied = reserved.copy()
            occupied.flat[payload] = True
            # A closer off-phase cell must actually invalidate phase-free collection.
            offphase = active & ~phase & ~occupied
            noise = []
            if witness == 'capacity':
                pool = offphase & (distance < int(distance.flat[selected].max()))
                if not pool.any():
                    continue
                noise.append(int(_choose(rng, np.flatnonzero(pool))))
            extra_count = min(int(rng.integers(0, 7)), int(offphase.sum()))
            off = _pick_cells(rng, offphase, max(extra_count, len(noise)), noise, style)
            if off is None:
                continue
            payload += off
            occupied.flat[off] = True
            if t is not None:
                outside_span = phase & ~((distance >= low) & (distance <= high)) & ~occupied
                force = []
                if witness == 'terminal':
                    before = outside_span & (distance < low)
                    if not before.any():
                        continue
                    force = [int(_choose(rng, np.flatnonzero(before)))]
                number = min(int(rng.integers(0, 7)), int(outside_span.sum()))
                outside = _pick_cells(rng, outside_span, max(number, len(force)), force, style)
                payload += outside
                occupied.flat[outside] = True
            minimum = 1 if witness == 'capacity' else 0
            unreachable_count = int(rng.integers(minimum, min(8, int(inactive.sum()))+1))
            unreachable = _pick_cells(rng, inactive & ~occupied, unreachable_count, style=style)
            if unreachable is None:
                continue
            payload += unreachable
            occupied.flat[unreachable] = True
            if len(payload) < 2:
                options = (offphase | inactive) & ~occupied
                additional = _pick_cells(rng, options, 2-len(payload), style=style)
                if additional is None:
                    continue
                payload += additional
                occupied.flat[additional] = True
            decoy = None
            if witness == 'capacity' or rng.random() < .55:
                options = (offphase | inactive) & ~occupied
                if not options.any():
                    if witness == 'capacity':
                        continue
                else:
                    decoy = int(_choose(rng, np.flatnonzero(options)))
            colors = palette if palette is not None else [int(x) for x in rng.choice(10, 7, replace=False)]
            bg, wc, slot, ref, pc, tc, dc = colors
            grid = np.full(wall.shape, bg, dtype=np.int8)
            grid[wall] = wc
            r, c = pos
            grid[r:r+3, c:c+3] = slot
            grid[r+1, c+1] = ref
            grid.flat[ticks[:2]] = ref
            if t is not None:
                grid.flat[ticks[2]] = tc
            if decoy is not None:
                grid.flat[decoy] = dc
            grid.flat[payload] = pc
            selected_mask = np.zeros(wall.shape, dtype=bool)
            selected_mask.flat[selected] = True
            output = _write_output(grid, pos, selected_mask, bg, pc if t is None else tc)
            try:
                info = _parse(grid)
                if not np.array_equal(_select(_eligible(info), info['distance']), selected_mask):
                    continue
            except ValueError:
                continue
            return {'input': grid, 'output': output}
    raise RuntimeError("Bounded scene construction exhausted.")


def _key(grid):
    grid = np.asarray(grid)
    return grid.shape, grid.tobytes() if grid.dtype == np.int8 else tuple(int(x) for x in grid.flat)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    _require(_integer(seed) and int(seed) >= 0, "Seed must be a nonnegative integer.")
    rng = np.random.default_rng(seed)
    palette = [int(x) for x in rng.choice(10, 7, replace=False)] if rng.random() < .25 else None
    capacity_period = int(rng.integers(2, 7))
    terminal_period = int(_choose(rng, [p for p in range(1, 7) if p != capacity_period]))
    requests = [('capacity', capacity_period), ('terminal', terminal_period)]+[(None, None)]*(num_train-2)
    requests = [requests[int(i)] for i in rng.permutation(num_train)]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, specs in [('train', requests), ('test', [(None, None)]*num_test)]:
        for witness, period in specs:
            for _ in range(12):
                pair = _build_pair(rng, palette, witness, split == 'test', period)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Bounded duplicate retries exhausted.")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info, selected):
    d, grid = info['distance'], info['grid']
    payload = grid == info['payload']
    phase = (d > 0) & ((d-info['a']) % info['period'] == 0)
    eligible = _eligible(info)
    retained = eligible & ~selected
    offphase = payload & (d > 0) & ~phase
    metric = _manhattan(grid.shape, info['pos'])
    path_reversal = bool(selected.any() and retained.any() and metric[selected].max() > metric[retained].min())
    before = payload & phase & (d < info['low'])
    phase_essential = bool(selected.any() and offphase.any() and
                           ((selected.sum() < 9 and np.any(offphase & (d >= info['low']) & (d <= info['high']))) or
                            np.any(offphase & (d < int(d[selected].max())) & (d >= info['low']) & (d <= info['high']))))
    # Measure whether some payload component is split at the capacity frontier.
    split_cluster = False
    remaining = set(int(x) for x in np.flatnonzero(payload))
    h, w = grid.shape
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component = [start]
        for cell in component:
            r, c = divmod(cell, w)
            for dr, dc in _STEPS:
                nr, nc = r+dr, c+dc
                n = nr*w+nc
                if 0 <= nr < h and 0 <= nc < w and n in remaining:
                    remaining.remove(n)
                    component.append(n)
        if any(selected.flat[x] for x in component) and any(retained.flat[x] for x in component):
            split_cluster = True
    return dict(shape=list(grid.shape), pos=list(info['pos']), period=info['period'], phase=info['a'] % info['period'],
                a=info['a'], b=info['b'], terminal=None if info['terminal'] is None else info['terminal'][2],
                direction=0 if info['terminal'] is None else int(np.sign(info['terminal'][2]-info['a'])),
                selected=int(selected.sum()), eligible=int(eligible.sum()), excess=int(retained.sum()),
                unreachable=int(np.count_nonzero(payload & (d < 0))), offphase=int(offphase.sum()),
                before_span=int(before.sum()), path_reversal=path_reversal, phase_essential=phase_essential,
                inward=bool(selected.any() and d[selected].min() < info['a']),
                outward=bool(selected.any() and d[selected].max() > info['b']), decoys=len(info['decoys']),
                max_distance=int(d.max()), max_detour=int((d-metric)[d >= 0].max()),
                split_cluster=split_cluster, walls=int(info['wall'].sum()),
                palette=[info[k] for k in ('background', 'wall_color', 'slot', 'reference', 'payload')])


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "Invalid puzzle schema.")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists.")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "Invalid pair schema.")
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape, "Dimensions must be preserved.")
            key = (a.shape, tuple(int(x) for x in a.flat))
            _require(key not in seen, "Duplicate input.")
            seen.add(key)
            info = _parse(a)
            wave = _wave_distances(info['wall'], info['pos'])
            _require(np.array_equal(wave, info['distance']), "Path audit disagrees.")
            eligible = _eligible(info, distance=wave)
            selected = _select(eligible, wave)
            expected = _write_output(a, info['pos'], selected, info['background'], info['paint'])
            _require(np.array_equal(b, expected), "Incorrect harvest or prefix.")
            changed_outside = (a != b) & ~info['counter']
            _require(np.array_equal(changed_outside, selected), "Unexpected external modification.")
            _require(int(selected.sum()) == min(9, int(eligible.sum())), "Capacity/count mismatch.")
            _require(np.all(b[info['wall']] == a[info['wall']]), "Walls changed.")
            features[split].append(_features(info, selected))
    train = features['train']
    _require(any(f['terminal'] is None and f['selected'] == 9 and f['excess'] > 0
                 and f['unreachable'] > 0 and f['path_reversal'] and f['phase_essential']
                 and f['inward'] and f['outward'] and f['decoys'] > 0 for f in train),
             "Missing boundary/capacity/path/cadence demonstration.")
    _require(any(f['terminal'] is not None and f['selected'] in (2, 4) and f['before_span'] > 0 for f in train),
             "Missing partial terminal stopping/recoloring demonstration.")
    _require(len({f['period'] for f in train}) > 1, "Training periods must vary.")
    return features


def _fixture(terminal=False, blocked=False, excess=False):
    """Hand-specified coordinates and labels, independent of scene construction."""
    grid = np.zeros((19, 24) if excess else (14, 16), dtype=np.int8)
    grid[0] = grid[-1] = 9
    grid[:, 0] = grid[:, -1] = 9
    pos = (2, 2)
    grid[2:5, 2:5] = 3
    grid[3, 3] = 2
    grid[3, 6] = grid[3, 8] = 2
    selected = [(1, 5), (5, 5), (3, 10)]
    retained = [(7, 3), (1, 2)]
    if excess:
        selected += [(6, 6), (7, 5), (8, 6), (9, 7), (10, 6), (5, 11)]
        retained += [(3, 14), (14, 4), (13, 7)]
    elif terminal or blocked:
        retained += [(3, 12)]
    else:
        selected += [(3, 12)]
    for cell in selected+retained:
        grid[cell] = 8
    grid[1, 4] = 7  # An off-phase singleton is never a terminal.
    if terminal:
        grid[10, 3] = 6  # Radius six bounds the [2,6] progression.
    if blocked:
        grid[:, 11] = 9
    expected = grid.copy()
    for cell in selected:
        expected[cell] = 0
    for i in range(len(selected)):
        expected[2+i//3, 2+i%3] = 6 if terminal else 8
    return grid, expected, selected


def self_test():
    for terminal, blocked, excess in [(False, False, False), (True, False, False),
                                      (False, True, False), (False, False, True)]:
        grid, expected, selected = _fixture(terminal, blocked, excess)
        _require(np.array_equal(solve(grid), expected), "Hand-specified regression failed.")
        mask = np.zeros(grid.shape, dtype=bool)
        for cell in selected:
            mask[cell] = True
        for turns in range(4):
            inp = np.rot90(grid, turns).copy()
            moved = np.rot90(mask, turns)
            info = _parse(inp)
            wanted = _write_output(inp, info['pos'], moved, 0, 6 if terminal else 8)
            _require(np.array_equal(solve(inp), wanted), "Displayed-order rotation regression failed.")
            _require(np.array_equal(solve(9-inp), 9-wanted), "Local color permutation failed.")
    two_decoys, expected, _ = _fixture()
    two_decoys[1, 3] = 6
    expected[1, 3] = 6
    _require(np.array_equal(solve(two_decoys), expected), "Two off-phase singleton hues were misread.")
    tied = np.ones((2, 5), dtype=bool)
    try:
        _select(tied, np.ones((2, 5), dtype=int))
    except ValueError:
        pass
    else:
        raise AssertionError("Capacity tie accepted.")
    # One puzzle at a time: no retained 42-seed grid dump or execution trace.
    signatures, structures, shapes, records = set(), set(), set(), []
    direct_rejected_puzzles = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        features = validate_generated_puzzle(puzzle)
        signature = []
        rejects_direct = False
        for split in ('train', 'test'):
            for pair, f in zip(puzzle[split], features[split]):
                grid = np.asarray(pair['input'])
                roles = {}
                normalized = tuple(roles.setdefault(int(x), len(roles)) for x in grid.flat)
                structures.add((grid.shape, normalized))
                shapes.add(grid.shape)
                signature.append(_key(grid))
                records.append(f)
                if split == 'train':
                    alternate = _direct_prediction(_parse(grid))
                    rejects_direct |= alternate is None or not np.array_equal(alternate, pair['output'])
        signatures.add(tuple(signature))
        _require(rejects_direct, "Training admits the complete direct-radius alternative.")
        direct_rejected_puzzles += int(rejects_direct)
    _require(len(signatures) == 42 and len(structures) >= 180 and len(shapes) >= 60, "Structural variation collapsed.")
    _require({f['period'] for f in records} == set(range(1, 7)), "Cadence variation collapsed.")
    _require({f['direction'] for f in records} == {-1, 0, 1}, "Terminal direction variation collapsed.")
    _require(max(max(f['shape']) for f in records) == 30 and max(f['max_distance'] for f in records) >= 40,
             "Size/path variation collapsed.")
    _require(any(f['split_cluster'] for f in records), "No capacity-split payload cluster.")
    _require(any(f['selected'] == 0 for f in records), "No zero-harvest case.")
    for role in range(5):
        _require(len({f['palette'][role] for f in records}) == 10, "Role colors collapsed.")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(700000+10*nt+ne, nt, ne)
            _require(len(p['train']) == nt and len(p['test']) == ne, "Explicit counts ignored.")
    for nt, ne in [(1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                   (3.0, 1), (4, False), (None, 1), (4, None)]:
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted.")
    first, second = generate_puzzle(12345, 2, 1), generate_puzzle(12345, 2, 1)
    _require(all(np.array_equal(a[k], b[k]) for s in first for a, b in zip(first[s], second[s])
                 for k in ('input', 'output')), "Whole-puzzle determinism failed.")
    broken = generate_puzzle(54321, 2, 1)
    broken['train'][0]['output'] = broken['train'][0]['input'].copy()
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("Corrupted label accepted.")
    return dict(seeds=42, pairs=len(records), normalized_inputs=len(structures), shapes=len(shapes),
                periods=sorted({f['period'] for f in records}),
                selected_counts=sorted({f['selected'] for f in records}),
                max_path_distance=max(f['max_distance'] for f in records),
                path_reversal_pairs=sum(f['path_reversal'] for f in records),
                capacity_split_clusters=sum(f['split_cluster'] for f in records),
                direct_radius_rejected_puzzles=direct_rejected_puzzles)
