"""Concentric collector: path-ordered colored cells become nested ring slots.

Only input-visible geometry and colors determine the transformation.  The
outer border identifies walls; the four inner corner cells identify background.
A unique even square, with a 2x2 center marker, is the empty ring memory.
"""
from collections import deque

import numpy as np


_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "Invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "Grid must contain integers")
    _require(np.all((a >= 0) & (a <= 9)), "Invalid ARC colors")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _parse(value):
    a = _grid(value)
    h, w = a.shape
    _require(h >= 8 and w >= 8, "The field must accommodate a ring memory")
    wall_color = int(a[0, 0])
    _require(np.all(a[0] == wall_color) and np.all(a[-1] == wall_color)
             and np.all(a[:, 0] == wall_color) and np.all(a[:, -1] == wall_color),
             "The complete outer border must identify walls")
    background = int(a[1, 1])
    _require(background != wall_color and all(int(a[r, c]) == background
             for r, c in ((1, 1), (1, w-2), (h-2, 1), (h-2, w-2))),
             "Four inner corners must identify background")
    candidates = []
    for marker in range(10):
        if marker in (background, wall_color):
            continue
        coords = np.argwhere(a == marker)
        if len(coords) != 4:
            continue
        mr, mc = map(int, coords.min(axis=0))
        if not np.all(a[mr:mr+2, mc:mc+2] == marker):
            continue
        for capacity in range(2, 10):
            side = 2 * capacity
            r, c = mr-capacity+1, mc-capacity+1
            if not (2 <= r and 2 <= c and r+side <= h-2 and c+side <= w-2):
                continue
            slot = int(a[r, c])
            if slot in (background, wall_color, marker):
                continue
            patch = a[r:r+side, c:c+side]
            expected = np.full((side, side), slot, dtype=np.int8)
            expected[capacity-1:capacity+1, capacity-1:capacity+1] = marker
            if np.array_equal(patch, expected) and np.count_nonzero(a == slot) == side*side-4:
                candidates.append((r, c, capacity, slot, marker))
    _require(len(candidates) == 1, "Input must have exactly one recognizable empty ring memory")
    r, c, capacity, slot, marker = candidates[0]
    footprint = np.zeros(a.shape, dtype=bool)
    footprint[r:r+2*capacity, c:c+2*capacity] = True
    payload = ~np.isin(a, (background, wall_color, slot, marker))
    _require(np.any(payload), "There must be visible payload")
    return dict(grid=a, background=background, wall_color=wall_color,
                wall=a == wall_color, slot=slot, marker=marker,
                pos=(r, c), capacity=capacity, footprint=footprint, payload=payload)


def _distances(wall, footprint):
    """Every memory cell is a distance-zero source; payload is traversable."""
    h, w = wall.shape
    d = np.full((h, w), -1, dtype=np.int16)
    starts = [tuple(map(int, cell)) for cell in np.argwhere(footprint)]
    queue = deque(starts)
    d[footprint] = 0
    while queue:
        r, c = queue.popleft()
        for dr, dc in _STEPS:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and not wall[nr, nc] and d[nr, nc] < 0:
                d[nr, nc] = d[r, c]+1
                queue.append((nr, nc))
    return d


def _wave_distances(wall, footprint):
    """Array wavefront cross-check, separate from the queue used by solve."""
    d = np.full(wall.shape, -1, dtype=np.int16)
    d[footprint] = 0
    frontier, seen = footprint.copy(), footprint.copy()
    for depth in range(1, wall.size+1):
        spread = np.zeros(wall.shape, dtype=bool)
        spread[1:] |= frontier[:-1]
        spread[:-1] |= frontier[1:]
        spread[:, 1:] |= frontier[:, :-1]
        spread[:, :-1] |= frontier[:, 1:]
        frontier = spread & ~seen & ~wall
        if not np.any(frontier):
            break
        d[frontier] = depth
        seen |= frontier
    return d


def _selection(info, d):
    a, capacity = info['grid'], info['capacity']
    cells = [tuple(map(int, cell)) for cell in np.argwhere(info['payload'] & (d >= 0))]
    cells.sort(key=lambda p: (int(d[p]), p[0], p[1]))
    if len(cells) > capacity:
        _require(d[cells[capacity-1]] < d[cells[capacity]], "Capacity cannot cut a simultaneous arrival")
    chosen = cells[:capacity]
    groups = {}
    for p in chosen:
        groups.setdefault(int(d[p]), set()).add(int(a[p]))
    _require(all(len(colors) == 1 for colors in groups.values()),
             "Simultaneously selected payload must have the same color")
    # Coordinates break ties only for bookkeeping. Equal-distance selected cells
    # have the same color, so neither erasure nor ring order depends on that tie.
    return chosen, cells, [int(a[p]) for p in chosen]


def _write(info, chosen, colors):
    out = info['grid'].copy()
    for p in chosen:
        out[p] = info['background']
    r, c = info['pos']
    side = 2*info['capacity']
    edge = np.minimum(np.arange(side), np.arange(side)[::-1])
    depth = np.minimum(edge[:, None], edge[None, :])
    patch = out[r:r+side, c:c+side]
    for index, color in enumerate(colors):
        patch[depth == index] = color
    return out


def _center_origin_rejected(info, chosen, reachable, colors):
    """A strict inversion rejects distances measured from the center marker."""
    center = np.zeros(info['grid'].shape, dtype=bool)
    r, c = info['pos']
    k = info['capacity']
    center[r+k-1:r+k+1, c+k-1:c+k+1] = True
    d = _distances(info['wall'], center)
    retained = reachable[len(chosen):]
    if chosen and retained and max(d[p] for p in chosen) > min(d[p] for p in retained):
        return True
    return any(d[p] > d[q] and colors[i] != colors[j]
               for i, p in enumerate(chosen) for j, q in enumerate(chosen) if i < j)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Erase nearest reachable cells and record their colors outside-to-inside."""
    info = _parse(input_grid)
    d = _distances(info['wall'], info['footprint'])
    chosen, _, colors = _selection(info, d)
    return _write(info, chosen, colors)


def _rectangle(wall, bounds):
    r0, c0, r1, c1 = bounds
    wall[r0, c0:c1+1] = True
    wall[r1, c0:c1+1] = True
    wall[r0:r1+1, c0] = True
    wall[r0:r1+1, c1] = True


def _overlaps(bounds, pos, side, margin=1):
    r0, c0, r1, c1 = bounds
    r, c = pos
    return not (r1 < r-margin or c1 < c-margin or r0 >= r+side+margin or c0 >= c+side+margin)


def _geometry(rng, capacity, kind):
    side = 2*capacity
    h = int(rng.integers(max(12, side+5), 31))
    w = int(rng.integers(max(12, side+5), 31))
    pos = (int(rng.integers(2, h-side-1)), int(rng.integers(2, w-side-1)))
    r, c = pos
    wall = np.zeros((h, w), dtype=bool)
    if kind == 'maze':
        wall[1:-1, 1:-1] = True
        nodes = [(rr, cc) for rr in range(1, h-1, 2) for cc in range(1, w-1, 2)]
        start = nodes[int(rng.integers(len(nodes)))]
        stack, visited = [start], {start}
        wall[start] = False
        while stack:
            rr, cc = stack[-1]
            options = [(rr+2*dr, cc+2*dc) for dr, dc in _STEPS
                       if 1 <= rr+2*dr < h-1 and 1 <= cc+2*dc < w-1
                       and (rr+2*dr, cc+2*dc) not in visited]
            if not options:
                stack.pop()
                continue
            nr, nc = options[int(rng.integers(len(options)))]
            wall[(rr+nr)//2, (cc+nc)//2] = False
            wall[nr, nc] = False
            visited.add((nr, nc))
            stack.append((nr, nc))
        for _ in range(int(rng.integers(1, 5))):
            rr, cc = int(rng.integers(1, h-3)), int(rng.integers(1, w-3))
            wall[rr:min(h-1, rr+int(rng.integers(2, 6))),
                 cc:min(w-1, cc+int(rng.integers(2, 6)))] = False
    elif kind == 'partitions':
        rooms = [(1, 1, h-2, w-2)]
        for _ in range(int(rng.integers(3, 10))):
            if not rooms:
                break
            j = int(rng.integers(len(rooms)))
            r0, c0, r1, c1 = rooms.pop(j)
            horizontal = (rng.random() < .5 and r1-r0 >= 5) or c1-c0 < 5
            if horizontal and r1-r0 >= 5:
                cut = int(rng.integers(r0+2, r1-1))
                wall[cut, c0:c1+1] = True
                for _ in range(int(rng.integers(0, 4))):
                    gap = int(rng.integers(c0, c1+1))
                    wall[cut, gap:min(c1+1, gap+int(rng.integers(1, 4)))] = False
                rooms.extend([(r0, c0, cut-1, c1), (cut+1, c0, r1, c1)])
            elif c1-c0 >= 5:
                cut = int(rng.integers(c0+2, c1-1))
                wall[r0:r1+1, cut] = True
                for _ in range(int(rng.integers(0, 4))):
                    gap = int(rng.integers(r0, r1+1))
                    wall[gap:min(r1+1, gap+int(rng.integers(1, 4))), cut] = False
                rooms.extend([(r0, c0, r1, cut-1), (r0, cut+1, r1, c1)])
    elif kind == 'detour':
        # A randomly positioned U, often with an extra rail, gives routes that
        # run away from the collector before returning close to it.
        top, left, right = r-1, c-1, c+side
        bottom = int(rng.integers(r+side, h-1))
        wall[top, left:right+1] = True
        wall[top:bottom+1, left] = True
        wall[top:bottom+1, right] = True
        if rng.random() < .4 and bottom+2 < h-1:
            wall[bottom+2, 2:max(3, right)] = True
    elif kind == 'courtyards':
        for offset in range(1, min(r-1, c-1, h-r-side-1, w-c-side-1)+1, 2):
            bounds = (r-offset, c-offset, r+side-1+offset, c+side-1+offset)
            _rectangle(wall, bounds)
            r0, c0, r1, c1 = bounds
            for _ in range(int(rng.integers(0, 5))):
                which = int(rng.integers(4))
                if which < 2:
                    x = int(rng.integers(c0+1, c1))
                    wall[r0 if which == 0 else r1, x:min(c1, x+int(rng.integers(1, 4)))] = False
                else:
                    x = int(rng.integers(r0+1, r1))
                    wall[x:min(r1, x+int(rng.integers(1, 4))), c0 if which == 2 else c1] = False
    elif kind == 'segments':
        for _ in range(int(rng.integers(2, 9))):
            r0, c0 = int(rng.integers(1, h-3)), int(rng.integers(1, w-3))
            r1, c1 = int(rng.integers(r0+2, h-1)), int(rng.integers(c0+2, w-1))
            primitive = int(rng.integers(3))
            if primitive != 1:
                wall[r0, c0:c1+1] = True
            if primitive != 0:
                wall[r0:r1+1, c1] = True
    elif kind != 'open':
        raise ValueError('Unknown geometry kind')
    # Independent small closed/open chambers can combine with any non-open
    # layout. They supply genuine unreachable regions, rather than a fixed lane.
    if kind != 'open':
        for _ in range(int(rng.integers(1, 5))):
            rh, rw = int(rng.integers(3, 6)), int(rng.integers(3, 6))
            r0, c0 = int(rng.integers(1, h-rh)), int(rng.integers(1, w-rw))
            bounds = (r0, c0, r0+rh-1, c0+rw-1)
            if _overlaps(bounds, pos, side):
                continue
            wall[r0+1:r0+rh-1, c0+1:c0+rw-1] = False
            _rectangle(wall, bounds)
            if rng.random() < .35:
                wall[r0, c0+1] = False
    wall[r:r+side, c:c+side] = False
    wall[0] = wall[-1] = True
    wall[:, 0] = wall[:, -1] = True
    for p in ((1, 1), (1, w-2), (h-2, 1), (h-2, w-2)):
        wall[p] = False
    return wall, pos


def _fallback(rng, capacity):
    """Parameterized U and closed pocket, guaranteeing a detour witness."""
    side = 2*capacity
    h, w = min(30, side+10), min(30, side+12)
    r, c = 5, 5
    wall = np.zeros((h, w), dtype=bool)
    wall[r-1, c-1:c+side+1] = True
    wall[r-1:h-3, c-1] = True
    wall[r-1:h-3, c+side] = True
    _rectangle(wall, (1, 1, 3, 3))
    wall[0] = wall[-1] = True
    wall[:, 0] = wall[:, -1] = True
    wall[1, 1] = False
    turns = int(rng.integers(4))
    mask = np.zeros(wall.shape, dtype=bool)
    mask[r:r+side, c:c+side] = True
    wall, mask = np.rot90(wall, turns).copy(), np.rot90(mask, turns).copy()
    rr, cc = np.argwhere(mask).min(axis=0)
    return wall, (int(rr), int(cc))


def _pick(rng, cells, count, width, style, forced=()):
    chosen = list(dict.fromkeys(int(x) for x in forced))
    available = set(map(int, cells))-set(chosen)
    if count < len(chosen) or count-len(chosen) > len(available):
        return None
    anchor = chosen[0] if chosen else int(rng.choice(sorted(available))) if available else 0
    ar, ac = divmod(anchor, width)
    horizontal = bool(rng.integers(2))
    while len(chosen) < count:
        options = np.array(sorted(available), dtype=int)
        if style == 'dispersed':
            x = int(rng.choice(options))
        else:
            rr, cc = options//width, options % width
            if style == 'compact':
                score = np.abs(rr-ar)+np.abs(cc-ac)
            else:
                score = (np.abs(rr-ar)*width+np.abs(cc-ac) if horizontal
                         else np.abs(cc-ac)*width+np.abs(rr-ar))
            best = options[score == score.min()]
            x = int(rng.choice(best))
        chosen.append(x)
        available.remove(x)
    return chosen


def _supply(rng, d, footprint, capacity, total, style, detour):
    h, w = d.shape
    active = (d > 0) & ~footprint
    corners = ((1, 1), (1, w-2), (h-2, 1), (h-2, w-2))
    for p in corners:
        active[p] = False
    all_cells = np.flatnonzero(active)
    if len(all_cells) < total:
        return None
    if total <= capacity:
        return _pick(rng, all_cells, total, w, style)
    rr, cc = np.indices(d.shape)
    fr, fc = np.argwhere(footprint).min(axis=0)
    side = 2*capacity
    direct = np.maximum(fr-rr, 0)+np.maximum(rr-fr-side+1, 0)+np.maximum(fc-cc, 0)+np.maximum(cc-fc-side+1, 0)
    for threshold in rng.permutation(np.unique(d[active])):
        near = np.flatnonzero(active & (d <= threshold))
        far = np.flatnonzero(active & (d > threshold))
        if len(near) < capacity or len(far) < total-capacity:
            continue
        boundary = np.flatnonzero(active & (d == threshold))
        forced_a, forced_b = None, None
        if detour:
            far_min = int(direct.flat[far].min())
            options = [int(x) for x in boundary if direct.flat[x] > far_min]
            if not options:
                continue
            forced_a = int(rng.choice(options))
            options = [int(x) for x in far if direct.flat[x] < direct.flat[forced_a]]
            forced_b = int(rng.choice(options))
        else:
            forced_a = int(rng.choice(boundary))
        selected = _pick(rng, near, capacity, w, style, [forced_a])
        retained = _pick(rng, far, total-capacity, w, style,
                         [] if forced_b is None else [forced_b])
        if selected is not None and retained is not None and len(set(int(d.flat[x]) for x in selected)) >= 2:
            return selected+retained
    return None


def _make_pair(rng, capacity, obligation, palette, split):
    side = 2*capacity
    if obligation == 'partial':
        total = int(rng.integers(2, capacity))
    elif obligation == 'witness':
        total = capacity+int(rng.integers(2, 9))
    else:
        mode = rng.random()
        total = 0 if mode < .06 else int(rng.integers(1, capacity+1)) if mode < .62 else capacity+int(rng.integers(1, 11))
    kinds = ['open', 'segments', 'partitions', 'courtyards', 'maze', 'detour']
    for attempt in range(80):
        partial_fallback = attempt >= 60 and obligation == 'partial'
        if partial_fallback:
            wall = np.zeros((side+8, side+10), dtype=bool)
            wall[0] = wall[-1] = True
            wall[:, 0] = wall[:, -1] = True
            pos = (4, 4)
        elif attempt >= 60:
            wall, pos = _fallback(rng, capacity)
        else:
            weights = [.15, .23, .22, .16, .12, .12]
            if obligation == 'witness':
                weights = [0, .12, .23, .10, .20, .35]
            elif split == 'test':
                weights = [.13, .19, .24, .14, .18, .12]
            wall, pos = _geometry(rng, capacity, str(rng.choice(kinds, p=weights)))
        h, w = wall.shape
        r, c = pos
        footprint = np.zeros(wall.shape, dtype=bool)
        footprint[r:r+side, c:c+side] = True
        d = _distances(wall, footprint)
        style = str(rng.choice(['dispersed', 'compact', 'line']))
        supply = _supply(rng, d, footprint, capacity, total, style, obligation == 'witness')
        if partial_fallback:
            active = (d > 0) & ~wall
            for p in ((1, 1), (1, w-2), (h-2, 1), (h-2, w-2)):
                active[p] = False
            forced = [r*w+c-1, (r-2)*w+c+capacity-1]
            supply = _pick(rng, np.flatnonzero(active), total, w, style, forced)
        if supply is None:
            continue
        inactive = (d < 0) & ~wall & ~footprint
        for p in ((1, 1), (1, w-2), (h-2, 1), (h-2, w-2)):
            inactive[p] = False
        minimum = 1 if obligation == 'witness' or total == 0 else 0
        limit = min(8, int(inactive.sum()))
        if minimum > limit:
            continue
        distraction_count = int(rng.integers(minimum, limit+1)) if minimum or rng.random() < .8 else 0
        distraction = _pick(rng, np.flatnonzero(inactive), distraction_count, w, style)
        local = palette if palette is not None else tuple(map(int, rng.choice(10, 4, replace=False)))
        background, wall_color, slot, marker = local
        pool = [x for x in range(10) if x not in local]
        richness = int(rng.integers(2, 7))
        colors = list(map(int, rng.choice(pool, richness, replace=False)))
        a = np.full(wall.shape, background, dtype=np.int8)
        a[wall] = wall_color
        a[footprint] = slot
        a[r+capacity-1:r+capacity+1, c+capacity-1:c+capacity+1] = marker
        ordered = sorted(supply, key=lambda x: (int(d.flat[x]), x))
        chosen = ordered[:capacity]
        if obligation in ('partial', 'witness') and len({int(d.flat[x]) for x in chosen}) < 2:
            continue
        groups = sorted({int(d.flat[x]) for x in chosen})
        # Simultaneous arrivals have a common color. Palette order has no meaning.
        distance_colors = {depth: int(rng.choice(colors)) for depth in groups}
        if obligation in ('partial', 'witness'):
            distance_colors[groups[0]], distance_colors[groups[-1]] = colors[:2]
        if partial_fallback:
            distance_colors[1], distance_colors[2] = max(colors[:2]), min(colors[:2])
            distance_colors[groups[-1]] = min(colors[:2])
        for x in chosen:
            a.flat[x] = distance_colors[int(d.flat[x])]
        if obligation == 'witness':
            code = [int(a.flat[x]) for x in chosen]
            if not any(x == y for x, y in zip(code, code[1:])):
                # With >=4 slots, two arrival groups may merge into a band while
                # a differently colored final group still establishes direction.
                if len(groups) >= 3:
                    distance_colors[groups[1]] = distance_colors[groups[0]]
                    for x in chosen:
                        a.flat[x] = distance_colors[int(d.flat[x])]
                else:
                    continue
        for x in ordered[capacity:]+distraction:
            a.flat[x] = int(rng.choice(colors))
        try:
            info = _parse(a)
            selected, reachable, code = _selection(info, d)
        except AssertionError:
            continue
        if obligation == 'partial' and (len(set(code)) < 2 or code == code[::-1]
                or code == sorted(code) or not _center_origin_rejected(info, selected, reachable, code)):
            continue
        if obligation == 'witness':
            row_code = [int(a[p]) for p in sorted(selected)]
            if row_code == code or code == code[::-1]:
                continue
        return {'input': a, 'output': _write(info, selected, code)}
    raise AssertionError('Bounded construction could not satisfy the evidence obligation')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    for value, lower, upper, name in ((num_train, 2, 10, 'num_train'), (num_test, 1, 4, 'num_test')):
        if not _integer(value) or not lower <= int(value) <= upper:
            raise ValueError(f'{name} must be an integer in {lower}..{upper}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    rng = np.random.default_rng(seed)
    # Palette policy and extrapolation direction vary across puzzles, not rules.
    fixed_palette = tuple(map(int, rng.choice(10, 4, replace=False))) if rng.random() < .45 else None
    larger_test = bool(rng.integers(2))
    capacities = [int(rng.integers(3, 7)), int(rng.integers(4, 7))]
    capacities.extend(int(rng.integers(3, 7)) for _ in range(num_train-2))
    obligations = ['partial', 'witness']+['random']*(num_train-2)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        specs = list(zip(capacities, obligations)) if split == 'train' else [
            (int(rng.integers(7, 10)) if larger_test else 2, 'random') for _ in range(num_test)]
        for capacity, obligation in specs:
            for _ in range(12):
                pair = _make_pair(rng, capacity, obligation, fixed_palette, split)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError('Unique input construction exhausted')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def pair_features(input_grid):
    """Input-derived measurements, without seed or construction metadata."""
    info = _parse(input_grid)
    a = info['grid']
    d = _distances(info['wall'], info['footprint'])
    chosen, reachable, code = _selection(info, d)
    rr, cc = np.indices(a.shape)
    r, c = info['pos']
    side = 2*info['capacity']
    direct = np.maximum(r-rr, 0)+np.maximum(rr-r-side+1, 0)+np.maximum(c-cc, 0)+np.maximum(cc-c-side+1, 0)
    retained = reachable[len(chosen):]
    path_reversal = bool(chosen and retained and max(int(direct[p]) for p in chosen) > min(int(direct[p]) for p in retained))
    active = (d > 0) & ~info['footprint']
    arrivals = [int(d[p]) for p in chosen]
    code_roles = {}
    return dict(height=a.shape[0], width=a.shape[1], capacity=info['capacity'],
                pos=info['pos'], roles=(info['background'], info['wall_color'], info['slot'], info['marker']),
                selected=len(chosen), reachable=len(reachable), excess=len(retained),
                unreachable=int(np.count_nonzero(info['payload'] & (d < 0))),
                code=code, normalized_code=tuple(code_roles.setdefault(x, len(code_roles)) for x in code),
                repeat=any(x == y for x, y in zip(code, code[1:])),
                simultaneous=len(arrivals) != len(set(arrivals)),
                recurrence=any(code[i] == code[j] and any(x != code[i] for x in code[i:j])
                               for i in range(len(code)) for j in range(i+2, len(code))),
                reverse_diff=code != code[::-1],
                row_order_diff=code != [int(a[p]) for p in sorted(chosen)],
                center_origin_diff=_center_origin_rejected(info, chosen, reachable, code),
                numeric_order_diff=code != sorted(code),
                path_reversal=path_reversal,
                max_distance=int(d[active].max()) if active.any() else 0,
                detour=int((d-direct)[active].max()) if active.any() else 0,
                openings=int(np.count_nonzero(active & (direct == 1))),
                internal_walls=int(info['wall'][1:-1, 1:-1].sum()))


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Invalid puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'Splits must be lists')
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, 'Invalid counts')
    seen, stats = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Invalid pair schema')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape, 'Canvas size must be preserved')
            _require(_key(a) not in seen, 'Duplicate input')
            seen.add(_key(a))
            info = _parse(a)
            d = _wave_distances(info['wall'], info['footprint'])
            chosen, reachable, colors = _selection(info, d)
            expected = a.copy()
            for p in chosen:
                expected[p] = info['background']
            # Check each ring by its four borders, without solve's depth array.
            r, c = info['pos']
            side = 2*info['capacity']
            for k, color in enumerate(colors):
                top, left, bottom, right = r+k, c+k, r+side-1-k, c+side-1-k
                expected[top, left:right+1] = color
                expected[bottom, left:right+1] = color
                expected[top:bottom+1, left] = color
                expected[top:bottom+1, right] = color
            _require(np.array_equal(b, expected), 'Wrong erasure, arrival order, or ring filling')
            erased = (~info['footprint']) & (a != b)
            _require(int(erased.sum()) == len(chosen), 'One erased cell per filled ring')
            _require(all(b[p] == a[p] for p in reachable[len(chosen):]), 'Reachable excess changed')
            _require(np.array_equal(b[info['wall']], a[info['wall']]), 'Walls changed')
            center = b[r+info['capacity']-1:r+info['capacity']+1,
                       c+info['capacity']-1:c+info['capacity']+1]
            _require(np.all(center == (colors[-1] if len(chosen) == info['capacity'] else info['marker'])),
                     'Center marker overwrite does not match saturation')
            stats[split].append(pair_features(a))
    train, tests = stats['train'], stats['test']
    _require(any(2 <= f['selected'] < f['capacity'] and len(set(f['code'])) >= 2
                 and f['reverse_diff'] and f['center_origin_diff'] and f['numeric_order_diff']
                 for f in train), 'Training needs ordered partial filling and explicit distance origin evidence')
    _require(any(f['excess'] and f['unreachable'] and f['path_reversal'] and f['repeat']
                 and f['row_order_diff'] and f['reverse_diff'] for f in train),
             'Training needs excess, distractors, detours, order, and band thickness evidence')
    capacities = [f['capacity'] for f in train]
    _require(all(f['capacity'] < min(capacities) for f in tests)
             or all(f['capacity'] > max(capacities) for f in tests),
             'Tests must extrapolate ring capacity in one direction')


def _signature(puzzle):
    return tuple(tuple((_key(_grid(p['input'])), _key(_grid(p['output'])))
                       for p in puzzle[s]) for s in ('train', 'test'))


def _expect_failure(function, exception=AssertionError):
    try:
        function()
    except exception:
        return
    raise AssertionError('Expected rejection did not occur')


def self_test() -> dict:
    """Bounded semantic fixtures, all count contracts, and 42-seed diversity."""
    puzzles, features, structures = [], [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        _require(_signature(puzzle) == _signature(generate_puzzle(seed)), 'Nondeterministic puzzle')
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                _require(np.array_equal(solve(pair['input']), pair['output']), 'solve disagrees')
                f = pair_features(pair['input'])
                f['split'] = split
                features.append(f)
                mapping = {}
                norm = tuple(mapping.setdefault(int(x), len(mapping)) for x in pair['input'].flat)
                structures.add((pair['input'].shape, norm))
    _require(len({_signature(p) for p in puzzles}) == 42, 'Seed diversity collapsed')
    _require(len(structures) >= 180, 'Geometry diversity collapsed')
    _require({f['capacity'] for f in features} == set(range(2, 10)), 'Capacity variation missing')
    for name in ('height', 'width', 'selected', 'excess', 'unreachable', 'max_distance', 'openings'):
        minimum = 6 if name in ('selected', 'excess', 'unreachable') else 10
        _require(len({f[name] for f in features}) >= minimum, 'Missing variation: '+name)
    for name in ('repeat', 'simultaneous', 'recurrence', 'path_reversal', 'row_order_diff'):
        _require(sum(f[name] for f in features) >= 8, 'Missing family feature: '+name)
    for role in range(4):
        _require({f['roles'][role] for f in features} == set(range(10)), 'Role palette range missing')
    _require(any(f['internal_walls'] == 0 for f in features), 'Open fields missing')
    _require(any(f['selected'] == 0 for f in features), 'Zero collection missing')
    _require(max(f['max_distance'] for f in features) >= 25, 'Long paths missing')
    tests = [f for f in features if f['split'] == 'test']
    _require(any(f['capacity'] == 2 for f in tests) and any(f['capacity'] >= 7 for f in tests),
             'Both extrapolation directions must occur')
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(1000+10*a+b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (3, None)):
        _expect_failure(lambda a=a, b=b: generate_puzzle(0, a, b), ValueError)
    # A hand-built line tests traversable colored payload, a cut cluster,
    # simultaneous same-color arrivals, and untouched unreachable supply.
    a = np.zeros((12, 18), dtype=np.int8)
    a[0] = a[-1] = 8
    a[:, 0] = a[:, -1] = 8
    a[4:10, 3:9] = 9
    a[6:8, 5:7] = 7
    a[5, 9:14] = [1, 1, 2, 3, 4]
    a[2:5, 14:17] = 8
    a[3, 15] = 6
    expected = a.copy()
    expected[5, 9:12] = 0
    expected[4:10, 3:9] = 1
    expected[6:8, 5:7] = 2
    _require(np.array_equal(solve(a), expected), 'Cell-wise collection/band fixture failed')
    # Capacity must not break an equal-distance group, even if all colors agree.
    tied = a.copy()
    tied[5, 9:14] = 0
    tied[3, 4:8] = 1
    _expect_failure(lambda: solve(tied))
    # Below capacity, mixed-color simultaneous arrivals have no ordered code.
    mixed = a.copy()
    mixed[5, 9:14] = 0
    mixed[3, 4:6] = [1, 2]
    _expect_failure(lambda: solve(mixed))
    # Rigid field rotation preserves a concentric output; role recoloring also
    # commutes with solve, since all color conventions are visible in the input.
    for turns in range(4):
        _require(np.array_equal(solve(np.rot90(a, turns)), np.rot90(expected, turns)), 'Rotation equivariance failed')
    permutation = np.array([5, 4, 3, 2, 1, 0, 9, 8, 7, 6], dtype=np.int8)
    _require(np.array_equal(solve(permutation[a]), permutation[expected]), 'Recoloring equivariance failed')
    # The solver covers the domain, including fields and supplies outside the
    # constructor's size/load sampling envelope.
    tiny = np.zeros((8, 8), dtype=np.int8)
    tiny[0] = tiny[-1] = 8
    tiny[:, 0] = tiny[:, -1] = 8
    tiny[2:6, 2:6] = 9
    tiny[3:5, 3:5] = 7
    tiny[1, 3] = 1
    tiny_expected = tiny.copy()
    tiny_expected[1, 3] = 0
    tiny_expected[2:6, 2:6] = 1
    tiny_expected[3:5, 3:5] = 7
    _require(np.array_equal(solve(tiny), tiny_expected), 'Minimum field boundary failed')
    many = np.zeros((8, 30), dtype=np.int8)
    many[0] = many[-1] = 8
    many[:, 0] = many[:, -1] = 8
    many[1, 1:7] = many[6, 1:7] = 8
    many[2:6, 1] = many[3:6, 6] = 8
    many[1, 1] = many[6, 1] = 0
    many[2:6, 2:6] = 9
    many[3:5, 3:5] = 7
    many[2, 6:29] = 1
    many_expected = many.copy()
    many_expected[2, 6:8] = 0
    many_expected[2:6, 2:6] = 1
    _require(np.array_equal(solve(many), many_expected), 'Large supply outside sampling envelope failed')
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ('train', 'test')}
    broken['train'][0]['output'][0, 0] = (int(broken['train'][0]['output'][0, 0])+1) % 10
    _expect_failure(lambda: validate_generated_puzzle(broken))
    return dict(seeds=42, pairs=len(features), distinct_structures=len(structures),
                capacities=sorted({f['capacity'] for f in features}),
                shapes=len({(f['height'], f['width']) for f in features}),
                normalized_codes=len({f['normalized_code'] for f in features}),
                path_reversals=sum(f['path_reversal'] for f in features),
                simultaneous_arrivals=sum(f['simultaneous'] for f in features),
                zero_cases=sum(f['selected'] == 0 for f in features),
                max_path=max(f['max_distance'] for f in features))
