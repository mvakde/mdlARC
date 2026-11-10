"""Arrow-bearing templates slide as finite rigid masks, then the arena is cropped.

Only Python's standard library and NumPy are used. No parent code or source data
is imported. The parser and solver receive only the input grid.
"""
from collections import Counter
import numpy as np

D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
CARDINAL = tuple(d for d in D8 if 0 in d)
DIAGONAL = tuple(d for d in D8 if 0 not in d)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}: {value}")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((a >= 0) & (a <= 9)), "invalid color")
    return a


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _puzzle_key(p):
    return tuple(tuple((_key(pair['input']), _key(pair['output'])) for pair in p[s])
                 for s in ('train', 'test'))


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in D4:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _bounds(cells):
    rr, cc = zip(*[(p[0], p[1]) for p in cells])
    return min(rr), min(cc), max(rr) - min(rr) + 1, max(cc) - min(cc) + 1


def _offsets(template):
    return tuple((r, c) for r, c, _ in template)


def _arrow(template):
    """A color entirely forming a 2..4-cell ray immediately behind the tip."""
    candidates = []
    colors = {v for r, c, v in template if (r, c) != (0, 0)}
    for color in colors:
        cells = {(r, c) for r, c, v in template if v == color}
        if not 2 <= len(cells) <= 4:
            continue
        for dr, dc in D8:
            if cells == {(-j * dr, -j * dc) for j in range(1, len(cells) + 1)}:
                candidates.append(((dr, dc), color, len(cells)))
    _require(len(candidates) == 1, "motif must have exactly one directional trail")
    return candidates[0]


def _fit(free, offsets):
    """Erode free space by a mask to obtain legal anchor positions."""
    h, w = free.shape
    result = np.ones_like(free, dtype=bool)
    for dr, dc in offsets:
        shifted = np.zeros_like(free, dtype=bool)
        r0, r1 = max(0, -dr), min(h, h - dr)
        c0, c1 = max(0, -dc), min(w, w - dc)
        if r0 < r1 and c0 < c1:
            shifted[r0:r1, c0:c1] = free[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        result &= shifted
    return result


def _box_offsets(template):
    r, c, h, w = _bounds(template)
    return tuple((rr, cc) for rr in range(r, r + h) for cc in range(c, c + w))


def _trace(fit, start, direction, corner=True):
    r, c = start
    dr, dc = direction
    path = [(r, c)]
    h, w = fit.shape
    while True:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w and fit[nr, nc]):
            return path
        if corner and dr and dc and (not fit[nr, c] or not fit[r, nc]):
            return path
        r, c = nr, nc
        path.append((r, c))


def _swept(template, path, direction):
    positions = set(path)
    dr, dc = direction
    if dr and dc:
        # Include the two side placements used to certify each successful step.
        for r, c in path[:-1]:
            positions.update(((r + dr, c), (r, c + dc)))
    return {(r + dy, c + dx) for r, c in positions for dy, dx, _ in template}


def _read_input(input_grid):
    a = _grid(input_grid)
    border = np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))
    exterior = int(border[0])
    _require(np.all(border == exterior), "outer canvas border must be uniform")
    arenas = []
    for color in set(map(int, np.unique(a))) - {exterior}:
        for component in _components(a == color):
            r, c, h, w = _bounds(component)
            if h < 8 or w < 8:
                continue
            sub = a[r:r + h, c:c + w]
            if (np.all(sub[0] == color) and np.all(sub[-1] == color)
                    and np.all(sub[:, 0] == color) and np.all(sub[:, -1] == color)):
                arenas.append((r, c, h, w, color))
    _require(len(arenas) == 1, "one unambiguous rectangular arena border required")
    r, c, h, w, wall = arenas[0]
    arena = a[r:r + h, c:c + w].copy()
    nonwall = arena[1:-1, 1:-1][arena[1:-1, 1:-1] != wall]
    _require(len(nonwall) > 0, "empty arena")
    fill = int(np.bincount(nonwall.astype(int), minlength=10).argmax())
    _require(np.count_nonzero(nonwall == fill) > len(nonwall) / 2, "unclear arena fill")
    anchor_mask = (arena != wall) & (arena != fill)
    anchors = []
    for cells in _components(anchor_mask):
        _require(len(cells) == 1, "arena initially contains separated seeds only")
        ar, ac = cells[0]
        anchors.append((ar, ac, int(arena[ar, ac])))
    _require(anchors, "arena needs an anchor seed")
    outside = a != exterior
    outside[r:r + h, c:c + w] = False
    templates, repetitions, specimen_boxes = {}, Counter(), []
    for cells in _components(outside):
        _require(len(cells) >= 5, "specimen is not a complete connected body")
        colors = Counter(int(a[p]) for p in cells)
        singleton = [v for v, n in colors.items() if n == 1]
        _require(len(singleton) == 1, "one distinctive anchor color per specimen")
        key = singleton[0]
        ar, ac = next(p for p in cells if int(a[p]) == key)
        template = tuple(sorted((rr - ar, cc - ac, int(a[rr, cc])) for rr, cc in cells))
        _require(key not in templates or templates[key] == template, "conflicting reference specimens")
        _arrow(template)
        templates[key] = template
        repetitions[key] += 1
        specimen_boxes.append(_bounds(cells))
    _require(templates, "missing specimen dictionary")
    keys = set(templates)
    _require(all(key in keys for _, _, key in anchors), "seed has no template")
    for key, template in templates.items():
        _require(key not in (exterior, fill, wall), "anchor conflicts with a structural role")
        for dr, dc, color in template:
            _require(color not in (exterior, fill, wall), "body cells must be visibly distinct from fills and walls")
            _require((dr, dc) == (0, 0) or color not in keys, "decoration cannot be an anchor key")
    free = arena != wall
    configs = {key: _fit(free, _offsets(t)) for key, t in templates.items()}
    motions, used, destinations = [], set(), set()
    for ar, ac, key in anchors:
        template = templates[key]
        direction = _arrow(template)[0]
        fit = configs[key]
        _require(fit[ar, ac], "virtual initial body intersects a wall")
        path = _trace(fit, (ar, ac), direction)
        swept = _swept(template, path, direction)
        _require(used.isdisjoint(swept), "body swept areas interact")
        used.update(swept)
        _require(path[-1] not in destinations, "shared destination")
        destinations.add(path[-1])
        point_path = _trace(free, (ar, ac), direction)
        box_fit = _fit(free, _box_offsets(template))
        box_path = _trace(box_fit, (ar, ac), direction) if box_fit[ar, ac] else [(ar, ac)]
        nr, nc = path[-1][0] + direction[0], path[-1][1] + direction[1]
        corner_witness = bool(direction[0] and direction[1] and 0 <= nr < h and 0 <= nc < w
                              and fit[nr, nc])
        motions.append(dict(key=key, start=(ar, ac), end=path[-1], path=path,
                            direction=direction, distance=len(path) - 1,
                            point_witness=path[-1] != point_path[-1],
                            box_witness=len(path) > len(box_path),
                            corner_witness=corner_witness))
    return dict(grid=a, box=(r, c, h, w), arena=arena, wall=wall, fill=fill,
                exterior=exterior, templates=templates, repetitions=dict(repetitions),
                specimen_boxes=specimen_boxes, anchors=anchors, free=free,
                configs=configs, motions=motions)


def _render(info):
    output = info['arena'].copy()
    for r, c, _ in info['anchors']:
        output[r, c] = info['fill']
    for motion in info['motions']:
        ar, ac = motion['end']
        for dr, dc, color in info['templates'][motion['key']]:
            _require(output[ar + dr, ac + dc] == info['fill'], "stamp overwrites a fixed cell or another body")
            output[ar + dr, ac + dc] = color
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read local specimens and walls, slide their occupied masks, and crop."""
    return _render(_read_input(input_grid))


def _choice(rng, values):
    return values[int(rng.integers(len(values)))]


def _template(rng, direction, key, trail, cargo, extent):
    dr, dc = direction
    length = int(rng.choice([2, 2, 3]))
    skeleton = {(0, 0)} | {(-j * dr, -j * dc) for j in range(1, length + 1)}
    occupied = set(skeleton)
    if dr and dc:
        # Staircase cells connect the diagonally touching trail into one body.
        occupied |= {(-j * dr, -(j - 1) * dc) for j in range(1, length + 1)}
    count = max(len(occupied), int(rng.integers(length + 3, 15)))
    style = int(rng.integers(3))
    while len(occupied) < count:
        frontier = sorted({(r + dy, c + dx) for r, c in occupied for dy, dx in D4} - occupied)
        frontier = [p for p in frontier if max(r for r, _ in occupied | {p})
                    - min(r for r, _ in occupied | {p}) < extent
                    and max(c for _, c in occupied | {p}) - min(c for _, c in occupied | {p}) < extent]
        neighbors = np.array([sum((r + dy, c + dx) in occupied for dy, dx in D4)
                              for r, c in frontier], dtype=float)
        weights = np.ones(len(frontier)) if style == 0 else (neighbors ** 2 if style == 1 else 1 / neighbors ** 2)
        occupied.add(frontier[int(rng.choice(len(frontier), p=weights / weights.sum()))])
    extras = sorted(occupied - skeleton)
    ncolors = 2 if len(extras) >= 4 and rng.random() < .4 else 1
    assignment = list(cargo[:ncolors]) * 2
    assignment += [_choice(rng, cargo[:ncolors]) for _ in range(len(extras) - len(assignment))]
    rng.shuffle(assignment)
    template = tuple(sorted([(0, 0, key)] + [(-j * dr, -j * dc, trail) for j in range(1, length + 1)]
                            + [(r, c, int(color)) for (r, c), color in zip(extras, assignment)]))
    _arrow(template)
    return template


def _walls(rng, h, w, richness):
    wall = np.zeros((h, w), dtype=bool)
    wall[0] = wall[-1] = True
    wall[:, 0] = wall[:, -1] = True
    # Filled corner bays change the usable outer outline, inside the crop border.
    for corner in rng.choice(4, size=int(rng.integers(0, 4)), replace=False):
        bh, bw = int(rng.integers(1, max(2, h // 3))), int(rng.integers(1, max(2, w // 3)))
        if rng.random() < .5:
            for i in range(bh):
                width = max(1, bw - i * bw // bh)
                rows = 1 + i if corner < 2 else h - 2 - i
                cols = slice(1, width + 1) if corner % 2 == 0 else slice(w - 1 - width, w - 1)
                wall[rows, cols] = True
        else:
            rows = slice(1, bh + 1) if corner < 2 else slice(h - 1 - bh, h - 1)
            cols = slice(1, bw + 1) if corner % 2 == 0 else slice(w - 1 - bw, w - 1)
            wall[rows, cols] = True
    for _ in range(int(rng.integers(1, 3 + richness))):
        side = int(rng.integers(4))
        if side < 2:
            col = int(rng.integers(2, w - 2))
            length = int(rng.integers(2, h - 2))
            rows = slice(0, length) if side == 0 else slice(h - length, h)
            wall[rows, col] = True
        else:
            row = int(rng.integers(2, h - 2))
            length = int(rng.integers(2, w - 2))
            cols = slice(0, length) if side == 2 else slice(w - length, w)
            wall[row, cols] = True
    # Posts are obstacles in the same visible wall role, never arbitrary colors.
    for _ in range(int(rng.integers(1, 5 + richness))):
        rr, cc = int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2))
        wall[rr, cc] = True
        if rng.random() < .25:
            wall[rr, cc + 1] = True
    return wall


def _candidates(free, template, direction):
    fit = _fit(free, _offsets(template))
    box = _fit(free, _box_offsets(template))
    h, w = free.shape
    result = []
    for rr, cc in np.argwhere(fit):
        start = int(rr), int(cc)
        path = _trace(fit, start, direction)
        if len(path) == 1:
            continue
        point = _trace(free, start, direction)
        box_path = _trace(box, start, direction) if box[start] else [start]
        nr, nc = path[-1][0] + direction[0], path[-1][1] + direction[1]
        corner = bool(direction[0] and direction[1] and 0 <= nr < h and 0 <= nc < w and fit[nr, nc])
        result.append(dict(start=start, path=path, point=path[-1] != point[-1],
                           box=len(path) > len(box_path), corner=corner))
    return result


def _placements(rng, wall, templates, directions, need_corner, need_box):
    free = ~wall
    candidate_lists = [_candidates(free, template, d) for template, d in zip(templates, directions)]
    # The first selected body certifies footprint-sensitive stopping. Required
    # training witnesses are real properties of its rendered input, not tags.
    priority = [p for p in candidate_lists[0] if p['point'] and len(p['path']) >= 3
                and (not need_corner or p['corner']) and (not need_box or p['box'])]
    if not priority:
        return None
    used, places = set(), []
    for i, (template, direction) in enumerate(zip(templates, directions)):
        candidates = priority if i == 0 else candidate_lists[i]
        order = list(map(int, rng.permutation(len(candidates))))
        # Keep long examples but favor shorter sweeps in crowded scenes.
        if len(templates) >= 3 and i:
            order.sort(key=lambda j: len(candidates[j]['path']))
        for j in order:
            candidate = candidates[j]
            sweep = _swept(template, candidate['path'], direction)
            if used.isdisjoint(sweep):
                places.append((i, candidate['start'], candidate['path']))
                used.update(sweep)
                break
        else:
            return None
    # Extra occurrences reuse existing templates, with independently sampled
    # multiplicity. Fitting, rather than a split-specific quota, limits counts.
    desired = int(rng.integers(len(templates), min(7, len(templates) + 4)))
    for _ in range(desired - len(templates)):
        i = int(rng.integers(len(templates)))
        template, direction = templates[i], directions[i]
        for j in rng.permutation(len(candidate_lists[i])):
            candidate = candidate_lists[i][int(j)]
            sweep = _swept(template, candidate['path'], direction)
            if used.isdisjoint(sweep):
                places.append((i, candidate['start'], candidate['path']))
                used.update(sweep)
                break
    return places


def _shelf(templates, ids, width):
    row, col, rowheight = 1, 1, 0
    placements = []
    for i in ids:
        r, c, h, w = _bounds(templates[i])
        if w + 2 > width:
            return None
        if col + w > width - 1:
            row += rowheight + 1
            col, rowheight = 1, 0
        placements.append((i, row - r, col - c))
        col += w + 1
        rowheight = max(rowheight, h)
    return row + rowheight + 1, placements


def _scene(rng, arena, templates, keys, counts, exterior):
    ids = [i for i, count in enumerate(counts) for _ in range(count)]
    rng.shuffle(ids)
    th, tw = arena.shape
    options = []
    for rw in range(5, 29):
        packed = _shelf(templates, ids, rw)
        if packed is None:
            continue
        rh, places = packed
        if rh + th + 3 <= 30 and max(rw, tw) + 2 <= 30:
            options.append(('vertical', rh, rw, places))
        if rw + tw + 3 <= 30 and max(rh, th) + 2 <= 30:
            options.append(('horizontal', rh, rw, places))
    if not options:
        return None
    layout, rh, rw, places = _choice(rng, options)
    axis = 0 if layout == 'vertical' else 1
    sizes = [(rh, rw), (th, tw)]
    minimum = [max(rh, th) + 2, max(rw, tw) + 2]
    minimum[axis] = sizes[0][axis] + sizes[1][axis] + 3
    shape = [int(rng.integers(n, min(31, n + 4))) for n in minimum]
    first = int(rng.integers(2))
    positions = [[0, 0], [0, 0]]
    slack = shape[axis] - sizes[0][axis] - sizes[1][axis] - 3
    begin = 1 + int(rng.integers(slack + 1))
    positions[first][axis] = begin
    positions[1 - first][axis] = begin + sizes[first][axis] + 1
    other = 1 - axis
    for i in (0, 1):
        positions[i][other] = int(rng.integers(1, shape[other] - sizes[i][other]))
    grid = np.full(shape, exterior, dtype=np.int8)
    rr, rc = positions[0]
    for i, ar, ac in places:
        for dr, dc, color in templates[i]:
            grid[rr + ar + dr, rc + ac + dc] = color
    tr, tc = positions[1]
    grid[tr:tr + th, tc:tc + tw] = arena
    return grid


def _generate_pair(rng, directions, need_corner=False, need_box=False, teach=False, test=False):
    n = len(directions)
    for attempt in range(240):
        palette = list(map(int, rng.permutation(10)))
        exterior, wall = palette[:2]
        fill = palette[2] if teach or rng.random() < .65 else exterior
        available = [v for v in palette if v not in (exterior, wall, fill)]
        keys, decorations = available[:n], available[n:]
        extent = int(rng.choice([4, 5, 6] if test else [4, 5]))
        try:
            templates = []
            for d, key in zip(directions, keys):
                local = list(map(int, rng.permutation(decorations)))
                templates.append(_template(rng, d, key, local[0], local[1:3], extent))
        except AssertionError:
            continue
        # Two types selecting different geometry defeat a one-template shortcut.
        if n > 1 and len(set(_offsets(t) for t in templates)) == 1:
            continue
        h = int(rng.integers(11 if n < 3 else 14, 22))
        w = int(rng.integers(11 if n < 3 else 14, 26))
        wall_mask = _walls(rng, h, w, int(rng.integers(3)))
        places = _placements(rng, wall_mask, templates, directions, need_corner, need_box)
        if places is None:
            continue
        arena = np.full((h, w), fill, dtype=np.int8)
        arena[wall_mask] = wall
        for i, start, _ in places:
            arena[start] = keys[i]
        counts = [int(rng.choice([2, 2, 2, 3])) for _ in templates]
        grid = _scene(rng, arena, templates, keys, counts, exterior)
        if grid is None:
            # Two copies are enough for a full dictionary; they are the fitting
            # baseline when three copies would exceed the 30-cell canvas.
            grid = _scene(rng, arena, templates, keys, [2] * n, exterior)
        if grid is None:
            continue
        # Construct the label from sampled masks and placement paths. The
        # input parser subsequently reconstructs those facts without receiving
        # the masks, paths, palette, specimen positions, or crop coordinates.
        output = arena.copy()
        for _, start, _ in places:
            output[start] = fill
        for i, _, path in places:
            ar, ac = path[-1]
            for dr, dc, color in templates[i]:
                output[ar + dr, ac + dc] = color
        try:
            info = _read_input(grid)
            _require(np.array_equal(_render(info), output), "construction/parser disagreement")
            _require(any(m['point_witness'] for m in info['motions']), "missing footprint witness")
            _require(not need_corner or any(m['corner_witness'] for m in info['motions']), "missing corner witness")
            _require(not need_box or any(m['box_witness'] for m in info['motions']), "missing mask witness")
        except AssertionError:
            continue
        return {'input': grid, 'output': output}
    raise RuntimeError(f"bounded construction failed for {directions}, corner={need_corner}, box={need_box}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    nt = _integer(num_train, 'num_train', 2, 10)
    nq = _integer(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    cardinal, diagonal = _choice(rng, CARDINAL), _choice(rng, DIAGONAL)
    pool = [cardinal, diagonal]
    others = [d for d in D8 if d not in pool]
    for i in rng.choice(len(others), size=int(rng.integers(0, 3)), replace=False):
        pool.append(others[int(i)])
    plans = [(tuple([cardinal]), False, True, True),
             (tuple([diagonal, _choice(rng, pool)]), True, False, False)]
    for _ in range(nt - 2):
        plans.append((tuple(_choice(rng, pool) for _ in range(int(rng.choice([1, 1, 2, 2, 3])))),
                      False, False, False))
    # Move the reserved witnesses through the demonstration order.
    plans = [plans[int(i)] for i in rng.permutation(len(plans))]
    heldout = [d for d in D8 if d not in pool]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            if split == 'train':
                directions, corner, box, teach = plans[i]
            else:
                directions = [_choice(rng, heldout)]
                directions += [_choice(rng, D8) for _ in range(int(rng.choice([1, 1, 2])))]
                corner, box, teach = False, False, False
            for _ in range(8):
                pair = _generate_pair(rng, directions, corner, box, teach, split == 'test')
                fingerprint = _key(pair['input'])
                if fingerprint not in seen:
                    break
            _require(fingerprint not in seen, "duplicate input")
            seen.add(fingerprint)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid pair counts")
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair schema")
            a, b = _grid(pair['input']), _grid(pair['output'])
            fingerprint = _key(a)
            _require(fingerprint not in seen, "duplicate input")
            seen.add(fingerprint)
            info = _read_input(a)
            _require(all(n >= 2 for n in info['repetitions'].values()), "generated dictionary needs repeated specimens")
            _require(np.array_equal(_render(info), b), "incorrect rigid-body translation or crop")
            _require(np.array_equal(info['arena'][info['arena'] == info['wall']], b[info['arena'] == info['wall']]),
                     "walls changed")
            _require(any(m['distance'] >= 2 for m in info['motions']), "must show maximal travel beyond one step")
            _require(any(m['point_witness'] for m in info['motions']), "point-anchor shortcut succeeds")
            expected_colors = Counter()
            for key, template in info['templates'].items():
                multiplicity = sum(k == key for _, _, k in info['anchors'])
                expected_colors.update({color: count * multiplicity
                                        for color, count in Counter(v for _, _, v in template).items()})
                for motion in info['motions']:
                    if motion['key'] == key:
                        ar, ac = motion['end']
                        _require(all(b[ar + dr, ac + dc] == color for dr, dc, color in template), "template changed")
            actual_colors = Counter(map(int, b[(b != info['wall']) & (b != info['fill'])]))
            _require(actual_colors == +expected_colors, "body color/anchor counts changed")
            if split == 'train':
                training.append(info)
    train_directions = {m['direction'] for info in training for m in info['motions']}
    _require(len(train_directions) >= 2, "training lacks direction variation")
    _require(any(m['corner_witness'] for info in training for m in info['motions']), "missing diagonal-clearance lesson")
    _require(any(m['box_witness'] for info in training for m in info['motions']), "missing occupied-mask lesson")
    _require(any(info['fill'] != info['exterior'] for info in training), "missing transparent transfer lesson")
    _require(any(len(info['templates']) >= 2 for info in training), "missing keyed-template lesson")


def measure_puzzle(puzzle):
    """Family features recovered from cells; no sampler metadata is consulted."""
    records = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info = _read_input(pair['input'])
            r, c, h, w = info['box']
            boxes = info['specimen_boxes']
            centers = [(rr + hh / 2, cc + ww / 2) for rr, cc, hh, ww in boxes]
            cr, cc = np.mean(centers, axis=0)
            dy, dx = cr - (r + h / 2), cc - (c + w / 2)
            side = ('bottom' if dy > 0 else 'top') if abs(dy) >= abs(dx) else ('right' if dx > 0 else 'left')
            geometries = tuple(sorted(_offsets(t) for t in info['templates'].values()))
            records.append(dict(split=split, input_shape=tuple(np.asarray(pair['input']).shape),
                                output_shape=(h, w), types=len(info['templates']),
                                anchors=len(info['anchors']), directions=tuple(m['direction'] for m in info['motions']),
                                distances=tuple(m['distance'] for m in info['motions']),
                                geometries=geometries, sizes=tuple(len(t) for t in info['templates'].values()),
                                trail_lengths=tuple(_arrow(t)[2] for t in info['templates'].values()),
                                multicolor=any(len({v for dr, dc, v in t if (dr, dc) != (0, 0)}) > 2
                                               for t in info['templates'].values()),
                                corner=sum(m['corner_witness'] for m in info['motions']),
                                point=sum(m['point_witness'] for m in info['motions']),
                                box=sum(m['box_witness'] for m in info['motions']),
                                reference_side=side, reference_counts=tuple(info['repetitions'].values()),
                                fills_differ=info['fill'] != info['exterior'],
                                regions=len(_components(info['free'])),
                                wall_geometry=tuple(map(tuple, np.argwhere(~info['free']))),
                                palette=(info['exterior'], info['fill'], info['wall'], tuple(info['templates']))))
    return records


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "seed diversity")
    records = [r for p in puzzles for r in measure_puzzle(p)]
    _require({d for r in records for d in r['directions']} == set(D8), "all compass orientations")
    _require({r['types'] for r in records} == {1, 2, 3}, "type-count diversity")
    _require(len({g for r in records for g in r['geometries']}) >= 120, "procedural body diversity")
    _require(len({r['input_shape'] for r in records}) >= 30, "canvas geometry diversity")
    _require(len({r['output_shape'] for r in records}) >= 35, "arena geometry diversity")
    _require(len({r['wall_geometry'] for r in records}) >= 150, "wall geometry diversity")
    _require({r['reference_side'] for r in records} == {'top', 'bottom', 'left', 'right'}, "evidence placements")
    _require({n for r in records for n in r['trail_lengths']} >= {2, 3}, "trail-length variation")
    _require({n for r in records for n in r['reference_counts']} >= {2, 3}, "specimen multiplicity")
    _require(len({r['anchors'] for r in records}) >= 4, "target multiplicity variation")
    _require(sum(r['multicolor'] for r in records) >= 15, "colored offsets")
    _require(any(r['regions'] > 1 for r in records), "partitioned usable space")
    for p in puzzles:
        batch = measure_puzzle(p)
        _require(len({r['geometries'] for r in batch}) >= 3, "within-puzzle template variation")
    # Determinism and count boundaries are cheap, independent of random quotas.
    for nt, nq in ((2, 1), (10, 4), (6, 3)):
        a = generate_puzzle(109 + nt, nt, nq)
        _require(_puzzle_key(a) == _puzzle_key(generate_puzzle(109 + nt, nt, nq)), "nondeterminism")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(1, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Whole-scene symmetry must transform the result without any extra rotation
    # of an individual body. Exercise every symmetry of one solved query.
    pair = puzzles[0]['test'][0]
    for flip in (False, True):
        for turns in range(4):
            a = np.fliplr(pair['input']) if flip else pair['input']
            b = np.fliplr(pair['output']) if flip else pair['output']
            _require(np.array_equal(solve(np.rot90(a, turns)), np.rot90(b, turns)), "scene symmetry")
    bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt wall label accepted")
    return {'seeds': 42, 'pairs': len(records),
            'body_geometries': len({g for r in records for g in r['geometries']}),
            'input_shapes': len({r['input_shape'] for r in records}),
            'arena_shapes': len({r['output_shape'] for r in records}),
            'point_witnesses': sum(r['point'] for r in records),
            'corner_witnesses': sum(r['corner'] for r in records),
            'mask_witnesses': sum(r['box'] for r in records)}
