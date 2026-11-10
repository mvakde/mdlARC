"""Soluble partitions: homologous chambers determine communicating vessels.

Only standard library and NumPy are used. Geometry, palette and doses are
visible in the input; generation labels never enter solve().
"""
from itertools import combinations
import numpy as np


HOST_SEEDS = (0, 41142710, 590040248, 110019890, 689868240, 298056817,
              327120574, 408522063, 461143988, 653092113, 847101825,
              527123735, 576282335, 582975015, 621882659, 740452263,
              213598398, 755465410, 460200674, 206569327, 824249686,
              372925185, 611046477, 405526999, 675673139, 37288738,
              120642667, 938710880, 96373561, 347913007, 206736861,
              874061638, 191213922, 583219372, 191816544, 471041137,
              811253947, 139607033, 685712466, 785972583, 237614115,
              72378391)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    parts = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        stack, points = [first], [first]
        while stack:
            r, c = stack.pop()
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in unseen:
                    unseen.remove(p)
                    stack.append(p)
                    points.append(p)
        part = np.zeros(mask.shape, dtype=bool)
        rr, cc = np.array(points).T
        part[rr, cc] = True
        parts.append(part)
    return parts


def _bbox(mask):
    p = np.argwhere(mask)
    return (*map(int, p.min(0)), *map(int, p.max(0) + 1))


def _rooms(walls):
    padded = np.pad(~walls, 1, constant_values=True)
    return [p[1:-1, 1:-1] for p in _components(padded)
            if not (p[0].any() or p[-1].any() or p[:, 0].any() or p[:, -1].any())]


def _relations(rooms):
    boxes = [_bbox(m) for m in rooms]
    return tuple(tuple(tuple(-1 if a[k + 2] <= b[k] else
                             1 if b[k + 2] <= a[k] else 0
                             for k in (0, 1)) for b in boxes) for a in boxes)


def _match(donor, target):
    """Unique orientation-preserving complete spatial graph isomorphism."""
    a, b = _relations(donor), _relations(target)
    n = len(a)
    options = [[j for j in range(n) if sorted(a[i]) == sorted(b[j])]
               for i in range(n)]
    answers = []

    def search(mapping, used):
        if len(answers) > 1:
            return
        if len(mapping) == n:
            answers.append(tuple(mapping[i] for i in range(n)))
            return
        i = min((i for i in range(n) if i not in mapping),
                key=lambda i: len(options[i]))
        for j in options[i]:
            if j not in used and all(a[i][k] == b[j][v] for k, v in mapping.items()):
                mapping[i] = j
                search(mapping, used | {j})
                del mapping[i]

    search({}, set())
    if len(answers) != 1:
        raise ValueError("chambers lack a unique visible correspondence")
    return answers[0]


def _extract(grid):
    g = np.asarray(grid)
    if (g.ndim != 2 or not np.issubdtype(g.dtype, np.integer) or
            not all(1 <= x <= 30 for x in g.shape) or
            np.any((g < 0) | (g > 9))):
        raise ValueError("invalid ARC grid")
    bg = int(g[0, 0])
    if not all(np.all(side == bg) for side in (g[0], g[-1], g[:, 0], g[:, -1])):
        raise ValueError("the family has a visible background border")
    candidates = []
    for wall in map(int, np.unique(g)):
        if wall == bg:
            continue
        parts = _components(g == wall)
        if len(parts) != 2:
            continue
        figures = []
        valid = True
        for part in parts:
            r, c, R, C = _bbox(part)
            crop = g[r:R, c:C].copy()
            local_wall = part[r:R, c:C]
            rooms = _rooms(local_wall)
            if not 3 <= len(rooms) <= 7:
                valid = False
                break
            inside = np.logical_or.reduce(rooms)
            if np.any((crop != bg) & ~local_wall & ~inside):
                valid = False
                break
            figures.append((crop, rooms, (r, c, R, C)))
        if not valid or len(figures[0][1]) != len(figures[1][1]):
            continue
        blank = [f for f in figures if np.all((f[0] == bg) | (f[0] == wall))]
        donor = [f for f in figures if not np.all((f[0] == bg) | (f[0] == wall))]
        if len(blank) != 1 or len(donor) != 1:
            continue
        d, dr, db = donor[0]
        t, tr, tb = blank[0]
        colors, doses = [], []
        for room in dr:
            values = set(map(int, np.unique(d[room]))) - {bg}
            if len(values) != 1 or wall in values:
                valid = False
                break
            color = next(iter(values))
            colors.append(color)
            doses.append(int(np.count_nonzero(d[room] == color)))
        if valid and len(set(colors)) >= 2:
            try:
                mapping = _match(dr, tr)
            except ValueError:
                continue
            candidates.append((bg, wall, d, t, dr, tr, colors, doses, mapping, db, tb))
    if len(candidates) != 1:
        raise ValueError("expected one recognizable donor/receiver framework pair")
    return candidates[0]


def _interfaces(walls, rooms):
    labels = np.full(walls.shape, -1, dtype=int)
    for i, m in enumerate(rooms):
        labels[m] = i
    h, w = walls.shape
    result = []
    for r, c in np.argwhere(walls):
        neighbors = [(int(r) + dr, int(c) + dc) for dr, dc in
                     ((-1, 0), (1, 0), (0, -1), (0, 1))]
        # Exterior-facing wall cells and zero-neighbor junction posts survive.
        if any(not (0 <= rr < h and 0 <= cc < w) or
               (not walls[rr, cc] and labels[rr, cc] < 0) for rr, cc in neighbors):
            continue
        adjacent = sorted({int(labels[rr, cc]) for rr, cc in neighbors
                           if labels[rr, cc] >= 0})
        if len(adjacent) >= 2:
            result.append((int(r), int(c), tuple(adjacent)))
    return result


def _vessels(walls, rooms, colors):
    opened = np.zeros(walls.shape, dtype=bool)
    for r, c, ids in _interfaces(walls, rooms):
        if len({colors[i] for i in ids}) == 1:
            opened[r, c] = True
    free = np.logical_or.reduce(rooms) | opened
    result = []
    for mask in _components(free):
        members = [i for i, room in enumerate(rooms) if np.any(mask & room)]
        if len({colors[i] for i in members}) != 1:
            raise ValueError("unlike liquids cannot communicate")
        result.append((mask, members))
    return opened, result


def _fill(vessel, volume):
    """Complete horizontal levels; the wet lower domain must communicate."""
    if not 1 <= volume <= int(vessel.sum()):
        raise ValueError("invalid vessel dose")
    wet = np.zeros(vessel.shape, dtype=bool)
    remaining = volume
    for r in range(vessel.shape[0] - 1, -1, -1):
        count = int(vessel[r].sum())
        if count > remaining:
            raise ValueError("partial horizontal levels are outside this family")
        wet[r] = vessel[r]
        remaining -= count
        if remaining == 0:
            break
    if remaining or len(_components(wet)) != 1:
        raise ValueError("the dose does not reach a connected complete meniscus")
    return wet


def _receive(target, tr, colors, doses, mapping, bg, wall):
    target_colors, target_doses = [0] * len(tr), [0] * len(tr)
    for i, j in enumerate(mapping):
        target_colors[j], target_doses[j] = colors[i], doses[i]
    opened, vessels = _vessels(target == wall, tr, target_colors)
    answer = target.copy()
    answer[opened] = bg
    for mask, members in vessels:
        wet = _fill(mask, sum(target_doses[i] for i in members))
        answer[wet] = target_colors[members[0]]
    return answer.astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    bg, wall, d, target, dr, tr, colors, doses, mapping, db, tb = _extract(input_grid)
    return _receive(target, tr, colors, doses, mapping, bg, wall)


def _plan(rng, n, fallback):
    if fallback:
        cols = (n + 1) // 2
        p = np.arange(2 * cols).reshape(2, cols)
        p[p >= n] = n - 1
        return p
    shapes = [(2, 3), (3, 2), (3, 3), (2, 4), (4, 2), (3, 4), (4, 3)]
    shapes = [s for s in shapes if s[0] * s[1] >= n]
    rows, cols = shapes[int(rng.integers(len(shapes)))]
    rectangles = [(0, 0, rows, cols)]
    while len(rectangles) < n:
        choices = [i for i, (r, c, R, C) in enumerate(rectangles)
                   if R - r > 1 or C - c > 1]
        i = int(rng.choice(choices))
        r, c, R, C = rectangles.pop(i)
        axis = int(rng.choice([k for k, span in enumerate((R - r, C - c)) if span > 1]))
        if axis == 0:
            cut = int(rng.integers(r + 1, R))
            rectangles.extend([(r, c, cut, C), (cut, c, R, C)])
        else:
            cut = int(rng.integers(c + 1, C))
            rectangles.extend([(r, c, R, cut), (r, cut, R, C)])
    p = np.full((rows, cols), -1, dtype=int)
    for i, (r, c, R, C) in enumerate(rectangles):
        p[r:R, c:C] = i
    # Remove outer corners only when all rooms survive connected.
    for point in ((0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)):
        i = int(p[point])
        if (p == i).sum() > 1 and rng.random() < 0.35:
            trial = p.copy()
            trial[point] = -1
            if len(_components(trial == i)) == 1:
                p = trial
    return p


def _geometry(rng, plan, fallback, top_shift=False):
    rows, cols = plan.shape
    hs = np.full(rows, 3) if fallback else rng.integers(2, 4, size=rows)
    ws = np.full(cols, 2) if fallback else rng.integers(2, 4, size=cols)
    ys, xs = np.r_[0, np.cumsum(hs + 1)], np.r_[0, np.cumsum(ws + 1)]
    labels = np.full((int(ys[-1]) + 1, int(xs[-1]) + 1), -1, dtype=int)
    # Draw each coarse chamber tile, removing internal edges of merged tiles.
    for r, c in np.ndindex(plan.shape):
        i = int(plan[r, c])
        if i < 0:
            continue
        labels[ys[r] + 1:ys[r + 1], xs[c] + 1:xs[c + 1]] = i
        if r and plan[r - 1, c] == i:
            labels[ys[r], xs[c] + 1:xs[c + 1]] = i
        if c and plan[r, c - 1] == i:
            labels[ys[r] + 1:ys[r + 1], xs[c]] = i
        if r and c and np.all(plan[r - 1:r + 1, c - 1:c + 1] == i):
            labels[ys[r], xs[c]] = i
    if not fallback:
        for i in range(int(plan.max()) + 1):
            m = labels == i
            r, c, R, C = _bbox(m)
            # Raised corner shelves retain at least one deepest cell.
            side = int(rng.integers(2))
            for depth in range(int(rng.integers(0, min(3, R - r)))):
                rr = R - 1 - depth
                cc = c + depth if side == 0 else C - 1 - depth
                if 0 <= cc < labels.shape[1] and labels[rr, cc] == i:
                    trial = labels == i
                    trial[rr, cc] = False
                    if trial.any() and len(_components(trial)) == 1:
                        labels[rr, cc] = -1
            # Independent chamber starts can reverse raster discovery order.
            if R - r >= 3 and rng.random() < 0.35:
                trial = labels == i
                trial[r] = False
                if trial.any() and len(_components(trial)) == 1:
                    labels[r, labels[r] == i] = -1
    if fallback and top_shift:
        # The fallback's upper-left chamber starts a row later in the donor.
        r = _bbox(labels == 0)[0]
        labels[r, labels[r] == 0] = -1
    interior = labels >= 0
    padded = np.pad(interior, 1)
    near = np.zeros(labels.shape, dtype=bool)
    for dr, dc in np.ndindex((3, 3)):
        near |= padded[dr:dr + labels.shape[0], dc:dc + labels.shape[1]]
    walls = near & ~interior
    return labels, walls


def _allocate(rng, volume, capacities):
    counts = np.ones(len(capacities), dtype=int)
    for _ in range(volume - len(counts)):
        choices = np.flatnonzero(counts < capacities)
        counts[int(rng.choice(choices))] += 1
    return counts


def _embed(rng, donor, target, bg, serial, fallback):
    dh, dw = donor.shape
    th, tw = target.shape
    modes = []
    if dw + tw + 3 <= 30:
        modes.append('side')
    if dh + th + 3 <= 30:
        modes.append('stack')
    if not modes:
        raise ValueError("figures do not fit")
    mode = modes[int(rng.integers(len(modes)))]
    if rng.integers(2):
        first, second = donor, target
    else:
        first, second = target, donor
    if mode == 'side':
        h, w = max(dh, th) + 2, dw + tw + 3
        h += int(rng.integers(min(3, 30 - h) + 1))
        positions = [(int(rng.integers(1, h - first.shape[0])), 1),
                     (int(rng.integers(1, h - second.shape[0])), first.shape[1] + 2)]
    else:
        h, w = dh + th + 3, max(dw, tw) + 2
        w += int(rng.integers(min(3, 30 - w) + 1))
        positions = [(1, int(rng.integers(1, w - first.shape[1]))),
                     (first.shape[0] + 2, int(rng.integers(1, w - second.shape[1])))]
    canvas = np.full((h, w), bg, dtype=np.int8)
    for obj, (r, c) in zip((first, second), positions):
        canvas[r:r + obj.shape[0], c:c + obj.shape[1]] = obj
    return canvas


def _make_pair(rng, n, merge_size, serial, bg, wall, force_fallback=False):
    for attempt in range(80 if force_fallback else 0, 81):
        fallback = attempt >= 80
        plan = _plan(rng, n, fallback)
        dl, dw = _geometry(rng, plan, fallback, top_shift=serial == 1)
        tl, tw = _geometry(rng, plan, fallback)
        drooms, trooms = [dl == i for i in range(n)], [tl == i for i in range(n)]
        if _relations(drooms) != _relations(trooms):
            continue
        adjacency = [set() for _ in range(n)]
        for r, c, ids in _interfaces(tw, trooms):
            for i in ids:
                adjacency[i].update(set(ids) - {i})
        subsets = []
        for choice in combinations(range(n), merge_size):
            subset = set(choice)
            reached = {choice[0]}
            for _ in range(merge_size):
                reached |= set().union(*(adjacency[i] for i in reached)) & subset
            extras = [i for i in range(n) if i not in subset and not (adjacency[i] & subset)]
            if reached == subset and (serial != 1 or extras):
                subsets.append((subset, extras))
        if not subsets:
            continue
        subset, extras = subsets[int(rng.integers(len(subsets)))]
        palette = [int(c) for c in rng.permutation(10) if c not in (bg, wall)]
        colors, cursor = [], 1
        for i in range(n):
            if i in subset:
                colors.append(palette[0])
            else:
                colors.append(palette[cursor])
                cursor += 1
        # Repeated color need not mean adjacency. Keep a disconnected copy.
        if extras and (serial == 1 or (not fallback and rng.random() < 0.5)):
            colors[int(rng.choice(extras))] = palette[0]
        opened, vessels = _vessels(tw, trooms, colors)
        if max(len(ids) for _, ids in vessels) != merge_size:
            continue
        doses = np.zeros(n, dtype=int)
        expected = np.full(tl.shape, bg, dtype=np.int8)
        expected[tw & ~opened] = wall
        good = True
        for mask, ids in vessels:
            capacities = np.array([drooms[i].sum() for i in ids], dtype=int)
            volumes = []
            for r in range(mask.shape[0]):
                wet = mask & (np.arange(mask.shape[0])[:, None] >= r)
                v = int(wet.sum())
                if (len(ids) <= v <= int(capacities.sum()) and
                        v < int(mask.sum()) and len(_components(wet)) == 1 and
                        (len(ids) == 1 or np.any(wet & opened))):
                    volumes.append(v)
            if not volumes:
                good = False
                break
            v = int(rng.choice(sorted(set(volumes))))
            allocations = _allocate(rng, v, capacities)
            for i, dose in zip(ids, allocations):
                doses[i] = dose
            expected[_fill(mask, v)] = colors[ids[0]]
        if not good:
            continue
        donor = np.full(dl.shape, bg, dtype=np.int8)
        donor[dw] = wall
        target = np.full(tl.shape, bg, dtype=np.int8)
        target[tw] = wall
        for i, room in enumerate(drooms):
            points = np.argwhere(room)
            style = serial % 4 if fallback else int(rng.integers(4))
            jitter = rng.random(len(points))
            if style == 0:
                score = points[:, 0] + jitter  # upper reservoir
            elif style == 1:
                score = jitter  # scattered dose
            elif style == 2:
                anchor = points[int(rng.integers(len(points)))]
                score = np.abs(points - anchor).sum(1) + jitter
            else:
                score = -points[:, 0] + jitter  # partly settled dose
            selected = points[np.argsort(score)[:doses[i]]]
            donor[tuple(selected.T)] = colors[i]
        try:
            inp = _embed(rng, donor, target, bg, serial, fallback)
            predicted = solve(inp)
            if not np.array_equal(predicted, expected):
                continue
            pair = {'input': inp, 'output': expected}
            stats = _measure(pair)
            if (not stats['redistribution'] or stats['merged'] != merge_size or
                    (serial == 1 and not stats['scan_shortcut_fails'])):
                continue
            return pair
        except ValueError:
            continue
    raise RuntimeError("bounded chamber construction failed")


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(rng, a, b):
    for value, low, high in ((a, 2, 10), (b, 1, 4)):
        if value is not None and (not _integer(value) or not low <= int(value) <= high):
            raise ValueError("illegal pair count")
    if a is None:
        a = int(rng.integers(max(2, (b or 0) + 1), 11))
    if b is None:
        b = int(rng.integers(1, min(4, a - 1) + 1))
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return int(a), int(b)


def _key(g):
    a = np.asarray(g, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    a, b = _counts(rng, num_train, num_test)
    puzzle, seen = {'train': [], 'test': []}, set()
    serial = 0
    first_roles = None
    for split, count in (('train', a), ('test', b)):
        for index in range(count):
            merge = 2 if split == 'train' else 3
            n = (3 + 2 * index if split == 'train' and index < 2 else
                 int(rng.integers(3 if split == 'train' else 4, 8)))
            for retry in range(20):
                if split == 'train' and index == 1:
                    bg = int(rng.choice([c for c in range(10) if c != first_roles[0]]))
                    wall = int(rng.choice([c for c in range(10) if c not in (bg, first_roles[1])]))
                else:
                    bg, wall = map(int, rng.choice(10, 2, replace=False))
                pair = _make_pair(rng, n, merge, serial, bg, wall)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("distinct-input construction exhausted")
            if split == 'train' and index == 0:
                first_roles = bg, wall
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
            serial += 1
    validate_generated_puzzle(puzzle)
    return puzzle


def _measure(pair):
    inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
    bg, wall, d, t, dr, tr, colors, doses, mapping, db, tb = _extract(inp)
    assert np.array_equal(out, solve(inp)), "wrong solved receiver"
    tc, td = [0] * len(tr), [0] * len(tr)
    for i, j in enumerate(mapping):
        tc[j], td[j] = colors[i], doses[i]
    opened, vessels = _vessels(t == wall, tr, tc)
    assert out.shape == t.shape
    assert np.array_equal(out == wall, (t == wall) & ~opened)
    assert np.all(out[(t == bg) & ~np.logical_or.reduce(tr)] == bg)
    assert opened.any() and np.any((out == bg) & np.logical_or.reduce(tr))
    redistribution = False
    for mask, ids in vessels:
        color = tc[ids[0]]
        wet = mask & (out == color)
        assert int(wet.sum()) == sum(td[i] for i in ids)
        assert 0 < int(wet.sum()) < int(mask.sum()), "every vessel retains dry space"
        assert np.array_equal(wet, _fill(mask, int(wet.sum())))
        # Opening cannot leak to exterior; every lower wet cell communicates.
        assert len(_components(wet)) == 1
        if len(ids) > 1:
            assert np.any(wet & opened)
            redistribution |= any(int((wet & tr[i]).sum()) != td[i] for i in ids)
    for color in set(colors):
        assert int((d == color).sum()) == int((out == color).sum())
    assert redistribution, "independent chamber doses accidentally explain output"
    areas = [int(m.sum()) for m in dr]
    target_areas = [int(tr[j].sum()) for j in mapping]
    r_overlap = min(db[2], tb[2]) > max(db[0], tb[0])
    separated_repeat = any(tc[a[1][0]] == tc[b[1][0]]
                           for a, b in combinations(vessels, 2))
    try:
        scan_answer = _receive(t, tr, colors, doses, tuple(range(len(dr))), bg, wall)
        scan_shortcut_fails = not np.array_equal(scan_answer, out)
    except ValueError:
        scan_shortcut_fails = True
    full_paint = t.copy()
    for i, j in enumerate(mapping):
        full_paint[tr[j]] = colors[i]
    assert not np.array_equal(full_paint, out), "ordinary B recoloring explains output"
    return {'rooms': len(dr), 'merged': max(len(ids) for _, ids in vessels),
            'background': bg, 'wall': wall, 'colors': len(set(colors)),
            'opened': int(opened.sum()), 'wet_opened': int(((out != bg) & (out != wall) & opened).sum()),
            'volume': sum(doses), 'redistribution': redistribution,
            'scan_changed': tuple(mapping) != tuple(range(len(dr))),
            'separated_repeat': separated_repeat,
            'scan_shortcut_fails': scan_shortcut_fails,
            'deformed': areas != target_areas,
            'nonrectangular': sum(int(m.sum()) != (_bbox(m)[2] - _bbox(m)[0]) *
                                  (_bbox(m)[3] - _bbox(m)[1]) for m in dr + tr),
            'layout': 'side' if r_overlap else 'stack',
            'source_components': len(_components((d != bg) & (d != wall))),
            'receiver_shape': t.shape,
            'structure': (t.shape, (t == wall).tobytes(), opened.tobytes())}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, stats = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for g in pair.values():
                g = np.asarray(g)
                assert g.ndim == 2 and np.issubdtype(g.dtype, np.integer)
                assert all(1 <= x <= 30 for x in g.shape) and np.all((g >= 0) & (g <= 9))
            key = _key(pair['input'])
            assert key not in seen
            seen.add(key)
            info = _measure(pair)
            assert info['merged'] == (2 if split == 'train' else 3)
            stats[split].append(info)
    assert stats['train'][0]['rooms'] == 3 and stats['train'][1]['rooms'] == 5
    assert stats['train'][1]['separated_repeat']
    assert stats['train'][1]['scan_shortcut_fails']
    for role in ('background', 'wall', 'structure'):
        assert stats['train'][0][role] != stats['train'][1][role]


def _signature(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def self_test():
    for n in range(3, 8):
        for merge in (2, 3):
            if n > merge:
                for seed in range(4):
                    _make_pair(np.random.default_rng(seed), n, merge, 0, 0, 1, True)
    _make_pair(np.random.default_rng(303), 5, 2, 1, 0, 1, True)
    records, structures, shapes, signatures = [], set(), set(), set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _signature(puzzle) == _signature(generate_puzzle(seed))
        signatures.add(_signature(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                m = _measure(pair)
                records.append(m)
                structures.add(m['structure'])
                shapes.add(pair['input'].shape)
    assert len(signatures) == 42 and len(structures) >= 150 and len(shapes) >= 35
    assert {m['rooms'] for m in records} == set(range(3, 8))
    assert {m['layout'] for m in records} == {'side', 'stack'}
    assert len({m['background'] for m in records}) >= 9
    assert len({m['wall'] for m in records}) >= 9
    assert len({m['volume'] for m in records}) >= 25
    assert sum(m['scan_changed'] for m in records) >= 5
    assert sum(m['nonrectangular'] > 0 for m in records) >= 50
    assert sum(m['deformed'] for m in records) >= 150
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(707, a, b))
    for args in ((None, None), (None, 4), (10, None), (2, None)):
        validate_generated_puzzle(generate_puzzle(99, *args))
    for args in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (2.5, 1), (True, 1), (4, False), ('4', 1)):
        try:
            generate_puzzle(0, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(12, 2, 1)
    bad['test'][0]['output'] = bad['test'][0]['output'].copy()
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output passed validation")
