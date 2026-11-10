"""Arrival loom: rail-facing enclosed rows become time-ordered square bands.

Only standard-library modules and NumPy are used. Solving reads pixels only.
The capped rail supplies orientation, tie order, and the central color.
"""

from __future__ import annotations

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad grid size")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(bool(np.all((0 <= a) & (a <= 9))), "bad color")
    return a.astype(np.int8, copy=False)


def _components(mask):
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    result = []
    for r, c in zip(*np.nonzero(mask)):
        r, c = int(r), int(c)
        if visited[r, c]:
            continue
        stack, cells = [(r, c)], []
        visited[r, c] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not visited[yy, xx]:
                    visited[yy, xx] = True
                    stack.append((yy, xx))
        result.append(cells)
    return result


def _holes(mask):
    h, w = mask.shape
    return [cells for cells in _components(~mask)
            if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells)]


def _profile(mask):
    holes = _holes(mask)
    active = [hole for hole in holes if any(c == mask.shape[1] - 2 for r, c in hole)]
    rows = {r for hole in active for r, c in hole}
    return holes, active, rows


def _scene(grid):
    """Canonical view: capped vertical rail, cap at top, objects on the left."""
    candidates = []
    for turn in range(4):
        a = np.rot90(grid, turn)
        h, w = a.shape
        if h < 4:
            continue
        for col in range(1, w - 1):
            rail, cap, bg = int(a[1, col]), int(a[0, col]), int(a[0, -1])
            if len({rail, cap, bg}) != 3:
                continue
            if not np.all(a[1:, col] == rail):
                continue
            if np.count_nonzero(a == rail) != h - 1 or np.count_nonzero(a == cap) != 1:
                continue
            if not np.all(a[:, col + 1:] == bg) or not np.any(a[:, :col] != bg):
                continue
            candidates.append((a, turn, bg, rail, cap, col))
    _require(len(candidates) == 1, "expected one uniquely oriented capped rail")
    return candidates[0]


def _inspect(input_grid):
    a, turn, bg, rail, cap, col = _scene(_grid(input_grid))
    objects = []
    for cells in _components(a[:, :col] != bg):
        rr, cc = np.array(cells).T
        r0, r1, c0, c1 = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[rr - r0, cc - c0] = True
        colors = set(map(int, a[rr, cc]))
        _require(len(colors) == 1, "objects must be monochrome")
        holes, active, rows = _profile(mask)
        for hole in active:
            _require({r for r, c in hole} == {r for r, c in hole if c == mask.shape[1] - 2},
                     "partially leading cavity is outside this family")
        enclosed = {p for hole in holes for p in hole}
        naive = {r for r in range(mask.shape[0]) if mask.shape[1] >= 3
                 and not mask[r, -2] and mask[r, -1]}
        objects.append(dict(row=r0, end=r1, left=c0, right=c1, cells=cells,
                            mask=mask, color=colors.pop(), gap=col - 1 - c1,
                            holes=holes, active=active, rows=rows,
                            open_rows={r for r in naive if (r, mask.shape[1] - 2) not in enclosed},
                            naive_rows=naive))
    objects.sort(key=lambda o: o['row'])
    _require(all(b['row'] > a['end'] + 1 for a, b in zip(objects, objects[1:])),
             "object travel bands must have a separating background row")
    _require(sum(len(o['rows']) for o in objects) <= 14, "too many emitted rows")
    return dict(array=a, turn=turn, bg=bg, rail=rail, cap=cap, col=col, objects=objects)


def _wrap(rail, events):
    result = np.full((2, 2), rail, dtype=np.int8)
    for color, thickness in events:
        for _ in range(thickness):
            result = np.pad(result, 1, constant_values=color)
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    scene = _inspect(input_grid)
    arrivals = sorted(scene['objects'], key=lambda o: (o['gap'], o['row']))
    return _wrap(scene['rail'], [(o['color'], len(o['rows'])) for o in arrivals])


def _edge_render(rail, events):
    """Validator's equivalent edge-distance construction, rather than padding."""
    layers = [color for color, thickness in reversed(events) for _ in range(thickness)] + [rail]
    side = 2 * len(layers)
    distances = np.minimum(np.arange(side), np.arange(side)[::-1])
    return np.array(layers, dtype=np.int8)[np.minimum(distances[:, None], distances[None, :])]


def _alternative(scene, mode):
    objects = scene['objects']
    arrivals = sorted(objects, key=lambda o: (o['gap'], o['row']))
    if mode == 'spatial':
        arrivals = objects
    elif mode == 'reverse_time':
        arrivals = arrivals[::-1]
    elif mode == 'reverse_tie':
        arrivals = sorted(objects, key=lambda o: (o['gap'], -o['row']))
    events = []
    for o in arrivals:
        q = len(o['rows'])
        if mode == 'one_per_object':
            q = int(bool(q))
        elif mode == 'one_per_hole':
            q = len(o['active'])
        elif mode == 'area':
            q = sum(map(len, o['active']))
        elif mode == 'all_holes':
            q = len({r for hole in o['holes'] for r, c in hole})
        elif mode == 'open_is_hole':
            q = len(o['naive_rows'])
        events.append((o['color'], q))
    return _edge_render(scene['rail'], events)


def _int(rng, low, high):
    return int(rng.integers(low, high + 1))


def _positive_partition(rng, total, count):
    return (1 + rng.multinomial(total - count, np.full(count, 1 / count))).tolist()


def _intervals(rng, height, count, total):
    sizes = _positive_partition(rng, total, count)
    walls = 1 + rng.multinomial(height - total - count - 1, np.full(count + 1, 1 / (count + 1)))
    row, result = int(walls[0]), []
    for i, size in enumerate(sizes):
        result.append((row, row + size))
        row += size + int(walls[i + 1])
    return result


def _carve(rng, mask, intervals, left, right, broad=False):
    # Stepped back edges are legal: every row still reaches the same front edge.
    for start, stop in intervals:
        fixed = _int(rng, left, right - int(broad))
        for r in range(start, stop):
            back = _int(rng, left, right - int(broad)) if rng.random() < .45 else fixed
            mask[r, back:right + 1] = False


def _erode(rng, mask):
    target_holes = {p for hole in _holes(mask) for p in hole}
    h, w = mask.shape
    def has_open_contact(m):
        return w >= 3 and any(not m[r, -2] and m[r, -1]
                              and (r, w - 2) not in target_holes for r in range(h))
    open_contact = has_open_contact(mask)
    for _ in range(_int(rng, 0, 7)):
        boundary = [(r, c) for r, c in zip(*np.nonzero(mask))
                    if r in (0, h - 1) or c in (0, w - 1)]
        r, c = boundary[_int(rng, 0, len(boundary) - 1)]
        trial = mask.copy()
        trial[r, c] = False
        if len(_components(trial)) != 1:
            continue
        rr, cc = np.nonzero(trial)
        if not (rr.min() == cc.min() == 0 and rr.max() == h - 1 and cc.max() == w - 1):
            continue
        if ({p for hole in _holes(trial) for p in hole} == target_holes
                and (not open_contact or has_open_contact(trial))):
            mask = trial
    return mask


def _mask(rng, kind, height, q=0, cavity_count=0, broad=False, rough=False):
    min_w = 6 if kind == 'mixed' else 4 if kind in ('tail', 'decoy') or broad else 2 if kind == 'solid' else 3
    w = _int(rng, min_w, 12 if kind == 'mixed' else 10)
    mask = np.ones((height, w), dtype=bool)
    if q:
        intervals = _intervals(rng, height, cavity_count, q)
        separator = _int(rng, 2, w - 3) if kind == 'mixed' else 0
        _carve(rng, mask, intervals, separator + 1, w - 2, broad=broad)
        if kind == 'mixed':
            count = _int(rng, 1, min(3, (height - 1) // 2))
            dq = _int(rng, count, height - count - 1)
            _carve(rng, mask, _intervals(rng, height, count, dq), 1, separator - 1)
        elif cavity_count > 1 and rough and rng.random() < .4:
            # Hollow lobes can be joined only along their extreme spine.
            for r in range(1, height - 1):
                if mask[r].all() and mask[r - 1].all() and mask[r + 1].all():
                    mask[r, :-1] = False
    elif kind in ('open', 'open_back'):
        mask[1:-1, 1:-1] = False
        side = int(rng.choice([0, 2, 3])) if kind == 'open_back' else _int(rng, 0, 3)
        if side < 2:
            mask[1:-1, 0 if side == 0 else -1] = False
        else:
            mask[0 if side == 2 else -1, -2] = False
    elif kind == 'tail':
        extension = _int(rng, 1, min(3, w - 3))
        body = w - extension
        mask[:, body:] = False
        count = _int(rng, 1, min(3, (height - 1) // 2))
        dq = _int(rng, count, height - count - 1)
        _carve(rng, mask[:, :body], _intervals(rng, height, count, dq), 1, body - 2)
        mask[0 if rng.random() < .5 else -1, body:] = True
    elif kind == 'decoy':
        count = _int(rng, 1, min(3, (height - 1) // 2))
        dq = _int(rng, count, height - count - 1)
        _carve(rng, mask, _intervals(rng, height, count, dq), 1, w - 3)
    elif kind == 'solid' and rough and rng.random() < .5:
        spine = _int(rng, 0, w - 1)
        for r in range(height):
            mask[r, :_int(rng, 0, spine)] = False
            mask[r, _int(rng, spine, w - 1) + 1:] = False
        # Keep the requested bounding width, without disconnecting the spine.
        mask[0, :spine + 1] = True
        mask[-1, spine:] = True
    if rough:
        mask = _erode(rng, mask)
    _require(len(_components(mask)) == 1, "builder disconnected an object")
    _require(len(_profile(mask)[2]) == q, "builder changed aperture quota")
    return mask


def _descriptors(rng, regime, demo=None, test=False):
    if demo == 'order_open':
        contributions = [_int(rng, 2, 3), _int(rng, 1, 2)]
        silents = ['open_back']
    elif demo == 'tie_tail':
        contributions = [_int(rng, 2, 3), _int(rng, 1, 2)]
        silents = ['tail']
    else:
        if test and regime == 'larger':
            total = _int(rng, 7, 14)
            count = _int(rng, 1, min(4, total))
        elif test and regime == 'smaller':
            total = _int(rng, 0, 2)
            count = _int(rng, 1, total) if total else 0
        elif test:
            total = _int(rng, 4, 8)
            count = _int(rng, 3, min(4, total))
        else:
            total = _int(rng, 3 if regime == 'smaller' else 0, 6)
            count = _int(rng, 1, min(2, total)) if total else 0
        contributions = _positive_partition(rng, total, count) if count else []
        silents = [str(rng.choice(['solid', 'open', 'tail', 'decoy']))
                   for _ in range(_int(rng, 0 if count else 1, 4))]
    descriptors = []
    for i, q in enumerate(contributions):
        # Minimal and tall multi-cavity objects both occur; count is not a size label.
        count = (1 if demo == 'order_open' and i == 0 else
                 2 if demo == 'tie_tail' and i == 0 else _int(rng, 1, min(3, q)))
        kind = 'mixed' if demo is None and rng.random() < .28 else 'active'
        descriptors.append(dict(kind=kind, q=q, count=count, minimum=q + count + 1,
                                broad=demo == 'order_open' and i == 0))
    descriptors += [dict(kind=k, q=0, count=0, minimum=2 if k == 'solid' else 3,
                         broad=False) for k in silents]
    def need():
        return sum(d['minimum'] for d in descriptors) + len(descriptors) + 1
    # Reduce optional silent objects and then optional cavity divisions, never q.
    while need() > 30 and any(d['q'] == 0 for d in descriptors):
        descriptors.pop(next(i for i in range(len(descriptors) - 1, -1, -1)
                             if descriptors[i]['q'] == 0))
    while need() > 30:
        d = max(descriptors, key=lambda d: d['count'])
        _require(d['count'] > 1, "quota cannot fit")
        d['count'] -= 1
        d['minimum'] -= 1
    return descriptors


def _make_input(rng, config, demo=None, test=False):
    descriptors = _descriptors(rng, config['regime'], demo, test)
    minimum = sum(d['minimum'] for d in descriptors) + len(descriptors) + 1
    available = 30 - minimum
    masks = []
    for d in descriptors:
        extra = _int(rng, 0, min(available, 4))
        available -= extra
        masks.append(_mask(rng, d['kind'], d['minimum'] + extra, d['q'], d['count'],
                           d['broad'], config['rough']))
    order = list(map(int, rng.permutation(len(masks))))
    masks = [masks[i] for i in order]
    descriptors = [descriptors[i] for i in order]
    base = sum(m.shape[0] for m in masks) + len(masks) + 1
    height = _int(rng, base, 30)
    extras = rng.multinomial(height - base, np.full(len(masks) + 1, 1 / (len(masks) + 1)))
    widest = max(m.shape[1] for m in masks)
    col = _int(rng, widest + 2, 28)
    band = _int(rng, 1, 29 - col)
    roles = config['roles'] if config['fixed_palette'] else tuple(map(int, rng.choice(10, 3, replace=False)))
    bg, rail, cap = roles
    available_colors = [c for c in range(10) if c not in roles]
    palette = config['palette'] if config['fixed_palette'] else list(map(int, rng.choice(
        available_colors, _int(rng, 2, 7), replace=False)))
    colors = list(map(int, rng.choice(palette, len(masks))))
    active_indices = [i for i, d in enumerate(descriptors) if d['q']]
    if len(active_indices) >= 2:
        colors[active_indices[0]], colors[active_indices[1]] = map(int, rng.choice(palette, 2, replace=False))
    gaps = [0 if rng.random() < .2 else _int(rng, 0, col - m.shape[1]) for m in masks]
    if demo == 'order_open':
        i, j = active_indices
        # The cap-nearer object arrives later: distance overrides spatial order.
        gaps[i] = _int(rng, 1, col - masks[i].shape[1])
        gaps[j] = _int(rng, 0, min(gaps[i] - 1, col - masks[j].shape[1]))
    elif demo == 'tie_tail':
        i, j = active_indices
        gaps[i] = gaps[j] = _int(rng, 0, min(col - masks[i].shape[1], col - masks[j].shape[1]))
    grid = np.full((height, col + 1 + band), bg, dtype=np.int8)
    grid[:, col] = rail
    grid[0, col] = cap
    row = 1 + int(extras[0])
    for i, mask in enumerate(masks):
        start = col - mask.shape[1] - gaps[i]
        view = grid[row:row + mask.shape[0], start:start + mask.shape[1]]
        view[mask] = colors[i]
        row += mask.shape[0] + 1 + int(extras[i + 1])
    return grid


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _unique_input(grid, seen):
    # At least four visible colors: 20 dihedral permutations of ten color names
    # are all distinct on this grid. At most 13 earlier inputs can be blocked.
    for sign in (1, -1):
        for offset in range(10):
            candidate = ((sign * grid.astype(int) + offset) % 10).astype(np.int8)
            if _key(candidate) not in seen:
                return candidate
    raise AssertionError("finite uniqueness orbit exhausted")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"illegal {name}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if a <= b:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(10, 3, replace=False)))
    palette = list(map(int, rng.choice([c for c in range(10) if c not in roles], _int(rng, 2, 7), replace=False)))
    config = dict(roles=roles, palette=palette, fixed_palette=bool(rng.random() < .4),
                  rough=bool(rng.random() < .7),
                  regime=str(rng.choice(['larger', 'smaller', 'recombine'])))
    base_turn = _int(rng, 0, 3)
    mixed = bool(rng.random() < .6)
    seen, puzzle = set(), {'train': [], 'test': []}
    for split, count in (('train', a), ('test', b)):
        for i in range(count):
            demo = ('order_open', 'tie_tail')[i] if split == 'train' and i < 2 else None
            inp = _make_input(rng, config, demo, split == 'test')
            turn = (base_turn + 1) % 4 if mixed and split == 'train' and i == 1 else _int(rng, 0, 3) if mixed and demo is None else base_turn
            inp = _unique_input(np.rot90(inp, turn).copy(), seen)
            seen.add(_key(inp))
            puzzle[split].append({'input': inp, 'output': solve(inp)})
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "bad puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "splits must be lists")
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "bad counts")
    seen, train_scenes = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "bad pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            scene = _inspect(inp)
            events = [(o['color'], len(o['rows'])) for o in sorted(
                scene['objects'], key=lambda o: (o['gap'], o['row']))]
            expected = _edge_render(scene['rail'], events)
            _require(np.array_equal(out, expected), "incorrect ordered frames")
            q = sum(t for c, t in events)
            _require(out.shape == (2 * (q + 1),) * 2, "incorrect emitted-row dimension")
            _require(np.all(out[q:q + 2, q:q + 2] == scene['rail']), "wrong central core")
            _require(set(map(int, np.unique(out))) == {scene['rail']} | {
                o['color'] for o in scene['objects'] if o['rows']}, "wrong material colors")
            _require(np.array_equal(out, out.T) and np.array_equal(out, out[::-1]), "broken concentric symmetry")
            if split == 'train':
                train_scenes.append((scene, out))
    # Collective demonstration requirements; these are not semantic restrictions
    # on an individual input (silent examples and all-solid inputs are legal).
    objects = [o for scene, out in train_scenes for o in scene['objects']]
    _require(any(o['open_rows'] for o in objects), "no enclosure contrast")
    _require(any(o['holes'] and not o['rows'] for o in objects), "no recessed contrast")
    _require(any(len(o['active']) >= 2 for o in objects), "no multi-cavity evidence")
    for mode in ('spatial', 'reverse_time', 'reverse_tie', 'one_per_object',
                 'one_per_hole', 'area', 'all_holes', 'open_is_hole'):
        _require(any(not np.array_equal(_alternative(scene, mode), out)
                     for scene, out in train_scenes), f"training permits shortcut: {mode}")


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def self_test():
    signatures, shapes, masks, colors, directions, quantities, counts = set(), set(), set(), set(), set(), set(), set()
    relations, silent, mixed_cavities, stepped, tied, zero = set(), 0, 0, 0, 0, 0
    first = None
    for seed in range(42):
        p = generate_puzzle(seed)
        first = p if first is None else first
        signatures.add(_puzzle_key(p))
        observed = [pair['output'].shape[0] for pair in p['train']]
        for split in ('train', 'test'):
            for pair in p[split]:
                scene = _inspect(pair['input'])
                objects = scene['objects']
                q = sum(len(o['rows']) for o in objects)
                shapes.add(pair['input'].shape)
                directions.add(scene['turn'])
                quantities.add(q)
                counts.add(len(objects))
                silent += q == 0
                colors.update(map(int, np.unique(pair['input'])))
                active = [o for o in objects if o['rows']]
                tied += any(x['gap'] == y['gap'] for i, x in enumerate(active) for y in active[i + 1:])
                zero += any(o['gap'] == 0 for o in objects)
                for o in objects:
                    masks.add((o['mask'].shape, o['mask'].tobytes()))
                    mixed_cavities += bool(o['active']) and len(o['holes']) > len(o['active'])
                    for hole in o['holes']:
                        rr, cc = zip(*hole)
                        stepped += len(hole) != (max(rr) - min(rr) + 1) * (max(cc) - min(cc) + 1)
                _require(np.array_equal(solve(pair['input'].copy()), pair['output']), "solve mismatch")
                if split == 'test':
                    side = pair['output'].shape[0]
                    relations.add('larger' if side > max(observed) else 'smaller' if side < min(observed) else 'overlap')
        _require(_puzzle_key(p) == _puzzle_key(generate_puzzle(seed)), "nondeterminism")
    _require(len(signatures) == 42 and len(shapes) >= 50 and len(masks) >= 120, "structural variation collapsed")
    _require(colors == set(range(10)) and directions == set(range(4)), "palette/orientation collapsed")
    _require({0, 1, 2, 3, 4, 7, 14} <= quantities, "emission range collapsed")
    _require({1, 2, 3, 4, 5} <= counts, "scene counts collapsed")
    _require(relations == {'larger', 'smaller', 'overlap'}, "shift modes collapsed")
    _require(silent >= 4 and mixed_cavities >= 5 and stepped >= 15 and tied >= 42 and zero >= 20,
             "meaningful family variation missing")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            generate_puzzle(42000 + 10 * a + b, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (2.5, 1), (3, False), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    # Small numeric witnesses: exact leading enclosure and the two size bounds.
    closed = np.zeros((6, 10), dtype=np.int8)
    closed[:, 7], closed[0, 7] = 5, 9
    closed[1:4, 2:5] = 2
    closed[2, 3] = 0
    expected = np.array([[2, 2, 2, 2], [2, 5, 5, 2],
                         [2, 5, 5, 2], [2, 2, 2, 2]], dtype=np.int8)
    opened = closed.copy()
    opened[2, 2] = 0  # An exterior doorway behind the same extreme wall.
    recessed = closed.copy()
    recessed[1, 5] = 2  # A tail beyond the closed chamber's local front wall.
    for turn in range(4):
        _require(np.array_equal(solve(np.rot90(closed, turn)), expected), "closed fixture")
        for silent_input in (opened, recessed):
            _require(np.array_equal(solve(np.rot90(silent_input, turn)), np.full((2, 2), 5)),
                     "open/recessed fixture must be silent")
    largest = np.zeros((18, 10), dtype=np.int8)
    largest[:, 7], largest[0, 7] = 5, 9
    largest[1:17, 3:7] = 2
    largest[2:16, 4:6] = 0
    big_out = solve(largest)
    _require(big_out.shape == (30, 30) and np.all(big_out[14:16, 14:16] == 5), "maximum quota fixture")
    _require(np.count_nonzero(big_out == 5) == 4 and np.count_nonzero(big_out == 2) == 896,
             "maximum fixture frame multiplicity")
    # Change only initial placement: arrival time must reorder material.
    travel = np.zeros((12, 12), dtype=np.int8)
    travel[:, 9], travel[0, 9] = 5, 9
    travel[1:4, 3:7] = 2
    travel[2, 5] = 0
    travel[5:9, 5:9] = 3
    travel[6:8, 6:8] = 0
    _require(np.array_equal(solve(travel), _edge_render(5, [(3, 2), (2, 1)])), "arrival fixture")
    farther = travel.copy()
    farther[5:9, :9] = 0
    farther[5:9, 2:6] = travel[5:9, 5:9]
    _require(np.array_equal(solve(farther), _edge_render(5, [(2, 1), (3, 2)])), "offset mutation fixture")
    _require(not np.array_equal(solve(travel), solve(farther)), "clearance was ignored")
    # Deliberate label corruption must fail family validation.
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in first[s]] for s in first}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
