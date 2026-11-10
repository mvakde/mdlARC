"""Rail-relative alignment and projection of enclosed leading cavities."""

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _normalize_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be an integer")
    return int(seed) % (1 << 64)


def _validate_supplied_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed_value, num_train, num_test):
    num_train = _validate_supplied_count(
        num_train, "num_train", 2, 10
    )
    num_test = _validate_supplied_count(
        num_test, "num_test", 1, 4
    )

    if num_train is None and num_test is None:
        num_test = 1 + seed_value % 4
        low = max(2, num_test + 1)
        num_train = low + (seed_value // 4) % (11 - low)
    elif num_train is None:
        low = max(2, num_test + 1)
        num_train = (
            low + (seed_value + seed_value // 7) % (11 - low)
        )
    elif num_test is None:
        high = min(4, num_train - 1)
        num_test = 1 + seed_value % high

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return int(num_train), int(num_test)


def _randint_inclusive(rng, low, high):
    return int(rng.integers(low, high + 1))


def _components(mask):
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    result = []

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue

            seen[row, col] = True
            stack = [(row, col)]
            component = []

            while stack:
                r, c = stack.pop()
                component.append((r, c))

                for nr, nc in (
                    (r - 1, c),
                    (r + 1, c),
                    (r, c - 1),
                    (r, c + 1),
                ):
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and mask[nr, nc]
                        and not seen[nr, nc]
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            result.append(component)

    return result


def _hole_components(local_foreground):
    height, width = local_foreground.shape
    holes = []

    for component in _components(~local_foreground):
        touches_boundary = any(
            r == 0
            or r == height - 1
            or c == 0
            or c == width - 1
            for r, c in component
        )
        if not touches_boundary:
            holes.append(component)

    return holes


def _mask_profile(mask):
    active, decoy = [], []
    for hole in _hole_components(mask):
        target = active if any(c == mask.shape[1] - 2 for r, c in hole) else decoy
        target.append(hole)
    return active, decoy


def _crop(mask):
    rows, cols = np.nonzero(mask)
    return mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1].copy()


def _scene(grid):
    """Infer direction and visible colors from pixels, never a sampled palette."""
    candidates = []
    for turn in range(4):
        canonical = np.rot90(grid, turn)
        h, w = canonical.shape
        for col in range(1, w - 1):
            rail = int(canonical[0, col])
            if not np.all(canonical[:, col] == rail):
                continue
            if np.count_nonzero(canonical == rail) != h:
                continue
            bg = int(canonical[0, -1])
            if bg == rail or not np.all(canonical[:, col + 1:] == bg):
                continue
            work = canonical[:, :col]
            if not np.any(work != bg):
                continue
            candidates.append((canonical, turn, bg, rail, col))
    _require(len(candidates) == 1, "rail and blank projection side must be unique")
    return candidates[0]


def _details(canonical, bg, rail_col):
    result = []
    for cells in _components(canonical[:, :rail_col] != bg):
        rows, cols = zip(*cells)
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in cells:
            mask[r - r0, c - c0] = True
        colors = {int(canonical[r, c]) for r, c in cells}
        _require(len(colors) == 1, "each connected object must be monochrome")
        active, decoy = _mask_profile(mask)
        active_rows = {r for hole in active for r, c in hole}
        naive = {r for r in range(1, mask.shape[0] - 1)
                 if mask.shape[1] >= 3 and not mask[r, -2]}
        holes = active + decoy
        enclosed = {p for hole in holes for p in hole}
        open_rows = {r for r in naive if (r, mask.shape[1] - 2) not in enclosed}
        result.append(dict(cells=cells, mask=mask, color=colors.pop(), row=r0,
                           end_row=r1, shift=rail_col - 1 - c1, active=active,
                           decoy=decoy, active_rows=active_rows, naive_rows=naive,
                           open_rows=open_rows))
    return result


def _apply_rule(grid, signal):
    canonical, turn, bg, rail, col = _scene(grid)
    result = canonical.copy()
    result[:, :col] = bg
    for obj in _details(canonical, bg, col):
        for r, c in obj['cells']:
            result[r, c + obj['shift']] = obj['color']
        for r in obj['active_rows']:
            result[obj['row'] + r, col + 1:] = signal
    return np.rot90(result, -turn).copy()


def _learn_signal(train_pairs):
    new = set()
    for pair in train_pairs:
        new.update(set(map(int, np.unique(pair['output']))) -
                   set(map(int, np.unique(pair['input']))))
    _require(len(new) == 1, "training must establish one output-only signal color")
    signal = new.pop()
    _require(all(not np.any(p['input'] == signal) for p in train_pairs),
             "signal must be absent from inputs")
    return signal


def _predict(grid, signal):
    """Second implementation: exterior reachability by array dilation.

    No object-builder state, hole-component helper, or query output is used.
    Full-row contact means row-local and whole-cavity readings coincide.
    """
    canonical, turn, bg, rail, col = _scene(grid)
    result = canonical.copy()
    result[:, :col] = bg
    for cells in _components(canonical[:, :col] != bg):
        rr, cc = np.array(cells).T
        r0, r1, c0, c1 = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())
        occupied = np.zeros((r1 - r0 + 3, c1 - c0 + 3), dtype=bool)
        occupied[rr - r0 + 1, cc - c0 + 1] = True
        exterior = np.zeros_like(occupied)
        exterior[0, :] = exterior[-1, :] = True
        exterior[:, 0] = exterior[:, -1] = True
        for _ in range(occupied.size):
            spread = exterior.copy()
            spread[1:] |= exterior[:-1]
            spread[:-1] |= exterior[1:]
            spread[:, 1:] |= exterior[:, :-1]
            spread[:, :-1] |= exterior[:, 1:]
            spread &= ~occupied
            if np.array_equal(spread, exterior):
                break
            exterior = spread
        enclosed = ~occupied & ~exterior
        # The padded object's extreme foreground column is -2; behind it is -3.
        active_rows = np.flatnonzero(enclosed[:, -3]) + r0 - 1
        result[rr, cc + col - 1 - c1] = canonical[rr, cc]
        result[active_rows, col + 1:] = signal
    return np.rot90(result, -turn).copy()


def solve(train_pairs, test_input):
    return _predict(test_input.copy(), _learn_signal(train_pairs))


def _intervals(rng, height, count):
    """Random separated intervals; sizes and positions vary independently."""
    if not count:
        return []
    spare = height - (2 * count + 1)
    bins = rng.multinomial(spare, np.full(2 * count + 1, 1 / (2 * count + 1)))
    row = 1 + int(bins[0])
    intervals = []
    for i in range(count):
        length = 1 + int(bins[2 * i + 1])
        intervals.append((row, row + length))
        row += length + 1 + int(bins[2 * i + 2])
    return intervals


def _cut_holes(mask, rng, count, left, right, stepped):
    for start, stop in _intervals(rng, mask.shape[0], count):
        back = _randint_inclusive(rng, left, right)
        for row in range(start, stop):
            edge = _randint_inclusive(rng, left, right) if stepped else back
            mask[row, edge:right + 1] = False


def _roughen(mask, rng):
    """Erode outlines while preserving connectivity and every cavity pixel."""
    target = {p for hole in _hole_components(mask) for p in hole}
    for _ in range(_randint_inclusive(rng, 2, 12)):
        h, w = mask.shape
        side = int(rng.integers(4))
        r = int(rng.integers(h)) if side < 2 else (0 if side == 2 else h - 1)
        c = (0 if side == 0 else w - 1) if side < 2 else int(rng.integers(w))
        if not mask[r, c]:
            continue
        trial = mask.copy()
        trial[r, c] = False
        if not trial.any() or len(_components(trial)) != 1:
            continue
        if set(np.nonzero(trial)[0]) != set(range(h)) or not trial[:, -1].any():
            continue
        actual = {p for hole in _hole_components(trial) for p in hole}
        if actual == target:
            mask = trial
    return _crop(mask)


def _make_object(rng, kind, max_height, max_width, complexity):
    minimum = 2 if kind == 'solid' else 5 if kind == 'multi' else 3
    height = _randint_inclusive(rng, minimum, min(max_height, 16))
    min_width = 2 if kind == 'solid' else 6 if kind == 'mixed' else 4 if kind == 'decoy' else 3
    width = _randint_inclusive(rng, min_width, max(max_width, min_width))
    mask = np.ones((height, width), dtype=bool)
    slots = min(4, (height - 1) // 2)
    stepped = complexity > 0 and rng.random() < .6
    if kind == 'solid':
        if complexity and width > 2 and height > 2 and rng.random() < .6:
            # A simply connected staircase / branching silhouette.
            spine = int(rng.integers(width))
            for row in range(height):
                left = int(rng.integers(spine + 1))
                right = _randint_inclusive(rng, spine, width - 1)
                mask[row, :left] = False
                mask[row, right + 1:] = False
    elif kind == 'open':
        mask[1:-1, 1:-1] = False
        side = int(rng.integers(4))
        if side < 2:
            mask[1:-1, 0 if side == 0 else -1] = False
        else:
            # A top/bottom doorway reaching the extreme-adjacent column.
            mask[0 if side == 2 else -1, -2] = False
    elif kind == 'decoy':
        if rng.random() < .6:
            # Recessed lobe with a one-cell-thick protrusion defining alignment.
            extension = _randint_inclusive(rng, 1, min(3, width - 3))
            body = width - extension
            mask[:, body:] = False
            _cut_holes(mask[:, :body], rng, _randint_inclusive(rng, 1, slots),
                       1, body - 2, stepped)
            row = int(rng.choice([0, height - 1]))
            mask[row, body:] = True
        else:
            _cut_holes(mask, rng, _randint_inclusive(rng, 1, slots),
                       1, _randint_inclusive(rng, 1, width - 3), stepped)
    elif kind == 'mixed':
        separator = _randint_inclusive(rng, 2, width - 3)
        _cut_holes(mask, rng, _randint_inclusive(rng, 1, slots),
                   separator + 1, width - 2, stepped)
        _cut_holes(mask, rng, _randint_inclusive(rng, 1, slots),
                   1, separator - 1, stepped)
    else:
        count = _randint_inclusive(rng, 2 if kind == 'multi' else 1,
                                  slots if kind == 'multi' or complexity == 2 else 1)
        _cut_holes(mask, rng, count, 1, width - 2, stepped)
        if complexity == 2 and count >= 2 and rng.random() < .6:
            # Hollow lobes connected along the extreme wall by a thin spine.
            for row in range(1, height - 1):
                if mask[row].all() and mask[row - 1].all() and mask[row + 1].all():
                    mask[row, :-1] = False
    if complexity and rng.random() < .7:
        mask = _roughen(mask, rng)
    return _crop(mask)


def _contrasts(rng):
    # Three near-matched bodies: a closed hole, an open doorway, and a tail.
    h, w = _randint_inclusive(rng, 4, 7), _randint_inclusive(rng, 3, 8)
    closed = np.ones((h, w), dtype=bool)
    closed[1:-1, 1:-1] = False
    opened = closed.copy()
    side = int(rng.integers(3))
    if side == 0:
        opened[1:-1, 0] = False
    else:
        opened[0 if side == 1 else -1, -2] = False
    extension = _randint_inclusive(rng, 1, 3)
    recessed = np.zeros((h, w + extension), dtype=bool)
    recessed[:, :w] = closed
    recessed[int(rng.choice([0, h - 1])), w:] = True
    if rng.random() < .5:
        closed, opened, recessed = (np.flipud(m).copy() for m in (closed, opened, recessed))
    return closed, opened, recessed


def _generate_pair(rng, regime, config, required, turn, pair_index):
    is_test = regime == 'target_test'
    relation, complexity = config['relation'], config['complexity']
    if relation == 'source_shift':
        low_h, high_h = (25, 30) if is_test else (6, 24)
    elif relation == 'smaller':
        low_h, high_h = (5, 18) if is_test else (22, 30)
    elif relation == 'equal':
        low_h, high_h = 12, 30
    else:
        low_h, high_h = 5, 30
    mandatory = [m.copy() for m in required]
    kinds = []
    if is_test and relation == 'source_shift':
        kinds = ['multi', 'decoy']
    silent = not mandatory and not kinds and rng.random() < .19
    max_count = 3 if complexity == 0 else 9
    count = max(len(mandatory) + len(kinds), _randint_inclusive(rng, 1, max_count))
    pool = ['solid', 'solid', 'open', 'decoy'] if silent else (
        ['solid', 'open', 'decoy', 'active', 'active', 'mixed', 'multi'])
    if complexity == 0:
        pool = ['solid', 'open', 'decoy'] if silent else ['solid', 'active', 'open']
    if relation == 'source_shift' and not is_test:
        pool = [k for k in pool if k not in ('multi', 'mixed')]
    while len(mandatory) + len(kinds) < count:
        kinds.append(str(rng.choice(pool)))
    def minimum(k):
        return 2 if k == 'solid' else 5 if k == 'multi' else 3
    def needed():
        return sum(m.shape[0] for m in mandatory) + sum(minimum(k) for k in kinds) + len(mandatory) + len(kinds)
    while needed() > high_h:
        kinds.pop()
    # One spare row ensures a solid object cannot impersonate the rail.
    budget = high_h - len(mandatory) - len(kinds) - sum(m.shape[0] for m in mandatory)
    masks = mandatory[:]
    rng.shuffle(kinds)
    for i, kind in enumerate(kinds):
        available = budget - sum(minimum(k) for k in kinds[i + 1:])
        effective = min(complexity, 1) if relation == 'source_shift' and not is_test else complexity
        mask = _make_object(rng, kind, available, 7 if complexity == 0 else 13, effective)
        masks.append(mask)
        budget -= mask.shape[0]
    rng.shuffle(masks)
    count = len(masks)
    base = sum(m.shape[0] for m in masks) + count - 1
    height = _randint_inclusive(rng, max(low_h, base + 1), high_h)
    widest = max(m.shape[1] for m in masks)
    width = _randint_inclusive(rng, widest + 3, 30)
    rail_col = _randint_inclusive(rng, widest + 1, width - 2)
    signal = config['signal']
    if config['color_mode'] == 'fixed':
        bg, rail, fg = config['roles']
    else:
        available_colors = [c for c in range(10) if c != signal]
        bg, rail, fg = map(int, rng.choice(available_colors, 3, replace=False))
        # A deterministic, visible distinction prevents accidental palette collapse.
        if (bg, rail) == config['previous_roles']:
            bg, rail = rail, bg
        config['previous_roles'] = (bg, rail)
    object_colors = [c for c in range(10) if c not in (signal, bg, rail)]
    grid = np.full((height, width), bg, dtype=np.int8)
    grid[:, rail_col] = rail
    extras = rng.multinomial(height - base, np.full(count + 1, 1 / (count + 1)))
    row = int(extras[0])
    shared_shift = int(rng.integers(rail_col - widest + 1)) if rng.random() < .16 else None
    shifts = []
    for mask in masks:
        limit = rail_col - mask.shape[1]
        shift = (shared_shift if shared_shift is not None else
                 0 if rng.random() < .25 else _randint_inclusive(rng, 1, limit))
        shifts.append(shift)
    if not is_test and not any(shifts):
        index = int(rng.integers(count))
        shifts[index] = _randint_inclusive(rng, 1, rail_col - masks[index].shape[1])
    for i, (mask, shift) in enumerate(zip(masks, shifts)):
        color = fg if config['color_mode'] != 'multi' else int(rng.choice(object_colors))
        start = rail_col - mask.shape[1] - shift
        view = grid[row:row + mask.shape[0], start:start + mask.shape[1]]
        view[mask] = color
        row += mask.shape[0] + 1 + int(extras[i + 1])
    # Orientation is an equivariant extension; every scene element turns together.
    grid = np.rot90(grid, turn).copy()
    return {'input': grid, 'output': _apply_rule(grid, signal)}


def _grid_signature(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _puzzle_signature(puzzle):
    return tuple(tuple((_grid_signature(p['input']), _grid_signature(p['output']))
                       for p in puzzle[s]) for s in ('train', 'test'))


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed_value = _normalize_seed(seed)
    train_count, test_count = _resolve_counts(seed_value, num_train, num_test)
    rng = np.random.default_rng(seed_value)
    signal = int(rng.integers(10))
    config = dict(signal=signal,
                  complexity=int(rng.choice(3, p=[.24, .40, .36])),
                  relation=str(rng.choice(['source_shift', 'shared', 'smaller', 'equal'],
                                          p=[.25, .35, .20, .20])),
                  color_mode=str(rng.choice(['fixed', 'vary', 'multi'], p=[.25, .40, .35])),
                  roles=tuple(map(int, rng.choice([c for c in range(10) if c != signal],
                                                 3, replace=False))),
                  previous_roles=None)
    orientation_mode = str(rng.choice(['fixed', 'shift', 'mixed'], p=[.4, .25, .35]))
    base_turn = int(rng.integers(4))
    other_turn = (base_turn + _randint_inclusive(rng, 1, 3)) % 4
    turns = [base_turn] * train_count + [other_turn if orientation_mode == 'shift'
                                        else base_turn] * test_count
    if orientation_mode == 'mixed':
        turns = list(map(int, rng.integers(4, size=train_count + test_count)))
        turns[1] = (turns[0] + _randint_inclusive(rng, 1, 3)) % 4
        if test_count > 1:
            turns[train_count + 1] = (turns[train_count] + _randint_inclusive(rng, 1, 3)) % 4
    requirements = [[] for _ in range(train_count)]
    order = list(map(int, rng.permutation(train_count)))
    for i, mask in enumerate(_contrasts(rng)):
        requirements[order[i % train_count]].append(mask)
    puzzle = {'train': [], 'test': []}
    seen = set()
    index = 0
    source_keys = {(_grid_signature(p['input']), _grid_signature(p['output']))
                   for p in _source_pairs()}
    for split, count, regime in [('train', train_count, 'source_train'),
                                  ('test', test_count, 'target_test')]:
        for i in range(count):
            pair = _generate_pair(rng, regime, config,
                                  requirements[i] if split == 'train' else [], turns[index], i)
            pair = _ensure_unique(pair, signal, seen, source_keys)
            seen.add(_grid_signature(pair['input']))
            puzzle[split].append(pair)
            index += 1
    validate_generated_puzzle(puzzle)
    return puzzle


def _ensure_unique(pair, signal, seen, source_keys):
    # The 18 rotations/reflections of nine visible colors give distinct inputs:
    # an odd cycle's reflection fixes at most one color, and scenes have >=3.
    # At most 13 prior inputs + 4 source pairs can be blocked, so this is bounded
    # construction with a guaranteed fallback, with no excluded source palette.
    colors = [c for c in range(10) if c != signal]
    for sign in (1, -1):
        for offset in range(9):
            lookup = np.arange(10, dtype=np.int8)
            for index, color in enumerate(colors):
                lookup[color] = colors[(sign * index + offset) % 9]
            candidate = {k: lookup[v] for k, v in pair.items()}
            ikey = _grid_signature(candidate['input'])
            if ikey not in seen and (ikey, _grid_signature(candidate['output'])) not in source_keys:
                return candidate
    raise AssertionError('color orbit exhausted despite <=17 excluded candidates')


def _check_grid(grid):
    _require(isinstance(grid, np.ndarray), 'grid must be a NumPy array')
    _require(grid.ndim == 2 and all(1 <= d <= 30 for d in grid.shape), 'invalid grid dimensions')
    _require(np.issubdtype(grid.dtype, np.integer), 'grid must have integer dtype')
    _require(bool(np.all((grid >= 0) & (grid <= 9))), 'invalid ARC color')


def _check_pair(pair, signal):
    _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair schema')
    inp, out = pair['input'], pair['output']
    _check_grid(inp)
    _check_grid(out)
    _require(inp.shape == out.shape, 'dimensions must be preserved')
    _require(not np.any(inp == signal), 'signal color occurs in input')
    canonical, turn, bg, rail, col = _scene(inp)
    objects = _details(canonical, bg, col)
    previous = -2
    for obj in objects:
        _require(obj['row'] > previous + 1, 'object row bands must remain disjoint')
        previous = obj['end_row']
        _require(obj['shift'] >= 0, 'object would cross rail')
        for hole in obj['active']:
            hole_rows = {r for r, c in hole}
            contact_rows = {r for r, c in hole if c == obj['mask'].shape[1] - 2}
            _require(hole_rows == contact_rows, 'ambiguous partially leading cavity')
    _require(np.array_equal(out, _predict(inp, signal)), 'output violates pixel-derived rule')
    # Comparing both algorithms also guards accidental implementation divergence.
    _require(np.array_equal(out, _apply_rule(inp, signal)), 'cavity projection disagrees')
    return objects


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'invalid pair counts')
    for split in ('train', 'test'):
        for p in puzzle[split]:
            _require(isinstance(p, dict) and set(p) == {'input', 'output'}, 'invalid pair schema')
            _check_grid(p['input'])
            _check_grid(p['output'])
    signal = _learn_signal(puzzle['train'])
    seen = set()
    source = {(_grid_signature(p['input']), _grid_signature(p['output'])) for p in _source_pairs()}
    training_objects = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            key = _grid_signature(pair['input'])
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            _require((key, _grid_signature(pair['output'])) not in source, 'exact source copy')
            objects = _check_pair(pair, signal)
            if split == 'train':
                training_objects.extend(objects)
            else:
                _require(np.array_equal(solve(puzzle['train'], pair['input']), pair['output']),
                         'query is not recoverable from training and input')
    _require(any(o['active'] for o in training_objects), 'no positive training example')
    _require(any(o['open_rows'] for o in training_objects), 'training lacks deep open contrast')
    _require(any(o['decoy'] and not o['active'] for o in training_objects), 'training lacks recessed contrast')
    _require(any(o['shift'] > 0 for o in training_objects), 'training never demonstrates motion')
    _require(any(len({r for r, c in h}) >= 2 for o in training_objects for h in o['active']),
             'training must demonstrate projection of all cavity rows')


def measure_pair(pair):
    """Review measurements reconstructed from returned pixels, without metadata."""
    grid, turn, bg, rail, col = _scene(pair['input'])
    objects = _details(grid, bg, col)
    holes = [hole for o in objects for hole in o['active'] + o['decoy']]
    open_sides = set()
    back_walls, cross_walls = set(), set()
    for obj in objects:
        mask = obj['mask']
        for hole in obj['active'] + obj['decoy']:
            for row in {r for r, c in hole}:
                c = min(c for r, c in hole if r == row) - 1
                thickness = 0
                while c >= 0 and mask[row, c]:
                    thickness += 1
                    c -= 1
                back_walls.add(thickness)
            for col in {c for r, c in hole}:
                occupied_rows = [r for r, c in hole if c == col]
                for start, step in ((min(occupied_rows) - 1, -1),
                                    (max(occupied_rows) + 1, 1)):
                    row, thickness = start, 0
                    while 0 <= row < mask.shape[0] and mask[row, col]:
                        thickness += 1
                        row += step
                    cross_walls.add(thickness)
        for region in _components(~mask):
            if not any(r in obj['open_rows'] and c == mask.shape[1] - 2 for r, c in region):
                continue
            for r, c in region:
                if r == 0:
                    open_sides.add('top')
                if r == mask.shape[0] - 1:
                    open_sides.add('bottom')
                if c == 0:
                    open_sides.add('back')
                if c == mask.shape[1] - 1:
                    open_sides.add('front')
    return dict(shape=tuple(pair['input'].shape), canonical_shape=tuple(grid.shape),
                direction=(-turn) % 4, background=bg, rail=rail,
                colors=tuple(sorted({o['color'] for o in objects})), work_width=col,
                band=grid.shape[1] - col - 1, count=len(objects),
                shifts=tuple(o['shift'] for o in objects),
                active_counts=tuple(len(o['active']) for o in objects),
                decoy_counts=tuple(len(o['decoy']) for o in objects),
                object_shapes=tuple(o['mask'].shape for o in objects),
                hole_shapes=tuple((max(r for r,c in h)-min(r for r,c in h)+1,
                                   max(c for r,c in h)-min(c for r,c in h)+1) for h in holes),
                stepped_holes=sum(len(h) != (max(r for r,c in h)-min(r for r,c in h)+1) *
                                  (max(c for r,c in h)-min(c for r,c in h)+1) for h in holes),
                open_sides=tuple(sorted(open_sides)),
                back_wall_thicknesses=tuple(sorted(back_walls)),
                cross_wall_thicknesses=tuple(sorted(cross_walls)),
                gaps=tuple(b['row'] - a['end_row'] - 1 for a, b in zip(objects, objects[1:])),
                thin_tail_decoys=sum(bool(o['decoy']) and not o['active'] and
                                     np.count_nonzero(o['mask'][:, -1]) <= 2 for o in objects),
                mixed_hole_objects=sum(bool(o['active']) and bool(o['decoy']) for o in objects),
                lobed_objects=sum(o['mask'].shape[0] >= 7 and any(
                    o['mask'][r].sum() == 1 and o['mask'][r - 1].sum() >= 2 and
                    o['mask'][r + 1].sum() >= 2 for r in range(1, o['mask'].shape[0] - 1))
                    for o in objects),
                open_objects=sum(bool(o['open_rows']) for o in objects),
                irregular_objects=sum(not (o['mask'][0].all() and o['mask'][-1].all() and
                                           o['mask'][:, 0].all() and o['mask'][:, -1].all()) for o in objects),
                shortcut_failure=any(o['active_rows'] != o['naive_rows'] for o in objects),
                silent=not any(o['active'] for o in objects),
                unchanged=bool(np.array_equal(pair['input'], pair['output'])),
                masks=tuple((o['mask'].shape, o['mask'].tobytes()) for o in objects))


def self_test() -> None:
    source = _source_pairs()
    for pair in source:
        _check_pair(pair, 2)
    _require(np.array_equal(solve(source[:3], source[3]['input']), source[3]['output']),
             'source query regression')
    _require(sum(measure_pair(p)['shortcut_failure'] for p in source) == 3,
             'source enclosure evidence changed')
    records, puzzles = [], []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        records.extend(measure_pair(p) for s in ('train', 'test') for p in puzzle[s])
    _require(len({_puzzle_signature(p) for p in puzzles}) == 42, 'cross-seed duplication')
    _require(len({len(p['train']) for p in puzzles}) >= 4 and
             len({len(p['test']) for p in puzzles}) >= 3, 'automatic counts collapsed')
    _require(len({r['shape'] for r in records}) >= 60, 'dimensions collapsed')
    _require(len({m for r in records for m in r['masks']}) >= 150, 'object geometry collapsed')
    _require({r['direction'] for r in records} == set(range(4)), 'missing orientation')
    _require({1, 2, 3, 4, 5, 6, 7}.issubset({r['count'] for r in records}), 'object counts collapsed')
    _require(any(30 in r['shape'] for r in records) and any(r['band'] == 1 for r in records),
             'missing wide grid or one-cell band')
    _require(sum(r['shortcut_failure'] for r in records) >= 80, 'enclosure shortcut still works')
    _require(sum(r['silent'] for r in records) >= 10, 'missing silent pairs')
    _require(sum(r['stepped_holes'] for r in records) >= 20, 'cavities are only rectangles')
    _require(sum(r['irregular_objects'] for r in records) >= 150, 'outlines collapsed')
    _require(any(max(r['active_counts']) >= 3 for r in records), 'no three-cavity object')
    _require(any(max(r['decoy_counts']) >= 3 for r in records), 'no several-decoy object')
    _require(any(any(w >= 3 for h, w in r['hole_shapes']) for r in records), 'holes too narrow')
    _require(any(any(h == 5 and a == 2 for (h, w), a in zip(r['object_shapes'], r['active_counts']))
                 for r in records), 'no compact double cavity')
    _require(any((2, 2) in r['object_shapes'] for r in records), 'no small solid')
    _require(any(any(h >= 8 and a == 0 for (h, w), a in zip(r['object_shapes'], r['active_counts']))
                 for r in records), 'missing tall no-active object')
    _require(any(any(h >= 8 and a == 1 for (h, w), a in zip(r['object_shapes'], r['active_counts']))
                 for r in records), 'missing tall single cavity')
    _require(sum(0 in r['shifts'] and max(r['shifts']) > 0 for r in records) >= 30,
             'missing zero/positive mixtures')
    _require(any(len(r['shifts']) > 1 and len(set(r['shifts'])) == 1 for r in records),
             'equal-distance composition absent')
    _require(len({r['shifts'][0] for r in records}) >= 8, 'top-object placement schedule')
    _require(len({r['shifts'][1] for r in records if len(r['shifts']) > 1}) >= 8,
             'second-object placement schedule')
    _require(sum(len(r['colors']) > 1 for r in records) >= 20, 'multicolor objects absent')
    within = [list(map(measure_pair, p['train'] + p['test'])) for p in puzzles]
    _require(sum(len({(r['background'], r['rail']) for r in rr}) > 1 for rr in within) >= 15,
             'visible roles never vary within puzzles')
    _require(sum(len({r['direction'] for r in rr}) == 1 for rr in within) >= 5,
             'no simple fixed-orientation puzzles')
    _require(sum(len({r['direction'] for r in rr}) > 1 for rr in within) >= 10,
             'no orientation variation within puzzles')
    _require(sum(r['irregular_objects'] > 0 and max(r['active_counts']) >= 2 and
                 len(r['colors']) > 1 for r in records) >= 5, 'axes do not combine')
    larger = smaller = 0
    for p in puzzles:
        train = [measure_pair(x) for x in p['train']]
        test = [measure_pair(x) for x in p['test']]
        larger += min(r['canonical_shape'][0] for r in test) > max(r['canonical_shape'][0] for r in train)
        smaller += max(r['canonical_shape'][0] for r in test) < min(r['canonical_shape'][0] for r in train)
    _require(larger >= 3 and smaller >= 3, 'train/test size relationships collapsed')
    _require(_puzzle_signature(generate_puzzle(99173, 7, 3)) ==
             _puzzle_signature(generate_puzzle(99173, 7, 3)), 'nondeterministic generator')
    for seed, a, b in [(2000, 2, 1), (2001, 10, 1), (2002, 10, 4),
                        (3000, None, 1), (3001, None, 4), (3002, 2, None), (3003, 10, None)]:
        p = generate_puzzle(seed, a, b)
        _require((a is None or len(p['train']) == a) and (b is None or len(p['test']) == b),
                 'requested count ignored')
    for a, b in [(1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (3.5, 1)]:
        try:
            generate_puzzle(4000, a, b)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    _require({side for r in records for side in r['open_sides']} ==
             {'top', 'bottom', 'back', 'front'}, 'missing opening direction')
    _require(len({v for r in records for v in r['back_wall_thicknesses']}) >= 5 and
             len({v for r in records for v in r['cross_wall_thicknesses']}) >= 5,
             'wall thickness away from the active face collapsed')
    _require(len({v for r in records for v in r['gaps']}) >= 4 and
             min(v for r in records for v in r['gaps']) >= 1, 'spacing collapsed or objects touch')
    _require(sum(r['thin_tail_decoys'] for r in records) >= 50, 'recessed tails absent')
    _require(sum(r['mixed_hole_objects'] for r in records) >= 30, 'mixed cavities absent')
    _require(sum(r['lobed_objects'] for r in records) >= 3, 'connected lobes absent')
    _require({_learn_signal(p['train']) for p in puzzles} == set(range(10)), 'signal colors collapsed')
    _require({r['background'] for r in records} == set(range(10)) and
             {r['rail'] for r in records} == set(range(10)), 'visible colors collapsed')
    shift_regimes = mixed_regimes = 0
    count_relations = [0, 0, 0]
    for p in puzzles:
        a = [measure_pair(pair) for pair in p['train']]
        b = [measure_pair(pair) for pair in p['test']]
        da, db = {r['direction'] for r in a}, {r['direction'] for r in b}
        shift_regimes += len(da) == len(db) == 1 and da != db
        mixed_regimes += len(da) > 1 and len(db) > 1
        count_relations[0] += any(x['count'] < y['count'] for x in b for y in a)
        count_relations[1] += any(x['count'] > y['count'] for x in b for y in a)
        count_relations[2] += any(x['count'] == y['count'] for x in b for y in a)
    _require(shift_regimes >= 3 and mixed_regimes >= 3, 'orientation schedules collapsed')
    _require(min(count_relations) >= 10, 'test count is tied to training count')
    # Regression checks are distinct from sampling preferences: legal old colors,
    # already aligned cavities, and equivariance under every complete rotation.
    ring = np.full((6, 9), 6, dtype=np.int8)
    ring[:, 5] = 5
    ring[1:5, 2:5] = 0
    ring[2:4, 3] = 6
    expected = ring.copy()
    expected[2:4, 6:] = 2
    for turn in range(4):
        _check_pair({'input': np.rot90(ring, turn).copy(),
                     'output': np.rot90(expected, turn).copy()}, 2)
    # Exercise the duplicate fallback directly with fourteen repeated scenes.
    pair = puzzles[0]['train'][0]
    seen = set()
    for _ in range(14):
        new_pair = _ensure_unique(pair, _learn_signal(puzzles[0]['train']), seen, set())
        seen.add(_grid_signature(new_pair['input']))
        _check_pair(new_pair, _learn_signal(puzzles[0]['train']))
    _require(len(seen) == 14, 'bounded uniqueness fallback failed')
    # A real bad label and an input-only signal leak must both be rejected.
    for failure in ('label', 'signal'):
        p = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
             for s in ('train', 'test')}
        if failure == 'label':
            p['test'][0]['output'][0, 0] = (int(p['test'][0]['output'][0, 0]) + 1) % 10
        else:
            p['test'][0]['input'][0, 0] = _learn_signal(p['train'])
        try:
            validate_generated_puzzle(p)
        except AssertionError:
            continue
        raise AssertionError('validator accepted corruption')


# Exact source evidence is used only for regression and source-copy rejection.
_SOURCE_FIXTURES = (('0000666666656666/0060666666656666/6000066666656666/0000006666656666/6660606666656666/6660606666656666/6660606666656666/6660006666656666/6666666666656666/6666666666656666/6666666666656666/6666666600056666/6666666606056666/6666666606056666/6666666600056666/6666666666656666', '6666600006656666/6666600606656666/6666660000656666/6666600000056666/6666666606052222/6666666606052222/6666666606052222/6666666600056666/6666666666656666/6666666666656666/6666666666656666/6666666600056666/6666666606052222/6666666606052222/6666666600056666/6666666666656666'), ('0006656666/0606656666/0000656666/6060656666/6000656666/6666656666/6666656666/0006656666/0606656666/0006656666/0666656666/0006656666/6666656666/0006656666/0606656666/0006656666', '6000656666/6060656666/6000056666/6606052222/6600056666/6666656666/6666656666/6600056666/6606052222/6600056666/6606656666/6600056666/6666656666/6600056666/6606052222/6600056666'), ('6000006566/6066606566/6000006566/6666666566/6600006566/6606606566/6606606566/6600006566/6666666566/6666006566/6666006566/6666666566/6000006566/6060606566/6000006566/6666666566/6000066566/6066066566/6000066566/6666666566', '6600000566/6606660522/6600000566/6666666566/6660000566/6660660522/6660660522/6660000566/6666666566/6666600566/6666600566/6666666566/6600000566/6606060522/6600000566/6666666566/6660000566/6660660522/6660000566/6666666566'), ('666666665666/666666665666/600006665666/606066665666/600000665666/666660665666/666660665666/666666665666/666600065666/666666065666/666600065666/666666665666/666000065666/666066065666/666000065666/666666665666/600066665666/606066665666/000066665666/606066665666/600066665666/666666665666/666660005666/666660605666/666660005666', '666666665666/666666665666/666000065666/666060665666/666000005666/666666605666/666666605666/666666665666/666660005666/666666605666/666660005666/666666665666/666600005666/666606605222/666600005666/666666665666/666660005666/666660605222/666600005666/666660605222/666660005666/666666665666/666660005666/666660605222/666660005666'))


def _source_pairs():
    return [{key: np.array([[int(c) for c in row] for row in text.split("/")], dtype=np.int8)
             for key, text in zip(("input", "output"), fixture)} for fixture in _SOURCE_FIXTURES]
