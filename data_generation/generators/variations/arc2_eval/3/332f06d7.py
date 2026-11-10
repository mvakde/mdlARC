import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _resolve_counts(rng, num_train, num_test):
    if num_train is not None:
        if (
            isinstance(num_train, (bool, np.bool_))
            or not isinstance(num_train, (int, np.integer))
        ):
            raise ValueError("num_train must be an integer or None")
        num_train = int(num_train)
        if not 2 <= num_train <= 10:
            raise ValueError("num_train must be between 2 and 10")

    if num_test is not None:
        if (
            isinstance(num_test, (bool, np.bool_))
            or not isinstance(num_test, (int, np.integer))
        ):
            raise ValueError("num_test must be an integer or None")
        num_test = int(num_test)
        if not 1 <= num_test <= 4:
            raise ValueError("num_test must be between 1 and 4")

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test

    token = int(rng.integers(0, 2**63, dtype=np.int64))

    if num_train is None and num_test is None:
        num_test = 1 + token % 4
        low = max(2, num_test + 1)
        num_train = low + (token // 4) % (11 - low)
    elif num_train is None:
        low = max(2, num_test + 1)
        num_train = low + token % (11 - low)
    else:
        highest_legal_test_count = min(4, num_train - 1)
        num_test = 1 + token % highest_legal_test_count

    return int(num_train), int(num_test)


def _square(mask):
    cells = np.argwhere(mask)
    _require(len(cells) > 0, "square marker is absent")

    r0, c0 = cells.min(axis=0)
    r1, c1 = cells.max(axis=0)

    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)

    _require(height == width, "marker is not square")
    _require(len(cells) == height * width, "marker is not solid")
    _require(
        bool(np.all(mask[r0:r1 + 1, c0:c1 + 1])),
        "marker has holes",
    )

    return (int(r0), int(c0)), height



_SOURCE_PAIRS = ((('322333333333', '322333333333', '311333333333', '311311111133', '311311111133', '311311311133', '311311311113', '311111311113', '311111331133', '311133330033', '333333330033', '333333333333'), ('300333333333', '300333333333', '311333333333', '311311111133', '311311111133', '311311311133', '311311311113', '311111311113', '311111331133', '311133331133', '333333331133', '333333333333')), (('33333333333333', '33333333333333', '33333111110033', '33333111110033', '33333113333333', '33333113333333', '33333113333333', '33333111113333', '33333111113333', '33333333313333', '31111111113333', '31111111113333', '32233333333333', '32233333333333'), ('33333333333333', '33333333333333', '33333111111133', '33333111111133', '33333113333333', '33333113333333', '33333113333333', '33333111003333', '33333111003333', '33333333313333', '31111111113333', '31111111113333', '32233333333333', '32233333333333')), (('3333333333333333', '2221111111111113', '2221111331111113', '2221133333111113', '3333333333333113', '3331111111333113', '3331111111333113', '3331111111333113', '3311113111333113', '3311113113331113', '3311113113331113', '3311133113331113', '3000133111111113', '3000133111111113', '3000133111111113', '3333333333333333'), ('3333333333333333', '2221111111111113', '2221111331111113', '2221133333111113', '3333333333333113', '3331111111333113', '3331111000333113', '3331111000333113', '3311113000333113', '3311113113331113', '3311113113331113', '3311133113331113', '3111133111111113', '3111133111111113', '3111133111111113', '3333333333333333')), (('3333333333', '3011333333', '3331311133', '3331313133', '3331113133', '3333333133', '3333111133', '3333133333', '3333133333', '3333233333'), ('3333333333', '3111333333', '3331311133', '3331313133', '3331113133', '3333333133', '3333111133', '3333133333', '3333133333', '3333033333')), (('33333333333333333333', '31111111333333111113', '31111111111111111113', '31111111111111111113', '31111111111111111113', '31111333333333111113', '31111111111113311113', '31111111111113331113', '31111111111113333113', '31111111111113333113', '33333333311113331113', '30000111311113311113', '30000111311113311113', '30000111311111311113', '30000111311111312222', '33331111111111312222', '33331111111113312222', '33311111111113312222', '33311111111111333333', '33333333333333333333'), ('33333333333333333333', '31110000333333111113', '31110000111111111113', '31110000111111111113', '31110000111111111113', '31111333333333111113', '31111111111113311113', '31111111111113331113', '31111111111113333113', '31111111111113333113', '33333333311113331113', '31111111311113311113', '31111111311113311113', '31111111311111311113', '31111111311111312222', '33331111111111312222', '33331111111113312222', '33311111111113312222', '33311111111111333333', '33333333333333333333')))

def _neighbors(p, shape):
    r, c = p
    for x, y in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
        if 0 <= x < shape[0] and 0 <= y < shape[1]:
            yield x, y


def _distances(mask, sources):
    distance = np.full(mask.shape, -1, dtype=np.int16)
    queue = [tuple(map(int, p)) for p in sources]
    for p in queue:
        distance[p] = 0
    for p in queue:
        for q in _neighbors(p, mask.shape):
            if mask[q] and distance[q] < 0:
                distance[q] = distance[p] + 1
                queue.append(q)
    return distance


def _valid_placements(allowed, k):
    return np.lib.stride_tricks.sliding_window_view(allowed, (k, k)).all(axis=(-1, -2))


def _components(mask):
    remaining = mask.copy()
    result = []
    while np.any(remaining):
        p = tuple(np.argwhere(remaining)[0])
        component = _distances(remaining, [p]) >= 0
        result.append(component)
        remaining[component] = False
    return result


def _structural_colors(inp, mover, goal):
    rest = set(map(int, np.unique(inp))) - {mover, goal}
    _require(len(rest) == 2, 'input must have four distinct roles')
    border = np.concatenate((inp[0], inp[-1], inp[1:-1, 0], inp[1:-1, -1]))
    a, b = sorted(rest)
    na, nb = int((border == a).sum()), int((border == b).sum())
    _require(na != nb, 'wall and interior colors are ambiguous')
    return (a, b) if na > nb else (b, a)


def _infer_markers(train_pairs):
    conventions = set()
    for pair in train_pairs:
        inp, out = pair['input'], pair['output']
        for goal in set(map(int, np.unique(inp))) - set(map(int, np.unique(out))):
            try:
                gp, k = _square(inp == goal)
                mover = int(out[gp])
                _, m = _square(inp == mover)
                _, n = _square(out == mover)
                _require(k == m == n and mover != goal, 'not goal replacement')
                wall, corridor = _structural_colors(inp, mover, goal)
                expected = inp.copy()
                expected[inp == mover] = corridor
                expected[inp == goal] = mover
                _require(np.array_equal(out, expected), 'not goal replacement')
                conventions.add((mover, goal))
            except AssertionError:
                continue
    _require(len(conventions) == 1, 'training must reveal one mover and goal convention')
    return next(iter(conventions))


def _analyze_input(inp, mover, goal):
    """Recover movement from visible cells only, including the original geometry.

    A cell-distance field follows the corridor toward the goal. When the goal
    is unreachable by the full square, its closest reachable front must be a
    SINGLE placement. Agreement of nearest-cell, farthest-cell and total
    distances avoids silently selecting a lateral alignment in a wide room.
    Configuration-space cycles and any number of downstream components are
    allowed. Distances are along corridor cells, never through a wall.
    """
    wall, corridor = _structural_colors(inp, mover, goal)
    start, k = _square(inp == mover)
    target, gk = _square(inp == goal)
    _require(k == gk, 'endpoint square sizes differ')
    allowed = inp != wall
    cell_distance = _distances(allowed, np.argwhere(inp == goal))
    _require(np.all(cell_distance[allowed] >= 0), 'disconnected corridor')
    # A second topological route would enclose a wall island. Thick chambers
    # are permitted; closed corridor loops are not.
    boundary = np.zeros(inp.shape, dtype=bool)
    boundary[0] = boundary[-1] = True
    boundary[:, 0] = boundary[:, -1] = True
    walls = ~allowed
    wall_distance = _distances(walls, np.argwhere(walls & boundary))
    _require(np.all(wall_distance[walls] >= 0), 'corridor has a shortcut loop')
    valid = _valid_placements(allowed, k)
    reach_distance = _distances(valid, [start])
    reachable = reach_distance >= 0
    success = bool(reachable[target])
    destination = target
    if not success:
        windows = np.lib.stride_tricks.sliding_window_view(cell_distance, (k, k))
        destinations = []
        for score in (windows.min(axis=(-1, -2)), windows.max(axis=(-1, -2)),
                      windows.sum(axis=(-1, -2))):
            best = np.argwhere(reachable & (score == score[reachable].min()))
            _require(len(best) == 1, 'several equally plausible stopping alignments')
            destinations.append(tuple(map(int, best[0])))
        _require(len(set(destinations)) == 1, 'ambiguous forward stopping front')
        destination = destinations[0]
    expected = inp.copy()
    expected[inp == mover] = corridor
    r, c = destination
    expected[r:r+k, c:c+k] = mover
    components = _components(valid)
    goal_component = next(v for v in components if v[target])
    return expected, {
        'wall': wall, 'corridor': corridor, 'mover': mover, 'goal': goal,
        'k': k, 'start': start, 'target': target, 'destination': destination,
        'success': success, 'travel': int(reach_distance[destination]),
        'components': len(components), 'goal_placements': int(goal_component.sum()),
        'reachable_placements': int(reachable.sum()),
        'placement_freedom': any(sum(bool(valid[q]) for q in _neighbors(tuple(p), valid.shape)) > 2
                                 for p in np.argwhere(reachable)),
    }


def solve(train_pairs, test_input):
    mover, goal = _infer_markers(train_pairs)
    return _analyze_input(np.array(test_input, copy=True), mover, goal)[0]


def _encoded_grid(grid):
    return tuple(''.join(str(int(v)) for v in row) for row in grid)


def _source_copy(pair):
    return (_encoded_grid(pair['input']), _encoded_grid(pair['output'])) in _SOURCE_PAIRS


def _sample_walk(rng, q, complexity, max_side):
    """Grow a self-avoiding route on a randomly spaced rectangular lattice.

    Occupied lattice nodes are chambers and occupied edges are passages.
    Unused edges remain wall, even between adjacent occupied nodes. A one-cell
    wall gap between parallel passages prevents accidental shortcuts.
    """
    route_limit = max_side - 2
    limit = min(6, (route_limit + 1) // (q + 1))
    nr, nc = int(rng.integers(1, limit+1)), int(rng.integers(2, limit+1))
    if complexity == 2:
        nr, nc = max(2, nr), max(2, nc)
    def coordinates(n):
        gaps = np.full(n-1, q+1, dtype=int)
        budget = route_limit - q - int(gaps.sum())
        for j in rng.permutation(n-1):
            addition = int(rng.integers(0, min(5, budget)+1))
            gaps[j] += addition
            budget -= addition
        return np.concatenate(([0], np.cumsum(gaps)))
    rows, cols = coordinates(nr), coordinates(nc)
    target_length = int(rng.integers(2, min(nr*nc, (4, 10, 25)[complexity])+1))
    best = []
    for _ in range(12):
        node = (int(rng.integers(nr)), int(rng.integers(nc)))
        path = [node]
        used = {node}
        while len(path) < target_length:
            options = [p for p in _neighbors(path[-1], (nr, nc)) if p not in used]
            if not options:
                break
            if len(path) > 1 and rng.random() < (0.65 if complexity == 0 else 0.35):
                dr, dc = np.subtract(path[-1], path[-2])
                straight = (path[-1][0]+dr, path[-1][1]+dc)
                if straight in options:
                    options = [straight]
            node = options[int(rng.integers(len(options)))]
            path.append(node)
            used.add(node)
        if len(path) > len(best):
            best = path
        if len(best) >= target_length:
            break
    return [(int(rows[r]), int(cols[c])) for r, c in best]


def _paint_room(mask, p, q):
    r, c = p
    mask[r:r+q, c:c+q] = True


def _edge_slice(a, b, q, width, offset):
    if a[0] == b[0]:
        lo, hi = sorted((a[1], b[1]))
        return (slice(a[0]+offset, a[0]+offset+width), slice(lo+q, hi))
    lo, hi = sorted((a[0], b[0]))
    return (slice(lo+q, hi), slice(a[1]+offset, a[1]+offset+width))


def _terminal(rng, node, neighbor, q, k):
    r, c = node
    offset = int(rng.integers(q-k+1))
    if r == neighbor[0]:
        return r+offset, c if c < neighbor[1] else c+q-k
    return (r if r < neighbor[0] else r+q-k), c+offset


def _make_layout(rng, k, blocked, complexity, max_side):
    # The room width and every individual passage width are distinct choices.
    q = k + int(rng.choice(4, p=(0.38, 0.27, 0.23, 0.12)))
    path = _sample_walk(rng, q, complexity, max_side)
    mask = np.zeros((30, 30), dtype=bool)
    for p in path:
        _paint_room(mask, p, q)
    first = int(rng.integers(len(path)-1)) if blocked else len(path)
    offsets, widths = [], []
    for i, (a, b) in enumerate(zip(path, path[1:])):
        low = 1 if blocked and i >= first else k
        width = int(rng.integers(low, q+1))
        if i == first:
            width = int(rng.integers(1, k))
        if complexity == 0 and i != first and rng.random() < 0.7:
            width = q
        offset = int(rng.integers(q-width+1))
        if i == first:
            # A k-wide collar makes the stopping alignment visible. No center
            # or wall-hugging convention is needed in the preceding chamber.
            collar_offset = int(rng.integers(q-k+1))
            offset = collar_offset + int(rng.integers(k-width+1))
        mask[_edge_slice(a, b, q, width, offset)] = True
        widths.append(width)
        offsets.append(offset)
    desired = None
    if blocked:
        a, b = path[first], path[first+1]
        r, c = a
        if r == b[0]:
            c = c+q-k if b[1] > c else c
            mask[r:r+q, c:c+k] = False
            desired = r+collar_offset, c
        else:
            r = r+q-k if b[0] > r else r
            mask[r:r+k, c:c+q] = False
            desired = r, c+collar_offset
        _paint_room(mask, desired, k)
    start = _terminal(rng, path[0], path[1], q, k)
    if blocked and first == 0:
        # A short starting approach is allowed, including an occasional
        # immediately blocked square. Required training stops still move.
        if path[0][0] == path[1][0]:
            start = desired[0], start[1]
        else:
            start = start[0], desired[1]
    target = _terminal(rng, path[-1], path[-2], q, k)
    if not mask[start[0]:start[0]+k, start[1]:start[1]+k].all():
        raise AssertionError('trimmed starting marker')
    points = np.argwhere(mask)
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)+1
    mask = mask[r0:r1, c0:c1]
    start = start[0]-int(r0), start[1]-int(c0)
    target = target[0]-int(r0), target[1]-int(c0)
    if desired is not None:
        desired = desired[0]-int(r0), desired[1]-int(c0)
    return mask, start, target, desired


def _fallback_layout(k, blocked, variant, max_side):
    # Guaranteed finite construction, also giving enough distinct placements
    # for every legal count request. Only used after bounded random attempts.
    mask = np.zeros((k, min(3*k+8, max_side-2)), dtype=bool)
    mask[:] = True
    start, target = (0, 0), (0, mask.shape[1]-k)
    desired = None
    if blocked:
        cut = k+1+variant % min(5, mask.shape[1]-2*k-1)
        mask[-1, cut:cut+1] = False
        desired = 0, cut-k
    return mask, start, target, desired


def _generate_pair(rng, k, blocked, colors, complexity, orientation, must_move, used, scale):
    wall, corridor, mover, goal = colors
    min_side, max_side = scale
    for attempt in range(96):
        try:
            if attempt < 64:
                mask, start, target, desired = _make_layout(rng, k, blocked, complexity, max_side)
            else:
                mask, start, target, desired = _fallback_layout(k, blocked, attempt-64, max_side)
            padding = []
            for dimension in mask.shape:
                lower = max(2, min_side-dimension)
                upper = min(max(12, lower), max_side-dimension)
                extra = int(rng.integers(lower, upper+1))
                before = int(rng.integers(extra+1))
                padding.append((before, extra-before))
            mask = np.pad(mask, padding)
            start = tuple(start[i]+padding[i][0] for i in range(2))
            target = tuple(target[i]+padding[i][0] for i in range(2))
            if desired is not None:
                desired = tuple(desired[i]+padding[i][0] for i in range(2))
            inp = np.full(mask.shape, wall, dtype=np.int8)
            inp[mask] = corridor
            r, c = start
            inp[r:r+k, c:c+k] = mover
            r, c = target
            inp[r:r+k, c:c+k] = goal
            out, info = _analyze_input(inp, mover, goal)
            _require((info['wall'], info['corridor']) == (wall, corridor),
                     'canvas margins obscure the structural color roles')
            _require(info['success'] != blocked, 'wrong requested movement outcome')
            if desired is not None:
                _require(info['destination'] == desired, 'collar is not the first reachable obstruction')
            _require(not must_move or info['travel'] > 0, 'demonstration must move')
            turns = int(rng.integers(4)) if orientation is None else orientation
            reflect = bool(rng.integers(2)) if orientation is None else False
            def transform(a):
                a = np.rot90(a, turns)
                return np.array(np.fliplr(a) if reflect else a, dtype=np.int8, copy=True)
            pair = {'input': transform(inp), 'output': transform(out)}
            signature = _encoded_grid(pair['input'])
            _require(signature not in used and not _source_copy(pair), 'duplicate configuration')
            # Recheck after rotation, with no access to the construction path.
            _require(np.array_equal(_analyze_input(pair['input'], mover, goal)[0], pair['output']),
                     'orientation changed interpretation')
            used.add(signature)
            return pair
        except AssertionError:
            continue
    raise AssertionError('bounded construction exhausted')


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    mover, goal = map(int, rng.choice(10, 2, replace=False))
    others = [v for v in range(10) if v not in (mover, goal)]
    base = tuple(map(int, rng.choice(others, 2, replace=False)))
    variable_palette = rng.random() < 0.6
    orientation = int(rng.integers(4)) if rng.random() < 0.16 else None
    # Preserve the original extrapolation as one regime; also exercise
    # familiar sizes, smaller tests and mixed scale/outcome generalization.
    regime = int(rng.integers(4))
    upper = 3 if regime == 0 else int(rng.integers(3, 6))
    lower = int(rng.integers(2, upper+1)) if regime == 2 else 1
    shared_k = int(rng.integers(max(2, lower), upper+1))
    same_size_contrast = rng.random() < 0.35
    train_specs = [(shared_k, False, True),
                   (shared_k if same_size_contrast else int(rng.integers(max(2, lower), upper+1)), True, True)]
    for _ in range(num_train-2):
        k = int(rng.integers(lower, upper+1))
        train_specs.append((k, bool(k > 1 and rng.integers(2)), False))
    train_specs = [train_specs[int(i)] for i in rng.permutation(num_train)]
    test_specs = []
    for _ in range(num_test):
        if regime == 0:
            k, blocked = 4, True
        elif regime == 1:
            k = int(rng.choice([spec[0] for spec in train_specs]))
            blocked = bool(k > 1 and rng.integers(2))
        elif regime == 2:
            k = int(rng.integers(1, lower))
            blocked = bool(k > 1 and rng.integers(2))
        else:
            k = int(rng.integers(1, 6))
            blocked = bool(k > 1 and rng.integers(2))
        test_specs.append((k, blocked, False))
    puzzle, used = {'train': [], 'test': []}, set()
    first_palette = None
    for split, specs in (('train', train_specs), ('test', test_specs)):
        for index, (k, blocked, must_move) in enumerate(specs):
            pair_palette = tuple(map(int, rng.choice(others, 2, replace=False))) if variable_palette else base
            if split == 'train':
                if index == 0:
                    first_palette = pair_palette
                elif index == 1 and variable_palette and pair_palette == first_palette:
                    pair_palette = pair_palette[::-1]
            colors = pair_palette + (mover, goal)
            complexity = int(rng.choice(3, p=(0.2, 0.35, 0.45)))
            if split == 'test' and regime != 0:
                complexity = max(complexity, int(rng.integers(1, 3)))
            scale = ((10, 16) if split == 'train' else (20, 30)) if regime == 0 else (4, 30)
            puzzle[split].append(_generate_pair(rng, k, blocked, colors, complexity,
                                                orientation, must_move, used, scale))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    n, m = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, 'illegal pair counts')
    seen = set()
    for pair in puzzle['train'] + puzzle['test']:
        _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
        for a in pair.values():
            _require(isinstance(a, np.ndarray) and a.ndim == 2, 'grid must be a 2D NumPy array')
            _require(np.issubdtype(a.dtype, np.integer), 'grid must contain integers')
            _require(all(1 <= d <= 30 for d in a.shape), 'grid exceeds canvas bounds')
            _require(np.all((0 <= a) & (a <= 9)), 'invalid color')
        _require(pair['input'].shape == pair['output'].shape, 'shape must be preserved')
        signature = _encoded_grid(pair['input'])
        _require(signature not in seen, 'duplicate input')
        _require(not _source_copy(pair), 'exact source pair copy')
        seen.add(signature)
    mover, goal = _infer_markers(puzzle['train'])
    information = {}
    for split in ('train', 'test'):
        information[split] = []
        for pair in puzzle[split]:
            expected, info = _analyze_input(pair['input'], mover, goal)
            _require(np.array_equal(pair['output'], expected), 'incorrect footprint movement or altered scenery')
            information[split].append(info)
    training = information['train']
    _require(any(v['success'] and v['travel'] > 0 for v in training), 'training needs moving success')
    _require(any(not v['success'] and v['travel'] > 0 for v in training), 'training needs moving collision')
    palettes = {(v['wall'], v['corridor']) for v in training}
    if len(palettes) == 1:
        _require(all((v['wall'], v['corridor']) in palettes for v in information['test']),
                 'structural palette variation was not demonstrated')
    # No artificial count of placement components, k-1 reachability, square
    # canvas requirement, or fixed test outcome is part of the invariant.


def _trace_route(allowed, start, target):
    """Find a route with the fewest bends among shortest orthogonal routes."""
    distance = _distances(allowed, [target])
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    costs = np.full(allowed.shape + (4,), 10000, dtype=np.int16)
    costs[target] = 0
    for p in sorted(map(tuple, np.argwhere(distance >= 0)), key=lambda p: int(distance[p])):
        if p == target:
            continue
        options = [(j, (p[0]+dr, p[1]+dc)) for j, (dr, dc) in enumerate(directions)
                   if 0 <= p[0]+dr < allowed.shape[0] and 0 <= p[1]+dc < allowed.shape[1]
                   and distance[p[0]+dr, p[1]+dc] == distance[p]-1]
        for old in range(4):
            costs[p+(old,)] = min(int(costs[q+(j,)]) + int(old != j) for j, q in options)
    p, old = start, int(np.argmin(costs[start]))
    route, headings = [p], []
    while p != target:
        options = []
        for j, (dr, dc) in enumerate(directions):
            q = p[0]+dr, p[1]+dc
            if 0 <= q[0] < allowed.shape[0] and 0 <= q[1] < allowed.shape[1] and distance[q] == distance[p]-1:
                options.append((int(costs[q+(j,)])+int(old != j), j, q))
        _, old, p = min(options)
        route.append(p)
        headings.append(old)
    return route, headings


def _route_features(inp, info):
    """Reconstruct features from pixels, without construction metadata.

    Widths are cross-sections perpendicular to a shortest cell route. Turns
    before stopping are measured separately in full-square placement space.
    """
    allowed = inp != info['wall']
    route, headings = _trace_route(allowed, info['start'], info['target'])
    _, mover_headings = _trace_route(_valid_placements(allowed, info['k']),
                                     info['start'], info['destination'])
    widths = []
    for (r, c), direction in zip(route, headings):
        dr, dc = (0, 1) if direction < 2 else (1, 0)
        width = 1
        for sign in (-1, 1):
            x, y = r+sign*dr, c+sign*dc
            while 0 <= x < allowed.shape[0] and 0 <= y < allowed.shape[1] and allowed[x, y]:
                width += 1
                x, y = x+sign*dr, y+sign*dc
        widths.append(width)
    profile = [v for i, v in enumerate(widths) if i == 0 or v != widths[i-1]]
    turns = sum(a != b for a, b in zip(headings, headings[1:]))
    legs = []
    for i, d in enumerate(headings):
        if i == 0 or d != headings[i-1]:
            legs.append(1)
        else:
            legs[-1] += 1
    k = info['k']
    narrow_lengths = []
    for i, width in enumerate(widths):
        if width < k:
            if i == 0 or widths[i-1] >= k:
                narrow_lengths.append(1)
            else:
                narrow_lengths[-1] += 1
    first_narrow = next((i for i, v in enumerate(profile) if v < k), len(profile))
    smaller_reaches = True
    if k > 1:
        small_valid = _valid_placements(allowed, k-1)
        smaller_reaches = bool(_distances(small_valid, [info['start']])[info['target']] >= 0)
    cells = np.argwhere(allowed)
    lo, hi = cells.min(axis=0), cells.max(axis=0)+1
    normalized = np.zeros(inp.shape, dtype=np.int8)
    for code, role in enumerate(('wall', 'corridor', 'mover', 'goal')):
        normalized[inp == info[role]] = code
    signature = _encoded_grid(normalized[lo[0]:hi[0], lo[1]:hi[1]])
    return {
        'turns': turns, 'leg_lengths': tuple(legs), 'route_length': len(headings),
        'reachable_turns': sum(a != b for a, b in zip(mover_headings, mover_headings[1:])),
        'width_profile': tuple(profile), 'width_changes': max(0, len(profile)-1),
        'passable_narrowings': sum(a > b >= k for a, b in zip(profile, profile[1:])),
        'impassable_sections': sum(v < k and (i == 0 or profile[i-1] >= k)
                                   for i, v in enumerate(profile)),
        'throat_lengths': tuple(narrow_lengths),
        'passable_narrowings_before_throat': sum(a > b >= k for a, b in
                                                  zip(profile[:first_narrow], profile[1:first_narrow])),
        'smaller_reaches': smaller_reaches,
        'goal_edge': any(info['target'][i] in (0, inp.shape[i]-k) for i in range(2)),
        'start_edge': any(info['start'][i] in (0, inp.shape[i]-k) for i in range(2)),
        'margins': (int(lo[0]), int(lo[1]), int(inp.shape[0]-hi[0]), int(inp.shape[1]-hi[1])),
        'heading': headings[0], 'shape': inp.shape, 'structure': signature,
    }


def _fingerprint(puzzle):
    return tuple((_encoded_grid(p['input']), _encoded_grid(p['output']))
                 for s in ('train', 'test') for p in puzzle[s])


def self_test() -> None:
    # Fixed source regression tests use the original cells without imposing
    # synthetic generation restrictions. The fifth prediction uses only the
    # four training pairs and the source test input.
    sources = [{'input': np.array([[int(v) for v in r] for r in a], dtype=np.int8),
                'output': np.array([[int(v) for v in r] for r in b], dtype=np.int8)}
               for a, b in _SOURCE_PAIRS]
    for pair in sources:
        expected, info = _analyze_input(pair['input'], 0, 2)
        _require(np.array_equal(expected, pair['output']), 'source regression')
    _require(np.array_equal(solve(sources[:4], sources[4]['input']), sources[4]['output']),
             'source train-only prediction failed')
    # Geometry-controlled size contrast: exactly the same 5 -> 4 -> 3 -> 2
    # passage with a wide goal approach passes a 2-square and blocks a
    # 3-square. The first PASSABLE reduction must not stop either square.
    for k in (2, 3):
        inp = np.full((9, 27), 3, dtype=np.int8)
        for left, right, width in ((1, 7, 5), (7, 12, 4), (12, 17, 3), (17, 19, 2), (19, 26, 5)):
            inp[2:2+width, left:right] = 1
        inp[2:2+k, 1:1+k] = 0
        inp[2:2+k, 26-k:26] = 2
        predicted, info = _analyze_input(inp, 0, 2)
        expected = inp.copy()
        expected[inp == 0] = 1
        destination = (2, 24) if k == 2 else (2, 14)
        r, c = destination
        expected[r:r+k, c:c+k] = 0
        _require(np.array_equal(predicted, expected), 'independent-width contrast failed')
    ambiguous = np.full((8, 18), 3, dtype=np.int8)
    ambiguous[1:5, 1:8] = ambiguous[1:5, 9:17] = 1
    ambiguous[2:4, 8] = 1
    ambiguous[1:4, 1:4] = 0
    ambiguous[1:4, 14:17] = 2
    try:
        _analyze_input(ambiguous, 0, 2)
    except AssertionError:
        pass
    else:
        raise AssertionError('ambiguous lateral stopping alignment accepted')
    automatic = [generate_puzzle(seed) for seed in range(42)]
    rows, within = [], []
    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
        mover, goal = _infer_markers(puzzle['train'])
        local = []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                expected, info = _analyze_input(pair['input'], mover, goal)
                _require(np.array_equal(expected, pair['output']), 'prediction disagrees')
                if split == 'test':
                    _require(np.array_equal(solve(puzzle['train'], pair['input']), pair['output']),
                             'test convention not recoverable')
                row = dict(info, **_route_features(pair['input'], info), split=split)
                rows.append(row)
                local.append(row)
        within.append(local)
    _require(len({len(p['train']) for p in automatic}) >= 4, 'train count coverage')
    _require(len({len(p['test']) for p in automatic}) >= 3, 'test count coverage')
    _require(len({_fingerprint(p) for p in automatic}) == 42, 'cross-seed repetition')
    _require(len({r['structure'] for r in rows}) >= 150, 'too few cropped structures')
    _require(len({r['shape'] for r in rows}) >= 50, 'too few canvas shapes')
    _require(any(r['shape'][0] != r['shape'][1] for r in rows), 'rectangular canvases missing')
    _require(max(max(r['shape']) for r in rows) == 30, 'large canvases missing')
    for role in ('wall', 'corridor', 'mover', 'goal'):
        _require({r[role] for r in rows} == set(range(10)), role+' does not use all colors')
    _require(max(r['turns'] for r in rows) >= 8, 'long winding routes missing')
    _require(max(r['reachable_turns'] for r in rows if not r['success']) >= 6,
             'complex routes only occur beyond the blockage')
    _require(any(r['turns'] == 0 for r in rows), 'straight routes missing')
    _require(any(r['turns'] == 1 for r in rows), 'one-bend routes missing')
    _require(len({r['width_profile'] for r in rows}) >= 80, 'width profiles collapsed')
    _require(sum(r['passable_narrowings'] > 0 for r in rows) >= 25, 'passable narrowings missing')
    _require(sum(r['impassable_sections'] >= 2 for r in rows) >= 5, 'multiple throats missing')
    _require(any(r['components'] >= 3 for r in rows), 'downstream placement components missing')
    _require(any(r['k'] > 2 and not r['smaller_reaches'] for r in rows), 'k-1 reachability still forced')
    _require(len({length for r in rows for length in r['throat_lengths']}) >= 4,
             'throat lengths collapsed')
    _require(any(not r['success'] and r['passable_narrowings_before_throat'] >= 2 for r in rows),
             'multiple passable narrowings before blockage missing')
    _require(sum(not r['success'] and r['goal_placements'] > 1 for r in rows) >= 20,
             'blocked goals lack approaches')
    _require(any(r['success'] and r['placement_freedom'] for r in rows), 'wide successful corridors missing')
    _require(any(not r['success'] and r['placement_freedom'] for r in rows), 'wide blocked corridors missing')
    _require({r['goal_edge'] for r in rows} == {False, True}, 'endpoint placement collapsed')
    _require({r['heading'] for r in rows} == {0, 1, 2, 3}, 'starting direction collapsed')
    _require(len({r['margins'] for r in rows}) >= 40, 'margin variation collapsed')
    test_rows = [r for r in rows if r['split'] == 'test']
    _require({r['success'] for r in test_rows} == {False, True}, 'fixed test outcome')
    _require(len({(r['k'], r['success']) for r in test_rows}) >= 7, 'test size/outcome correlation')
    _require(any({r['success'] for r in local if r['split'] == 'test'} == {False, True}
                 for local in within), 'mixed test outcomes missing')
    _require(any(all(max(r['shape']) <= 16 and r['k'] <= 3 for r in local if r['split'] == 'train')
                 and all(min(r['shape']) >= 20 and r['k'] == 4 and not r['success']
                         for r in local if r['split'] == 'test') for local in within),
             'source scale extrapolation missing')
    for relation in ('smaller', 'larger', 'familiar'):
        def has_shift(local):
            sizes = {r['k'] for r in local if r['split'] == 'train'}
            return any((r['k'] < min(sizes) if relation == 'smaller' else
                        r['k'] > max(sizes) if relation == 'larger' else r['k'] in sizes)
                       for r in local if r['split'] == 'test')
        _require(any(has_shift(local) for local in within), relation+' test sizes missing')
    for feature in ('shape', 'turns', 'k', 'width_profile', 'margins'):
        _require(sum(len({r[feature] for r in local}) > 1 for local in within) >= 25,
                 feature+' lacks within-puzzle diversity')
    _require(sum(len({(r['wall'], r['corridor']) for r in local if r['split'] == 'train'}) > 1
                 for local in within) >= 10, 'within-puzzle structural recoloring missing')
    _require(sum(r['turns'] >= 6 and r['width_changes'] >= 4 for r in rows) >= 10,
             'combined route and width variation missing')
    # Mutations check that validation examines the supplied output and shared
    # marker convention, rather than accepting construction-time intentions.
    original = automatic[0]
    for mutation in ('wall', 'goal', 'square', 'test_roles'):
        bad = {s: [{key: value.copy() for key, value in pair.items()} for pair in original[s]]
               for s in ('train', 'test')}
        mover, goal = _infer_markers(bad['train'])
        if mutation == 'test_roles':
            for a in bad['test'][0].values():
                moving, target = a == mover, a == goal
                a[moving], a[target] = goal, mover
        else:
            pair = next(p for p in bad['train'] if np.any(p['output'] == goal))
            _, info = _analyze_input(pair['input'], mover, goal)
            if mutation == 'wall':
                point = tuple(np.argwhere(pair['input'] == info['wall'])[0])
            elif mutation == 'goal':
                point = tuple(np.argwhere(pair['input'] == goal)[0])
            else:
                point = tuple(np.argwhere(pair['output'] == mover)[0])
            pair['output'][point] = info['corridor']
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            continue
        raise AssertionError('validator accepted '+mutation+' mutation')
    for counts in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(814, *counts)
        validate_generated_puzzle(puzzle)
        _require(all(v is None or len(puzzle[s]) == v for s, v in zip(('train', 'test'), counts)),
                 'requested count ignored')
        _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(814, *counts)), 'nondeterminism')
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4), (2.0, 1), (2, True)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError('invalid count request accepted')
