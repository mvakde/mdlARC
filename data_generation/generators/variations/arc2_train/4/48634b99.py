import numpy as np


# Rejecting these inputs is sufficient to reject every exact source-task pair.
_SOURCE_INPUT_ROWS = (
    (
        "7777778777777777",
        "7787778777777777",
        "7787778777777777",
        "7787778779777777",
        "7787778779777777",
        "7787778778777777",
        "7787778778777777",
        "7777778777778777",
        "7777778777778787",
        "7777778777777787",
        "7877777777777787",
        "7877877777778787",
        "7777877777778787",
        "7777777877777787",
        "7777877877777787",
        "7777877777777787",
    ),
    (
        "7777777777777778",
        "7877777777787778",
        "7877777777787777",
        "7877777777777777",
        "7877777778777777",
        "7777777878777777",
        "7777877878787787",
        "7877877778787787",
        "7877877778777777",
        "7877877778777877",
        "7877877778777977",
        "7877877778777777",
        "7877877777777878",
        "7777877777787878",
        "7777877777787777",
        "7777877777777777",
    ),
    (
        "7787777778777787",
        "7787787778777787",
        "7787787778777787",
        "7787777778777787",
        "7777777779777787",
        "7777778779787787",
        "7877778779787787",
        "7877777779777787",
        "7777787877778787",
        "7777787877778787",
        "7878777777777777",
        "7878777777777777",
        "7877777777787777",
        "7877777787787777",
        "7877787787777777",
        "7877787777777777",
    ),
    (
        "7777777777777777",
        "7777777877777778",
        "7777777877777778",
        "7777877877777778",
        "7777877877777778",
        "7777777877777778",
        "7777777877777778",
        "7777777877877778",
        "7777777877877778",
        "7777777777877778",
        "7977877877877778",
        "7977877877777777",
        "7977777777777787",
        "7877787777778787",
        "7877787777778777",
        "7877777777777777",
    ),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _normalize_optional_count(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _normalize_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be a nonnegative integer")
    seed_int = int(seed)
    if seed_int < 0:
        raise ValueError("seed must be a nonnegative integer")
    return seed_int


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _normalize_optional_count("num_train", num_train)
    test_count = _normalize_optional_count("num_test", num_test)

    if train_count is not None and not 2 <= train_count <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if test_count is not None and not 1 <= test_count <= 4:
        raise ValueError("num_test must be between 1 and 4")

    code = seed ^ (seed >> 32) ^ (seed >> 64)

    if train_count is None and test_count is None:
        test_count = 1 + code % 4
        minimum_train = max(2, test_count + 1)
        train_count = minimum_train + (code // 4) % (11 - minimum_train)
    elif train_count is None:
        minimum_train = max(2, test_count + 1)
        train_count = minimum_train + (code // 5) % (11 - minimum_train)
    elif test_count is None:
        maximum_test = min(4, train_count - 1)
        test_count = 1 + (code // 7) % maximum_test

    if not 2 <= train_count <= 10:
        raise ValueError("resolved num_train is illegal")
    if not 1 <= test_count <= 4:
        raise ValueError("resolved num_test is illegal")
    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def _paint(grid, axis, lane, start, stop, color):
    if axis == 0:
        grid[start:stop, lane] = color
    else:
        grid[lane, start:stop] = color


def _line(grid, axis, lane):
    return grid[:, lane] if axis == 0 else grid[lane, :]


def _runs(grid, axis, background):
    found = []
    for lane in range(grid.shape[1 - axis]):
        line = _line(grid, axis, lane)
        start = None
        for pos in range(len(line) + 1):
            occupied = pos < len(line) and int(line[pos]) != background
            if occupied and start is None:
                start = pos
            if not occupied and start is not None:
                found.append((lane, start, pos, tuple(int(v) for v in line[start:pos])))
                start = None
    return found


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _is_source_input(grid):
    if grid.shape != (16, 16):
        return False
    return tuple(''.join(str(int(v)) for v in row) for row in grid) in _SOURCE_INPUT_ROWS


def _input_solutions(grid):
    """Infer every valid bar-axis interpretation without consulting an output."""
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) != 3:
        return []
    order = np.argsort(counts)
    accent, structural, background = (int(colors[i]) for i in order)
    if counts[order[2]] <= counts[order[1]]:
        return []
    accent_points = np.argwhere(grid == accent)
    n = len(accent_points)
    answers = []
    for axis in (0, 1):
        lanes = set(int(v) for v in accent_points[:, 1 - axis])
        if len(lanes) != 1:
            continue
        lane = next(iter(lanes))
        along = sorted(int(v) for v in accent_points[:, axis])
        if along != list(range(along[0], along[0] + n)):
            continue
        runs = _runs(grid, axis, background)
        marked = [r for r in runs if r[0] == lane and r[1] <= along[0] and along[-1] < r[2]]
        if len(marked) != 1:
            continue
        source = marked[0]
        if source[2] - source[1] != 2 * n:
            continue
        if along == list(range(source[1], source[1] + n)):
            endpoint = 0
        elif along == list(range(source[2] - n, source[2])):
            endpoint = 1
        else:
            continue
        if any(v != (accent if i in along else structural)
               for i, v in zip(range(source[1], source[2]), source[3])):
            continue
        target_length = 2 * n + 2
        targets = [r for r in runs if r[2] - r[1] == target_length
                   and all(v == structural for v in r[3])]
        if len(targets) != 1:
            continue
        target = targets[0]
        output = grid.copy()
        _paint(output, axis, source[0], source[1], source[2], structural)
        begin = target[1] if endpoint == 0 else target[2] - n - 1
        _paint(output, axis, target[0], begin, begin + n + 1, accent)
        answers.append({'output': output, 'axis': axis, 'endpoint': endpoint,
                        'source_length': 2 * n, 'target_length': target_length,
                        'runs': runs, 'source': source, 'target': target,
                        'colors': (background, structural, accent)})
    return answers


def _analyze_pair(input_grid, output_grid):
    _require(input_grid.shape == output_grid.shape, 'grid dimensions changed')
    solutions = _input_solutions(input_grid)
    _require(len(solutions) == 1, 'input does not determine a unique bar-axis rule')
    solution = solutions[0]
    _require(np.array_equal(solution['output'], output_grid), 'output violates the inferred transfer')
    _require(not np.array_equal(input_grid, output_grid), 'unchanged output')
    return solution


def _choose_start(rng, available):
    if rng.random() < .30:
        return available[0] if rng.random() < .5 else available[-1]
    return int(rng.choice(available))


def _available_starts(occupancy, lane, length):
    along = occupancy.shape[0]
    choices = []
    for start in range(along - length + 1):
        if not np.any(occupancy[max(0, start - 1):min(along, start + length + 1), lane]):
            choices.append(start)
    return choices


def _build_pair(rng, regime, n, endpoint, axis, colors, odd_midpoint):
    if regime not in ('source_train', 'target_test'):
        raise ValueError('unknown pair regime')
    source_length, target_length = 2 * n, 2 * n + 2
    background, structural, accent = colors
    for _ in range(100):
        along = int(rng.integers(max(6, target_length), 31))
        simple = bool(rng.random() < .22)
        across = int(rng.integers(9 if simple else 5, 31))
        requested = int(rng.integers(4, min(7, (across + 1) // 2 + 1))) if simple else int(rng.integers(4, min(21, across * 2 + 1)))
        same_lane = bool(not simple and rng.random() < .35 and source_length + target_length + 1 <= along)
        occupancy = np.zeros((along, across), dtype=bool)
        bars = []
        sparse_lanes = list(range(int(rng.integers(2)), across, 2)) if simple else None
        if simple and len(sparse_lanes) < requested:
            sparse_lanes = list(range(0, across, 2))
        target_lane = int(rng.choice(sparse_lanes)) if simple else int(rng.integers(across))
        source_lane = target_lane if same_lane else int(rng.choice([v for v in (sparse_lanes if simple else range(across)) if v != target_lane]))
        required = [(target_length, target_lane, 'target'), (source_length, source_lane, 'source')]
        if rng.random() < .5:
            required.reverse()
        failed = False
        for length, lane, tag in required:
            starts = _available_starts(occupancy, lane, length)
            if not starts:
                failed = True
                break
            start = _choose_start(rng, starts)
            occupancy[start:start + length, lane] = True
            bars.append((lane, start, length, tag))
        if failed:
            continue
        extras = []
        if odd_midpoint:
            extras.append(source_length + 1)
        if rng.random() < .65:
            extras.append(2)
        if rng.random() < .50:
            extras.append(source_length)
        max_length = min(along, max(12, target_length + 4))
        pool = [length for length in range(2, max_length + 1)
                if length != target_length and (odd_midpoint or length % 2 == 0)]
        while len(extras) < requested - 2:
            if rng.random() < .55:
                extras.append(2)
            else:
                extras.append(int(rng.choice(pool)))
        rng.shuffle(extras)
        for length in extras:
            lanes = list(sparse_lanes) if simple else list(range(across))
            rng.shuffle(lanes)
            if simple:
                lanes = [lane for lane in lanes if not np.any(occupancy[:, lane])]
            options = [(lane, _available_starts(occupancy, lane, length)) for lane in lanes]
            options = [(lane, starts) for lane, starts in options if starts]
            if not options:
                failed = True
                break
            if rng.random() < .45:
                shared = [item for item in options if np.any(occupancy[:, item[0]])]
                if shared:
                    options = shared
            lane, starts = options[int(rng.integers(len(options)))]
            start = _choose_start(rng, starts)
            occupancy[start:start + length, lane] = True
            bars.append((lane, start, length, 'extra'))
        if failed or len(bars) < 4 or np.count_nonzero(occupancy) >= occupancy.size * .48:
            continue
        shape = (along, across) if axis == 0 else (across, along)
        grid = np.full(shape, background, dtype=np.int8)
        for lane, start, length, _ in bars:
            _paint(grid, axis, lane, start, start + length, structural)
        source = next(item for item in bars if item[3] == 'source')
        begin = source[1] if endpoint == 0 else source[1] + n
        _paint(grid, axis, source[0], begin, begin + n, accent)
        solutions = _input_solutions(grid)
        if len(solutions) != 1 or solutions[0]['axis'] != axis or _is_source_input(grid):
            continue
        return grid, solutions[0]['output']
    raise RuntimeError('failed to construct an unambiguous pair')


def _palette(rng):
    return tuple(int(x) for x in rng.choice(10, size=3, replace=False))


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _normalize_seed(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    scheme = int(rng.integers(3))  # withheld middle, familiar, or extrapolated size
    base = int(rng.integers(1, 9))
    train_ns = [base, base + (2 if scheme == 0 else 1)]
    test_n = base + (1 if scheme == 0 else 2 if scheme == 2 else int(rng.integers(2)))
    while len(train_ns) < train_count:
        if scheme == 0:
            choices = [v for v in range(1, 11) if v != test_n]
        elif scheme == 2:
            choices = list(range(1, base + 2))
        else:
            choices = list(range(1, 11))
        train_ns.append(int(rng.choice(choices)))
    rng.shuffle(train_ns)
    test_ns = [test_n if i == 0 or rng.random() < .7 else int(rng.choice(train_ns))
               for i in range(test_count)]
    endpoints = [0, 1] + [int(rng.integers(2)) for _ in range(train_count - 2 + test_count)]
    if rng.random() < .5:
        endpoints[0], endpoints[1] = endpoints[1], endpoints[0]
    orientation_mode = int(rng.integers(3))
    first_axis = int(rng.integers(2))
    if orientation_mode == 0:
        axes = [first_axis] * (train_count + test_count)
    elif orientation_mode == 1:
        axes = [first_axis, 1 - first_axis] + [int(rng.integers(2)) for _ in range(train_count + test_count - 2)]
    else:
        axes = [first_axis] * train_count + [1 - first_axis] * test_count
    stable_palette = bool(rng.integers(2))
    shared_palette = _palette(rng)
    palettes = [shared_palette if stable_palette else _palette(rng)
                for _ in range(train_count + test_count)]
    odd_family = bool(rng.integers(2))
    odd_flags = [False] * (train_count + test_count)
    if odd_family:
        odd_flags[0] = True
        for i in range(1, len(odd_flags)):
            odd_flags[i] = bool(rng.integers(2))
    puzzle = {'train': [], 'test': []}
    used = set()
    specifications = [('train', 'source_train', n) for n in train_ns] + [
        ('test', 'target_test', n) for n in test_ns]
    for i, (section, regime, n) in enumerate(specifications):
        for _ in range(20):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            inp, out = _build_pair(pair_rng, regime, n, endpoints[i], axes[i], palettes[i], odd_flags[i])
            key = _grid_key(inp)
            if key not in used:
                used.add(key)
                puzzle[section].append({'input': inp, 'output': out})
                break
        else:
            raise RuntimeError('failed to construct distinct pairs')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list), 'splits must be lists')
    train_count, test_count = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count, 'invalid pair counts')
    inputs, pairs = set(), set()
    analyses = {'train': [], 'test': []}
    for section in ('train', 'test'):
        for pair in puzzle[section]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            inp, out = pair['input'], pair['output']
            for grid in (inp, out):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, 'grid must be a 2D array')
                _require(all(1 <= size <= 30 for size in grid.shape), 'invalid dimensions')
                _require(np.issubdtype(grid.dtype, np.integer), 'noninteger grid')
                _require(0 <= int(grid.min()) and int(grid.max()) <= 9, 'invalid color')
            key, pair_key = _grid_key(inp), (_grid_key(inp), _grid_key(out))
            _require(key not in inputs and pair_key not in pairs, 'duplicate pair or input')
            _require(not _is_source_input(inp), 'source pair copied')
            inputs.add(key)
            pairs.add(pair_key)
            analysis = _analyze_pair(inp, out)
            _require(len(analysis['runs']) >= 3, 'bar field too small')
            analyses[section].append(analysis)
    _require({a['endpoint'] for a in analyses['train']} == {0, 1}, 'training omits an endpoint')
    # Odd midpoint distractors are an extension to the original all-even task.
    def odd_mid(a):
        return any(r[2] - r[1] == a['source_length'] + 1 for r in a['runs'])
    _require(not any(odd_mid(a) for a in analyses['test']) or any(odd_mid(a) for a in analyses['train']),
             'test uses an undemonstrated odd midpoint distractor')
    # Palette and orientation are visible in each input, so they are pair properties.


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ('input', 'output')) for s in ('train', 'test'))


def self_test() -> None:
    _require(_puzzles_equal(generate_puzzle(517), generate_puzzle(517)), 'nondeterministic puzzle')
    _require(not _puzzles_equal(generate_puzzle(517), generate_puzzle(518)), 'seed has no effect')
    for tr, te in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(800 + (tr or 0) * 10 + (te or 0), tr, te)
        validate_generated_puzzle(p)
        _require(tr is None or len(p['train']) == tr, 'train count ignored')
        _require(te is None or len(p['test']) == te, 'test count ignored')
    for tr, te in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(16, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid count accepted')
    train_counts, test_counts, shapes, patterns, palettes = set(), set(), set(), set(), set()
    axes, transitions, repeated, multilane, odd, touching, same_lane = set(), set(), 0, 0, 0, 0, 0
    simple, combined, large, small = 0, 0, 0, 0
    mixed_axes, mixed_palettes, withheld, fewer_test_bars = 0, 0, 0, 0
    for seed in range(42):
        p = generate_puzzle(seed)
        train_counts.add(len(p['train']))
        test_counts.add(len(p['test']))
        puzzle_analyses = {'train': [], 'test': []}
        for section in ('train', 'test'):
            for pair in p[section]:
                a = _analyze_pair(pair['input'], pair['output'])
                puzzle_analyses[section].append(a)
                shapes.add(pair['input'].shape)
                patterns.add(tuple(sorted(r[2] - r[1] for r in a['runs'])))
                palettes.add(a['colors'])
                axes.add(a['axis'])
                transitions.add(a['source_length'])
                repeat = sum(r[2] - r[1] == a['source_length'] for r in a['runs']) > 1
                repeated += repeat
                counts = [sum(r[0] == lane for r in a['runs']) for lane in range(pair['input'].shape[1-a['axis']])]
                multi = max(counts) > 1
                multilane += multi
                has_odd = any((r[2] - r[1]) % 2 for r in a['runs'])
                odd += has_odd
                touch = any(abs(x[0]-y[0]) == 1 and max(x[1],y[1]) < min(x[2],y[2])
                            for x in a['runs'] for y in a['runs'])
                touching += touch
                same_lane += a['source'][0] == a['target'][0]
                simple += not multi and not repeat and not touch
                combined += multi and repeat and touch and has_odd
                large += a['source_length'] > 10
                small += pair['input'].shape[a['axis']] <= 12
        all_analyses = puzzle_analyses['train'] + puzzle_analyses['test']
        mixed_axes += len({a['axis'] for a in all_analyses}) > 1
        mixed_palettes += len({a['colors'] for a in all_analyses}) > 1
        train_lengths = {a['source_length'] for a in puzzle_analyses['train']}
        withheld += any(a['source_length'] not in train_lengths for a in puzzle_analyses['test'])
        fewer_test_bars += min(len(a['runs']) for a in puzzle_analyses['test']) < max(
            len(a['runs']) for a in puzzle_analyses['train'])
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, 'automatic count diversity')
    _require(len(shapes) >= 50 and len(patterns) >= 40 and len(palettes) >= 40, 'layout/color diversity')
    _require(axes == {0, 1} and len(transitions) >= 7, 'orientation/transition diversity')
    _require(min(repeated, multilane, odd, touching, same_lane) > 0, 'missing reviewed layout variation')
    _require(min(simple, combined, large, small) > 0, 'missing simple, combined, large, or small scenes')
    _require(min(mixed_axes, mixed_palettes, withheld, fewer_test_bars) > 0,
             'missing within-puzzle or train/test variation')
