import numpy as np


def _normalize(shape):
    points = {(int(r), int(c)) for r, c in shape}
    low_r = min(r for r, _ in points)
    low_c = min(c for _, c in points)
    return tuple(sorted((r - low_r, c - low_c) for r, c in points))


def _orient(shape, rng):
    points = list(shape)
    for _ in range(int(rng.integers(4))):
        points = [(c, -r) for r, c in points]
    if rng.integers(2):
        points = [(r, -c) for r, c in points]
    return _normalize(points)


def _dimensions(shape):
    return (max(r for r, _ in shape) + 1,
            max(c for _, c in shape) + 1)


def _components(grid, foreground):
    unseen = {(int(r), int(c)) for r, c in np.argwhere(grid == foreground)}
    found = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        component = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for neighbor in ((r - 1, c), (r + 1, c),
                             (r, c - 1), (r, c + 1)):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        found.append(component)
    return found


def _translation(component, height):
    fall = height - 1 - max(r for r, _ in component)
    return {(r + fall, c) for r, c in component}


def _swept(component, height):
    fall = height - 1 - max(r for r, _ in component)
    return {(r + step, c) for step in range(fall + 1)
            for r, c in component}


def _transform(grid, background, foreground):
    output = np.full(grid.shape, background, dtype=np.int8)
    occupied = set()
    for component in _components(grid, foreground):
        destination = (component if len(component) <= 3 else
                       _translation(component, grid.shape[0]))
        if destination & occupied:
            raise RuntimeError('destinations overlap')
        occupied.update(destination)
    for r, c in occupied:
        output[r, c] = foreground
    return output


def _resolve_counts(rng, num_train, num_test):
    for name, value, low, high in (
        ('num_train', num_train, 2, 10),
        ('num_test', num_test, 1, 4),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or
            not low <= int(value) <= high
        ):
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_test = int(num_test)
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_train = int(num_train)
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    else:
        num_train, num_test = int(num_train), int(num_test)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return num_train, num_test


def _is_separated(candidate, blocked):
    for r, c in candidate:
        if any(neighbor in blocked for neighbor in
               ((r, c), (r - 1, c), (r + 1, c),
                (r, c - 1), (r, c + 1))):
            return False
    return True


def _small_shape(size, rng):
    if size == 1:
        shape = [(0, 0)]
    elif size == 2:
        shape = [(0, 0), (0, 1)]
    elif rng.integers(2):
        shape = [(0, 0), (0, 1), (0, 2)]
    else:
        shape = [(0, 0), (0, 1), (1, 0)]
    return _orient(shape, rng)


def _mover_shape(rng, family=None):
    if family is None:
        family = ('rectangle', 'bar', 'hollow', 'step', 'poly')[
            int(rng.choice(5, p=[.29, .15, .16, .22, .18]))]
    if family == 'rectangle':
        h = int(rng.integers(2, 5))
        w = int(rng.integers(2, 6))
        shape = [(r, c) for r in range(h) for c in range(w)]
    elif family == 'bar':
        length = int(rng.integers(4, 9))
        shape = [(0, c) for c in range(length)]
    elif family == 'hollow':
        h = int(rng.integers(3, 6))
        w = int(rng.integers(3, 6))
        shape = [(r, c) for r in range(h) for c in range(w)
                 if r in (0, h - 1) or c in (0, w - 1)]
        if rng.integers(4) == 0:
            shape.remove((0, int(rng.integers(1, w - 1))))
    elif family == 'step':
        h1, w1, h2, w2 = [int(rng.integers(2, 4)) for _ in range(4)]
        offset = int(rng.integers(1, w1))
        shape = ([(r, c) for r in range(h1) for c in range(w1)] +
                 [(h1 - 1 + r, offset + c)
                  for r in range(h2) for c in range(w2)])
    elif family == 'poly':
        target = int(rng.integers(4, 14))
        points = {(0, 0)}
        while len(points) < target:
            r, c = tuple(points)[int(rng.integers(len(points)))]
            dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[
                int(rng.integers(4))]
            points.add((r + dr, c + dc))
        shape = points
    else:
        raise ValueError('unknown shape family')
    return _orient(shape, rng)


def _sample_dimensions(rng, regime, index):
    if regime == 'source_train' and index == 0:
        return int(rng.integers(7, 13)), int(rng.integers(7, 13))
    if regime == 'source_train' and index == 1:
        return int(rng.integers(11, 19)), int(rng.integers(11, 19))
    if regime == 'target_test' and rng.integers(4) != 0:
        return int(rng.integers(11, 21)), int(rng.integers(11, 21))
    return int(rng.integers(7, 21)), int(rng.integers(7, 21))


def _sample_fall(rng, maximum):
    if maximum == 0:
        return 0
    category = int(rng.choice(4, p=[.13, .24, .18, .45]))
    if category == 0:
        return 0
    if category == 1:
        return 1
    if category == 2:
        return min(2, maximum)
    return int(rng.integers(1, maximum + 1))


def _generate_pair(rng, regime, index, background, foreground):
    for attempt in range(240):
        height, width = _sample_dimensions(rng, regime, index)
        anchor = regime == 'source_train' and index == 0
        complex_demo = regime == 'source_train' and index == 1
        max_movers = min(4, max(1, (width - 1) // 3))
        if anchor:
            mover_count = 1
        elif width >= 14 and height >= 13 and rng.integers(5) != 0:
            mover_count = int(rng.integers(2, max_movers + 1))
        else:
            mover_count = int(rng.integers(1, max_movers + 1))
        shapes = []
        for mover_index in range(mover_count):
            if anchor and mover_index == 0:
                family = 'rectangle' if rng.integers(2) else 'bar'
                shape = _orient([(r, c) for r in range(2) for c in range(2)]
                                if family == 'rectangle' else
                                [(0, c) for c in range(4)], rng)
            elif complex_demo and mover_index == 0:
                shape = _mover_shape(rng, 'step' if rng.integers(2)
                                     else 'hollow')
            else:
                shape = _mover_shape(rng)
            shapes.append(shape)
        rng.shuffle(shapes)
        sizes = [_dimensions(shape) for shape in shapes]
        packed = sum(w for _, w in sizes) + len(shapes) - 1
        if packed > width or any(h >= height for h, _ in sizes):
            continue
        padding = rng.multinomial(width - packed,
                                  np.full(len(shapes) + 1, 1 / (len(shapes) + 1)))
        lefts = []
        cursor = int(padding[0])
        for j, (_, shape_width) in enumerate(sizes):
            lefts.append(cursor)
            cursor += shape_width + (1 + int(padding[j + 1])
                                     if j + 1 < len(shapes) else 0)
        moving = []
        for shape, (shape_height, _), left in zip(shapes, sizes, lefts):
            maximum = height - shape_height
            fall = _sample_fall(rng, maximum)
            if anchor and fall == 0:
                fall = int(rng.integers(1, maximum + 1))
            moving.append({(height - shape_height - fall + r, left + c)
                           for r, c in shape})
        if all(max(r for r, _ in obj) == height - 1 for obj in moving):
            continue
        moving_input = set().union(*moving)
        moving_dest = set().union(*(_translation(obj, height) for obj in moving))
        swept = set().union(*(_swept(obj, height) for obj in moving))
        if anchor:
            small_sizes = [3]
            if rng.integers(2):
                small_sizes.append(int(rng.integers(1, 4)))
        else:
            count = int(rng.integers(1, min(4, max(2, height * width // 65)) + 1))
            small_sizes = [int(rng.integers(1, 4)) for _ in range(count)]
        stationary = set()
        placed_all = True
        for size in small_sizes:
            placed = None
            for _ in range(140):
                shape = _small_shape(size, rng)
                sh, sw = _dimensions(shape)
                if sh > height or sw > width:
                    continue
                top = int(rng.integers(height - sh + 1))
                left = int(rng.integers(width - sw + 1))
                candidate = {(top + r, left + c) for r, c in shape}
                if anchor and size == 3 and max(r for r, _ in candidate) == height - 1:
                    continue
                if (not candidate & swept and
                    _is_separated(candidate, moving_input | stationary) and
                    _is_separated(candidate, moving_dest | stationary)):
                    placed = candidate
                    break
            if placed is None:
                placed_all = False
                break
            stationary.update(placed)
        if not placed_all:
            continue
        total = len(moving_input) + len(stationary)
        if total / (height * width) > .26:
            continue
        if total / (height * width) < .06 and rng.integers(5) != 0:
            continue
        grid = np.full((height, width), background, dtype=np.int8)
        for r, c in moving_input | stationary:
            grid[r, c] = foreground
        if len(_components(grid, foreground)) != len(moving) + len(small_sizes):
            continue
        output = _transform(grid, background, foreground)
        return grid, output
    raise RuntimeError('could not construct separated components')

def _parse_grid(text):
    rows = [
        row.strip()
        for row in text.strip().splitlines()
    ]
    return np.array(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_PAIRS = (
    (
        _parse_grid(
            """
            7777777
            7667667
            7767667
            7777767
            7777767
            7777777
            7777777
            """
        ),
        _parse_grid(
            """
            7777777
            7667777
            7767777
            7777667
            7777667
            7777767
            7777767
            """
        ),
    ),
    (
        _parse_grid(
            """
            777777777777777
            755777777555777
            755777557555777
            777757557555757
            777777557777757
            777777777777757
            777777777777777
            777777777777777
            777777777777777
            777777777777777
            777777777777777
            777777777777777
            777777777777777
            777777777777777
            777777777777777
            """
        ),
        _parse_grid(
            """
            777777777777777
            777777777777777
            777777777777777
            777757777777757
            777777777777757
            777777777777757
            777777777777777
            777777777777777
            777777777777777
            777777777777777
            777777777777777
            777777777777777
            777777557555777
            755777557555777
            755777557555777
            """
        ),
    ),
    (
        _parse_grid(
            """
            7777777777777777
            7117777777777777
            7117177777711117
            7117177777717717
            7117777777711117
            7117777771777777
            7777777771777777
            7777777771777777
            7777777177777777
            7777771177777777
            7777771177777777
            7777777777777777
            7777777777777777
            7777777777777777
            7777777777777777
            7777777777777777
            """
        ),
        _parse_grid(
            """
            7777777777777777
            7777777777777777
            7777177777777777
            7777177777777777
            7777777777777777
            7777777771777777
            7777777771777777
            7777777771777777
            7777777777777777
            7777777777777777
            7777777777777777
            7117777777777777
            7117777777777777
            7117777177711117
            7117771177717717
            7117771177711117
            """
        ),
    ),
    (
        _parse_grid(
            """
            77777777777777
            74474747447447
            74474777447447
            77774774447447
            77777774447447
            77777777777777
            77777777777777
            77777777777777
            77777777777777
            77777777777777
            77777777777777
            77777777777777
            77777777777777
            77777777777777
            """
        ),
        _parse_grid(
            """
            77777777777777
            77774747777777
            77774777777777
            77774777777777
            77777777777777
            77777777777777
            77777777777777
            77777777777777
            77777777777777
            77777777777777
            77777777447447
            77777777447447
            74477774447447
            74477774447447
            """
        ),
    ),
)



def _array_key(array):
    return tuple(array.shape), tuple(int(value) for value in array.ravel())


def _is_source_pair(input_grid, output_grid):
    return any(np.array_equal(input_grid, source_input) and
               np.array_equal(output_grid, source_output)
               for source_input, source_output in _SOURCE_PAIRS)


def _palette_plan(rng, num_train, num_test):
    shared_background = bool(rng.integers(2))
    base = int(rng.integers(10))
    palettes = []
    for index in range(num_train + num_test):
        if shared_background:
            background = base
        elif index and rng.integers(4) == 0:
            background = palettes[-1][1]
        else:
            background = int(rng.integers(10))
        available = [c for c in range(10) if c != background]
        if index == 1:
            available = [c for c in available if c != palettes[0][1]]
        if index >= num_train and rng.integers(5) < 3:
            reused = [fg for _, fg in palettes[:num_train] if fg in available]
            foreground = (int(reused[int(rng.integers(len(reused)))])
                          if reused else int(rng.choice(available)))
        elif index > 1 and rng.integers(3) == 0:
            reused = [fg for _, fg in palettes[:index] if fg in available]
            foreground = (int(reused[int(rng.integers(len(reused)))])
                          if reused else int(rng.choice(available)))
        else:
            foreground = int(rng.choice(available))
        palettes.append((background, foreground))
    return palettes


def _analyze_pair(input_grid, output_grid):
    assert input_grid.shape == output_grid.shape
    values, counts = np.unique(input_grid, return_counts=True)
    assert len(values) == 2 and counts[0] != counts[1]
    background = int(values[int(np.argmax(counts))])
    foreground = int(values[int(np.argmin(counts))])
    assert set(int(x) for x in np.unique(output_grid)) == {background, foreground}
    components = _components(input_grid, foreground)
    small = [piece for piece in components if len(piece) <= 3]
    large = [piece for piece in components if len(piece) >= 4]
    assert small and large
    assert any(max(r for r, _ in piece) < input_grid.shape[0] - 1
               for piece in large)
    occupied = set()
    for piece in components:
        destination = (piece if len(piece) <= 3 else
                       _translation(piece, input_grid.shape[0]))
        assert _is_separated(destination, occupied)
        occupied.update(destination)
    for mover in large:
        path = _swept(mover, input_grid.shape[0])
        assert not any(path & fixed for fixed in small)
        assert not any(path & other for other in large if other is not mover)
    expected = np.full(input_grid.shape, background, dtype=np.int8)
    for r, c in occupied:
        expected[r, c] = foreground
    assert np.array_equal(expected, output_grid)
    assert not np.array_equal(input_grid, output_grid)
    return {
        'background': background,
        'foreground': foreground,
        'stationary_sizes': tuple(sorted(map(len, small))),
        'moving_sizes': tuple(sorted(map(len, large))),
        'moving_shapes': tuple(_normalize(piece) for piece in large),
        'falls': tuple(input_grid.shape[0] - 1 - max(r for r, _ in piece)
                       for piece in large),
        'density': sum(map(len, components)) / input_grid.size,
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen_inputs, seen_pairs = set(), set()
    analyses = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            input_grid, output_grid = pair['input'], pair['output']
            input_key = _array_key(input_grid)
            pair_key = (input_key, _array_key(output_grid))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert not _is_source_pair(input_grid, output_grid)
            info = _analyze_pair(input_grid, output_grid)
            analyses[split].append(info)
            assert 7 <= input_grid.shape[0] <= 20
            assert 7 <= input_grid.shape[1] <= 20
            if split == 'train' and len(analyses['train']) == 1:
                assert max(input_grid.shape) <= 12
                assert info['moving_sizes'] == (4,)
                assert info['falls'][0] > 0
                assert any(
                    len(component) == 3 and
                    max(r for r, _ in component) < input_grid.shape[0] - 1
                    for component in _components(input_grid, info['foreground'])
                )
            if split == 'train' and len(analyses['train']) == 2:
                assert min(input_grid.shape) >= 11
    train = analyses['train']
    assert any(3 in info['stationary_sizes'] for info in train)
    assert any(4 in info['moving_sizes'] for info in train)
    assert len({info['foreground'] for info in train}) >= 2
    assert any(any(fall > 0 for fall in info['falls']) for info in train)
    # A single size threshold and downward direction explain all demonstrations.
    assert all(info['moving_sizes'] and info['stationary_sizes']
               for info in train + analyses['test'])


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    palettes = _palette_plan(rng, num_train, num_test)
    pair_seeds = rng.integers(0, np.iinfo(np.uint64).max,
                              size=num_train + num_test, dtype=np.uint64)
    puzzle = {'train': [], 'test': []}
    seen = set()
    for global_index, (background, foreground) in enumerate(palettes):
        split = 'train' if global_index < num_train else 'test'
        regime = 'source_train' if split == 'train' else 'target_test'
        local_index = (global_index if split == 'train' else
                       global_index - num_train)
        for retry in range(12):
            child_rng = np.random.default_rng(np.random.SeedSequence(
                [int(pair_seeds[global_index]), global_index, retry]))
            try:
                input_grid, output_grid = _generate_pair(
                    child_rng, regime, local_index, background, foreground)
            except RuntimeError:
                continue
            key = _array_key(input_grid)
            if key in seen or _is_source_pair(input_grid, output_grid):
                continue
            seen.add(key)
            puzzle[split].append({'input': input_grid, 'output': output_grid})
            break
        else:
            raise RuntimeError('could not construct a distinct pair')
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple((split, _array_key(pair['input']), _array_key(pair['output']))
                 for split in ('train', 'test') for pair in puzzle[split])


def self_test() -> None:
    automatic = generate_puzzle(314159)
    validate_generated_puzzle(automatic)
    explicit = generate_puzzle(271828, 7, 3)
    assert _puzzle_key(explicit) == _puzzle_key(generate_puzzle(271828, 7, 3))
    assert _puzzle_key(generate_puzzle(1, 6, 2)) != _puzzle_key(
        generate_puzzle(2, 6, 2))
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(1000 + 10 * train_count + test_count,
                                 train_count, test_count)
        assert (len(puzzle['train']), len(puzzle['test'])) == (train_count, test_count)
    for index, (train_count, test_count) in enumerate(
        ((None, 1), (None, 4), (2, None), (10, None))
    ):
        puzzle = generate_puzzle(2000 + index, train_count, test_count)
        if train_count is not None:
            assert len(puzzle['train']) == train_count
        if test_count is not None:
            assert len(puzzle['test']) == test_count
    for args in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (2, 3), (4, 4), (3.0, 1)):
        try:
            generate_puzzle(0, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f'illegal counts accepted: {args}')
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    dimensions, shapes, sizes, counts, backgrounds = set(), set(), set(), set(), set()
    short_falls = aligned_with_faller = small_grids = low_density = total = 0
    changed_backgrounds = reused_foreground = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        infos = []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                grid = pair['input']
                info = _analyze_pair(grid, pair['output'])
                infos.append(info)
                dimensions.add(grid.shape)
                shapes.update(info['moving_shapes'])
                sizes.update(info['moving_sizes'])
                counts.add((len(info['moving_sizes']), len(info['stationary_sizes'])))
                backgrounds.add(info['background'])
                short_falls += any(fall in (1, 2) for fall in info['falls'])
                aligned_with_faller += (0 in info['falls'] and
                                        any(fall > 0 for fall in info['falls']))
                small_grids += max(grid.shape) <= 9
                low_density += info['density'] < .06
                total += 1
        changed_backgrounds += len({i['background'] for i in infos}) > 1
        train_foregrounds = {i['foreground'] for i in infos[:len(puzzle['train'])]}
        reused_foreground += any(i['foreground'] in train_foregrounds
                                 for i in infos[len(puzzle['train']):])
    assert len(dimensions) >= 45 and len(shapes) >= 50
    assert len(sizes) >= 10 and len(counts) >= 7 and len(backgrounds) >= 8
    assert short_falls >= 20 and aligned_with_faller >= 5 and small_grids >= 5
    assert low_density < total // 4
    assert changed_backgrounds >= 10 and reused_foreground >= 10
