import numpy as np


_SOURCE_SPECS = (
    (4, 5, ((1, 2), (3, 1))),
    (3, 6, ((1, 2), (2, 0))),
    (5, 8, ((1, 1), (2, 1))),
    (4, 1, ((3, 0),)),
    (2, 7, ((1, 1),)),
    (4, 4, ((1, 2), (3, 0))),
)
_SHIFTS = tuple((dr, dc) for dr in (-1, 0, 1)
                for dc in (-1, 0, 1) if (dr, dc) != (0, 0))


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer or None')
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f'{name} must be between {low} and {high}')
    return value


def _counts(seed, num_train, num_test):
    train = _count('num_train', num_train, 2, 10)
    test = _count('num_test', num_test, 1, 4)
    if train is None and test is None:
        test = 1 + int(seed) % 4
        low = max(2, test + 1)
        train = low + (int(seed) // 4) % (11 - low)
    elif train is None:
        low = max(2, test + 1)
        train = low + int(seed) % (11 - low)
    elif test is None:
        test = 1 + int(seed) % min(4, train - 1)
    if train <= test:
        raise ValueError('num_train must exceed num_test')
    return train, test


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    index = int(np.argmax(counts))
    assert counts[index] > grid.size // 2
    assert len(colors) >= 2
    return int(colors[index])


def _apply_rule(grid, background, construction, shift=(-1, -1)):
    height, width = grid.shape
    mask = grid != background
    augmented = grid.copy()
    for r, c in np.argwhere(mask):
        target = ((int(r) + shift[0]) % height,
                  (int(c) + shift[1]) % width)
        if not mask[target]:
            augmented[target] = construction
    return np.tile(augmented, (3, 3))


def _stats(mask):
    height, width = mask.shape
    positions = np.argwhere(mask)
    visible = []
    collisions = 0
    for r, c in positions:
        r, c = int(r), int(c)
        target = ((r - 1) % height, (c - 1) % width)
        if mask[target]:
            collisions += 1
        else:
            visible.append((r, c))
    return {
        'count': len(positions), 'collisions': collisions,
        'visible': len(visible),
        'top': any(r == 0 and c > 0 for r, c in visible),
        'left': any(c == 0 and r > 0 for r, c in visible),
        'corner': any(r == 0 and c == 0 for r, c in visible),
        'interior': any(r > 0 and c > 0 for r, c in visible),
    }


def _dimensions(rng, test=False, minimum=2):
    if test and rng.random() < 0.17:
        return 4, 4
    small_probability = 0.42 if test else 0.57
    if rng.random() < small_probability:
        high = 5
        low = minimum
    else:
        low, high = max(minimum, 5), 10
    height = int(rng.integers(low, high + 1))
    if rng.random() < 0.42:
        width = height
    else:
        if rng.random() < small_probability:
            width = int(rng.integers(minimum, 6))
        else:
            width = int(rng.integers(max(minimum, 5), 11))
    return height, width


def _direction_witness(rng):
    for _ in range(64):
        height, width = _dimensions(rng, minimum=3)
        cells = {(0, int(rng.integers(1, width))),
                 (int(rng.integers(1, height)), 0)}
        if rng.random() < 0.48:
            cells.add((int(rng.integers(1, height)),
                       int(rng.integers(1, width))))
        mask = np.zeros((height, width), dtype=bool)
        for cell in cells:
            mask[cell] = True
        stats = _stats(mask)
        if (stats['top'] and stats['left'] and stats['visible'] >= 2
                and stats['collisions'] == 0
                and mask.sum() < mask.size / 2):
            return mask
    mask = np.zeros((4, 5), dtype=bool)
    mask[0, 2] = mask[2, 0] = mask[3, 3] = True
    return mask


def _collision_example(rng, other_shape):
    for _ in range(64):
        height, width = _dimensions(rng, minimum=3)
        if (height, width) == other_shape:
            continue
        length = int(rng.integers(2, min(height, width, 6)))
        row = int(rng.integers(0, height - length + 1))
        col = int(rng.integers(0, width - length + 1))
        mask = np.zeros((height, width), dtype=bool)
        for step in range(length):
            mask[row + step, col + step] = True
        extras = int(rng.integers(0, 4))
        for _ in range(extras):
            if mask.sum() + 1 >= mask.size / 2:
                break
            mask[int(rng.integers(height)), int(rng.integers(width))] = True
        stats = _stats(mask)
        if stats['collisions'] and stats['visible'] and mask.sum() < mask.size / 2:
            return mask
    mask = np.zeros((5, 4), dtype=bool)
    mask[1, 1] = mask[2, 2] = True
    return mask


def _motif(rng, height, width, mode):
    area = height * width
    limit = max(1, (area - 1) // 2)
    if mode == 'singleton' or limit == 1:
        return {(0, 0)}
    if mode == 'line':
        horizontal = bool(rng.integers(2))
        length = int(rng.integers(2, min(7, width if horizontal else height, limit + 1) + 1))
        return {(0, c) for c in range(length)} if horizontal else {(r, 0) for r in range(length)}
    if mode == 'block' and height >= 3 and width >= 3 and limit >= 4:
        bh = int(rng.integers(2, min(4, height) + 1))
        bw = int(rng.integers(2, min(4, width, max(2, limit // bh)) + 1))
        return {(r, c) for r in range(bh) for c in range(bw)}
    if mode == 'diagonal':
        length = int(rng.integers(2, min(6, height, width, limit) + 1))
        return {(i, i) for i in range(length)}
    if mode == 'irregular':
        target = int(rng.integers(3, min(limit, 15) + 1)) if limit >= 3 else limit
        cells = {(height // 2, width // 2)}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            choices = sorted(frontier)
            cells.add(choices[int(rng.integers(len(choices)))])
        return cells
    if mode == 'separated':
        target = int(rng.integers(2, min(limit, 9) + 1))
        cells = set()
        choices = [(r, c) for r in range(height) for c in range(width)]
        for index in rng.permutation(len(choices)):
            r, c = choices[int(index)]
            if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in cells):
                cells.add((r, c))
                if len(cells) == target:
                    break
        return cells
    if mode == 'clusters' and height >= 4 and width >= 3 and limit >= 4:
        gap = int(rng.integers(2, height - 1))
        first_col = int(rng.integers(0, width - 1))
        second_col = int(rng.integers(0, width - 1))
        return {(0, first_col), (0, first_col + 1),
                (gap, second_col), (gap, second_col + 1)}
    target = int(rng.integers(2, min(limit, max(3, int(area * 0.36)), 22) + 1))
    indices = rng.choice(area, size=target, replace=False)
    return {(int(index) // width, int(index) % width) for index in indices}


def _place_motif(rng, height, width, cells, boundary):
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    cells = {(r - min_r, c - min_c) for r, c in cells}
    mh = 1 + max(r for r, _ in cells)
    mw = 1 + max(c for _, c in cells)
    if boundary == 'interior':
        if mh > height - 1 or mw > width - 1:
            return None
        row = int(rng.integers(1, height - mh + 1))
        col = int(rng.integers(1, width - mw + 1))
    elif boundary == 'top':
        if mw > width - 1:
            return None
        row = 0
        col = int(rng.integers(1, width - mw + 1))
    elif boundary == 'left':
        if mh > height - 1:
            return None
        row = int(rng.integers(1, height - mh + 1))
        col = 0
    elif boundary == 'corner':
        row = col = 0
    else:
        row = int(rng.integers(0, height - mh + 1))
        col = int(rng.integers(0, width - mw + 1))
    placed = {(r + row, c + col) for r, c in cells}
    if boundary == 'corner':
        placed.add((0, 0))
    if boundary == 'mixed' and not any(r > 0 and c > 0 for r, c in placed):
        placed.add((int(rng.integers(1, height)), int(rng.integers(1, width))))
    if len(placed) >= height * width / 2:
        return None
    mask = np.zeros((height, width), dtype=bool)
    for cell in placed:
        mask[cell] = True
    return mask


def _general_mask(rng, test, used_structures):
    if test and rng.random() < 0.075:
        side = int(rng.integers(3, 9))
        cycle = np.eye(side, dtype=bool)
        if _structure_key(cycle) not in used_structures:
            return cycle
    if rng.random() < 0.12:
        for index in rng.permutation(4):
            small = np.zeros((2, 2), dtype=bool)
            small[int(index) // 2, int(index) % 2] = True
            if _structure_key(small) not in used_structures:
                return small
    modes = ('singleton', 'line', 'block', 'irregular', 'separated',
             'clusters', 'diagonal', 'random')
    boundaries = ('interior', 'top', 'left', 'corner', 'mixed', 'free')
    mode = modes[int(rng.integers(len(modes)))]
    boundary = boundaries[int(rng.integers(len(boundaries)))]
    for attempt in range(80):
        height, width = _dimensions(rng, test=test)
        if boundary != 'free' and (height < 3 or width < 3):
            continue
        cells = _motif(rng, height, width, mode)
        mask = _place_motif(rng, height, width, cells, boundary)
        if mask is None:
            continue
        key = _structure_key(mask)
        if key in used_structures:
            continue
        stats = _stats(mask)
        if not test and stats['visible'] == 0:
            continue
        if boundary == 'top' and not stats['top']:
            continue
        if boundary == 'left' and not stats['left']:
            continue
        if boundary == 'corner' and not stats['corner']:
            continue
        if boundary == 'interior' and (stats['top'] or stats['left'] or stats['corner']):
            continue
        return mask
    # A deterministic escape for rare placement or duplication failures.
    mask = np.zeros((10, 10), dtype=bool)
    for r in range(10):
        for c in range(10):
            mask.fill(False)
            mask[r, c] = True
            if _structure_key(mask) not in used_structures:
                return mask.copy()
    raise AssertionError('no unused mask')


def _structure_key(mask):
    return (mask.shape, tuple(map(tuple, np.argwhere(mask))))


def _grid_key(grid):
    return (grid.shape, tuple(int(x) for x in grid.ravel()))


def _source_copy(input_grid, output_grid):
    for side, color, positions in _SOURCE_SPECS:
        source = np.zeros((side, side), dtype=np.int8)
        for r, c in positions:
            source[r, c] = color
        if np.array_equal(input_grid, source) and np.array_equal(
                output_grid, _apply_rule(source, 0, 2)):
            return True
    return False


def _make_pair(mask, background, foreground_colors, construction, rng):
    positions = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
    rng.shuffle(positions)
    grid = np.full(mask.shape, background, dtype=np.int8)
    for index, position in enumerate(positions):
        color = foreground_colors[index] if index < len(foreground_colors) else \
            foreground_colors[int(rng.integers(len(foreground_colors)))]
        grid[position] = color
    output = _apply_rule(grid, background, construction)
    if _source_copy(grid, output):
        replacement = next(color for color in range(10)
                           if color not in (background, construction, foreground_colors[0]))
        grid[mask] = replacement
        output = _apply_rule(grid, background, construction)
    return {'input': grid, 'output': output}


def _compatible_shifts(train):
    candidates = set(_SHIFTS)
    for pair in train:
        grid = pair['input']
        background = _background(grid)
        new_colors = set(map(int, np.unique(pair['output']))) - set(map(int, np.unique(grid)))
        if len(new_colors) != 1:
            return set()
        construction = next(iter(new_colors))
        candidates = {shift for shift in candidates if np.array_equal(
            pair['output'], _apply_rule(grid, background, construction, shift))}
    return candidates


def _predict_from_training(train, test_input):
    assert _compatible_shifts(train) == {(-1, -1)}
    constructions = []
    for pair in train:
        new = set(map(int, np.unique(pair['output']))) - set(map(int, np.unique(pair['input'])))
        assert len(new) == 1
        constructions.append(next(iter(new)))
    assert len(set(constructions)) == 1
    return _apply_rule(test_input, _background(test_input), constructions[0])


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    construction = int(rng.integers(10))
    palette = [int(x) for x in rng.permutation(10) if x != construction]
    backgrounds = palette[:2]
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max,
                              size=train_count + test_count, dtype=np.int64)
    pair_rngs = [np.random.default_rng(int(value)) for value in pair_seeds]

    witness = _direction_witness(pair_rngs[0])
    collision = _collision_example(pair_rngs[1], witness.shape)
    masks = [witness, collision]
    used_structures = {_structure_key(witness), _structure_key(collision)}
    for index in range(2, train_count + test_count):
        mask = _general_mask(pair_rngs[index], index >= train_count, used_structures)
        masks.append(mask)
        used_structures.add(_structure_key(mask))

    train = []
    train_foregrounds = set()
    trained_multi = False
    first_colors = []
    for index in range(train_count):
        pair_rng = pair_rngs[index]
        background = backgrounds[index] if index < 2 else palette[int(pair_rng.integers(len(palette)))]
        choices = [color for color in palette if color != background]
        if index == 1:
            other = [color for color in choices if color not in first_colors]
            if other:
                choices = other
        first = choices[int(pair_rng.integers(len(choices)))]
        first_colors.append(first)
        selected = [first]
        if (index != 1 and masks[index].sum() >= 2
                and _stats(masks[index])['collisions'] == 0
                and pair_rng.random() < 0.55):
            extras = [color for color in palette if color not in (background, first)]
            pair_rng.shuffle(extras)
            extra_count = min(len(extras), int(masks[index].sum()) - 1,
                              int(pair_rng.integers(1, 3)))
            selected.extend(extras[:extra_count])
            trained_multi = True
        pair = _make_pair(masks[index], background, selected, construction, pair_rng)
        train.append(pair)
        train_foregrounds.update(int(x) for x in np.unique(pair['input']) if x != background)

    test = []
    start_novel = int(rng.integers(2))
    for test_index in range(test_count):
        index = train_count + test_index
        pair_rng = pair_rngs[index]
        background = palette[int(pair_rng.integers(len(palette)))]
        familiar = [color for color in train_foregrounds if color != background]
        novel = [color for color in palette if color != background and color not in train_foregrounds]
        use_novel = bool(start_novel) if test_count == 1 else \
            bool((test_index + start_novel) % 2)
        choices = novel if use_novel and novel else familiar
        if not choices:
            choices = [color for color in palette if color != background]
        first = choices[int(pair_rng.integers(len(choices)))]
        selected = [first]
        if trained_multi and masks[index].sum() >= 2 and pair_rng.random() < 0.45:
            extras = [color for color in palette if color not in (background, first)]
            pair_rng.shuffle(extras)
            selected.extend(extras[:min(2, len(extras), int(masks[index].sum()) - 1)])
        test.append(_make_pair(masks[index], background, selected, construction, pair_rng))

    # Shuffle examples after building the two required demonstrations.
    train_order = rng.permutation(train_count)
    train = [train[int(i)] for i in train_order]
    puzzle = {'train': train, 'test': test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen_inputs, seen_pairs, seen_structures = set(), set(), set()
    constructions, backgrounds, train_colors, train_shapes = [], [], set(), set()
    train_stats, multi_train = [], False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert 0 <= int(grid.min()) and int(grid.max()) <= 9
            input_grid, output_grid = pair['input'], pair['output']
            h, w = input_grid.shape
            assert 2 <= h <= 10 and 2 <= w <= 10
            assert output_grid.shape == (3 * h, 3 * w)
            input_key = _grid_key(input_grid)
            pair_key = (input_key, _grid_key(output_grid))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            background = _background(input_grid)
            backgrounds.append(background)
            mask = input_grid != background
            structure = _structure_key(mask)
            assert structure not in seen_structures
            seen_structures.add(structure)
            assert not _source_copy(input_grid, output_grid)
            stats = _stats(mask)
            input_colors = set(map(int, np.unique(input_grid)))
            output_colors = set(map(int, np.unique(output_grid)))
            new_colors = output_colors - input_colors
            if split == 'train':
                assert stats['visible'] >= 1 and len(new_colors) == 1
                constructions.extend(new_colors)
                train_stats.append(stats)
                train_shapes.add((h, w))
                foregrounds = input_colors - {background}
                train_colors.update(foregrounds)
                multi_train |= len(foregrounds) > 1 and stats['collisions'] == 0
            else:
                assert new_colors <= set(constructions)
                if len(input_colors - {background}) > 1:
                    assert multi_train
            assert constructions
            construction = constructions[0]
            assert construction not in input_colors
            assert new_colors <= {construction}
            assert np.array_equal(output_grid,
                                  _apply_rule(input_grid, background, construction))
    assert len(set(constructions)) == 1
    assert len(set(backgrounds)) >= 2
    assert len(train_colors) >= 2
    assert len(train_shapes) >= 2
    assert any(item['collisions'] >= 1 for item in train_stats)
    assert any(item['top'] for item in train_stats)
    assert any(item['left'] for item in train_stats)
    assert _compatible_shifts(puzzle['train']) == {(-1, -1)}
    for pair in puzzle['test']:
        assert np.array_equal(_predict_from_training(puzzle['train'], pair['input']),
                              pair['output'])


def _puzzles_equal(first, second):
    return all(np.array_equal(a[key], b[key])
               for split in ('train', 'test')
               for a, b in zip(first[split], second[split])
               for key in ('input', 'output')) and all(
                   len(first[split]) == len(second[split]) for split in ('train', 'test'))


def self_test() -> None:
    assert _puzzles_equal(generate_puzzle(8128), generate_puzzle(8128))
    assert _puzzles_equal(generate_puzzle(99173, 7, 3), generate_puzzle(99173, 7, 3))
    assert not _puzzles_equal(generate_puzzle(70), generate_puzzle(71))
    for seed, train, test in ((301, 2, 1), (302, 10, 1), (303, 10, 4),
                              (401, None, 1), (402, None, 4),
                              (403, 2, None), (404, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        if train is not None:
            assert len(puzzle['train']) == train
        if test is not None:
            assert len(puzzle['test']) == test
        validate_generated_puzzle(puzzle)
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (2.0, 1), (True, 1)):
        try:
            generate_puzzle(500, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    train_counts, test_counts = set(), set()
    shapes, structures, colors = set(), set(), set()
    categories = {'train': set(), 'test': set()}
    multi_splits, crowded, familiar, novel = set(), 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        train_colors = set()
        for pair in puzzle['train']:
            background = _background(pair['input'])
            train_colors.update(set(map(int, np.unique(pair['input']))) - {background})
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                grid, output = pair['input'], pair['output']
                background = _background(grid)
                mask = grid != background
                stats = _stats(mask)
                shapes.add(grid.shape)
                structures.add(_structure_key(mask))
                colors.update(map(int, np.unique(grid)))
                assert np.array_equal(output, _predict_from_training(puzzle['train'], grid))
                assert not _source_copy(grid, output)
                category = ('corner' if stats['corner'] else
                            'both' if stats['top'] and stats['left'] else
                            'top' if stats['top'] else
                            'left' if stats['left'] else 'interior')
                categories[split].add(category)
                if len(set(map(int, np.unique(grid))) - {background}) > 1:
                    multi_splits.add(split)
                if stats['count'] > 2:
                    crowded += 1
                if split == 'test':
                    if (set(map(int, np.unique(grid))) - {background}) & train_colors:
                        familiar += 1
                    if (set(map(int, np.unique(grid))) - {background}) - train_colors:
                        novel += 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 20 and len(structures) >= 130
    assert len(colors) == 10 and crowded >= 35
    assert {'train', 'test'} <= multi_splits
    assert familiar >= 10 and novel >= 10
    assert {'corner', 'both', 'top', 'left', 'interior'} <= categories['train']
    assert {'corner', 'both', 'top', 'left', 'interior'} <= categories['test']
