"""Generate symmetric figures with a width-controlled sliding upper half."""
import hashlib
import numpy as np

_SOURCE_INPUT_HASHES = {
    '23d7f58539ebd04a32105506e6dd4058fcf17248e09abb6da96674592ee02c88',
    '8da8132c03d95776c220395a22c54f1678af561e8e315af3662bc526c3e413fb',
    '156af76a83c2cb62432098a6c5f11063bb5df1b3614068f76666c8a48e3820b1',
}


def _count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer or None')
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    num_train, num_test = _count(num_train, 'num_train'), _count(num_test, 'num_test')
    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError('num_train must be between 2 and 10')
    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError('num_test must be between 1 and 4')
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError('num_train must exceed num_test')
        return num_train, num_test
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        minimum = max(2, num_test + 1)
        num_train = minimum + (seed // 4) % (11 - minimum)
    elif num_train is None:
        minimum = max(2, num_test + 1)
        num_train = minimum + seed % (11 - minimum)
    else:
        num_test = 1 + seed % min(4, num_train - 1)
    return num_train, num_test


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _palettes(rng, train_count, test_count):
    all_pairs = [(a, b) for a in range(10) for b in range(a + 1, 10)]
    first = _choose(rng, all_pairs)
    second = _choose(rng, [p for p in all_pairs if p != first])
    train = [first, second]
    while len(train) < train_count:
        train.append(_choose(rng, train if rng.random() < .4 else all_pairs))
    test = []
    for _ in range(test_count):
        kind = int(rng.integers(3))
        if kind == 0:
            test.append(_choose(rng, train))
        elif kind == 1:
            anchor = _choose(rng, train)
            shared = _choose(rng, anchor)
            choices = [p for p in all_pairs if shared in p and p not in train
                       and len(set(p) & set(anchor)) == 1]
            test.append(_choose(rng, choices or [p for p in all_pairs if p not in train]))
        else:
            used = {c for p in train for c in p}
            novel = [p for p in all_pairs if p not in train and not (set(p) & used)]
            test.append(_choose(rng, novel or [p for p in all_pairs if p not in train]))
    return train, test


def _row_colors(rng, width, a, b):
    # Choose afresh per upper row. The lower half copies complete rows.
    mode = int(rng.integers(6))
    if mode == 0:
        return [a if rng.random() < .5 else b] * width
    if mode == 1:
        return [a] + [b] * (width - 2) + [a] if width > 2 else [a, b][:width]
    if mode == 2:
        return [a if i % 2 else b for i in range(width)]
    if mode == 3:
        cut = int(rng.integers(1, width)) if width > 1 else 1
        return [a] * cut + [b] * (width - cut)
    if mode == 4:
        values = [a] * width
        values[int(rng.integers(width))] = b
        return values
    return [a if rng.random() < .5 else b for _ in range(width)]


def _geometry(rng, middle_width, half_height, force_overwide):
    if rng.random() < .24 and not (force_overwide and middle_width == 9):
        same_parity = [w for w in range(1, 11) if w % 2 == middle_width % 2]
        widths = [int(_choose(rng, same_parity)) for _ in range(half_height - 1)] + [middle_width]
        if force_overwide:
            index = int(rng.integers(half_height - 1))
            widths[index] = int(_choose(rng, [w for w in same_parity if w > middle_width]))
        elif all(w == middle_width for w in widths):
            widths[0] = int(_choose(rng, [w for w in same_parity if w != middle_width]))
        widest = max(widths)
        return widths, [(widest - w) // 2 for w in widths], widest

    widths = [int(rng.integers(1, 11)) for _ in range(half_height - 1)] + [middle_width]
    if force_overwide:
        index = int(rng.integers(half_height - 1))
        widths[index] = int(rng.integers(middle_width + 1, 11))
    elif all(width == middle_width for width in widths):
        widths[0] = 1
    lefts = [0] * half_height
    for row in range(half_height - 2, -1, -1):
        below_left, below_width = lefts[row + 1], widths[row + 1]
        low = max(-4, below_left - widths[row] + 1)
        high = min(4, below_left + below_width - 1)
        lefts[row] = int(rng.integers(low, high + 1))
    minimum = min(lefts)
    lefts = [left - minimum for left in lefts]
    bounding_width = max(left + width for left, width in zip(lefts, widths))
    return widths, lefts, bounding_width


def _edge_position(rng, maximum):
    if maximum == 0:
        return 0
    choice = rng.random()
    if choice < .18:
        return 0
    if choice < .36:
        return maximum
    return int(rng.integers(maximum + 1))


def _canvas_size(rng, figure_height, bounding_width, distance):
    min_height = max(10, figure_height + 1)
    min_width = max(12, bounding_width + distance)
    height = 16 if rng.random() < .22 and min_height <= 16 else int(rng.integers(min_height, 31))
    width = 16 if rng.random() < .22 and min_width <= 16 else int(rng.integers(min_width, 31))
    return height, width


def _apply_rule(grid, background, colors, top, figure_height, distance):
    output = np.full_like(grid, background)
    a, b = colors
    split = top + figure_height // 2
    for row, col in np.argwhere(grid != background):
        value = int(grid[row, col])
        output[row, col + (distance if row < split else 0)] = b if value == a else a
    return output


def _make_pair(rng, palette, middle_width, force_overwide, previous_background):
    a, b = palette
    half_height = int(rng.integers(2, 7))
    widths, lefts, bounding_width = _geometry(rng, middle_width, half_height, force_overwide)
    figure_height = 2 * half_height
    height, width = _canvas_size(rng, figure_height, bounding_width, middle_width)
    # Keep the uniform background uniquely dominant even for broad figures
    # on compact canvases.
    while height * width <= 4 * sum(widths):
        if width < 30:
            width += 1
        else:
            height += 1
    top = _edge_position(rng, height - figure_height)
    left = _edge_position(rng, width - bounding_width - middle_width)
    choices = [c for c in range(10) if c not in palette and c != previous_background]
    background = _choose(rng, choices)
    grid = np.full((height, width), background, dtype=np.int8)
    for index, (row_width, row_left) in enumerate(zip(widths, lefts)):
        colors = _row_colors(rng, row_width, a, b)
        if index == half_height - 1:
            colors[0], colors[-1] = a, b
        start = left + row_left
        grid[top + index, start:start + row_width] = colors
        grid[top + figure_height - 1 - index, start:start + row_width] = colors
    output = _apply_rule(grid, background, palette, top, figure_height, middle_width)
    if grid.shape == (16, 16) and hashlib.sha256(grid.tobytes()).hexdigest() in _SOURCE_INPUT_HASHES:
        replacement = _choose(rng, [c for c in range(10) if c not in palette and c != background])
        grid[grid == background] = replacement
        output[output == background] = replacement
        background = replacement
    return {'input': grid, 'output': output}, background


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    train_palettes, test_palettes = _palettes(rng, num_train, num_test)
    # Puzzle-wide rule: upper half moves right, and the two foreground roles swap.
    # Two distinct widths in training teach the distance rule.
    first = int(rng.integers(2, 7))
    second = _choose(rng, [d for d in range(2, 9) if d != first])
    train_distances = [first, second] + [int(rng.integers(2, 9)) for _ in range(num_train - 2)]
    test_distances = [_choose(rng, train_distances) if rng.random() < .38
                      else int(rng.integers(2, 10)) for _ in range(num_test)]
    train, test = [], []
    previous_background = None
    for index, (palette, distance) in enumerate(zip(train_palettes, train_distances)):
        pair, previous_background = _make_pair(rng, palette, distance,
                                               index == 0 or rng.random() < .45,
                                               previous_background)
        train.append(pair)
    for palette, distance in zip(test_palettes, test_distances):
        pair, previous_background = _make_pair(rng, palette, distance,
                                               rng.random() < .55, previous_background)
        test.append(pair)
    puzzle = {'train': train, 'test': test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _inspect(pair):
    assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
    grid, output = pair['input'], pair['output']
    for item in (grid, output):
        assert isinstance(item, np.ndarray) and item.ndim == 2
        assert 1 <= item.shape[0] <= 30 and 1 <= item.shape[1] <= 30
        assert np.issubdtype(item.dtype, np.integer)
        assert bool(np.all((item >= 0) & (item <= 9)))
    assert grid.shape == output.shape and grid.dtype == output.dtype
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 3
    background = int(colors[np.argmax(counts)])
    foreground = tuple(int(c) for c in colors if c != background)
    mask = grid != background
    occupied = np.argwhere(mask)
    top, left = occupied.min(axis=0)
    bottom, right = occupied.max(axis=0)
    top, left, bottom, right = map(int, (top, left, bottom, right))
    figure_height = bottom - top + 1
    assert figure_height % 2 == 0 and 4 <= figure_height <= 12
    assert int(np.count_nonzero(mask)) < grid.size / 2
    rows = []
    for row in range(top, bottom + 1):
        cols = np.flatnonzero(mask[row])
        assert len(cols) and len(cols) == int(cols[-1] - cols[0] + 1)
        rows.append((int(cols[0]), len(cols)))
    for offset in range(figure_height // 2):
        assert np.array_equal(grid[top + offset], grid[bottom - offset])
    for (l1, w1), (l2, w2) in zip(rows, rows[1:]):
        assert max(l1, l2) < min(l1 + w1, l2 + w2)
    distance = rows[figure_height // 2 - 1][1]
    assert rows[figure_height // 2] == rows[figure_height // 2 - 1]
    assert 2 <= distance <= 9
    assert right + distance < grid.shape[1]
    expected = _apply_rule(grid, background, foreground, top, figure_height, distance)
    assert np.array_equal(output, expected) and not np.array_equal(grid, output)
    assert set(np.unique(output)) == set(colors)
    if grid.shape == (16, 16):
        assert hashlib.sha256(grid.astype(np.int8).tobytes()).hexdigest() not in _SOURCE_INPUT_HASHES
    return {'background': background, 'palette': foreground, 'canvas': grid.shape,
            'top': top, 'left': left, 'height': figure_height,
            'bounding_width': right - left + 1, 'distance': distance, 'rows': tuple(rows)}


def _candidate_output(grid, info, moving_upper, direction, swap, use_middle):
    background = info['background']
    a, b = info['palette']
    split = info['top'] + info['height'] // 2
    distance = info['distance'] if use_middle else info['bounding_width']
    result = np.full_like(grid, background)
    for row, col in np.argwhere(grid != background):
        move = (row < split) == moving_upper
        target = int(col) + (direction * distance if move else 0)
        if not 0 <= target < grid.shape[1]:
            return None
        value = int(grid[row, col])
        result[row, target] = (b if value == a else a) if swap else value
    return result


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    train, test = puzzle['train'], puzzle['test']
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    infos = {'train': [], 'test': []}
    for split, pairs in (('train', train), ('test', test)):
        for pair in pairs:
            info = _inspect(pair)
            infos[split].append(info)
            key = _grid_key(pair['input'])
            assert key not in seen
            seen.add(key)
    train_info = infos['train']
    assert len({i['distance'] for i in train_info}) >= 2
    assert any(i['bounding_width'] > i['distance'] for i in train_info)
    assert len({i['palette'] for i in train_info}) >= 2
    assert len({i['background'] for i in train_info}) >= 2
    # Fit plausible competing rules using training grids alone.
    candidates = []
    for moving_upper in (False, True):
        for direction in (-1, 1):
            for swap in (False, True):
                for use_middle in (False, True):
                    option = moving_upper, direction, swap, use_middle
                    if all(np.array_equal(_candidate_output(pair['input'], info, *option), pair['output'])
                           for pair, info in zip(train, train_info)):
                        candidates.append(option)
    assert candidates == [(True, 1, True, True)]
    for pair, info in zip(test, infos['test']):
        predicted = _candidate_output(pair['input'].copy(), info, *candidates[0])
        assert np.array_equal(predicted, pair['output'])


def _equal(first, second):
    return all(len(first[split]) == len(second[split]) and
               all(np.array_equal(a[key], b[key]) for a, b in zip(first[split], second[split])
                   for key in ('input', 'output')) for split in ('train', 'test'))


def self_test() -> None:
    train_counts, test_counts, heights, widths, distances = set(), set(), set(), set(), set()
    layouts, palettes, backgrounds = set(), set(), set()
    staggered = centered = overwide = near_edge = right_edge = 0
    same_palette = novel_palette = partial_palette = disjoint_palette = 0
    monochrome = compact = large_train = trainlike_test = unseen_test_width = test_overwide = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _equal(puzzle, generate_puzzle(seed))
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        train_palettes = {_inspect(p)['palette'] for p in puzzle['train']}
        train_colors = {color for palette in train_palettes for color in palette}
        train_distances = {_inspect(p)['distance'] for p in puzzle['train']}
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _inspect(pair)
                heights.add(info['height'])
                widths.add(info['bounding_width'])
                distances.add(info['distance'])
                layouts.add((info['canvas'], info['rows'], info['top'], info['left']))
                palettes.add(info['palette'])
                backgrounds.add(info['background'])
                staggered += len({2 * left + width for left, width in info['rows']}) > 1
                centered += len({2 * left + width for left, width in info['rows']}) == 1
                overwide += info['bounding_width'] > info['distance']
                near_edge += info['top'] == 0 or info['left'] == 0 or info['top'] + info['height'] == info['canvas'][0]
                right_edge += info['left'] + info['bounding_width'] + info['distance'] == info['canvas'][1]
                monochrome += any(len(set(pair['input'][info['top'] + row, left:left + width])) == 1
                                  for row, (left, width) in enumerate(info['rows'][:info['height'] // 2]))
                compact += info['canvas'] == (16, 16)
                large_train += split == 'train' and info['height'] >= 10
                if split == 'test':
                    same_palette += info['palette'] in train_palettes
                    novel_palette += info['palette'] not in train_palettes
                    partial_palette += info['palette'] not in train_palettes and bool(set(info['palette']) & train_colors)
                    disjoint_palette += not bool(set(info['palette']) & train_colors)
                    unseen_test_width += info['distance'] not in train_distances
                    test_overwide += info['bounding_width'] > info['distance']
                    trainlike_test += info['height'] in (6, 8)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(heights) >= 4 and len(widths) >= 8 and len(distances) >= 7
    assert len(layouts) >= 120 and len(palettes) >= 20 and len(backgrounds) >= 8
    assert staggered >= 80 and overwide >= 70 and near_edge >= 25 and right_edge >= 10
    assert same_palette >= 10 and novel_palette >= 10
    assert centered >= 15 and monochrome >= 30 and compact >= 3
    assert large_train >= 20 and trainlike_test >= 10 and test_overwide >= 10
    assert unseen_test_width >= 5 and partial_palette >= 5 and disjoint_palette >= 5
    for seed, nt, ns in ((700, 2, 1), (701, 10, 1), (702, 10, 4),
                         (800, None, 1), (801, None, 4), (802, 2, None), (803, 10, None)):
        puzzle = generate_puzzle(seed, nt, ns)
        validate_generated_puzzle(puzzle)
        assert nt is None or len(puzzle['train']) == nt
        assert ns is None or len(puzzle['test']) == ns
    for arguments in ({'num_train': 1}, {'num_train': 11}, {'num_test': 0},
                      {'num_test': 5}, {'num_train': 2, 'num_test': 2},
                      {'num_train': 3, 'num_test': 4}):
        try:
            generate_puzzle(900, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    assert not _equal(generate_puzzle(10001, 6, 3), generate_puzzle(10002, 6, 3))
