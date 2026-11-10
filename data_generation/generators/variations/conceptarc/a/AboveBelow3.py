import numpy as np


_SOURCE_INPUTS = {
    tuple(text.split('/')) for text in (
        '00000000/03330000/00000000/04440000/04440000/04440000/00000333/00333000',
        '0022220000/0000000000/0000000000/0066660000/0066660000/0066660000/0066660000/2222200000/0000000000/0000222000',
        '0044000/0000000/0011000/0011000/0444000/0000000/0000000',
        '001111100/000000000/000000000/000000000/005555500/005555500/005555500/005555500/005555500/000000000/000000000/001111100',
        '000011100/000000000/004444400/004000400/004000400/004000400/004444400/000000000/000111100',
        '0000000000/0005555000/0000000000/0005555000/0005555000/0005555000/0005555000/0000000000/0000000000/0000000000',
    )
}


def _rows(grid):
    return tuple(''.join(str(int(v)) for v in row) for row in grid)


def _signature(grid):
    return grid.shape, grid.astype(np.uint8, copy=False).tobytes()


def _choose_counts(rng, num_train, num_test):
    for name, value, low, high in (
        ('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError(f'{name} must be an integer from {low} through {high}')
    if num_train is None and num_test is None:
        num_train = int(rng.integers(2, 11))
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _different_color(rng, background, excluded):
    choices = [v for v in range(1, 10) if v != background and v != excluded]
    return int(rng.choice(choices))


def _sample_pair(rng, regime, anchor=False, background_mode=None):
    train = regime == 'source_train'
    assert train or regime == 'target_test'
    if background_mode == 'zero':
        background = 0
    elif background_mode == 'nonzero':
        background = int(rng.integers(1, 10))
    else:
        background = 0 if rng.random() < 0.67 else int(rng.integers(1, 10))
    body_color = _different_color(rng, background, -1)
    same_color = not anchor and rng.random() < (0.23 if train else 0.35)
    bar_color = body_color if same_color else _different_color(rng, background, body_color)

    hollow = not anchor and rng.random() < (0.22 if train else 0.36)
    body_height = int(rng.integers(3 if hollow else 2, 8 if train else 10))
    body_width = int(rng.integers(3 if hollow else 2, 9 if train else 12))
    gap = int(rng.integers(1, 5 if train else 7))
    top_pad = int(rng.integers(0, 4))
    left_pad = int(rng.integers(0, 7))
    right_pad = int(rng.integers(0, 7))
    if left_pad + right_pad == 0:
        right_pad = 1
    width = left_pad + body_width + right_pad
    body_top = top_pad + gap + 1
    body_bottom = body_top + body_height - 1

    if body_width > 2 and rng.random() < 0.47:
        bar_width = int(rng.integers(2, body_width))
    else:
        bar_width = body_width
    bar_offset = int(rng.integers(0, body_width - bar_width + 1))
    bar_left = left_pad + bar_offset
    if top_pad == 0 and (bar_left == 0 or bar_left + bar_width == width):
        top_pad = 1
        body_top += 1
        body_bottom += 1

    lower_count = int(rng.choice([0, 1, 2, 3], p=[0.28, 0.36, 0.28, 0.08]))
    lower = []
    previous_row = body_bottom
    previous_color = body_color
    previous_span = (left_pad, left_pad + body_width)
    for _ in range(lower_count):
        row = previous_row + int(rng.integers(1, 4))
        length = int(rng.integers(2, min(6, width + 1)))
        left = int(rng.integers(0, width - length + 1))
        color_options = [body_color, bar_color,
                         _different_color(rng, background, body_color)]
        color = int(rng.choice(color_options))
        if row == body_bottom + 1 and color == body_color:
            color = _different_color(rng, background, body_color)
        if row == previous_row + 1 and color == previous_color:
            a, b = previous_span
            if not (left + length <= a or left >= b):
                color = _different_color(rng, background, color)
        lower.append((row, left, length, color))
        previous_row, previous_color = row, color
        previous_span = (left, left + length)

    last_row = lower[-1][0] if lower else body_bottom
    height = last_row + 1 + int(rng.integers(1, 4))
    if height > 30:
        height = 30
    assert last_row < height - 1
    inp = np.full((height, width), background, dtype=np.int8)
    if hollow:
        inp[body_top, left_pad:left_pad + body_width] = body_color
        inp[body_bottom, left_pad:left_pad + body_width] = body_color
        inp[body_top:body_bottom + 1, left_pad] = body_color
        inp[body_top:body_bottom + 1, left_pad + body_width - 1] = body_color
    else:
        inp[body_top:body_bottom + 1, left_pad:left_pad + body_width] = body_color
    inp[top_pad, bar_left:bar_left + bar_width] = bar_color
    for row, left, length, color in lower:
        inp[row, left:left + length] = color

    # Keep the background visually dominant and identifiable even for large solids.
    foreground = int(np.count_nonzero(inp != background))
    while inp.size - foreground <= foreground:
        if inp.shape[1] < 30 and (inp.shape[1] <= inp.shape[0] or inp.shape[0] == 30):
            inp = np.pad(inp, ((0, 0), (0, 1)), constant_values=background)
        else:
            assert inp.shape[0] < 30
            inp = np.pad(inp, ((0, 1), (0, 0)), constant_values=background)
    out = inp.copy()
    out[top_pad, bar_left:bar_left + bar_width] = background
    out[body_top - 1, bar_left:bar_left + bar_width] = bar_color
    return inp, out


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _choose_counts(rng, num_train, num_test)
    puzzle = {'train': [], 'test': []}
    seen = set()
    anchor_index = int(rng.integers(0, train_count))
    zero_index, nonzero_index = rng.choice(train_count, size=2, replace=False)
    for part, count, regime in (
        ('train', train_count, 'source_train'),
        ('test', test_count, 'target_test'),
    ):
        for index in range(count):
            for attempt in range(40):
                background_mode = None
                if part == 'train':
                    if index == zero_index:
                        background_mode = 'zero'
                    elif index == nonzero_index:
                        background_mode = 'nonzero'
                inp, out = _sample_pair(
                    rng, regime, anchor=(part == 'train' and index == anchor_index),
                    background_mode=background_mode)
                signature = _signature(inp)
                if signature not in seen and _rows(inp) not in _SOURCE_INPUTS:
                    seen.add(signature)
                    puzzle[part].append({'input': inp, 'output': out})
                    break
            else:
                raise RuntimeError('could not construct a distinct pair')
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((grid >= 0) & (grid <= 9))


def _analyze_pair(inp, out):
    _check_grid(inp)
    _check_grid(out)
    assert inp.shape == out.shape and inp.dtype == out.dtype
    assert not np.array_equal(inp, out)
    values, counts = np.unique(inp, return_counts=True)
    background = int(values[np.argmax(counts)])
    assert counts.max() > inp.size / 2
    assert all(int(inp[r, c]) == background for r, c in (
        (0, 0), (0, -1), (-1, 0), (-1, -1)))
    active = np.flatnonzero(np.any(inp != background, axis=1))
    assert len(active) >= 3
    bar_row = int(active[0])
    bar_cols = np.flatnonzero(inp[bar_row] != background)
    bar_left, bar_right = int(bar_cols[0]), int(bar_cols[-1])
    assert len(bar_cols) >= 2 and len(bar_cols) == bar_right - bar_left + 1
    bar_color = int(inp[bar_row, bar_left])
    assert np.all(inp[bar_row, bar_left:bar_right + 1] == bar_color)
    top = int(active[1])
    gap = top - bar_row - 1
    assert gap >= 1
    body_cols = np.flatnonzero(inp[top] != background)
    left, right = int(body_cols[0]), int(body_cols[-1])
    assert len(body_cols) >= 2 and len(body_cols) == right - left + 1
    body_color = int(inp[top, left])
    assert np.all(inp[top, left:right + 1] == body_color)
    assert left <= bar_left <= bar_right <= right
    bottom = top
    while bottom + 1 < inp.shape[0] and np.any(inp[bottom + 1, left:right + 1] == body_color):
        bottom += 1
    assert bottom > top
    body = inp[top:bottom + 1, left:right + 1]
    assert np.all(inp[top:bottom + 1, :left] == background)
    assert np.all(inp[top:bottom + 1, right + 1:] == background)
    if np.all(body == body_color):
        kind = 'solid'
    else:
        assert body.shape[0] >= 3 and body.shape[1] >= 3
        expected_body = np.full(body.shape, background, dtype=inp.dtype)
        expected_body[0, :] = body_color
        expected_body[-1, :] = body_color
        expected_body[:, 0] = body_color
        expected_body[:, -1] = body_color
        assert np.array_equal(body, expected_body)
        kind = 'hollow'
    lower = []
    for row in range(bottom + 1, inp.shape[0]):
        columns = np.flatnonzero(inp[row] != background)
        if len(columns) == 0:
            continue
        start, end = int(columns[0]), int(columns[-1])
        assert 2 <= len(columns) <= 5 and len(columns) == end - start + 1
        color = int(inp[row, start])
        assert np.all(inp[row, start:end + 1] == color)
        if row == bottom + 1:
            assert color != body_color
        if lower and row == lower[-1][0] + 1 and color == lower[-1][3]:
            previous_start, previous_end = lower[-1][1:3]
            assert end < previous_start or start > previous_end
        lower.append((row, start, end, color))
    assert len(lower) <= 3
    assert np.all(inp[bar_row + 1:top] == background)
    expected = inp.copy()
    expected[bar_row, bar_left:bar_right + 1] = background
    assert np.all(expected[top - 1, bar_left:bar_right + 1] == background)
    expected[top - 1, bar_left:bar_right + 1] = bar_color
    assert np.array_equal(out, expected)
    return {
        'background': background, 'bar_color': bar_color,
        'body_color': body_color, 'body_height': bottom - top + 1,
        'body_width': right - left + 1, 'kind': kind, 'gap': gap,
        'bar_row': bar_row, 'bar_left': bar_left,
        'bar_width': bar_right - bar_left + 1, 'body_left': left,
        'body_top': top, 'body_bottom': bottom, 'lower': tuple(lower),
        'lower_count': len(lower),
        'adjacent_lower': any(row == bottom + 1 or
                              (i and row == lower[i - 1][0] + 1)
                              for i, (row, _, _, _) in enumerate(lower)),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen_inputs, seen_pairs = set(), set()
    metas = {'train': [], 'test': []}
    for part in ('train', 'test'):
        for pair in puzzle[part]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = pair['input'], pair['output']
            meta = _analyze_pair(inp, out)
            metas[part].append(meta)
            input_sig = _signature(inp)
            pair_sig = input_sig, _signature(out)
            assert input_sig not in seen_inputs and pair_sig not in seen_pairs
            assert _rows(inp) not in _SOURCE_INPUTS
            seen_inputs.add(input_sig)
            seen_pairs.add(pair_sig)
    # The sole puzzle convention is that the detached upper bar closes its gap.
    # Both demonstrations show that exact move; colors and geometry are input-local.
    assert any(meta['kind'] == 'solid' and
               meta['bar_color'] != meta['body_color']
               for meta in metas['train'])
    assert 0 in {meta['background'] for meta in metas['train']}
    assert any(meta['background'] != 0 for meta in metas['train'])
    for part in ('train', 'test'):
        for meta in metas[part]:
            assert meta['bar_width'] <= meta['body_width']
            assert meta['bar_row'] <= 3
            assert 2 <= meta['body_height'] <= 9
            assert 2 <= meta['body_width'] <= 11
            assert 1 <= meta['gap'] <= 6


def _same_puzzle(first, second):
    return all(
        len(first[part]) == len(second[part]) and all(
            np.array_equal(a[k], b[k])
            for a, b in zip(first[part], second[part])
            for k in ('input', 'output')
        ) for part in ('train', 'test')
    )


def self_test() -> None:
    source_input = np.array([
        [int(value) for value in row] for row in (
            '00000000', '03330000', '00000000', '04440000',
            '04440000', '04440000', '00000333', '00333000',
        )], dtype=np.int8)
    source_output = source_input.copy()
    source_output[1, 1:4] = 0
    source_output[2, 1:4] = 3
    source_meta = _analyze_pair(source_input, source_output)
    assert source_meta['adjacent_lower'] and source_meta['lower_count'] == 2
    assert source_meta['lower'][0][3] == source_meta['lower'][1][3]
    counts_train, counts_test, shapes, layouts = set(), set(), set(), set()
    backgrounds, palettes, geometry = set(), set(), set()
    train_hollow = train_same = train_zero = 0
    test_hollow = test_same = test_easy = test_combined = 0
    adjacent = edge_body = left_narrow = right_narrow = 0
    middle_narrow = same_adjacent = isolated_hollow = isolated_same = 0
    one_color = nonzero_background = lower_color_body = 0
    minimum_height = minimum_width = 30
    maximum_height = maximum_width = 0
    minimum_aspect = 99.0
    maximum_aspect = 0.0
    puzzle_background_variety = 0
    puzzle_color_variety = puzzle_gap_variety = puzzle_lower_variety = 0
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        counts_train.add(len(puzzle['train']))
        counts_test.add(len(puzzle['test']))
        local_backgrounds = set()
        local_colors, local_gaps, local_lower_counts = set(), set(), set()
        for part in ('train', 'test'):
            for pair in puzzle[part]:
                inp, out = pair['input'], pair['output']
                meta = _analyze_pair(inp, out)
                signatures.add(_signature(inp))
                shapes.add(inp.shape)
                local_backgrounds.add(meta['background'])
                local_colors.add(meta['body_color'])
                local_gaps.add(meta['gap'])
                local_lower_counts.add(meta['lower_count'])
                backgrounds.add(meta['background'])
                nonzero_background += meta['background'] != 0
                one_color += len(np.unique(inp)) == 2
                minimum_height = min(minimum_height, inp.shape[0])
                maximum_height = max(maximum_height, inp.shape[0])
                minimum_width = min(minimum_width, inp.shape[1])
                maximum_width = max(maximum_width, inp.shape[1])
                minimum_aspect = min(minimum_aspect, inp.shape[1] / inp.shape[0])
                maximum_aspect = max(maximum_aspect, inp.shape[1] / inp.shape[0])
                palette = tuple(sorted(int(v) for v in np.unique(inp)))
                palettes.add(palette)
                layout = (part, inp.shape, meta['kind'], meta['body_height'],
                          meta['body_width'], meta['gap'], meta['bar_width'],
                          meta['bar_left'] - meta['body_left'], meta['lower_count'])
                layouts.add(layout)
                geometry.add((meta['kind'], meta['body_height'], meta['body_width'],
                              meta['gap'], meta['bar_width'], meta['lower_count']))
                adjacent += meta['adjacent_lower']
                edge_body += (meta['body_left'] == 0 or
                              meta['body_left'] + meta['body_width'] == inp.shape[1])
                left_narrow += (meta['bar_width'] < meta['body_width'] and
                                meta['bar_left'] == meta['body_left'])
                right_narrow += (meta['bar_width'] < meta['body_width'] and
                                 meta['bar_left'] + meta['bar_width'] ==
                                 meta['body_left'] + meta['body_width'])
                middle_narrow += (meta['bar_width'] < meta['body_width'] and
                                  meta['bar_left'] > meta['body_left'] and
                                  meta['bar_left'] + meta['bar_width'] <
                                  meta['body_left'] + meta['body_width'])
                lower = meta['lower']
                same_adjacent += any(
                    lower[j][0] == lower[j - 1][0] + 1 and
                    lower[j][3] == lower[j - 1][3]
                    for j in range(1, len(lower)))
                lower_color_body += any(row[3] == meta['body_color'] for row in lower)
                if part == 'train':
                    train_hollow += meta['kind'] == 'hollow'
                    train_same += meta['bar_color'] == meta['body_color']
                    train_zero += meta['lower_count'] == 0
                else:
                    test_hollow += meta['kind'] == 'hollow'
                    test_same += meta['bar_color'] == meta['body_color']
                    test_easy += (meta['kind'] == 'solid' and
                                  meta['bar_color'] != meta['body_color'] and
                                  meta['gap'] <= 3)
                    test_combined += (meta['kind'] == 'hollow' and
                                      meta['bar_color'] == meta['body_color'] and
                                      meta['gap'] >= 3)
                    isolated_hollow += (meta['kind'] == 'hollow' and
                                        meta['bar_color'] != meta['body_color'] and
                                        meta['gap'] <= 2)
                    isolated_same += (meta['kind'] == 'solid' and
                                      meta['bar_color'] == meta['body_color'] and
                                      meta['gap'] <= 2)
        puzzle_background_variety += len(local_backgrounds) > 1
        puzzle_color_variety += len(local_colors) > 1
        puzzle_gap_variety += len(local_gaps) > 1
        puzzle_lower_variety += len(local_lower_counts) > 1
    assert len(counts_train) >= 4 and len(counts_test) >= 3
    assert len(shapes) >= 30 and len(layouts) >= 150
    assert len(signatures) >= 150 and len(geometry) >= 80
    assert len(backgrounds) >= 5 and len(palettes) >= 30
    assert minimum_height <= 8 and maximum_height >= 20
    assert minimum_width <= 5 and maximum_width >= 20
    assert minimum_aspect < 0.4 and maximum_aspect > 2.0
    assert min(puzzle_background_variety, puzzle_color_variety,
               puzzle_gap_variety, puzzle_lower_variety) >= 35
    assert min(same_adjacent, isolated_hollow, isolated_same,
               middle_narrow, one_color, nonzero_background,
               lower_color_body) > 0
    assert min(train_hollow, train_same, train_zero, test_hollow,
               test_same, test_easy, test_combined, adjacent, edge_body,
               left_narrow, right_narrow) > 0
    for train, test in ((2, 1), (10, 1), (10, 4),
                        (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(812, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle['train']) == train
        if test is not None:
            assert len(puzzle['test']) == test
        assert _same_puzzle(puzzle, generate_puzzle(812, train, test))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    broken = generate_puzzle(991, 2, 1)
    pair = broken['test'][0]
    change = np.argwhere(pair['input'] != pair['output'])[0]
    pair['output'] = pair['output'].copy()
    pair['output'][tuple(change)] = pair['input'][tuple(change)]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted a damaged move')
