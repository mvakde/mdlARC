import numpy as np


_SOURCE_TEXT_PAIRS = (
    (
        (
            "0007", "0777", "0700", "0777", "4444",
            "0020", "0222", "2200", "0202",
        ),
        ("3300", "3000", "0033", "3000"),
    ),
    (
        (
            "0077", "0077", "0770", "7700", "4444",
            "2020", "0202", "0220", "0020",
        ),
        ("0300", "3000", "3003", "0003"),
    ),
    (
        (
            "7070", "0077", "7077", "7700", "4444",
            "0022", "0000", "2002", "0202",
        ),
        ("0300", "3300", "0300", "0030"),
    ),
    (
        (
            "7707", "0770", "0777", "0770", "4444",
            "0000", "0202", "2220", "2002",
        ),
        ("0030", "3000", "0000", "0000"),
    ),
    (
        (
            "7777", "0777", "7000", "7070", "4444",
            "0222", "0000", "2022", "0200",
        ),
        ("0000", "3000", "0300", "0003"),
    ),
)


def _array_from_rows(rows):
    return np.asarray(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_PAIRS = tuple(
    (_array_from_rows(input_rows), _array_from_rows(output_rows))
    for input_rows, output_rows in _SOURCE_TEXT_PAIRS
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _validated_optional_count(value, name, low, high):
    if value is None:
        return None

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")

    return value


def _resolve_counts(rng, num_train, num_test):
    train_count = _validated_optional_count(
        num_train, "num_train", 2, 10
    )
    test_count = _validated_optional_count(
        num_test, "num_test", 1, 4
    )

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    if train_count is None and test_count is None:
        # The source has four demonstrations; keep short puzzles common.
        if rng.random() < 0.55:
            train_count = int(rng.integers(2, 5))
        else:
            train_count = int(rng.integers(5, 11))
        test_count = int(rng.integers(1, min(4, train_count - 1) + 1))
    elif train_count is None:
        train_count = int(rng.integers(test_count + 1, 11))
    else:
        largest_test = min(4, train_count - 1)
        test_count = int(rng.integers(1, largest_test + 1))

    if not (
        2 <= train_count <= 10
        and 1 <= test_count <= 4
        and train_count > test_count
    ):
        raise ValueError(
            "the requested pair counts cannot form a legal puzzle"
        )

    return train_count, test_count


def _sample_puzzle_parameters(rng):
    colors = tuple(int(c) for c in rng.choice(10, size=5, replace=False))
    orientation_mode = rng.choice(('horizontal', 'vertical', 'mixed'), p=(0.36, 0.27, 0.37))
    parameters = {
        'colors': colors,  # background, first foreground, divider, second foreground, highlight
        'recolor': bool(rng.random() < 0.48),
        'variable_size': bool(rng.random() < 0.68),
        'orientation_mode': str(orientation_mode),
        'base_shape': (4, 4) if rng.random() < 0.22 else _sample_shape(rng, str(orientation_mode)),
    }
    if rng.random() < 0.14:
        parameters['variable_size'] = False
        parameters['orientation_mode'] = 'horizontal'
        parameters['base_shape'] = (4, 4)
    return parameters


def _sample_shape(rng, orientation):
    # Both orientations have room for a 14-cell panel dimension and a separator.
    if rng.random() < 0.16:
        return 4, 4
    height = int(rng.integers(3, 12))
    width = int(rng.integers(3, 12))
    if rng.random() < 0.20:
        if orientation == 'horizontal':
            height = int(rng.integers(11, 15))
        elif orientation == 'vertical':
            width = int(rng.integers(11, 15))
        elif rng.random() < 0.5:
            height = int(rng.integers(11, 15))
        else:
            width = int(rng.integers(11, 15))
    return height, width


def _roles_for_pair(rng, parameters, pair_index):
    if not parameters['recolor']:
        return parameters['colors']
    highlight = parameters['colors'][4]
    available = [c for c in range(10) if c != highlight]
    colors = [int(c) for c in rng.choice(available, size=4, replace=False)]
    # Ensure an actual background change is shown in the first two demonstrations.
    if pair_index == 1 and colors[0] == parameters['colors'][0]:
        colors[0], colors[1] = colors[1], colors[0]
    return (*colors, highlight)


def _random_mask(rng, height, width):
    p = float(rng.uniform(0.18, 0.82))
    row_bias = rng.uniform(-0.23, 0.23, size=(height, 1))
    col_bias = rng.uniform(-0.19, 0.19, size=(1, width))
    return rng.random((height, width)) < np.clip(p + row_bias + col_bias, 0.04, 0.96)


def _rect_mask(rng, height, width, edge=False):
    mask = np.zeros((height, width), dtype=bool)
    for _ in range(int(rng.integers(1, 4))):
        rh = int(rng.integers(1, height + 1))
        rw = int(rng.integers(1, width + 1))
        if edge:
            row = int(rng.choice((0, height - rh)))
            col = int(rng.choice((0, width - rw)))
        else:
            row = int(rng.integers(0, height - rh + 1))
            col = int(rng.integers(0, width - rw + 1))
        mask[row:row + rh, col:col + rw] = True
    return mask


def _masks(rng, height, width, regime, fixed_palette, teacher=False):
    modes = ('speckle', 'near_match', 'near_complement', 'match', 'complement',
             'patches', 'bands', 'pockets', 'blank_panel', 'full_panel',
             'all_highlight', 'all_background', 'row_extremes', 'mixed_rows')
    weights = np.array((18, 9, 9, 3, 4, 11, 9, 8, 7,
                        7 if fixed_palette else 0, 7, 5, 9, 4), dtype=float)
    if regime == 'target_test':
        weights[[3, 4, 8, 9, 10, 11, 12, 13]] *= 1.45
    weights /= weights.sum()
    mode = str(rng.choice(modes, p=weights))
    top = _random_mask(rng, height, width)
    bottom = _random_mask(rng, height, width)
    if mode in ('near_match', 'near_complement', 'match', 'complement'):
        bottom = top.copy() if mode in ('near_match', 'match') else ~top
        if mode.startswith('near'):
            flips = int(rng.integers(1, max(2, height * width // 4 + 1)))
            positions = rng.choice(height * width, size=flips, replace=False)
            bottom.flat[positions] = ~bottom.flat[positions]
    elif mode == 'patches':
        top = _rect_mask(rng, height, width)
        bottom = _rect_mask(rng, height, width)
        if rng.random() < 0.5:
            top = ~top
        if rng.random() < 0.5:
            bottom = ~bottom
    elif mode == 'bands':
        axis = int(rng.integers(2))
        for panel in (top, bottom):
            bands = rng.random(height if axis == 0 else width) < rng.uniform(0.22, 0.78)
            panel[:] = bands[:, None] if axis == 0 else bands[None, :]
    elif mode == 'pockets':
        top = _rect_mask(rng, height, width, edge=True)
        bottom = _rect_mask(rng, height, width, edge=True)
        if rng.random() < 0.5:
            top = ~top
        if rng.random() < 0.5:
            bottom = ~bottom
    elif mode == 'blank_panel':
        (top if rng.random() < 0.5 else bottom)[:] = False
    elif mode == 'full_panel':
        (top if rng.random() < 0.5 else bottom)[:] = True
    elif mode == 'all_highlight':
        top[:] = False
        bottom[:] = False
    elif mode == 'all_background':
        bottom = ~top
    elif mode == 'row_extremes':
        for panel in (top, bottom):
            count = int(rng.integers(1, min(4, height + 1)))
            rows = rng.choice(height, size=count, replace=False)
            for row in rows:
                panel[row, :] = bool(rng.integers(2))
    elif mode == 'mixed_rows':
        for panel in (top, bottom):
            for row in range(height):
                if np.all(panel[row]) or not np.any(panel[row]):
                    panel[row, int(rng.integers(width))] = not bool(panel[row, 0])

    # Independent small chance of an extreme row in either split and either panel.
    if mode not in ('match', 'complement', 'all_highlight', 'all_background', 'mixed_rows') and rng.random() < 0.22:
        panel = top if rng.random() < 0.5 else bottom
        for row in rng.choice(height, size=int(rng.integers(1, min(4, height + 1))), replace=False):
            panel[row, :] = bool(rng.integers(2))

    if teacher:
        # Four distinct positions prove all four cellwise cases, including the highlight.
        positions = rng.choice(height * width, size=4, replace=False)
        for position, state in zip(positions, (0, 1, 2, 3)):
            top.flat[position] = bool(state & 2)
            bottom.flat[position] = bool(state & 1)
    elif not fixed_palette:
        # A recolored input has to identify its background from that input.
        if np.all(top):
            top.flat[int(rng.integers(height * width))] = False
        if np.all(bottom):
            bottom.flat[int(rng.integers(height * width))] = False
    # A pair with two monochrome but differently colored panels is ambiguous.
    if np.all(top) and np.all(bottom):
        bottom.flat[int(rng.integers(height * width))] = False
    return top, bottom


def _construct_pair(top, bottom, roles, orientation):
    background, first_fg, divider, second_fg, highlight = roles
    height, width = top.shape
    first = np.where(top, first_fg, background)
    second = np.where(bottom, second_fg, background)
    if orientation == 'horizontal':
        input_grid = np.vstack((first, np.full((1, width), divider), second))
    else:
        input_grid = np.hstack((first, np.full((height, 1), divider), second))
    output_grid = np.where(~top & ~bottom, highlight, background)
    return input_grid.astype(np.int8), output_grid.astype(np.int8)


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _pair_key(pair):
    return (_grid_key(pair['input']), _grid_key(pair['output']))


def _is_source_pair(input_grid, output_grid):
    return any(np.array_equal(input_grid, a) and np.array_equal(output_grid, b)
               for a, b in _SOURCE_PAIRS)


def _check_basic_grid(grid, label):
    _require(isinstance(grid, np.ndarray), f'{label} must be an array')
    _require(grid.ndim == 2, f'{label} must be two-dimensional')
    _require(all(1 <= size <= 30 for size in grid.shape), f'{label} size is illegal')
    _require(np.issubdtype(grid.dtype, np.integer), f'{label} needs integer dtype')
    _require(bool(np.all((grid >= 0) & (grid <= 9))), f'{label} has illegal colors')


def _analyze_pair(pair, learned=None):
    inp, out = pair['input'], pair['output']
    rows, cols = inp.shape
    if rows % 2 == 1 and out.shape == ((rows - 1) // 2, cols):
        orientation = 'horizontal'
        height = (rows - 1) // 2
        first, divider_cells, second = inp[:height], inp[height], inp[height + 1:]
    elif cols % 2 == 1 and out.shape == (rows, (cols - 1) // 2):
        orientation = 'vertical'
        width = (cols - 1) // 2
        first, divider_cells, second = inp[:, :width], inp[:, width], inp[:, width + 1:]
    else:
        raise AssertionError('input and output are not equal aligned panels with divider')
    _require(first.shape == second.shape, 'panels differ in shape')
    _require(len(np.unique(divider_cells)) == 1, 'divider is not solid')
    divider = int(divider_cells.flat[0])
    first_colors = set(map(int, np.unique(first)))
    second_colors = set(map(int, np.unique(second)))
    _require(1 <= len(first_colors) <= 2 and 1 <= len(second_colors) <= 2,
             'each panel may have only background and one foreground')
    _require(divider not in first_colors | second_colors, 'divider color occurs in a panel')
    common = first_colors & second_colors
    if len(common) == 1:
        background = next(iter(common))
    elif len(common) == 0 and learned is not None and learned['fixed_background'] is not None:
        _require(len(first_colors) == 1 or len(second_colors) == 1,
                 'a disjoint palette needs a monochrome panel')
        background = learned['fixed_background']
        _require(background in first_colors | second_colors, 'learned background is absent')
    else:
        raise AssertionError('panel background cannot be identified')
    _require(len(common) <= 1, 'panel foreground colors overlap')
    first_fg = first_colors - {background}
    second_fg = second_colors - {background}
    _require(len(first_fg) <= 1 and len(second_fg) <= 1, 'multiple panel foregrounds')
    _require(not first_fg or not second_fg or first_fg != second_fg,
             'panel foreground roles must be distinct')
    _require(background != divider, 'divider equals background')
    first_mask = first != background
    second_mask = second != background
    out_colors = set(map(int, np.unique(out)))
    if learned is None:
        _require(0 in _state_set(first_mask, second_mask), 'first training pair lacks highlight case')
        highlights = out_colors - {background}
        _require(len(highlights) == 1, 'first training pair must reveal highlight')
        highlight = next(iter(highlights))
    else:
        highlight = learned['highlight']
    _require(highlight not in set(map(int, np.unique(inp))), 'highlight occurs in input')
    _require(out_colors <= {background, highlight}, 'output has unexpected colors')
    expected = np.where(~first_mask & ~second_mask, highlight, background)
    _require(np.array_equal(out, expected), 'output violates shared-background rule')
    return {'orientation': orientation, 'shape': first.shape, 'background': background,
            'highlight': highlight, 'divider': divider, 'first_fg': next(iter(first_fg), None),
            'second_fg': next(iter(second_fg), None), 'states': _state_set(first_mask, second_mask),
            'first_mask': first_mask, 'second_mask': second_mask}


def _state_set(first, second):
    return set(map(int, np.unique(2 * first.astype(np.int8) + second.astype(np.int8))))


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'bad puzzle schema')
    _require(isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list), 'splits must be lists')
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, 'illegal pair counts')
    seen_inputs, seen_pairs = set(), set()
    learned = None
    infos = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for index, pair in enumerate(puzzle[split]):
            label = f'{split}[{index}]'
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, f'{label} schema')
            _check_basic_grid(pair['input'], f'{label}.input')
            _check_basic_grid(pair['output'], f'{label}.output')
            key = _grid_key(pair['input'])
            pair_key = _pair_key(pair)
            _require(key not in seen_inputs and pair_key not in seen_pairs, 'duplicate pair')
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            _require(not _is_source_pair(pair['input'], pair['output']), 'copied source pair')
            info = _analyze_pair(pair, learned)
            infos[split].append(info)
            if learned is None:
                learned = {'highlight': info['highlight'], 'fixed_background': info['background']}
            else:
                _require(info['highlight'] == learned['highlight'], 'highlight changed within puzzle')
    # The first demonstration covers the full truth table; all later outputs have its color key.
    _require(infos['train'][0]['states'] == {0, 1, 2, 3}, 'training does not establish the full rule')
    # If a pair has no common panel color, its palette must have been demonstrated in training.
    complete_train_roles = {
        (i['background'], i['first_fg'], i['divider'], i['second_fg'])
        for i in infos['train'] if i['first_fg'] is not None and i['second_fg'] is not None
    }
    for split in ('train', 'test'):
        for pair, info in zip(puzzle[split], infos[split]):
            first, second = info['first_mask'], info['second_mask']
            if np.all(first) or np.all(second):
                # A completely filled panel is only used with a trained palette.
                _require(any(info['background'] == role[0] and info['divider'] == role[2]
                             and (info['first_fg'] is None or info['first_fg'] == role[1])
                             and (info['second_fg'] is None or info['second_fg'] == role[3])
                             for role in complete_train_roles), 'full panel needs known palette')
            _require(pair['input'].shape != pair['output'].shape, 'unchanged input/output geometry')


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    params = _sample_puzzle_parameters(rng)
    puzzle = {'train': [], 'test': []}
    used = set()
    orientations = []
    for index in range(train_count + test_count):
        if params['orientation_mode'] == 'mixed':
            orientation = ('horizontal', 'vertical')[index] if index < 2 else str(rng.choice(('horizontal', 'vertical')))
        else:
            orientation = params['orientation_mode']
        orientations.append(orientation)
    for index, orientation in enumerate(orientations):
        split = 'train' if index < train_count else 'test'
        regime = 'source_train' if split == 'train' else 'target_test'
        shape = _sample_shape(rng, orientation) if params['variable_size'] else params['base_shape']
        roles = _roles_for_pair(rng, params, index)
        # First training pair keeps the palette sampled at puzzle level.
        if index == 0:
            roles = params['colors']
        for attempt in range(80):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            first, second = _masks(pair_rng, *shape, regime, not params['recolor'], teacher=index == 0)
            inp, out = _construct_pair(first, second, roles, orientation)
            if _grid_key(inp) in used or _is_source_pair(inp, out):
                continue
            used.add(_grid_key(inp))
            puzzle[split].append({'input': inp, 'output': out})
            break
        else:
            raise RuntimeError('could not construct unique pair')
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x['input'], y['input']) and np.array_equal(x['output'], y['output'])
        for x, y in zip(a[s], b[s])) for s in ('train', 'test'))


def _split_input_only(grid):
    rows, cols = grid.shape
    if rows % 2:
        middle = rows // 2
        divider = int(grid[middle, 0])
        if np.all(grid[middle] == divider) and not np.any(grid[:middle] == divider) \
                and not np.any(grid[middle + 1:] == divider):
            return grid[:middle], grid[middle + 1:]
    if cols % 2:
        middle = cols // 2
        divider = int(grid[0, middle])
        if np.all(grid[:, middle] == divider) and not np.any(grid[:, :middle] == divider) \
                and not np.any(grid[:, middle + 1:] == divider):
            return grid[:, :middle], grid[:, middle + 1:]
    raise AssertionError('input alone does not reveal its divider')


def _predict_from_training(train_pairs, test_input):
    # Only demonstrations and the test input are read here.
    sample_first, sample_second = _split_input_only(train_pairs[0]['input'])
    shared = set(map(int, np.unique(sample_first))) & set(map(int, np.unique(sample_second)))
    _require(len(shared) == 1, 'training background is ambiguous')
    trained_background = next(iter(shared))
    highlight_colors = set(map(int, np.unique(train_pairs[0]['output']))) - {trained_background}
    _require(len(highlight_colors) == 1, 'training highlight is ambiguous')
    highlight = next(iter(highlight_colors))
    first, second = _split_input_only(test_input)
    shared = set(map(int, np.unique(first))) & set(map(int, np.unique(second)))
    background = next(iter(shared)) if len(shared) == 1 else trained_background
    return np.where((first == background) & (second == background), highlight, background)


def self_test() -> None:
    auto = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in auto:
        validate_generated_puzzle(puzzle)
        demonstrations = [{'input': pair['input'].copy(), 'output': pair['output'].copy()}
                          for pair in puzzle['train']]
        for pair in puzzle['test']:
            prediction = _predict_from_training(demonstrations, pair['input'].copy())
            _require(np.array_equal(prediction, pair['output']), 'test cannot be predicted from training')
    _require(_puzzles_equal(generate_puzzle(123456, 7, 3), generate_puzzle(123456, 7, 3)),
             'explicit generation is nondeterministic')
    _require(_puzzles_equal(generate_puzzle(654321), generate_puzzle(654321)),
             'automatic generation is nondeterministic')
    for n, m in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(8000 + n + m, n, m)
        _require((len(p['train']), len(p['test'])) == (n, m), 'explicit counts changed')
    for n, m in ((None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(9000 + (n or 0) + (m or 0), n, m)
        _require(n is None or len(p['train']) == n, 'partial train count changed')
        _require(m is None or len(p['test']) == m, 'partial test count changed')
    for request in ({'num_train': 1}, {'num_train': 11}, {'num_test': 0},
                    {'num_test': 5}, {'num_train': 2, 'num_test': 2},
                    {'num_train': 2, 'num_test': 3}, {'num_train': 4, 'num_test': 4},
                    {'num_train': 3.0, 'num_test': 1}):
        try:
            generate_puzzle(99, **request)
        except ValueError:
            pass
        else:
            raise AssertionError(f'illegal count accepted: {request}')
    _require(len({len(p['train']) for p in auto}) >= 4, 'train counts lack diversity')
    _require(len({len(p['test']) for p in auto}) >= 3, 'test counts lack diversity')
    _require(len({_grid_key(p['train'][0]['input']) for p in auto}) >= 40,
             'seeds lack structural diversity')
    infos = [(s, _analyze_pair(pair, {'highlight': _analyze_pair(p['train'][0])['highlight'],
                                     'fixed_background': _analyze_pair(p['train'][0])['background']}))
             for p in auto for s in ('train', 'test') for pair in p[s]]
    _require(len({i['shape'] for _, i in infos}) >= 20, 'size diversity too narrow')
    _require({i['orientation'] for _, i in infos} == {'horizontal', 'vertical'}, 'orientation diversity missing')
    _require(any(i['states'] == {0} for _, i in infos), 'all-highlight output missing')
    _require(any(0 not in i['states'] for _, i in infos), 'all-background output missing')
    _require(any(s == 'train' and (np.any(np.all(i['first_mask'], axis=1)) or
                    np.any(np.all(~i['second_mask'], axis=1))) for s, i in infos),
             'training extreme rows missing')
    _require(any(s == 'test' and not (np.any(np.all(i['first_mask'], axis=1)) or
                    np.any(np.all(~i['second_mask'], axis=1))) for s, i in infos),
             'ordinary test rows missing')
    _require(sum(len(p['train']) <= 4 and len(p['test']) == 1 for p in auto) >= 8,
             'compact source-like counts are too rare')
    _require(sum(len({q['output'].shape for s in ('train', 'test') for q in p[s]}) > 1
                 for p in auto) >= 15, 'within-puzzle size variation is too rare')
    _require(sum(len({tuple(_analyze_pair(q, {'highlight': _analyze_pair(p['train'][0])['highlight'],
                                           'fixed_background': _analyze_pair(p['train'][0])['background']})[k]
                       for k in ('background', 'divider')) for s in ('train', 'test') for q in p[s]}) > 1
                 for p in auto) >= 12, 'within-puzzle recoloring is too rare')
    _require(sum(len({_analyze_pair(q, {'highlight': _analyze_pair(p['train'][0])['highlight'],
                                       'fixed_background': _analyze_pair(p['train'][0])['background']})['orientation']
                       for s in ('train', 'test') for q in p[s]}) > 1 for p in auto) >= 8,
             'mixed-orientation puzzles are too rare')
    _require(any(s == 'train' and i['states'] == {0} for s, i in infos) and
             any(s == 'test' and i['states'] == {0} for s, i in infos),
             'uniform highlight must occur in both splits')
    _require(any(s == 'train' and 0 not in i['states'] for s, i in infos) and
             any(s == 'test' and 0 not in i['states'] for s, i in infos),
             'uniform background must occur in both splits')
    _require(any(np.array_equal(i['first_mask'], i['second_mask']) for _, i in infos)
             and any(np.array_equal(i['first_mask'], ~i['second_mask']) for _, i in infos),
             'exact mask relationships missing')
    _require(any(not np.any(i['first_mask']) or not np.any(i['second_mask']) for _, i in infos)
             and any(np.all(i['first_mask']) or np.all(i['second_mask']) for _, i in infos),
             'blank or filled panels missing')
