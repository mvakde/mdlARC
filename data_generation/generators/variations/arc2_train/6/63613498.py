import numpy as np


_SOURCE_PAIR_SIGNATURES = {
    (
        "0305000000\n"
        "3305000000\n"
        "0005004400\n"
        "5555004000\n"
        "0000000000\n"
        "0000000000\n"
        "0077700000\n"
        "0000000090\n"
        "8800000990\n"
        "8800000000",
        "0305000000\n"
        "3305000000\n"
        "0005004400\n"
        "5555004000\n"
        "0000000000\n"
        "0000000000\n"
        "0077700000\n"
        "0000000050\n"
        "8800000550\n"
        "8800000000",
    ),
    (
        "1105000700\n"
        "0115007770\n"
        "0015000700\n"
        "5555000000\n"
        "0000000000\n"
        "0000066000\n"
        "7770006600\n"
        "0070000600\n"
        "0000800000\n"
        "0008888000",
        "1105000700\n"
        "0115007770\n"
        "0015000700\n"
        "5555000000\n"
        "0000000000\n"
        "0000055000\n"
        "7770005500\n"
        "0070000500\n"
        "0000800000\n"
        "0008888000",
    ),
    (
        "0405000100\n"
        "4445001110\n"
        "0005000000\n"
        "5555000000\n"
        "0000003000\n"
        "0000003300\n"
        "0000003000\n"
        "0666000000\n"
        "0060000000\n"
        "0000000000",
        "0405000500\n"
        "4445005550\n"
        "0005000000\n"
        "5555000000\n"
        "0000003000\n"
        "0000003300\n"
        "0000003000\n"
        "0666000000\n"
        "0060000000\n"
        "0000000000",
    ),
    (
        "0005000600\n"
        "2225000660\n"
        "0025000600\n"
        "5555030000\n"
        "0000030000\n"
        "0000033000\n"
        "8880000040\n"
        "8000000040\n"
        "0000777040\n"
        "0000007000",
        "0005000600\n"
        "2225000660\n"
        "0025000600\n"
        "5555030000\n"
        "0000030000\n"
        "0000033000\n"
        "8880000040\n"
        "8000000040\n"
        "0000555040\n"
        "0000005000",
    ),
}


def _grid_text(grid: np.ndarray) -> str:
    return "\n".join(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _array_signature(grid: np.ndarray) -> tuple:
    return (
        grid.shape,
        tuple(int(value) for value in grid.ravel()),
    )


def _normalize(coords) -> tuple:
    cells = [(int(point[0]), int(point[1])) for point in coords]
    if not cells:
        return ()

    min_row = min(row for row, _ in cells)
    min_col = min(col for _, col in cells)
    return tuple(
        sorted(
            (row - min_row, col - min_col)
            for row, col in cells
        )
    )


def _shape_size(shape: tuple) -> tuple[int, int]:
    return (
        max(row for row, _ in shape) + 1,
        max(col for _, col in shape) + 1,
    )


def _reflection_variants(shape: tuple) -> set[tuple]:
    height, width = _shape_size(shape)
    variants = {
        _normalize(
            (row, width - 1 - col)
            for row, col in shape
        ),
        _normalize(
            (height - 1 - row, col)
            for row, col in shape
        ),
    }
    variants.discard(shape)
    return variants


def _components(mask: np.ndarray) -> list[tuple]:
    height, width = mask.shape
    unseen = {
        (int(row), int(col))
        for row, col in np.argwhere(mask)
    }
    result = []

    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack = [start]
        component = []

        while stack:
            row, col = stack.pop()
            component.append((row, col))

            for next_row, next_col in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if (
                    0 <= next_row < height
                    and 0 <= next_col < width
                    and (next_row, next_col) in unseen
                ):
                    unseen.remove((next_row, next_col))
                    stack.append((next_row, next_col))

        result.append(tuple(sorted(component)))

    return result


def _find_canonical_divider(grid: np.ndarray, control: int) -> tuple:
    matches = []
    for rotation in range(4):
        turned = np.rot90(grid, rotation)
        control_cells = {(int(row), int(col)) for row, col in np.argwhere(turned == control)}
        if not control_cells:
            continue
        drow = max(row for row, _ in control_cells)
        dcol = max(col for _, col in control_cells)
        if drow < 3 or dcol < 3 or drow + 3 >= turned.shape[0] or dcol + 3 >= turned.shape[1]:
            continue
        expected = {(row, dcol) for row in range(drow + 1)}
        expected.update((drow, col) for col in range(dcol + 1))
        if control_cells == expected:
            matches.append((turned, rotation, drow, dcol))
    assert len(matches) == 1
    return matches[0]


def _connected(shape: tuple) -> bool:
    return len(_components(_shape_array(shape))) == 1


def _shape_array(shape: tuple) -> np.ndarray:
    height, width = _shape_size(shape)
    mask = np.zeros((height, width), dtype=bool)
    for row, col in shape:
        mask[row, col] = True
    return mask


def _turns(shape: tuple) -> set[tuple]:
    variants = set()
    current = shape
    for _ in range(3):
        current = _normalize((col, -row) for row, col in current)
        if current != shape:
            variants.add(current)
    return variants


def _near_shapes(shape: tuple) -> set[tuple]:
    variants = set()
    for removed in shape:
        remaining = set(shape) - {removed}
        if not _connected(_normalize(remaining)):
            continue
        for row, col in remaining:
            for added in ((row - 1, col), (row + 1, col),
                          (row, col - 1), (row, col + 1)):
                if added in remaining:
                    continue
                trial = _normalize(remaining | {added})
                if trial != shape and _shape_size(trial)[0] <= 5 and _shape_size(trial)[1] <= 5:
                    variants.add(trial)
    return variants


def _random_polyomino(rng: np.random.Generator, area: int,
                      max_height: int, max_width: int,
                      forbidden=()) -> tuple:
    forbidden = set(forbidden)
    for _ in range(300):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for row, col in cells:
                for candidate in ((row - 1, col), (row + 1, col),
                                  (row, col - 1), (row, col + 1)):
                    if candidate in cells:
                        continue
                    trial = cells | {candidate}
                    rows = [point[0] for point in trial]
                    cols = [point[1] for point in trial]
                    if max(rows) - min(rows) + 1 <= max_height and max(cols) - min(cols) + 1 <= max_width:
                        frontier.add(candidate)
            if not frontier:
                break
            choices = sorted(frontier)
            cells.add(choices[int(rng.integers(len(choices)))])
        if len(cells) == area:
            result = _normalize(cells)
            if result not in forbidden:
                return result
    raise RuntimeError('polyomino vocabulary exhausted')


def _choose_motif(rng: np.random.Generator, areas: tuple,
                  max_height: int, max_width: int, forbidden: set) -> tuple:
    for _ in range(160):
        area = int(areas[int(rng.integers(len(areas)))])
        if rng.random() < 0.25:
            simple = [
                tuple((0, col) for col in range(area)),
                tuple((row, 0) for row in range(area)),
            ]
            if area == 4:
                simple.append(((0, 0), (0, 1), (1, 0), (1, 1)))
            possible = [shape for shape in simple if
                        _shape_size(shape)[0] <= max_height and
                        _shape_size(shape)[1] <= max_width and shape not in forbidden]
            if possible:
                return possible[int(rng.integers(len(possible)))]
        try:
            return _random_polyomino(rng, area, max_height, max_width,
                                     forbidden=forbidden)
        except RuntimeError:
            pass
    raise RuntimeError('motif vocabulary exhausted')


def _new_distractor(rng: np.random.Generator, motif: tuple,
                    forbidden: set, kind: str) -> tuple:
    if kind == 'rotation':
        possibilities = _turns(motif) - forbidden
    elif kind == 'near':
        possibilities = _near_shapes(motif) - forbidden
    elif kind == 'equal_box':
        possibilities = {shape for shape in _near_shapes(motif) - forbidden
                         if _shape_size(shape) == _shape_size(motif)}
    else:
        possibilities = set()
    if possibilities:
        ordered = sorted(possibilities)
        return ordered[int(rng.integers(len(ordered)))]
    area_deltas = (0, -1, 1, -2, 2)
    for index in rng.permutation(len(area_deltas)):
        area = min(8, max(3, len(motif) + area_deltas[int(index)]))
        try:
            return _random_polyomino(rng, area, 5, 5, forbidden=forbidden)
        except RuntimeError:
            pass
    raise RuntimeError('distractor vocabulary exhausted')


def _place_shape(grid: np.ndarray, shape: tuple, color: int,
                 background: int, divider_row: int, divider_col: int,
                 rng: np.random.Generator) -> tuple:
    height, width = grid.shape
    sh, sw = _shape_size(shape)
    origins = []
    for top in range(height - sh + 1):
        for left in range(width - sw + 1):
            cells = tuple((top + row, left + col) for row, col in shape)
            if any(row < divider_row and col < divider_col for row, col in cells):
                continue
            if all(0 <= nr < height and 0 <= nc < width and grid[nr, nc] == background
                   for row, col in cells
                   for nr, nc in ((row, col), (row - 1, col), (row + 1, col),
                                  (row, col - 1), (row, col + 1))
                   if 0 <= nr < height and 0 <= nc < width):
                origins.append(cells)
    if not origins:
        raise RuntimeError('no separated placement')
    chosen = origins[int(rng.integers(len(origins)))]
    for row, col in chosen:
        grid[row, col] = color
    return chosen


def _sample_candidate_colors(rng: np.random.Generator, available: tuple,
                             count: int, mode: str) -> list[int]:
    shuffled = [int(available[i]) for i in rng.permutation(len(available))]
    colors = shuffled[:count]
    if mode == 'distractor_repeat':
        colors[2] = colors[1]
    elif mode == 'match_repeat':
        colors[1] = colors[0]
    elif mode == 'several_share':
        colors[1] = colors[0]
        colors[2] = colors[0]
        for i in range(3, count):
            if rng.random() < 0.5:
                colors[i] = colors[0]
    elif mode == 'monochrome':
        colors = [colors[0]] * count
    return colors


def sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    palette = tuple(int(x) for x in rng.permutation(10))
    return {
        'palette': palette,
        'shared_roles': bool(rng.integers(0, 2)),
        'shared_guide': bool(rng.integers(0, 2)),
        'shared_rotation': bool(rng.integers(0, 2)),
        'drow': int(rng.integers(3, 6)),
        'dcol': int(rng.integers(3, 6)),
        'rotation': int(rng.integers(4)),
    }


def _local_style(parameters: dict, pair_index: int) -> tuple:
    palette = parameters['palette']
    offset = 0 if parameters['shared_roles'] else pair_index
    background = palette[offset % 10]
    control = palette[(offset + 4) % 10]
    if parameters['shared_guide']:
        drow, dcol = parameters['drow'], parameters['dcol']
    else:
        drow = 3 + (parameters['drow'] - 3 + pair_index) % 3
        dcol = 3 + (parameters['dcol'] - 3 + 2 * pair_index) % 3
    rotation = parameters['rotation'] if parameters['shared_rotation'] else (parameters['rotation'] + pair_index) % 4
    return background, control, drow, dcol, rotation


def generate_pair(rng: np.random.Generator, regime: str,
                  parameters: dict, pair_index: int,
                  forbidden_shapes: set) -> tuple[np.ndarray, np.ndarray]:
    background, control, drow, dcol, rotation = _local_style(parameters, pair_index)
    if regime == 'source_train':
        areas = (3, 4, 4, 5, 5, 6)
        counts = (3, 3, 4, 4, 5)
        size_bins = ((10, 12), (12, 16), (16, 20))
        size_probs = (0.38, 0.42, 0.20)
        mirror_probability = 0.30
    elif regime == 'target_test':
        areas = (3, 4, 4, 5, 5, 6, 7)
        counts = (3, 4, 4, 5, 5, 6)
        size_bins = ((10, 12), (12, 16), (16, 21))
        size_probs = (0.28, 0.42, 0.30)
        mirror_probability = 0.42
    else:
        raise ValueError('unknown regime')
    motif = _choose_motif(rng, areas, drow, dcol, forbidden_shapes)
    candidate_count = int(counts[int(rng.integers(len(counts)))])
    size_bin = int(rng.choice(3, p=size_probs))
    low, high = size_bins[size_bin]
    height = int(rng.integers(low, high + 1))
    width = int(rng.integers(low, high + 1))
    if height <= drow + 2 or width <= dcol + 2:
        raise RuntimeError('canvas too small for guide')

    reflected = sorted(_reflection_variants(motif))
    include_mirror = bool(reflected) and rng.random() < mirror_probability
    candidate_shapes = [motif]
    forbidden = {motif}
    if not include_mirror:
        forbidden.update(reflected)
    else:
        mirror = reflected[int(rng.integers(len(reflected)))]
        candidate_shapes.append(mirror)
        forbidden.add(mirror)
    kinds = ('near', 'rotation', 'equal_box', 'random')
    while len(candidate_shapes) < candidate_count:
        kind = kinds[int(rng.integers(len(kinds)))]
        shape = _new_distractor(rng, motif, forbidden, kind)
        candidate_shapes.append(shape)
        forbidden.add(shape)

    if regime == 'source_train' and pair_index == 0:
        color_mode = 'match_repeat'
    elif regime == 'source_train' and pair_index == 1:
        color_mode = 'distractor_repeat'
    else:
        modes = ('distinct', 'distractor_repeat', 'match_repeat',
                 'several_share', 'monochrome')
        color_mode = modes[int(rng.integers(len(modes)))]
    available = tuple(color for color in range(10) if color not in (background, control))
    candidate_colors = _sample_candidate_colors(rng, available, candidate_count, color_mode)
    template_color = candidate_colors[0] if rng.random() < 0.25 else int(available[int(rng.integers(len(available)))])

    grid = np.full((height, width), background, dtype=np.int8)
    grid[:drow + 1, dcol] = control
    grid[drow, :dcol + 1] = control
    mh, mw = _shape_size(motif)
    top = int(rng.integers(drow - mh + 1))
    left = int(rng.integers(dcol - mw + 1))
    for row, col in motif:
        grid[top + row, left + col] = template_color

    order = [int(i) for i in rng.permutation(candidate_count)]
    order.sort(key=lambda i: -len(candidate_shapes[i]))
    matching_cells = None
    for index in order:
        cells = _place_shape(grid, candidate_shapes[index], candidate_colors[index],
                             background, drow, dcol, rng)
        if index == 0:
            matching_cells = cells
    assert matching_cells is not None
    output = grid.copy()
    for row, col in matching_cells:
        output[row, col] = control
    return np.rot90(grid, rotation).copy(), np.rot90(output, rotation).copy()


def _decode_input(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    counts = np.bincount(grid.ravel().astype(int), minlength=10)
    background = int(np.argmax(counts))
    assert int(np.count_nonzero(counts == counts[background])) == 1
    solutions = []
    for control in range(10):
        if control == background or counts[control] == 0:
            continue
        try:
            canonical, k, drow, dcol = _find_canonical_divider(grid, control)
        except AssertionError:
            continue
        height, width = canonical.shape
        compartment = np.zeros((height, width), dtype=bool)
        compartment[:drow, :dcol] = True
        mask = (canonical != background) & (canonical != control)
        components = _components(mask)
        inside, outside = [], []
        valid = True
        for component in components:
            flags = [bool(compartment[row, col]) for row, col in component]
            colors = {int(canonical[row, col]) for row, col in component}
            if (not all(flags) and any(flags)) or len(colors) != 1:
                valid = False
                break
            (inside if all(flags) else outside).append(component)
        if not valid or len(inside) != 1 or not 3 <= len(outside) <= 6:
            continue
        motif = _normalize(inside[0])
        matches = [i for i, component in enumerate(outside)
                   if _normalize(component) == motif]
        if len(matches) != 1:
            continue
        matching_index = matches[0]
        matching_cells = outside[matching_index]
        predicted = canonical.copy()
        for row, col in matching_cells:
            predicted[row, col] = control
        masks = [_normalize(component) for component in outside]
        candidate_colors = [int(canonical[component[0][0], component[0][1]])
                            for component in outside]
        template_color = int(canonical[inside[0][0][0], inside[0][0][1]])
        reflections = _reflection_variants(motif)
        info = {
            'background': background, 'control': control, 'canonical_k': k,
            'drow': drow, 'dcol': dcol, 'template_mask': motif,
            'template_area': len(motif), 'template_color': template_color,
            'candidate_count': len(outside), 'candidate_colors': tuple(candidate_colors),
            'match_color': candidate_colors[matching_index],
            'match_index': matching_index,
            'match_position': (min(row for row, _ in matching_cells),
                               min(col for _, col in matching_cells)),
            'has_reflection': any(i != matching_index and shape in reflections
                                  for i, shape in enumerate(masks)),
            'has_rotation': any(i != matching_index and shape in _turns(motif)
                                for i, shape in enumerate(masks)),
            'has_near': any(i != matching_index and shape in _near_shapes(motif)
                            for i, shape in enumerate(masks)),
            'has_equal_box': any(i != matching_index and len(shape) == len(motif)
                                 and _shape_size(shape) == _shape_size(motif)
                                 for i, shape in enumerate(masks)),
            'is_symmetric': not bool(reflections),
            'candidate_masks': tuple(masks),
            'object_color_count': len(set(candidate_colors + [template_color])),
            'density': float(np.count_nonzero(grid != background) / grid.size),
        }
        solutions.append((np.rot90(predicted, -k).copy(), info))
    assert len(solutions) == 1
    return solutions[0]


def _analyze_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> dict:
    assert input_grid.shape == output_grid.shape
    prediction, info = _decode_input(input_grid)
    assert np.array_equal(prediction, output_grid)
    assert np.any(input_grid != output_grid)
    return info


def _check_count(name: str, value: int | None, lower: int, upper: int) -> None:
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer or None')
    if not lower <= int(value) <= upper:
        raise ValueError(f'{name} outside legal range')


def _resolve_counts(rng: np.random.Generator, num_train: int | None,
                    num_test: int | None) -> tuple[int, int]:
    _check_count('num_train', num_train, 2, 10)
    _check_count('num_test', num_test, 1, 4)
    if num_train is None and num_test is None:
        test_count = int(rng.integers(1, 5))
        train_count = int(rng.integers(max(2, test_count + 1), 11))
    elif num_train is None:
        test_count = int(num_test)
        train_count = int(rng.integers(test_count + 1, 11))
    elif num_test is None:
        train_count = int(num_train)
        test_count = int(rng.integers(1, min(4, train_count - 1) + 1))
    else:
        train_count, test_count = int(num_train), int(num_test)
    if train_count <= test_count:
        raise ValueError('num_train must exceed num_test')
    return train_count, test_count


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    parameters = sample_puzzle_parameters(rng)
    puzzle = {'train': [], 'test': []}
    seen = set()
    used_shapes = set()
    pair_index = 0
    for section, count, regime in (('train', train_count, 'source_train'),
                                   ('test', test_count, 'target_test')):
        for _ in range(count):
            for attempt in range(240):
                pair_rng = np.random.default_rng(int(rng.integers(2**63)))
                try:
                    input_grid, output_grid = generate_pair(
                        pair_rng, regime, parameters, pair_index, used_shapes)
                    info = _analyze_pair(input_grid, output_grid)
                except (AssertionError, RuntimeError):
                    continue
                signature = _array_signature(input_grid)
                if signature in seen or (_grid_text(input_grid), _grid_text(output_grid)) in _SOURCE_PAIR_SIGNATURES:
                    continue
                break
            else:
                raise RuntimeError(f'could not construct pair {pair_index}')
            puzzle[section].append({'input': input_grid, 'output': output_grid})
            seen.add(signature)
            used_shapes.add(info['template_mask'])
            pair_index += 1
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen_inputs, seen_pairs, seen_shapes = set(), set(), set()
    analyses = {'train': [], 'test': []}
    for section in ('train', 'test'):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            input_grid, output_grid = pair['input'], pair['output']
            inp, out = _array_signature(input_grid), _array_signature(output_grid)
            assert inp not in seen_inputs and (inp, out) not in seen_pairs
            seen_inputs.add(inp); seen_pairs.add((inp, out))
            assert (_grid_text(input_grid), _grid_text(output_grid)) not in _SOURCE_PAIR_SIGNATURES
            info = _analyze_pair(input_grid, output_grid)
            assert info['template_mask'] not in seen_shapes
            seen_shapes.add(info['template_mask'])
            analyses[section].append(info)
    train, test = analyses['train'], analyses['test']
    # The first two examples disprove color-based selection rules.
    assert train[0]['match_color'] in [c for i, c in enumerate(train[0]['candidate_colors'])
                                       if i != train[0]['match_index']]
    assert train[1]['match_color'] not in [c for i, c in enumerate(train[1]['candidate_colors'])
                                           if i != train[1]['match_index']]
    distractor_colors = [c for i, c in enumerate(train[1]['candidate_colors'])
                         if i != train[1]['match_index']]
    assert len(set(distractor_colors)) < len(distractor_colors)
    # A fixed training convention must not silently change on a test pair.
    for field in ('background', 'control', 'canonical_k', 'drow', 'dcol'):
        train_values = {info[field] for info in train}
        if len(train_values) == 1:
            assert all(info[field] in train_values for info in test)


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[s]) == len(second[s]) and
               all(np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s])
                   for k in ('input', 'output')) for s in ('train', 'test'))


def self_test() -> None:
    automatic = generate_puzzle(12345)
    explicit = generate_puzzle(67890, 6, 3)
    assert _puzzles_equal(explicit, generate_puzzle(67890, 6, 3))
    assert not _puzzles_equal(generate_puzzle(91, 5, 2), generate_puzzle(92, 5, 2))
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(1000 + 10 * train_count + test_count, train_count, test_count)
        assert (len(puzzle['train']), len(puzzle['test'])) == (train_count, test_count)
    for index, (train_count, test_count) in enumerate(((None, 1), (None, 4),
                                                        (2, None), (10, None))):
        puzzle = generate_puzzle(2000 + index, train_count, test_count)
        if train_count is not None:
            assert len(puzzle['train']) == train_count
        if test_count is not None:
            assert len(puzzle['test']) == test_count
    for args in ({'num_train': 1}, {'num_train': 11}, {'num_test': 0},
                 {'num_test': 5}, {'num_train': 2, 'num_test': 2},
                 {'num_train': 3, 'num_test': 4}):
        try:
            generate_puzzle(3000, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f'illegal counts accepted: {args}')
    train_counts, test_counts = set(), set()
    dimensions, masks, rotations, counts = set(), set(), set(), set()
    color_patterns, reflections, symmetries = set(), set(), set()
    backgrounds, controls, guide_sizes, densities = set(), set(), set(), []
    layouts, signatures = set(), set()
    split_patterns = {'train': set(), 'test': set()}
    split_areas = {'train': set(), 'test': set()}
    split_counts = {'train': set(), 'test': set()}
    split_dimensions = {'train': set(), 'test': set()}
    split_reflections = {'train': set(), 'test': set()}
    decoy_features = {'has_near': set(), 'has_rotation': set(),
                      'has_equal_box': set()}
    role_profiles = {'background': set(), 'control': set(),
                     'canonical_k': set(), 'guide': set()}
    same_template_match = set()
    match_regions = set()
    aspects = set()
    compact_five_candidate_test = False
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle['train'])); test_counts.add(len(puzzle['test']))
        pair_infos = []
        for section in ('train', 'test'):
            for pair in puzzle[section]:
                prediction, info = _decode_input(pair['input'])
                assert np.array_equal(prediction, pair['output'])
                pair_infos.append(info)
                dimensions.add(pair['input'].shape)
                split_dimensions[section].add(pair['input'].shape)
                masks.add(info['template_mask'])
                rotations.add(info['canonical_k'])
                counts.add(info['candidate_count'])
                split_counts[section].add(info['candidate_count'])
                split_areas[section].add(info['template_area'])
                split_reflections[section].add(info['has_reflection'])
                backgrounds.add(info['background']); controls.add(info['control'])
                guide_sizes.add((info['drow'], info['dcol']))
                densities.append(info['density'])
                reflections.add(info['has_reflection'])
                symmetries.add(info['is_symmetric'])
                same_template_match.add(info['template_color'] == info['match_color'])
                for feature in decoy_features:
                    decoy_features[feature].add(info[feature])
                row, col = info['match_position']
                match_regions.add('upper_right' if row < info['drow'] else
                                  'lower_left' if col < info['dcol'] else 'lower_right')
                height, width = pair['input'].shape
                aspects.add('wide' if width > height else 'tall' if height > width else 'square')
                if section == 'test' and (height, width) == (10, 10) and info['candidate_count'] >= 5:
                    compact_five_candidate_test = True
                colors = info['candidate_colors']
                histogram = sorted(colors.count(color) for color in set(colors))
                color_patterns.add(tuple(histogram))
                pattern = ('monochrome' if len(set(colors)) == 1 else
                           'distinct' if len(set(colors)) == len(colors) else
                           'several_share' if max(histogram) >= 3 else
                           'match_shared' if colors.count(info['match_color']) > 1 else
                           'distractor_shared')
                split_patterns[section].add(pattern)
                layouts.add((pair['input'].shape, info['candidate_count'],
                             info['match_position'], info['template_mask']))
                signatures.add(_array_signature(pair['input']))
        for field in ('background', 'control', 'canonical_k'):
            role_profiles[field].add(len({info[field] for info in pair_infos}) > 1)
        role_profiles['guide'].add(len({(info['drow'], info['dcol']) for info in pair_infos}) > 1)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(dimensions) >= 25 and len(masks) >= 25
    assert len(rotations) == 4 and len(counts) >= 3
    assert len(backgrounds) >= 7 and len(controls) >= 7
    assert len(guide_sizes) >= 5 and len(color_patterns) >= 4
    assert reflections == {False, True} and symmetries == {False, True}
    assert min(densities) < 0.15 and max(densities) > 0.25
    assert len(layouts) >= 0.8 * len(signatures)
    assert all(profile == {False, True} for profile in role_profiles.values())
    assert all(values == {False, True} for values in decoy_features.values())
    assert all(values == {False, True} for values in split_reflections.values())
    assert same_template_match == {False, True}
    assert match_regions == {'upper_right', 'lower_left', 'lower_right'}
    assert aspects == {'wide', 'tall', 'square'}
    assert compact_five_candidate_test
    assert len(split_dimensions['train'] & split_dimensions['test']) >= 5
    assert len(split_areas['train'] & split_areas['test']) >= 3
    assert len(split_counts['train'] & split_counts['test']) >= 3
    assert {'distinct', 'distractor_shared', 'match_shared',
            'several_share', 'monochrome'} <= split_patterns['test']
