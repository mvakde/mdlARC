import numpy as np


_SOURCE_PAIR_ROWS = (
    (
        (
            "0000000000",
            "0444000000",
            "0444000000",
            "0444000666",
            "0444000666",
            "0444000666",
            "0444000666",
            "0444000666",
            "0444000000",
            "0000000000",
        ),
        (
            "0000000000",
            "0444000000",
            "0444000000",
            "0444000666",
            "0444888666",
            "0444888666",
            "0444888666",
            "0444000666",
            "0444000000",
            "0000000000",
        ),
    ),
    (
        (
            "0000000000",
            "0222200000",
            "0222200000",
            "0222200000",
            "0000000000",
            "0000000000",
            "0000000000",
            "7777770000",
            "7777770000",
            "7777770000",
        ),
        (
            "0000000000",
            "0222200000",
            "0222200000",
            "0222200000",
            "0088000000",
            "0088000000",
            "0088000000",
            "7777770000",
            "7777770000",
            "7777770000",
        ),
    ),
    (
        (
            "3333333330",
            "3333333330",
            "3333333330",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0009999990",
            "0009999990",
        ),
        (
            "3333333330",
            "3333333330",
            "3333333330",
            "0000888800",
            "0000888800",
            "0000888800",
            "0000888800",
            "0000888800",
            "0009999990",
            "0009999990",
        ),
    ),
    (
        (
            "1111000000",
            "1111000222",
            "1111000222",
            "1111000222",
            "1111000222",
            "1111000222",
            "1111000222",
            "1111000000",
            "0000000000",
            "0000000000",
        ),
        (
            "1111000000",
            "1111000222",
            "1111888222",
            "1111888222",
            "1111888222",
            "1111888222",
            "1111000222",
            "1111000000",
            "0000000000",
            "0000000000",
        ),
    ),
)

_SOURCE_PAIR_TEXTS = {
    ("/".join(input_rows), "/".join(output_rows))
    for input_rows, output_rows in _SOURCE_PAIR_ROWS
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _validate_optional_count(
    name: str,
    value: int | None,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}, inclusive"
        )
    return value


def _resolve_counts(
    seed_index: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _validate_optional_count("num_train", num_train, 2, 10)
    num_test = _validate_optional_count("num_test", num_test, 1, 4)

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test

    if num_train is None and num_test is None:
        # This seed-indexed schedule guarantees broad count diversity while
        # remaining deterministic. All other generation uses the local RNG.
        resolved_test = 1 + (seed_index % 4)
        minimum_train = max(2, resolved_test + 1)
        span = 10 - minimum_train + 1
        resolved_train = minimum_train + ((seed_index // 4) % span)
        return resolved_train, resolved_test

    if num_train is None:
        minimum_train = max(2, int(num_test) + 1)
        resolved_train = int(rng.integers(minimum_train, 11))
        return resolved_train, int(num_test)

    maximum_test = min(4, int(num_train) - 1)
    if maximum_test < 1:
        raise ValueError("no legal num_test exists for the requested num_train")
    resolved_test = int(rng.integers(1, maximum_test + 1))
    return int(num_train), resolved_test



def _sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    connector = int(rng.integers(0, 10))
    orientation_mode = str(rng.choice(('mixed', 'mixed', 'mixed', 'horizontal', 'vertical')))
    common_canvas = bool(rng.random() < 0.30)
    if common_canvas:
        if rng.random() < 0.45:
            canvas = (10, 10)
        else:
            canvas = (int(rng.integers(9, 24)), int(rng.integers(9, 24)))
    else:
        canvas = None
    return {'connector': connector, 'orientation_mode': orientation_mode, 'canvas': canvas}


def _grid_signature(grid: np.ndarray) -> tuple:
    return (tuple(int(v) for v in grid.shape), grid.tobytes())


def _grid_text(grid: np.ndarray) -> str:
    return '/'.join(''.join(str(int(v)) for v in row) for row in grid)


def _is_source_pair(inp: np.ndarray, out: np.ndarray) -> bool:
    return (_grid_text(inp), _grid_text(out)) in _SOURCE_PAIR_TEXTS


def _rectangle_bbox(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    ys, xs = np.where(grid == color)
    _require(bool(ys.size), 'missing rectangle color')
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    region = grid[y0:y1 + 1, x0:x1 + 1]
    _require(bool(np.all(region == color)) and int(ys.size) == region.size,
             'object is not one solid rectangle')
    return y0, y1, x0, x1


def _count_boundary_sides(boxes: tuple, height: int, width: int) -> int:
    sides = set()
    for y0, y1, x0, x1 in boxes:
        if y0 == 0:
            sides.add('top')
        if y1 == height - 1:
            sides.add('bottom')
        if x0 == 0:
            sides.add('left')
        if x1 == width - 1:
            sides.add('right')
    return len(sides)


def _geometry_key(info: dict) -> tuple:
    return (info['orientation'], info['overlap'], info['gap'],
            info['aligned_ends'], info['boundary_count'])


def _interpret_input(inp: np.ndarray, connector: int) -> tuple[dict, np.ndarray]:
    colors = tuple(int(v) for v in np.unique(inp))
    _require(len(colors) == 3 and connector not in colors,
             'input needs a background and two colors, excluding connector')
    candidates = []
    for background in colors:
        object_colors = tuple(c for c in colors if c != background)
        try:
            boxes = tuple(_rectangle_bbox(inp, c) for c in object_colors)
        except AssertionError:
            continue
        a, b = boxes
        for orientation in ('horizontal', 'vertical'):
            if orientation == 'horizontal':
                first, second = sorted(boxes, key=lambda box: box[2])
                gap_start, gap_end = first[3] + 1, second[2] - 1
                shared_start = max(first[0], second[0])
                shared_end = min(first[1], second[1])
                gap_region = inp[shared_start:shared_end + 1, gap_start:gap_end + 1]
            else:
                first, second = sorted(boxes, key=lambda box: box[0])
                gap_start, gap_end = first[1] + 1, second[0] - 1
                shared_start = max(first[2], second[2])
                shared_end = min(first[3], second[3])
                gap_region = inp[gap_start:gap_end + 1, shared_start:shared_end + 1]
            overlap = shared_end - shared_start + 1
            gap = gap_end - gap_start + 1
            if gap < 1 or overlap < 3 or not bool(np.all(gap_region == background)):
                continue
            expected = inp.copy()
            if orientation == 'horizontal':
                expected[shared_start + 1:shared_end, gap_start:gap_end + 1] = connector
                spans = ((first[0], first[1]), (second[0], second[1]))
                depths = (first[3] - first[2] + 1, second[3] - second[2] + 1)
                opposing = first[2] == 0 and second[3] == inp.shape[1] - 1
            else:
                expected[gap_start:gap_end + 1, shared_start + 1:shared_end] = connector
                spans = ((first[2], first[3]), (second[2], second[3]))
                depths = (first[1] - first[0] + 1, second[1] - second[0] + 1)
                opposing = first[0] == 0 and second[1] == inp.shape[0] - 1
            aligned = sum((spans[0][0] == spans[1][0], spans[0][1] == spans[1][1]))
            contains = ((spans[0][0] <= spans[1][0] and spans[0][1] >= spans[1][1])
                        or (spans[1][0] <= spans[0][0] and spans[1][1] >= spans[0][1]))
            info = {
                'background': background, 'orientation': orientation,
                'overlap': overlap, 'gap': gap, 'connector_cells': gap * (overlap - 2),
                'aligned_ends': aligned, 'containment': contains,
                'depths': depths, 'object_palette': tuple(sorted(object_colors)),
                'boundary_count': _count_boundary_sides(boxes, *inp.shape),
                'opposing_boundaries': opposing,
                'shape': tuple(int(v) for v in inp.shape),
            }
            candidates.append((info, expected))
    _require(len(candidates) == 1, 'input has no unique two-rectangle interpretation')
    return candidates[0]


def _sample_longitudinal(rng: np.random.Generator, width: int,
                         edge_mode: str, min_gap: int, prefer_thin: bool) -> tuple:
    triples = []
    for gap in range(min_gap, min(14, width - 2) + 1):
        for first_depth in range(1, min(10, width - gap - 1) + 1):
            for second_depth in range(1, min(10, width - gap - first_depth) + 1):
                slack = width - gap - first_depth - second_depth
                if edge_mode == 'interior' and slack < 2:
                    continue
                if edge_mode == 'opposing' and slack != 0:
                    continue
                if edge_mode == 'one' and slack < 1:
                    continue
                triples.append((gap, first_depth, second_depth, slack))
    _require(bool(triples), 'canvas has no feasible longitudinal geometry')
    if prefer_thin:
        thin = [t for t in triples if 1 in t[1:3]]
        if thin:
            triples = thin
    gaps = sorted({t[0] for t in triples})
    gap = gaps[int(rng.integers(0, len(gaps)))]
    choices = [t for t in triples if t[0] == gap]
    gap, first_depth, second_depth, slack = choices[int(rng.integers(0, len(choices)))]
    if edge_mode == 'interior':
        before = int(rng.integers(1, slack))
    elif edge_mode == 'opposing':
        before = 0
    elif edge_mode == 'one':
        before = 0 if rng.random() < 0.5 else slack
    else:
        before = int(rng.integers(0, slack + 1))
    return gap, first_depth, second_depth, before


def _sample_transverse(rng: np.random.Generator, height: int,
                       edge_mode: str, minimum_overlap: int,
                       avoid_overlap: int | None) -> tuple:
    minimum_slack = 2 if edge_mode == 'interior' else 0
    overlaps = list(range(minimum_overlap, min(14, height - minimum_slack) + 1))
    if avoid_overlap in overlaps and len(overlaps) > 1:
        overlaps.remove(avoid_overlap)
    overlap = overlaps[int(rng.integers(0, len(overlaps)))]
    room = height - overlap - minimum_slack
    modes = []
    for pre_mode in (0, 1, 2):
        for post_mode in (0, 1, 2):
            if int(pre_mode != 0) + int(post_mode != 0) <= room:
                modes.append((pre_mode, post_mode))
    pre_mode, post_mode = modes[int(rng.integers(0, len(modes)))]
    pre_amount = 0
    post_amount = 0
    if pre_mode:
        pre_amount = int(rng.integers(1, min(5, room - int(post_mode != 0)) + 1))
    if post_mode:
        post_amount = int(rng.integers(1, min(5, room - pre_amount) + 1))
    pre = (pre_amount, 0) if pre_mode == 1 else ((0, pre_amount) if pre_mode == 2 else (0, 0))
    post = (post_amount, 0) if post_mode == 1 else ((0, post_amount) if post_mode == 2 else (0, 0))
    slack = height - overlap - max(pre) - max(post)
    if edge_mode == 'interior':
        before = int(rng.integers(1, slack))
    elif edge_mode == 'contact':
        before = 0 if rng.random() < 0.5 else slack
    else:
        before = int(rng.integers(0, slack + 1))
    shared_start = before + max(pre)
    return ((shared_start - pre[0], shared_start + overlap - 1 + post[0]),
            (shared_start - pre[1], shared_start + overlap - 1 + post[1]),
            shared_start, overlap)


def _build_pair(rng: np.random.Generator, shape: tuple[int, int], orientation: str,
                background: int, colors: tuple[int, int], connector: int,
                minimum_overlap: int, min_gap: int,
                avoid_overlap: int | None) -> tuple[np.ndarray, np.ndarray]:
    horizontal = orientation == 'horizontal'
    height, width = shape if horizontal else shape[::-1]
    long_mode = str(rng.choice(('interior', 'one', 'opposing', 'free')))
    trans_mode = str(rng.choice(('interior', 'contact', 'free', 'free')))
    prefer_thin = bool(rng.random() < 0.28)
    gap, depth_a, depth_b, left_margin = _sample_longitudinal(
        rng, width, long_mode, min_gap, prefer_thin)
    (y_a0, y_a1), (y_b0, y_b1), shared_start, overlap = _sample_transverse(
        rng, height, trans_mode, minimum_overlap, avoid_overlap)
    x_a0 = left_margin
    x_a1 = x_a0 + depth_a - 1
    x_b0 = x_a1 + gap + 1
    x_b1 = x_b0 + depth_b - 1
    inp = np.full((height, width), background, dtype=np.int8)
    inp[y_a0:y_a1 + 1, x_a0:x_a1 + 1] = colors[0]
    inp[y_b0:y_b1 + 1, x_b0:x_b1 + 1] = colors[1]
    out = inp.copy()
    out[shared_start + 1:shared_start + overlap - 1, x_a1 + 1:x_b0] = connector
    if not horizontal:
        inp, out = inp.T.copy(), out.T.copy()
    return inp, out


def _choose_colors(rng: np.random.Generator, connector: int,
                   previous_palettes: list[tuple[int, int]]) -> tuple[int, tuple[int, int]]:
    available = [c for c in range(10) if c != connector]
    if previous_palettes and rng.random() < 0.38:
        palette = previous_palettes[int(rng.integers(0, len(previous_palettes)))]
    else:
        pair = rng.choice(np.asarray(available), size=2, replace=False)
        palette = tuple(sorted((int(pair[0]), int(pair[1]))))
    background = int(rng.choice(np.asarray([c for c in available if c not in palette])))
    colors = palette if rng.random() < 0.5 else palette[::-1]
    return background, colors


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(int(seed), rng, num_train, num_test)
    parameters = _sample_puzzle_parameters(rng)
    connector = parameters['connector']
    pair_seeds = rng.integers(0, np.iinfo(np.uint64).max,
                              size=train_count + test_count, dtype=np.uint64)
    first_orientation = str(rng.choice(('horizontal', 'vertical')))
    puzzle = {'train': [], 'test': []}
    seen_inputs = set()
    train_geometries = set()
    palettes = []
    first_overlap = None
    for index, pair_seed in enumerate(pair_seeds):
        is_train = index < train_count
        local_index = index if is_train else index - train_count
        pair_rng = np.random.default_rng(pair_seed)
        if parameters['orientation_mode'] == 'mixed':
            if is_train and local_index < 2:
                orientation = first_orientation if local_index == 0 else (
                    'vertical' if first_orientation == 'horizontal' else 'horizontal')
            else:
                orientation = str(pair_rng.choice(('horizontal', 'vertical')))
        else:
            orientation = parameters['orientation_mode']
        shape = parameters['canvas']
        if shape is None:
            shape = (int(pair_rng.integers(9, 31)), int(pair_rng.integers(9, 31)))
        background, colors = _choose_colors(pair_rng, connector, palettes)
        minimum_overlap = 5 if is_train and local_index == 0 else 3
        min_gap = 2 if is_train and local_index == 0 else 1
        avoid_overlap = first_overlap if is_train and local_index == 1 else None
        made = False
        for _ in range(24):
            inp, out = _build_pair(pair_rng, shape, orientation, background, colors,
                                   connector, minimum_overlap, min_gap, avoid_overlap)
            signature = _grid_signature(inp)
            if signature in seen_inputs or _is_source_pair(inp, out):
                continue
            try:
                info, predicted = _interpret_input(inp, connector)
            except AssertionError:
                continue
            if (np.array_equal(predicted, out) and info['background'] == background
                    and (is_train or _geometry_key(info) not in train_geometries)):
                made = True
                break
        _require(made, 'bounded pair construction could not find a unique input')
        if is_train and local_index == 0:
            first_overlap = info['overlap']
        seen_inputs.add(signature)
        if is_train:
            train_geometries.add(_geometry_key(info))
        palettes.append(tuple(sorted(colors)))
        puzzle['train' if is_train else 'test'].append({'input': inp, 'output': out})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'},
             'puzzle must have only train and test lists')
    _require(isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list),
             'train and test must be lists')
    train_count, test_count = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             'illegal train/test counts')
    connector = None
    seen_inputs, seen_pairs = set(), set()
    analyses = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'},
                     'pair must have only input and output')
            inp, out = pair['input'], pair['output']
            for grid in (inp, out):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         'grid must be a two-dimensional NumPy array')
                _require(all(1 <= n <= 30 for n in grid.shape), 'grid size outside ARC bounds')
                _require(np.issubdtype(grid.dtype, np.integer), 'grid dtype must be integer')
                _require(0 <= int(grid.min()) and int(grid.max()) <= 9, 'illegal ARC color')
            _require(inp.shape == out.shape, 'output dimensions changed')
            inp_sig, out_sig = _grid_signature(inp), _grid_signature(out)
            _require(inp_sig not in seen_inputs and (inp_sig, out_sig) not in seen_pairs,
                     'duplicate pair or input')
            seen_inputs.add(inp_sig)
            seen_pairs.add((inp_sig, out_sig))
            _require(not _is_source_pair(inp, out), 'exact source pair copied')
            changed = inp != out
            _require(bool(np.any(changed)), 'connector must change cells')
            old_colors = np.unique(inp[changed])
            new_colors = np.unique(out[changed])
            _require(len(old_colors) == len(new_colors) == 1,
                     'changes need one background and one connector color')
            pair_connector = int(new_colors[0])
            _require(pair_connector != int(old_colors[0]), 'connector did not change color')
            if connector is None:
                connector = pair_connector
            _require(pair_connector == connector, 'connector color changed within puzzle')
            info, expected = _interpret_input(inp, connector)
            _require(info['background'] == int(old_colors[0]), 'wrong background replaced')
            _require(np.array_equal(out, expected), 'output differs from inset full-gap bridge')
            _require(int(changed.sum()) == info['connector_cells'], 'connector area is wrong')
            analyses[split].append(info)
    train_orientations = {a['orientation'] for a in analyses['train']}
    _require(all(a['orientation'] in train_orientations for a in analyses['test']),
             'test orientation absent from training')
    _require(analyses['train'][0]['overlap'] >= 5 and analyses['train'][0]['gap'] >= 2,
             'first training example must show a visible inset and full gap')
    _require(analyses['train'][0]['overlap'] != analyses['train'][1]['overlap'],
             'training must vary overlap')
    train_geometries = {_geometry_key(info) for info in analyses['train']}
    _require(all(_geometry_key(info) not in train_geometries for info in analyses['test']),
             'test must recombine geometry beyond the training layouts')


def _puzzles_equal(first: dict, second: dict) -> bool:
    for split in ('train', 'test'):
        if len(first[split]) != len(second[split]):
            return False
        for a, b in zip(first[split], second[split]):
            if not np.array_equal(a['input'], b['input']) or not np.array_equal(a['output'], b['output']):
                return False
    return True


def self_test() -> None:
    train_counts, test_counts = set(), set()
    puzzle_signatures = set()
    all_info = {'train': [], 'test': []}
    common_canvas_count = 0
    source_size_count = 0
    variable_canvas_count = 0
    single_orientation_count = 0
    repeated_palette_count = 0
    new_palette_count = 0
    background_vary_count = 0
    alignment_vary_count = 0
    combination_counts = {'vertical_aligned_interior': 0,
                          'horizontal_aligned_edge': 0, 'thin_long_gap': 0}
    connector_colors = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _require(_puzzles_equal(puzzle, generate_puzzle(seed)), 'generation is not deterministic')
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        puzzle_signatures.add(tuple((split, tuple((_grid_signature(p['input']),
            _grid_signature(p['output'])) for p in puzzle[split])) for split in ('train', 'test')))
        shapes, palettes, backgrounds = set(), [], set()
        orientations = set()
        alignments = set()
        conn = int(np.unique(puzzle['train'][0]['output'][
            puzzle['train'][0]['input'] != puzzle['train'][0]['output']])[0])
        connector_colors.add(conn)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info, expected = _interpret_input(pair['input'], conn)
                _require(np.array_equal(pair['output'], expected), 'independent input reconstruction failed')
                all_info[split].append(info)
                shapes.add(info['shape'])
                palettes.append(info['object_palette'])
                backgrounds.add(info['background'])
                orientations.add(info['orientation'])
                alignments.add(info['aligned_ends'])
                combination_counts['vertical_aligned_interior'] += (
                    info['orientation'] == 'vertical' and info['aligned_ends'] >= 1
                    and info['boundary_count'] == 0)
                combination_counts['horizontal_aligned_edge'] += (
                    info['orientation'] == 'horizontal' and info['aligned_ends'] >= 1
                    and info['boundary_count'] >= 1)
                combination_counts['thin_long_gap'] += (
                    1 in info['depths'] and info['gap'] >= 9)
        common_canvas_count += len(shapes) == 1
        source_size_count += shapes == {(10, 10)}
        variable_canvas_count += len(shapes) > 1
        single_orientation_count += len(orientations) == 1
        repeated_palette_count += len(set(palettes)) < len(palettes)
        new_palette_count += len(set(palettes)) > 1
        background_vary_count += len(backgrounds) > 1
        alignment_vary_count += len(alignments) > 1
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, 'automatic counts lack range')
    _require(len(puzzle_signatures) >= 40, 'cross-seed puzzles repeat')
    _require(len(connector_colors) >= 5 and background_vary_count >= 15,
             'color roles do not vary enough')
    _require(common_canvas_count >= 3 and single_orientation_count >= 3,
             'common-canvas or single-orientation puzzles absent')
    _require(source_size_count >= 2 and variable_canvas_count >= 15,
             'source-size or variable-canvas puzzles absent')
    _require(repeated_palette_count >= 8, 'object palettes never repeat within puzzles')
    _require(new_palette_count >= 20 and alignment_vary_count >= 25,
             'within-puzzle palette or alignment variation too narrow')
    _require(all(count >= 5 for count in combination_counts.values()),
             'compatible orientation, edge, alignment, and gap combinations absent')
    _require(any(i['connector_cells'] == 1 for infos in all_info.values() for i in infos),
             'one-cell connectors absent')
    _require(len({i['shape'] for infos in all_info.values() for i in infos}) >= 80,
             'canvas shapes too repetitive')
    for split in ('train', 'test'):
        infos = all_info[split]
        _require({i['orientation'] for i in infos} == {'horizontal', 'vertical'},
                 f'{split} lacks an orientation')
        _require(len({i['overlap'] for i in infos}) >= 6 and len({i['gap'] for i in infos}) >= 9,
                 f'{split} overlap or gap diversity is too narrow')
        _require({i['aligned_ends'] for i in infos} == {0, 1, 2},
                 f'{split} lacks an alignment class')
        _require(any(i['containment'] for i in infos) and any(not i['containment'] for i in infos),
                 f'{split} lacks containment or partial offset')
        _require(any(1 in i['depths'] for i in infos) and any(i['gap'] >= 9 for i in infos),
                 f'{split} lacks one-cell slabs or long gaps')
        _require(any(i['boundary_count'] == 0 for i in infos) and
                 any(i['boundary_count'] >= 2 for i in infos),
                 f'{split} lacks interior or boundary-contact examples')
    _require(any(i['orientation'] == 'horizontal' and i['boundary_count'] >= 2
                 for i in all_info['train']), 'horizontal boundary-contact training absent')
    for train, test in ((2, 1), (10, 1), (10, 4), (5, 3)):
        puzzle = generate_puzzle(1000 + train * 10 + test, train, test)
        _require((len(puzzle['train']), len(puzzle['test'])) == (train, test),
                 'explicit count request ignored')
    for train, test in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(1200 + (train or 0) * 10 + (test or 0), train, test)
        _require((train is None or len(puzzle['train']) == train) and
                 (test is None or len(puzzle['test']) == test), 'partial count request ignored')
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(7, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count request accepted')
