import numpy as np


_AUTO_COUNT_OPTIONS = ((2, 1), (3, 1), (4, 1), (4, 2), (5, 2),
                       (5, 3), (6, 3), (6, 4), (7, 1), (8, 2),
                       (9, 3), (10, 4))


def _rows(*rows):
    return np.array([[int(c) for c in row] for row in rows], dtype=np.int8)


_SOURCE_PAIRS = (
    (_rows('0000000', '0003000', '0033300', '0333300', '0330000', '0033000', '6000000'),
     _rows('0000000', '0006000', '0066600', '0666600', '0660000', '0066000', '0000000')),
    (_rows('0000000', '0222000', '0020000', '0222200', '0022200', '0002000', '4000000'),
     _rows('0000000', '0444000', '0040000', '0444400', '0044400', '0004000', '0000000')),
    (_rows('0000000', '0888000', '0888880', '0008800', '0088000', '0088800', '2000000'),
     _rows('0000000', '0222000', '0222220', '0002200', '0022000', '0022200', '0000000')),
)


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _validate_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer or None')
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f'{name} must be between {low} and {high}')
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    num_train = _validate_count('num_train', num_train, 2, 10)
    num_test = _validate_count('num_test', num_test, 1, 4)
    if num_train is None and num_test is None:
        return _AUTO_COUNT_OPTIONS[seed % len(_AUTO_COUNT_OPTIONS)]
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return num_train, num_test


def _neighbors(cell):
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _normalize(cells):
    cells = tuple(cells)
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _connected(cells):
    remaining = set(cells)
    if not remaining:
        return False
    stack = [remaining.pop()]
    while stack:
        for neighbor in _neighbors(stack.pop()):
            if neighbor in remaining:
                remaining.remove(neighbor)
                stack.append(neighbor)
    return not remaining


def _grow(rng, size):
    cells = {(0, 0)}
    style = int(rng.integers(0, 3))
    while len(cells) < size:
        frontier = sorted({n for cell in cells for n in _neighbors(cell)
                           if n not in cells and -5 <= n[0] <= 5 and -5 <= n[1] <= 5})
        weights = np.array([sum(n in cells for n in _neighbors(f)) for f in frontier], dtype=float)
        if style == 0:
            weights = (weights + 0.5) ** 2
        elif style == 1:
            weights = 1 / (weights + 0.5) ** 2
        else:
            weights[:] = 1
        cells.add(frontier[int(rng.choice(len(frontier), p=weights / weights.sum()))])
    return _normalize(cells)


def _shape(rng, low, high):
    kinds = ['bar', 'rectangle', 'hollow', 'bent', 'irregular']
    for _ in range(36):
        kind = kinds[int(rng.integers(0, len(kinds)))]
        if kind == 'bar':
            n = int(rng.integers(2, 13))
            cells = [(0, c) for c in range(n)] if rng.integers(0, 2) else [(r, 0) for r in range(n)]
        elif kind == 'rectangle':
            h, w = int(rng.integers(2, 6)), int(rng.integers(2, 6))
            cells = [(r, c) for r in range(h) for c in range(w)]
        elif kind == 'hollow':
            h, w = int(rng.integers(3, 7)), int(rng.integers(3, 7))
            cells = [(r, c) for r in range(h) for c in range(w)
                     if r in (0, h - 1) or c in (0, w - 1)]
        elif kind == 'bent':
            a, b = int(rng.integers(2, 8)), int(rng.integers(2, 8))
            cells = {(0, c) for c in range(a)} | {(r, 0) for r in range(b)}
            if rng.integers(0, 2):
                branch = int(rng.integers(1, 4))
                cells |= {(r, a - 1) for r in range(branch)}
            cells = list(cells)
        else:
            if high < 3:
                continue
            cells = _grow(rng, int(rng.integers(max(3, low), min(23, high) + 1)))
        if low <= len(cells) <= high:
            cells = _normalize(cells)
            if rng.integers(0, 2):
                cells = _normalize((c, r) for r, c in cells)
            if rng.integers(0, 2):
                max_c = max(c for _, c in cells)
                cells = _normalize((r, max_c - c) for r, c in cells)
            return cells
    return _grow(rng, int(rng.integers(max(2, low), min(23, high) + 1)))


def _marker_candidates(height, width, category):
    corners = [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]
    if category == 'corner':
        return corners
    if category == 'edge':
        return [(r, c) for r in range(height) for c in range(width)
                if (r in (0, height - 1) or c in (0, width - 1)) and (r, c) not in corners]
    return [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]


def _make_pair(rng, background, object_color, swatch_color, marker_category,
               fixed_corner, size_range, anchor=None):
    for _ in range(100):
        cells = _shape(rng, *size_range)
        shape_h = max(r for r, _ in cells) + 1
        shape_w = max(c for _, c in cells) + 1
        if anchor is None:
            height = int(rng.integers(max(5, shape_h + 2), min(19, shape_h + 8)))
            width = int(rng.integers(max(5, shape_w + 2), min(19, shape_w + 8)))
        else:
            height, width = anchor
        if shape_h > height - 2 or shape_w > width - 2 or len(cells) + 1 >= height * width / 2:
            continue
        markers = _marker_candidates(height, width, marker_category)
        if fixed_corner is not None:
            markers = [markers[fixed_corner]]
        marker = markers[int(rng.integers(0, len(markers)))]
        interior = bool(rng.random() < 0.62)
        origins = []
        for r0 in range(height - shape_h + 1):
            for c0 in range(width - shape_w + 1):
                translated = [(r + r0, c + c0) for r, c in cells]
                if any(max(abs(r - marker[0]), abs(c - marker[1])) <= 1 for r, c in translated):
                    continue
                inside = r0 > 0 and c0 > 0 and r0 + shape_h < height and c0 + shape_w < width
                if inside == interior:
                    origins.append((r0, c0))
        if not origins:
            continue
        r0, c0 = origins[int(rng.integers(0, len(origins)))]
        inp = np.full((height, width), background, dtype=np.int8)
        for r, c in cells:
            inp[r + r0, c + c0] = object_color
        inp[marker] = swatch_color
        out = inp.copy()
        out[inp == object_color] = swatch_color
        out[marker] = background
        return {'input': inp, 'output': out}
    raise RuntimeError('unable to place object and isolated swatch')


def _grid_key(grid):
    return (grid.shape, tuple(int(x) for x in grid.ravel()))


def _is_source_pair(inp, out):
    return any(np.array_equal(inp, a) and np.array_equal(out, b) for a, b in _SOURCE_PAIRS)


def _random_colors(rng, background, prior=None):
    available = [c for c in range(10) if c != background]
    if prior is None:
        chosen = rng.choice(available, size=2, replace=False)
        return int(chosen[0]), int(chosen[1])
    old_object, old_swatch = prior
    objects = [c for c in available if c != old_object]
    obj = int(objects[int(rng.integers(0, len(objects)))])
    swatches = [c for c in available if c not in (obj, old_swatch)]
    swatch = int(swatches[int(rng.integers(0, len(swatches)))])
    return obj, swatch


def _test_colors(rng, background, train_roles, mode):
    train_objects = {o for o, _ in train_roles}
    train_swatches = {s for _, s in train_roles}
    available = [c for c in range(10) if c != background]
    if mode == 0:  # Source-like: new object color, old object color as swatch.
        swatches = [c for c in train_objects if c in available]
        swatch = int(swatches[int(rng.integers(0, len(swatches)))]) if swatches else None
        objects = [c for c in available if c not in train_objects | train_swatches and c != swatch]
        if not objects:
            objects = [c for c in available if c not in train_objects and c != swatch]
        if objects:
            return int(objects[int(rng.integers(0, len(objects)))]), swatch
    elif mode == 1:  # Reverse demonstrated color roles.
        choices = [(o, s) for o in train_swatches for s in train_objects
                   if o in available and s in available and o != s]
        if choices:
            return choices[int(rng.integers(0, len(choices)))]
    elif mode == 2:  # New literal pairing, preferably new role colors.
        choices = [(o, s) for o in available for s in available if o != s
                   and (o, s) not in train_roles]
        if choices:
            novel = [pair for pair in choices if pair[0] not in train_objects
                     and pair[1] not in train_swatches]
            pool = novel or choices
            return pool[int(rng.integers(0, len(pool)))]
    else:  # A familiar mapping with a new shape and placement.
        familiar = [pair for pair in train_roles if background not in pair]
        if familiar:
            return familiar[int(rng.integers(0, len(familiar)))]
    return _random_colors(rng, background)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, rng, num_train, num_test)

    # The repaint-and-erase meaning is the only puzzle-level semantic convention.
    fixed_corner = int(rng.integers(0, 4)) if rng.random() < 0.22 else None
    background_fixed = bool(rng.random() < 0.58)
    common_background = int(rng.integers(0, 10))
    first_background = common_background
    anchor_test = bool(rng.random() < 0.34)
    train_categories = list(rng.permutation(['corner', 'edge', 'interior']))
    test_color_modes = list(rng.permutation(4))
    test_size_modes = list(rng.permutation(3))
    puzzle = {'train': [], 'test': []}
    used_inputs, used_shapes = set(), set()
    train_roles, train_sizes = [], []

    def add_pair(split, pair_rng, bg, obj, sw, category, size_range, anchor=None):
        for _ in range(60):
            try:
                pair = _make_pair(pair_rng, bg, obj, sw, category, fixed_corner,
                                  size_range, anchor)
            except RuntimeError:
                if anchor is not None:
                    anchor = None
                    continue
                raise
            inp, out = pair['input'], pair['output']
            info = _analyze_pair(pair)
            key = _grid_key(inp)
            if key in used_inputs or info['shape'] in used_shapes or _is_source_pair(inp, out):
                continue
            used_inputs.add(key)
            used_shapes.add(info['shape'])
            puzzle[split].append(pair)
            return info
        raise RuntimeError('unable to find a distinct pair')

    for i in range(num_train):
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        if background_fixed or i == 0:
            bg = first_background
        elif i == 1:
            bg = int(pair_rng.choice([c for c in range(10) if c != first_background]))
        else:
            bg = int(pair_rng.integers(0, 10))
        if i == 0:
            obj, sw = _random_colors(pair_rng, bg)
        elif i == 1 and pair_rng.random() < 0.5 and bg not in train_roles[0]:
            obj, sw = train_roles[0][1], train_roles[0][0]
        elif i == 1:
            obj, sw = _random_colors(pair_rng, bg, train_roles[0])
        else:
            obj, sw = _random_colors(pair_rng, bg)
        train_roles.append((obj, sw))
        category = 'corner' if fixed_corner is not None else train_categories[i % 3]
        low, high = (2, 5) if i >= 2 and pair_rng.random() < 0.17 else (5, 18)
        info = add_pair('train', pair_rng, bg, obj, sw, category, (low, high))
        train_sizes.append(info['size'])

    for i in range(num_test):
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        bg = common_background if background_fixed else int(pair_rng.integers(0, 10))
        obj, sw = _test_colors(pair_rng, bg, train_roles, test_color_modes[i % 4])
        size_mode = test_size_modes[i % 3]
        if size_mode == 0 and min(train_sizes) > 2:
            size_range = (2, min(train_sizes) - 1)
        elif size_mode == 1:
            chosen = train_sizes[int(pair_rng.integers(0, len(train_sizes)))]
            size_range = (chosen, chosen)
        else:
            size_range = (max(train_sizes) + 1, 23)
        if size_range[0] > size_range[1]:
            size_range = (3, 20)
        category = 'corner' if fixed_corner is not None else ['corner', 'edge', 'interior'][int(pair_rng.integers(0, 3))]
        anchor = puzzle['train'][0]['input'].shape if anchor_test and i == 0 else None
        add_pair('test', pair_rng, bg, obj, sw, category, size_range, anchor)

    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    inp, out = pair['input'], pair['output']
    _check(inp.shape == out.shape, 'input/output shape changed')
    colors, counts = np.unique(inp, return_counts=True)
    _check(len(colors) == 3, 'input needs exactly three color roles')
    _check(sum(counts == 1) == 1, 'input needs one singleton swatch')
    background = int(colors[int(np.argmax(counts))])
    _check(int(counts.max()) > inp.size / 2, 'background must dominate')
    swatch = int(colors[int(np.flatnonzero(counts == 1)[0])])
    obj = int(next(c for c in colors if c not in (background, swatch)))
    cells = {tuple(int(x) for x in p) for p in np.argwhere(inp == obj)}
    _check(len(cells) >= 2 and _connected(cells), 'object must be a connected multi-cell shape')
    marker = tuple(int(x) for x in np.argwhere(inp == swatch)[0])
    _check(all(max(abs(r - marker[0]), abs(c - marker[1])) > 1 for r, c in cells),
           'swatch needs a full eight-neighbor gap')
    expected = inp.copy()
    expected[inp == obj] = swatch
    expected[marker] = background
    _check(np.array_equal(out, expected), 'incorrect repaint-and-erase output')
    _check(not np.array_equal(inp, out), 'degenerate pair')
    return {'background': background, 'object': obj, 'swatch': swatch,
            'size': len(cells), 'shape': _normalize(cells), 'marker': marker,
            'interior': all(0 < r < inp.shape[0] - 1 and 0 < c < inp.shape[1] - 1
                            for r, c in cells)}


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _check(isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list), 'splits must be lists')
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    _check(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, 'invalid pair counts')
    inputs, pairs, shapes, train_info = set(), set(), set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            for grid in (pair['input'], pair['output']):
                _check(isinstance(grid, np.ndarray) and grid.ndim == 2, 'grid must be a 2D array')
                _check(np.issubdtype(grid.dtype, np.integer), 'grid must be integral')
                _check(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30, 'invalid grid size')
                _check(int(grid.min()) >= 0 and int(grid.max()) <= 9, 'invalid ARC color')
            input_key = _grid_key(pair['input'])
            pair_key = (input_key, _grid_key(pair['output']))
            _check(input_key not in inputs and pair_key not in pairs, 'duplicate input or pair')
            inputs.add(input_key)
            pairs.add(pair_key)
            _check(not _is_source_pair(pair['input'], pair['output']), 'copied source pair')
            info = _analyze_pair(pair)
            _check(info['shape'] not in shapes, 'translated duplicate object shape')
            shapes.add(info['shape'])
            if split == 'train':
                train_info.append(info)
    _check(len({x['object'] for x in train_info}) >= 2, 'training object colors must vary')
    _check(len({x['swatch'] for x in train_info}) >= 2, 'training swatch colors must vary')


def _infer_from_visible_training(train_pairs, test_input):
    # This check receives only visible training pairs and the test input.
    for pair in train_pairs:
        _analyze_pair(pair)
    colors, counts = np.unique(test_input, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    swatch = int(colors[int(np.flatnonzero(counts == 1)[0])])
    obj = int(next(c for c in colors if c not in (background, swatch)))
    marker = tuple(int(x) for x in np.argwhere(test_input == swatch)[0])
    prediction = test_input.copy()
    prediction[test_input == obj] = swatch
    prediction[marker] = background
    return prediction


def _marker_kind(marker, shape):
    r, c = marker
    h, w = shape
    corners = ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))
    if marker in corners:
        return 'corner', corners.index(marker)
    if r in (0, h - 1) or c in (0, w - 1):
        return 'edge', None
    return 'interior', None


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _check(len({len(p['train']) for p in puzzles}) >= 4, 'train counts lack variety')
    _check(len({len(p['test']) for p in puzzles}) >= 3, 'test counts lack variety')
    signatures, dimensions, shapes, layouts = set(), set(), set(), set()
    categories, sizes, backgrounds, positions = set(), set(), set(), set()
    interior_count = edge_count = variable_swatch = variable_background = 0
    variable_size = fixed_corner_puzzles = role_overlap = multi_swatch = 0
    repeated_color_pairs = same_canvas = background_role_reuse = combined = 0
    source_like = reversal = wholly_new = familiar = 0
    genuinely_new_source_like = branching_count = bent_count = rectangular_canvas = 0
    marker_categories = set()
    gaps, directions = set(), set()
    size_relations = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_grid_key(pair['input']) for split in ('train', 'test') for pair in puzzle[split]))
        train_info = [_analyze_pair(pair) for pair in puzzle['train']]
        test_info = [_analyze_pair(pair) for pair in puzzle['test']]
        all_pairs = puzzle['train'] + puzzle['test']
        all_info = train_info + test_info
        kinds = [_marker_kind(x['marker'], pair['input'].shape)
                 for x, pair in zip(all_info, all_pairs)]
        marker_categories.update(kind for kind, _ in kinds)
        variable_swatch += len({kind for kind, _ in kinds[:len(train_info)]}) > 1
        fixed_corner_puzzles += len(set(kinds)) == 1 and kinds[0][0] == 'corner'
        variable_background += len({x['background'] for x in train_info}) > 1
        train_sizes = {x['size'] for x in train_info}
        variable_size += len(train_sizes) > 1
        train_objects = {x['object'] for x in train_info}
        train_swatches = {x['swatch'] for x in train_info}
        train_color_pairs = [(x['object'], x['swatch']) for x in train_info]
        role_overlap += bool(train_objects & train_swatches)
        multi_swatch += any(len({s for o, s in train_color_pairs if o == color}) > 1
                             for color in train_objects)
        repeated_color_pairs += len(set(train_color_pairs)) < len(train_color_pairs)
        same_canvas += any(a['input'].shape == b['input'].shape
                           for a in puzzle['train'] for b in puzzle['test'])
        background_role_reuse += any(x['background'] in (y['object'], y['swatch'])
                                     for x in all_info for y in all_info if x is not y)
        combined += (len({kind for kind, _ in kinds[:len(train_info)]}) > 1
                     and len({x['background'] for x in train_info}) > 1
                     and len(train_sizes) > 1)
        for pair, info in zip(puzzle['test'], test_info):
            n = info['size']
            size_relations.add('equal' if n in train_sizes else 'smaller' if n < min(train_sizes)
                               else 'larger' if n > max(train_sizes) else 'between')
            source_like += info['object'] not in train_objects and info['swatch'] in train_objects
            genuinely_new_source_like += (info['object'] not in train_objects | train_swatches
                                          and info['swatch'] in train_objects)
            reversal += info['object'] in train_swatches and info['swatch'] in train_objects
            wholly_new += (info['object'] not in train_objects | train_swatches
                           and info['swatch'] not in train_objects | train_swatches)
            familiar += (info['object'], info['swatch']) in train_color_pairs
            _check(np.array_equal(_infer_from_visible_training(puzzle['train'], pair['input']),
                                  pair['output']), 'visible-input prediction failed')
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _analyze_pair(pair)
                inp = pair['input']
                dimensions.add(inp.shape)
                shapes.add(info['shape'])
                layouts.add((inp.shape, info['shape'], info['marker']))
                sizes.add(info['size'])
                backgrounds.add(info['background'])
                positions.add(info['marker'])
                interior_count += info['interior']
                edge_count += not info['interior']
                rectangular_canvas += inp.shape[0] != inp.shape[1]
                object_cells = np.argwhere(inp == info['object'])
                marker_row, marker_col = info['marker']
                gaps.add(min(max(abs(int(r) - marker_row), abs(int(c) - marker_col))
                             for r, c in object_cells))
                center_row, center_col = object_cells.mean(axis=0)
                directions.add((int(np.sign(center_row - marker_row)),
                                int(np.sign(center_col - marker_col))))
                h = max(r for r, _ in info['shape']) + 1
                w = max(c for _, c in info['shape']) + 1
                category = ('bar' if h == 1 or w == 1 else 'rectangle' if info['size'] == h * w
                            else 'hollow' if info['size'] == 2 * (h + w) - 4 and
                            all(r in (0, h - 1) or c in (0, w - 1) for r, c in info['shape'])
                            else 'other')
                categories.add(category)
                if category == 'other':
                    degree = max(sum(neighbor in info['shape'] for neighbor in _neighbors(cell))
                                 for cell in info['shape'])
                    branching_count += degree >= 3
                    bent_count += degree == 2
    _check(len(signatures) == 42, 'seeds repeated puzzles')
    _check(len(dimensions) >= 20 and len(shapes) >= 120 and len(layouts) >= 160,
           'structural diversity too low')
    _check(len(sizes) >= 15 and len(backgrounds) >= 8 and len(positions) >= 25,
           'size, background, or swatch position diversity too low')
    _check(categories == {'bar', 'rectangle', 'hollow', 'other'}, 'geometry classes missing')
    _check(branching_count >= 10 and bent_count >= 10, 'branching or thin bends missing')
    _check(interior_count >= 40 and edge_count >= 30, 'object placement lacks variety')
    _check(rectangular_canvas >= 100 and len(gaps) >= 4 and len(directions) >= 4,
           'aspect ratios or object-swatch spacing lack variety')
    _check(variable_swatch >= 15 and variable_background >= 8, 'within-puzzle variation too low')
    _check(variable_size >= 30 and fixed_corner_puzzles >= 3,
           'size variation or fixed-corner examples missing')
    _check(marker_categories == {'corner', 'edge', 'interior'}, 'swatch locations lack variety')
    _check(role_overlap >= 10 and multi_swatch >= 5 and repeated_color_pairs <= 15,
           'training color-role variation too weak')
    _check(same_canvas >= 5 and background_role_reuse >= 3 and combined >= 3,
           'independent variation axes lack combinations')
    _check(source_like >= 5 and genuinely_new_source_like >= 3 and reversal >= 5
           and wholly_new >= 1 and familiar >= 3,
           'target color regimes missing')
    _check({'smaller', 'equal', 'larger'} <= size_relations, 'test size shifts missing')
    for seed, nt, ns in ((101, 2, 1), (102, 10, 1), (103, 10, 4), (104, 6, 2),
                         (201, None, 1), (202, None, 4), (203, 2, None), (204, 10, None)):
        puzzle = generate_puzzle(seed, nt, ns)
        if nt is not None:
            _check(len(puzzle['train']) == nt, 'explicit train count changed')
        if ns is not None:
            _check(len(puzzle['test']) == ns, 'explicit test count changed')
    for args in ({'num_train': 1}, {'num_train': 11}, {'num_test': 0}, {'num_test': 5},
                 {'num_train': 2, 'num_test': 2}, {'num_train': 3, 'num_test': 4},
                 {'num_train': 4, 'num_test': 4}):
        try:
            generate_puzzle(999, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f'illegal counts accepted: {args}')
    for seed, nt, ns in ((314159, 7, 3), (271828, None, None)):
        a, b = generate_puzzle(seed, nt, ns), generate_puzzle(seed, nt, ns)
        _check(all(np.array_equal(pa[k], pb[k]) for split in ('train', 'test')
                   for pa, pb in zip(a[split], b[split]) for k in ('input', 'output')),
               'generation is not deterministic')
