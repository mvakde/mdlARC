import numpy as np


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _normalize_count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    train_count = _normalize_count(num_train, "num_train")
    test_count = _normalize_count(num_test, "num_test")

    if train_count is not None and not 2 <= train_count <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if test_count is not None and not 1 <= test_count <= 4:
        raise ValueError("num_test must be between 1 and 4")

    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        low = max(2, test_count + 1)
        train_count = low + (seed // 4) % (11 - low)
    elif train_count is None:
        low = max(2, test_count + 1)
        train_count = low + seed % (11 - low)
    elif test_count is None:
        maximum = min(4, train_count - 1)
        test_count = 1 + seed % maximum

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def sample_puzzle_parameters(rng):
    colors = rng.choice(np.arange(10), size=3, replace=False)
    return {
        "background": int(colors[0]),
        "structural": int(colors[1]),
        "marker": int(colors[2]),
    }


def _components(mask):
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    result = []

    for start_r in range(height):
        for start_c in range(width):
            if not mask[start_r, start_c] or seen[start_r, start_c]:
                continue

            stack = [(start_r, start_c)]
            seen[start_r, start_c] = True
            cells = []

            while stack:
                r, c = stack.pop()
                cells.append((r, c))

                if r > 0 and mask[r - 1, c] and not seen[r - 1, c]:
                    seen[r - 1, c] = True
                    stack.append((r - 1, c))
                if r + 1 < height and mask[r + 1, c] and not seen[r + 1, c]:
                    seen[r + 1, c] = True
                    stack.append((r + 1, c))
                if c > 0 and mask[r, c - 1] and not seen[r, c - 1]:
                    seen[r, c - 1] = True
                    stack.append((r, c - 1))
                if c + 1 < width and mask[r, c + 1] and not seen[r, c + 1]:
                    seen[r, c + 1] = True
                    stack.append((r, c + 1))

            result.append(np.asarray(cells, dtype=np.int16))

    return result


def _rectangle_mask(shape, r0, r1, c0, c1):
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _projection_mask(shape, bar):
    height, width = shape
    mask = np.zeros(shape, dtype=bool)
    thickness = bar["thickness"]

    if bar["orientation"] == "horizontal":
        marker_c = bar["marker_c"]
        band_c0 = marker_c - (thickness - 1)
        band_c1 = marker_c + (thickness - 1)

        if bar["direction"] == "up":
            mask[0 : bar["r0"], band_c0 : band_c1 + 1] = True
        else:
            mask[
                bar["r1"] + 1 : height,
                band_c0 : band_c1 + 1,
            ] = True
    else:
        marker_r = bar["marker_r"]
        band_r0 = marker_r - (thickness - 1)
        band_r1 = marker_r + (thickness - 1)

        if bar["direction"] == "left":
            mask[band_r0 : band_r1 + 1, 0 : bar["c0"]] = True
        else:
            mask[
                band_r0 : band_r1 + 1,
                bar["c1"] + 1 : width,
            ] = True

    return mask


def _analyze_roles(grid, background, structural, marker):
    _check(
        np.count_nonzero(grid == marker) == 2,
        "There must be exactly two marker cells",
    )

    object_mask = grid != background
    components = _components(object_mask)
    _check(len(components) == 2, "The input must contain two separate bars")

    bars = {}

    for component in components:
        rows = component[:, 0]
        cols = component[:, 1]
        r0 = int(rows.min())
        r1 = int(rows.max())
        c0 = int(cols.min())
        c1 = int(cols.max())

        rect = grid[r0 : r1 + 1, c0 : c1 + 1]
        rect_height, rect_width = rect.shape

        _check(
            component.shape[0] == rect_height * rect_width,
            "Each object must fill its bounding rectangle",
        )
        _check(
            np.all(rect != background),
            "A bar rectangle cannot contain background holes",
        )
        _check(
            np.all((rect == structural) | (rect == marker)),
            "A bar contains an unexpected color",
        )
        _check(
            np.count_nonzero(rect == marker) == 1,
            "Each bar must contain exactly one marker",
        )
        _check(
            rect_height != rect_width,
            "Bars must have a unique long orientation",
        )

        marker_position = np.argwhere(rect == marker)[0]
        marker_r = r0 + int(marker_position[0])
        marker_c = c0 + int(marker_position[1])

        if rect_width > rect_height:
            orientation = "horizontal"
            thickness = rect_height
            length = rect_width

            _check(thickness >= 2, "Training-family bars have thickness >= 2")
            _check(
                length >= 2 * thickness - 1,
                "The bar is too short for its centered boundary motif",
            )
            _check(
                marker_r in (r0, r1),
                "A horizontal marker must lie on a long side",
            )

            offset = marker_c - c0
            _check(
                thickness - 1 <= offset <= length - thickness,
                "The marker lacks the required side clearance",
            )
            direction = "up" if marker_r == r0 else "down"
        else:
            orientation = "vertical"
            thickness = rect_width
            length = rect_height

            _check(thickness >= 2, "Training-family bars have thickness >= 2")
            _check(
                length >= 2 * thickness - 1,
                "The bar is too short for its centered boundary motif",
            )
            _check(
                marker_c in (c0, c1),
                "A vertical marker must lie on a long side",
            )

            offset = marker_r - r0
            _check(
                thickness - 1 <= offset <= length - thickness,
                "The marker lacks the required side clearance",
            )
            direction = "left" if marker_c == c0 else "right"

        _check(
            orientation not in bars,
            "There must be one horizontal and one vertical bar",
        )

        bars[orientation] = {
            "orientation": orientation,
            "r0": r0,
            "r1": r1,
            "c0": c0,
            "c1": c1,
            "marker_r": marker_r,
            "marker_c": marker_c,
            "thickness": int(thickness),
            "length": int(length),
            "direction": direction,
        }

    _check(
        set(bars) == {"horizontal", "vertical"},
        "The two bars must have orthogonal orientations",
    )

    horizontal_projection = _projection_mask(
        grid.shape, bars["horizontal"]
    )
    vertical_projection = _projection_mask(grid.shape, bars["vertical"])

    _check(
        np.any(horizontal_projection),
        "The horizontal bar must have a nonempty projection",
    )
    _check(
        np.any(vertical_projection),
        "The vertical bar must have a nonempty projection",
    )
    _check(
        np.all(grid[horizontal_projection] == background),
        "The horizontal projection path must contain only background",
    )
    _check(
        np.all(grid[vertical_projection] == background),
        "The vertical projection path must contain only background",
    )
    _check(
        not np.any(horizontal_projection & vertical_projection),
        "The two projection footprints must not conflict",
    )

    return {
        "background": int(background),
        "structural": int(structural),
        "marker": int(marker),
        "bars": bars,
    }


def _analyze_input(grid):
    colors, counts = np.unique(grid, return_counts=True)
    _check(len(colors) == 3, "Each input must use exactly three colors")

    colors = [int(color) for color in colors]
    count_by_color = {
        int(color): int(count) for color, count in zip(colors, counts)
    }
    marker_candidates = [
        color for color in colors if count_by_color[color] == 2
    ]
    _check(
        marker_candidates,
        "The marker color must occur exactly once in each bar",
    )

    valid_analyses = []

    for marker in marker_candidates:
        remaining = [color for color in colors if color != marker]
        for background in remaining:
            structural = remaining[0] if remaining[1] == background else remaining[1]
            try:
                analysis = _analyze_roles(
                    grid, background, structural, marker
                )
            except AssertionError:
                continue
            valid_analyses.append(analysis)

    _check(
        len(valid_analyses) == 1,
        "Input color roles and bar segmentation must be unambiguous",
    )
    return valid_analyses[0]


def _expected_output(input_grid, analysis):
    result = input_grid.copy()
    structural = analysis["structural"]
    marker = analysis["marker"]
    height, width = input_grid.shape

    for orientation in ("horizontal", "vertical"):
        bar = analysis["bars"][orientation]
        thickness = bar["thickness"]

        if orientation == "horizontal":
            marker_c = bar["marker_c"]
            c0 = marker_c - (thickness - 1)
            c1 = marker_c + (thickness - 1)

            if bar["direction"] == "up":
                rows = slice(0, bar["r0"])
            else:
                rows = slice(bar["r1"] + 1, height)

            result[rows, c0 : c1 + 1] = structural
            result[rows, marker_c] = marker
        else:
            marker_r = bar["marker_r"]
            r0 = marker_r - (thickness - 1)
            r1 = marker_r + (thickness - 1)

            if bar["direction"] == "left":
                cols = slice(0, bar["c0"])
            else:
                cols = slice(bar["c1"] + 1, width)

            result[r0 : r1 + 1, cols] = structural
            result[marker_r, cols] = marker

    return result


import hashlib


_SOURCE_INPUT_HASHES = {
    '79bdd633e883e9e920576bb72956f9fbf64a2ca03e65a69a7decee3d0d35ff6f',
    '696de825c5470c933b092624c7c40f3c47d2252b39ff0fa2019ed542108fc06d',
    'f227ef1c229bcea5c7dbe436fea584c31878414c0e17befcac3519241f4136c2',
    '3ef9e24ab3e23f736811c8e035285600b4b5e8d4fc0558d75181008b21e02bf7',
    '67f4c8629e69bda3d05f58df3cce4dfeb4997d88d92fd18da1777a47681e2bee',
}


def _grid_key(grid):
    canonical = np.ascontiguousarray(grid, dtype=np.int8)
    return tuple(canonical.shape), canonical.tobytes()


def _source_input(grid):
    canonical = np.ascontiguousarray(grid, dtype=np.int8)
    digest = hashlib.sha256(bytes(canonical.shape) + canonical.tobytes()).hexdigest()
    return digest in _SOURCE_INPUT_HASHES


def _arrangement(analysis):
    h = analysis['bars']['horizontal']
    v = analysis['bars']['vertical']
    dr = (h['r0'] + h['r1']) - (v['r0'] + v['r1'])
    dc = (h['c0'] + h['c1']) - (v['c0'] + v['c1'])
    if abs(dr) <= 2:
        return 'aligned_rows'
    if abs(dc) <= 2:
        return 'aligned_cols'
    return ('h' if dr < 0 else 'v') + ('_ul' if (dr < 0) == (dc < 0) else '_ur')


def _sample_shape(rng, min_span, mode):
    for _ in range(40):
        if mode == 'small_square':
            side = int(rng.integers(10, 15))
            shape = (side, side)
        elif mode == 'medium_square':
            side = int(rng.integers(17, 23))
            shape = (side, side)
        elif mode == 'large_square':
            side = int(rng.integers(23, 31))
            shape = (side, side)
        elif mode == 'rectangle':
            shape = (int(rng.integers(10, 31)), int(rng.integers(10, 31)))
            if abs(shape[0] - shape[1]) < 3:
                continue
        else:
            raise ValueError('Unknown shape mode')
        if min(shape) >= min_span:
            return shape
    # A requested thick bar cannot fit on a tiny square.
    side = max(16, min_span)
    if mode == 'rectangle':
        return side, min(30, side + 5)
    return side, side


def _sample_length(rng, thickness, span):
    minimum = 2 * thickness - 1
    if minimum == span:
        return minimum
    mode = int(rng.integers(0, 3))
    if mode == 0:
        upper = min(span, minimum + 3)
        return int(rng.integers(minimum, upper + 1))
    if mode == 1:
        lower = max(minimum, (minimum + span) // 2)
        return int(rng.integers(lower, span + 1))
    return int(rng.integers(minimum, span + 1))


def _sample_offset(rng, thickness, length):
    low, high = thickness - 1, length - thickness
    if low == high:
        return low
    if int(rng.integers(0, 3)) == 0:
        return low if int(rng.integers(0, 2)) == 0 else high
    return int(rng.integers(low, high + 1))


def _construct_pair(rng, spec, palette):
    ht, vt = spec['thicknesses']
    hd, vd = spec['directions']
    for attempt in range(2400):
        height, width = _sample_shape(rng, max(2 * ht, 2 * vt), spec['shape_mode'])
        hl = _sample_length(rng, ht, width)
        vl = _sample_length(rng, vt, height)
        hr0 = int(rng.integers(1 if hd == 'up' else 0,
                                height - ht + (0 if hd == 'down' else 1)))
        hc0 = int(rng.integers(0, width - hl + 1))
        vr0 = int(rng.integers(0, height - vl + 1))
        vc0 = int(rng.integers(1 if vd == 'left' else 0,
                                width - vt + (0 if vd == 'right' else 1)))
        hr1, hc1 = hr0 + ht - 1, hc0 + hl - 1
        vr1, vc1 = vr0 + vl - 1, vc0 + vt - 1
        hm_c = hc0 + _sample_offset(rng, ht, hl)
        vm_r = vr0 + _sample_offset(rng, vt, vl)
        hbar = {'orientation': 'horizontal', 'r0': hr0, 'r1': hr1,
                'c0': hc0, 'c1': hc1, 'marker_r': hr0 if hd == 'up' else hr1,
                'marker_c': hm_c, 'thickness': ht, 'length': hl,
                'direction': hd}
        vbar = {'orientation': 'vertical', 'r0': vr0, 'r1': vr1,
                'c0': vc0, 'c1': vc1, 'marker_r': vm_r,
                'marker_c': vc0 if vd == 'left' else vc1,
                'thickness': vt, 'length': vl, 'direction': vd}
        shape = (height, width)
        hmask = _rectangle_mask(shape, hr0, hr1, hc0, hc1)
        vmask = _rectangle_mask(shape, vr0, vr1, vc0, vc1)
        if np.any(hmask & vmask):
            continue
        hp = _projection_mask(shape, hbar)
        vp = _projection_mask(shape, vbar)
        if not np.any(hp) or not np.any(vp):
            continue
        if np.any(hp & (vmask | vp)) or np.any(vp & hmask):
            continue
        background, structural, marker = palette
        grid = np.full(shape, background, dtype=np.int8)
        grid[hmask | vmask] = structural
        grid[hbar['marker_r'], hbar['marker_c']] = marker
        grid[vbar['marker_r'], vbar['marker_c']] = marker
        try:
            analysis = _analyze_input(grid)
        except AssertionError:
            continue
        if _source_input(grid):
            continue
        return grid, _expected_output(grid, analysis)
    raise ValueError('Unable to construct a separated pair')


def _new_palette(rng, previous=None):
    for _ in range(30):
        colors = tuple(int(v) for v in rng.choice(10, size=3, replace=False))
        if colors != previous:
            return colors
    raise ValueError('Unable to recolor pair')


def _new_spec(rng, shape_mode=None, directions=None, thicknesses=None):
    if shape_mode is None:
        shape_mode = ('small_square', 'medium_square',
                      'large_square', 'rectangle')[int(rng.integers(4))]
    if thicknesses is None:
        upper = 4 if shape_mode == 'small_square' else 7
        thicknesses = (int(rng.integers(2, upper)), int(rng.integers(2, upper)))
    if directions is None:
        directions = ('up' if int(rng.integers(2)) == 0 else 'down',
                      'left' if int(rng.integers(2)) == 0 else 'right')
    if shape_mode == 'small_square' and max(thicknesses) >= 5:
        shape_mode = 'medium_square'
    return {'thicknesses': thicknesses, 'directions': directions,
            'shape_mode': shape_mode}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    if seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    base_palette = _new_palette(rng)
    local_colors = bool(rng.integers(0, 2))
    first_directions = ('up', 'left') if int(rng.integers(2)) == 0 else ('up', 'right')
    second_directions = ('down', 'right' if first_directions[1] == 'left' else 'left')
    shape_modes = ('small_square', 'medium_square', 'large_square', 'rectangle')
    first_mode = shape_modes[int(rng.integers(4))]
    second_mode = shape_modes[int(rng.integers(4))]
    if second_mode == first_mode:
        second_mode = shape_modes[(shape_modes.index(first_mode) + int(rng.integers(1, 4))) % 4]
    first_upper = 4 if first_mode == 'small_square' else 7
    second_upper = 4 if second_mode == 'small_square' else 7
    first_thicknesses = (int(rng.integers(2, first_upper)),
                         int(rng.integers(2, first_upper)))
    while True:
        second_thicknesses = (int(rng.integers(2, second_upper)),
                              int(rng.integers(2, second_upper)))
        if len(set(first_thicknesses + second_thicknesses)) >= 2:
            break
    train_specs = [
        _new_spec(rng, first_mode, first_directions, first_thicknesses),
        _new_spec(rng, second_mode, second_directions, second_thicknesses),
    ]
    train_specs.extend(_new_spec(rng) for _ in range(train_count - 2))
    train_specs = [train_specs[int(i)] for i in rng.permutation(train_count)]
    test_specs = []
    for index in range(test_count):
        exemplar = train_specs[int(rng.integers(train_count))]
        shift = int(rng.integers(4))
        if seed % 7 == 0 and index == 0:
            spec = _new_spec(rng, directions=('up', 'left'), thicknesses=(6, 3))
        elif shift == 0:
            spec = _new_spec(rng, directions=exemplar['directions'],
                             thicknesses=exemplar['thicknesses'])
        elif shift == 1:
            spec = _new_spec(rng, directions=exemplar['directions'])
        elif shift == 2:
            d = exemplar['directions']
            spec = _new_spec(rng, directions=('down' if d[0] == 'up' else 'up',
                                               'right' if d[1] == 'left' else 'left'),
                             thicknesses=exemplar['thicknesses'])
        else:
            spec = _new_spec(rng)
        test_specs.append(spec)
    puzzle = {'train': [], 'test': []}
    used = set()
    palettes = [base_palette]
    if local_colors:
        palettes.append(_new_palette(rng, base_palette))
    for split, specs in (('train', train_specs), ('test', test_specs)):
        for index, spec in enumerate(specs):
            if local_colors and split == 'train' and index == 1:
                palette = palettes[1]
            elif local_colors and (split == 'test' or index > 1) and int(rng.integers(2)) == 0:
                palette = _new_palette(rng)
            else:
                palette = base_palette
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            for _ in range(12):
                input_grid, output_grid = _construct_pair(pair_rng, spec, palette)
                key = _grid_key(input_grid)
                if key not in used:
                    used.add(key)
                    puzzle[split].append({'input': input_grid, 'output': output_grid})
                    break
            else:
                raise ValueError('Unable to make unique input')
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid, name):
    _check(isinstance(grid, np.ndarray), f'{name} must be an ndarray')
    _check(grid.ndim == 2, f'{name} must be two-dimensional')
    _check(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30,
           f'{name} dimensions are outside 1..30')
    _check(np.issubdtype(grid.dtype, np.integer), f'{name} needs integer dtype')
    _check(np.all((grid >= 0) & (grid <= 9)), f'{name} has an invalid color')


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'},
           'Puzzle needs exactly train and test')
    _check(isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list),
           'Splits must be lists')
    train_count, test_count = len(puzzle['train']), len(puzzle['test'])
    _check(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
           'Invalid pair counts')
    seen = set()
    train_analyses = []
    palettes = {'train': set(), 'test': set()}
    shapes = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {'input', 'output'},
                   'Pair needs input and output')
            inp, out = pair['input'], pair['output']
            _validate_grid(inp, 'input')
            _validate_grid(out, 'output')
            _check(inp.shape == out.shape, 'Shape changed')
            key = _grid_key(inp)
            _check(key not in seen, 'Duplicate input')
            seen.add(key)
            _check(not _source_input(inp), 'An original source input was copied')
            analysis = _analyze_input(inp)
            _check(np.array_equal(out, _expected_output(inp, analysis)),
                   'Incorrect projection')
            _check(not np.array_equal(inp, out), 'Invisible transformation')
            for bar in analysis['bars'].values():
                _check(2 <= bar['thickness'] <= 6, 'Unsupported thickness')
            palettes[split].add((analysis['background'], analysis['structural'],
                                 analysis['marker']))
            shapes.add(inp.shape)
            if split == 'train':
                train_analyses.append(analysis)
    _check(len(shapes) >= 2, 'Puzzle lacks shape variation')
    directions = {orientation: set() for orientation in ('horizontal', 'vertical')}
    thicknesses = set()
    for analysis in train_analyses:
        for orientation, bar in analysis['bars'].items():
            directions[orientation].add(bar['direction'])
            thicknesses.add(bar['thickness'])
    _check(directions['horizontal'] == {'up', 'down'} and
           directions['vertical'] == {'left', 'right'},
           'Training must demonstrate every side direction')
    _check(len(thicknesses) >= 2, 'Training must demonstrate thickness scaling')
    if len(palettes['train'] | palettes['test']) > 1:
        _check(len(palettes['train']) > 1, 'Local recoloring must appear in training')


def _puzzles_equal(first, second):
    for split in ('train', 'test'):
        if len(first[split]) != len(second[split]):
            return False
        for a, b in zip(first[split], second[split]):
            if not np.array_equal(a['input'], b['input']) or not np.array_equal(a['output'], b['output']):
                return False
    return True


def self_test() -> None:
    for seed in (0, 1, 17, 41):
        validate_generated_puzzle(generate_puzzle(seed))
    explicit = generate_puzzle(918273, 7, 3)
    _check(_puzzles_equal(explicit, generate_puzzle(918273, 7, 3)),
           'Generation is not deterministic')
    _check(not _puzzles_equal(explicit, generate_puzzle(918274, 7, 3)),
           'Different seeds should yield different puzzles')
    for nt, ns in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                   (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(300 + (nt or 0) * 10 + (ns or 0), nt, ns))
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(300, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal pair count accepted')
    counts = {'train': set(), 'test': set()}
    shapes, layouts, inputs = set(), set(), set()
    test_directions, arrangements, lengths = set(), set(), set()
    six_train = six_test = recolored = 0
    ten_square = twenty_square = large_square = rectangle = 0
    long_bars = asymmetric_markers = short_short = long_long = 0
    aligned = vertical_six_test = 0
    mixed_lengths = mixed_arrangements = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        counts['train'].add(len(puzzle['train']))
        counts['test'].add(len(puzzle['test']))
        local_palettes = set()
        puzzle_lengths = set()
        puzzle_arrangements = set()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input']
                analysis = _analyze_input(inp)
                _check(np.array_equal(pair['output'], _expected_output(inp, analysis)),
                       'Self-test projection mismatch')
                shapes.add(inp.shape)
                ten_square += inp.shape == (10, 10)
                twenty_square += inp.shape == (20, 20)
                large_square += inp.shape[0] == inp.shape[1] and inp.shape[0] >= 23
                rectangle += inp.shape[0] != inp.shape[1]
                inputs.add(_grid_key(inp))
                local_palettes.add((analysis['background'], analysis['structural'], analysis['marker']))
                arrangement = _arrangement(analysis)
                arrangements.add(arrangement)
                puzzle_arrangements.add(arrangement)
                aligned += arrangement.startswith('aligned')
                h, v = analysis['bars']['horizontal'], analysis['bars']['vertical']
                lengths.update((h['length'], v['length']))
                for bar in (h, v):
                    puzzle_lengths.add(bar['length'])
                    long_bars += bar['length'] >= 18
                    offset = (bar['marker_c'] - bar['c0'] if bar['orientation'] == 'horizontal'
                              else bar['marker_r'] - bar['r0'])
                    left_tail = offset - (bar['thickness'] - 1)
                    right_tail = bar['length'] - bar['thickness'] - offset
                    asymmetric_markers += abs(left_tail - right_tail) >= 5
                h_distance = int(np.count_nonzero(_projection_mask(inp.shape, h)) //
                                 (2 * h['thickness'] - 1))
                v_distance = int(np.count_nonzero(_projection_mask(inp.shape, v)) //
                                 (2 * v['thickness'] - 1))
                short_short += h_distance <= 2 and v_distance <= 2
                long_long += h_distance >= 8 and v_distance >= 8
                layouts.add((inp.shape, h['thickness'], h['length'], h['direction'],
                             h['r0'], h['c0'], h['marker_c'], v['thickness'],
                             v['length'], v['direction'], v['r0'], v['c0'], v['marker_r']))
                if split == 'train' and (h['thickness'] == 6 or v['thickness'] == 6):
                    six_train += 1
                if split == 'test':
                    test_directions.add((h['direction'], v['direction']))
                    vertical_six_test += v['thickness'] == 6
                    if h['thickness'] == 6 or v['thickness'] == 6:
                        six_test += 1
        recolored += len(local_palettes) > 1
        mixed_lengths += len(puzzle_lengths) >= 3
        mixed_arrangements += len(puzzle_arrangements) >= 2
    _check(len(counts['train']) >= 4 and len(counts['test']) >= 3,
           'Automatic counts lack diversity')
    _check(len(shapes) >= 25 and len(layouts) >= 120 and len(inputs) >= 120,
           'Structural diversity too narrow')
    _check(len(test_directions) == 4 and len(arrangements) >= 5,
           'Directions or arrangements too narrow')
    _check(max(lengths) >= 18 and six_train >= 5 and six_test >= 5 and recolored >= 5,
           'Length, thickness, or palette diversity too narrow')
    _check(ten_square >= 3 and twenty_square >= 5 and large_square >= 20 and
           rectangle >= 20, 'Source canvas scales or aspect ratios are missing')
    _check(long_bars >= 30 and asymmetric_markers >= 30 and aligned >= 10,
           'Long bars, off-center markers, or aligned placements are missing')
    _check(short_short >= 5 and long_long >= 5 and vertical_six_test >= 2,
           'Projection distances or test thicknesses lack diversity')
    _check(mixed_lengths >= 30 and mixed_arrangements >= 30,
           'Within-puzzle geometry lacks diversity')
