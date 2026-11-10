import numpy as np


_ZERO = "0000000000"
_SOURCE_INPUT_ROWS = {
    (
        _ZERO, _ZERO, "0044444000", "0000400000", "0000400000",
        "0000400000", "0044444000", _ZERO, _ZERO, _ZERO,
    ),
    (
        _ZERO, _ZERO, "0400000400", "0444444400", "0400000400",
        _ZERO, _ZERO, _ZERO, _ZERO, _ZERO,
    ),
    (
        _ZERO, _ZERO, "4444444400", "4040000000", "0404000000",
        "0040400000", "0004040000", "0000404000", "0044444440",
        _ZERO,
    ),
    (
        _ZERO, "0400000040", "0400000040", "0444444440",
        "0400000040", "0444444440", "0400000040", "0400000040",
        _ZERO, _ZERO,
    ),
    (
        _ZERO, _ZERO, "0444440000", "0044400000", "0044400000",
        "0044400000", "0044400000", "0044400000", "0044400000",
        "0444440000",
    ),
    (
        _ZERO, "4040404040", "4404040440", "4000000040",
        "4040404040", "4404040440", "4000000040", _ZERO,
        _ZERO, _ZERO,
    ),
}


def _pair_counts(
    rng: np.random.Generator,
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    if num_train is not None:
        if (
            isinstance(num_train, bool)
            or not isinstance(num_train, (int, np.integer))
            or not 2 <= num_train <= 10
        ):
            raise ValueError("num_train must be an integer from 2 to 10")

    if num_test is not None:
        if (
            isinstance(num_test, bool)
            or not isinstance(num_test, (int, np.integer))
            or not 1 <= num_test <= 4
        ):
            raise ValueError("num_test must be an integer from 1 to 4")

    if num_train is None and num_test is None:
        num_test = 1 + int(seed) % 4
        num_train = num_test + 1 + (int(seed) // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))

    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")

    return int(num_train), int(num_test)


def _continuous(indices):
    return len(indices) >= 3 and int(indices[-1] - indices[0] + 1) == len(indices)


def _components(mask, diagonal=True):
    cells = set(zip(*np.where(mask)))
    steps = ([(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
             if diagonal else [(-1, 0), (1, 0), (0, -1), (0, 1)])
    count = 0
    while cells:
        count += 1
        pending = [cells.pop()]
        while pending:
            row, col = pending.pop()
            for dr, dc in steps:
                neighbor = (row + dr, col + dc)
                if neighbor in cells:
                    cells.remove(neighbor)
                    pending.append(neighbor)
    return count


def _outer_orientation(mask):
    rows, cols = np.where(mask)
    assert len(rows)
    top, bottom = int(rows.min()), int(rows.max())
    left, right = int(cols.min()), int(cols.max())
    assert bottom - top >= 2 and right - left >= 2
    t, b = np.flatnonzero(mask[top]), np.flatnonzero(mask[bottom])
    l, r = np.flatnonzero(mask[:, left]), np.flatnonzero(mask[:, right])
    horizontal = (_continuous(t) and _continuous(b)
                  and min(int(t[-1]), int(b[-1])) - max(int(t[0]), int(b[0])) >= 1)
    vertical = (_continuous(l) and _continuous(r)
                and min(int(l[-1]), int(r[-1])) - max(int(l[0]), int(r[0])) >= 1)
    assert horizontal != vertical, 'ambiguous outer bars'
    return 'H' if horizontal else 'V'


def _source_copy(grid):
    return grid.shape == (10, 10) and tuple(
        ''.join(str(int(v)) for v in row) for row in grid
    ) in _SOURCE_INPUT_ROWS


def _walk(rng, separation, length, start, end, alternate):
    path = [start]
    for row in range(1, separation - 1):
        old = path[-1]
        remaining = separation - 1 - row
        if abs(end - old) >= remaining:
            step = int(np.sign(end - old))
        elif alternate:
            step = 1 if row % 2 else -1
        else:
            step = int(rng.choice((-1, 0, 1), p=(0.39, 0.22, 0.39)))
        path.append(max(0, min(length - 1, old + step)))
    path.append(end)
    if abs(path[-1] - path[-2]) > 1:
        return None
    return path


def _local_figure(rng, length, separation, style):
    trim = min(4, length - 3)
    for _ in range(40):
        starts = [int(rng.integers(0, trim + 1)) for _ in range(2)]
        ends = [length - 1 - int(rng.integers(0, trim + 1)) for _ in range(2)]
        starts[int(rng.integers(0, 2))] = 0
        ends[int(rng.integers(0, 2))] = length - 1
        if (min(ends) - max(starts) >= 1
                and all(ends[i] - starts[i] >= 2 for i in range(2))):
            break
    else:
        starts, ends = [0, 0], [length - 1, length - 1]
    mask = np.zeros((separation, length), dtype=bool)
    mask[0, starts[0]:ends[0] + 1] = True
    mask[-1, starts[1]:ends[1] + 1] = True
    common = list(range(max(starts), min(ends) + 1))
    if style == 'straight':
        count = int(rng.integers(1, min(4, len(common)) + 1))
        for col in rng.choice(common, size=count, replace=False):
            mask[1:-1, int(col)] = True
    elif style == 'thick':
        thickness = int(rng.integers(2, min(4, len(common)) + 1))
        first = int(rng.integers(common[0], common[-1] - thickness + 2))
        mask[1:-1, first:first + thickness] = True
    else:
        count = (int(rng.integers(1, min(3, len(common)) + 1))
                 if style in ('zigzag', 'alternating', 'lattice') else 1)
        for start in rng.choice(common, size=count, replace=False):
            start = int(start)
            end = int(rng.choice([v for v in common if abs(v - start) <= separation - 1]))
            if style == 'lattice' and rng.random() < 0.4:
                path = [start] * separation
            else:
                path = _walk(rng, separation, length, start, end,
                             style == 'alternating')
            if path is None:
                continue
            for row in range(1, separation - 1):
                mask[row, path[row]] = True
                if style == 'branched' and rng.random() < 0.75:
                    direction = -1 if rng.random() < 0.5 else 1
                    reach = int(rng.integers(1, min(4, length - 1)))
                    for delta in range(1, reach + 1):
                        col = path[row] + direction * delta
                        if 0 < col < length - 1:
                            mask[row, col] = True
        if style == 'lattice' and separation > 3 and rng.random() < 0.5:
            row = int(rng.integers(1, separation - 1))
            occupied = np.flatnonzero(mask[row])
            if len(occupied) > 1:
                mask[row, int(occupied.min()):int(occupied.max()) + 1] = True
    if separation >= 4 and rng.random() < 0.13:
        row = int(rng.integers(1, separation - 1))
        mask[row, :] = True
    return mask


def _dimensions(rng, fixed):
    if fixed or rng.random() < 0.18:
        return 10, 10
    return int(rng.integers(7, 17)), int(rng.integers(7, 17))


def _make_input(rng, height, width, background, figure, orientation, regime):
    styles = ('straight', 'thick', 'zigzag', 'alternating', 'branched', 'lattice')
    weights = ((0.25, 0.14, 0.23, 0.12, 0.15, 0.11)
               if regime == 'source_train' else
               (0.13, 0.18, 0.22, 0.18, 0.14, 0.15))
    for _ in range(500):
        if orientation == 'H':
            max_length, max_separation = width - 1, height - 1
        else:
            max_length, max_separation = height - 1, width - 1
        length = int(rng.integers(3, min(max_length, 13) + 1))
        separation = int(rng.integers(3, min(max_separation, 11) + 1))
        style = str(rng.choice(styles, p=weights))
        local = _local_figure(rng, length, separation, style)
        shape = local if orientation == 'H' else local.T
        if _components(shape) != 1:
            continue
        try:
            if _outer_orientation(shape) != orientation:
                continue
        except AssertionError:
            continue
        y = int(rng.integers(1, height - shape.shape[0] + 1))
        x = int(rng.integers(0, width - shape.shape[1]))
        grid = np.full((height, width), background, dtype=np.int8)
        patch = grid[y:y + shape.shape[0], x:x + shape.shape[1]]
        patch[shape] = figure
        if not _source_copy(grid):
            return grid
    raise RuntimeError('Could not place a unique pair of outer bars')


def _roles(rng, accent, base, variable, index, section):
    if not variable:
        return base
    if section == 'train' and index == 0:
        return base
    if section == 'train' and index == 1:
        return base[1], base[0]
    if rng.random() < 0.5:
        return base if rng.random() < 0.5 else (base[1], base[0])
    choices = [v for v in range(10) if v != accent]
    selected = rng.choice(choices, size=2, replace=False)
    return int(selected[0]), int(selected[1])


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _pair_counts(rng, seed, num_train, num_test)
    colors = [int(v) for v in rng.choice(10, size=3, replace=False)]
    base, accent = (colors[0], colors[1]), colors[2]
    source_like = bool(rng.random() < 0.27)
    variable_palette = not source_like and bool(rng.random() < 0.75)
    variable_size = not source_like and bool(rng.random() < 0.83)
    train_axes = ['H', 'V'] + [str(rng.choice(('H', 'V')))
                                 for _ in range(train_count - 2)]
    test_axes = [str(rng.choice(('H', 'V'))) for _ in range(test_count)]
    rng.shuffle(train_axes)
    puzzle = {'train': [], 'test': []}
    used = set()
    for section, axes in (('train', train_axes), ('test', test_axes)):
        regime = 'source_train' if section == 'train' else 'target_test'
        for index, axis in enumerate(axes):
            background, figure = _roles(rng, accent, base, variable_palette,
                                        index, section)
            for _ in range(100):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                h, w = _dimensions(pair_rng, not variable_size)
                if variable_size and section == 'train' and index == 1:
                    if (h, w) == puzzle['train'][0]['input'].shape:
                        h = 7 if h != 7 else 16
                input_grid = _make_input(pair_rng, h, w, background, figure,
                                         axis, regime)
                key = (input_grid.shape, input_grid.tobytes())
                if key not in used:
                    break
            else:
                raise RuntimeError('Could not generate distinct inputs')
            used.add(key)
            output_grid = input_grid.copy()
            if axis == 'H':
                output_grid[0, :] = accent
            else:
                output_grid[:, -1] = accent
            puzzle[section].append({'input': input_grid, 'output': output_grid})
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
    inp, out = pair['input'], pair['output']
    for grid in (inp, out):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
        assert np.issubdtype(grid.dtype, np.integer)
        assert np.all((grid >= 0) & (grid <= 9))
    assert inp.shape == out.shape and min(inp.shape) >= 7
    background = int(inp[0, 0])
    assert np.all(inp[0] == background) and np.all(inp[:, -1] == background)
    colors = set(int(v) for v in np.unique(inp))
    assert len(colors) == 2 and background in colors
    figure = next(iter(colors - {background}))
    mask = inp == figure
    assert _components(mask) == 1
    axis = _outer_orientation(mask)
    assert not _source_copy(inp)
    new = set(int(v) for v in np.unique(out)) - colors
    assert len(new) == 1
    accent = next(iter(new))
    expected = inp.copy()
    if axis == 'H':
        expected[0, :] = accent
    else:
        expected[:, -1] = accent
    assert np.array_equal(out, expected)
    return axis, background, figure, accent


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[k], list) for k in ('train', 'test'))
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    inputs, pairs = set(), set()
    train_axes, train_roles, train_shapes = set(), set(), set()
    test_roles, test_shapes, accents = set(), set(), set()
    for section in ('train', 'test'):
        for pair in puzzle[section]:
            axis, bg, fig, accent = _inspect_pair(pair)
            accents.add(accent)
            inp, out = pair['input'], pair['output']
            ikey = (inp.shape, inp.tobytes())
            pkey = (ikey, out.tobytes())
            assert ikey not in inputs and pkey not in pairs
            inputs.add(ikey)
            pairs.add(pkey)
            if section == 'train':
                train_axes.add(axis)
                train_roles.add((bg, fig))
                train_shapes.add(inp.shape)
            else:
                test_roles.add((bg, fig))
                test_shapes.add(inp.shape)
    assert len(accents) == 1 and train_axes == {'H', 'V'}
    if len(train_roles) == 1:
        assert test_roles == train_roles
    if len(train_shapes) == 1:
        assert test_shapes == train_shapes


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k])
        for x, y in zip(a[s], b[s]) for k in ('input', 'output'))
        for s in ('train', 'test'))


def _prediction(train, test_input):
    # Derive the output color solely from demonstrations and orient the test input.
    accents = set()
    for pair in train:
        changed = pair['input'] != pair['output']
        assert changed.any()
        accents.update(int(v) for v in np.unique(pair['output'][changed]))
    assert len(accents) == 1
    bg = int(test_input[0, 0])
    figure = next(int(v) for v in np.unique(test_input) if int(v) != bg)
    axis = _outer_orientation(test_input == figure)
    predicted = test_input.copy()
    if axis == 'H':
        predicted[0] = next(iter(accents))
    else:
        predicted[:, -1] = next(iter(accents))
    return predicted


def self_test() -> None:
    train_counts, test_counts, sizes, layouts, palettes = set(), set(), set(), set(), set()
    fingerprints, patterns, connector_counts, components = set(), set(), set(), set()
    varied_sizes = varied_palettes = swaps = short_bars = zero_connectors = thick = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        axes, roles, shapes, signature = {}, {}, {}, []
        for section in ('train', 'test'):
            axes[section], roles[section], shapes[section] = [], [], []
            for pair in puzzle[section]:
                inp, out = pair['input'], pair['output']
                axis, bg, fig, accent = _inspect_pair(pair)
                assert np.array_equal(_prediction(puzzle['train'], inp), out)
                axes[section].append(axis)
                roles[section].append((bg, fig))
                shapes[section].append(inp.shape)
                palettes.add((bg, fig, accent))
                sizes.add(inp.shape)
                signature.append((inp.shape, inp.tobytes(), out.tobytes()))
                mask = inp == fig
                rr, cc = np.where(mask)
                top, bottom, left, right = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())
                cropped = mask[top:bottom + 1, left:right + 1]
                layouts.add((cropped.shape, cropped.tobytes()))
                components.add(_components(mask, False))
                if axis == 'H':
                    bars = (int(mask[top].sum()), int(mask[bottom].sum()))
                    connectors = sum(bool(np.all(mask[top:bottom + 1, col]))
                                     for col in range(left + 1, right))
                    thick += int(any(np.count_nonzero(mask[row, left + 1:right]) >= 3
                                     for row in range(top + 1, bottom)))
                else:
                    bars = (int(mask[:, left].sum()), int(mask[:, right].sum()))
                    connectors = sum(bool(np.all(mask[row, left:right + 1]))
                                     for row in range(top + 1, bottom))
                    thick += int(any(np.count_nonzero(mask[top + 1:bottom, col]) >= 3
                                     for col in range(left + 1, right)))
                short_bars += int(min(bars) == 3)
                zero_connectors += int(connectors == 0)
                connector_counts.add(int(connectors))
        patterns.add((tuple(axes['train']), tuple(axes['test'])))
        varied_sizes += int(len(set(shapes['train'])) > 1)
        varied_palettes += int(len(set(roles['train'])) > 1)
        all_roles = set(roles['train'] + roles['test'])
        swaps += int(any((b, a) in all_roles for a, b in all_roles))
        fingerprints.add(tuple(signature))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(sizes) >= 30 and len(layouts) >= 120 and len(palettes) >= 30
    assert len(fingerprints) == 42 and len(patterns) >= 20
    assert varied_sizes >= 12 and varied_palettes >= 12 and swaps >= 12
    assert short_bars >= 5 and zero_connectors >= 20 and thick >= 20
    assert max(components) > 1 and max(connector_counts) >= 2
    for seed, (nt, ns) in enumerate(((2, 1), (10, 1), (10, 4),
                                     (None, 1), (None, 4), (2, None),
                                     (10, None)), 100):
        puzzle = generate_puzzle(seed, nt, ns)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed, nt, ns))
        if nt is not None:
            assert len(puzzle['train']) == nt
        if ns is not None:
            assert len(puzzle['test']) == ns
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (2, 3), (3, 3), (4, 4)):
        try:
            generate_puzzle(200, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Accepted illegal counts: {nt}, {ns}')
