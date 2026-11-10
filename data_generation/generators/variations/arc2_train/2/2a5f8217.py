import numpy as np


_SOURCE_PAIR_ROWS = {
    (
        (
            "000000000",
            "011100100",
            "010100110",
            "000000000",
            "000000099",
            "001100009",
            "000107000",
            "666007700",
            "606000000",
        ),
        (
            "000000000",
            "066600700",
            "060600770",
            "000000000",
            "000000099",
            "009900009",
            "000907000",
            "666007700",
            "606000000",
        ),
    ),
    (
        (
            "010000",
            "111000",
            "010000",
            "000080",
            "000888",
            "000080",
        ),
        (
            "080000",
            "888000",
            "080000",
            "000080",
            "000888",
            "000080",
        ),
    ),
    (
        (
            "01000100009",
            "11100100099",
            "00100000009",
            "00000000100",
            "03000601100",
            "33300600100",
            "00000000000",
            "00000000700",
            "00010007770",
            "00111000070",
            "00000000000",
        ),
        (
            "07000600009",
            "77700600099",
            "00700000009",
            "00000000900",
            "03000609900",
            "33300600900",
            "00000000000",
            "00000000700",
            "00030007770",
            "00333000070",
            "00000000000",
        ),
    ),
    (
        (
            "0000008800000",
            "0010088801100",
            "0111008011100",
            "0000000001000",
            "0000001000000",
            "0040011100000",
            "0440001000020",
            "0040000000222",
            "0000001000000",
            "0000011000060",
            "0010001000666",
            "8010000000060",
            "8000000000000",
        ),
        (
            "0000008800000",
            "0020088808800",
            "0222008088800",
            "0000000008000",
            "0000006000000",
            "0040066600000",
            "0440006000020",
            "0040000000222",
            "0000004000000",
            "0000044000060",
            "0080004000666",
            "8080000000060",
            "8000000000000",
        ),
    ),
}


def _check_count_value(
    name: str,
    value: int | None,
    lower: int,
    upper: int,
) -> None:
    if value is None:
        return

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    if not lower <= int(value) <= upper:
        raise ValueError(
            f"{name} must be between {lower} and {upper}"
        )


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    _check_count_value("num_train", num_train, 2, 10)
    _check_count_value("num_test", num_test, 1, 4)

    if num_train is None and num_test is None:
        test_count = int(rng.integers(1, 5))
        train_count = int(
            rng.integers(max(2, test_count + 1), 11)
        )
    elif num_train is None:
        test_count = int(num_test)
        train_count = int(
            rng.integers(max(2, test_count + 1), 11)
        )
    elif num_test is None:
        train_count = int(num_train)
        maximum_test = min(4, train_count - 1)
        test_count = int(rng.integers(1, maximum_test + 1))
    else:
        train_count = int(num_train)
        test_count = int(num_test)

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count



def _normalize_shape(cells):
    top = min(r for r, _ in cells)
    left = min(c for _, c in cells)
    return tuple(sorted((r - top, c - left) for r, c in cells))


def _transform_shape(shape, turn, reflect=False):
    cells = [(r, -c if reflect else c) for r, c in shape]
    for _ in range(turn):
        cells = [(c, -r) for r, c in cells]
    return _normalize_shape(cells)


def _shape_box(shape):
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _random_shape(rng, kind=None):
    if kind is None:
        kind = rng.choice(['bar', 'outline', 'growth'], p=[0.19, 0.17, 0.64])
    if kind == 'bar':
        size = 1 if rng.random() < 0.08 else int(rng.integers(2, 11))
        if rng.random() < 0.5:
            return tuple((0, i) for i in range(size))
        return tuple((i, 0) for i in range(size))
    if kind == 'outline':
        height = int(rng.integers(3, 7))
        width = int(rng.integers(3, 7))
        cells = {(r, c) for r in range(height) for c in range(width)
                 if r in (0, height - 1) or c in (0, width - 1)}
        if rng.random() < 0.42:
            # An open outline remains connected after a non-corner gap.
            side = int(rng.integers(4))
            if side == 0 and width > 3:
                cells.remove((0, int(rng.integers(1, width - 1))))
            elif side == 1 and width > 3:
                cells.remove((height - 1, int(rng.integers(1, width - 1))))
            elif side == 2 and height > 3:
                cells.remove((int(rng.integers(1, height - 1)), 0))
            elif side == 3 and height > 3:
                cells.remove((int(rng.integers(1, height - 1)), width - 1))
        return _normalize_shape(cells)
    side = int(rng.integers(3, 8))
    target = int(rng.integers(3, min(14, side * side) + 1))
    cells = {(int(rng.integers(side)), int(rng.integers(side)))}
    while len(cells) < target:
        boundary = set()
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < side and 0 <= nc < side and (nr, nc) not in cells:
                    boundary.add((nr, nc))
        cells.add(sorted(boundary)[int(rng.integers(len(boundary)))])
    return _normalize_shape(cells)


def _sample_shapes(rng, count, lesson):
    shapes = []
    seen = set()
    if lesson:
        for _ in range(300):
            first = _random_shape(rng, 'growth')
            h, w = _shape_box(first)
            relatives = sorted({
                _transform_shape(first, turn, flip)
                for turn in range(4) for flip in (False, True)
            } - {first})
            relatives = [s for s in relatives if _shape_box(s) == (h, w)]
            if relatives:
                second = relatives[int(rng.integers(len(relatives)))]
                shapes = [first, second]
                seen.update(shapes)
                break
        else:
            raise ValueError('could not construct an orientation lesson')
    kinds = ['bar', 'outline', 'growth']
    for _ in range(3000):
        if len(shapes) >= count:
            return shapes
        # Rich pairs mix thin, hollow/open, and organically grown masks.
        kind = kinds[(len(shapes) - 2) % 3] if count >= 5 and len(shapes) >= 2 else None
        shape = _random_shape(rng, kind)
        if shape not in seen:
            shapes.append(shape)
            seen.add(shape)
    raise ValueError('could not sample distinct masks')


def _sample_labels(rng, count, background, placeholder, lesson, mode=None):
    palette = [c for c in range(10) if c not in (background, placeholder)]
    if mode is None:
        mode = rng.choice(['distinct', 'one_pair', 'groups', 'dominant'],
                          p=[0.38, 0.24, 0.23, 0.15])
    if mode == 'distinct' or count == 1:
        labels = rng.choice(palette, count, replace=False).tolist()
    elif mode == 'one_pair':
        labels = rng.choice(palette, count - 1, replace=False).tolist()
        labels.insert(int(rng.integers(count)), labels[int(rng.integers(len(labels)))])
    elif mode == 'groups':
        groups = min(max(2, (count + 1) // 2), len(palette))
        colors = rng.choice(palette, groups, replace=False).tolist()
        labels = [colors[i % groups] for i in range(count)]
        rng.shuffle(labels)
    else:
        groups = 2 if lesson else int(rng.integers(1, min(3, count) + 1))
        colors = rng.choice(palette, groups, replace=False).tolist()
        labels = [colors[0]] * count
        for i in range(1, groups):
            labels[i] = colors[i]
        rng.shuffle(labels)
    if lesson and labels[0] == labels[1]:
        choices = [c for c in palette if c != labels[0]]
        labels[1] = int(rng.choice(choices))
    return [int(c) for c in labels]


def _can_place(occupied, shape, top, left):
    height, width = occupied.shape
    for dr, dc in shape:
        r, c = top + dr, left + dc
        if occupied[r, c]:
            return False
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < height and 0 <= nc < width and occupied[nr, nc]:
                return False
    return True


def _sample_dimensions(rng, specs, shapes):
    cells = sum(len(shapes[i]) for i, _ in specs)
    max_h = max(_shape_box(shapes[i])[0] for i, _ in specs)
    max_w = max(_shape_box(shapes[i])[1] for i, _ in specs)
    for _ in range(80):
        density = float(rng.uniform(2.5, 5.8))
        area = max(25, cells * density)
        aspect = float(np.exp(rng.uniform(-0.82, 0.82)))
        height = int(np.ceil(np.sqrt(area * aspect)))
        width = int(np.ceil(np.sqrt(area / aspect)))
        if rng.random() < 0.08:
            if rng.random() < 0.5:
                height = 30
            else:
                width = 30
        height = min(30, max(max_h, height))
        width = min(30, max(max_w, width))
        if height * width >= 2.4 * cells and height >= max_h and width >= max_w:
            return height, width
    return 30, 30


def _pack_objects(rng, shapes, labels, specs, height, width,
                  background, placeholder, attempts=8):
    for _ in range(attempts):
        occupied = np.zeros((height, width), dtype=bool)
        order = list(specs)
        rng.shuffle(order)
        order.sort(key=lambda item: -len(shapes[item[0]]))
        placements = []
        for index, is_placeholder in order:
            shape = shapes[index]
            h, w = _shape_box(shape)
            if h > height or w > width:
                break
            positions = [(r, c) for r in range(height - h + 1)
                         for c in range(width - w + 1)]
            rng.shuffle(positions)
            if rng.random() < 0.22:
                positions.sort(key=lambda pos: 0 if (
                    pos[0] == 0 or pos[1] == 0 or
                    pos[0] + h == height or pos[1] + w == width
                ) else 1)
            chosen = next(((r, c) for r, c in positions
                           if _can_place(occupied, shape, r, c)), None)
            if chosen is None:
                break
            top, left = chosen
            for dr, dc in shape:
                occupied[top + dr, left + dc] = True
            placements.append((index, is_placeholder, top, left))
        if len(placements) != len(specs):
            continue
        input_grid = np.full((height, width), background, dtype=np.int8)
        output_grid = input_grid.copy()
        for index, is_placeholder, top, left in placements:
            for dr, dc in shapes[index]:
                input_grid[top + dr, left + dc] = placeholder if is_placeholder else labels[index]
                output_grid[top + dr, left + dc] = labels[index]
        if np.count_nonzero(input_grid == background) <= height * width // 2:
            continue
        return input_grid, output_grid
    return None


def _generate_pair(rng, regime, background, placeholder, shape_count,
                   lesson=False, label_mode=None):
    shapes = _sample_shapes(rng, shape_count, lesson)
    simple_pair = rng.random() < 0.25
    if not simple_pair and shape_count <= 5 and rng.random() < 0.34:
        for _ in range(40):
            extra = _random_shape(rng)
            if extra not in shapes:
                shapes.append(extra)
                break
    labels = _sample_labels(rng, len(shapes), background, placeholder,
                            lesson, label_mode)
    specs = []
    for index in range(shape_count):
        exemplar_copies = 2 if not simple_pair and rng.random() < 0.25 else 1
        placeholder_copies = (int(rng.choice([1, 2, 3], p=[0.57, 0.35, 0.08]))
                              if not simple_pair else 1)
        specs.extend([(index, False)] * exemplar_copies)
        specs.extend([(index, True)] * placeholder_copies)
    for index in range(shape_count, len(shapes)):
        specs.extend([(index, False)] * (2 if rng.random() < 0.25 else 1))
    for _ in range(20):
        height, width = _sample_dimensions(rng, specs, shapes)
        packed = _pack_objects(rng, shapes, labels, specs, height, width,
                               background, placeholder)
        if packed is not None:
            return packed
    packed = _pack_objects(rng, shapes, labels, specs, 30, 30,
                           background, placeholder, attempts=80)
    if packed is None:
        raise ValueError('unable to place the objects')
    return packed


def _array_signature(grid):
    return grid.shape, tuple(int(v) for v in grid.ravel())


def _rows_signature(grid):
    return tuple(''.join(str(int(v)) for v in row) for row in grid)


def _components_of_color(grid, color):
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    components = []
    for start_r in range(height):
        for start_c in range(width):
            if seen[start_r, start_c] or int(grid[start_r, start_c]) != color:
                continue
            stack = [(start_r, start_c)]
            seen[start_r, start_c] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (0 <= nr < height and 0 <= nc < width and
                            not seen[nr, nc] and int(grid[nr, nc]) == color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(tuple(cells))
    return components


def _mode_color(grid):
    values, counts = np.unique(grid, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1
    return int(values[np.argmax(counts)])


def _analyze_pair(input_grid, output_grid):
    assert input_grid.shape == output_grid.shape
    background = _mode_color(input_grid)
    assert _mode_color(output_grid) == background
    changed = input_grid != output_grid
    assert np.any(changed)
    colors = np.unique(input_grid[changed])
    assert len(colors) == 1
    placeholder = int(colors[0])
    assert placeholder != background
    assert np.array_equal(changed, input_grid == placeholder)
    assert not np.any(output_grid == placeholder)
    # Different objects may not touch orthogonally, even when their colors differ.
    a, b = input_grid[:, :-1], input_grid[:, 1:]
    assert not np.any((a != background) & (b != background) & (a != b))
    a, b = input_grid[:-1, :], input_grid[1:, :]
    assert not np.any((a != background) & (b != background) & (a != b))
    exemplar_colors = {}
    exemplar_counts = {}
    labeled_components = []
    for color in np.unique(input_grid):
        color = int(color)
        if color in (background, placeholder):
            continue
        for cells in _components_of_color(input_grid, color):
            mask = _normalize_shape(cells)
            assert mask not in exemplar_colors or exemplar_colors[mask] == color
            exemplar_colors[mask] = color
            exemplar_counts[mask] = exemplar_counts.get(mask, 0) + 1
            labeled_components.append((mask, color))
    placeholders = _components_of_color(input_grid, placeholder)
    assert placeholders
    placeholder_counts = {}
    for cells in placeholders:
        mask = _normalize_shape(cells)
        assert mask in exemplar_colors
        placeholder_counts[mask] = placeholder_counts.get(mask, 0) + 1
        assert all(int(output_grid[r, c]) == exemplar_colors[mask] for r, c in cells)
    assert set(placeholder_counts).issubset(exemplar_colors)
    active = list(placeholder_counts)
    conflicts = []
    for i, first in enumerate(active):
        for second in active[i + 1:]:
            if (len(first) == len(second) and _shape_box(first) == _shape_box(second)
                    and exemplar_colors[first] != exemplar_colors[second]):
                conflicts.append((first, second))
    orientation_conflict = any(
        second in {_transform_shape(first, turn, flip)
                   for turn in range(4) for flip in (False, True)}
        for first, second in conflicts
    )
    return {
        'background': background,
        'placeholder': placeholder,
        'shape_count': len(active),
        'placeholder_count': len(placeholders),
        'exemplar_count': sum(exemplar_counts.values()),
        'inactive_count': len(set(exemplar_colors) - set(active)),
        'duplicate_placeholder': any(n > 1 for n in placeholder_counts.values()),
        'duplicate_exemplar': any(n > 1 for n in exemplar_counts.values()),
        'target_colors': tuple(exemplar_colors[s] for s in active),
        'shape_masks': tuple(active),
        'thin': any(min(_shape_box(s)) == 1 for s in active),
        'large': any(len(s) >= 10 for s in active),
        'full_mask_conflict': bool(conflicts),
        'orientation_conflict': orientation_conflict,
    }


def _solve_from_training(train_pairs, test_input):
    placeholder_colors = set()
    for pair in train_pairs:
        changed = pair['input'] != pair['output']
        placeholder_colors.update(int(x) for x in np.unique(pair['input'][changed]))
    assert len(placeholder_colors) == 1
    placeholder = next(iter(placeholder_colors))
    background = _mode_color(test_input)
    exemplars = {}
    for color in np.unique(test_input):
        color = int(color)
        if color in (background, placeholder):
            continue
        for cells in _components_of_color(test_input, color):
            mask = _normalize_shape(cells)
            assert mask not in exemplars or exemplars[mask] == color
            exemplars[mask] = color
    prediction = test_input.copy()
    for cells in _components_of_color(test_input, placeholder):
        mask = _normalize_shape(cells)
        assert mask in exemplars
        for r, c in cells:
            prediction[r, c] = exemplars[mask]
    return prediction


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    placeholder = int(rng.integers(10))
    backgrounds = [c for c in range(10) if c != placeholder]
    first_two = rng.choice(backgrounds, 2, replace=False).tolist()
    shift = rng.choice(['expanded', 'same', 'simpler'], p=[0.3, 0.4, 0.3])
    if shift == 'simpler':
        train_shape_counts = [int(rng.integers(3, 6))]
        train_shape_counts.extend(int(rng.integers(3, 7)) for _ in range(train_count - 1))
    else:
        train_shape_counts = [int(rng.integers(2, 6))]
        train_shape_counts.extend(int(rng.integers(1, 7)) for _ in range(train_count - 1))
    maximum = max(train_shape_counts)
    minimum = min(train_shape_counts)
    test_shape_counts = []
    for _ in range(test_count):
        if shift == 'expanded':
            test_shape_counts.append(int(rng.integers(min(7, maximum + 1), 8)))
        elif shift == 'simpler':
            test_shape_counts.append(int(rng.integers(1, minimum)))
        else:
            test_shape_counts.append(int(rng.choice(train_shape_counts)))
    # Independent deterministic random streams preserve count-request behavior.
    children = np.random.SeedSequence(rng.integers(0, 2**32, 4,
                             dtype=np.uint32).tolist()).spawn(train_count + test_count)
    puzzle = {'train': [], 'test': []}
    seen = set()
    for index in range(train_count + test_count):
        split = 'train' if index < train_count else 'test'
        local = index if split == 'train' else index - train_count
        pair_rng = np.random.default_rng(children[index])
        background = int(first_two[index]) if index < 2 else int(pair_rng.choice(backgrounds))
        count = train_shape_counts[local] if split == 'train' else test_shape_counts[local]
        lesson = index == 0 or (count >= 2 and pair_rng.random() <
                                 (0.45 if split == 'test' else 0.30))
        label_mode = 'distinct' if shift == 'expanded' and split == 'train' else None
        if shift == 'expanded' and split == 'test':
            label_mode = pair_rng.choice(['one_pair', 'groups', 'dominant'])
        for _ in range(24):
            input_grid, output_grid = _generate_pair(pair_rng, split,
                background, placeholder, count, lesson, label_mode)
            signature = _array_signature(input_grid)
            if signature not in seen:
                seen.add(signature)
                puzzle[split].append({'input': input_grid, 'output': output_grid})
                break
        else:
            raise ValueError('could not create a unique pair')
    validate_generated_puzzle(puzzle)
    return puzzle


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
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            inp = _array_signature(pair['input'])
            pair_sig = (inp, _array_signature(pair['output']))
            assert inp not in seen_inputs and pair_sig not in seen_pairs
            seen_inputs.add(inp)
            seen_pairs.add(pair_sig)
            assert (_rows_signature(pair['input']), _rows_signature(pair['output'])) not in _SOURCE_PAIR_ROWS
            analyses[split].append(_analyze_pair(pair['input'], pair['output']))
    placeholders = {a['placeholder'] for group in analyses.values() for a in group}
    assert len(placeholders) == 1
    assert len({a['background'] for a in analyses['train']}) >= 2
    assert all(a['background'] != a['placeholder'] for group in analyses.values() for a in group)
    assert any(a['full_mask_conflict'] and a['orientation_conflict']
               for a in analyses['train'])
    for pair in puzzle['test']:
        assert np.array_equal(_solve_from_training(puzzle['train'], pair['input']),
                              pair['output'])


def _puzzles_equal(first, second):
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s])
        for k in ('input', 'output')) for s in ('train', 'test'))


def self_test() -> None:
    explicit = generate_puzzle(1729, 7, 3)
    assert _puzzles_equal(explicit, generate_puzzle(1729, 7, 3))
    assert not _puzzles_equal(explicit, generate_puzzle(1730, 7, 3))
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1),
                        (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(2000 + (train or 0) * 10 + (test or 0), train, test)
        if train is not None:
            assert len(puzzle['train']) == train
        if test is not None:
            assert len(puzzle['test']) == test
        validate_generated_puzzle(puzzle)
    for args in ({'num_train': 1}, {'num_train': 11}, {'num_test': 0},
                 {'num_test': 5}, {'num_train': 2, 'num_test': 2},
                 {'num_train': 3, 'num_test': 4}, {'num_train': 2.0}):
        try:
            generate_puzzle(4000, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f'illegal counts accepted: {args}')
    train_counts, test_counts, roles, dimensions = set(), set(), set(), set()
    shapes, layouts, inputs = set(), set(), set()
    feature_totals = {k: 0 for k in ('thin', 'large', 'full_mask_conflict',
        'orientation_conflict', 'duplicate_placeholder', 'duplicate_exemplar')}
    inactive_total = 0
    repeated_label_total = 0
    distinct_label_total = 0
    total_pairs = 0
    train_shape_counts, test_shape_counts = set(), set()
    for seed in range(5000, 5042):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        roles.add(_analyze_pair(puzzle['train'][0]['input'],
                                puzzle['train'][0]['output'])['placeholder'])
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                total_pairs += 1
                a = _analyze_pair(pair['input'], pair['output'])
                dimensions.add(pair['input'].shape)
                binary = pair['input'] != a['background']
                layouts.add((binary.shape, tuple(bool(x) for x in binary.ravel())))
                inputs.add(_array_signature(pair['input']))
                shapes.update(a['shape_masks'])
                (train_shape_counts if split == 'train' else test_shape_counts).add(a['shape_count'])
                for name in feature_totals:
                    feature_totals[name] += int(a[name])
                inactive_total += int(a['inactive_count'] > 0)
                repeated_label_total += int(len(set(a['target_colors'])) < a['shape_count'])
                distinct_label_total += int(len(set(a['target_colors'])) == a['shape_count'])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(roles) >= 7 and len(dimensions) >= 30
    assert len(shapes) >= 60 and len(layouts) >= int(0.9 * total_pairs)
    assert len(inputs) == total_pairs
    assert len(train_shape_counts) >= 5 and len(test_shape_counts) >= 5
    assert all(value > 0 for value in feature_totals.values())
    assert inactive_total > 0 and repeated_label_total > 0 and distinct_label_total > 0
