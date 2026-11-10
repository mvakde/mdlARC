import numpy as np


# The original four solved pairs are retained solely to exclude exact copies.
_SOURCE_PAIRS = (
    (
        (
            "030000",
            "030200",
            "000200",
            "080022",
            "000022",
            "666000",
        ),
        (
            "000030",
            "000032",
            "000002",
            "000822",
            "000022",
            "000666",
        ),
    ),
    (
        (
            "0000000000000",
            "0022000000000",
            "0002000550000",
            "0000000550000",
            "0000330000000",
            "0000300000000",
            "0000000000000",
            "0660008000000",
            "0000000000000",
            "0000000000000",
        ),
        (
            "0000000000000",
            "0000000002200",
            "0000000000255",
            "0000000000055",
            "0000000000033",
            "0000000000030",
            "0000000000000",
            "0000000000668",
            "0000000000000",
            "0000000000000",
        ),
    ),
    (
        (
            "022000",
            "602000",
            "600000",
            "008030",
            "000030",
            "800000",
        ),
        (
            "000022",
            "000062",
            "000060",
            "000083",
            "000003",
            "000008",
        ),
    ),
    (
        (
            "0008020000",
            "0000022000",
            "2033300000",
            "2000404000",
            "0000444000",
            "0008000000",
            "0000006000",
            "0800006000",
            "0005500000",
            "0000080000",
        ),
        (
            "0000000820",
            "0000000022",
            "0000002333",
            "0000002404",
            "0000000444",
            "0000000008",
            "0000000006",
            "0000000086",
            "0000000055",
            "0000000008",
        ),
    ),
)



def _dominant_color(grid):
    counts = np.bincount(grid.astype(np.int64).ravel(), minlength=10)
    winners = np.flatnonzero(counts == counts.max())
    assert len(winners) == 1
    color = int(winners[0])
    assert int(counts[color]) > grid.size // 2
    return color


def _components(grid, background):
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    answer = []
    for row in range(height):
        for col in range(width):
            if seen[row, col] or int(grid[row, col]) == background:
                continue
            color = int(grid[row, col])
            stack = [(row, col)]
            seen[row, col] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c),
                               (r, c - 1), (r, c + 1)):
                    if (0 <= nr < height and 0 <= nc < width
                            and not seen[nr, nc]
                            and int(grid[nr, nc]) == color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            answer.append((color, tuple(sorted(cells))))
    return answer


def _bounds(cells):
    rows, cols = zip(*cells)
    return min(rows), max(rows), min(cols), max(cols)


def _spans(cells):
    spans = {}
    for row, col in cells:
        if row in spans:
            lo, hi = spans[row]
            spans[row] = (min(lo, col), max(hi, col))
        else:
            spans[row] = (col, col)
    return spans


def _settlement_order(components):
    """Order pieces by visible row-wise precedence, allowing column overlap."""
    spans = [_spans(cells) for _, cells in components]
    to_right = [set() for _ in components]
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            common = spans[i].keys() & spans[j].keys()
            if not common:
                continue
            i_left = all(spans[i][r][1] < spans[j][r][0] for r in common)
            j_left = all(spans[j][r][1] < spans[i][r][0] for r in common)
            assert i_left != j_left, 'interleaved pieces have ambiguous motion'
            to_right[i if i_left else j].add(j if i_left else i)
    pending = set(range(len(components)))
    order = []
    while pending:
        ready = [i for i in pending if not (to_right[i] & pending)]
        assert ready, 'cyclic collision order'
        index = max(ready, key=lambda i: (_bounds(components[i][1])[3],
                                           _bounds(components[i][1])[2], -i))
        order.append(index)
        pending.remove(index)
    return order


def _settle(grid, background):
    height, width = grid.shape
    components = _components(grid, background)
    order = _settlement_order(components)
    output = np.full_like(grid, background)
    occupied = set()
    shifts = [0] * len(components)
    for index in order:
        color, cells = components[index]
        shift = 0
        while all(col + shift + 1 < width and
                  (row, col + shift + 1) not in occupied
                  for row, col in cells):
            shift += 1
        shifts[index] = shift
        for row, col in cells:
            dest = (row, col + shift)
            assert dest not in occupied
            occupied.add(dest)
            output[dest] = color
    return output, components, shifts


def _rowwise(grid, background):
    output = np.full_like(grid, background)
    for row in range(grid.shape[0]):
        values = grid[row][grid[row] != background]
        if len(values):
            output[row, -len(values):] = values
    return output


def _signature(grid):
    return (grid.shape, grid.tobytes())


def _source_copy(input_grid, output_grid):
    actual_in = tuple(''.join(str(int(v)) for v in row) for row in input_grid)
    actual_out = tuple(''.join(str(int(v)) for v in row) for row in output_grid)
    return any(actual_in == old_in and actual_out == old_out
               for old_in, old_out in _SOURCE_PAIRS)


def _pair_features(input_grid, background):
    output, components, shifts = _settle(input_grid, background)
    count = len(components)
    bounds = [_bounds(cells) for _, cells in components]
    colors = {color for color, _ in components}
    foreground = sum(len(cells) for _, cells in components)
    overlap_columns = 0
    reversed_order = 0
    same_color_singletons = False
    for i in range(count):
        for j in range(i + 1, count):
            ai, aj = bounds[i], bounds[j]
            if max(ai[2], aj[2]) <= min(ai[3], aj[3]) and (ai[1] < aj[0] or aj[1] < ai[0]):
                overlap_columns += 1
            if (ai[0] < aj[0] and ai[2] > aj[2]) or (aj[0] < ai[0] and aj[2] > ai[2]):
                reversed_order += 1
            if (components[i][0] == components[j][0]
                    and len(components[i][1]) == len(components[j][1]) == 1):
                same_color_singletons = True
    adjacent_unlike = any(
        int(input_grid[r, c]) != background
        and int(input_grid[r, c + 1]) != background
        and input_grid[r, c] != input_grid[r, c + 1]
        for r in range(input_grid.shape[0])
        for c in range(input_grid.shape[1] - 1)
    )
    return {
        'objects': count,
        'colors': len(colors),
        'foreground': foreground,
        'density': foreground / input_grid.size,
        'multirow': sum(b[0] < b[1] for b in bounds),
        'horizontal': sum(b[2] < b[3] for b in bounds),
        'blocked': sum(shift < input_grid.shape[1] - 1 - b[3]
                       for shift, b in zip(shifts, bounds)),
        'moved_multirow': sum(shift > 0 and b[0] < b[1]
                              for shift, b in zip(shifts, bounds)),
        'zero_moves': sum(shift == 0 for shift in shifts),
        'one_moves': sum(shift == 1 for shift in shifts),
        'long_moves': sum(shift >= 3 for shift in shifts),
        'overlap_columns': overlap_columns,
        'reversed_order': reversed_order,
        'adjacent_unlike': adjacent_unlike,
        'same_color_singletons': same_color_singletons,
        'rowwise_differs': not np.array_equal(output, _rowwise(input_grid, background)),
        'shifts': tuple(shifts),
    }


def _validate_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer or None')
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return value


def _resolve_counts(rng, num_train, num_test):
    train = _validate_count(num_train, 'num_train', 2, 10)
    test = _validate_count(num_test, 'num_test', 1, 4)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError('num_train must exceed num_test')
    return train, test


def _shape_array(rng, kind, compact=False):
    if kind == 0:
        array = np.ones((1, 1), dtype=bool)
    elif kind == 1:
        array = np.ones((1, int(rng.integers(2, 4 if compact else 6))), dtype=bool)
    elif kind == 2:
        array = np.ones((2, int(rng.integers(2, 3 if compact else 4))), dtype=bool)
    elif kind == 3:
        length = int(rng.integers(2, 4 if compact else 5))
        array = np.zeros((2, length), dtype=bool)
        array[0, :] = True
        array[1, 0] = True
    else:
        patterns = (
            ('#.#', '###'),
            ('###', '.#.'),
            ('##.', '.##'),
            ('.#.', '###', '.#.'),
            ('###', '#..', '#..'),
            ('##..', '.###'),
        )
        lines = patterns[int(rng.integers(len(patterns)))]
        array = np.array([[c == '#' for c in line] for line in lines], dtype=bool)
    array = np.rot90(array, int(rng.integers(4)))
    if rng.integers(2):
        array = np.fliplr(array)
    return tuple((int(r), int(c)) for r, c in np.argwhere(array))


def _dimensions(rng, demonstration):
    if demonstration:
        return int(rng.integers(8, 13)), int(rng.integers(9, 17))
    mode = int(rng.choice(3, p=(0.43, 0.39, 0.18)))
    if mode == 0:
        if rng.random() < 0.12:
            return 6, 6
        return int(rng.integers(6, 11)), int(rng.integers(6, 12))
    if mode == 1:
        return int(rng.integers(8, 17)), int(rng.integers(9, 20))
    return int(rng.integers(11, 23)), int(rng.integers(18, 31))


def _object_count(rng, height, width, regime, demonstration):
    cap = min(10, max(3, height * width // 8))
    if demonstration:
        return int(rng.integers(5, min(cap, 8) + 1))
    lower = 3
    if regime == 'target_test' and rng.random() < 0.40:
        lower = min(cap, max(5, cap - 2))
    return int(rng.integers(lower, cap + 1))


def _colors(rng, count, background, repeat_singletons):
    available = [c for c in range(10) if c != background]
    if rng.random() < 0.45:
        palette_size = int(rng.integers(2, min(4, count) + 1))
    else:
        palette_size = int(rng.integers(2, min(8, count, 9) + 1))
    palette = [int(v) for v in rng.choice(available, size=palette_size, replace=False)]
    if palette_size >= count and rng.random() < 0.35:
        colors = [int(v) for v in rng.permutation(palette)[:count]]
    else:
        weights = rng.dirichlet(np.ones(palette_size) * float(rng.choice((0.5, 1.0, 2.0))))
        colors = [int(v) for v in rng.choice(palette, size=count, p=weights)]
    if repeat_singletons:
        i, j = rng.choice(count, size=2, replace=False)
        colors[int(j)] = colors[int(i)]
        return colors, (int(i), int(j))
    return colors, ()


def _placement_ok(cells, color, placed):
    own = set(cells)
    for other_color, other_cells in placed:
        other = set(other_cells)
        if own & other:
            return False
        if color == other_color and any((r + dr, c + dc) in other
                                        for r, c in own
                                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))):
            return False
    try:
        _settlement_order(placed + [(color, tuple(cells))])
    except AssertionError:
        return False
    return True


def _construct_pair(rng, regime, primary_background, demonstration):
    for _ in range(300):
        height, width = _dimensions(rng, demonstration)
        count = _object_count(rng, height, width, regime, demonstration)
        if rng.random() < 0.62:
            background = primary_background
        else:
            background = int(rng.integers(10))
        compact = height * width <= 50
        probabilities = (0.28, 0.38, 0.10, 0.19, 0.05) if compact else (0.25, 0.30, 0.13, 0.21, 0.11)
        kinds = list(rng.choice(5, size=count, p=probabilities))
        if demonstration:
            kinds[int(rng.integers(count))] = int(rng.choice((3, 4)))
        repeat_singletons = count >= 4 and rng.random() < 0.35
        colors, repeat_indexes = _colors(rng, count, background, repeat_singletons)
        for i in repeat_indexes:
            kinds[i] = 0
        shapes = [_shape_array(rng, int(kind), compact) for kind in kinds]
        total_cells = sum(len(s) for s in shapes)
        if total_cells >= height * width * 0.43:
            continue
        placed = []
        success = True
        for index in rng.permutation(count):
            index = int(index)
            shape = shapes[index]
            shape_h = max(r for r, _ in shape) + 1
            shape_w = max(c for _, c in shape) + 1
            if shape_h > height or shape_w > width:
                success = False
                break
            for _ in range(90):
                top = int(rng.integers(height - shape_h + 1))
                edge_choice = rng.random()
                if edge_choice < 0.22:
                    left = width - shape_w
                elif edge_choice < 0.34 and width - shape_w >= 1:
                    left = width - shape_w - 1
                else:
                    left = int(rng.integers(width - shape_w + 1))
                cells = tuple((top + r, left + c) for r, c in shape)
                if _placement_ok(cells, colors[index], placed):
                    placed.append((colors[index], cells))
                    break
            else:
                success = False
                break
        if not success:
            continue
        grid = np.full((height, width), background, dtype=np.int8)
        for color, cells in placed:
            for row, col in cells:
                grid[row, col] = color
        if len(_components(grid, background)) != count:
            continue
        output, _, shifts = _settle(grid, background)
        if not any(shifts) or _source_copy(grid, output):
            continue
        features = _pair_features(grid, background)
        if demonstration and not (features['blocked'] >= 1
                                  and features['moved_multirow'] >= 1
                                  and features['rowwise_differs']):
            continue
        return {'input': grid, 'output': output}
    raise RuntimeError('could not construct a valid pair')


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    inputs = set()
    pairs = set()
    train_features = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            input_grid, output_grid = pair['input'], pair['output']
            for grid in (input_grid, output_grid):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            assert input_grid.shape == output_grid.shape
            input_signature = _signature(input_grid)
            pair_signature = (input_signature, _signature(output_grid))
            assert input_signature not in inputs and pair_signature not in pairs
            inputs.add(input_signature)
            pairs.add(pair_signature)
            assert not _source_copy(input_grid, output_grid)
            background = _dominant_color(input_grid)
            assert _dominant_color(output_grid) == background
            expected, components, shifts = _settle(input_grid, background)
            assert 3 <= len(components) <= 10
            assert np.array_equal(expected, output_grid)
            assert any(shift > 0 for shift in shifts)
            assert not np.array_equal(input_grid, output_grid)
            assert all(np.array_equal(np.bincount(input_grid[r].astype(int), minlength=10),
                                      np.bincount(output_grid[r].astype(int), minlength=10))
                       for r in range(input_grid.shape[0]))
            if split == 'train':
                train_features.append(_pair_features(input_grid, background))
    # A solver sees direction, collision blocking, and rigid multirow travel.
    assert any(f['blocked'] >= 1 and f['moved_multirow'] >= 1
               and f['rowwise_differs'] for f in train_features)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    primary_background = int(rng.integers(10))
    puzzle = {'train': [], 'test': []}
    used = set()
    for split, count in (('train', train_count), ('test', test_count)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            for _ in range(20):
                pair = _construct_pair(pair_rng, 'source_train' if split == 'train' else 'target_test',
                                       primary_background, split == 'train' and index == 0)
                sig = _signature(pair['input'])
                if sig not in used:
                    used.add(sig)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('could not find a unique pair')
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    a = generate_puzzle(123456789)
    b = generate_puzzle(123456789)
    assert all(np.array_equal(x[k], y[k])
               for split in ('train', 'test')
               for x, y in zip(a[split], b[split]) for k in ('input', 'output'))
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4),
                                          (None, 1), (None, 4), (2, None),
                                          (10, None), (6, 2))):
        p = generate_puzzle(8000 + index, train, test)
        validate_generated_puzzle(p)
        if train is not None:
            assert len(p['train']) == train
        if test is not None:
            assert len(p['test']) == test
    for index, arguments in enumerate(({'num_train': 1}, {'num_train': 11},
                                       {'num_test': 0}, {'num_test': 5},
                                       {'num_train': 2, 'num_test': 2},
                                       {'num_train': 3, 'num_test': 4},
                                       {'num_train': True, 'num_test': 1})):
        try:
            generate_puzzle(9000 + index, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    counts_train, counts_test, signatures, dimensions = set(), set(), set(), set()
    backgrounds = set()
    totals = {k: 0 for k in ('zero_moves', 'one_moves', 'long_moves',
                             'overlap_columns', 'reversed_order',
                             'adjacent_unlike', 'same_color_singletons',
                             'rowwise_differs')}
    dense = sparse = compact = expansive = simple_test = crowded_test = 0
    for seed in range(1000, 1042):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        counts_train.add(len(puzzle['train']))
        counts_test.add(len(puzzle['test']))
        signatures.add(tuple(_signature(pair['input'])
                             for split in ('train', 'test') for pair in puzzle[split]))
        train_counts = []
        test_counts = []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                grid = pair['input']
                bg = _dominant_color(grid)
                backgrounds.add(bg)
                features = _pair_features(grid, bg)
                dimensions.add(grid.shape)
                (train_counts if split == 'train' else test_counts).append(features['objects'])
                for key in totals:
                    totals[key] += int(features[key])
                dense += features['density'] >= 0.20
                sparse += features['density'] <= 0.10
                compact += grid.shape[1] <= 10
                expansive += grid.shape[1] >= 20
                # Independently recover the test answer from training evidence
                # and the test input, never from generation metadata.
                if split == 'test':
                    assert any(_pair_features(p['input'], _dominant_color(p['input']))['rowwise_differs']
                               for p in puzzle['train'])
                    prediction, _, _ = _settle(grid.copy(), bg)
                    assert np.array_equal(prediction, pair['output'])
        simple_test += min(test_counts) <= max(train_counts)
        crowded_test += max(test_counts) > max(train_counts)
    assert len(counts_train) >= 4 and len(counts_test) >= 3
    assert len(signatures) == 42 and len(dimensions) >= 30 and len(backgrounds) >= 7
    assert compact >= 20 and expansive >= 10 and dense >= 10 and sparse >= 10
    assert simple_test >= 5 and crowded_test >= 5
    assert all(value > 0 for value in totals.values())
