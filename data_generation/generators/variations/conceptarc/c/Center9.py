"""ARC family: move one pixel to the midpoint of two parallel lines."""

import numpy as np

_SOURCE_INPUTS = {
    tuple(s.split('/')) for s in (
        '060006/060086/060006/060006/060006/060006',
        '000000/333333/000000/000000/000070/333333',
        '000000000/888888888/000000000/000000000/000000000/000000000/004000000/888888888/000000000',
        '04000004/04000004/04600004/04000004/04000004/04000004/04000004/04000004',
        '030003/030003/030083/030003/030003/030003',
        '7777777777/0000000000/0000000000/0000000000/0000000000/0000000000/0000000000/0000000000/0000000700/0000000000/7777777777',
        '5555555/0000000/0000000/2000000/5555555/0000000/0000000',
    )
}


def _rows(grid):
    return tuple(''.join(str(int(x)) for x in row) for row in grid)


def _counts(rng, num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10),
                                   ('num_test', num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                  not isinstance(value, (int, np.integer)) or
                                  not low <= int(value) <= high):
            raise ValueError(f'{name} must be an integer from {low} to {high}')
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_test = int(num_test)
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_train = int(num_train)
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    else:
        num_train, num_test = int(num_train), int(num_test)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return num_train, num_test


def _color(rng, excluded):
    return int(rng.choice([x for x in range(10) if x not in excluded]))


def _pair(rng, orientation, line, marker, background, train,
          side=None, large_move=False):
    same = line == marker
    gap_min = 8 if same and large_move else (6 if same or large_move else 4)
    small = rng.random() < (.28 if train else .34)
    gap_max = (8 if small else (22 if train else 26))
    gap = int(rng.choice(np.arange(gap_min, gap_max + 1, 2)))
    axis_limit = 11 if small else 30
    margin_count = int(rng.choice([0, 1, 2], p=(.20, .35, .45) if train
                                  else (.32, .32, .36)))
    if gap == 4 and margin_count == 0:
        margin_count = 1
    available = axis_limit - gap - 1
    if margin_count == 0:
        before = after = 0
    elif margin_count == 1:
        margin = int(rng.integers(1, min(7, available + 1)))
        before, after = (margin, 0) if rng.integers(2) else (0, margin)
    else:
        before = int(rng.integers(1, min(6, available)))
        after = int(rng.integers(1, min(6, available - before + 1)))
    axis_size = gap + before + after + 1
    along_size = int(rng.integers(6, 12 if small else 27))
    if rng.random() < .22:
        along_size = axis_size
    middle = before + gap // 2
    moves = list(range(1, gap // 2))
    if same:
        moves = [d for d in moves if gap // 2 - d >= 2]
    if large_move:
        moves = [d for d in moves if d >= 2]
    distance = int(rng.choice(moves))
    if side is None:
        side = int(rng.choice([-1, 1]))
    axis = middle + side * distance
    along = (int(rng.choice([0, along_size - 1]))
             if rng.random() < (.22 if train else .42)
             else int(rng.integers(1, along_size - 1)))
    height, width = ((axis_size, along_size) if orientation == 'horizontal'
                     else (along_size, axis_size))
    inp = np.full((height, width), background, dtype=np.int8)
    if orientation == 'horizontal':
        inp[before, :] = inp[before + gap, :] = line
        old, new = (axis, along), (middle, along)
    else:
        inp[:, before] = inp[:, before + gap] = line
        old, new = (along, axis), (along, middle)
    inp[old] = marker
    out = inp.copy()
    out[old] = background
    out[new] = marker
    return {'input': inp, 'output': out}


def _inspect(pair):
    """Reconstruct the rule and features from the grids, not sampled metadata."""
    inp, out = pair['input'], pair['output']
    assert inp.shape == out.shape
    height, width = inp.shape
    colors, counts = np.unique(inp, return_counts=True)
    background = int(colors[np.argmax(counts)])
    rows = [r for r in range(height) if inp[r, 0] != background and
            np.all(inp[r] == inp[r, 0])]
    cols = [c for c in range(width) if inp[0, c] != background and
            np.all(inp[:, c] == inp[0, c])]
    if len(rows) == 2 and not cols:
        orientation, (first, second) = 'horizontal', rows
        line = int(inp[first, 0])
        assert np.all(inp[second] == line)
        rem = inp.copy()
        rem[first, :] = rem[second, :] = background
        axis_size, along_size = height, width
    elif len(cols) == 2 and not rows:
        orientation, (first, second) = 'vertical', cols
        line = int(inp[0, first])
        assert np.all(inp[:, second] == line)
        rem = inp.copy()
        rem[:, first] = rem[:, second] = background
        axis_size, along_size = width, height
    else:
        raise AssertionError('exactly two full same-colored lines required')
    gap = second - first
    assert gap >= 4 and gap % 2 == 0
    positions = np.argwhere(rem != background)
    assert len(positions) == 1
    row, col = map(int, positions[0])
    marker = int(inp[row, col])
    axis = row if orientation == 'horizontal' else col
    along = col if orientation == 'horizontal' else row
    middle = (first + second) // 2
    assert first < axis < second and axis != middle
    if marker == line:
        assert axis - first >= 2 and second - axis >= 2
    expected = inp.copy()
    expected[row, col] = background
    expected[(middle, col) if orientation == 'horizontal' else (row, middle)] = marker
    assert np.array_equal(out, expected)
    assert np.count_nonzero(inp != out) == 2
    return dict(orientation=orientation, background=background,
                line=line, marker=marker, height=height, width=width,
                first=first, second=second, gap=gap,
                distance=abs(axis - middle), side=-1 if axis < middle else 1,
                along=along, edge=along in (0, along_size - 1),
                margins=int(first > 0) + int(second < axis_size - 1))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    ntrain, ntest = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest
    seen_inputs, seen_pairs, features = set(), set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            inp, out = _rows(pair['input']), _rows(pair['output'])
            assert inp not in seen_inputs and inp not in _SOURCE_INPUTS
            assert (inp, out) not in seen_pairs
            seen_inputs.add(inp)
            seen_pairs.add((inp, out))
            features[split].append(_inspect(pair))
    train = features['train']
    assert {f['orientation'] for f in train} == {'horizontal', 'vertical'}
    assert {f['side'] for f in train} == {-1, 1}
    assert sum(f['line'] != f['marker'] for f in train) >= 2
    assert any(a['line'] == b['marker'] and a['marker'] == b['line']
               and a['line'] != a['marker'] for a in train for b in train)
    if ntrain >= 3:
        assert any(f['line'] == f['marker'] for f in train)
    assert any(f['distance'] >= 2 for f in train)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    ntrain, ntest = _counts(rng, num_train, num_test)
    a, b = map(int, rng.choice(np.arange(1, 10), size=2, replace=False))
    first_orientation = str(rng.choice(['horizontal', 'vertical']))
    first_side = int(rng.choice([-1, 1]))
    second_side = (int(rng.choice([-1, 1])) if ntrain >= 3
                   else -first_side)
    same_index = int(rng.integers(2, ntrain)) if ntrain >= 3 else -1
    large_index = int(rng.integers(ntrain))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', ntrain), ('test', ntest)):
        for index in range(count):
            for _ in range(60):
                local = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                if split == 'train' and index < 2:
                    line, marker = (a, b) if index == 0 else (b, a)
                    orientation = first_orientation if index == 0 else (
                        'vertical' if first_orientation == 'horizontal' else 'horizontal')
                    side = first_side if index == 0 else second_side
                else:
                    orientation = str(local.choice(['horizontal', 'vertical']))
                    side = (-first_side if split == 'train' and index == 2
                            and first_side == second_side else None)
                    line = _color(local, {0})
                    same = (index == same_index if split == 'train'
                            else local.random() < .34)
                    marker = line if same else _color(local, {0, line})
                background = (0 if local.random() < .72 else
                              _color(local, {0, line, marker}))
                pair = _pair(local, orientation, line, marker, background,
                             split == 'train', side,
                             split == 'train' and index == large_index)
                signature = _rows(pair['input'])
                if signature not in seen and signature not in _SOURCE_INPUTS:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('could not construct unique pair')
    validate_generated_puzzle(puzzle)
    return puzzle


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s])
        for k in ('input', 'output')) for s in ('train', 'test'))


def self_test() -> None:
    train_counts, test_counts, dimensions, layouts, palettes = set(), set(), set(), set(), set()
    gaps, distances, margins = set(), set(), set()
    same_train = nonzero = train_edge = train_rect = 0
    mixed_background = mixed_margins = 0
    small_test = vertical_first = combined_test = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        train_features = [_inspect(pair) for pair in puzzle['train']]
        mixed_background += len({f['background'] for f in train_features}) > 1
        mixed_margins += len({f['margins'] for f in train_features}) > 1
        vertical_first += _inspect(puzzle['test'][0])['orientation'] == 'vertical'
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _inspect(pair)
                dimensions.add((f['height'], f['width']))
                layouts.add((f['height'], f['width'], f['orientation'],
                             f['first'], f['second'], f['along'],
                             f['side'], f['distance']))
                palettes.add((f['background'], f['line'], f['marker']))
                gaps.add(f['gap'])
                distances.add(f['distance'])
                margins.add(f['margins'])
                nonzero += f['background'] != 0
                if split == 'train':
                    same_train += f['line'] == f['marker']
                    train_edge += f['edge']
                    train_rect += f['height'] != f['width']
                else:
                    small_test += max(f['height'], f['width']) <= 11
                    combined_test += (f['margins'] == 0 and f['edge'] and
                                      f['line'] == f['marker'])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(dimensions) >= 70 and len(layouts) >= 180 and len(palettes) >= 90
    assert len(gaps) >= 9 and len(distances) >= 5 and margins == {0, 1, 2}
    assert same_train >= 15 and nonzero >= 35 and train_edge >= 20 and train_rect >= 70
    assert mixed_background >= 25 and mixed_margins >= 35
    assert small_test >= 20 and vertical_first >= 8 and combined_test >= 2
    for counts in ((2, 1), (10, 1), (10, 4), (None, 1),
                   (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(12345, *counts)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(12345, *counts))
        assert all(want is None or got == want for got, want in
                   zip((len(puzzle['train']), len(puzzle['test'])), counts))
    for counts in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (2, 4)):
        try:
            generate_puzzle(3, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts were accepted')
