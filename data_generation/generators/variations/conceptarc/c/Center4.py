import numpy as np


_SOURCE_ROWS = {
    (
        "0000000",
        "0400007",
        "0000000",
        "0003000",
        "0000000",
        "0500000",
        "0000060",
    ),
    (
        "000000600",
        "020000600",
        "020000600",
        "020070000",
        "000070000",
        "000070400",
        "000000400",
        "000090000",
        "000090000",
    ),
    ("222", "282", "222"),
    (
        "0000000",
        "0333330",
        "0300030",
        "0304030",
        "0300030",
        "0333330",
        "0000000",
    ),
    (
        "000000000000",
        "000000000000",
        "001111111100",
        "000000000000",
        "000000000000",
        "000006600000",
        "000006600000",
        "000000000000",
        "000444444000",
        "000400004000",
        "000444444000",
        "000000000000",
    ),
    (
        "0000000000000",
        "0000000080000",
        "0777700000000",
        "0777700000000",
        "0777700066000",
        "0777700066000",
        "0000004000000",
        "0000000000000",
        "0000222000000",
        "0000222000000",
        "0000222000000",
        "0000000000000",
        "0000000000000",
    ),
}


def _rows(grid):
    return tuple(''.join(str(int(v)) for v in row) for row in grid)


def _component(grid, start):
    color = int(grid[start])
    found = {start}
    pending = [start]
    while pending:
        row, col = pending.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = row + dr, col + dc
            if (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]
                    and (nr, nc) not in found and int(grid[nr, nc]) == color):
                found.add((nr, nc))
                pending.append((nr, nc))
    return found


def _components(grid):
    seen = set()
    result = []
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            if grid[row, col] and (row, col) not in seen:
                cells = _component(grid, (row, col))
                seen.update(cells)
                result.append((int(grid[row, col]), cells))
    return result


def _erase_center(grid):
    result = grid.copy()
    middle = grid.shape[0] // 2
    for cell in _component(grid, (middle, middle)):
        result[cell] = 0
    return result


def _shape(cells):
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _center_shape(size, kind, rng):
    m = size // 2
    if size % 2:
        if kind == 'single':
            return {(m, m)}
        if kind in ('horizontal', 'vertical'):
            lengths = [n for n in (3, 5, 7) if n <= size - 2]
            length = int(rng.choice(lengths))
            if kind == 'horizontal':
                return {(m, c) for c in range(m - length // 2, m + length // 2 + 1)}
            return {(r, m) for r in range(m - length // 2, m + length // 2 + 1)}
        if kind == 'block':
            dims = [(3, 3)]
            if size >= 9:
                dims += [(3, 5), (5, 3), (5, 5)]
            h, w = dims[int(rng.integers(len(dims)))]
            return {(r, c) for r in range(m - h // 2, m + h // 2 + 1)
                    for c in range(m - w // 2, m + w // 2 + 1)}
        if kind == 'plus':
            arm = int(rng.choice([1, 2] if size >= 9 else [1]))
            return ({(m, c) for c in range(m - arm, m + arm + 1)} |
                    {(r, m) for r in range(m - arm, m + arm + 1)})
    else:
        if kind == 'block':
            dims = [(2, 2), (2, 4), (4, 2), (4, 4)]
            if size >= 10:
                dims += [(2, 6), (6, 2)]
            dims = [(h, w) for h, w in dims if max(h, w) <= size - 2]
            h, w = dims[int(rng.integers(len(dims)))]
            return {(r, c) for r in range(m - h // 2, m + h // 2)
                    for c in range(m - w // 2, m + w // 2)}
        if kind == 'cross':
            core = {(r, c) for r in (m - 1, m) for c in (m - 1, m)}
            arms = {(m - 2, m - 1), (m - 2, m), (m + 1, m - 1),
                    (m + 1, m), (m - 1, m - 2), (m, m - 2),
                    (m - 1, m + 1), (m, m + 1)}
            return core | arms
    raise ValueError('invalid center shape')


def _center_kinds(size):
    if size == 5:
        return ('single', 'horizontal', 'vertical', 'plus')
    return ('single', 'horizontal', 'vertical', 'block', 'plus') if size % 2 else ('block', 'cross')


def _frame(h, w):
    return {(r, c) for r in range(h) for c in range(w)
            if r in (0, h - 1) or c in (0, w - 1)}


def _other_shape(rng, size):
    kind = int(rng.choice(6, p=[.21, .19, .18, .18, .15, .09]))
    if kind == 0:
        return {(0, 0)}
    if kind in (1, 2):
        length = int(rng.integers(2, min(9, size - 1) + 1))
        return {(0, c) for c in range(length)} if kind == 1 else {(r, 0) for r in range(length)}
    if kind == 3:
        h = int(rng.integers(2, min(5, size - 2) + 1))
        w = int(rng.integers(2, min(5, size - 2) + 1))
        return {(r, c) for r in range(h) for c in range(w)}
    if kind == 4:
        h = int(rng.integers(3, min(8, size - 2) + 1))
        w = int(rng.integers(3, min(8, size - 2) + 1))
        if h == w and size >= 8:
            w = w + 1 if w < min(8, size - 2) else w - 1
        return _frame(h, w)
    return {(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)}


def _clear(candidate, occupied):
    return all(max(abs(r - rr), abs(c - cc)) >= 2
               for r, c in candidate for rr, cc in occupied)


def _place(grid, shape, color, rng, placement='scatter'):
    size = grid.shape[0]
    h = max(r for r, _ in shape) + 1
    w = max(c for _, c in shape) + 1
    if h > size or w > size:
        return False
    occupied = set(zip(*np.nonzero(grid)))
    options = []
    for r0 in range(size - h + 1):
        for c0 in range(size - w + 1):
            cells = {(r0 + r, c0 + c) for r, c in shape}
            if _clear(cells, occupied):
                options.append((r0, c0))
    if not options:
        return False
    if placement == 'near':
        m = (size - 1) / 2
        options.sort(key=lambda p: abs(p[0] + (h - 1) / 2 - m) +
                     abs(p[1] + (w - 1) / 2 - m))
        options = options[:max(1, len(options) // 3)]
    r0, c0 = options[int(rng.integers(len(options)))]
    for r, c in shape:
        grid[r0 + r, c0 + c] = color
    return True


def _enclose(grid, center, color, rng):
    size = grid.shape[0]
    r0, r1 = min(r for r, _ in center), max(r for r, _ in center)
    c0, c1 = min(c for _, c in center), max(c for _, c in center)
    options = []
    for top in (2, 3):
        for bottom in (2, 3):
            for left in (2, 3):
                for right in (2, 3):
                    a, b = r0 - top, r1 + bottom
                    x, y = c0 - left, c1 + right
                    if a >= 0 and x >= 0 and b < size and y < size:
                        options.append((a, b, x, y))
    if not options:
        return False
    a, b, x, y = options[int(rng.integers(len(options)))]
    cells = {(a + r, x + c) for r, c in _frame(b - a + 1, y - x + 1)}
    if not _clear(cells, center):
        return False
    for cell in cells:
        grid[cell] = color
    return True


def _make_pair(rng, size, kind, center_color, decoy=False, field=False, regime='source_train'):
    center = _center_shape(size, kind, rng)
    grid = np.zeros((size, size), dtype=np.int8)
    palette_size = int(rng.integers(2, 8))
    other_colors = [int(x) for x in rng.choice(
        [x for x in range(1, 10) if x != center_color],
        size=palette_size - 1, replace=False)]
    if field:
        grid[:, :] = other_colors[0]
        for cell in center:
            grid[cell] = center_color
        return grid, _erase_center(grid)

    for cell in center:
        grid[cell] = center_color
    modes = ('scatter', 'near', 'enclose', 'dense', 'sparse')
    weights = (.27, .23, .19, .18, .13) if regime == 'source_train' else (.18, .25, .23, .23, .11)
    mode = str(rng.choice(modes, p=weights))
    if mode == 'enclose':
        _enclose(grid, center, int(rng.choice(other_colors + [center_color])), rng)
    if decoy:
        if not _place(grid, _shape(center), center_color, rng):
            return None
    if mode == 'sparse':
        goal = int(rng.integers(1, 3))
    elif mode == 'dense':
        goal = int(rng.integers(6, 11))
    elif mode == 'enclose':
        goal = int(rng.integers(1, 5))
    else:
        goal = int(rng.integers(2, 7))
    for _ in range(goal):
        shape = _other_shape(rng, size)
        color = center_color if rng.random() < .23 else int(rng.choice(other_colors))
        _place(grid, shape, color, rng, 'near' if mode == 'near' else 'scatter')
    if len(_components(grid)) < 2:
        if not _place(grid, {(0, 0)}, int(rng.choice(other_colors)), rng):
            return None
    return grid, _erase_center(grid)


def _valid_count(value, lower, upper, name):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not lower <= value <= upper:
        raise ValueError(f'{name} must be an integer from {lower} to {upper}')
    return int(value)


def _size(rng, regime, classic=False, minimum=5):
    if regime == 'source_train':
        band = int(rng.choice(3, p=[.28, .38, .34]))
        lo, hi = ((5, 10), (11, 16), (17, 23))[band]
    else:
        band = int(rng.choice(3, p=[.34, .32, .34]))
        lo, hi = ((5, 11), (12, 19), (20, 27))[band]
    values = [n for n in range(max(lo, minimum), hi) if not classic or n % 2]
    if not values:
        values = [n for n in range(minimum, 24 if regime == 'source_train' else 28)
                  if not classic or n % 2]
    return int(rng.choice(values))


def _new_pair(rng, size, kind, center_color, decoy, field, regime, seen):
    for _ in range(40):
        pair = _make_pair(rng, size, kind, center_color, decoy, field, regime)
        if pair is None:
            continue
        input_grid, output_grid = pair
        key = (input_grid.shape, input_grid.tobytes())
        if key not in seen and _rows(input_grid) not in _SOURCE_ROWS:
            seen.add(key)
            return {'input': input_grid, 'output': output_grid}
    raise RuntimeError('could not construct a unique pair')


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    num_train = _valid_count(num_train, 2, 10, 'num_train')
    num_test = _valid_count(num_test, 1, 4, 'num_test')
    if num_train is None and num_test is None:
        num_train = 2 + int(seed) % 9
        num_test = 1 + (int(seed) // 9) % min(4, num_train - 1)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')

    # A subset retains the source's odd-training / even-test extrapolation.
    classic = bool(rng.random() < .24)
    seen = set()
    train = []
    anchor_size = _size(rng, 'source_train', classic, minimum=11)
    anchor_kinds = ('single', 'horizontal', 'vertical') if anchor_size % 2 else ('block',)
    anchor_kind = str(rng.choice(anchor_kinds))
    anchor_color = int(rng.integers(1, 10))
    train.append(_new_pair(rng, anchor_size, anchor_kind, anchor_color,
                           True, False, 'source_train', seen))
    anchor_length = len(_component(train[0]['input'],
                                   (anchor_size // 2, anchor_size // 2)))

    for index in range(1, num_train):
        for _ in range(40):
            size = _size(rng, 'source_train', classic)
            kind = str(rng.choice(_center_kinds(size)))
            field = bool(rng.random() < (.26 if index == 1 else .09))
            if field:
                kind = 'single' if size % 2 else 'block'
            color = int(rng.integers(1, 10))
            if index == 1 and color == anchor_color:
                continue
            pair = _new_pair(rng, size, kind, color,
                             bool(index > 1 and rng.random() < .25 and size >= 11 and
                                  kind in ('single', 'horizontal', 'vertical', 'block')),
                             field, 'source_train', seen)
            cells = _component(pair['input'], (size // 2, size // 2))
            if index == 1 and (len(cells) == anchor_length or
                               {int(v) for v in np.unique(pair['input']) if v} ==
                               {int(v) for v in np.unique(train[0]['input']) if v}):
                seen.remove((pair['input'].shape, pair['input'].tobytes()))
                continue
            train.append(pair)
            break
        else:
            raise RuntimeError('could not construct varied training pairs')
    rng.shuffle(train)

    test = []
    even_target = int(rng.integers(num_test)) if classic else -1
    for index in range(num_test):
        size = _size(rng, 'target_test')
        if index == even_target and size % 2:
            size = size + 1 if size < 27 else size - 1
        if index != even_target and rng.random() < .20:
            if size % 2 == 0:
                size = size - 1 if size > 5 else size + 1
            kind = 'single'
        else:
            kind = str(rng.choice(_center_kinds(size)))
        color = int(rng.integers(1, 10))
        test.append(_new_pair(rng, size, kind, color,
                              bool(rng.random() < .32 and size >= 11 and
                                   kind in ('single', 'horizontal', 'vertical', 'block')),
                              False, 'target_test', seen))
    puzzle = {'train': train, 'test': test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen_inputs, seen_pairs = set(), set()
    train_center_colors, train_center_sizes, train_color_sets = set(), set(), set()
    train_decoy = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = pair['input'], pair['output']
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape and inp.shape[0] == inp.shape[1]
            size = inp.shape[0]
            m = size // 2
            assert inp[m, m] != 0
            center = _component(inp, (m, m))
            if size % 2 == 0:
                assert {(m-1, m-1), (m-1, m), (m, m-1), (m, m)} <= center
            assert np.array_equal(out, _erase_center(inp))
            assert not np.array_equal(inp, out)
            key = (inp.shape, inp.tobytes())
            pair_key = (key, out.tobytes())
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            assert _rows(inp) not in _SOURCE_ROWS
            color = int(inp[m, m])
            others = [(c, cells) for c, cells in _components(inp) if cells != center]
            # The center remains a distinct object even under diagonal grouping.
            assert all(_clear(center, cells) for c, cells in others if c == color)
            if split == 'train':
                train_center_colors.add(color)
                train_center_sizes.add(len(center))
                train_color_sets.add(frozenset(int(v) for v in np.unique(inp) if v))
                train_decoy |= any(c == color and _shape(cells) == _shape(center)
                                   for c, cells in others)
    assert len(train_center_colors) >= 2
    assert len(train_center_sizes) >= 2
    assert len(train_color_sets) >= 2
    assert train_decoy


def _signature(puzzle):
    return tuple((split, tuple((pair['input'].shape, pair['input'].tobytes(),
                                pair['output'].tobytes()) for pair in puzzle[split]))
                 for split in ('train', 'test'))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        assert _signature(puzzle) == _signature(generate_puzzle(seed))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                changed = set(zip(*np.nonzero(inp != out)))
                center = _component(inp, (inp.shape[0] // 2, inp.shape[1] // 2))
                assert changed == center
                assert all(out[cell] == 0 for cell in center)
    assert len({_signature(p) for p in puzzles}) >= 40
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    assert any(any(pair['input'].shape[0] % 2 == 0 for pair in p['train']) for p in puzzles)
    assert any(any(pair['input'].shape[0] <= 11 for pair in p['test']) for p in puzzles)
    assert any(any(pair['input'].shape[0] >= 18 for pair in p['train']) for p in puzzles)
    assert any(all(pair['input'].shape[0] % 2 for pair in p['train']) and
               any(pair['input'].shape[0] % 2 == 0 for pair in p['test']) for p in puzzles)
    for a, b in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                 (2, None), (10, None)):
        p = generate_puzzle(101, a, b)
        validate_generated_puzzle(p)
        assert _signature(p) == _signature(generate_puzzle(101, a, b))
        if a is not None:
            assert len(p['train']) == a
        if b is not None:
            assert len(p['test']) == b
    for a, b in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4), (4, 4), (2, 4)):
        try:
            generate_puzzle(101, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal pair counts accepted')
