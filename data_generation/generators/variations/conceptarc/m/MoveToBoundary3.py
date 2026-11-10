import numpy as np


# Exact source inputs, used only to reject accidental copies.
_SOURCE_FOREGROUND = (
    ((2, 4), (3, 4), (4, 4), (5, 4), (5, 5), (5, 6)),
    ((4, 6), (4, 7), (5, 6), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6)),
    ((3, 1), (3, 2), (3, 3), (4, 1), (4, 3), (5, 1), (5, 2), (5, 3)),
    ((3, 0), (3, 1), (3, 2), (4, 2), (5, 0), (5, 1), (5, 2)),
    tuple([(2, c) for c in range(8)] + [(6, c) for c in range(5)]),
)


def _components(grid: np.ndarray, foreground: int) -> list[set[tuple[int, int]]]:
    remaining = set(zip(*np.where(grid == foreground)))
    parts = []
    while remaining:
        start = remaining.pop()
        part = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    part.add(neighbor)
                    stack.append(neighbor)
        parts.append(part)
    return parts


def _align(grid: np.ndarray, background: int, foreground: int) -> np.ndarray:
    result = np.full(grid.shape, background, dtype=np.int8)
    for part in _components(grid, foreground):
        shift = grid.shape[1] - 1 - max(c for _, c in part)
        for r, c in part:
            result[r, c + shift] = foreground
    return result


def _hole(part: set[tuple[int, int]]) -> bool:
    rows = [r for r, _ in part]
    cols = [c for _, c in part]
    top, bottom, left, right = min(rows), max(rows), min(cols), max(cols)
    empty = {(r, c) for r in range(top, bottom + 1)
             for c in range(left, right + 1) if (r, c) not in part}
    frontier = [cell for cell in empty if cell[0] in (top, bottom)
                or cell[1] in (left, right)]
    outside = set(frontier)
    while frontier:
        r, c = frontier.pop()
        for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if neighbor in empty and neighbor not in outside:
                outside.add(neighbor)
                frontier.append(neighbor)
    return bool(empty - outside)


def _shape(rng: np.random.Generator, kind: str,
           max_height: int, max_width: int) -> set[tuple[int, int]]:
    if kind == 'bar':
        width = int(rng.integers(2, min(max_width, 14) + 1))
        return {(0, c) for c in range(width)}
    if kind == 'vertical':
        height = int(rng.integers(2, min(max_height, 10) + 1))
        return {(r, 0) for r in range(height)}
    if kind == 'ring':
        height = int(rng.integers(3, min(max_height, 8) + 1))
        width = int(rng.integers(3, min(max_width, 9) + 1))
        return {(r, c) for r in range(height) for c in range(width)
                if r in (0, height - 1) or c in (0, width - 1)}
    if kind == 'elbow':
        height = int(rng.integers(2, min(max_height, 8) + 1))
        width = int(rng.integers(2, min(max_width, 9) + 1))
        stem = 0 if rng.integers(2) else width - 1
        foot = 0 if rng.integers(2) else height - 1
        return ({(r, stem) for r in range(height)}
                | {(foot, c) for c in range(width)})
    if kind == 'notch':
        height = int(rng.integers(3, min(max_height, 8) + 1))
        width = int(rng.integers(3, min(max_width, 9) + 1))
        side = int(rng.integers(2))
        result = set()
        for r in range(height):
            row_width = int(rng.integers(1, width + 1))
            if side:
                result.update((r, c) for c in range(row_width))
            else:
                result.update((r, c) for c in range(width - row_width, width))
        result.update((height - 1, c) for c in range(width))
        return result

    # Connected random growth provides many silhouettes beyond the basic motifs.
    target = int(rng.integers(5, min(20, max_height * max_width) + 1))
    cells = {(int(rng.integers(max_height)), int(rng.integers(max_width)))}
    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            frontier.update((rr, cc) for rr, cc in
                            ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                            if 0 <= rr < max_height and 0 <= cc < max_width
                            and (rr, cc) not in cells)
        if not frontier:
            break
        candidates = sorted(frontier)
        cells.add(candidates[int(rng.integers(len(candidates)))])
    top = min(r for r, _ in cells)
    left = min(c for _, c in cells)
    return {(r - top, c - left) for r, c in cells}


def _counts(rng: np.random.Generator, num_train: int | None,
            num_test: int | None) -> tuple[int, int]:
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= value <= high):
            raise ValueError('illegal pair count')
    if num_test is None:
        num_test = int(rng.integers(1, min(4, (num_train or 10) - 1) + 1))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_train <= num_test:
        raise ValueError('train count must exceed test count')
    return int(num_train), int(num_test)


def _dimensions(rng: np.random.Generator) -> tuple[int, int]:
    if rng.random() < 0.12:
        return 10, 10
    return int(rng.integers(10, 26)), int(rng.integers(10, 27))


def _choose_shift(rng: np.random.Generator, maximum: int) -> int:
    mode = int(rng.integers(10))
    if mode == 0:
        return 0
    if mode <= 2:
        return 1
    if mode <= 4:
        return maximum
    return int(rng.integers(1, maximum + 1))


def _make_pair(rng: np.random.Generator, role: str,
               palette: tuple[int, int]) -> dict:
    height, width = _dimensions(rng)
    diagnostic = role in ('train_multi', 'test_multi')
    if role == 'train_single':
        count = 1
    elif diagnostic:
        count = 2 if rng.random() < 0.68 or height < 13 else 3
    else:
        count = int(rng.integers(1, 4 if height >= 13 else 3))

    max_height = max(2, (height - count + 1) // count)
    max_width = min(width - 3, 14)
    shapes = []
    for _ in range(count):
        if role == 'train_single':
            kind = ('elbow', 'notch', 'irregular')[int(rng.integers(3))]
        else:
            kinds = ['bar', 'vertical', 'elbow', 'notch', 'irregular']
            if max_height >= 3:
                kinds.append('ring')
            kind = kinds[int(rng.integers(len(kinds)))]
        shapes.append(_shape(rng, kind, max_height, max_width))

    if role == 'train_single':
        # A whole-object shift must differ from aligning each row separately.
        shape = shapes[0]
        row_rights = {max(c for rr, c in shape if rr == r)
                      for r in {rr for rr, _ in shape}}
        if len(row_rights) == 1:
            shapes[0] = {(r, 0) for r in range(3)} | {(2, c) for c in range(3)}

    heights = [max(r for r, _ in shape) + 1 for shape in shapes]
    spare = height - sum(heights) - (count - 1)
    assert spare >= 0
    margins = rng.multinomial(spare, np.ones(count + 1) / (count + 1))
    tops = []
    cursor = int(margins[0])
    for index, shape_height in enumerate(heights):
        tops.append(cursor)
        cursor += shape_height + 1 + int(margins[index + 1])

    maximums = [width - (max(c for _, c in shape) + 1) for shape in shapes]
    shifts = [_choose_shift(rng, maximum) for maximum in maximums]
    if diagnostic:
        if role == 'test_multi' and rng.random() < 0.55:
            shifts[0] = maximums[0]
        if shifts[1] == shifts[0]:
            alternatives = [s for s in range(maximums[1] + 1)
                            if abs(s - shifts[0]) >= 2]
            if not alternatives:
                alternatives = [s for s in range(maximums[1] + 1)
                                if s != shifts[0]]
            shifts[1] = alternatives[int(rng.integers(len(alternatives)))]
    elif count >= 2 and rng.random() < 0.22:
        common = int(rng.integers(1, min(maximums) + 1))
        shifts = [common] * count
    if not any(shifts):
        shifts[0] = 1

    background, foreground = palette
    grid = np.full((height, width), background, dtype=np.int8)
    for shape, top, shift in zip(shapes, tops, shifts):
        left = width - (max(c for _, c in shape) + 1) - shift
        for r, c in shape:
            grid[top + r, left + c] = foreground
    assert np.count_nonzero(grid == foreground) * 2 < height * width
    return {'input': grid, 'output': _align(grid, background, foreground)}


def _input_key(grid: np.ndarray) -> tuple:
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    base = tuple(int(x) for x in rng.choice(10, size=2, replace=False))
    recolor = bool(rng.integers(2))
    train_roles = ['train_single', 'train_multi'] + ['train_extra'] * (train_count - 2)
    test_roles = ['test_multi'] + ['test_extra'] * (test_count - 1)
    rng.shuffle(train_roles)
    rng.shuffle(test_roles)
    puzzle = {'train': [], 'test': []}
    seen = set()
    previous_palette = None
    for section, roles in (('train', train_roles), ('test', test_roles)):
        for role in roles:
            if recolor and previous_palette is not None:
                palettes = [(b, f) for b in range(10) for f in range(10)
                            if b != f and (b, f) != previous_palette]
                palette = palettes[int(rng.integers(len(palettes)))]
            else:
                palette = base
            previous_palette = palette
            for _ in range(16):
                pair = _make_pair(rng, role, palette)
                key = _input_key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[section].append(pair)
                    break
            else:
                raise AssertionError('could not construct distinct pair')
    validate_generated_puzzle(puzzle)
    return puzzle


def _source_copy(grid: np.ndarray, background: int, foreground: int) -> bool:
    if grid.shape != (10, 10) or (background, foreground) != (0, 2):
        return False
    occupied = frozenset(zip(*np.where(grid == foreground)))
    return any(occupied == frozenset(source) for source in _SOURCE_FOREGROUND)


def _pair_features(pair: dict) -> dict:
    grid = pair['input']
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) == 2 and counts[0] != counts[1]
    background = int(values[np.argmax(counts)])
    foreground = int(values[np.argmin(counts)])
    parts = _components(grid, foreground)
    shifts = [grid.shape[1] - 1 - max(c for _, c in part) for part in parts]
    return {'background': background, 'foreground': foreground,
            'parts': parts, 'shifts': shifts,
            'row_shape': any(len({max(c for rr, c in part if rr == r)
                                  for r in {rr for rr, _ in part}}) > 1
                             for part in parts),
            'hole': any(_hole(part) for part in parts)}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen = set()
    train_features = []
    test_features = []
    for section in ('train', 'test'):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in (pair['input'], pair['output']):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = pair['input'], pair['output']
            assert inp.shape == out.shape
            key = _input_key(inp)
            assert key not in seen
            seen.add(key)
            feature = _pair_features(pair)
            bg, fg = feature['background'], feature['foreground']
            assert not _source_copy(inp, bg, fg)
            assert feature['parts'] and len(feature['parts']) <= 3
            assert all(shift >= 0 for shift in feature['shifts'])
            assert any(shift > 0 for shift in feature['shifts'])
            assert np.array_equal(out, _align(inp, bg, fg))
            assert np.count_nonzero(out == fg) == np.count_nonzero(inp == fg)
            output_parts = _components(out, fg)
            assert len(output_parts) == len(feature['parts'])
            assert all(max(c for _, c in part) == inp.shape[1] - 1
                       for part in output_parts)
            (train_features if section == 'train' else test_features).append(feature)

    # Every puzzle demonstrates whole-object alignment and independent shifts.
    assert any(len(f['parts']) == 1 and f['row_shape'] for f in train_features)
    assert any(len(f['parts']) >= 2 and len(set(f['shifts'])) >= 2
               for f in train_features)
    assert any(len(f['parts']) >= 2 and len(set(f['shifts'])) >= 2
               for f in test_features)


def _signature(puzzle: dict) -> tuple:
    return tuple((section, tuple((p['input'].shape, p['input'].tobytes(),
                                  p['output'].tobytes()) for p in puzzle[section]))
                 for section in ('train', 'test'))


def self_test() -> None:
    signatures = set()
    train_counts, test_counts = set(), set()
    dimensions, palettes, shapes, object_counts = set(), set(), set(), set()
    ring_train = ring_test = bar_train = long_bar = zero_shift = one_shift = 0
    left_train = off_left_test = vertical = equal_shift = combined = 0
    hole_sizes = set()
    mixed_palettes = ten_square = tall = wide = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = _signature(puzzle)
        assert signature == _signature(generate_puzzle(seed))
        signatures.add(signature)
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        pair_palettes = set()
        for section in ('train', 'test'):
            for pair in puzzle[section]:
                inp, out = pair['input'], pair['output']
                feature = _pair_features(pair)
                bg, fg = feature['background'], feature['foreground']
                assert np.array_equal(out, _align(inp, bg, fg))
                assert not _source_copy(inp, bg, fg)
                dimensions.add(inp.shape)
                palettes.add((bg, fg))
                pair_palettes.add((bg, fg))
                object_counts.add(len(feature['parts']))
                leftmost = min(c for part in feature['parts'] for _, c in part)
                left_train += section == 'train' and leftmost == 0
                off_left_test += section == 'test' and leftmost > 0
                equal_shift += (len(feature['parts']) >= 2
                                and len(set(feature['shifts'])) == 1)
                combined += (len(feature['parts']) >= 2
                             and len(set(feature['shifts'])) >= 2
                             and feature['hole'])
                ten_square += inp.shape == (10, 10)
                tall += inp.shape[0] > inp.shape[1]
                wide += inp.shape[1] > inp.shape[0]
                for part, shift in zip(feature['parts'], feature['shifts']):
                    top = min(r for r, _ in part)
                    left = min(c for _, c in part)
                    shapes.add(tuple(sorted((r - top, c - left) for r, c in part)))
                    zero_shift += shift == 0
                    one_shift += shift == 1
                    rows = {r for r, _ in part}
                    width = max(c for _, c in part) - min(c for _, c in part) + 1
                    height = max(rows) - min(rows) + 1
                    vertical += width == 1 and height >= 3
                    if _hole(part):
                        hole_sizes.add((height - 2, width - 2))
                    if len(rows) == 1 and len(part) >= 8:
                        long_bar += 1
                    if section == 'train':
                        ring_train += _hole(part)
                        bar_train += len(rows) == 1 and len(part) >= 2
                    else:
                        ring_test += _hole(part)
        mixed_palettes += len(pair_palettes) > 1
    assert len(signatures) == 42
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(dimensions) >= 80 and len(palettes) >= 20
    assert len(shapes) >= 80 and object_counts == {1, 2, 3}
    assert ring_train > 0 and ring_test > 0 and bar_train > 0
    assert long_bar > 0 and vertical > 0 and len(hole_sizes) >= 6
    assert zero_shift > 0 and one_shift > 0
    assert left_train > 0 and off_left_test > 0
    assert equal_shift > 0 and combined > 0
    assert mixed_palettes >= 8 and ten_square > 0 and tall > 0 and wide > 0

    for requested_train, requested_test in ((2, 1), (10, 1), (10, 4),
                                            (None, 1), (None, 4),
                                            (2, None), (10, None)):
        puzzle = generate_puzzle(1234, requested_train, requested_test)
        validate_generated_puzzle(puzzle)
        assert requested_train is None or len(puzzle['train']) == requested_train
        assert requested_test is None or len(puzzle['test']) == requested_test
        assert _signature(puzzle) == _signature(generate_puzzle(
            1234, requested_train, requested_test))
    for requested_train, requested_test in ((1, None), (11, None), (None, 0),
                                            (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(1234, requested_train, requested_test)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts were accepted')
