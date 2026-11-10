import numpy as np


_NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_NEIGHBORS_8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),             (0, 1),
    (1, -1),  (1, 0),    (1, 1),
)


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        (int(grid.shape[0]), int(grid.shape[1])),
        tuple(int(value) for value in grid.ravel()),
    )


def _pair_key(pair: dict) -> tuple:
    return _grid_key(pair["input"]), _grid_key(pair["output"])


def _rows_key(rows: tuple[str, ...]) -> tuple:
    return (
        (len(rows), len(rows[0])),
        tuple(int(character) for row in rows for character in row),
    )


_SOURCE_PAIR_ROWS = (
    (
        (
            "00000",
            "88000",
            "88000",
            "00000",
            "00000",
        ),
        (
            "00000",
            "00880",
            "00880",
            "00000",
            "00000",
        ),
    ),
    (
        (
            "0000000000",
            "0888000000",
            "0808000000",
            "0888000000",
            "0000000000",
            "0000000000",
            "0800000000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
        (
            "0000000000",
            "0000888000",
            "0000808000",
            "0000888000",
            "0000000000",
            "0000000000",
            "0080000000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
    ),
    (
        (
            "000000000000",
            "088880000000",
            "088000000000",
            "000880000000",
            "088880000000",
            "000000000000",
            "000000000000",
            "000000000000",
            "000088000000",
            "000088000000",
            "000000000000",
            "000000000000",
        ),
        (
            "000000000000",
            "000008888000",
            "000008800000",
            "000000088000",
            "000008888000",
            "000000000000",
            "000000000000",
            "000000000000",
            "000000880000",
            "000000880000",
            "000000000000",
            "000000000000",
        ),
    ),
    (
        (
            "000000000000",
            "080808000000",
            "080808000000",
            "008080000000",
            "008080000000",
            "080808000000",
            "000000000000",
            "000000000000",
            "000008000000",
            "000088000000",
            "000000000000",
            "000000000000",
        ),
        (
            "000000000000",
            "000000808080",
            "000000808080",
            "000000080800",
            "000000080800",
            "000000808080",
            "000000000000",
            "000000000000",
            "000000080000",
            "000000880000",
            "000000000000",
            "000000000000",
        ),
    ),
)

_SOURCE_PAIR_KEYS = frozenset(
    (_rows_key(input_rows), _rows_key(output_rows))
    for input_rows, output_rows in _SOURCE_PAIR_ROWS
)


def _validate_optional_count(
    value: int | None,
    name: str,
    minimum: int,
    maximum: int,
) -> None:
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    if not minimum <= int(value) <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    _validate_optional_count(num_train, "num_train", 2, 10)
    _validate_optional_count(num_test, "num_test", 1, 4)

    if num_train is None and num_test is None:
        # This seed-dependent cycle guarantees diverse automatic counts over
        # consecutive fixed seeds while remaining completely deterministic.
        selector = int(seed)
        test_count = 1 + selector % 4
        minimum_train = test_count + 1
        number_of_choices = 11 - minimum_train
        train_count = (
            minimum_train
            + (selector // 4) % number_of_choices
        )
    elif num_train is None:
        test_count = int(num_test)
        minimum_train = max(2, test_count + 1)
        train_count = int(rng.integers(minimum_train, 11))
    elif num_test is None:
        train_count = int(num_train)
        maximum_test = min(4, train_count - 1)
        if maximum_test < 1:
            raise ValueError("no legal test count exists")
        test_count = int(rng.integers(1, maximum_test + 1))
    else:
        train_count = int(num_train)
        test_count = int(num_test)

    if not 2 <= train_count <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if not 1 <= test_count <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    colors = rng.choice(10, size=2, replace=False)
    return {
        "background": int(colors[0]),
        "foreground": int(colors[1]),
    }


def _components_of_cells(
    cells: set[tuple[int, int]],
    diagonal: bool = True,
) -> list[set[tuple[int, int]]]:
    offsets = _NEIGHBORS_8 if diagonal else _NEIGHBORS_4
    remaining = set(cells)
    components = []

    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component = {start}
        stack = [start]

        while stack:
            row, column = stack.pop()
            for delta_row, delta_column in offsets:
                neighbor = (
                    row + delta_row,
                    column + delta_column,
                )
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)

        components.append(component)

    return components


def _components(
    mask: np.ndarray,
    diagonal: bool = True,
) -> list[set[tuple[int, int]]]:
    cells = {
        (int(position[0]), int(position[1]))
        for position in np.argwhere(mask)
    }
    return _components_of_cells(cells, diagonal=diagonal)


def _is_connected(
    cells: set[tuple[int, int]],
    diagonal: bool,
) -> bool:
    return bool(cells) and len(
        _components_of_cells(cells, diagonal=diagonal)
    ) == 1


def _bbox(
    cells: set[tuple[int, int]],
) -> tuple[int, int, int, int]:
    rows = [row for row, _ in cells]
    columns = [column for _, column in cells]
    return min(rows), max(rows), min(columns), max(columns)


def _bbox_width(cells: set[tuple[int, int]]) -> int:
    _, _, left, right = _bbox(cells)
    return right - left + 1


def _bbox_height(cells: set[tuple[int, int]]) -> int:
    top, bottom, _, _ = _bbox(cells)
    return bottom - top + 1


def _normalize(
    cells: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    top, _, left, _ = _bbox(cells)
    return {
        (row - top, column - left)
        for row, column in cells
    }


def _shape_density(cells: set[tuple[int, int]]) -> float:
    return len(cells) / (
        _bbox_height(cells) * _bbox_width(cells)
    )


def _frontier(cells, height, width):
    return sorted({(r + dr, c + dc) for r, c in cells
                   for dr, dc in _NEIGHBORS_8
                   if 0 <= r + dr < height and 0 <= c + dc < width
                   and (r + dr, c + dc) not in cells})


def _has_hole(cells):
    top, bottom, left, right = _bbox(cells)
    empty = {(r, c) for r in range(top, bottom + 1)
             for c in range(left, right + 1) if (r, c) not in cells}
    if not empty:
        return False
    for part in _components_of_cells(empty, diagonal=False):
        if all(top < r < bottom and left < c < right for r, c in part):
            return True
    return False


def _make_shape(rng, width, kind):
    assert 1 <= width <= 7
    if width == 1:
        height = int(rng.integers(1, 7))
        return {(r, 0) for r in range(height)}
    if kind == 'hollow' and width >= 3:
        height = int(rng.integers(3, 9))
        cells = {(r, c) for r in range(height) for c in range(width)
                 if r in (0, height - 1) or c in (0, width - 1)}
        # Keep at least one enclosed background cell. The ring may be thickened.
        for r in range(1, height - 1):
            for c in range(1, width - 1):
                if rng.random() < .12:
                    trial = cells | {(r, c)}
                    if _has_hole(trial) and _is_connected(trial, True):
                        cells = trial
        return cells
    for _ in range(100):
        height = int(rng.integers(2, 9 if width >= 4 else 7))
        row = int(rng.integers(0, height))
        cells = {(row, 0)}
        for col in range(1, width):
            choices = [v for v in (row - 1, row, row + 1)
                       if 0 <= v < height and (kind != 'diagonal' or v != row)]
            row = int(rng.choice(choices))
            cells.add((row, col))
        if kind == 'diagonal':
            extras = int(rng.integers(0, max(2, width // 2 + 1)))
        elif kind == 'thick':
            extras = int(rng.integers(width, max(width + 1, height * width // 2 + 1)))
        else:
            extras = int(rng.integers(1, max(2, width + height // 2)))
        for _ in range(extras):
            frontier = _frontier(cells, height, width)
            if frontier:
                cells.add(frontier[int(rng.integers(len(frontier)))])
        cells = _normalize(cells)
        if _bbox_width(cells) != width or not _is_connected(cells, True):
            continue
        if kind == 'diagonal':
            parts = _components_of_cells(cells, diagonal=False)
            if len(parts) <= 1 or all(_bbox_width(p) == width for p in parts):
                continue
        return cells
    raise RuntimeError('shape construction failed')


def _sets_touch(first, second):
    return any((r + dr, c + dc) in second for r, c in first
               for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def _relationship(sources, destinations):
    row_overlap = any(not ({r for r, _ in a}.isdisjoint({r for r, _ in b}))
                      for i, a in enumerate(sources) for b in sources[i+1:])
    output_touch = any(_sets_touch(a, b)
                       for i, a in enumerate(destinations)
                       for b in destinations[i+1:])
    crossing = any(a & b for i, a in enumerate(sources)
                   for j, b in enumerate(destinations) if i != j)
    return row_overlap, output_touch, crossing


def _place_scene(rng, shapes, background, foreground, mode='any', allow_touch=True):
    max_width = max(_bbox_width(s) for s in shapes)
    max_height = max(_bbox_height(s) for s in shapes)
    total_cells = sum(map(len, shapes))
    for attempt in range(90):
        width = int(rng.integers(max(2 * max_width + 2, 8),
                                  min(30, 2 * max_width + 13) + 1))
        height = int(rng.integers(max(max_height + 2, 6),
                                   min(27, max(max_height + 10, 13)) + 1))
        if height * width <= 3 * total_cells:
            continue
        sources, destinations = [], []
        for shape in shapes:
            sh, sw = _bbox_height(shape), _bbox_width(shape)
            candidates = []
            for top in range(height - sh + 1):
                for left in range(width - 2 * sw + 1):
                    source = {(top + r, left + c) for r, c in shape}
                    if any(_sets_touch(source, old) for old in sources):
                        continue
                    dest = {(r, c + sw) for r, c in source}
                    if any(dest & old for old in destinations):
                        continue
                    candidates.append((source, dest))
            if not candidates:
                break
            if not sources:
                # A position near the left leaves room for the second object to
                # demonstrate close placements and simultaneous motion.
                if mode in ('touch', 'cross'):
                    lower = [p for p in candidates if min(c for _, c in p[0]) < width // 3]
                    if lower:
                        candidates = lower
            elif len(sources) == 1 and mode in ('touch', 'cross', 'overlap', 'separate'):
                if mode == 'touch':
                    candidates = [p for p in candidates if _sets_touch(p[1], destinations[0])]
                elif mode == 'cross':
                    candidates = [p for p in candidates
                                  if p[0] & destinations[0] or p[1] & sources[0]]
                elif mode == 'overlap':
                    old_rows = {r for r, _ in sources[0]}
                    candidates = [p for p in candidates
                                  if old_rows & {r for r, _ in p[0]}]
                else:
                    old_rows = {r for r, _ in sources[0]}
                    candidates = [p for p in candidates
                                  if old_rows.isdisjoint({r for r, _ in p[0]})]
            if not candidates:
                break
            source, dest = candidates[int(rng.integers(len(candidates)))]
            sources.append(source)
            destinations.append(dest)
        if len(sources) != len(shapes):
            continue
        _, output_touch, _ = _relationship(sources, destinations)
        if output_touch and not allow_touch:
            continue
        grid = np.full((height, width), background, dtype=np.int8)
        for source in sources:
            for r, c in source:
                grid[r, c] = foreground
        return grid
    raise RuntimeError('scene placement failed')


def _apply_transformation(input_grid, background, foreground):
    output = np.full(input_grid.shape, background, dtype=np.int8)
    destinations = set()
    for component in _components(input_grid == foreground, diagonal=True):
        width = _bbox_width(component)
        moved = {(r, c + width) for r, c in component}
        if any(c >= input_grid.shape[1] for _, c in moved) or moved & destinations:
            raise RuntimeError('invalid destination')
        destinations.update(moved)
    for r, c in destinations:
        output[r, c] = foreground
    return output


def _pair_colors(rng, parameters, index, split):
    if not parameters['vary_colors'] or (split == 'train' and index == 0):
        return parameters['colors']
    for _ in range(30):
        colors = tuple(map(int, rng.choice(10, size=2, replace=False)))
        if split != 'train' or colors != parameters['colors']:
            return colors
    raise RuntimeError('color selection failed')


def _make_pair(rng, split, index, parameters, touch_demo):
    background, foreground = _pair_colors(rng, parameters, index, split)
    if split == 'train' and index == 0:
        # The first demonstration always forces a diagonal-only join to move
        # by its combined width, next to an object with a different width.
        widths = [int(rng.integers(3, 5)), int(rng.choice((1, 2, 5)))]
        if widths[1] == widths[0]:
            widths[1] = 1
        shapes = [_make_shape(rng, widths[0], 'diagonal'),
                  _make_shape(rng, widths[1], 'branch')]
        mode = parameters['interaction'] if parameters['interaction'] in ('touch', 'cross') else str(rng.choice(('overlap', 'separate')))
        allow_touch = True
    elif split == 'train' and index == 1:
        width = int(rng.integers(3, 6))
        kind = 'hollow' if rng.random() < .7 else str(rng.choice(('branch', 'thick')))
        shapes = [_make_shape(rng, width, kind)]
        if rng.random() < .5:
            other_width = int(rng.integers(1, 5))
            shapes.append(_make_shape(rng, other_width, str(rng.choice(('branch', 'diagonal')))))
        mode = str(rng.choice(('overlap', 'separate', 'any'))) if len(shapes) > 1 else 'any'
        allow_touch = True
    else:
        if split == 'test':
            count = int(rng.choice((1, 2, 2, 3)))
            primary_width = int(rng.integers(4, 8))
            kinds = ('diagonal', 'hollow', 'thick', 'branch')
            kind = str(rng.choice(kinds, p=(.27, .30, .25, .18)))
            shapes = [_make_shape(rng, primary_width, kind)]
            for extra_index in range(count - 1):
                if count == 2 and extra_index == 0 and rng.random() < .28:
                    width = primary_width
                else:
                    width = int(rng.integers(1, min(6, primary_width + 1)))
                shapes.append(_make_shape(rng, width, str(rng.choice(kinds))))
            if count > 1 and touch_demo and _bbox_width(shapes[0]) != _bbox_width(shapes[1]) and rng.random() < .45:
                mode = 'touch'
            else:
                mode = str(rng.choice(('overlap', 'separate', 'any')))
            allow_touch = touch_demo
        else:
            count = int(rng.choice((1, 2, 2, 3)))
            kinds = ('diagonal', 'hollow', 'thick', 'branch')
            shapes = []
            for _ in range(count):
                width = int(rng.integers(1, 6))
                shapes.append(_make_shape(rng, width, str(rng.choice(kinds))))
            mode = str(rng.choice(('overlap', 'separate', 'any'))) if count > 1 else 'any'
            allow_touch = True
    grid = _place_scene(rng, shapes, background, foreground, mode, allow_touch)
    return {'input': grid, 'output': _apply_transformation(grid, background, foreground)}


def _analyze_pair(input_grid, output_grid):
    values, counts = np.unique(input_grid, return_counts=True)
    assert len(values) == 2 and counts[0] != counts[1]
    background = int(values[int(np.argmax(counts))])
    foreground = int(values[int(np.argmin(counts))])
    assert set(map(int, np.unique(output_grid))) == {background, foreground}
    components = _components(input_grid == foreground, diagonal=True)
    assert 1 <= len(components) <= 3
    destinations = []
    expected = np.full(input_grid.shape, background, dtype=np.int8)
    for component in components:
        width = _bbox_width(component)
        moved = {(r, c + width) for r, c in component}
        assert all(c < input_grid.shape[1] for _, c in moved)
        assert all(not moved & previous for previous in destinations)
        destinations.append(moved)
        for r, c in moved:
            expected[r, c] = foreground
    assert np.array_equal(output_grid, expected)
    assert not np.array_equal(input_grid, output_grid)
    rows_overlap, outputs_touch, crossing = _relationship(components, destinations)
    return {'background': background, 'foreground': foreground,
            'components': components, 'widths': [_bbox_width(c) for c in components],
            'row_overlap': rows_overlap, 'outputs_touch': outputs_touch,
            'crossing': crossing,
            'diagonal': any(len(_components_of_cells(c, False)) > 1 for c in components),
            'hole': any(_has_hole(c) for c in components)}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    train, test = puzzle['train'], puzzle['test']
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen_inputs, seen_pairs, analyses = set(), set(), []
    for pair in train + test:
        assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
        inp, out = pair['input'], pair['output']
        for grid in (inp, out):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
            assert np.issubdtype(grid.dtype, np.integer)
            assert grid.min() >= 0 and grid.max() <= 9
        assert inp.shape == out.shape
        key, pair_key = _grid_key(inp), _pair_key(pair)
        assert key not in seen_inputs and pair_key not in seen_pairs
        assert pair_key not in _SOURCE_PAIR_KEYS
        seen_inputs.add(key)
        seen_pairs.add(pair_key)
        analyses.append(_analyze_pair(inp, out))
    train_info, test_info = analyses[:len(train)], analyses[len(train):]
    first = train_info[0]
    assert len(first['components']) == 2 and len(set(first['widths'])) == 2
    assert any(len(_components_of_cells(c, False)) > 1 and _bbox_width(c) >= 3
               for c in first['components'])
    assert len({w for a in train_info for w in a['widths']}) >= 2
    if len({(a['background'], a['foreground']) for a in analyses}) > 1:
        assert len({(a['background'], a['foreground']) for a in train_info}) > 1
    if any(a['outputs_touch'] for a in test_info):
        assert any(a['outputs_touch'] for a in train_info)
    for analysis in test_info:
        assert any(width >= 4 for width in analysis['widths'])
    # Inputs must provide sufficient background to identify both roles.
    for pair in train + test:
        values, counts = np.unique(pair['input'], return_counts=True)
        assert len(values) == 2 and max(counts) > 2 * min(counts)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _resolve_counts(int(seed), rng, num_train, num_test)
    base_colors = sample_puzzle_parameters(rng)
    parameters = {'colors': (base_colors['background'], base_colors['foreground']),
                  'vary_colors': bool(rng.random() < .45),
                  'interaction': str(rng.choice(('none', 'none', 'touch', 'cross')))}
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, count in (('train', train_count), ('test', test_count)):
        for index in range(count):
            for _ in range(90):
                pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                try:
                    pair = _make_pair(pair_rng, split, index, parameters,
                                      any(_analyze_pair(p['input'], p['output'])['outputs_touch']
                                          for p in puzzle['train']))
                except (RuntimeError, AssertionError):
                    continue
                key = _pair_key(pair)
                if _grid_key(pair['input']) in seen or key in _SOURCE_PAIR_KEYS:
                    continue
                seen.add(_grid_key(pair['input']))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError(f'could not construct {split}[{index}]')
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple(tuple(_pair_key(p) for p in puzzle[split]) for split in ('train', 'test'))


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    for seed, nt, ns in ((700, 2, 1), (701, 10, 1), (702, 10, 4)):
        p = generate_puzzle(seed, nt, ns)
        assert (len(p['train']), len(p['test'])) == (nt, ns)
    for seed, args in enumerate(((None, 1), (None, 4), (2, None), (10, None)), 800):
        p = generate_puzzle(seed, *args)
        validate_generated_puzzle(p)
    assert _puzzle_key(generate_puzzle(950, 6, 2)) == _puzzle_key(generate_puzzle(950, 6, 2))
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(1200, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    pairs = [pair for p in puzzles for pair in p['train'] + p['test']]
    info = [_analyze_pair(p['input'], p['output']) for p in pairs]
    assert len({p['input'].shape for p in pairs}) >= 20
    assert len({len(a['components']) for a in info}) == 3
    assert any(a['hole'] for a in info)
    assert any(a['row_overlap'] for a in info)
    assert any(a['crossing'] for a in info)
    assert any(a['outputs_touch'] for a in info)
    assert any(len({(a['background'], a['foreground'])
                    for a in [_analyze_pair(x['input'], x['output'])
                              for x in p['train'] + p['test']]}) > 1 for p in puzzles)
    train_info = [[_analyze_pair(pair['input'], pair['output']) for pair in p['train']]
                  for p in puzzles]
    test_info = [[_analyze_pair(pair['input'], pair['output']) for pair in p['test']]
                 for p in puzzles]
    assert all(info[0]['diagonal'] and len(set(info[0]['widths'])) == 2
               for info in train_info)
    assert sum(any(a['hole'] for a in info) for info in train_info) >= 20
    assert sum(any(a['outputs_touch'] for a in info) for info in train_info) >= 8
    assert sum(any(a['crossing'] for a in info) for info in train_info) >= 8
    assert sum(a['hole'] for info in test_info for a in info) >= 15
    assert sum(a['diagonal'] for info in test_info for a in info) >= 15
    assert sum(a['row_overlap'] for info in test_info for a in info) >= 10
    assert sum(a['outputs_touch'] for info in test_info for a in info) >= 5
    assert sum(a['crossing'] for info in test_info for a in info) >= 5
    assert {len(a['components']) for info in test_info for a in info} == {1, 2, 3}
    assert sum(len(a['widths']) == 2 and len(set(a['widths'])) == 1
               for info in test_info for a in info) >= 5
    assert len({g.shape for p in puzzles for pair in p['train'] + p['test']
                for g in (pair['input'],)}) >= 50
    assert all(len({pair['input'].shape for pair in p['train'] + p['test']}) > 1
               for p in puzzles)
    assert all(len({len(a['components']) for a in ti + si}) > 1
               for ti, si in zip(train_info, test_info))
