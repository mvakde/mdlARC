import numpy as np


_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))

_SOURCE_ROWS = (
    (
        (
            "0000000000",
            "0555555000",
            "0555555000",
            "0005005000",
            "0000555000",
            "0000555000",
            "0000500000",
            "0000000000",
        ),
        (
            "0000000000",
            "0882882000",
            "0882882000",
            "0002002000",
            "0000288000",
            "0000288000",
            "0000200000",
            "0000000000",
        ),
    ),
    (
        (
            "00000000000",
            "00550000000",
            "00555550000",
            "00055500000",
            "00055555000",
            "00050055000",
            "00000555000",
            "00000000000",
            "00000000000",
        ),
        (
            "00000000000",
            "00880000000",
            "00882220000",
            "00028800000",
            "00028888000",
            "00020088000",
            "00000222000",
            "00000000000",
            "00000000000",
        ),
    ),
    (
        (
            "000000000",
            "055555000",
            "000055000",
            "000500000",
            "000555000",
            "000555000",
            "000000000",
            "000000000",
        ),
        (
            "000000000",
            "022288000",
            "000088000",
            "000200000",
            "000288000",
            "000288000",
            "000000000",
            "000000000",
        ),
    ),
    (
        (
            "00550555000",
            "00550050000",
            "05555550000",
            "00005550000",
            "00055500000",
            "00000550000",
            "00000550000",
            "00000000000",
        ),
        (
            "00880222000",
            "00880020000",
            "02228820000",
            "00008820000",
            "00022200000",
            "00000880000",
            "00000880000",
            "00000000000",
        ),
    ),
)


def _piece_cells(
    kind: str,
    orientation: str,
    row: int,
    col: int,
) -> frozenset[tuple[int, int]]:
    if kind == "square":
        return frozenset(
            (row + dr, col + dc)
            for dr in range(2)
            for dc in range(2)
        )

    if orientation == "H":
        return frozenset((row, col + dc) for dc in range(3))

    return frozenset((row + dr, col) for dr in range(3))


def _enumerate_candidates(
    foreground: frozenset[tuple[int, int]],
) -> list[tuple[str, str, frozenset[tuple[int, int]]]]:
    if not foreground:
        return []

    min_r = min(r for r, _ in foreground)
    max_r = max(r for r, _ in foreground)
    min_c = min(c for _, c in foreground)
    max_c = max(c for _, c in foreground)

    candidates = []

    for row in range(min_r, max_r + 1):
        for col in range(min_c, max_c + 1):
            for kind, orientation in (
                ("square", "S"),
                ("bar", "H"),
                ("bar", "V"),
            ):
                cells = _piece_cells(kind, orientation, row, col)
                if cells.issubset(foreground):
                    candidates.append((kind, orientation, cells))

    return candidates


def _exact_covers(
    foreground: frozenset[tuple[int, int]],
    limit: int = 2,
) -> list[tuple[tuple[str, str, frozenset[tuple[int, int]]], ...]]:
    foreground = frozenset(foreground)
    if not foreground:
        return []

    candidates = _enumerate_candidates(foreground)
    by_cell = {cell: [] for cell in foreground}

    for index, candidate in enumerate(candidates):
        for cell in candidate[2]:
            by_cell[cell].append(index)

    answers = []

    def search(
        remaining: frozenset[tuple[int, int]],
        chosen: list[tuple[str, str, frozenset[tuple[int, int]]]],
    ) -> None:
        if len(answers) >= limit:
            return

        if not remaining:
            answers.append(tuple(chosen))
            return

        def cell_key(cell: tuple[int, int]) -> tuple[int, tuple[int, int]]:
            option_count = sum(
                candidates[index][2].issubset(remaining)
                for index in by_cell[cell]
            )
            return option_count, cell

        cell = min(remaining, key=cell_key)

        for index in by_cell[cell]:
            cells = candidates[index][2]
            if cells.issubset(remaining):
                search(
                    remaining - cells,
                    chosen + [candidates[index]],
                )
                if len(answers) >= limit:
                    return

    search(foreground, [])
    return answers


def _component_count(cells: frozenset[tuple[int, int]], diagonal=False) -> int:
    unseen = set(cells)
    count = 0

    while unseen:
        count += 1
        stack = [unseen.pop()]

        while stack:
            row, col = stack.pop()
            directions = (_NEIGHBORS + ((-1, -1), (-1, 1), (1, -1), (1, 1))
                          if diagonal else _NEIGHBORS)
            for dr, dc in directions:
                neighbor = (row + dr, col + dc)
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)

    return count


def _piece_contact_edges(
    cover: tuple[tuple[str, str, frozenset[tuple[int, int]]], ...],
) -> int:
    edges = 0

    for first_index in range(len(cover)):
        first = cover[first_index][2]

        for second_index in range(first_index + 1, len(cover)):
            second = cover[second_index][2]

            if any(
                (row + dr, col + dc) in second
                for row, col in first
                for dr, dc in _NEIGHBORS
            ):
                edges += 1

    return edges



def _connected_group(rng, shapes, density):
    """Place an edge-connected group without overlapping its pieces."""
    placed = []
    occupied = set()
    for kind, orientation in shapes:
        if not placed:
            cells = _piece_cells(kind, orientation, 0, 0)
        else:
            rows = [r for r, _ in occupied]
            cols = [c for _, c in occupied]
            options = []
            for row in range(min(rows) - 3, max(rows) + 4):
                for col in range(min(cols) - 3, max(cols) + 4):
                    candidate = _piece_cells(kind, orientation, row, col)
                    if candidate & occupied:
                        continue
                    contacts = sum(
                        any((r + dr, c + dc) in other for r, c in candidate
                            for dr, dc in _NEIGHBORS)
                        for _, _, other in placed
                    )
                    if contacts:
                        options.append((contacts, candidate))
            assert options
            scores = np.array([v[0] for v in options], dtype=float)
            weights = scores ** (2 if density == 'dense' else -2)
            weights /= weights.sum()
            cells = options[int(rng.choice(len(options), p=weights))][1]
        placed.append((kind, orientation, cells))
        occupied.update(cells)
    return placed


def _translate(pieces, dr, dc):
    return [(kind, orientation,
             frozenset((r + dr, c + dc) for r, c in cells))
            for kind, orientation, cells in pieces]


def _diagonal_contacts(first, second):
    return sum((r + dr, c + dc) in second for r, c in first
               for dr in (-1, 1) for dc in (-1, 1))


def _layout_once(rng, shapes, group_count, style):
    order = [shapes[int(i)] for i in rng.permutation(len(shapes))]
    if group_count == 1:
        groups = [order]
    else:
        cuts = sorted(int(x) for x in rng.choice(
            np.arange(1, len(order)), size=group_count - 1, replace=False))
        groups = [order[a:b] for a, b in zip([0] + cuts, cuts + [len(order)])]
    pieces = _connected_group(rng, groups[0], style)
    for group in groups[1:]:
        local = _connected_group(rng, group, style)
        occupied = {cell for _, _, cells in pieces for cell in cells}
        local_cells = {cell for _, _, cells in local for cell in cells}
        row_min = min(r for r, _ in occupied)
        row_max = max(r for r, _ in occupied)
        col_min = min(c for _, c in occupied)
        col_max = max(c for _, c in occupied)
        height = max(r for r, _ in local_cells) - min(r for r, _ in local_cells) + 1
        width = max(c for _, c in local_cells) - min(c for _, c in local_cells) + 1
        shifts = []
        for dr in range(row_min - height - 4, row_max + 6):
            for dc in range(col_min - width - 4, col_max + 6):
                shifted = {(r + dr, c + dc) for r, c in local_cells}
                if shifted & occupied:
                    continue
                if any((r + d_r, c + d_c) in occupied
                       for r, c in shifted for d_r, d_c in _NEIGHBORS):
                    continue
                corners = _diagonal_contacts(occupied, shifted)
                if style == 'corner' and not corners:
                    continue
                if style == 'separated' and corners:
                    continue
                both = occupied | shifted
                span_h = max(r for r, _ in both) - min(r for r, _ in both) + 1
                span_w = max(c for _, c in both) - min(c for _, c in both) + 1
                if span_h <= 25 and span_w <= 25:
                    shifts.append((dr, dc, span_h * span_w))
        if not shifts:
            return None
        # Choose near and far placements; their bounding boxes can overlap.
        shifts.sort(key=lambda x: x[2])
        if rng.random() < .70:
            shifts = shifts[:max(1, len(shifts) // 3)]
        dr, dc, _ = shifts[int(rng.integers(len(shifts)))]
        pieces.extend(_translate(local, dr, dc))
    return pieces


def _fallback_layout(shapes):
    # Independent isolated pieces guarantee a finite construction path.
    pieces = []
    for index, (kind, orientation) in enumerate(shapes):
        pieces.append((kind, orientation,
                       _piece_cells(kind, orientation,
                                    5 * (index // 4), 5 * (index % 4))))
    return pieces


def _find_layout(rng, shapes, group_count, style):
    for _ in range(160):
        pieces = _layout_once(rng, shapes, group_count, style)
        if pieces is None:
            continue
        foreground = frozenset(cell for _, _, cells in pieces for cell in cells)
        rows = [r for r, _ in foreground]
        cols = [c for _, c in foreground]
        if max(rows) - min(rows) + 1 > 30 or max(cols) - min(cols) + 1 > 30:
            continue
        covers = _exact_covers(foreground, limit=2)
        if len(covers) == 1:
            return foreground, covers[0]
    pieces = _fallback_layout(shapes)
    foreground = frozenset(cell for _, _, cells in pieces for cell in cells)
    covers = _exact_covers(foreground, limit=2)
    assert len(covers) == 1
    return foreground, covers[0]


def _trimmed_signature(mask):
    points = np.argwhere(mask)
    low = points.min(axis=0)
    high = points.max(axis=0) + 1
    crop = mask[low[0]:high[0], low[1]:high[1]]
    return crop.shape, np.packbits(crop.reshape(-1)).tobytes()


def _array_key(array):
    return array.shape, np.asarray(array, dtype=np.int16).tobytes()


def _source_pair_keys():
    result = set()
    for input_rows, output_rows in _SOURCE_ROWS:
        arrays = [np.array([[int(v) for v in row] for row in rows],
                           dtype=np.int8)
                  for rows in (input_rows, output_rows)]
        result.add((_array_key(arrays[0]), _array_key(arrays[1])))
    return result


_SOURCE_PAIR_KEYS = _source_pair_keys()


def _resolve_counts(rng, num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10),
                                   ('num_test', num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
            raise ValueError(f'{name} is outside its legal range')
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _draw_counts(rng, regime, index):
    # Training includes two mixed-role anchors. Other pairs sample both
    # primitive counts independently, including zero for one role.
    if regime == 'source_train' and index < 2:
        squares = int(rng.integers(1, 5))
        bars = int(rng.integers(1, 5))
    else:
        total = int(rng.integers(1, 11))
        if regime == 'target_test' and rng.random() < .42:
            total = int(rng.integers(6, 11))
        squares = int(rng.integers(total + 1))
        bars = total - squares
    return squares, bars


def _draw_shapes(rng, squares, bars, orientation_mode):
    if orientation_mode == 'H':
        horizontal = bars
    elif orientation_mode == 'V':
        horizontal = 0
    else:
        horizontal = int(rng.integers(bars + 1))
    return ([('square', 'S')] * squares
            + [('bar', 'H')] * horizontal
            + [('bar', 'V')] * (bars - horizontal))


def _render_pair(rng, foreground, cover, square_color, bar_color,
                 input_palette):
    rows = [r for r, _ in foreground]
    cols = [c for _, c in foreground]
    min_r, min_c = min(rows), min(cols)
    span_h = max(rows) - min_r + 1
    span_w = max(cols) - min_c + 1
    size_mode = int(rng.choice(3, p=[.35, .43, .22]))
    if size_mode == 0:
        pad_h = int(rng.integers(0, 4))
        pad_w = int(rng.integers(0, 4))
    elif size_mode == 1:
        pad_h = int(rng.integers(2, 10))
        pad_w = int(rng.integers(2, 10))
    else:
        pad_h = int(rng.integers(8, 25))
        pad_w = int(rng.integers(8, 25))
    height = min(30, span_h + pad_h)
    width = min(30, span_w + pad_w)
    if size_mode == 2 and rng.random() < .3:
        height = 30
    if size_mode == 2 and rng.random() < .3:
        width = 30
    # A large background majority makes its role visible, even when a piece
    # reaches the border. Expand only the amount necessary for this.
    while height * width <= 2 * len(foreground) and (height < 30 or width < 30):
        if (height <= width and height < 30) or width == 30:
            height += 1
        else:
            width += 1
    top = int(rng.integers(height - span_h + 1))
    left = int(rng.integers(width - span_w + 1))
    if rng.random() < .24:
        top = int(rng.choice([0, height - span_h]))
    if rng.random() < .24:
        left = int(rng.choice([0, width - span_w]))
    if input_palette is None:
        possible = [v for v in range(10) if v not in (square_color, bar_color)]
        background, neutral = [int(v) for v in rng.choice(possible, 2,
                                                           replace=False)]
    else:
        background, neutral = input_palette
    input_grid = np.full((height, width), background, dtype=np.int8)
    output_grid = input_grid.copy()
    for row, col in foreground:
        input_grid[row - min_r + top, col - min_c + left] = neutral
    for kind, _, cells in cover:
        color = square_color if kind == 'square' else bar_color
        for row, col in cells:
            output_grid[row - min_r + top, col - min_c + left] = color
    return {'input': input_grid, 'output': output_grid}


def _make_pair(rng, regime, index, square_color, bar_color,
               input_palette, orientation_mode, require_bar=False):
    squares, bars = _draw_counts(rng, regime, index)
    if require_bar and bars == 0:
        squares -= 1
        bars = 1
    shapes = _draw_shapes(rng, squares, bars, orientation_mode)
    maximum = min(4, len(shapes))
    group_count = int(rng.choice(np.arange(1, maximum + 1),
                                 p=np.array([.42, .32, .18, .08][:maximum])
                                 / sum([.42, .32, .18, .08][:maximum])))
    style = str(rng.choice(['loose', 'dense', 'corner', 'separated'],
                           p=[.32, .30, .20, .18]))
    if group_count == 1 and style in ('corner', 'separated'):
        style = 'loose' if rng.random() < .5 else 'dense'
    foreground, cover = _find_layout(rng, shapes, group_count, style)
    return _render_pair(rng, foreground, cover, square_color, bar_color,
                        input_palette)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    square_color, bar_color = [int(x) for x in rng.choice(10, 2, replace=False)]
    palette = None
    if rng.random() < .45:
        available = [v for v in range(10) if v not in (square_color, bar_color)]
        palette = tuple(int(x) for x in rng.choice(available, 2,
                                                   replace=False))
    orientation_policy = str(rng.choice(['free', 'hold_H', 'hold_V'],
                                        p=[.76, .12, .12]))
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max,
                              size=num_train + num_test, dtype=np.int64)
    puzzle = {'train': [], 'test': []}
    seen_inputs = set()
    seen_crops = set()
    for split, count, offset in (('train', num_train, 0),
                                 ('test', num_test, num_train)):
        regime = 'source_train' if split == 'train' else 'target_test'
        for index in range(count):
            pair_rng = np.random.default_rng(int(pair_seeds[offset + index]))
            for attempt in range(120):
                if orientation_policy == 'free':
                    orientation = str(pair_rng.choice(['H', 'V', 'mixed'],
                                                       p=[.28, .28, .44]))
                elif split == 'train':
                    orientation = orientation_policy[-1]
                else:
                    orientation = str(pair_rng.choice(
                        ['H', 'V', 'mixed'], p=[.25, .25, .5]))
                    if index == 0:
                        orientation = 'V' if orientation_policy == 'hold_H' else 'H'
                pair = _make_pair(pair_rng, regime, index, square_color,
                                  bar_color, palette, orientation,
                                  orientation_policy != 'free'
                                  and split == 'test' and index == 0)
                key = _array_key(pair['input'])
                pair_key = (key, _array_key(pair['output']))
                background = int(np.unique(pair['input'], return_counts=True)[0][
                    np.argmax(np.unique(pair['input'], return_counts=True)[1])])
                crop = _trimmed_signature(pair['input'] != background)
                if (key not in seen_inputs and crop not in seen_crops
                        and pair_key not in _SOURCE_PAIR_KEYS):
                    seen_inputs.add(key)
                    seen_crops.add(crop)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('failed to create a distinct pair')
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    input_grid, output_grid = pair['input'], pair['output']
    assert input_grid.shape == output_grid.shape
    values, counts = np.unique(input_grid, return_counts=True)
    assert len(values) == 2
    background = int(values[np.argmax(counts)])
    neutral = int(values[np.argmin(counts)])
    assert int(counts.max()) > int(counts.min())
    foreground = frozenset((int(r), int(c)) for r, c in np.argwhere(
        input_grid == neutral))
    covers = _exact_covers(foreground, limit=2)
    assert len(covers) == 1
    cover = covers[0]
    square_cells = set()
    bar_cells = set()
    horizontal = vertical = 0
    for kind, orientation, cells in cover:
        if kind == 'square':
            square_cells.update(cells)
        else:
            bar_cells.update(cells)
            horizontal += orientation == 'H'
            vertical += orientation == 'V'
    square_values = {int(output_grid[r, c]) for r, c in square_cells}
    bar_values = {int(output_grid[r, c]) for r, c in bar_cells}
    assert len(square_values) <= 1 and len(bar_values) <= 1
    assert np.all(output_grid[input_grid == background] == background)
    square_color = next(iter(square_values), None)
    bar_color = next(iter(bar_values), None)
    if square_color is not None:
        assert square_color not in (background, neutral)
    if bar_color is not None:
        assert bar_color not in (background, neutral, square_color)
    assert not np.array_equal(output_grid, input_grid)
    for r, c in square_cells:
        assert int(output_grid[r, c]) == square_color
    for r, c in bar_cells:
        assert int(output_grid[r, c]) == bar_color
    assert np.array_equal(input_grid == background,
                          output_grid == background)
    edge = any(r in (0, input_grid.shape[0] - 1) or
               c in (0, input_grid.shape[1] - 1) for r, c in foreground)
    diagonal = 0
    for i, (_, _, a) in enumerate(cover):
        for _, _, b in cover[i+1:]:
            diagonal += bool(_diagonal_contacts(a, b))
    return {'background': background, 'neutral': neutral,
            'square_color': square_color, 'bar_color': bar_color,
            'squares': len(square_cells) // 4, 'bars': len(bar_cells) // 3,
            'horizontal': horizontal, 'vertical': vertical,
            'components': _component_count(foreground),
            'eight_components': _component_count(foreground, diagonal=True),
            'contacts': _piece_contact_edges(cover),
            'diagonal': diagonal, 'edge': edge,
            'crop': _trimmed_signature(input_grid != background),
            'size': input_grid.shape}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    inputs, pairs, crops = set(), set(), set()
    analyses = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            key = _array_key(pair['input'])
            pair_key = (key, _array_key(pair['output']))
            assert key not in inputs and pair_key not in pairs
            assert pair_key not in _SOURCE_PAIR_KEYS
            analysis = _analyze_pair(pair)
            assert analysis['crop'] not in crops
            inputs.add(key)
            pairs.add(pair_key)
            crops.add(analysis['crop'])
            analyses[split].append(analysis)
    square_colors = {a['square_color'] for s in analyses.values() for a in s
                     if a['square_color'] is not None}
    bar_colors = {a['bar_color'] for s in analyses.values() for a in s
                  if a['bar_color'] is not None}
    assert len(square_colors) == len(bar_colors) == 1
    square_color = next(iter(square_colors))
    bar_color = next(iter(bar_colors))
    assert square_color != bar_color
    assert any(a['square_color'] == square_color for a in analyses['train'])
    assert any(a['bar_color'] == bar_color for a in analyses['train'])
    assert all(a['squares'] and a['bars'] for a in analyses['train'][:2])
    assert all(a['squares'] + a['bars'] >= 1
               for s in analyses.values() for a in s)


def _puzzle_key(puzzle):
    return tuple((split, tuple((_array_key(p['input']), _array_key(p['output']))
                               for p in puzzle[split]))
                 for split in ('train', 'test'))


def _predict_from_training(train_pairs, test_input):
    """Infer role colors from demonstrations, then solve an unseen silhouette."""
    role_colors = {'square': set(), 'bar': set()}
    for pair in train_pairs:
        grid = pair['input']
        values, counts = np.unique(grid, return_counts=True)
        background = int(values[np.argmax(counts)])
        cells = frozenset((int(r), int(c)) for r, c in np.argwhere(
            grid != background))
        covers = _exact_covers(cells, limit=2)
        assert len(covers) == 1
        for kind, _, part in covers[0]:
            role_colors[kind].update(int(pair['output'][r, c])
                                     for r, c in part)
    assert all(len(colors) == 1 for colors in role_colors.values())
    values, counts = np.unique(test_input, return_counts=True)
    background = int(values[np.argmax(counts)])
    cells = frozenset((int(r), int(c)) for r, c in np.argwhere(
        test_input != background))
    covers = _exact_covers(cells, limit=2)
    assert len(covers) == 1
    prediction = np.full(test_input.shape, background, dtype=np.int8)
    for kind, _, part in covers[0]:
        color = next(iter(role_colors[kind]))
        for r, c in part:
            prediction[r, c] = color
    return prediction


def self_test() -> None:
    fixed = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in fixed}) == 42
    assert len({len(p['train']) for p in fixed}) >= 4
    assert len({len(p['test']) for p in fixed}) >= 3
    analyses = []
    for puzzle in fixed:
        validate_generated_puzzle(puzzle)
        for pair in puzzle['test']:
            prediction = _predict_from_training(
                [{'input': p['input'].copy(), 'output': p['output'].copy()}
                 for p in puzzle['train']], pair['input'].copy())
            assert np.array_equal(prediction, pair['output'])
        analyses.extend(_analyze_pair(pair) for split in ('train', 'test')
                       for pair in puzzle[split])
    assert len({a['size'] for a in analyses}) >= 30
    assert len({a['crop'] for a in analyses}) >= 150
    assert len({(a['squares'], a['bars']) for a in analyses}) >= 20
    assert {a['components'] for a in analyses} >= {1, 2, 3}
    assert any(a['edge'] for a in analyses)
    assert any(a['eight_components'] < a['components'] for a in analyses)
    assert any(a['contacts'] >= 7 for a in analyses)
    assert any(a['contacts'] <= 2 for a in analyses)
    assert any(a['horizontal'] and not a['vertical'] for a in analyses)
    assert any(a['vertical'] and not a['horizontal'] for a in analyses)
    assert any(a['squares'] > a['bars'] for a in analyses)
    assert any(a['bars'] > a['squares'] for a in analyses)
    assert any(a['squares'] == 0 for a in analyses)
    assert any(a['bars'] == 0 for a in analyses)
    assert any(max(a['size']) == 30 for a in analyses)
    assert _puzzle_key(generate_puzzle(8675309, 7, 3)) == _puzzle_key(
        generate_puzzle(8675309, 7, 3))
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(1000 + train_count * 10 + test_count,
                            train_count, test_count)
        assert (len(p['train']), len(p['test'])) == (train_count, test_count)
    for i, (train_count, test_count) in enumerate(
            ((None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(2000 + i, train_count, test_count)
        if train_count is not None:
            assert len(p['train']) == train_count
        if test_count is not None:
            assert len(p['test']) == test_count
    for kwargs in ({'num_train': 1}, {'num_train': 11}, {'num_test': 0},
                   {'num_test': 5}, {'num_train': 2, 'num_test': 2},
                   {'num_train': 3, 'num_test': 4}):
        try:
            generate_puzzle(999, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f'illegal counts accepted: {kwargs}')
