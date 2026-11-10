import numpy as np


def _rows(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[int(ch) for ch in row] for row in rows], dtype=np.int8)


_SOURCE_PAIRS = (
    (
        _rows((
            "8888888888888888",
            "8888888888888888",
            "8888888888888888",
            "8888888888888888",
            "8888888888888888",
            "8888888888888888",
            "8888888888888888",
            "8888888888888888",
            "8888888888888888",
            "8888888888888888",
            "8338888888888888",
            "3223888888888888",
            "3242388888888888",
            "3242388888888888",
            "3222388888888888",
            "3333888888888888",
        )),
        _rows((
            "8888888888888888",
            "8888888888888888",
            "8888888888888888",
            "8888888888888888",
            "8888888888888888",
            "8888888888888888",
            "8888888888888888",
            "8888888888888888",
            "8888888888888888",
            "8448888888888888",
            "4884888888888888",
            "8888488888888888",
            "8888848888888888",
            "8888848888888888",
            "8888848888888888",
            "8888488888888888",
        )),
    ),
    (
        _rows((
            "8388888",
            "8338888",
            "8313888",
            "8331388",
            "8313888",
            "8338888",
            "8388888",
        )),
        _rows((
            "1818888",
            "1881888",
            "1888188",
            "1888818",
            "1888188",
            "1881888",
            "1818888",
        )),
    ),
    (
        _rows((
            "88488",
            "88448",
            "84448",
            "84348",
            "84448",
        )),
        _rows((
            "83838",
            "83883",
            "38883",
            "38883",
            "38883",
        )),
    ),
)


def _coerce_count(
    name: str,
    value: int | None,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}, inclusive"
        )
    return result


def _coerce_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be a non-negative integer")
    result = int(seed)
    if result < 0:
        raise ValueError("seed must be a non-negative integer")
    return result


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _coerce_count("num_train", num_train, 2, 10)
    test_count = _coerce_count("num_test", num_test, 1, 4)

    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        minimum_train = test_count + 1
        number_of_choices = 11 - minimum_train
        train_count = minimum_train + (seed // 4) % number_of_choices
    elif train_count is None:
        assert test_count is not None
        train_count = int(rng.integers(test_count + 1, 11))
    elif test_count is None:
        maximum_test = min(4, train_count - 1)
        if maximum_test < 1:
            raise ValueError("no legal test count exists for num_train")
        test_count = int(rng.integers(1, maximum_test + 1))

    assert train_count is not None
    assert test_count is not None

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def _dilate4(mask: np.ndarray) -> np.ndarray:
    result = mask.copy()
    result[1:, :] |= mask[:-1, :]
    result[:-1, :] |= mask[1:, :]
    result[:, 1:] |= mask[:, :-1]
    result[:, :-1] |= mask[:, 1:]
    return result


def _contour(mask: np.ndarray) -> np.ndarray:
    return _dilate4(mask) & ~mask


def _diagonal_only_neighbors(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    diagonal = (
        padded[:-2, :-2]
        | padded[:-2, 2:]
        | padded[2:, :-2]
        | padded[2:, 2:]
    )
    return diagonal & ~_dilate4(mask)


def _depth_map(mask: np.ndarray) -> np.ndarray:
    depth = np.full(mask.shape, -1, dtype=np.int16)
    remaining = mask.copy()
    level = 0

    while np.any(remaining):
        padded = np.pad(remaining, 1, constant_values=False)
        interior = (
            remaining
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
        boundary = remaining & ~interior
        assert np.any(boundary)
        depth[boundary] = level
        remaining[boundary] = False
        level += 1

    return depth


def _is_connected(mask: np.ndarray) -> bool:
    positions = np.argwhere(mask)
    if positions.size == 0:
        return False

    height, width = mask.shape
    start = tuple(int(x) for x in positions[0])
    visited = np.zeros(mask.shape, dtype=bool)
    visited[start] = True
    queue = [start]
    head = 0

    while head < len(queue):
        row, col = queue[head]
        head += 1
        for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nrow = row + drow
            ncol = col + dcol
            if (
                0 <= nrow < height
                and 0 <= ncol < width
                and mask[nrow, ncol]
                and not visited[nrow, ncol]
            ):
                visited[nrow, ncol] = True
                queue.append((nrow, ncol))

    return bool(np.all(visited[mask]))


def _is_hole_free(mask: np.ndarray) -> bool:
    background = ~mask
    if not np.any(background):
        return False

    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    queue: list[tuple[int, int]] = []

    boundary_positions = []
    for col in range(width):
        boundary_positions.append((0, col))
        boundary_positions.append((height - 1, col))
    for row in range(height):
        boundary_positions.append((row, 0))
        boundary_positions.append((row, width - 1))

    for row, col in boundary_positions:
        if background[row, col] and not visited[row, col]:
            visited[row, col] = True
            queue.append((row, col))

    head = 0
    while head < len(queue):
        row, col = queue[head]
        head += 1
        for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nrow = row + drow
            ncol = col + dcol
            if (
                0 <= nrow < height
                and 0 <= ncol < width
                and background[nrow, ncol]
                and not visited[nrow, ncol]
            ):
                visited[nrow, ncol] = True
                queue.append((nrow, ncol))

    return bool(np.all(visited[background]))


def _crop(mask: np.ndarray) -> np.ndarray:
    positions = np.argwhere(mask)
    if not len(positions):
        raise ValueError('empty footprint')
    low = positions.min(axis=0)
    high = positions.max(axis=0) + 1
    return mask[low[0]:high[0], low[1]:high[1]].copy()


def _shape(rng: np.random.Generator, min_depth: int, small: bool) -> np.ndarray:
    """Compose a footprint before canvas placement or color assignment."""
    for _ in range(180):
        mode = int(rng.integers(4))
        upper = 9 if small else 17
        if mode == 0:  # Independently changing row spans.
            height = int(rng.integers(3 + 2 * (min_depth == 2), upper + 1))
            width = int(rng.integers(3 + 2 * (min_depth == 2), min(upper, 13) + 1))
            mask = np.zeros((height, width + 8), dtype=bool)
            center = width // 2 + 4
            span = int(rng.integers(3, width + 1))
            for row in range(height):
                center = int(np.clip(center + int(rng.integers(-1, 2)), width // 2, width // 2 + 8))
                span = int(np.clip(span + int(rng.integers(-2, 3)), 3, width))
                left = center - span // 2
                mask[row, left:left + span] = True
        elif mode == 1:  # Thickened, turning or branching path.
            field = np.zeros((25, 25), dtype=bool)
            row = col = 12
            length = int(rng.integers(1, 7 if small else 15))
            for _step in range(length):
                field[row, col] = True
                direction = int(rng.integers(4))
                dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[direction]
                row = int(np.clip(row + dr, 4, 20))
                col = int(np.clip(col + dc, 4, 20))
            mask = _dilate4(field)
            if min_depth == 2:
                mask = _dilate4(mask)
        elif mode == 2:  # Mirrored taper, optionally wider at its waist.
            height = int(rng.integers(5, upper + 1))
            field = np.zeros((height, 21), dtype=bool)
            edge = int(rng.integers(1, 4))
            slope = int(rng.integers(1, 4))
            plateau = int(rng.integers(2, 7))
            for row in range(height):
                rise = min(row, height - 1 - row)
                span = edge + min(plateau, rise // slope + (rise > 0))
                if min_depth == 2:
                    span += 2
                field[row, 10 - span // 2:10 - span // 2 + span] = True
            mask = field
        else:  # Corner cuts and open notches in a block.
            height = int(rng.integers(5 if min_depth == 2 else 3, upper + 1))
            width = int(rng.integers(5 if min_depth == 2 else 3, upper + 1))
            mask = np.ones((height, width), dtype=bool)
            for r0, c0, rs, cs in ((0, 0, 1, 1), (0, width - 1, 1, -1),
                                   (height - 1, 0, -1, 1), (height - 1, width - 1, -1, -1)):
                cut = int(rng.integers(0, min(4, height // 3 + 1, width // 3 + 1)))
                for dr in range(cut):
                    for dc in range(cut - dr):
                        mask[r0 + rs * dr, c0 + cs * dc] = False
            if height >= 5 and width >= 5 and rng.random() < .75:
                side = int(rng.integers(4))
                opening = int(rng.integers(1, max(2, min(height, width) // 3 + 1)))
                if side < 2:
                    start = int(rng.integers(1, width - opening))
                    rows = slice(0, opening) if side == 0 else slice(height - opening, height)
                    mask[rows, start:start + opening] = False
                else:
                    start = int(rng.integers(1, height - opening))
                    cols = slice(0, opening) if side == 2 else slice(width - opening, width)
                    mask[start:start + opening, cols] = False
        mask = _crop(mask)
        if rng.random() < .5:
            mask = mask[:, ::-1]
        mask = np.rot90(mask, int(rng.integers(4))).copy()
        if (max(mask.shape) <= 21 and _is_connected(mask) and _is_hole_free(mask)
                and int(_depth_map(mask).max()) >= min_depth):
            return mask
    raise ValueError('could not make a solid silhouette')


def _place_mask(rng: np.random.Generator, local: np.ndarray, contact: int) -> np.ndarray:
    """Place the same footprint with zero, one, or two touched edges."""
    h, w = local.shape
    for _ in range(70):
        margins = [int(rng.integers(1, 7)) for _ in range(4)]  # top, bottom, left, right
        if contact:
            first = int(rng.integers(4))
            margins[first] = 0
            if contact == 2:
                second = int(rng.choice([side for side in range(4) if side != first]))
                margins[second] = 0
        if h + margins[0] + margins[1] > 30 or w + margins[2] + margins[3] > 30:
            continue
        mask = np.zeros((h + margins[0] + margins[1], w + margins[2] + margins[3]), dtype=bool)
        mask[margins[0]:margins[0] + h, margins[2]:margins[2] + w] = local
        if np.count_nonzero(mask) < mask.size // 2 and np.any(_contour(mask)):
            return mask
    raise ValueError('could not place a visibly distinct object')


def _components(mask: np.ndarray) -> int:
    remaining = mask.copy()
    count = 0
    while np.any(remaining):
        count += 1
        start = tuple(int(x) for x in np.argwhere(remaining)[0])
        remaining[start] = False
        queue = [start]
        for row, col in queue:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = row + dr, col + dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and remaining[nr, nc]:
                    remaining[nr, nc] = False
                    queue.append((nr, nc))
    return count


def _choose_accent(rng: np.random.Generator, eligible: np.ndarray, style: str) -> np.ndarray:
    positions = np.argwhere(eligible)
    if not len(positions):
        raise ValueError('no interior cells')
    result = np.zeros(eligible.shape, dtype=bool)
    if style == 'single':
        row, col = positions[int(rng.integers(len(positions)))]
        result[row, col] = True
    elif style == 'large':
        count = max(1, int(np.ceil(len(positions) * float(rng.uniform(.55, 1.0)))))
        for row, col in positions[rng.choice(len(positions), size=count, replace=False)]:
            result[row, col] = True
    elif style == 'separated':
        order = rng.permutation(len(positions))
        for index in order:
            row, col = positions[index]
            if not np.any(_dilate4(result)[row, col]):
                result[row, col] = True
            if np.count_nonzero(result) >= min(4, max(2, len(positions) // 3)):
                break
        if _components(result) < 2:
            raise ValueError('no separated accent sites')
    else:  # A connected irregular patch.
        row, col = positions[int(rng.integers(len(positions)))]
        result[row, col] = True
        target = min(len(positions), int(rng.integers(2, 8)))
        while np.count_nonzero(result) < target:
            frontier = np.argwhere(eligible & _dilate4(result) & ~result)
            if not len(frontier):
                break
            row, col = frontier[int(rng.integers(len(frontier)))]
            result[row, col] = True
    return result


def _build_pair(rng: np.random.Generator, mask: np.ndarray, colors: tuple[int, ...],
                accent_style: str) -> dict:
    background, shell, accent = colors[:3]
    middle = colors[3] if len(colors) == 4 else None
    depth = _depth_map(mask)
    eligible = mask & (depth >= (2 if middle is not None else 1))
    accent_mask = _choose_accent(rng, eligible, accent_style)
    input_grid = np.full(mask.shape, background, dtype=np.int8)
    input_grid[mask] = shell
    if middle is not None:
        middle_area = mask & (depth >= 1)
        depth_one = np.argwhere(mask & (depth == 1))
        if not len(depth_one):
            raise ValueError('middle layer is missing')
        # Intruding shell cells break a mechanical distance-layer shortcut.
        if len(depth_one) > 2 and rng.random() < .5:
            count = int(rng.integers(1, max(2, len(depth_one) // 3 + 1)))
            middle_area[tuple(depth_one[rng.choice(len(depth_one), count, replace=False)].T)] = False
        input_grid[middle_area] = middle
    input_grid[accent_mask] = accent
    output_grid = np.full(mask.shape, background, dtype=np.int8)
    output_grid[_contour(mask)] = accent
    return {'input': input_grid, 'output': output_grid}


def _grid_key(grid: np.ndarray) -> tuple:
    return tuple(grid.shape), grid.tobytes()


def _pair_key(pair: dict) -> tuple:
    return _grid_key(pair['input']), _grid_key(pair['output'])


def _matches_source_pair(pair: dict) -> bool:
    return any(pair['input'].shape == source_input.shape
               and np.array_equal(pair['input'], source_input)
               and np.array_equal(pair['output'], source_output)
               for source_input, source_output in _SOURCE_PAIRS)


def _analyze_pair(pair: dict) -> dict:
    grid, output = pair['input'], pair['output']
    counts = np.bincount(grid.ravel(), minlength=10)
    background = int(np.argmax(counts))
    assert counts[background] > grid.size // 2
    mask = grid != background
    assert np.any(mask) and _is_connected(mask) and _is_hole_free(mask)
    depth = _depth_map(mask)
    assert depth.max() >= 1
    boundary_colors = set(int(x) for x in np.unique(grid[mask & (depth == 0)]))
    assert len(boundary_colors) == 1
    shell = next(iter(boundary_colors))
    object_colors = set(int(x) for x in np.unique(grid[mask]))
    assert 2 <= len(object_colors) <= 3 and shell in object_colors
    inner = object_colors - {shell}
    if len(inner) == 1:
        accent = next(iter(inner))
        middle = None
        assert np.all(depth[grid == accent] >= 1)
    else:
        at_depth_one = set(int(x) for x in np.unique(grid[mask & (depth == 1)]))
        middle_options = inner & at_depth_one
        assert len(middle_options) == 1
        middle = next(iter(middle_options))
        accent = next(iter(inner - {middle}))
        assert np.all(depth[grid == accent] >= 2)
    ring = _contour(mask)
    assert np.any(ring)
    expected = np.full(grid.shape, background, dtype=grid.dtype)
    expected[ring] = accent
    assert np.array_equal(output, expected)
    assert np.any(_diagonal_only_neighbors(mask))
    assert not np.array_equal(grid, output)
    return {'background': background, 'shell': shell, 'middle': middle,
            'accent': accent, 'mask': mask, 'depth': depth,
            'accent_area': int(np.count_nonzero(grid == accent)),
            'accent_components': _components(grid == accent),
            'contacts': sum(bool(np.any(edge)) for edge in
                            (mask[0], mask[-1], mask[:, 0], mask[:, -1]))}


def _colors(rng: np.random.Generator, training: list[dict], layer_count: int,
            regime: str) -> tuple[int, ...]:
    sampled = [int(x) for x in rng.choice(10, size=4, replace=False)]
    if regime == 'test' and training and rng.random() < .28:
        anchor = training[int(rng.integers(len(training)))]
        sampled[1] = anchor['accent']
        sampled[2] = anchor['shell']
        available = [x for x in range(10) if x not in sampled[1:3]]
        sampled[0] = int(rng.choice(available))
        sampled[3] = int(rng.choice([x for x in available if x != sampled[0]]))
    return tuple(sampled[:3 + (layer_count == 3)])


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen_inputs, seen_pairs = set(), set()
    details = {'train': [], 'test': []}
    for regime in ('train', 'test'):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert pair['input'].shape == pair['output'].shape
            assert _grid_key(pair['input']) not in seen_inputs
            assert _pair_key(pair) not in seen_pairs
            seen_inputs.add(_grid_key(pair['input']))
            seen_pairs.add(_pair_key(pair))
            assert not _matches_source_pair(pair)
            details[regime].append(_analyze_pair(pair))
    train = details['train']
    assert {d['middle'] is None for d in train} == {True, False}
    assert any(d['accent_components'] > 1 for d in train)
    assert any(d['contacts'] == 0 for d in train)
    assert any(d['contacts'] > 0 for d in train)
    assert len({d['background'] for d in train}) >= 2
    assert len({d['shell'] for d in train}) >= 2
    assert len({d['accent'] for d in train}) >= 2
    # Layer types and clipping are demonstrated in training; literal colors
    # have no puzzle-level output mapping. Each input identifies its own roles.
    for pair in puzzle['test']:
        inferred = _analyze_pair({'input': pair['input'], 'output': pair['output']})
        assert inferred['accent'] in set(int(x) for x in np.unique(pair['input']))


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed_value = _coerce_seed(seed)
    rng = np.random.default_rng(seed_value)
    train_count, test_count = _resolve_counts(seed_value, rng, num_train, num_test)
    # No arbitrary latent color convention is supported by the source task.
    # The shared convention is the demonstrated geometric and role rule.
    layer_order = list(rng.permutation([2, 3]))
    clip_order = list(rng.permutation([0, 1]))
    puzzle = {'train': [], 'test': []}
    train_details = []
    seen = set()
    for regime, count in (('train', train_count), ('test', test_count)):
        for index in range(count):
            if regime == 'train' and index < 2:
                layers = layer_order[index]
                contact = clip_order[index]
                accent_style = 'separated' if layers == 2 else str(rng.choice(['single', 'connected', 'separated']))
            else:
                layers = 3 if rng.random() < .42 else 2
                contact = int(rng.choice([0, 1, 2], p=[.42, .36, .22]))
                accent_style = str(rng.choice(['single', 'connected', 'separated', 'large'],
                                              p=[.25, .25, .25, .25]))
            for attempt in range(300):
                try:
                    small = bool(rng.random() < (.55 if regime == 'test' else .35))
                    fallback = attempt >= 250
                    source_scale = (not fallback and layers == 2 and accent_style != 'separated'
                                    and rng.random() < (.22 if regime == 'test' else .08))
                    if fallback:
                        local = np.ones((7 + attempt % 4, 7 + (attempt // 4) % 4), dtype=bool)
                        if attempt % 2:
                            local[0, 2:4] = False
                        mask = _place_mask(rng, local, contact)
                    elif source_scale:
                        local = np.array([[0, 1, 0], [0, 1, 1], [1, 1, 1],
                                          [1, 1, 1], [1, 1, 1]], dtype=bool)
                        local = np.rot90(local, int(rng.integers(4))).copy()
                        mask = np.pad(local, ((0, 0), (1, 1)) if local.shape[0] == 5
                                      else ((1, 1), (0, 0)))
                    else:
                        local = _shape(rng, layers - 1, small)
                        mask = _place_mask(rng, local, contact)
                    colors = _colors(rng, train_details, layers, regime)
                    pair = _build_pair(rng, mask, colors, accent_style)
                    if _grid_key(pair['input']) in seen or _matches_source_pair(pair):
                        continue
                    detail = _analyze_pair(pair)
                    if regime == 'train' and index == 1:
                        first = train_details[0]
                        if any(detail[key] == first[key] for key in ('background', 'shell', 'accent')):
                            continue
                    break
                except (ValueError, AssertionError):
                    continue
            else:
                raise ValueError(f'could not construct pair {regime} {index}')
            puzzle[regime].append(pair)
            seen.add(_grid_key(pair['input']))
            if regime == 'train':
                train_details.append(detail)
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple((regime, tuple(_pair_key(pair) for pair in puzzle[regime]))
                 for regime in ('train', 'test'))


def _expect_value_error(**kwargs: int | None) -> None:
    try:
        generate_puzzle(123, **kwargs)
    except ValueError:
        return
    raise AssertionError(f'expected ValueError for {kwargs}')


def self_test() -> None:
    for args in ({}, {'num_train': 6, 'num_test': 3}, {'num_train': None, 'num_test': 1},
                 {'num_train': None, 'num_test': 4}, {'num_train': 2, 'num_test': None},
                 {'num_train': 10, 'num_test': None}, {'num_train': 2, 'num_test': 1},
                 {'num_train': 10, 'num_test': 1}, {'num_train': 10, 'num_test': 4}):
        first = generate_puzzle(777, **args)
        second = generate_puzzle(777, **args)
        assert _puzzle_key(first) == _puzzle_key(second)
        validate_generated_puzzle(first)
    for args in ({'num_train': 1}, {'num_train': 11}, {'num_test': 0}, {'num_test': 5},
                 {'num_train': 2, 'num_test': 2}, {'num_train': 3, 'num_test': 4}):
        _expect_value_error(**args)
    train_counts, test_counts, palettes, dimensions, layouts = set(), set(), set(), set(), set()
    accent_areas, contacts, component_counts, layer_counts = set(), set(), set(), set()
    small, broad, concave, three_layer_tests = 0, 0, 0, 0
    review_counts = {(split, feature): 0 for split in ('train', 'test') for feature in
                     ('single', 'separated', 'five_by_five', 'intrusion', 'row_notch',
                      'interior', 'two_edges', 'horizontal', 'vertical',
                      'three_layer_clipped_separated', 'notched_three_layer')}
    novel_test_accents = role_swapped_tests = 0
    all_keys = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        key = _puzzle_key(puzzle)
        assert key not in all_keys
        all_keys.add(key)
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        train_colors = {int(v) for pair in puzzle['train'] for v in np.unique(pair['input'])}
        train_shells = {_analyze_pair(pair)['shell'] for pair in puzzle['train']}
        train_accents = {_analyze_pair(pair)['accent'] for pair in puzzle['train']}
        for regime in ('train', 'test'):
            for pair in puzzle[regime]:
                detail = _analyze_pair(pair)
                mask = detail['mask']
                height, width = mask.shape
                has_middle = detail['middle'] is not None
                separated = detail['accent_components'] > 1
                row_notch = any(
                    int(np.ptp(np.flatnonzero(row))) + 1 != int(np.count_nonzero(row))
                    for row in mask if np.any(row)
                )
                intrusion = bool(np.any(mask & (detail['depth'] >= 1)
                                        & (pair['input'] == detail['shell'])))
                for feature, present in (
                    ('single', detail['accent_area'] == 1),
                    ('separated', separated),
                    ('five_by_five', (height, width) == (5, 5)),
                    ('intrusion', intrusion),
                    ('row_notch', row_notch),
                    ('interior', detail['contacts'] == 0),
                    ('two_edges', detail['contacts'] == 2),
                    ('horizontal', width > height),
                    ('vertical', height > width),
                    ('three_layer_clipped_separated', has_middle and detail['contacts'] > 0 and separated),
                    ('notched_three_layer', has_middle and row_notch),
                ):
                    review_counts[(regime, feature)] += bool(present)
                if regime == 'test':
                    novel_test_accents += detail['accent'] not in train_colors
                    role_swapped_tests += (detail['shell'] in train_accents
                                           and detail['accent'] in train_shells)
                palettes.add((detail['background'], detail['shell'], detail['accent']))
                dimensions.add(mask.shape)
                layouts.add((mask.shape, np.packbits(mask).tobytes()))
                accent_areas.add(detail['accent_area'])
                contacts.add(detail['contacts'])
                component_counts.add(detail['accent_components'])
                layer_counts.add(2 + (detail['middle'] is not None))
                small += max(mask.shape) <= 6
                broad += np.count_nonzero(mask) >= 50
                concave += bool(np.any(~mask & (np.cumsum(mask, axis=1) > 0) &
                                       (np.cumsum(mask[:, ::-1], axis=1)[:, ::-1] > 0)))
                three_layer_tests += regime == 'test' and detail['middle'] is not None
                expected = np.full(mask.shape, detail['background'], dtype=np.int8)
                expected[_contour(mask)] = detail['accent']
                assert np.array_equal(pair['output'], expected)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(palettes) >= 40 and len(dimensions) >= 20 and len(layouts) >= 100
    assert min(accent_areas) == 1 and max(accent_areas) >= 12
    assert {0, 1, 2} <= contacts and max(component_counts) >= 2
    assert layer_counts == {2, 3} and small >= 3 and broad >= 5
    assert concave >= 5 and three_layer_tests >= 5
    for split in ('train', 'test'):
        for feature in ('single', 'separated', 'five_by_five', 'intrusion',
                        'row_notch', 'interior', 'two_edges', 'horizontal',
                        'vertical', 'three_layer_clipped_separated',
                        'notched_three_layer'):
            assert review_counts[(split, feature)] > 0, (split, feature)
    assert novel_test_accents > 0 and role_swapped_tests > 0
