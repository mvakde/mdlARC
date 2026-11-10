import numpy as np


_AUTO_COUNT_OPTIONS = (
    (2, 1),
    (3, 1),
    (4, 1),
    (4, 2),
    (5, 2),
    (5, 3),
    (6, 1),
    (6, 2),
    (6, 3),
    (7, 4),
    (8, 1),
    (8, 3),
    (9, 2),
    (9, 4),
    (10, 1),
    (10, 4),
)


_SOURCE_RAW_PAIRS = (
    (
        (
            "11000000010010010101",
            "01000001111001100000",
        ),
        (
            "11666660010010010101",
            "01666661111001100000",
        ),
    ),
    (
        (
            "11100110111100000101",
            "11101100101011000111",
            "00111001010110101110",
            "01010001101110001111",
        ),
        (
            "11100110111100666101",
            "11101100101011666111",
            "00111001010110101110",
            "01010001101110001111",
        ),
    ),
    (
        (
            "00100100011111000000",
            "10000110010000010000",
            "10110010101010001000",
            "00100011001111111001",
        ),
        (
            "00100100011111000666",
            "10000110010000010666",
            "10110010101010001666",
            "00100011001111111001",
        ),
    ),
    (
        (
            "511111500000000001101001000010",
            "010100011110111000011101011111",
            "110001010111011000010011001000",
        ),
        (
            "511111500000000001101001000010",
            "010100011110111666611101011111",
            "110001010111011666610011001000",
        ),
    ),
    (
        (
            "000010101000001000011010",
            "100011110000110110111110",
            "000001110110100011001111",
            "001100000111110110110001",
        ),
        (
            "066610101000001000011010",
            "166611110000110110111110",
            "066601110110100011001111",
            "001100000111110110110001",
        ),
    ),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _rows_signature(rows: tuple[str, ...]) -> tuple:
    height = len(rows)
    width = len(rows[0])
    values = tuple(int(ch) for row in rows for ch in row)
    return height, width, values


_SOURCE_PAIR_SIGNATURES = tuple(
    (_rows_signature(input_rows), _rows_signature(output_rows))
    for input_rows, output_rows in _SOURCE_RAW_PAIRS
)


def _grid_signature(grid: np.ndarray) -> tuple:
    return (
        int(grid.shape[0]),
        int(grid.shape[1]),
        tuple(int(value) for value in grid.ravel()),
    )


def _pair_is_source(input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
    signature = (_grid_signature(input_grid), _grid_signature(output_grid))
    return signature in _SOURCE_PAIR_SIGNATURES


def _is_integer_argument(value) -> bool:
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
    )


def _validate_count_argument(name: str, value, minimum: int, maximum: int) -> None:
    if value is None:
        return
    if not _is_integer_argument(value):
        raise ValueError(f"{name} must be an integer or None")
    if not minimum <= int(value) <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    _validate_count_argument("num_train", num_train, 2, 10)
    _validate_count_argument("num_test", num_test, 1, 4)

    if num_train is not None:
        num_train = int(num_train)
    if num_test is not None:
        num_test = int(num_test)

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test

    if num_train is None and num_test is None:
        return _AUTO_COUNT_OPTIONS[seed % len(_AUTO_COUNT_OPTIONS)]

    if num_train is None:
        minimum_train = max(2, num_test + 1)
        num_train = int(rng.integers(minimum_train, 11))
        return num_train, num_test

    maximum_test = min(4, num_train - 1)
    if maximum_test < 1:
        raise ValueError("no legal num_test exists for the requested num_train")
    num_test = int(rng.integers(1, maximum_test + 1))
    return num_train, num_test


def sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    colors = rng.permutation(10)
    background = int(colors[0])
    highlight = int(colors[1])
    blocker_count = int(rng.integers(1, 8))
    blockers = tuple(int(value) for value in colors[2:2 + blocker_count])

    return {
        "background": background,
        "highlight": highlight,
        "blockers": blockers,
    }


def _true_runs(row: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    start = None

    for column, value in enumerate(row):
        if bool(value):
            if start is None:
                start = column
        elif start is not None:
            runs.append((start, column))
            start = None

    if start is not None:
        runs.append((start, len(row)))

    return runs


def _largest_background_rectangles(
    grid: np.ndarray,
    background: int,
) -> tuple[int, list[tuple[int, int, int, int]]]:
    height, width = grid.shape
    candidates = []

    for top in range(height - 1):
        common_background = np.ones(width, dtype=bool)

        for bottom in range(top, height):
            common_background &= grid[bottom] == background
            rectangle_height = bottom - top + 1

            if rectangle_height < 2:
                continue

            for left, right in _true_runs(common_background):
                rectangle_width = right - left
                if rectangle_width >= 2:
                    area = rectangle_height * rectangle_width
                    candidates.append(
                        (
                            area,
                            top,
                            left,
                            rectangle_height,
                            rectangle_width,
                        )
                    )

    if not candidates:
        return 0, []

    maximum_area = max(candidate[0] for candidate in candidates)
    winners = [
        (top, left, rectangle_height, rectangle_width)
        for area, top, left, rectangle_height, rectangle_width in candidates
        if area == maximum_area
    ]
    return maximum_area, winners



def _rectangle_mask(height, width, top, left, rh, rw):
    mask = np.zeros((height, width), dtype=bool)
    mask[top:top + rh, left:left + rw] = True
    return mask


def _rectangles(grid, background):
    height, width = grid.shape
    found = []
    for top in range(height - 1):
        common = np.ones(width, dtype=bool)
        for bottom in range(top, height):
            common &= grid[bottom] == background
            rh = bottom - top + 1
            if rh >= 2:
                for left, right in _true_runs(common):
                    if right - left >= 2:
                        found.append((rh * (right - left), top, left, rh, right - left))
    return found


def _geometry(rng, style, profile):
    if style == 'shortwide':
        minimum_height = 2 if profile in ('simple', 'line') else 5
        height = int(rng.integers(minimum_height, 11))
        width = int(rng.integers(18, 31))
    elif style == 'tall':
        height = int(rng.integers(18, 31))
        width = int(rng.integers(5, 12))
    elif style == 'compact':
        height = int(rng.integers(5, 11))
        width = int(rng.integers(5, 12))
    elif style == 'square':
        side = int(rng.integers(11, 30))
        height = max(5, min(30, side + int(rng.integers(-3, 4))))
        width = max(5, min(30, side + int(rng.integers(-3, 4))))
    else:
        height = int(rng.integers(20, 31))
        width = int(rng.integers(20, 31))

    if profile in ('mixed', 'overlap', 'rival'):
        shapes = [(3, 3), (2, 4), (3, 4), (4, 3), (4, 4), (2, 5), (5, 2)]
        if profile == 'overlap':
            shapes = [(3, 3), (3, 4), (4, 3), (4, 4), (5, 3), (3, 5)]
        rh, rw = shapes[int(rng.integers(len(shapes)))]
        rh = min(rh, height - 1)
        rw = min(rw, width - 1)
    else:
        rh = int(rng.integers(2, min(height, 10) + 1))
        rw = int(rng.integers(2, min(width, 10) + 1))
        if profile == 'line' and rh * rw >= max(height, width):
            rh, rw = (2, int(rng.integers(2, min(7, width) + 1)))
    vertical = int(rng.integers(3))
    horizontal = int(rng.integers(3))
    top = (0 if vertical == 0 else height - rh if vertical == 1
           else int(rng.integers(1, height - rh)) if height - rh > 1 else 0)
    left = (0 if horizontal == 0 else width - rw if horizontal == 1
            else int(rng.integers(1, width - rw)) if width - rw > 1 else 0)
    return height, width, top, left, rh, rw


def _add_overlap(mask, target, rng):
    height, width = mask.shape
    top, left, rh, rw = target
    choices = []
    for side in range(4):
        if side == 0 and top > 0:
            for length in range(2, rw):
                for start in range(rw - length + 1):
                    choices.append((top - 1, left + start, 1, length))
        elif side == 1 and top + rh < height:
            for length in range(2, rw):
                for start in range(rw - length + 1):
                    choices.append((top + rh, left + start, 1, length))
        elif side == 2 and left > 0:
            for length in range(2, rh):
                for start in range(rh - length + 1):
                    choices.append((top + start, left - 1, length, 1))
        elif side == 3 and left + rw < width:
            for length in range(2, rh):
                for start in range(rh - length + 1):
                    choices.append((top + start, left + rw, length, 1))
    if not choices:
        return False
    order = rng.permutation(len(choices))
    for index in order:
        y, x, h, w = choices[int(index)]
        if np.all(mask[y:y + h, x:x + w]):
            continue
        trial = mask.copy()
        trial[y:y + h, x:x + w] = True
        area, winners = _largest_background_rectangles(trial.astype(np.int8), 1)
        if area == rh * rw and winners == [target]:
            mask[y:y + h, x:x + w] = True
            return True
    return False


def _add_rival(mask, target, rng):
    height, width = mask.shape
    target_area = target[2] * target[3]
    if target_area <= 4:
        return False
    shapes = [(h, w) for h in range(2, min(height, 9) + 1)
              for w in range(2, min(width, 9) + 1)
              if 4 <= h * w < target_area]
    if not shapes:
        return False
    rng.shuffle(shapes)
    shapes.sort(key=lambda shape: shape[0] * shape[1], reverse=True)
    for rh, rw in shapes[:12]:
        positions = [(y, x) for y in range(height - rh + 1)
                     for x in range(width - rw + 1)
                     if not np.any(mask[max(0, y - 1):min(height, y + rh + 1),
                                        max(0, x - 1):min(width, x + rw + 1)])]
        if positions:
            y, x = positions[int(rng.integers(len(positions)))]
            mask[y:y + rh, x:x + rw] = True
            return True
    return False


def _add_line(mask, target_area, rng):
    height, width = mask.shape
    orientations = [axis for axis, limit in ((0, width), (1, height))
                    if limit > target_area]
    if not orientations:
        return False
    rng.shuffle(orientations)
    for axis in orientations:
        limit = width if axis == 0 else height
        length = int(rng.integers(target_area + 1, min(limit, target_area + 5) + 1))
        positions = []
        if axis == 0:
            for y in range(height):
                for x in range(width - length + 1):
                    if not np.any(mask[y, x:x + length]):
                        positions.append((y, x))
        else:
            for x in range(width):
                for y in range(height - length + 1):
                    if not np.any(mask[y:y + length, x]):
                        positions.append((y, x))
        if positions:
            y, x = positions[int(rng.integers(len(positions)))]
            if axis == 0:
                mask[y, x:x + length] = True
            else:
                mask[y:y + length, x] = True
            return True
    return False


def _pair_features(grid, background, target):
    top, left, rh, rw = target
    target_area = rh * rw
    candidates = _rectangles(grid, background)
    other = [(area, y, x, h, w) for area, y, x, h, w in candidates
             if not (top <= y and left <= x and
                     y + h <= top + rh and x + w <= left + rw)]
    overlap = [item for item in other
               if item[1] < top + rh and top < item[1] + item[3]
               and item[2] < left + rw and left < item[2] + item[4]]
    unchanged = grid == background
    unchanged[top:top + rh, left:left + rw] = False
    longest_horizontal = max([right - l for row in unchanged
                              for l, right in _true_runs(row)] + [0])
    longest_vertical = max([bottom - t for column in unchanged.T
                            for t, bottom in _true_runs(column)] + [0])
    attachments = 0
    if top > 0:
        attachments += int(np.count_nonzero(grid[top - 1, left:left + rw] == background))
    if top + rh < grid.shape[0]:
        attachments += int(np.count_nonzero(grid[top + rh, left:left + rw] == background))
    if left > 0:
        attachments += int(np.count_nonzero(grid[top:top + rh, left - 1] == background))
    if left + rw < grid.shape[1]:
        attachments += int(np.count_nonzero(grid[top:top + rh, left + rw] == background))
    return {'competitors': len(other),
            'runner_up': max((item[0] for item in other), default=0),
            'overlap': len(overlap), 'attachments': attachments,
            'long_line': max(longest_horizontal, longest_vertical) > target_area,
            'horizontal_long': longest_horizontal > target_area,
            'vertical_long': longest_vertical > target_area,
            'wider_loser': any(item[4] > rw for item in other),
            'taller_loser': any(item[3] > rh for item in other)}


def generate_pair(rng, regime, puzzle_parameters, profile, style, active_blockers):
    if regime not in ('source_train', 'target_test'):
        raise ValueError('unknown pair regime')
    background = puzzle_parameters['background']
    highlight = puzzle_parameters['highlight']
    for _ in range(600):
        height, width, top, left, rh, rw = _geometry(rng, style, profile)
        target = (top, left, rh, rw)
        target_area = rh * rw
        mask = _rectangle_mask(height, width, top, left, rh, rw)
        if profile in ('overlap', 'mixed') and not _add_overlap(mask, target, rng):
            continue
        if profile in ('rival', 'mixed') and not _add_rival(mask, target, rng):
            continue
        if profile in ('line', 'mixed') and not _add_line(mask, target_area, rng):
            continue
        if profile not in ('simple',) and rng.random() < 0.38:
            _add_overlap(mask, target, rng)
        if profile not in ('simple', 'rival') and rng.random() < 0.28:
            _add_rival(mask, target, rng)
        if profile != 'simple' and rng.random() < 0.26:
            _add_line(mask, target_area, rng)
        density = float(rng.uniform(0.06, 0.32))
        noise = rng.random((height, width)) < density
        if profile == 'simple':
            noise[max(0, top - 1):min(height, top + rh + 1),
                  max(0, left - 1):min(width, left + rw + 1)] = False
        mask |= noise
        # A sparse blocker texture keeps every planted region visible; the
        # completed mask, not the planting recipe, decides the winner.
        input_grid = np.full((height, width), background, dtype=np.int8)
        non_background = np.argwhere(~mask)
        if len(non_background) < len(active_blockers):
            continue
        colors = np.asarray(active_blockers, dtype=np.int8)
        input_grid[~mask] = rng.choice(colors, size=len(non_background))
        chosen = rng.permutation(len(non_background))[:len(colors)]
        for index, color in zip(chosen, colors):
            y, x = non_background[int(index)]
            input_grid[y, x] = color
        area, winners = _largest_background_rectangles(input_grid, background)
        if area != target_area or winners != [target]:
            continue
        features = _pair_features(input_grid, background, target)
        if profile in ('overlap', 'mixed') and features['overlap'] < 1:
            continue
        if profile in ('rival', 'mixed') and features['runner_up'] < max(4, int(0.60 * target_area)):
            continue
        if profile in ('line', 'mixed') and not features['long_line']:
            continue
        if profile == 'simple' and features['competitors'] != 0:
            continue
        output_grid = input_grid.copy()
        output_grid[top:top + rh, left:left + rw] = highlight
        if not _pair_is_source(input_grid, output_grid):
            return input_grid, output_grid
    raise RuntimeError('failed to construct an unambiguous pair')


def _analyze_pair(input_grid, output_grid):
    _require(input_grid.shape == output_grid.shape, 'input and output dimensions differ')
    changed = input_grid != output_grid
    _require(bool(np.any(changed)), 'pair must change')
    rows, columns = np.nonzero(changed)
    top, left = int(rows.min()), int(columns.min())
    bottom, right = int(rows.max()) + 1, int(columns.max()) + 1
    rh, rw = bottom - top, right - left
    _require(rh >= 2 and rw >= 2, 'answer must be two-dimensional')
    expected = _rectangle_mask(*input_grid.shape, top, left, rh, rw)
    _require(np.array_equal(changed, expected), 'changes do not form one rectangle')
    _require(len(np.unique(input_grid[changed])) == 1, 'answer input is not uniform')
    _require(len(np.unique(output_grid[changed])) == 1, 'answer output is not uniform')
    background = int(input_grid[top, left])
    highlight = int(output_grid[top, left])
    _require(background != highlight, 'background equals highlight')
    _require(highlight not in np.unique(input_grid), 'highlight is in input')
    area, winners = _largest_background_rectangles(input_grid, background)
    _require(area == rh * rw and winners == [(top, left, rh, rw)],
             'answer is not the unique largest eligible rectangle')
    blockers = frozenset(int(value) for value in np.unique(input_grid)
                         if int(value) != background)
    _require(bool(blockers), 'input has no blocker')
    info = {'background': background, 'highlight': highlight,
            'blockers': blockers, 'top': top, 'left': left,
            'height': rh, 'width': rw, 'area': area,
            'grid_height': int(input_grid.shape[0]),
            'grid_width': int(input_grid.shape[1])}
    info.update(_pair_features(input_grid, background, (top, left, rh, rw)))
    return info


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'},
             'puzzle must have train and test')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')),
             'splits must be lists')
    train_count, test_count = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             'invalid pair counts')
    seen_inputs, seen_pairs = set(), set()
    infos = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'},
                     'pair must have input and output')
            for role in ('input', 'output'):
                grid = pair[role]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         'grid must be a two-dimensional NumPy array')
                _require(all(1 <= value <= 30 for value in grid.shape),
                         'grid dimensions are out of range')
                _require(np.issubdtype(grid.dtype, np.integer), 'grid must be integer')
                _require(bool(np.all((grid >= 0) & (grid <= 9))), 'grid colors are out of range')
            key = _grid_signature(pair['input'])
            pair_key = (key, _grid_signature(pair['output']))
            _require(key not in seen_inputs and pair_key not in seen_pairs,
                     'duplicate pair or input')
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            _require(not _pair_is_source(pair['input'], pair['output']),
                     'pair copies source')
            infos[split].append(_analyze_pair(pair['input'], pair['output']))
    conventions = {(info['background'], info['highlight'])
                   for split in infos.values() for info in split}
    _require(len(conventions) == 1, 'background/highlight convention changes')
    train_colors = set().union(*(info['blockers'] for info in infos['train']))
    for info in infos['test']:
        _require(info['blockers'] <= train_colors, 'test has an unseen blocker')
    _require(any(info['grid_height'] <= 10 and info['grid_width'] >= 18
                 for info in infos['train']), 'training lacks a short wide example')
    _require(any(info['runner_up'] >= max(4, int(.60 * info['area']))
                 and info['long_line'] for info in infos['train']),
             'training lacks area and one-dimensional comparisons')
    _require(any(info['overlap'] for info in infos['train']),
             'training lacks an overlapping competitor')


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if not _is_integer_argument(seed) or int(seed) < 0:
        raise ValueError('seed must be a non-negative integer')
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    parameters = sample_puzzle_parameters(rng)
    blocker_count = len(parameters['blockers'])
    homogeneous_style = 'shortwide' if rng.random() < 0.10 else None
    # Training always displays every blocker color used by tests. Each pair
    # samples a subset independently after the first complete demonstration.
    plans = []
    for split, count in (('train', train_count), ('test', test_count)):
        for index in range(count):
            if split == 'train' and index == 0:
                profile, style = 'mixed', 'shortwide'
            elif split == 'train' and index == 1:
                profile = 'overlap'
                style = (homogeneous_style or
                         ('compact', 'tall', 'square', 'large')[int(rng.integers(4))])
            else:
                if split == 'test':
                    choices = ('simple', 'overlap', 'rival', 'line', 'mixed', 'mixed')
                    styles = ('compact', 'tall', 'square', 'large', 'large', 'shortwide')
                else:
                    choices = ('simple', 'overlap', 'rival', 'line', 'mixed')
                    styles = ('shortwide', 'tall', 'compact', 'square', 'large')
                profile = choices[int(rng.integers(len(choices)))]
                style = homogeneous_style or styles[int(rng.integers(len(styles)))]
            if split == 'train' and index == 0:
                active = parameters['blockers']
            else:
                selected = rng.random(blocker_count) < float(rng.uniform(.25, .85))
                selected[0] = True
                active = tuple(color for color, take in zip(parameters['blockers'], selected) if take)
            plans.append((split, profile, style, active))
    puzzle = {'train': [], 'test': []}
    used = set()
    for split, profile, style, active in plans:
        for _ in range(20):
            pair_seed = int(rng.integers(0, 2**63, dtype=np.int64))
            pair_rng = np.random.default_rng(pair_seed)
            inp, out = generate_pair(pair_rng, 'source_train' if split == 'train'
                                     else 'target_test', parameters, profile, style, active)
            key = _grid_signature(inp)
            if key not in used:
                used.add(key)
                puzzle[split].append({'input': inp, 'output': out})
                break
        else:
            raise RuntimeError('failed to construct a unique pair')
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first, second):
    if any(len(first[s]) != len(second[s]) for s in ('train', 'test')):
        return False
    return all(np.array_equal(a[role], b[role])
               for split in ('train', 'test')
               for a, b in zip(first[split], second[split])
               for role in ('input', 'output'))


def _predict_from_training(train_pairs, test_input):
    roles = set()
    for pair in train_pairs:
        changed = pair['input'] != pair['output']
        _require(bool(np.any(changed)), 'training pair does not reveal a color role')
        _require(len(np.unique(pair['input'][changed])) == 1 and
                 len(np.unique(pair['output'][changed])) == 1,
                 'training pair has ambiguous color roles')
        roles.add((int(pair['input'][changed][0]),
                   int(pair['output'][changed][0])))
    _require(len(roles) == 1, 'training color roles disagree')
    background, highlight = next(iter(roles))
    height, width = test_input.shape
    heights = np.zeros(width, dtype=int)
    best_area, best_rectangles = 0, set()
    for bottom in range(height):
        heights = np.where(test_input[bottom] == background, heights + 1, 0)
        for left in range(width - 1):
            minimum_height = int(heights[left])
            for right in range(left + 1, width):
                minimum_height = min(minimum_height, int(heights[right]))
                if minimum_height < 2:
                    break
                rectangle_width = right - left + 1
                area = rectangle_width * minimum_height
                rectangle = (bottom - minimum_height + 1, left,
                             minimum_height, rectangle_width)
                if area > best_area:
                    best_area, best_rectangles = area, {rectangle}
                elif area == best_area:
                    best_rectangles.add(rectangle)
    _require(len(best_rectangles) == 1, 'training-only prediction is ambiguous')
    top, left, rh, rw = next(iter(best_rectangles))
    predicted = test_input.copy()
    predicted[top:top + rh, left:left + rw] = highlight
    return predicted


def self_test() -> None:
    automatic = generate_puzzle(314159)
    validate_generated_puzzle(automatic)
    explicit = generate_puzzle(271828, num_train=6, num_test=2)
    _require((len(explicit['train']), len(explicit['test'])) == (6, 2),
             'explicit count mismatch')
    _require(_puzzles_equal(automatic, generate_puzzle(314159)),
             'automatic generation is not deterministic')
    _require(_puzzles_equal(explicit, generate_puzzle(271828, 6, 2)),
             'explicit generation is not deterministic')
    _require(not _puzzles_equal(explicit, generate_puzzle(271829, 6, 2)),
             'different seeds produced the same puzzle')
    for index, counts in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(700000 + index, *counts)
        _require((len(puzzle['train']), len(puzzle['test'])) == counts,
                 'boundary counts mismatch')
    for index, kwargs in enumerate(({'num_test': 1}, {'num_test': 4},
                                    {'num_train': 2}, {'num_train': 10})):
        puzzle = generate_puzzle(710000 + index, **kwargs)
        for key, count in kwargs.items():
            _require(len(puzzle['train' if key == 'num_train' else 'test']) == count,
                     'partial count mismatch')
    for index, kwargs in enumerate(({'num_train': 1}, {'num_train': 11},
                                    {'num_test': 0}, {'num_test': 5},
                                    {'num_train': 2, 'num_test': 2},
                                    {'num_train': 3, 'num_test': 4},
                                    {'num_train': True}, {'num_test': 1.5})):
        try:
            generate_puzzle(720000 + index, **kwargs)
        except ValueError:
            continue
        raise AssertionError('illegal count accepted')
    train_counts, test_counts = set(), set()
    shapes, targets, conventions, palette_sizes = set(), set(), set(), set()
    profiles = {'simple': 0, 'overlap': 0, 'close': 0, 'line': 0,
                'attached': 0, 'multicolor_test': 0, 'wider': 0, 'taller': 0,
                'two_by_two': 0, 'compact': 0, 'tall_canvas': 0,
                'large_canvas': 0, 'size_30': 0, 'height_2': 0,
                'interior': 0, 'corner': 0, 'horizontal_long': 0,
                'vertical_long': 0, 'several_competitors': 0,
                'combined': 0}
    within_shape = 0
    within_target_shape = 0
    homogeneous = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        training = [{role: pair[role].copy() for role in ('input', 'output')}
                    for pair in puzzle['train']]
        for pair in puzzle['test']:
            predicted = _predict_from_training(training, pair['input'].copy())
            _require(np.array_equal(predicted, pair['output']),
                     'training-only prediction disagrees with test output')
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        local_shapes = set()
        local_targets = set()
        short_wide_only = True
        first = _analyze_pair(puzzle['train'][0]['input'], puzzle['train'][0]['output'])
        conventions.add((first['background'], first['highlight']))
        palette_sizes.add(len(set().union(*(_analyze_pair(p['input'], p['output'])['blockers']
                                            for p in puzzle['train']))))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _analyze_pair(pair['input'], pair['output'])
                shape = (info['grid_height'], info['grid_width'])
                shapes.add(shape)
                local_shapes.add(shape)
                local_targets.add((info['height'], info['width']))
                targets.add((info['height'], info['width'], info['top'], info['left']))
                profiles['simple'] += int(info['competitors'] == 0)
                profiles['overlap'] += int(info['overlap'] > 0)
                profiles['close'] += int(info['runner_up'] >= .70 * info['area'])
                profiles['line'] += int(info['long_line'])
                profiles['attached'] += int(info['attachments'] >= 2)
                profiles['wider'] += int(info['wider_loser'])
                profiles['taller'] += int(info['taller_loser'])
                profiles['two_by_two'] += int((info['height'], info['width']) == (2, 2))
                profiles['compact'] += int(info['grid_height'] <= 11 and info['grid_width'] <= 11)
                profiles['tall_canvas'] += int(info['grid_height'] >= 18 and info['grid_width'] <= 11)
                profiles['large_canvas'] += int(info['grid_height'] >= 20 and info['grid_width'] >= 20)
                profiles['size_30'] += int(30 in shape)
                profiles['height_2'] += int(info['grid_height'] == 2)
                profiles['interior'] += int(0 < info['top'] < info['grid_height'] - info['height']
                                            and 0 < info['left'] < info['grid_width'] - info['width'])
                profiles['corner'] += int(info['top'] in (0, info['grid_height'] - info['height'])
                                          and info['left'] in (0, info['grid_width'] - info['width']))
                profiles['horizontal_long'] += int(info['horizontal_long'])
                profiles['vertical_long'] += int(info['vertical_long'])
                profiles['several_competitors'] += int(info['competitors'] >= 2)
                profiles['combined'] += int(info['overlap'] > 0 and info['attachments'] >= 2
                                            and info['runner_up'] >= .70 * info['area'])
                short_wide_only &= info['grid_height'] <= 10 and info['grid_width'] >= 18
                if split == 'test':
                    profiles['multicolor_test'] += int(len(info['blockers']) >= 2)
        within_shape += int(len(local_shapes) >= 2)
        within_target_shape += int(len(local_targets) >= 2)
        homogeneous += int(short_wide_only)
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             'automatic counts are insufficiently varied')
    _require(len(shapes) >= 35 and len(targets) >= 75 and len(conventions) >= 20,
             'structural or color diversity is insufficient')
    _require(len(palette_sizes) >= 4 and within_shape >= 35
             and within_target_shape >= 35 and homogeneous >= 1,
             'within-puzzle or palette diversity is insufficient')
    for name, count in profiles.items():
        minimum = 2 if name == 'height_2' else 4 if name in ('simple', 'two_by_two') else 10
        _require(count >= minimum,
                 f'insufficient {name} coverage: {count}')
