import numpy as np


# A generated input is rejected if it exactly matches any source-task input.
# This is stronger than merely rejecting an exact source input/output pair.
_SOURCE_INPUT_ROWS = (
    (
        "02020", "22022", "02220", "02200", "02222", "20202", "44444",
        "20222", "02200", "20202", "20002", "22020", "20220",
    ),
    (
        "02222", "00002", "20222", "00220", "22220", "22002", "44444",
        "00000", "00200", "20002", "00020", "02020", "02220",
    ),
    (
        "22022", "20222", "20000", "02020", "22202", "20200", "44444",
        "20022", "00202", "22000", "00202", "02022", "02202",
    ),
    (
        "00022", "00202", "20022", "22002", "00002", "02000", "44444",
        "20000", "22000", "20200", "00200", "00022", "20020",
    ),
    (
        "20202", "20202", "00020", "02220", "20220", "22202", "44444",
        "22000", "02222", "00220", "02000", "02202", "20000",
    ),
    (
        "20220", "20022", "22200", "22222", "02200", "22222", "44444",
        "00022", "20002", "22202", "02200", "20220", "20222",
    ),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _signature_from_rows(rows: tuple[str, ...]) -> tuple:
    return (
        (len(rows), len(rows[0])),
        tuple(int(char) for row in rows for char in row),
    )


_SOURCE_INPUT_SIGNATURES = frozenset(
    _signature_from_rows(rows) for rows in _SOURCE_INPUT_ROWS
)


def _array_signature(grid: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.ravel()),
    )


def _pair_signature(pair: dict) -> tuple:
    return (
        _array_signature(pair["input"]),
        _array_signature(pair["output"]),
    )


def _validated_optional_count(
    name: str,
    value: int | None,
    lower: int,
    upper: int,
) -> int | None:
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    result = int(value)
    if not lower <= result <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return result


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _validated_optional_count(
        "num_train", num_train, 2, 10
    )
    test_count = _validated_optional_count(
        "num_test", num_test, 1, 4
    )

    if (
        train_count is not None
        and test_count is not None
        and train_count <= test_count
    ):
        raise ValueError("num_train must be greater than num_test")

    if train_count is None and test_count is None:
        # This deterministic schedule guarantees broad count diversity for
        # consecutive seeds while respecting every legal bound.
        test_count = 1 + seed % 4
        train_count = (
            test_count
            + 1
            + (seed // 4) % (10 - test_count)
        )
    elif train_count is None:
        minimum_train = max(2, test_count + 1)
        train_count = int(rng.integers(minimum_train, 11))
    elif test_count is None:
        maximum_test = min(4, train_count - 1)
        if maximum_test < 1:
            raise ValueError("no legal num_test exists for num_train")
        test_count = int(rng.integers(1, maximum_test + 1))

    if not (
        2 <= train_count <= 10
        and 1 <= test_count <= 4
        and train_count > test_count
    ):
        raise ValueError("illegal train/test count combination")

    return train_count, test_count


def _sample_shape(rng, orientation, minimum=1):
    # The axis parallel to the divider can occupy all 30 cells.
    def axis(limit, floor):
        choices = ((floor, min(limit, 2)), (max(floor, 3), min(limit, 6)),
                   (max(floor, 7), min(limit, 14)), (max(floor, 15), limit))
        choices = [(a, b) for a, b in choices if a <= b]
        lo, hi = choices[int(rng.integers(len(choices)))]
        return int(rng.integers(lo, hi + 1))

    if orientation == 'horizontal':
        height, width = axis(14, minimum), axis(30, minimum)
    else:
        height, width = axis(30, minimum), axis(14, minimum)
    if height * width == 1:
        if orientation == 'horizontal':
            width = 2
        else:
            height = 2
    return height, width


def _sample_parameters(rng):
    neutral, discrepancy, active, divider = (
        int(x) for x in rng.choice(10, size=4, replace=False)
    )
    orientation = 'horizontal' if rng.integers(2) == 0 else 'vertical'
    return {
        'neutral': neutral,
        'discrepancy': discrepancy,
        'active': active,
        'divider': divider,
        'orientation': orientation,
        'layout_mode': ('fixed', 'free', 'transfer', 'mixed')[int(rng.integers(4))],
        'palette_mode': ('fixed', 'active', 'divider', 'both')[int(rng.integers(4))],
        'density_mode': ('ordinary', 'shared', 'challenge')[int(rng.integers(3))],
        'fixed_shape': _sample_shape(rng, orientation, 3),
    }


def _choose_roles(rng, parameters, split, index, train_roles):
    mode = parameters['palette_mode']
    active, divider = parameters['active'], parameters['divider']
    if mode == 'fixed' or (split == 'train' and index == 0):
        return active, divider
    forbidden = {parameters['neutral'], parameters['discrepancy']}
    if split == 'test':
        # Every role color used by tests has already appeared in training.
        actives = sorted({x for x, _ in train_roles})
        dividers = sorted({x for _, x in train_roles})
        options = [(a, d) for a in actives for d in dividers if a != d]
        return options[int(rng.integers(len(options)))]
    if mode in ('active', 'both'):
        candidates = [x for x in range(10) if x not in forbidden | {active, divider}]
        active = int(rng.choice(candidates))
    if mode in ('divider', 'both'):
        candidates = [x for x in range(10) if x not in forbidden | {active, divider}]
        divider = int(rng.choice(candidates))
    return active, divider


def _orientation(parameters, rng, split, index):
    base = parameters['orientation']
    other = 'vertical' if base == 'horizontal' else 'horizontal'
    mode = parameters['layout_mode']
    if mode == 'fixed':
        return base
    if mode == 'transfer':
        return base if split == 'train' else other
    if mode == 'mixed' and split == 'train' and index < 2:
        return base if index == 0 else other
    return base if rng.integers(2) == 0 else other


def _scores(rng, shape, style):
    height, width = shape
    noise = rng.random(shape)
    yy, xx = np.indices(shape)
    if style == 'scattered':
        return noise
    if style == 'blocks':
        marks = np.zeros(shape, dtype=float)
        for _ in range(int(rng.integers(1, 4))):
            top = int(rng.integers(height))
            left = int(rng.integers(width))
            bottom = int(rng.integers(top + 1, height + 1))
            right = int(rng.integers(left + 1, width + 1))
            marks[top:bottom, left:right] = 1
        return marks + 0.25 * noise
    if style == 'stripes':
        axis = yy if rng.integers(2) == 0 else xx
        period = int(rng.integers(2, 6))
        band = int(rng.integers(1, period))
        phase = int(rng.integers(period))
        return (((axis + phase) % period) < band).astype(float) + 0.25 * noise
    if style == 'motif':
        motif = rng.integers(0, 2, size=(int(rng.integers(2, 5)),
                                          int(rng.integers(2, 5))))
        return motif[yy % motif.shape[0], xx % motif.shape[1]] + 0.25 * noise
    if style == 'patch':
        cy, cx = int(rng.integers(height)), int(rng.integers(width))
        radius = max(height, width)
        return 1 - (abs(yy - cy) + abs(xx - cx)) / radius + 0.18 * noise
    if style == 'isolated':
        marks = np.zeros(shape, dtype=float)
        count = max(1, min(height * width, int(rng.integers(1, 7))))
        indices = rng.choice(height * width, size=count, replace=False)
        marks.flat[indices] = 1
        return marks + 0.2 * noise
    # Independent spatial components can coexist in one panel or mask.
    first = _scores(rng, shape, 'blocks' if rng.integers(2) == 0 else 'stripes')
    second = _scores(rng, shape, 'motif' if rng.integers(2) == 0 else 'patch')
    return first + second + 0.1 * noise


def _pattern(rng, shape, count, style=None):
    total = shape[0] * shape[1]
    if style is None:
        style = ('scattered', 'blocks', 'stripes', 'motif', 'patch',
                 'isolated', 'mixed')[int(rng.integers(7))]
    result = np.zeros(total, dtype=np.int8)
    if count:
        scores = _scores(rng, shape, style).ravel()
        indices = np.argpartition(scores, total - count)[total - count:]
        result[indices] = 1
    return result.reshape(shape)


def _disagreement_count(rng, cells, split, density_mode, anchor=False,
                        second_palette_demo=False):
    if anchor:
        low = max(2, int(np.ceil(0.25 * cells)))
        high = min(cells - 2, int(np.floor(0.75 * cells)))
        return int(rng.integers(low, high + 1))
    if density_mode == 'ordinary':
        low = max(1, int(np.ceil(0.25 * cells)))
        high = min(cells - 1, int(np.floor(0.75 * cells)))
        return int(rng.integers(low, high + 1))
    if second_palette_demo:
        return int(rng.integers(1, cells))
    styles = ('ordinary', 'low', 'high', 'identical', 'complement', 'single')
    if split == 'test' and density_mode == 'challenge':
        weights = (0.26, 0.18, 0.18, 0.14, 0.14, 0.10)
    else:
        weights = (0.46, 0.17, 0.17, 0.07, 0.07, 0.06)
    style = str(rng.choice(styles, p=weights))
    if style == 'identical':
        return 0
    if style == 'complement':
        return cells
    if style == 'single':
        return 1 if rng.integers(2) == 0 else cells - 1
    if style == 'low' or style == 'high':
        few = int(rng.integers(1, max(2, cells // 5 + 1)))
        return few if style == 'low' else cells - few
    low = max(1, int(np.ceil(0.25 * cells)))
    high = min(cells - 1, int(np.floor(0.75 * cells)))
    if low > high:
        low, high = 1, cells - 1
    return int(rng.integers(low, high + 1))


def _generate_pair(rng, shape, orientation, neutral, active, divider,
                   discrepancy, split, density_mode, anchor=False,
                   second_palette_demo=False):
    height, width = shape
    cells = height * width
    differences = _disagreement_count(rng, cells, split, density_mode,
                                      anchor, second_palette_demo)
    mask = _pattern(rng, shape, differences)
    if anchor or differences in (0, cells):
        foreground_count = int(rng.integers(1, cells))
    elif rng.random() < 0.09:
        foreground_count = 0 if rng.integers(2) == 0 else cells
    else:
        foreground_count = int(rng.integers(1, cells))
    first = _pattern(rng, shape, foreground_count)
    if anchor:
        mismatches = rng.permutation(np.flatnonzero(mask))[:2]
        matches = rng.permutation(np.flatnonzero(1 - mask))[:2]
        first.flat[mismatches[0]] = 0
        first.flat[mismatches[1]] = 1
        first.flat[matches[0]] = 0
        first.flat[matches[1]] = 1
    second = first ^ mask
    # Both state colors appear in the combined panels, including solid outputs.
    if not np.any(first) and not np.any(second):
        first.flat[0] = 1
        second = first ^ mask
    if np.all(first) and np.all(second):
        first.flat[0] = 0
        second = first ^ mask
    first_panel = np.where(first, active, neutral).astype(np.int8)
    second_panel = np.where(second, active, neutral).astype(np.int8)
    if orientation == 'horizontal':
        separator = np.full((1, width), divider, dtype=np.int8)
        input_grid = np.vstack((first_panel, separator, second_panel))
    else:
        separator = np.full((height, 1), divider, dtype=np.int8)
        input_grid = np.hstack((first_panel, separator, second_panel))
    output_grid = np.where(mask, discrepancy, neutral).astype(np.int8)
    return input_grid, output_grid


def _check_grid(grid, name):
    _require(isinstance(grid, np.ndarray), f'{name} must be a NumPy array')
    _require(grid.ndim == 2, f'{name} must be two-dimensional')
    rows, columns = grid.shape
    _require(1 <= rows <= 30 and 1 <= columns <= 30,
             f'{name} has illegal dimensions')
    _require(np.issubdtype(grid.dtype, np.integer), f'{name} must be integer')
    _require(int(grid.min()) >= 0 and int(grid.max()) <= 9,
             f'{name} contains an illegal ARC color')


def _parse_input(input_grid):
    rows, columns = input_grid.shape
    candidates = []
    if rows >= 3 and rows % 2:
        middle = rows // 2
        divider = int(input_grid[middle, 0])
        first, second = input_grid[:middle], input_grid[middle + 1:]
        if np.all(input_grid[middle] == divider) and not np.any(first == divider) and not np.any(second == divider):
            candidates.append(('horizontal', first, second, divider))
    if columns >= 3 and columns % 2:
        middle = columns // 2
        divider = int(input_grid[0, middle])
        first, second = input_grid[:, :middle], input_grid[:, middle + 1:]
        if np.all(input_grid[:, middle] == divider) and not np.any(first == divider) and not np.any(second == divider):
            candidates.append(('vertical', first, second, divider))
    _require(len(candidates) == 1, 'input needs one unique complete central divider')
    orientation, first, second, divider = candidates[0]
    states = frozenset(int(x) for x in np.unique(np.concatenate((first.ravel(), second.ravel()))))
    _require(len(states) == 2 and divider not in states,
             'two aligned panels must share exactly two state colors')
    _require(first.shape == second.shape and first.size >= 2,
             'panels must be equal, aligned, and contain at least two cells')
    return orientation, first, second, divider, states


def _facts(pair):
    orientation, first, second, divider, states = _parse_input(pair['input'])
    output = pair['output']
    _require(output.shape == first.shape, 'output must have one panel shape')
    equal = first == second
    equal_colors = frozenset(int(x) for x in np.unique(output[equal]))
    different_colors = frozenset(int(x) for x in np.unique(output[~equal]))
    _require(len(equal_colors) <= 1 and len(different_colors) <= 1,
             'each comparison outcome must have a single output color')
    return {'orientation': orientation, 'shape': first.shape,
            'divider': divider, 'states': states, 'first': first,
            'second': second, 'equal': equal,
            'equal_colors': equal_colors, 'different_colors': different_colors,
            'output': output,
            'fraction': float(np.count_nonzero(~equal) / first.size)}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'},
             'puzzle must contain only train and test')
    _require(isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list),
             'splits must be lists')
    train_count, test_count = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             'illegal pair counts')
    seen_inputs, seen_pairs = set(), set()
    facts = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for index, pair in enumerate(puzzle[split]):
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'},
                     f'{split}[{index}] must contain input and output')
            _check_grid(pair['input'], f'{split}[{index}].input')
            _check_grid(pair['output'], f'{split}[{index}].output')
            input_signature = _array_signature(pair['input'])
            pair_signature = _pair_signature(pair)
            _require(input_signature not in seen_inputs and pair_signature not in seen_pairs,
                     'duplicate generated pair')
            _require(input_signature not in _SOURCE_INPUT_SIGNATURES,
                     'generated source input copy')
            seen_inputs.add(input_signature)
            seen_pairs.add(pair_signature)
            facts[split].append(_facts(pair))
    neutral_candidates = set().union(*(f['equal_colors'] for f in facts['train']))
    discrepancy_candidates = set().union(*(f['different_colors'] for f in facts['train']))
    _require(len(neutral_candidates) == len(discrepancy_candidates) == 1,
             'training must reveal both output colors')
    neutral = next(iter(neutral_candidates))
    discrepancy = next(iter(discrepancy_candidates))
    _require(neutral != discrepancy, 'output colors must differ')
    comparisons = set()
    for split in ('train', 'test'):
        for f in facts[split]:
            _require(neutral in f['states'], 'neutral must be a shared input state')
            _require(discrepancy not in f['states'] and discrepancy != f['divider'],
                     'discrepancy must be absent from input')
            _require(f['divider'] not in f['states'], 'divider must be distinct')
            expected = np.where(f['equal'], neutral, discrepancy)
            _require(np.array_equal(f['output'], expected), 'output is not cellwise XOR')
            if split == 'train':
                active = next(iter(f['states'] - {neutral}))
                comparisons.update((int(a == active), int(b == active))
                                   for a, b in zip(f['first'].ravel(), f['second'].ravel()))
    _require(comparisons == {(0, 0), (0, 1), (1, 0), (1, 1)},
             'training must cover all four binary comparisons collectively')


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be a non-negative integer')
    seed = int(seed)
    if seed < 0:
        raise ValueError('seed must be a non-negative integer')
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    parameters = _sample_parameters(rng)
    puzzle = {'train': [], 'test': []}
    seen_inputs = set()
    train_roles = []
    for split, count in (('train', train_count), ('test', test_count)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            orientation = _orientation(parameters, pair_rng, split, index)
            if parameters['layout_mode'] == 'fixed':
                shape = parameters['fixed_shape']
            else:
                minimum = 3 if split == 'train' and index == 0 else 2 if split == 'train' and index == 1 else 1
                shape = _sample_shape(pair_rng, orientation, minimum)
            active, divider = _choose_roles(pair_rng, parameters, split, index, train_roles)
            second_palette_demo = split == 'train' and index == 1 and parameters['palette_mode'] != 'fixed'
            for _ in range(64):
                input_grid, output_grid = _generate_pair(
                    pair_rng, shape, orientation, parameters['neutral'], active,
                    divider, parameters['discrepancy'], split,
                    parameters['density_mode'],
                    anchor=split == 'train' and index == 0,
                    second_palette_demo=second_palette_demo)
                signature = _array_signature(input_grid)
                if signature not in seen_inputs and signature not in _SOURCE_INPUT_SIGNATURES:
                    break
            else:
                raise RuntimeError('unable to construct a unique input')
            seen_inputs.add(signature)
            puzzle[split].append({'input': input_grid, 'output': output_grid})
            if split == 'train':
                train_roles.append((active, divider))
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_signature(puzzle):
    return tuple(tuple(_pair_signature(pair) for pair in puzzle[split])
                 for split in ('train', 'test'))


def self_test() -> None:
    automatic = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
    _require(len({len(p['train']) for p in automatic}) >= 4, 'train counts collapsed')
    _require(len({len(p['test']) for p in automatic}) >= 3, 'test counts collapsed')
    _require(len({_puzzle_signature(p) for p in automatic}) == 42,
             'distinct seeds produced duplicate puzzles')
    for seed, counts in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(3000 + seed, *counts)
        _require((len(puzzle['train']), len(puzzle['test'])) == counts,
                 'explicit count changed')
    for seed, counts in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(4000 + seed, *counts)
        _require((counts[0] is None or len(puzzle['train']) == counts[0]) and
                 (counts[1] is None or len(puzzle['test']) == counts[1]),
                 'partial count changed')
    for seed, counts in enumerate(((1, None), (11, None), (None, 0),
                                   (None, 5), (2, 2), (3, 4), (2.0, 1))):
        try:
            generate_puzzle(5000 + seed, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts were accepted')
    _require(_puzzle_signature(generate_puzzle(24680)) ==
             _puzzle_signature(generate_puzzle(24680)), 'non-deterministic generation')
    # Reconstruct the output solely from the visible training pairs and a test input.
    for puzzle in automatic:
        train_facts = [_facts(pair) for pair in puzzle['train']]
        neutral = next(iter(set().union(*(f['equal_colors'] for f in train_facts))))
        discrepancy = next(iter(set().union(*(f['different_colors'] for f in train_facts))))
        for pair in puzzle['test']:
            _, first, second, _, _ = _parse_input(pair['input'].copy())
            prediction = np.where(first == second, neutral, discrepancy)
            _require(np.array_equal(prediction, pair['output']),
                     'test output not recoverable from training and test input')
    all_facts = [[_facts(pair) for split in ('train', 'test') for pair in p[split]]
                 for p in automatic]
    _require(any(len({f['orientation'] for f in fs}) > 1 for fs in all_facts),
             'no within-puzzle orientation variation')
    _require(any(len({f['shape'] for f in fs}) > 1 for fs in all_facts),
             'no within-puzzle shape variation')
    _require(any(len({(f['states'], f['divider']) for f in fs}) > 1 for fs in all_facts),
             'no within-puzzle input palette variation')
    _require(any(any(min(f['shape']) <= 2 for f in fs) for fs in all_facts),
             'thin panels absent')
    _require(any(any(max(f['shape']) >= 15 for f in fs) for fs in all_facts),
             'long parallel axis absent')
    _require(any(any(f['fraction'] == 0 for f in fs) for fs in all_facts),
             'identical panels absent')
    _require(any(any(f['fraction'] == 1 for f in fs) for fs in all_facts),
             'complementary panels absent')
    _require(any(any(0.3 <= f['fraction'] <= 0.7 for f in fs[len(automatic[i]['train']):])
                 for i, fs in enumerate(all_facts)), 'ordinary-density tests absent')
    _require(any(all(0.25 <= f['fraction'] <= 0.75 for f in fs)
                 for fs in all_facts), 'no puzzle without a density shift')
    _require(any(len({('low' if f['fraction'] < 0.25 else
                        'high' if f['fraction'] > 0.75 else 'ordinary')
                      for f in fs[len(automatic[i]['train']):]}) > 1
                 for i, fs in enumerate(all_facts) if len(automatic[i]['test']) > 1),
             'test density levels never mix within a puzzle')
    _require(any({f['orientation'] for f in fs[:len(automatic[i]['train'])]} == {'horizontal'}
                 and {f['orientation'] for f in fs[len(automatic[i]['train']):]} == {'vertical'}
                 for i, fs in enumerate(all_facts)), 'horizontal-to-vertical transfer absent')
    _require(any({f['orientation'] for f in fs[:len(automatic[i]['train'])]} == {'vertical'}
                 and {f['orientation'] for f in fs[len(automatic[i]['train']):]} == {'horizontal'}
                 for i, fs in enumerate(all_facts)), 'vertical-to-horizontal transfer absent')
    _require(any(len({f['orientation'] for f in fs}) == 1 for fs in all_facts),
             'unchanged orientation absent')
    _require(any(len({f['states'] for f in fs}) > 1 and
                 len({f['divider'] for f in fs}) > 1 and
                 len({f['shape'] for f in fs}) > 1 and
                 len({f['orientation'] for f in fs}) > 1
                 for fs in all_facts), 'compatible variation axes did not combine')
    _require(any(any((f['shape'][1] if f['orientation'] == 'horizontal'
                      else f['shape'][0]) == 30 for f in fs)
                 for fs in all_facts), 'full parallel dimension absent')
    _require(any(any(len(np.unique(f['first'])) == 1 for f in fs)
                 for fs in all_facts), 'uniform panels absent')
    _require(any(any(len(np.unique(f['output'])) == 1 for f in fs)
                 for fs in all_facts), 'solid outputs absent')
    _require(any(any(int(np.count_nonzero(~f['equal'])) == 1 for f in fs)
                 for fs in all_facts), 'one-cell mismatches absent')
    _require(any(any(int(np.count_nonzero(~f['equal'])) == f['first'].size - 1
                     for f in fs) for fs in all_facts), 'one-cell matches absent')
    _require(any(len({(int(a), int(b)) for a, b in zip(
                 f['first'].ravel(), f['second'].ravel())}) < 4
                 for fs in all_facts for f in fs),
             'truth-table coverage is still forced in every pair')
    _require(any(any(f['fraction'] < 0.25 for f in fs[:len(automatic[i]['train'])])
                 for i, fs in enumerate(all_facts)), 'sparse training absent')
    _require(any(any(f['fraction'] > 0.75 for f in fs[:len(automatic[i]['train'])])
                 for i, fs in enumerate(all_facts)), 'dense training absent')
