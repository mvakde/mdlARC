"""Synthetic three-row path-fragment assembly puzzles."""

import numpy as np


def _rows(*rows):
    return np.array([[int(ch) for ch in row] for row in rows], dtype=np.int8)


_SOURCE_PAIRS = (
    (_rows('00000080000', '22208880000', '00200000666'), _rows('000000000', '222008666', '002888000')),
    (_rows('00033001022', '22000011002', '02000000000'), _rows('00000000', '22000122', '02331102')),
    (_rows('01100000220', '11002202200', '00000200000'), _rows('01122022', '11002220', '00000000')),
    (_rows('020000000', '220110222', '000010000'), _rows('0211000', '2201222', '0000000')),
    (_rows('02011003055', '22001033005', '00001000000'), _rows('02110000', '22010355', '00013305')),
)


def _neighbors(cell):
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _endpoints(cells):
    cells = set(cells)
    return [p for p in cells if sum(q in cells for q in _neighbors(p)) == 1]


def _is_simple_path(cells):
    cells = set(cells)
    if len(cells) < 2:
        return False
    seen = {next(iter(cells))}
    stack = list(seen)
    while stack:
        for q in _neighbors(stack.pop()):
            if q in cells and q not in seen:
                seen.add(q)
                stack.append(q)
    return len(seen) == len(cells) and len(_endpoints(cells)) == 2 and all(
        1 <= sum(q in cells for q in _neighbors(p)) <= 2 for p in cells
    )


def _build_shapes():
    library = {}
    for width in range(2, 6):
        shapes = []
        for mask in range(1, 1 << (3 * width)):
            cells = {(i // width, i % width) for i in range(3 * width) if mask & (1 << i)}
            if not cells or min(r for r, _ in cells) or min(c for _, c in cells):
                continue
            if max(c for _, c in cells) != width - 1 or not _is_simple_path(cells):
                continue
            ends = _endpoints(cells)
            left = [p for p in ends if p[1] == 0]
            right = [p for p in ends if p[1] == width - 1]
            if len(left) != 1 or len(right) != 1:
                continue
            shapes.append((tuple(sorted(cells)), max(r for r, _ in cells) + 1, left[0][0], right[0][0]))
        library[width] = tuple(shapes)
    return library


_SHAPES = _build_shapes()


def _grid_key(grid):
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


def _pair_key(pair):
    return (_grid_key(pair['input']), _grid_key(pair['output']))


def _is_source_pair(pair):
    return any(np.array_equal(pair['input'], a) and np.array_equal(pair['output'], b) for a, b in _SOURCE_PAIRS)


def _component_infos(grid, background):
    unseen = set(map(tuple, np.argwhere(grid != background)))
    infos = []
    while unseen:
        start = unseen.pop()
        cells = {start}
        stack = [start]
        while stack:
            for q in _neighbors(stack.pop()):
                if q in unseen:
                    unseen.remove(q)
                    cells.add(q)
                    stack.append(q)
        assert _is_simple_path(cells), 'each fragment must be an unbranched path'
        colors = {int(grid[r, c]) for r, c in cells}
        assert len(colors) == 1, 'fragments must be monochromatic'
        r0, r1 = min(r for r, _ in cells), max(r for r, _ in cells)
        c0, c1 = min(c for _, c in cells), max(c for _, c in cells)
        ends = _endpoints(cells)
        left = [p for p in ends if p[1] == c0]
        right = [p for p in ends if p[1] == c1]
        assert c1 > c0 and len(left) == len(right) == 1, 'one endpoint is needed on each extreme column'
        infos.append({
            'cells': cells, 'color': next(iter(colors)), 'min_row': r0, 'min_col': c0,
            'max_col': c1, 'height': r1-r0+1, 'width': c1-c0+1,
            'left_row': left[0][0]-r0, 'right_row': right[0][0]-r0,
            'relative_cells': tuple(sorted((r-r0, c-c0) for r, c in cells)),
        })
    infos.sort(key=lambda info: info['min_col'])
    return infos


def _analyze_input(grid, background):
    assert grid.shape[0] == 3, 'three-row input required'
    infos = _component_infos(grid, background)
    assert 2 <= len(infos) <= 5, 'two to five separate fragments required'
    assert all(2 <= info['width'] <= 5 for info in infos), 'fragment width out of range'
    gaps = [b['min_col'] - a['max_col'] - 1 for a, b in zip(infos, infos[1:])]
    assert all(g >= 1 for g in gaps), 'fragment bounding boxes must have blank separator columns'
    assert infos[0]['min_col'] >= 0 and infos[-1]['max_col'] < grid.shape[1]
    width = sum(info['width'] for info in infos)
    assert width < grid.shape[1] <= 30, 'input must include removable blank columns'
    output = np.full((3, width), background, dtype=np.int8)
    tops = []
    x = 0
    prev_right = None
    for i, info in enumerate(infos):
        top = info['min_row'] if i == 0 else prev_right - info['left_row']
        assert 0 <= top <= 3-info['height'], 'endpoint alignment must fit the three-row canvas'
        for r, c in info['relative_cells']:
            assert output[top+r, x+c] == background, 'fragments must not overlap'
            output[top+r, x+c] = info['color']
        tops.append(top)
        prev_right = top + info['right_row']
        x += info['width']
    assert _is_simple_path(set(map(tuple, np.argwhere(output != background)))), 'output must be a single path'
    return {'components': infos, 'background': background, 'gaps': gaps,
            'leading': infos[0]['min_col'], 'trailing': grid.shape[1]-1-infos[-1]['max_col'],
            'tops': tops, 'shifts': [tops[i]-infos[i]['min_row'] for i in range(1, len(infos))],
            'output': output}


def _input_solutions(grid):
    candidates = {int(grid[0, c]) for c in range(grid.shape[1]) if np.all(grid[:, c] == grid[0, c])}
    solutions = []
    for bg in candidates:
        try:
            solutions.append(_analyze_input(grid, bg))
        except AssertionError:
            pass
    return solutions


def _construct_pair(rng, regime, background, palette, shift='any', source_like=False,
                    require_repeat=False, novel_color=None, show_both_margins=False):
    for _ in range(1200):
        count = 4 if source_like else int(rng.choice((2, 3, 4, 5), p=(0.10, 0.34, 0.38, 0.18)))
        if regime == 'source_train' and shift in ('up', 'down') and count == 2:
            count = 3
        width_weights = (0.59, 0.27, 0.11, 0.03) if regime == 'source_train' else (0.31, 0.32, 0.25, 0.12)
        widths = [2] * 4 if source_like else [int(rng.choice((2, 3, 4, 5), p=width_weights)) for _ in range(count)]
        output_width = sum(widths)
        if output_width + count - 1 > 29:
            continue
        shapes, tops, assembled = [], [], set()
        x = 0
        failed = False
        for i, width in enumerate(widths):
            choices = []
            for shape in _SHAPES[width]:
                cells, height, left, right = shape
                possible_tops = range(4-height) if i == 0 else (tops[-1] + shapes[-1][3] - left,)
                for top in possible_tops:
                    if not 0 <= top <= 3-height:
                        continue
                    new = {(r+top, c+x) for r, c in cells}
                    if _is_simple_path(assembled | new):
                        choices.append((shape, top, new))
            if not choices:
                failed = True
                break
            # Height is sampled separately from width so broad shapes do not swamp narrow ones.
            heights = sorted({choice[0][1] for choice in choices})
            height_weights = (0.32, 0.55, 0.13) if regime == 'source_train' else (0.17, 0.47, 0.36)
            weights = np.array([height_weights[h-1] for h in heights])
            weights /= weights.sum()
            height = heights[int(rng.choice(len(heights), p=weights))]
            subset = [choice for choice in choices if choice[0][1] == height]
            shape, top, new = subset[int(rng.integers(len(subset)))]
            shapes.append(shape)
            tops.append(top)
            assembled |= new
            x += width
        if failed:
            continue
        input_tops = tops.copy()
        for i, shape in enumerate(shapes[1:], 1):
            input_tops[i] = int(rng.integers(0, 4-shape[1]))
        shifts = [tops[i]-input_tops[i] for i in range(1, count)]
        if shift == 'none':
            input_tops = tops.copy()
        elif shift == 'up' and not any(s < 0 for s in shifts):
            continue
        elif shift == 'down' and not any(s > 0 for s in shifts):
            continue
        elif shift == 'mixed' and not (any(s < 0 for s in shifts) and any(s > 0 for s in shifts)):
            continue
        gaps = [1] * 3 if source_like else [int(rng.integers(1, 5)) for _ in range(count-1)]
        spare = 30-output_width-sum(gaps)
        if spare < (2 if show_both_margins else 0):
            continue
        if show_both_margins:
            leading = int(rng.integers(1, min(3, spare-1)+1))
            trailing = int(rng.integers(1, min(3, spare-leading)+1))
        elif source_like and rng.integers(2) == 0:
            leading = trailing = 0
        else:
            leading = int(rng.integers(0, min(3, spare)+1))
            trailing = int(rng.integers(0, min(3, spare-leading)+1))
        input_width = output_width + sum(gaps) + leading + trailing
        colors = [int(palette[int(rng.integers(len(palette)))]) for _ in range(count)]
        if require_repeat:
            j = int(rng.integers(1, count))
            colors[j] = colors[j-1]
        if novel_color is not None:
            colors[int(rng.integers(count))] = novel_color
        if background in colors:
            continue
        input_grid = np.full((3, input_width), background, dtype=np.int8)
        output_grid = np.full((3, output_width), background, dtype=np.int8)
        ix, ox = leading, 0
        for i, (shape, color) in enumerate(zip(shapes, colors)):
            for r, c in shape[0]:
                input_grid[input_tops[i]+r, ix+c] = color
                output_grid[tops[i]+r, ox+c] = color
            ix += widths[i]
            ox += widths[i]
            if i < count-1:
                ix += gaps[i]
        pair = {'input': input_grid, 'output': output_grid}
        solutions = _input_solutions(input_grid)
        if len(solutions) != 1 or not np.array_equal(solutions[0]['output'], output_grid) or _is_source_pair(pair):
            continue
        return pair
    raise ValueError('unable to construct a valid pair')


def _resolve_counts(seed, rng, num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high):
            raise ValueError(f'{name} must be an integer from {low} to {high}')
    if num_train is None and num_test is None:
        test = 1 + seed % 4
        train = max(2, test+1) + (seed//4) % (11-max(2, test+1))
    elif num_train is None:
        test = int(num_test)
        train = int(rng.integers(max(2, test+1), 11))
    elif num_test is None:
        train = int(num_train)
        test = int(rng.integers(1, min(4, train-1)+1))
    else:
        train, test = int(num_train), int(num_test)
    if train <= test:
        raise ValueError('num_train must exceed num_test')
    return train, test


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    base_background = int(rng.integers(10))
    variable_background = bool(rng.integers(2))
    train_palette = tuple(int(c) for c in rng.choice([c for c in range(10) if c != base_background], size=int(rng.integers(4, 7)), replace=False))
    puzzle = {'train': [], 'test': []}
    used_inputs = set()
    used_colors = set()
    # The first two demonstrations establish both translation directions in variable order.
    shifts = ['up', 'down']
    if rng.integers(2):
        shifts.reverse()
    for split, count in (('train', train_count), ('test', test_count)):
        for index in range(count):
            for _ in range(80):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63, dtype=np.int64)))
                bg = int(pair_rng.integers(10)) if variable_background else base_background
                available = [c for c in train_palette if c != bg]
                if split == 'train':
                    shift = shifts[index] if index < 2 else str(pair_rng.choice(('any', 'none', 'mixed'), p=(0.55, 0.25, 0.20)))
                    novel = None
                    source_like = False
                    repeat = index == 0 or bool(pair_rng.integers(3) == 0)
                else:
                    shift = str(pair_rng.choice(('any', 'none', 'mixed'), p=(0.60, 0.15, 0.25)))
                    source_like = bool(pair_rng.integers(4) == 0)
                    repeat = bool(pair_rng.integers(2))
                    unseen = [c for c in range(10) if c != bg and c not in used_colors]
                    novel = int(pair_rng.choice(unseen)) if index == 0 and unseen and pair_rng.integers(4) != 0 else None
                    if pair_rng.integers(3) == 0:
                        available = [c for c in range(10) if c != bg]
                try:
                    pair = _construct_pair(pair_rng, split == 'train' and 'source_train' or 'target_test', bg,
                                           available, shift, source_like, repeat, novel,
                                           show_both_margins=(split == 'train' and index == 0))
                except ValueError:
                    continue
                key = _grid_key(pair['input'])
                if key in used_inputs:
                    continue
                puzzle[split].append(pair)
                used_inputs.add(key)
                if split == 'train':
                    analysis = _input_solutions(pair['input'])[0]
                    used_colors.update(info['color'] for info in analysis['components'])
                break
            else:
                raise ValueError('unable to create a unique pair')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[x], list) for x in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    inputs, pairs = set(), set()
    training_shifts = []
    training_margins = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
            key = _pair_key(pair)
            assert key[0] not in inputs and key not in pairs
            inputs.add(key[0])
            pairs.add(key)
            assert not _is_source_pair(pair)
            solutions = _input_solutions(pair['input'])
            assert len(solutions) == 1, 'background and assembly must be unique from the input'
            analysis = solutions[0]
            assert np.array_equal(pair['output'], analysis['output']), 'incorrect endpoint assembly'
            assert pair['output'].shape[0] == 3
            assert pair['input'].shape[1] > pair['output'].shape[1]
            assert len(analysis['gaps']) == len(analysis['components'])-1
            if split == 'train':
                training_shifts.extend(analysis['shifts'])
                training_margins.append((analysis['leading'], analysis['trailing']))
    assert any(x < 0 for x in training_shifts) and any(x > 0 for x in training_shifts), 'training must show both translation directions'
    assert any(lead > 0 and trail > 0 for lead, trail in training_margins), 'training must show both exterior margins'


def _puzzle_key(puzzle):
    return tuple(tuple(_pair_key(p) for p in puzzle[split]) for split in ('train', 'test'))


def self_test() -> None:
    automatic = [generate_puzzle(i) for i in range(42)]
    assert len({len(p['train']) for p in automatic}) >= 4
    assert len({len(p['test']) for p in automatic}) >= 3
    assert len({_puzzle_key(p) for p in automatic}) >= 40
    for p in automatic:
        validate_generated_puzzle(p)
    for i, (a, b) in enumerate(((2, 1), (10, 1), (10, 4), (6, 3), (None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(100+i, a, b)
        assert (a is None or len(p['train']) == a) and (b is None or len(p['test']) == b)
        validate_generated_puzzle(p)
    assert _puzzle_key(generate_puzzle(987654, 7, 3)) == _puzzle_key(generate_puzzle(987654, 7, 3))
    for kwargs in ({'num_train': 1}, {'num_train': 11}, {'num_test': 0}, {'num_test': 5}, {'num_train': 2, 'num_test': 2}):
        try:
            generate_puzzle(999, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    all_pairs = [(s, p) for puzzle in automatic for s in ('train', 'test') for p in puzzle[s]]
    analyses = [(s, _input_solutions(p['input'])[0]) for s, p in all_pairs]
    assert len({_grid_key(p['input']) for _, p in all_pairs}) >= 150
    assert len({(p['input'].shape, p['output'].shape) for _, p in all_pairs}) >= 20
    assert len({tuple(info['relative_cells']) for _, a in analyses for info in a['components']}) >= 35
    assert any(a['leading'] > 0 for _, a in analyses) and any(a['trailing'] > 0 for _, a in analyses)
    assert any(max(info['height'] for info in a['components']) == 3 for s, a in analyses if s == 'train')
    assert any(sum(info['height'] == 3 for info in a['components']) >= 2 for _, a in analyses)
    assert any(all(shift == 0 for shift in a['shifts']) for _, a in analyses)
    assert len({len(a['components']) for s, a in analyses if s == 'test'}) >= 3
    assert len({a['output'].shape[1] for s, a in analyses if s == 'test'}) >= 6
    assert any(len({i['color'] for i in a['components']}) < len(a['components']) for s, a in analyses if s == 'test')
    assert {len(a['components']) for s, a in analyses if s == 'test'} == {2, 3, 4, 5}
    assert {g for s, a in analyses if s == 'test' for g in a['gaps']} == {1, 2, 3, 4}
    assert {c['width'] for s, a in analyses if s == 'test' for c in a['components']} == {2, 3, 4, 5}
    assert {min(2, sum(c['height'] == 3 for c in a['components'])) for s, a in analyses if s == 'test'} == {0, 1, 2}
    assert sum(a['leading'] > 0 and a['trailing'] > 0 for s, a in analyses if s == 'test') >= 10
    assert sum(len(set(a['gaps'])) > 1 for s, a in analyses if s == 'test') >= 15
    assert sum(all(x == 0 for x in a['shifts']) for s, a in analyses if s == 'test') >= 5
    assert sum(len({c['color'] for c in a['components']}) < len(a['components']) for s, a in analyses if s == 'test') >= 15
    assert sum(any(x['color'] == y['color'] for x, y in zip(a['components'], a['components'][1:])) for s, a in analyses if s == 'test') >= 10
    assert sum(p['input'].shape == (3, 11) and p['output'].shape == (3, 8) for s, p in all_pairs if s == 'test') >= 2
    assert sum(len({int(_input_solutions(p['input'])[0]['background']) for p in puzzle['train'] + puzzle['test']}) > 1 for puzzle in automatic) >= 10
    assert sum(len({int(_input_solutions(p['input'])[0]['background']) for p in puzzle['train'] + puzzle['test']}) == 1 for puzzle in automatic) >= 10
    assert sum(len({len(_input_solutions(p['input'])[0]['components']) for p in puzzle['train']}) > 1 for puzzle in automatic) >= 30
    assert sum(len({p['output'].shape[1] for p in puzzle['train']}) > 1 for puzzle in automatic) >= 30
    assert sum(any(c['color'] not in {
        info['color'] for train_pair in puzzle['train']
        for info in _input_solutions(train_pair['input'])[0]['components']
    } for test_pair in puzzle['test'] for c in _input_solutions(test_pair['input'])[0]['components']) for puzzle in automatic) >= 10
