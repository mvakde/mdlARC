"""Layer reversal puzzles: complete visible regular shapes through foreground masks."""

import hashlib
import numpy as np


_SOURCE_HASHES = {
    'd74a2881bc1275daf18a3448446bdd27d1d1d4845cf25dcec35a6b9ed60a6209',
    'f8f3fd518e1fa1375c9a0295b661376d80cc259c6a731f65ce048ae0c3b744a3',
    '9c3edbeef028213cb5d9f533ec31500c240261387a0eccc8e7f0e21fdf5561e1',
    'cb7467dd30555b9a3368c76d78703e4462ced02399af5cf00523e5605b6fa528',
}


def _integer(rng, low, high):
    return int(rng.integers(low, high + 1))


def _box(shape, r0, r1, c0, c1):
    result = np.zeros(shape, dtype=bool)
    result[r0:r1 + 1, c0:c1 + 1] = True
    return result


def _counts(rng, num_train, num_test):
    for name, value, lo, hi in (
        ('num_train', num_train, 2, 10),
        ('num_test', num_test, 1, 4),
    ):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not lo <= value <= hi):
            raise ValueError(name)
    if num_train is None and num_test is None:
        num_test = _integer(rng, 1, 4)
        num_train = _integer(rng, max(2, num_test + 1), 10)
    elif num_train is None:
        num_train = _integer(rng, max(2, num_test + 1), 10)
    elif num_test is None:
        num_test = _integer(rng, 1, min(4, num_train - 1))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _axis_shapes(rng, category):
    multi = category == 'multi'
    height = _integer(rng, 19 if multi else 14, 30)
    width = _integer(rng, 19 if multi else 14, 30)
    shape = (height, width)
    targets = []
    masks = []
    if not multi:
        bar = category == 'bar'
        thickness = _integer(rng, 1, 3) if bar else _integer(rng, 3, min(9, height - 5))
        span = _integer(rng, 8 if bar else 5, width - 1)
        r0 = _integer(rng, 2, height - thickness - 2)
        c0 = _integer(rng, 0, width - span)
        r1, c1 = r0 + thickness - 1, c0 + span - 1
        targets.append(_box(shape, r0, r1, c0, c1))
        side_clip = not bar and thickness >= 4 and bool(rng.integers(3) == 0)
        if side_clip:
            overlap = _integer(rng, 1, max(1, span // 2))
            extension = _integer(rng, 1, 3)
            mr0 = _integer(rng, r0 + 1, r1 - 1)
            mr1 = min(height - 1, r1 + _integer(rng, 1, 3))
            if c0 >= extension and (c1 + extension >= width or rng.integers(2)):
                mc0, mc1 = c0 - extension, c0 + overlap - 1
            else:
                mc0, mc1 = c1 - overlap + 1, min(width - 1, c1 + extension)
        else:
            gap = _integer(rng, 1, min(5, span - 4))
            mc0 = _integer(rng, c0 + 2, c1 - gap - 1)
            mc1 = mc0 + gap - 1
            mr0 = max(0, r0 - _integer(rng, 1, 4))
            mr1 = min(height - 1, r1 + _integer(rng, 1, 4))
        masks.append(_box(shape, mr0, mr1, mc0, mc1))
    else:
        count = 3 if height >= 25 and rng.integers(5) == 0 else 2
        slot = (height - 4) // count
        bounds = []
        for index in range(count):
            thickness = _integer(rng, 1, min(5, slot - 1))
            start = 2 + index * slot
            r0 = _integer(rng, start, start + slot - thickness)
            r1 = r0 + thickness - 1
            span = _integer(rng, 10, width)
            c0 = _integer(rng, 0, width - span)
            c1 = c0 + span - 1
            targets.append(_box(shape, r0, r1, c0, c1))
            bounds.append((r0, r1))
        mask_count = _integer(rng, 1, 3)
        intervals = []
        for _ in range(mask_count):
            gap = _integer(rng, 1, min(5, width - 7))
            options = [col for col in range(3, width - gap - 2)
                       if all(col > old1 + 1 or col + gap - 1 < old0 - 1
                              for old0, old1 in intervals)]
            if not options:
                raise AssertionError('no space for mask')
            mc0 = int(rng.choice(options))
            mc1 = mc0 + gap - 1
            intervals.append((mc0, mc1))
            if rng.integers(3) == 0:
                chosen = _integer(rng, 0, count - 1)
                mr0 = max(0, bounds[chosen][0] - _integer(rng, 1, 2))
                mr1 = min(height - 1, bounds[chosen][1] + _integer(rng, 1, 2))
            else:
                first = _integer(rng, 0, count - 2)
                last = _integer(rng, first + 1, count - 1)
                mr0 = max(0, bounds[first][0] - _integer(rng, 1, 3))
                mr1 = min(height - 1, bounds[last][1] + _integer(rng, 1, 3))
            masks.append(_box(shape, mr0, mr1, mc0, mc1))
    if rng.integers(2):
        targets = [part.T.copy() for part in targets]
        masks = [part.T.copy() for part in masks]
    return targets, masks


def _slope_shapes(rng, include_rect=True):
    height = _integer(rng, 19, 29)
    width = min(30, height + _integer(rng, -1, 3))
    shape = (height, width)
    line = np.zeros(shape, dtype=bool)
    line_r0 = _integer(rng, 1, 3)
    line_c0 = _integer(rng, 1, 3)
    length = min(height - line_r0 - 1, width - line_c0 - 1) - _integer(rng, 0, 2)
    for step in range(length):
        line[line_r0 + step, line_c0 + step] = True
    targets = [line]
    mode = int(rng.integers(3)) if include_rect else int(rng.integers(2))
    if include_rect:
        rect_height = _integer(rng, 4, 8)
        rect_width = _integer(rng, 3, 6)
        rect_c0 = width - rect_width - 2
        if mode == 2:
            rect_r0 = height - rect_height - _integer(rng, 2, 3)
        else:
            rect_r0 = _integer(rng, 3, min(8, height - rect_height - 3))
        targets.append(_box(shape, rect_r0, rect_r0 + rect_height - 1,
                            rect_c0, rect_c0 + rect_width - 1))
    top = _integer(rng, 5, 8)
    if mode == 1:
        bottom = min(height - 8, top + _integer(rng, 5, 8))
    elif mode == 2:
        bottom = min(height - 10, top + _integer(rng, 5, 8))
    else:
        bottom = height - _integer(rng, 4, 6)
    if bottom < top + 4:
        raise AssertionError('wedge too short')
    left_start = width // 2 + _integer(rng, -2, 2)
    right_start = (rect_c0 - _integer(rng, 0, 2)) if include_rect else width - _integer(rng, 5, 7)
    left_step = _integer(rng, 1, 3)
    right_step = _integer(rng, 0, 3)
    wedge = np.zeros(shape, dtype=bool)
    for row in range(top, bottom + 1):
        distance = row - top
        left = max(1, left_start - distance * left_step // 2)
        right = min(width - 2, right_start + distance * right_step // 2)
        if left <= right:
            wedge[row, left:right + 1] = True
    masks = [wedge]
    if mode == 1:
        row = bottom + _integer(rng, 2, 3)
        col = line_c0 + row - line_r0
        if row + 1 < height - 3 and col + 2 < width:
            masks.append(_box(shape, row, row + 1, max(0, col - 1), min(width - 1, col + 2)))
    elif mode == 2:
        row = rect_r0 + _integer(rng, 1, rect_height - 2)
        masks.append(_box(shape, row, min(height - 1, row + 1),
                          rect_c0 - 1, width - 1))
    if rng.integers(2):
        targets = [np.fliplr(part) for part in targets]
        masks = [np.fliplr(part) for part in masks]
    if rng.integers(2):
        targets = [np.flipud(part) for part in targets]
        masks = [np.flipud(part) for part in masks]
    if rng.integers(2):
        targets = [part.T.copy() for part in targets]
        masks = [part.T.copy() for part in masks]
    return targets, masks


def _paint(rng, background, targets, masks, preferred=None):
    shape = targets[0].shape
    assert all(part.shape == shape and np.any(part) for part in targets + masks)
    used = np.zeros(shape, dtype=bool)
    for part in targets:
        assert not np.any(used & part)
        used |= part
    used[:] = False
    for part in masks:
        assert not np.any(used & part)
        used |= part
    assert all(np.any(target & np.logical_or.reduce(masks)) for target in targets)
    palette = [color for color in range(10) if color != background]
    colors = list(map(int, rng.choice(palette, len(targets) + len(masks), replace=False)))
    if preferred is not None:
        first_target, first_mask = preferred
        assert first_target != background and first_mask != background
        assert first_target != first_mask
        colors = [first_target, first_mask] + [c for c in colors if c not in preferred]
        if len(colors) < len(targets) + len(masks):
            colors += [c for c in palette if c not in colors]
        colors = colors[:len(targets) + len(masks)]
        target_colors = [colors[0]] + colors[2:2 + len(targets) - 1]
        mask_colors = [colors[1]] + colors[2 + len(targets) - 1:]
    else:
        target_colors = colors[:len(targets)]
        mask_colors = colors[len(targets):]
    input_grid = np.full(shape, background, dtype=np.int8)
    for target, color in zip(targets, target_colors):
        input_grid[target] = color
    for mask, color in zip(masks, mask_colors):
        input_grid[mask] = color
    output_grid = np.full(shape, background, dtype=np.int8)
    for mask, color in zip(masks, mask_colors):
        output_grid[mask] = color
    for target, color in zip(targets, target_colors):
        output_grid[target] = color
    return {'input': input_grid, 'output': output_grid}


def _signature(pair):
    first = pair['input'].astype(np.uint8, copy=False)
    second = pair['output'].astype(np.uint8, copy=False)
    return hashlib.sha256(bytes(first.shape) + first.tobytes() + second.tobytes()).hexdigest()


def _reconstruct(input_grid, output_grid):
    """Infer completed footprints from visible colors, without generation metadata."""
    background = int(input_grid[0, 0])
    assert all(int(input_grid[r, c]) == background for r, c in
               ((0, 0), (0, -1), (-1, 0), (-1, -1)))
    assert np.array_equal(input_grid == background, output_grid == background)
    changed = input_grid != output_grid
    assert np.any(changed)
    targets = set(map(int, np.unique(output_grid[changed])))
    masks = set(map(int, np.unique(input_grid[changed])))
    assert background not in targets | masks and not targets & masks
    assert 1 <= len(targets) <= 3 and 1 <= len(masks) <= 3
    all_colors = set(map(int, np.unique(input_grid)))
    assert all_colors == targets | masks | {background}
    prediction = input_grid.copy()
    occupied = np.zeros(input_grid.shape, dtype=bool)
    kinds = []
    hidden_counts = []
    for color in sorted(targets):
        rows, cols = np.where(input_grid == color)
        assert len(rows) >= 4
        r0, r1, c0, c1 = int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())
        diagonal = (r1 > r0 and c1 > c0 and r1 - r0 == c1 - c0
                    and (np.all(rows - cols == rows[0] - cols[0])
                         or np.all(rows + cols == rows[0] + cols[0])))
        footprint = np.zeros(input_grid.shape, dtype=bool)
        if diagonal:
            rr = np.arange(r0, r1 + 1)
            cc = rr - int(rows[0] - cols[0]) if np.all(rows - cols == rows[0] - cols[0]) else int(rows[0] + cols[0]) - rr
            assert np.all((cc >= 0) & (cc < input_grid.shape[1]))
            footprint[rr, cc] = True
            visible = input_grid[rr, cc] == color
            gaps = np.flatnonzero(~visible)
            assert len(gaps) >= 1 and gaps[0] >= 3 and len(visible) - gaps[-1] - 1 >= 3
            gap_breaks = np.flatnonzero(np.diff(gaps) > 1)
            assert all(gaps[next_index] - gaps[index] - 1 >= 2
                       for index, next_index in zip(gap_breaks, gap_breaks + 1))
            kinds.append('diagonal')
        else:
            footprint[r0:r1 + 1, c0:c1 + 1] = True
            patch = input_grid[r0:r1 + 1, c0:c1 + 1]
            assert np.any(np.all(patch == color, axis=0)) or np.any(np.all(patch == color, axis=1))
            if r1 == r0 or c1 == c0 or min(r1 - r0 + 1, c1 - c0 + 1) <= 3:
                kinds.append('bar')
            else:
                kinds.append('rectangle')
        assert not np.any(occupied & footprint)
        occupied |= footprint
        hidden = footprint & (input_grid != color)
        assert np.any(hidden)
        assert set(map(int, np.unique(input_grid[hidden]))) <= masks
        assert np.array_equal(footprint, output_grid == color)
        prediction[footprint] = color
        hidden_counts.append(int(hidden.sum()))
    assert np.array_equal(prediction, output_grid)
    for color in masks:
        cells = input_grid == color
        assert np.any(cells & ~occupied)
        rr = np.flatnonzero(np.any(cells, axis=1))
        assert len(rr) >= 2 and np.all(np.diff(rr) == 1)
        for row in rr:
            cc = np.flatnonzero(cells[row])
            assert len(cc) and np.all(np.diff(cc) == 1)
    return {'background': background, 'targets': targets, 'masks': masks,
            'kinds': kinds, 'hidden': hidden_counts}


def _new_pair(rng, background, category, preferred=None):
    for _ in range(240):
        try:
            if category == 'slope':
                targets, masks = _slope_shapes(rng, True)
            elif category == 'diagonal':
                targets, masks = _slope_shapes(rng, False)
            else:
                targets, masks = _axis_shapes(rng, category)
            pair = _paint(rng, background, targets, masks, preferred)
            _reconstruct(pair['input'], pair['output'])
            return pair
        except AssertionError:
            continue
    raise RuntimeError('Could not construct an unambiguous scene')


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    stable_palette = train_count <= 4 and test_count <= 2
    stable_background = stable_palette or bool(rng.integers(2))
    base_background = _integer(rng, 0, 9)
    anchors = list(map(int, rng.choice([c for c in range(10) if c != base_background], 2, replace=False)))
    if rng.integers(2):
        first_two = ['rect', 'bar']
    else:
        first_two = ['bar', 'rect']
    if not stable_palette and rng.integers(4) == 0:
        first_two[1] = 'multi'
    train_categories = first_two + [str(rng.choice(['rect', 'bar'] if stable_palette else
                                                   ['rect', 'bar', 'multi', 'diagonal', 'slope'],
                                                   p=None if stable_palette else
                                                   [.22, .22, .25, .14, .17]))
                                     for _ in range(train_count - 2)]
    test_categories = [str(rng.choice(['rect', 'bar'] if stable_palette else
                                      ['rect', 'bar', 'multi', 'slope'],
                                      p=None if stable_palette else [.10, .12, .43, .35]))
                       for _ in range(test_count)]
    if not stable_palette and test_count >= 2 and all(kind in ('rect', 'bar') for kind in test_categories):
        test_categories[0] = 'multi'
    puzzle = {'train': [], 'test': []}
    seen = set()
    for regime, categories in (('train', train_categories), ('test', test_categories)):
        for index, category in enumerate(categories):
            for _ in range(12):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                if stable_background:
                    background = base_background
                elif regime == 'train' and index == 0:
                    background = base_background
                else:
                    background = _integer(pair_rng, 0, 9)
                preferred = None
                if stable_palette:
                    preferred = (anchors[index % 2], anchors[1 - index % 2])
                elif regime == 'train' and index < 2 and background == base_background:
                    preferred = (anchors[index], anchors[1 - index])
                pair = _new_pair(pair_rng, background, category, preferred)
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen and _signature(pair) not in _SOURCE_HASHES:
                    seen.add(key)
                    puzzle[regime].append(pair)
                    break
            else:
                raise RuntimeError('Could not construct distinct pairs')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen_inputs = set()
    for regime in ('train', 'test'):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in (pair['input'], pair['output']):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.all((grid >= 0) & (grid <= 9))
            first, second = pair['input'], pair['output']
            assert first.shape == second.shape
            key = (first.shape, first.tobytes())
            assert key not in seen_inputs
            seen_inputs.add(key)
            assert _signature(pair) not in _SOURCE_HASHES
            facts = _reconstruct(first, second)
            if regime == 'train':
                assert any(kind in ('bar', 'rectangle', 'diagonal') for kind in facts['kinds'])
            else:
                assert facts['targets'] and facts['masks']
    training_kinds = [kind for pair in puzzle['train']
                      for kind in _reconstruct(pair['input'], pair['output'])['kinds']]
    assert any(kind in ('bar', 'rectangle') for kind in training_kinds)
    assert all('diagonal' not in _reconstruct(pair['input'], pair['output'])['kinds']
               for pair in puzzle['train'][:2])
    if len(puzzle['train']) <= 4 and len(puzzle['test']) <= 2:
        palettes = {tuple(map(int, np.unique(pair['input'])))
                    for regime in ('train', 'test') for pair in puzzle[regime]}
        assert len(palettes) == 1 and len(next(iter(palettes))) == 3
        first, second = puzzle['train'][:2]
        first_changed = first['input'] != first['output']
        second_changed = second['input'] != second['output']
        assert set(map(int, np.unique(first['output'][first_changed]))) == \
            set(map(int, np.unique(second['input'][second_changed])))
        assert set(map(int, np.unique(first['input'][first_changed]))) == \
            set(map(int, np.unique(second['output'][second_changed])))
    # The only puzzle-wide operation is geometric completion; color identities
    # and background may change by pair, as the original examples demonstrate.


def _same_puzzle(left, right):
    return all(len(left[regime]) == len(right[regime]) and
               all(np.array_equal(a[key], b[key])
                   for a, b in zip(left[regime], right[regime])
                   for key in ('input', 'output'))
               for regime in ('train', 'test'))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert all(_same_puzzle(puzzles[i], generate_puzzle(i)) for i in range(42))
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    pairs = [pair for p in puzzles for regime in ('train', 'test') for pair in p[regime]]
    assert len({pair['input'].shape for pair in pairs}) >= 30
    assert len({int(pair['input'][0, 0]) for pair in pairs}) >= 8
    assert any(pair['input'].shape == (16, 16) for pair in pairs)
    assert any(len({int(pair['input'][0, 0]) for pair in p['train'] + p['test']}) > 1
               for p in puzzles)
    assert any(len({int(pair['input'][0, 0]) for pair in p['train'] + p['test']}) == 1
               for p in puzzles)
    assert len({(pair['input'].shape, pair['input'].tobytes()) for pair in pairs}) == len(pairs)
    assert any(len(_reconstruct(pair['input'], pair['output'])['targets']) >= 2
               for pair in pairs)
    assert any('diagonal' in _reconstruct(pair['input'], pair['output'])['kinds']
               for pair in pairs)
    assert len({int((pair['input'] != pair['output']).sum()) for pair in pairs}) >= 18
    facts = [(_reconstruct(pair['input'], pair['output']), pair)
             for pair in pairs]
    assert sum('diagonal' in f['kinds'] for f, _ in facts) >= 25
    assert sum(len(f['targets']) >= 2 for f, _ in facts) >= 35
    assert sum(len(f['masks']) >= 2 for f, _ in facts) >= 25
    assert sum(sum(f['hidden']) <= 2 for f, _ in facts) >= 5
    assert sum(sum(f['hidden']) >= 12 for f, _ in facts) >= 40
    assert sum(any(color in pair['output'][0] or color in pair['output'][-1]
                   or color in pair['output'][:, 0] or color in pair['output'][:, -1]
                   for color in f['targets']) for f, pair in facts) >= 20
    assert sum(len({tuple(map(int, np.unique(pair['input'])))
                    for regime in ('train', 'test') for pair in puzzle[regime]}) == 1
               for puzzle in puzzles) >= 2
    assert sum(len({int(pair['input'][0, 0])
                    for regime in ('train', 'test') for pair in puzzle[regime]}) > 1
               for puzzle in puzzles) >= 8
    assert sum(any((pair['input'] != pair['output']).sum() <= 5
                   for regime in ('train', 'test') for pair in puzzle[regime])
               and any((pair['input'] != pair['output']).sum() >= 12
                       for regime in ('train', 'test') for pair in puzzle[regime])
               for puzzle in puzzles) >= 15
    mask_widths = set()
    wedge_count = 0
    partial_crossings = 0
    unequal_masks = 0
    for f, pair in facts:
        first, second = pair['input'], pair['output']
        widths = []
        for color in f['masks']:
            rows, cols = np.where(first == color)
            r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
            if np.all(first[r0:r1 + 1, c0:c1 + 1] == color):
                width = min(int(r1 - r0 + 1), int(c1 - c0 + 1))
                widths.append(width)
                mask_widths.add(width)
            else:
                wedge_count += 1
            if len(f['targets']) >= 2:
                crossings = sum(np.any((first == color) & (second == target))
                                for target in f['targets'])
                partial_crossings += int(crossings < len(f['targets']))
        unequal_masks += int(len(set(widths)) >= 2)
    assert {1, 2, 3, 4, 5} <= mask_widths
    assert wedge_count >= 25 and partial_crossings >= 25
    assert unequal_masks >= 10
    for counts in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(1042, *counts)
        assert counts[0] is None or len(puzzle['train']) == counts[0]
        assert counts[1] is None or len(puzzle['test']) == counts[1]
        assert _same_puzzle(puzzle, generate_puzzle(1042, *counts))
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1)):
        try:
            generate_puzzle(7, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
