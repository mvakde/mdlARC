import numpy as np


_SOURCE_INPUTS = {
    ('000000', '030300', '033000', '003330', '033030', '000000'),
    ('000000', '040040', '044400', '040440', '004040', '000000'),
    ('000000', '060066', '060600', '060600', '006066', '006660'),
    ('0000000000', '0000000000', '0700000700', '0000000000',
     '0000000000', '0000700000', '0000000000', '0000000000',
     '0700000700', '0000000000'),
    ('00000000', '00000000', '00000000', '00101100', '00000100',
     '00000000', '00100000', '00000000', '00000000', '00000000'),
}
_STYLES = ('corners', 'corner_interior', 'opposite', 'four_sides',
           'broken_outline', 'short_strokes', 'cluster_extremes',
           'scattered', 'near_interior', 'near_edge', 'mixed')


def _check_counts(rng, num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= value <= high):
            raise ValueError('Illegal pair count')
    if num_test is None:
        upper = 4 if num_train is None else min(4, int(num_train) - 1)
        num_test = int(rng.integers(1, upper + 1))
    if num_train is None:
        num_train = int(rng.integers(max(2, int(num_test) + 1), 11))
    if num_train <= num_test:
        raise ValueError('Training pair count must exceed test pair count')
    return int(num_train), int(num_test)


def _rows(grid):
    return tuple(''.join(str(int(x)) for x in row) for row in grid)


def _key(grid):
    return grid.shape, tuple(int(x) for x in grid.flat)


def _roles(grid):
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) == 2
    order = np.argsort(counts)
    assert counts[order[-1]] > counts[order[-2]]
    return int(values[order[-1]]), int(values[order[-2]])


def _bounds(grid, foreground):
    points = np.argwhere(grid == foreground)
    return (int(points[:, 0].min()), int(points[:, 0].max()),
            int(points[:, 1].min()), int(points[:, 1].max()))


def _gap_types(grid, background, foreground):
    r0, r1, c0, c1 = _bounds(grid, foreground)
    missing = grid[r0:r1 + 1, c0:c1 + 1] == background
    inner = bool(missing[1:-1, 1:-1].any())
    edge = bool(missing[0].any() or missing[-1].any()
                or missing[:, 0].any() or missing[:, -1].any())
    return inner, edge


def _ensure_extremes(rng, cells, side):
    last = side - 1
    if not any(r == 0 for r, _ in cells):
        cells.add((0, int(rng.integers(side))))
    if not any(r == last for r, _ in cells):
        cells.add((last, int(rng.integers(side))))
    if not any(c == 0 for _, c in cells):
        cells.add((int(rng.integers(side)), 0))
    if not any(c == last for _, c in cells):
        cells.add((int(rng.integers(side)), last))
    return cells


def _add_random(rng, cells, target, pool):
    available = [cell for cell in pool if cell not in cells]
    amount = min(max(0, target - len(cells)), len(available))
    if amount:
        cells.update(available[int(k)] for k in rng.choice(
            len(available), size=amount, replace=False))
    return cells


def _sample_cues(rng, side, style):
    last = side - 1
    corners = {(0, 0), (0, last), (last, 0), (last, last)}
    interior = [(r, c) for r in range(1, last) for c in range(1, last)]
    border = [(r, c) for r in range(side) for c in range(side)
              if r in (0, last) or c in (0, last)]
    all_cells = [(r, c) for r in range(side) for c in range(side)]
    if style == 'corners':
        cells = set(corners)
    elif style == 'corner_interior':
        cells = set(corners)
        _add_random(rng, cells, 4 + int(rng.integers(1, 4)), interior)
    elif style == 'opposite':
        cells = {(0, 0), (last, last)} if int(rng.integers(2)) else {
            (0, last), (last, 0)}
    elif style == 'four_sides':
        cells = {(0, int(rng.integers(1, last))),
                 (last, int(rng.integers(1, last))),
                 (int(rng.integers(1, last)), 0),
                 (int(rng.integers(1, last)), last)}
    elif style == 'broken_outline':
        count = int(rng.integers(max(4, len(border) // 2),
                                 max(5, 4 * len(border) // 5) + 1))
        cells = {border[int(k)] for k in rng.choice(
            len(border), size=count, replace=False)}
        _ensure_extremes(rng, cells, side)
        _add_random(rng, cells, len(cells) + int(rng.integers(3)), interior)
    elif style == 'short_strokes':
        cells = {(0, int(rng.integers(1, last))),
                 (last, int(rng.integers(1, last))),
                 (int(rng.integers(1, last)), 0),
                 (int(rng.integers(1, last)), last)}
        rr, cc = int(rng.integers(1, last)), int(rng.integers(1, last))
        cells.update((rr, min(last, cc + k)) for k in range(
            int(rng.integers(2, min(5, side) + 1))))
        cells.update((min(last, rr + k), cc) for k in range(
            int(rng.integers(2, min(5, side) + 1))))
    elif style == 'cluster_extremes':
        cells = {(0, int(rng.integers(side))),
                 (last, int(rng.integers(side))),
                 (int(rng.integers(side)), 0),
                 (int(rng.integers(side)), last)}
        start = max(1, side // 2 - 1)
        cluster = [(r, c) for r in range(start, min(last, start + 3))
                   for c in range(start, min(last, start + 3))]
        _add_random(rng, cells, len(cells) + int(rng.integers(2, 7)), cluster)
    elif style == 'scattered':
        cells = _ensure_extremes(rng, set(), side)
        target = max(4, int(round(float(rng.uniform(.12, .65)) * side * side)))
        _add_random(rng, cells, target, all_cells)
    elif style == 'near_interior':
        cells = set(all_cells)
        holes = int(rng.integers(1, min(5, len(interior)) + 1))
        cells.difference_update(interior[int(k)] for k in rng.choice(
            len(interior), size=holes, replace=False))
    elif style == 'near_edge':
        cells = set(all_cells)
        removable = [cell for cell in border if cell not in corners]
        holes = int(rng.integers(1, min(6, len(removable)) + 1))
        cells.difference_update(removable[int(k)] for k in rng.choice(
            len(removable), size=holes, replace=False))
    elif style == 'mixed':
        cells = set(corners)
        target = int(rng.integers(max(5, side * side // 3),
                                  max(6, 2 * side * side // 3) + 1))
        _add_random(rng, cells, target, all_cells)
        if all(cell in cells for cell in interior):
            cells.remove(interior[int(rng.integers(len(interior)))])
        optional_edge = [cell for cell in border if cell not in corners]
        if all(cell in cells for cell in optional_edge):
            cells.remove(optional_edge[int(rng.integers(len(optional_edge)))])
    else:
        raise ValueError('Unknown cue style')
    return cells


def _extra(rng, side, minimum):
    cap = 30 - side
    if float(rng.random()) < .75:
        cap = min(cap, max(minimum, side // 2 + 4))
    return int(rng.integers(minimum, cap + 1))


def _sample_geometry(rng, side, count):
    for _ in range(60):
        mode = int(rng.integers(5))
        if mode == 0:  # Four interior margins.
            eh, ew = _extra(rng, side, 2), _extra(rng, side, 2)
            top, left = int(rng.integers(1, eh)), int(rng.integers(1, ew))
        elif mode == 1:  # Exactly one touched edge.
            eh, ew = _extra(rng, side, 2), _extra(rng, side, 2)
            if int(rng.integers(2)):
                top = 0 if int(rng.integers(2)) else eh
                left = int(rng.integers(1, ew))
            else:
                top = int(rng.integers(1, eh))
                left = 0 if int(rng.integers(2)) else ew
        elif mode == 2:  # Two adjacent touched edges.
            eh, ew = _extra(rng, side, 1), _extra(rng, side, 1)
            top = 0 if int(rng.integers(2)) else eh
            left = 0 if int(rng.integers(2)) else ew
        elif mode == 3:  # Square spans one canvas dimension.
            extra = _extra(rng, side, 2)
            if int(rng.integers(2)):
                eh, ew, top = 0, extra, 0
                left = int(rng.integers(1, extra))
            else:
                eh, ew, left = extra, 0, 0
                top = int(rng.integers(1, extra))
        else:
            eh, ew = _extra(rng, side, 2), _extra(rng, side, 2)
            top, left = int(rng.integers(eh + 1)), int(rng.integers(ew + 1))
        h, w = side + eh, side + ew
        if h * w > 2 * count:
            return h, w, top, left
    return 30, 30, int(rng.integers(31 - side)), int(rng.integers(31 - side))


def _sample_colors(rng, num_train, num_test):
    palette = [int(x) for x in rng.permutation(np.arange(1, 10))]
    zero_foreground = bool(rng.integers(2))
    train = [(0, palette[0]),
             (palette[1], 0 if zero_foreground else palette[2])]
    for index in range(2, num_train):
        foreground_options = ([0] if zero_foreground else []) + palette[:5]
        foreground = foreground_options[int(rng.integers(len(foreground_options)))]
        background_options = [x for x in range(10) if x != foreground]
        background = background_options[int(rng.integers(len(background_options)))]
        train.append((background, foreground))
    used = {fg for _, fg in train}
    new = [x for x in palette if x not in used]
    test = []
    for index in range(num_test):
        if index == 0 and num_test >= 2:
            foreground = train[int(rng.integers(num_train))][1]
        elif index == 1 and num_test >= 2:
            foreground = new[int(rng.integers(len(new)))]
        elif int(rng.integers(2)):
            foreground = train[int(rng.integers(num_train))][1]
        else:
            foreground = new[int(rng.integers(len(new)))]
        background_options = [x for x in range(10) if x != foreground]
        background = background_options[int(rng.integers(len(background_options)))]
        test.append((background, foreground))
    return train, test


def _make_pair(rng, side, style, background, foreground):
    cells = _sample_cues(rng, side, style)
    h, w, row, col = _sample_geometry(rng, side, len(cells))
    inp = np.full((h, w), background, dtype=np.int8)
    out = np.full((h, w), background, dtype=np.int8)
    for r, c in cells:
        inp[row + r, col + c] = foreground
    out[row:row + side, col:col + side] = foreground
    return {'input': inp, 'output': out}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _check_counts(rng, num_train, num_test)
    train_colors, test_colors = _sample_colors(rng, num_train, num_test)
    first_side = int(rng.integers(4, 9))
    puzzle = {'train': [], 'test': []}
    seen = set()
    for regime, count, colors in (('train', num_train, train_colors),
                                  ('test', num_test, test_colors)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            for _ in range(80):
                if regime == 'train' and index == 0:
                    side, style = first_side, 'mixed'
                else:
                    maximum = 18 if regime == 'train' else 22
                    side = int(pair_rng.integers(4, maximum + 1))
                    if regime == 'train' and index == 1 and side == first_side:
                        side = 4 + (side - 3) % (maximum - 3)
                    style = _STYLES[int(pair_rng.integers(len(_STYLES)))]
                    if side == 22 and style in ('near_interior', 'near_edge'):
                        side = 21
                background, foreground = colors[index]
                pair = _make_pair(pair_rng, side, style, background, foreground)
                key = _key(pair['input'])
                if key in seen or _rows(pair['input']) in _SOURCE_INPUTS:
                    continue
                seen.add(key)
                puzzle[regime].append(pair)
                break
            else:
                raise RuntimeError('Unable to construct a distinct pair')
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    n_train, n_test = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= n_train <= 10 and 1 <= n_test <= 4 and n_train > n_test
    seen_inputs, seen_pairs = set(), set()
    train_sides, train_foregrounds, train_backgrounds = set(), set(), set()
    test_foregrounds = []
    for regime in ('train', 'test'):
        for index, pair in enumerate(puzzle[regime]):
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = pair['input'], pair['output']
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((grid >= 0) & (grid <= 9)))
            assert inp.shape == out.shape
            input_key, output_key = _key(inp), _key(out)
            assert input_key not in seen_inputs
            assert (input_key, output_key) not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add((input_key, output_key))
            assert _rows(inp) not in _SOURCE_INPUTS
            background, foreground = _roles(inp)
            r0, r1, c0, c1 = _bounds(inp, foreground)
            side = r1 - r0 + 1
            assert 4 <= side <= 22 and side == c1 - c0 + 1
            assert int((inp == foreground).sum()) < side * side
            expected = np.full(inp.shape, background, dtype=np.int8)
            expected[r0:r1 + 1, c0:c1 + 1] = foreground
            assert np.array_equal(out, expected) and not np.array_equal(inp, out)
            if regime == 'train':
                assert side <= 18
                train_sides.add(side)
                train_backgrounds.add(background)
                train_foregrounds.add(foreground)
                if index == 0:
                    assert 4 <= side <= 8
                    assert _gap_types(inp, background, foreground) == (True, True)
            else:
                test_foregrounds.append(foreground)
    assert len(train_sides) >= 2
    assert 0 in train_backgrounds and any(x != 0 for x in train_backgrounds)
    assert all(fg != 0 or 0 in train_foregrounds for fg in test_foregrounds)
    if n_test >= 2:
        assert any(fg in train_foregrounds for fg in test_foregrounds)
        assert any(fg not in train_foregrounds for fg in test_foregrounds)


def _puzzle_equal(first, second):
    if set(first) != set(second):
        return False
    for regime in ('train', 'test'):
        if len(first[regime]) != len(second[regime]):
            return False
        for a, b in zip(first[regime], second[regime]):
            if not np.array_equal(a['input'], b['input']):
                return False
            if not np.array_equal(a['output'], b['output']):
                return False
    return True


def self_test() -> None:
    train_counts, test_counts, signatures = set(), set(), set()
    sizes, sides, layouts, backgrounds = set(), set(), set(), set()
    densities, gaps, edges, cue_counts = set(), set(), set(), set()
    sparse_train = zero_foreground = reused_test = new_test = 0
    review = {name: 0 for name in (
        'four_corners', 'corner_plus_interior', 'opposite_corners',
        'four_sides', 'broken_outline', 'short_runs', 'interior_holes_only',
        'edge_gaps_only', 'one_edge', 'spans_dimension', 'unequal_margins',
        'large_canvas', 'large_square', 'rectangular_canvas',
        'dense', 'zero_foreground_test', 'large_sparse_train',
        'smaller_test', 'denser_test', 'familiar_test_side',
        'familiar_test_position', 'sparse_one_edge_nonzero_background')}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _puzzle_equal(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        signatures.add(tuple(_key(pair['input']) for regime in ('train', 'test')
                             for pair in puzzle[regime]))
        train_sides, train_boxes, train_fg = set(), set(), set()
        train_positions, train_densities, train_canvases = set(), [], set()
        for regime in ('train', 'test'):
            for pair in puzzle[regime]:
                inp, out = pair['input'], pair['output']
                background, foreground = _roles(inp)
                points = np.argwhere(inp == foreground)
                lo, hi = points.min(axis=0), points.max(axis=0)
                r0, c0 = (int(x) for x in lo)
                r1, c1 = (int(x) for x in hi)
                side = r1 - r0 + 1
                predicted = inp.copy()
                predicted[r0:r1 + 1, c0:c1 + 1] = foreground
                assert np.array_equal(predicted, out)
                assert side == c1 - c0 + 1 and not np.array_equal(inp, out)
                assert _rows(inp) not in _SOURCE_INPUTS
                sizes.add(inp.shape)
                sides.add(side)
                layouts.add((inp.shape, r0, c0, side))
                backgrounds.add(background)
                cue_count = len(points)
                cue_counts.add(cue_count)
                density = cue_count / (side * side)
                densities.add(int(density * 10))
                gaps.add(_gap_types(inp, background, foreground))
                margins = (r0, inp.shape[0] - r1 - 1,
                           c0, inp.shape[1] - c1 - 1)
                touched = sum(x == 0 for x in margins)
                edges.add(touched)
                local = inp[r0:r1 + 1, c0:c1 + 1] == foreground
                border = np.zeros((side, side), dtype=bool)
                border[0] = border[-1] = True
                border[:, 0] = border[:, -1] = True
                inner_gap, edge_gap = _gap_types(inp, background, foreground)
                corners = {(0, 0), (0, side - 1),
                           (side - 1, 0), (side - 1, side - 1)}
                cue_positions = set(map(tuple, np.argwhere(local)))
                review['four_corners'] += cue_positions == corners
                review['corner_plus_interior'] += (
                    cue_count == 5 and corners.issubset(cue_positions))
                review['opposite_corners'] += (
                    cue_count == 2 and cue_positions in (
                        {(0, 0), (side - 1, side - 1)},
                        {(0, side - 1), (side - 1, 0)}))
                review['four_sides'] += (
                    cue_count == 4 and not cue_positions & corners
                    and local[0].any() and local[-1].any()
                    and local[:, 0].any() and local[:, -1].any())
                review['broken_outline'] += (
                    cue_count >= 5 and not local[1:-1, 1:-1].any()
                    and edge_gap)
                review['short_runs'] += (
                    bool(np.any(local[:, :-1] & local[:, 1:]))
                    and bool(np.any(local[:-1, :] & local[1:, :]))
                    and edge_gap)
                review['interior_holes_only'] += inner_gap and not edge_gap
                review['edge_gaps_only'] += edge_gap and not inner_gap
                review['one_edge'] += touched == 1
                review['spans_dimension'] += (
                    margins[0] == margins[1] == 0
                    or margins[2] == margins[3] == 0)
                review['unequal_margins'] += len(set(margins)) >= 3
                review['large_canvas'] += max(inp.shape) > 14
                review['large_square'] += side > 9
                review['rectangular_canvas'] += inp.shape[0] != inp.shape[1]
                review['dense'] += density >= .75
                review['sparse_one_edge_nonzero_background'] += (
                    density <= .25 and touched == 1 and background != 0)
                if regime == 'train':
                    train_sides.add(side)
                    train_boxes.add((r0, r1, c0, c1))
                    train_positions.add((r0, c0))
                    train_densities.append(density)
                    train_canvases.add(inp.shape)
                    train_fg.add(foreground)
                    sparse_train += density <= .25
                    zero_foreground += foreground == 0
                    review['large_sparse_train'] += (
                        side >= 10 and density <= .25 and all(x > 0 for x in margins))
                else:
                    reused_test += foreground in train_fg
                    new_test += foreground not in train_fg
                    review['zero_foreground_test'] += foreground == 0
                    review['smaller_test'] += side < max(train_sides)
                    review['denser_test'] += density > min(train_densities)
                    review['familiar_test_side'] += side in train_sides
                    review['familiar_test_position'] += (r0, c0) in train_positions
        assert len(train_sides) >= 2 and len(train_boxes) >= 2
        assert len(train_canvases) >= 2 and len(set(train_densities)) >= 2
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) >= 40 and len(layouts) >= 100
    assert len(sizes) >= 45 and len(sides) >= 15
    assert len(backgrounds) >= 8 and len(cue_counts) >= 25
    assert len(densities) >= 8 and len(gaps) >= 3
    assert {0, 1, 2}.issubset(edges)
    assert sparse_train >= 10 and zero_foreground >= 10
    assert reused_test >= 15 and new_test >= 15
    assert all(value > 0 for value in review.values()), review
    for num_train in range(2, 11):
        for num_test in range(1, min(4, num_train - 1) + 1):
            puzzle = generate_puzzle(1000 + 10 * num_train + num_test,
                                     num_train, num_test)
            validate_generated_puzzle(puzzle)
            assert (len(puzzle['train']), len(puzzle['test'])) == (
                num_train, num_test)
    for index, kwargs in enumerate(({'num_test': 1}, {'num_test': 4},
                                    {'num_train': 2}, {'num_train': 10})):
        puzzle = generate_puzzle(2000 + index, **kwargs)
        validate_generated_puzzle(puzzle)
        assert _puzzle_equal(puzzle, generate_puzzle(2000 + index, **kwargs))
    for kwargs in ({'num_train': 1}, {'num_train': 11}, {'num_test': 0},
                   {'num_test': 5}, {'num_train': 2, 'num_test': 2},
                   {'num_train': 2, 'num_test': 3},
                   {'num_train': 4, 'num_test': 4}, {'num_train': True}):
        try:
            generate_puzzle(1, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid counts accepted: {kwargs}')
