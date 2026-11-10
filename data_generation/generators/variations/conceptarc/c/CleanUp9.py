"""Generate noisy hollow-square completion puzzles."""
import numpy as np

_SOURCE_INPUTS = {
    ("040023", "411210", "010023", "020010", "021110", "020040"),
    ("055656", "050006", "030005", "030305", "055555", "660503"),
    ("000809", "844740", "400040", "470070", "990799", "448490"),
    ("343436", "300003", "400044", "300003", "204003", "322434"),
    ("000000000000", "011111000000", "010002002000", "210022000000",
     "010001002000", "021111000000", "000000000000", "000002021111",
     "020000020001", "000200010001", "000000010001", "000002021112"),
}


def _key(grid):
    return grid.shape, tuple(int(x) for x in grid.flat)


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[k])
                 for k in ('train', 'test'))


def _perimeter(r, c, side):
    return sorted({(r, x) for x in range(c, c + side)}
                  | {(r + side - 1, x) for x in range(c, c + side)}
                  | {(x, c) for x in range(r, r + side)}
                  | {(x, c + side - 1) for x in range(r, r + side)})


def _render(shape, squares, background, color):
    out = np.full(shape, background, dtype=np.int8)
    for r, c, side in squares:
        for rr, cc in _perimeter(r, c, side):
            out[rr, cc] = color
    return out


def _separate(a, b):
    ar, ac, s = a[:3]
    br, bc, t = b[:3]
    return ar + s < br or br + t < ar or ac + s < bc or bc + t < ac


def _hypotheses(grid, background):
    """Score the entire surviving perimeter without per-side or corner quotas."""
    h, w = grid.shape
    candidates = {}
    for color in np.unique(grid):
        color = int(color)
        if color == background:
            continue
        mask = (grid == color).astype(np.int16)
        row = np.pad(np.cumsum(mask, axis=1), ((0, 0), (1, 0)))
        col = np.pad(np.cumsum(mask, axis=0), ((1, 0), (0, 0)))
        found = []
        for side in range(4, min(h, w) + 1):
            perimeter_size = 4 * side - 4
            for r in range(h - side + 1):
                bottom = r + side - 1
                for c in range(w - side + 1):
                    right = c + side - 1
                    hits = int(row[r, right + 1] - row[r, c]
                               + row[bottom, right + 1] - row[bottom, c]
                               + col[bottom + 1, c] - col[r, c]
                               + col[bottom + 1, right] - col[r, right]
                               - mask[r, c] - mask[r, right]
                               - mask[bottom, c] - mask[bottom, right])
                    score = 8 * hits - 3 * perimeter_size
                    if score >= 18:
                        found.append((score, hits, r, c, side, color))
        candidates[color] = sorted(found, reverse=True)
    return candidates


def _recover(grid, background):
    hypotheses = _hypotheses(grid, background)
    possibilities = []
    for color, ranked in hypotheses.items():
        chosen = []
        total = 0
        for score, hits, r, c, side, _ in ranked:
            square = (r, c, side)
            if all(_separate(square, other) for other in chosen):
                chosen.append(square)
                total += score
        possibilities.append((total, color, sorted(chosen)))
    return sorted(possibilities, reverse=True), hypotheses


def _layout(rng, category):
    if category == 'small':
        side = int(rng.integers(4, 6))
        return (6, 6), [(int(rng.integers(7 - side)),
                         int(rng.integers(7 - side)), side)]
    if category == 'full':
        side = int(rng.integers(6, 14))
        return (side, side), [(0, 0, side)]
    if category == 'single':
        h, w = int(rng.integers(7, 18)), int(rng.integers(7, 18))
        side = int(rng.integers(4, min(h, w) + 1))
        placement = int(rng.integers(4))
        if placement == 0:
            r = int(rng.integers(1, h - side)) if h - side >= 2 else 0
            c = int(rng.integers(1, w - side)) if w - side >= 2 else 0
        elif placement == 1:
            r = int(rng.choice([0, h - side]))
            c = int(rng.integers(w - side + 1))
        elif placement == 2:
            r = int(rng.integers(h - side + 1))
            c = int(rng.choice([0, w - side]))
        else:
            r = int(rng.integers(h - side + 1))
            c = int(rng.integers(w - side + 1))
        return (h, w), [(r, c, side)]
    count = 2 if category == 'double' else 3
    sides = [int(x) for x in rng.integers(4, 8, size=count)]
    gaps = [int(x) for x in rng.integers(1, 5, size=count - 1)]
    before, after = int(rng.integers(4)), int(rng.integers(4))
    primary = before + sum(sides) + sum(gaps) + after
    if primary > 30:
        take = min(after, primary - 30)
        after -= take
        primary -= take
    if primary > 30:
        take = min(before, primary - 30)
        before -= take
        primary -= take
    cross = max(8, max(sides) + int(rng.integers(0, 7)))
    horizontal = bool(rng.integers(2))
    shape = (cross, primary) if horizontal else (primary, cross)
    squares = []
    cursor = before
    for i, side in enumerate(sides):
        offset = int(rng.integers(cross - side + 1))
        squares.append((offset, cursor, side) if horizontal else (cursor, offset, side))
        cursor += side + (gaps[i] if i < count - 1 else 0)
    return shape, squares


def _damage(rng, square, mode):
    r, c, side = square
    edges = [[(r, x) for x in range(c, c + side)],
             [(r + side - 1, x) for x in range(c, c + side)],
             [(x, c) for x in range(r, r + side)],
             [(x, c + side - 1) for x in range(r, r + side)]]
    perimeter = _perimeter(r, c, side)
    if mode == 'light':
        selected = rng.choice(len(perimeter), size=int(rng.integers(1, 4)), replace=False)
        return {perimeter[int(i)] for i in selected}
    if mode == 'sparse_side':
        side_index = int(rng.integers(4))
        edge = edges[side_index]
        n = int(rng.integers(max(2, side - 2), side))
        chosen = {edge[int(i)] for i in rng.choice(side, size=n, replace=False)}
        rest = [p for p in perimeter if p not in chosen and p not in edges[side_index ^ 1]]
        extra = int(rng.integers(0, min(3, len(rest)) + 1))
        chosen.update(rest[int(i)] for i in rng.choice(len(rest), size=extra, replace=False))
        return chosen
    if mode == 'corners':
        corners = [(r, c), (r, c + side - 1),
                   (r + side - 1, c), (r + side - 1, c + side - 1)]
        n = int(rng.integers(2, 5))
        chosen = {corners[int(i)] for i in rng.choice(4, size=n, replace=False)}
        rest = [p for p in perimeter if p not in chosen]
        extra = int(rng.integers(0, max(2, side // 2) + 1))
        chosen.update(rest[int(i)] for i in rng.choice(len(rest), size=min(extra, len(rest)), replace=False))
        return chosen
    if mode == 'clean_side':
        protected = set(edges[int(rng.integers(4))])
        rest = [p for p in perimeter if p not in protected]
        count = int(rng.integers(2, max(3, len(rest) // 2 + 1)))
        return {rest[int(i)] for i in rng.choice(len(rest), size=min(count, len(rest)), replace=False)}
    rate = float(rng.uniform(.18, .53))
    count = max(1, int(round(len(perimeter) * rate)))
    return {perimeter[int(i)] for i in rng.choice(len(perimeter), size=count, replace=False)}


def _make_pair(rng, shape, squares, background, color, noise_pool, regime):
    output = _render(shape, squares, background, color)
    border = output == color
    modes = ('light', 'uniform', 'sparse_side', 'corners', 'clean_side')
    weights = (.13, .27, .23, .20, .17) if regime == 'target_test' else (.22, .24, .19, .18, .17)
    for _ in range(90):
        inp = output.copy()
        mode = str(rng.choice(modes, p=weights))
        hues = [int(x) for x in rng.choice(noise_pool, size=int(rng.integers(1, 5)), replace=False)]
        damaged = set()
        for square in squares:
            damaged.update(_damage(rng, square, mode if rng.random() < .7 else str(rng.choice(modes))))
        deletion_rate = float(rng.choice([0.0, 0.0, 0.0, .25, .5, .75, 1.0]))
        for rr, cc in damaged:
            inp[rr, cc] = background if rng.random() < deletion_rate else int(rng.choice(hues))
        vacant = [(int(r), int(c)) for r, c in np.argwhere(~border)]
        rng.shuffle(vacant)
        same_count = int(rng.choice([0, 0, 0, 1, 1, 2, 3]))
        dense = len(vacant) > 55 and rng.random() < .24
        other_count = (int(rng.integers(12, min(31, len(vacant) // 3 + 1)))
                       if dense else int(rng.integers(0, 8)))
        for rr, cc in vacant[:same_count]:
            inp[rr, cc] = color
        for rr, cc in vacant[same_count:same_count + other_count]:
            inp[rr, cc] = hues[0] if dense and rng.random() < .8 else int(rng.choice(hues))
        values, counts = np.unique(inp, return_counts=True)
        if int(values[np.argmax(counts)]) != background or np.count_nonzero(counts == counts.max()) != 1:
            continue
        rows = tuple(''.join(map(str, row)) for row in inp)
        if rows in _SOURCE_INPUTS or np.array_equal(inp, output):
            continue
        possibilities, hypotheses = _recover(inp, background)
        expected = sorted(squares)
        if not possibilities or possibilities[0][1:] != (color, expected):
            continue
        if len(possibilities) > 1 and possibilities[0][0] - possibilities[1][0] < 5:
            continue
        ranked = hypotheses[color]
        scores = {x[2:5]: x[0] for x in ranked}
        if any(scores[square] - max((x[0] for x in ranked
                                    if x[2:5] != square and not _separate(square, x[2:5])),
                                   default=-100) < 4 for square in expected):
            continue
        return inp, output
    raise RuntimeError('Could not construct an unambiguous noisy square')


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    for value, low, high, name in ((num_train, 2, 10, 'num_train'),
                                   (num_test, 1, 4, 'num_test')):
        if value is not None and (not isinstance(value, (int, np.integer))
                                  or isinstance(value, (bool, np.bool_))
                                  or not low <= value <= high):
            raise ValueError(f'{name} must be an integer from {low} through {high}')
    if num_train is None and num_test is None:
        num_train = int(rng.integers(2, 9))
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    palette = [int(x) for x in rng.permutation(10)]
    fixed_background = rng.random() < .43
    backgrounds = ([palette[0]] * (num_train + num_test) if fixed_background else
                   [palette[i % 10] for i in range(num_train + num_test)])
    train_categories = ['small'] + [str(rng.choice(
        ['small', 'single', 'full', 'double', 'triple'],
        p=[.24, .26, .12, .30, .08])) for _ in range(num_train - 1)]
    rng.shuffle(train_categories)
    test_categories = [str(rng.choice(
        ['small', 'single', 'full', 'double', 'triple'],
        p=[.17, .25, .17, .31, .10])) for _ in range(num_test)]
    puzzle = {'train': [], 'test': []}
    for i, (split, category) in enumerate(
            [('train', x) for x in train_categories] +
            [('test', x) for x in test_categories]):
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
        background = backgrounds[i]
        color = palette[(i + 3) % 10]
        if color == background:
            color = palette[(i + 4) % 10]
        noise_pool = [x for x in palette if x not in (background, color)]
        shape, squares = _layout(pair_rng, category)
        inp, output = _make_pair(pair_rng, shape, squares, background,
                                 color, noise_pool,
                                 'source_train' if split == 'train' else 'target_test')
        puzzle[split].append({'input': inp, 'output': output})
    return puzzle


def _output_squares(output, background):
    colors = set(int(x) for x in np.unique(output)) - {background}
    assert len(colors) == 1
    color = colors.pop()
    h, w = output.shape
    unseen = set(zip(*np.nonzero(output == color)))
    squares = []
    while unseen:
        stack = [unseen.pop()]
        cells = set(stack)
        while stack:
            r, c = stack.pop()
            for point in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if point in unseen:
                    unseen.remove(point)
                    cells.add(point)
                    stack.append(point)
        top, left = min(r for r, _ in cells), min(c for _, c in cells)
        bottom, right = max(r for r, _ in cells), max(c for _, c in cells)
        side = bottom - top + 1
        assert side >= 4 and side == right - left + 1
        assert cells == set(_perimeter(top, left, side))
        squares.append((top, left, side))
    squares.sort()
    assert all(_separate(a, b) for i, a in enumerate(squares) for b in squares[i + 1:])
    assert np.array_equal(output, _render((h, w), squares, background, color))
    return squares, color


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[k], list) for k in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    inputs, pairs = set(), set()
    small_train = 0
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = pair['input'], pair['output']
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            input_key, pair_key = _key(inp), (_key(inp), _key(out))
            assert input_key not in inputs and pair_key not in pairs
            inputs.add(input_key)
            pairs.add(pair_key)
            assert tuple(''.join(map(str, row)) for row in inp) not in _SOURCE_INPUTS
            assert not np.array_equal(inp, out)
            values, counts = np.unique(inp, return_counts=True)
            background = int(values[np.argmax(counts)])
            assert np.count_nonzero(counts == counts.max()) == 1
            squares, color = _output_squares(out, background)
            assert color != background and 1 <= len(squares) <= 3
            assert np.any((out == color) & (inp == color))
            assert np.any((out == color) & (inp != color))
            possibilities, _ = _recover(inp, background)
            assert possibilities and possibilities[0][1:] == (color, squares)
            if split == 'train' and inp.shape == (6, 6) and len(squares) == 1 and squares[0][2] in (4, 5):
                small_train += 1
    assert small_train >= 1


def self_test() -> None:
    signatures, train_counts, test_counts = set(), set(), set()
    layouts, shapes, backgrounds = set(), set(), set()
    deletions, noise_hues, same_strays = set(), set(), set()
    multi_train = zero_deletion = sparse_edge = noise_dominant = 0
    mixed_background = 0
    mixed_object_counts = inset = edge_touch = full = 0
    horizontal = vertical = mixed_sizes = clean_side = damaged_corners = 0
    sparse_opposite_clean = replacement_only_no_stray = large_clutter = 0
    small_test = triple_test = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_key(puzzle))
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        pair_bgs = set()
        pair_object_counts = set()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                values, counts = np.unique(inp, return_counts=True)
                bg = int(values[np.argmax(counts)])
                squares, color = _output_squares(out, bg)
                pair_bgs.add(bg)
                pair_object_counts.add(len(squares))
                backgrounds.add(bg)
                layouts.add((inp.shape, tuple(squares)))
                shapes.add(inp.shape)
                border = out == color
                ndelete = int(np.count_nonzero(border & (inp == bg)))
                deletions.add(ndelete)
                zero_deletion += ndelete == 0
                noise_hues.add(len(set(int(x) for x in np.unique(inp)) - {bg, color}))
                same_strays.add(int(np.count_nonzero((out == bg) & (inp == color))))
                other_strays = int(np.count_nonzero((out == bg) & (inp != bg) & (inp != color)))
                replacement_only_no_stray += ndelete == 0 and int(np.count_nonzero((out == bg) & (inp == color))) == 0
                large_clutter += other_strays >= 12
                noise_dominant += any(np.count_nonzero(inp == x) > np.count_nonzero(inp == color)
                                      for x in np.unique(inp) if x not in (bg, color))
                if split == 'train':
                    multi_train += len(squares) > 1
                else:
                    small_test += inp.shape == (6, 6) and len(squares) == 1
                    triple_test += len(squares) == 3
                inset += any(r > 0 and c > 0 and r + s < inp.shape[0] and c + s < inp.shape[1]
                             for r, c, s in squares)
                edge_touch += any(r == 0 or c == 0 or r + s == inp.shape[0] or c + s == inp.shape[1]
                                  for r, c, s in squares)
                full += len(squares) == 1 and squares[0] == (0, 0, inp.shape[0]) and inp.shape[0] == inp.shape[1]
                mixed_sizes += len({s for _, _, s in squares}) > 1
                if len(squares) > 1:
                    row_span = max(r + s for r, _, s in squares) - min(r for r, _, _ in squares)
                    col_span = max(c + s for _, c, s in squares) - min(c for _, c, _ in squares)
                    horizontal += col_span > row_span
                    vertical += row_span >= col_span
                for r, c, side in squares:
                    edges = [[(r, x) for x in range(c, c + side)],
                             [(r + side - 1, x) for x in range(c, c + side)],
                             [(x, c) for x in range(r, r + side)],
                             [(x, c + side - 1) for x in range(r, r + side)]]
                    surviving = [sum(inp[rr, cc] == color for rr, cc in edge) for edge in edges]
                    sparse_edge += any(n <= 1 for n in surviving)
                    clean_side += any(n == side for n in surviving)
                    sparse_opposite_clean += any(surviving[j] <= 1 and surviving[j ^ 1] == side
                                                 for j in range(4))
                    corners = ((r, c), (r, c + side - 1),
                               (r + side - 1, c), (r + side - 1, c + side - 1))
                    damaged_corners += sum(inp[rr, cc] != color for rr, cc in corners) >= 2
        mixed_background += len(pair_bgs) > 1
        mixed_object_counts += len(pair_object_counts) > 1
    assert len(signatures) >= 40
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(layouts) >= 90 and len(shapes) >= 30
    assert len(backgrounds) >= 6 and mixed_background >= 5
    assert len(deletions) >= 5 and 0 in deletions
    assert len(noise_hues) >= 3 and 1 in noise_hues
    assert 0 in same_strays and max(same_strays) >= 2
    assert multi_train >= 4 and zero_deletion >= 5 and sparse_edge >= 5
    assert noise_dominant >= 3
    assert mixed_object_counts >= 20
    assert inset >= 20 and edge_touch >= 20 and full >= 10
    assert horizontal >= 15 and vertical >= 15 and mixed_sizes >= 15
    assert clean_side >= 20 and damaged_corners >= 20 and sparse_opposite_clean >= 5
    assert replacement_only_no_stray >= 10 and large_clutter >= 5
    assert small_test >= 5 and triple_test >= 2
    for a, b in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                 (2, None), (10, None)):
        puzzle = generate_puzzle(902, a, b)
        validate_generated_puzzle(puzzle)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(902, a, b))
        assert a is None or len(puzzle['train']) == a
        assert b is None or len(puzzle['test']) == b
    for a, b in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (2, 3), (3, 4), (4, 4)):
        try:
            generate_puzzle(902, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal count request accepted')
