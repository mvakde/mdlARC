import numpy as np


_ORIGINAL_INPUTS = {
    tuple(s.splitlines()) for s in (
        '''0000000000
0000000000
0000000000
0000000600
0000000600
0060000600
0000000600
0000000000
0066600060
0000006060''',
        '''0000000000
0000000000
0006066600
0006000000
0000000000
0000000000
0000000000
0000006000
0000000000
0000000000''',
        '''0000000000
0000000000
0000000000
0000006000
0606006000
0000006000
0060006000
0000000000
0000000000
0000000000''',
        '''0000000000
6066600000
0000600000
0000600000
0000600000
0000600000
0066600000
0000000000
0000000000
0000000000''',
        '''0000000000
0000000000
0000000000
6666666660
0000000000
0000000006
0000000006
0000000006
0000000006
0000000006''',
    )
}


def _components(grid, fg):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    result = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] != fg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= yy < h and 0 <= xx < w and not seen[yy, xx] and grid[yy, xx] == fg:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
            result.append(tuple(sorted(cells)))
    return result


def _settle(grid, bg, fg):
    pieces = sorted(_components(grid, fg), key=lambda cells: (
        min(x for _, x in cells), min(y for y, _ in cells)))
    output = np.full(grid.shape, bg, dtype=np.int8)
    occupied = set()
    placed = []
    for cells in pieces:
        shift = 0
        while all(x - shift - 1 >= 0 and (y, x - shift - 1) not in occupied
                  for y, x in cells):
            shift += 1
        moved = tuple((y, x - shift) for y, x in cells)
        assert not any(cell in occupied for cell in moved)
        occupied.update(moved)
        for y, x in moved:
            output[y, x] = fg
        placed.append((cells, shift))
    return output, placed


def _shape(rng, max_h, max_w, multi=False):
    kinds = ['single', 'bar', 'stem', 'L', 'C', 'U', 'T']
    if multi:
        kinds = ['stem', 'L', 'C', 'U', 'T']
    if max_h < 2:
        kinds = ['single', 'bar']
    elif max_w < 2:
        kinds = ['single', 'stem']
    kind = str(rng.choice(kinds))
    if kind == 'single':
        return ((0, 0),)
    if kind == 'bar':
        width = int(rng.integers(2, min(max_w, 9) + 1))
        return tuple((0, x) for x in range(width))
    if kind == 'stem':
        height = int(rng.integers(2, min(max_h, 9) + 1))
        return tuple((y, 0) for y in range(height))
    height = int(rng.integers(2, min(max_h, 7) + 1))
    width = int(rng.integers(2, min(max_w, 7) + 1))
    side = int(rng.choice((0, width - 1)))
    end = int(rng.choice((0, height - 1)))
    if kind == 'L':
        cells = {(y, side) for y in range(height)} | {(end, x) for x in range(width)}
    elif kind == 'C':
        cells = {(y, side) for y in range(height)}
        cells |= {(0, x) for x in range(width)}
        cells |= {(height - 1, x) for x in range(width)}
    elif kind == 'U':
        cells = {(y, 0) for y in range(height)} | {(y, width - 1) for y in range(height)}
        cells |= {(end, x) for x in range(width)}
    else:
        stem_x = int(rng.integers(0, width))
        cells = {(end, x) for x in range(width)} | {(y, stem_x) for y in range(height)}
    return tuple(sorted(cells))


def _size(cells):
    return max(y for y, _ in cells) + 1, max(x for _, x in cells) + 1


def _put(grid, fg, cells, y0, x0):
    for y, x in cells:
        yy, xx = y + y0, x + x0
        assert 0 <= yy < grid.shape[0] and 0 <= xx < grid.shape[1]
        assert grid[yy, xx] != fg
        grid[yy, xx] = fg


def _contact_rows(cells):
    left = min(x for _, x in cells)
    return sorted({y for y, x in cells if x == left})


def _witness(rng, grid, fg, extras=False):
    h, w = grid.shape
    mover = _shape(rng, min(7, h - 2), min(6, w - 5), multi=True)
    mh, mw = _size(mover)
    length = int(rng.integers(1, min(7, w - mw - 3) + 1))
    bx = int(rng.integers(0, min(4, w - length - mw - 2)))
    mx = int(rng.integers(bx + length + 2, w - mw + 1))
    my = int(rng.integers(0, h - mh + 1))
    by = my + int(rng.choice(_contact_rows(mover)))
    _put(grid, fg, tuple((0, x) for x in range(length)), by, bx)
    _put(grid, fg, mover, my, mx)
    if extras:
        occupied_rows = set(range(my - 1, my + mh + 1))
        available = [y for y in range(h) if y not in occupied_rows]
        rng.shuffle(available)
        added = []
        for y in available:
            if len(added) >= 4 or rng.random() < 0.25:
                break
            if any(abs(y - other) < 2 for other in added):
                continue
            x = int(rng.integers(0, w))
            _put(grid, fg, ((0, 0),), y, x)
            added.append(y)


def _blockers(rng, grid, fg):
    h, w = grid.shape
    height = int(rng.integers(5, min(8, h) + 1))
    width = int(rng.integers(2, min(6, w - 5) + 1))
    form = str(rng.choice(('L', 'C', 'U')))
    mover = {(y, 0) for y in range(height)}
    if form in ('L', 'C'):
        mover |= {(0, x) for x in range(width)}
    if form in ('C', 'U'):
        mover |= {(height - 1, x) for x in range(width)}
    if form == 'U':
        mover |= {(y, width - 1) for y in range(height)}
    y0 = int(rng.integers(0, h - height + 1))
    x0 = int(rng.integers(4, w - width + 1))
    choices = list(range(0, height, 2))
    rng.shuffle(choices)
    count = int(rng.integers(2, min(3, len(choices)) + 1))
    for y in choices[:count]:
        x = int(rng.integers(0, x0 - 1))
        _put(grid, fg, ((0, 0),), y0 + y, x)
    _put(grid, fg, mover, y0, x0)


def _chain(rng, grid, fg):
    h, w = grid.shape
    middle = _shape(rng, min(5, h - 2), 4, multi=True)
    right = _shape(rng, min(5, h - 2), 4)
    mheight, mwidth = _size(middle)
    rheight, rwidth = _size(right)
    if w - mwidth - rwidth - 4 < 1:
        return False
    anchor_len = int(rng.integers(1, min(4, w - mwidth - rwidth - 4) + 1))
    left_row = int(rng.choice(_contact_rows(middle)))
    right_row = int(rng.choice(_contact_rows(right)))
    lo = max(left_row, right_row)
    hi = min(h - mheight + left_row, h - rheight + right_row)
    common = int(rng.integers(lo, hi + 1))
    gap1 = int(rng.integers(1, 4))
    gap2 = int(rng.integers(1, 4))
    total = anchor_len + gap1 + mwidth + gap2 + rwidth
    if total > w:
        gap1 = gap2 = 1
        total = anchor_len + gap1 + mwidth + gap2 + rwidth
    if total > w:
        return False
    ax = int(rng.integers(0, min(4, w - total + 1)))
    mx = ax + anchor_len + gap1
    rx = mx + mwidth + gap2
    _put(grid, fg, tuple((0, x) for x in range(anchor_len)), common, ax)
    _put(grid, fg, middle, common - left_row, mx)
    _put(grid, fg, right, common - right_row, rx)
    return True


def _independent(rng, grid, fg):
    h, w = grid.shape
    count = int(rng.integers(2, min(4, h // 3) + 1))
    for i in range(count):
        start = (i * h) // count
        stop = ((i + 1) * h) // count - (i < count - 1)
        shape = _shape(rng, max(1, stop - start), min(8, w - 2))
        sh, sw = _size(shape)
        y = int(rng.integers(start, stop - sh + 1))
        x = int(rng.integers(1, w - sw + 1))
        _put(grid, fg, shape, y, x)


def _palette(rng, previous=None):
    while True:
        bg, fg = map(int, rng.choice(10, size=2, replace=False))
        if previous is None or (bg, fg) != previous:
            return bg, fg


def _generate_pair(rng, mode, bg, fg):
    h = int(rng.integers(9, 25))
    w = int(rng.integers(11, 29))
    grid = np.full((h, w), bg, dtype=np.int8)
    if mode == 'witness':
        _witness(rng, grid, fg)
    elif mode == 'mixed':
        _witness(rng, grid, fg, extras=True)
    elif mode == 'chain':
        if not _chain(rng, grid, fg):
            _witness(rng, grid, fg, extras=True)
    elif mode == 'blockers':
        _blockers(rng, grid, fg)
    else:
        _independent(rng, grid, fg)
    output, _ = _settle(grid, bg, fg)
    return {'input': grid, 'output': output}


def _count(value, minimum, maximum, name):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not minimum <= int(value) <= maximum:
        raise ValueError(f'{name} must be an integer from {minimum} to {maximum}')
    return int(value)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    if num_train is not None and num_test is not None and num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))

    shared_palette = bool(rng.integers(0, 2))
    common_palette = _palette(rng)
    puzzle = {'train': [], 'test': []}
    seen = set()
    pair_number = 0
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            if split == 'train' and i == 0:
                mode = 'witness'
            elif split == 'train' and i == 1:
                mode = 'independent'
            else:
                mode = str(rng.choice(('witness', 'chain', 'mixed', 'independent',
                                       'blockers'),
                                      p=(0.22, 0.25, 0.20, 0.15, 0.18) if split == 'train'
                                      else (0.23, 0.24, 0.17, 0.18, 0.18)))
            if shared_palette:
                palette = common_palette
            elif pair_number == 0:
                palette = common_palette
            elif pair_number == 1:
                palette = _palette(rng, common_palette)
            else:
                palette = _palette(rng)
            for _ in range(80):
                pair = _generate_pair(rng, mode, *palette)
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('Could not generate a unique pair')
            pair_number += 1
    validate_generated_puzzle(puzzle)
    return puzzle


def _roles(grid):
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 2 and counts[0] != counts[1]
    bg = int(colors[np.argmax(counts)])
    fg = int(colors[np.argmin(counts)])
    return bg, fg


def _nonprefix(grid, fg):
    for row in grid:
        xs = np.flatnonzero(row == fg)
        if len(xs) and tuple(xs) != tuple(range(len(xs))):
            return True
    return False


def _informative(placed, output, fg):
    multirow_stop = any(len({y for y, _ in cells}) > 1 and
                        min(x for _, x in cells) - shift > 0
                        for cells, shift in placed)
    return multirow_stop and _nonprefix(output, fg)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen_inputs = set()
    seen_pairs = set()
    train_informative = False
    train_independent = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = pair['input'], pair['output']
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape
            bg, fg = _roles(inp)
            assert set(map(int, np.unique(out))) == {bg, fg}
            ikey = (inp.shape, tuple(map(int, inp.flat)))
            pkey = (ikey, out.shape, tuple(map(int, out.flat)))
            assert ikey not in seen_inputs and pkey not in seen_pairs
            seen_inputs.add(ikey)
            seen_pairs.add(pkey)
            input_rows = tuple(''.join(str(int(v)) for v in row) for row in inp)
            assert input_rows not in _ORIGINAL_INPUTS
            expected, placed = _settle(inp, bg, fg)
            assert np.array_equal(out, expected)
            assert not np.array_equal(inp, out)
            assert len(placed) >= 2
            assert np.array_equal(np.count_nonzero(inp == fg, axis=1),
                                  np.count_nonzero(out == fg, axis=1))
            assert any(shift > 0 for _, shift in placed)
            if split == 'train':
                train_informative |= _informative(placed, out, fg)
                train_independent |= any(min(x for _, x in cells) - shift == 0
                                         for cells, shift in placed)
    assert train_informative and train_independent


def _signature(puzzle):
    return tuple(tuple((p['input'].shape, p['input'].tobytes(),
                        p['output'].tobytes()) for p in puzzle[s])
                 for s in ('train', 'test'))


def _simulate_on_grid(inp, bg, fg):
    canvas = np.full(inp.shape, bg, dtype=np.int8)
    pieces = _components(inp, fg)
    pieces.sort(key=lambda cells: min(x for y, x in cells))
    for cells in pieces:
        current = list(cells)
        while min(x for y, x in current) > 0:
            candidate = [(y, x - 1) for y, x in current]
            if any(canvas[y, x] == fg for y, x in candidate):
                break
            current = candidate
        for y, x in current:
            assert canvas[y, x] == bg
            canvas[y, x] = fg
    return canvas


def self_test() -> None:
    signatures, train_counts, test_counts = set(), set(), set()
    dimensions, palettes, layouts = set(), set(), set()
    test_stops = test_nonprefix = anchored = moved_bars = chains = 0
    blocker_scenes = five_components = c_shapes = u_shapes = 0
    top = bottom = overlap = disjoint = 0
    shared_palettes = varied_palettes = 0
    placements = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = _signature(puzzle)
        assert signature == _signature(generate_puzzle(seed))
        signatures.add(signature)
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        puzzle_palettes = set()
        puzzle_dimensions = set()
        puzzle_component_counts = set()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                bg, fg = _roles(inp)
                puzzle_palettes.add((bg, fg))
                puzzle_dimensions.add(inp.shape)
                palettes.add((bg, fg))
                dimensions.add(inp.shape)
                expected, placed = _settle(inp, bg, fg)
                assert np.array_equal(out, expected)
                assert np.array_equal(out, _simulate_on_grid(inp, bg, fg))
                puzzle_component_counts.add(len(placed))
                five_components += len(placed) >= 5
                rows = [{y for y, x in cells} for cells, shift in placed]
                has_overlap = any(rows[i] & rows[j] for i in range(len(rows))
                                  for j in range(i + 1, len(rows)))
                overlap += has_overlap
                disjoint += not has_overlap
                top += any(min(y for y, x in cells) == 0 for cells, shift in placed)
                bottom += any(max(y for y, x in cells) == inp.shape[0] - 1
                              for cells, shift in placed)
                shapes = []
                for cells, shift in placed:
                    y0 = min(y for y, _ in cells)
                    x0 = min(x for _, x in cells)
                    shapes.append(tuple(sorted((y-y0, x-x0) for y, x in cells)))
                    local = set(shapes[-1])
                    sh = max(y for y, x in local) + 1
                    sw = max(x for y, x in local) + 1
                    if sh >= 3 and sw >= 3:
                        left = all((y, 0) in local for y in range(sh))
                        right = all((y, sw - 1) in local for y in range(sh))
                        horizontal_top = all((0, x) in local for x in range(sw))
                        horizontal_bottom = all((sh - 1, x) in local for x in range(sw))
                        c_shapes += (left or right) and horizontal_top and horizontal_bottom
                        u_shapes += left and right and (horizontal_top or horizontal_bottom)
                    placements.add((y0 == 0, y0 >= inp.shape[0] // 2,
                                    x0 == 0, len(cells) == 1))
                    if len({y for y, _ in cells}) == 1 and len(cells) >= 4:
                        anchored += (x0 == 0)
                        moved_bars += (shift > 0)
                blockers = [cells[0][0] for cells, shift in placed
                            if len(cells) == 1 and cells[0][1] - shift == 0]
                blocker_scenes += any(
                    len({y for y in blockers if y in {a for a, b in cells}}) >= 2
                    and len({a for a, b in cells}) > 1
                    and min(x for y, x in cells) - shift > 0
                    for cells, shift in placed)
                layouts.add(tuple(sorted(shapes)))
                chains += len(placed) >= 3 and sum(
                    min(x for _, x in cells) - shift > 0 for cells, shift in placed) >= 2
                if split == 'test':
                    test_stops += any(min(x for _, x in cells) - shift > 0 and shift > 0
                                      for cells, shift in placed)
                    test_nonprefix += _nonprefix(out, fg)
        assert len(puzzle_dimensions) > 1 and len(puzzle_component_counts) > 1
        shared_palettes += len(puzzle_palettes) == 1
        varied_palettes += len(puzzle_palettes) > 1
    assert len(signatures) == 42
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(dimensions) >= 35 and len(palettes) >= 20 and len(layouts) >= 35
    assert test_stops >= 10 and test_nonprefix >= 10
    assert anchored >= 4 and moved_bars >= 4 and chains >= 4
    assert blocker_scenes >= 4 and five_components >= 2
    assert c_shapes >= 8 and u_shapes >= 8
    assert top >= 8 and bottom >= 8 and overlap >= 20 and disjoint >= 8
    assert shared_palettes >= 5 and varied_palettes >= 5
    assert len(placements) >= 8
    for train, test in ((2, 1), (10, 1), (10, 4),
                        (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(901, train, test)
        validate_generated_puzzle(p)
        assert train is None or len(p['train']) == train
        assert test is None or len(p['test']) == test
        assert _signature(p) == _signature(generate_puzzle(901, train, test))
    for train, test in ((1, None), (11, None), (None, 0),
                        (None, 5), (2, 2), (2, 4), (4, 4)):
        try:
            generate_puzzle(7, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError((train, test))
