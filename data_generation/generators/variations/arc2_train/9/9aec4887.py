import numpy as np


_SOURCE_INPUTS = frozenset({
    (14, 16, ''.join((
        '0000000000000000','0000000000000000','0000000008000000',
        '0000000088800000','0000000008800000','0000000000000000',
        '0033300000000000','0600040000000000','0600040000000000',
        '0600040000000000','0022200000000000','0000000000000000',
        '0000000000000000','0000000000000000'))),
    (15, 16, ''.join((
        '0000000000000000','0800800000000000','0080800000000000',
        '0888800000000000','0080800000000000','0000000000000000',
        '0000000000000000','0000000444400000','0000002000010000',
        '0000002000010000','0000002000010000','0000002000010000',
        '0000000333300000','0000000000000000','0000000000000000'))),
    (15, 15, ''.join((
        '000000000000000','000777700000000','004000060000000',
        '004000060000000','004000060000000','004000060000000',
        '000111100000000','000000000000000','000000000000000',
        '000000880800000','000000088000000','000000080800000',
        '000000880800000','000000000000000','000000000000000'))),
    (16, 16, ''.join((
        '0000001111100000','0000020000040000','0000020000040000',
        '0000020000040000','0000020000040000','0000020000040000',
        '0000003333300000','0000000000000000','0000000000000000',
        '0088088000000000','0080880000000000','0008888000000000',
        '0000800000000000','0088088000000000','0000000000000000',
        '0000000000000000'))),
})


def _check(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def _key(grid):
    return grid.shape, grid.tobytes()


def _source_key(grid):
    return grid.shape[0], grid.shape[1], ''.join(map(str, grid.ravel()))


def _profile(mask):
    h, w = mask.shape
    sides = set()
    depths = set()
    ties = 0
    for y, x in np.argwhere(mask):
        d = (int(y), h - 1 - int(y), int(x), w - 1 - int(x))
        nearest = [i for i, value in enumerate(d) if value == min(d)]
        if len(nearest) == 1:
            sides.add(nearest[0])
            depths.add((nearest[0], min(d)))
        else:
            ties += 1
    return sides, depths, ties


def _tight(mask):
    return (mask[0].any() and mask[-1].any()
            and mask[:, 0].any() and mask[:, -1].any())


def _mask(rng, h, w, demonstration=False):
    mode = int(rng.integers(0, 7))
    yy, xx = np.indices((h, w))
    if mode == 0:  # thin, scattered points
        mask = rng.random((h, w)) < rng.uniform(.12, .26)
    elif mode == 1:  # medium irregular
        mask = rng.random((h, w)) < rng.uniform(.28, .52)
    elif mode == 2:  # dense irregular
        mask = rng.random((h, w)) < rng.uniform(.60, .82)
    elif mode == 3:  # hollow outlines with gaps
        edge = (yy == 0) | (yy == h - 1) | (xx == 0) | (xx == w - 1)
        mask = edge & (rng.random((h, w)) < .65)
        mask |= (~edge) & (rng.random((h, w)) < .12)
    elif mode == 4:  # short horizontal and vertical strokes
        mask = np.zeros((h, w), dtype=bool)
        for _ in range(int(rng.integers(2, 7))):
            y, x = int(rng.integers(h)), int(rng.integers(w))
            length = int(rng.integers(2, max(h, w) + 1))
            if rng.integers(2):
                mask[y, x:min(w, x + length)] = True
            else:
                mask[y:min(h, y + length), x] = True
    elif mode == 5:  # mirrored motifs, including symmetric squares
        mask = rng.random((h, w)) < rng.uniform(.15, .42)
        mask |= np.fliplr(mask)
        if rng.integers(2):
            mask |= np.flipud(mask)
    else:  # mostly nearest-side ties
        distance = np.stack((yy, h - 1 - yy, xx, w - 1 - xx))
        minimum = np.min(distance, axis=0)
        tie = np.sum(distance == minimum, axis=0) >= 2
        mask = tie & (rng.random((h, w)) < .85)
        mask |= (~tie) & (rng.random((h, w)) < .06)

    # The extremes define the complete stencil box; none has a fixed location.
    if not mask[0].any():
        mask[0, int(rng.integers(w))] = True
    if not mask[-1].any():
        mask[-1, int(rng.integers(w))] = True
    if not mask[:, 0].any():
        mask[int(rng.integers(h)), 0] = True
    if not mask[:, -1].any():
        mask[int(rng.integers(h)), -1] = True

    if demonstration:
        # A single teaching example supplies all four roles, a tie, and depth.
        for side in range(4):
            cells = []
            for y in range(h):
                for x in range(w):
                    d = (y, h - 1 - y, x, w - 1 - x)
                    if d[side] == min(d) and d.count(min(d)) == 1:
                        cells.append((y, x, min(d)))
            deepest = max(cell[2] for cell in cells)
            for depth_target in range(1, min(2, deepest) + 1):
                eligible = [(y, x) for y, x, depth in cells
                            if depth == depth_target]
                y, x = eligible[int(rng.integers(len(eligible)))]
                mask[y, x] = True
        ties = []
        for y in range(h):
            for x in range(w):
                d = (y, h - 1 - y, x, w - 1 - x)
                if d.count(min(d)) >= 2 and min(d) >= 1:
                    ties.append((y, x))
        y, x = ties[int(rng.integers(len(ties)))]
        mask[y, x] = True
    elif rng.random() < .13:
        distance = np.stack((yy, h - 1 - yy, xx, w - 1 - xx))
        tie = np.sum(distance == np.min(distance, axis=0), axis=0) >= 2
        mask[tie] = False
        mask[0, w // 2] = True
        mask[-1, w // 2] = True
        mask[h // 2, 0] = True
        mask[h // 2, -1] = True
    return mask


def _pick_colors(rng, settings, previous):
    stable = settings['stencil']
    background = settings['background'] if settings['fixed_background'] else int(rng.integers(10))
    if background == stable:
        background = (background + 1 + int(rng.integers(9))) % 10
    if previous and rng.random() < .34 and background not in previous:
        roles = previous[:4]
        stencil = previous[4]
    else:
        if settings['fixed_stencil']:
            stencil = stable
        else:
            stencil = int(rng.choice([c for c in range(10) if c != background]))
        available = [c for c in range(10) if c not in (background, stencil)]
        roles = tuple(int(c) for c in rng.choice(available, 4, replace=False))
    return background, tuple(roles), stencil


def _layout(rng, h, w, arrangement):
    if arrangement in ('frame_above', 'stencil_above'):
        gap = int(rng.integers(1, min(6, 30 - 2 * h - 1)))
        required = 2 * h + 2 + gap
        top_margin = int(rng.integers(0, min(5, 31 - required)))
        bottom_margin = int(rng.integers(0, min(5, 31 - required - top_margin)))
        canvas_h = required + top_margin + bottom_margin
        canvas_w = min(30, w + 2 + int(rng.integers(0, 10)))
        frame_x = int(rng.integers(0, canvas_w - w - 1))
        stencil_x = int(rng.integers(0, canvas_w - w + 1))
        if arrangement == 'frame_above':
            frame_y = top_margin
            stencil_y = frame_y + h + 2 + gap
        else:
            stencil_y = top_margin
            frame_y = stencil_y + h + gap
    else:
        gap = int(rng.integers(1, min(6, 30 - 2 * w - 1)))
        required = 2 * w + 2 + gap
        left_margin = int(rng.integers(0, min(5, 31 - required)))
        right_margin = int(rng.integers(0, min(5, 31 - required - left_margin)))
        canvas_w = required + left_margin + right_margin
        canvas_h = min(30, h + 2 + int(rng.integers(0, 10)))
        frame_y = int(rng.integers(0, canvas_h - h - 1))
        stencil_y = int(rng.integers(0, canvas_h - h + 1))
        if arrangement == 'frame_left':
            frame_x = left_margin
            stencil_x = frame_x + w + 2 + gap
        else:
            stencil_x = left_margin
            frame_x = stencil_x + w + gap
    return canvas_h, canvas_w, frame_y, frame_x, stencil_y, stencil_x


def _paint_output(background, colors, stencil, mask):
    h, w = mask.shape
    out = np.full((h + 2, w + 2), background, dtype=np.int8)
    out[0, 1:-1] = colors[0]
    out[-1, 1:-1] = colors[1]
    out[1:-1, 0] = colors[2]
    out[1:-1, -1] = colors[3]
    for y, x in np.argwhere(mask):
        d = (int(y), h - 1 - int(y), int(x), w - 1 - int(x))
        nearest = [side for side in range(4) if d[side] == min(d)]
        out[y + 1, x + 1] = colors[nearest[0]] if len(nearest) == 1 else stencil
    return out


def _make_pair(rng, h, w, arrangement, settings, previous, demonstration):
    background, colors, stencil = _pick_colors(rng, settings, previous)
    mask = _mask(rng, h, w, demonstration)
    canvas_h, canvas_w, fy, fx, sy, sx = _layout(rng, h, w, arrangement)
    grid = np.full((canvas_h, canvas_w), background, dtype=np.int8)
    grid[fy, fx + 1:fx + w + 1] = colors[0]
    grid[fy + h + 1, fx + 1:fx + w + 1] = colors[1]
    grid[fy + 1:fy + h + 1, fx] = colors[2]
    grid[fy + 1:fy + h + 1, fx + w + 1] = colors[3]
    region = grid[sy:sy + h, sx:sx + w]
    region[mask] = stencil
    return {'input': grid, 'output': _paint_output(background, colors, stencil, mask)}


def _extract(grid):
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    foreground = [int(v) for v in values if v != background]
    _check(len(foreground) == 5, 'Expected five distinct foreground roles')
    horizontal, vertical = [], []
    for color in foreground:
        cells = np.argwhere(grid == color)
        ys, xs = cells[:, 0], cells[:, 1]
        if np.all(ys == ys[0]) and len(xs) >= 3 and xs.max() - xs.min() + 1 == len(xs):
            horizontal.append((color, int(ys[0]), int(xs.min()), int(xs.max())))
        if np.all(xs == xs[0]) and len(ys) >= 3 and ys.max() - ys.min() + 1 == len(ys):
            vertical.append((color, int(xs[0]), int(ys.min()), int(ys.max())))
    interpretations = []
    for top in horizontal:
        for bottom in horizontal:
            if top[0] == bottom[0] or top[2:] != bottom[2:] or top[1] >= bottom[1]:
                continue
            h, w = bottom[1] - top[1] - 1, top[3] - top[2] + 1
            if h < 3 or w < 3:
                continue
            for left in vertical:
                if left[1] != top[2] - 1 or left[2:] != (top[1] + 1, bottom[1] - 1):
                    continue
                for right in vertical:
                    if right[1] != top[3] + 1 or right[2:] != left[2:]:
                        continue
                    roles = (top[0], bottom[0], left[0], right[0])
                    if len(set(roles)) != 4:
                        continue
                    other = set(foreground) - set(roles)
                    if len(other) != 1:
                        continue
                    stencil = other.pop()
                    cells = np.argwhere(grid == stencil)
                    sy, sx = cells.min(axis=0)
                    ey, ex = cells.max(axis=0)
                    if (int(ey - sy + 1), int(ex - sx + 1)) != (h, w):
                        continue
                    fy, fx = top[1], top[2] - 1
                    separated = (ey < fy - 1 or sy > fy + h + 2
                                 or ex < fx - 1 or sx > fx + w + 2)
                    if not separated:
                        continue
                    mask = grid[sy:ey + 1, sx:ex + 1] == stencil
                    if not _tight(mask):
                        continue
                    reconstructed = np.full(grid.shape, background, dtype=np.int8)
                    reconstructed[fy, fx + 1:fx + w + 1] = roles[0]
                    reconstructed[fy + h + 1, fx + 1:fx + w + 1] = roles[1]
                    reconstructed[fy + 1:fy + h + 1, fx] = roles[2]
                    reconstructed[fy + 1:fy + h + 1, fx + w + 1] = roles[3]
                    reconstructed[sy:ey + 1, sx:ex + 1][mask] = stencil
                    if np.array_equal(grid, reconstructed):
                        interpretations.append((background, roles, stencil, mask, (fy, fx), (int(sy), int(sx))))
    _check(len(interpretations) == 1, 'Frame and stencil must have one interpretation')
    return interpretations[0]


def _counts(rng, num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, 'num_train'), (num_test, 1, 4, 'num_test')):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= value <= high):
            raise ValueError(name + ' is outside its legal integer range')
    if num_test is None:
        num_test = int(rng.integers(1, min(4, (num_train or 10) - 1) + 1))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    settings = {
        'background': int(rng.integers(10)),
        'fixed_background': bool(rng.random() < .55),
        'stencil': int(rng.integers(10)),
        'fixed_stencil': bool(rng.random() < .55),
    }
    if settings['stencil'] == settings['background']:
        settings['stencil'] = (settings['stencil'] + 1) % 10

    # Two teaching examples cover the source square and the compatible rectangle.
    square = int(rng.integers(5, 9))
    rect_h = int(rng.integers(5, 10))
    rect_w = int(rng.integers(5, 10))
    if rect_w == rect_h:
        rect_w = 10 if rect_w == 9 else rect_w + 1
    train_shapes = [(square, square), (rect_h, rect_w)]
    for _ in range(train_count - 2):
        if rng.random() < .37:
            n = int(rng.integers(3, 10))
            train_shapes.append((n, n))
        else:
            train_shapes.append((int(rng.integers(3, 11)), int(rng.integers(3, 11))))
    rng.shuffle(train_shapes)
    teaching_candidates = [i for i, shape in enumerate(train_shapes) if min(shape) >= 5]
    teaching_index = teaching_candidates[int(rng.integers(len(teaching_candidates)))]
    arrangements = ('frame_above', 'stencil_above', 'frame_left', 'stencil_left')
    layout_mode = int(rng.choice(4, p=(.13, .18, .15, .54)))
    if layout_mode == 0:
        allowed_arrangements = (arrangements[int(rng.integers(4))],)
    elif layout_mode == 1:
        allowed_arrangements = arrangements[:2]
    elif layout_mode == 2:
        allowed_arrangements = arrangements[2:]
    else:
        allowed_arrangements = arrangements
    puzzle = {'train': [], 'test': []}
    seen = set()
    previous = None
    for split, count in (('train', train_count), ('test', test_count)):
        for index in range(count):
            if split == 'train':
                h, w = train_shapes[index]
                demonstration = (index == teaching_index)
            else:
                if rng.random() < .32:
                    eligible_shapes = [shape for shape in train_shapes if min(shape) >= 5]
                    h, w = eligible_shapes[int(rng.integers(len(eligible_shapes)))]
                elif rng.random() < .38:
                    n = int(rng.integers(5, 13))
                    h, w = n, n
                else:
                    h, w = int(rng.integers(5, 13)), int(rng.integers(5, 13))
                demonstration = False
            for _ in range(80):
                arrangement = allowed_arrangements[int(rng.integers(len(allowed_arrangements)))]
                pair = _make_pair(rng, h, w, arrangement, settings, previous, demonstration)
                key = _key(pair['input'])
                if key in seen or _source_key(pair['input']) in _SOURCE_INPUTS:
                    continue
                try:
                    _extract(pair['input'])
                except AssertionError:
                    continue
                puzzle[split].append(pair)
                seen.add(key)
                _, prior_roles, prior_stencil, _, _, _ = _extract(pair['input'])
                previous = (*prior_roles, prior_stencil)
                break
            else:
                raise RuntimeError('Could not construct a unique pair')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Invalid puzzle keys')
    _check(all(isinstance(puzzle[k], list) for k in ('train', 'test')), 'Splits must be lists')
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    _check(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, 'Invalid pair counts')
    seen = set()
    train_sides, train_depths, train_ties = set(), set(), 0
    squares, rectangles = 0, 0
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Invalid pair')
            for grid in pair.values():
                _check(isinstance(grid, np.ndarray) and grid.ndim == 2, 'Invalid grid type')
                _check(np.issubdtype(grid.dtype, np.integer), 'Noninteger grid')
                _check(all(1 <= n <= 30 for n in grid.shape), 'Grid dimension outside ARC bounds')
                _check(grid.min() >= 0 and grid.max() <= 9, 'Color outside ARC range')
            inp, out = pair['input'], pair['output']
            _check(_key(inp) not in seen, 'Duplicate input')
            seen.add(_key(inp))
            _check(_source_key(inp) not in _SOURCE_INPUTS, 'Copied source input')
            background, roles, stencil, mask, _, _ = _extract(inp)
            h, w = mask.shape
            _check(out.shape == (h + 2, w + 2), 'Wrong frame crop dimensions')
            _check(np.array_equal(out, _paint_output(background, roles, stencil, mask)), 'Wrong nearest-side output')
            if split == 'train':
                sides, depths, ties = _profile(mask)
                train_sides |= sides
                train_depths |= depths
                train_ties += ties
                squares += h == w
                rectangles += h != w
            else:
                _check(h >= 5 and w >= 5, 'Test shape too small for target regime')
    _check(squares and rectangles, 'Training must teach square and rectangular alignment')
    _check(train_sides == {0, 1, 2, 3} and train_ties, 'Training lacks a side or tie witness')
    _check({0, 1, 2, 3}.issubset({s for s, d in train_depths if d >= 1}),
           'Training lacks depth-one examples on all sides')


def _equal_puzzles(a, b):
    return all(np.array_equal(pa[k], pb[k])
               for split in ('train', 'test')
               for pa, pb in zip(a[split], b[split])
               for k in ('input', 'output')) and all(len(a[s]) == len(b[s]) for s in ('train', 'test'))


def self_test() -> None:
    counts_train, counts_test = set(), set()
    shapes, masks, layouts, backgrounds, role_tuples = set(), set(), set(), set(), set()
    square_train = rectangle_train = deep_train = sparse = dense = symmetric = zero_role = 0
    same_size = larger_test = horizontal = vertical = top_contact = 0
    missing_sides = tie_free = 0
    upper_left_empty = mostly_ties = test_tie_free = small_train = 0
    combined = fixed_layout = varied_layout = varied_background = repeated_roles = unseen_test_colors = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _check(_equal_puzzles(puzzle, generate_puzzle(seed)), 'Nondeterministic puzzle')
        counts_train.add(len(puzzle['train']))
        counts_test.add(len(puzzle['test']))
        train_shapes = set()
        pair_features = {'train': [], 'test': []}
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                bg, roles, stencil, mask, (fy, fx), (sy, sx) = _extract(pair['input'])
                h, w = mask.shape
                shapes.add((h, w))
                masks.add((h, w, mask.tobytes()))
                backgrounds.add(bg)
                role_tuples.add((*roles, stencil))
                layouts.add((fy - sy, fx - sx, pair['input'].shape))
                zero_role += 0 in (*roles, stencil)
                sparse += mask.mean() < .3
                dense += mask.mean() > .6
                symmetric += np.array_equal(mask, np.fliplr(mask))
                sides, depths, ties = _profile(mask)
                missing_sides += len(sides) < 4
                tie_free += ties == 0
                test_tie_free += split == 'test' and ties == 0
                upper_left_empty += not mask[0, 0]
                mostly_ties += ties > int(np.count_nonzero(mask)) / 2
                horizontal += abs(fx - sx) > w + 1
                vertical += abs(fy - sy) > h + 1
                top_contact += fy == 0 or sy == 0
                combined += (h != w and abs(fx - sx) > w + 1 and bg != 0
                             and any(d >= 1 for _, d in depths))
                arrangement = ('frame_above' if fy + h + 1 < sy else
                               'stencil_above' if sy + h - 1 < fy else
                               'frame_left' if fx + w + 1 < sx else 'stencil_left')
                pair_features[split].append((bg, (*roles, stencil), (h, w), arrangement))
                if split == 'train':
                    train_shapes.add((h, w))
                    square_train += h == w
                    rectangle_train += h != w
                    deep_train += any(d >= 2 for _, d in depths)
                    small_train += max(h, w) <= 4
                else:
                    same_size += (h, w) in train_shapes
                    larger_test += h > max(t[0] for t in train_shapes) or w > max(t[1] for t in train_shapes)
        all_features = pair_features['train'] + pair_features['test']
        fixed_layout += len({x[3] for x in all_features}) == 1
        varied_layout += len({x[3] for x in all_features}) > 1
        varied_background += len({x[0] for x in all_features}) > 1
        repeated_roles += len({x[1] for x in all_features}) < len(all_features)
        train_colors = {c for x in pair_features['train'] for c in x[1]}
        unseen_test_colors += any(set(x[1]) - train_colors for x in pair_features['test'])
    _check(len(counts_train) >= 4 and len(counts_test) >= 3, 'Automatic count diversity')
    _check(len(shapes) >= 30 and len(masks) >= 100 and len(layouts) >= 100, 'Structural diversity')
    _check(len(backgrounds) >= 7 and len(role_tuples) >= 60 and zero_role >= 20, 'Color diversity')
    _check(min(square_train, rectangle_train, deep_train, sparse, dense, symmetric) > 0,
           'Missing training shape, depth, or stencil variation')
    _check(min(same_size, larger_test, horizontal, vertical, top_contact, missing_sides, tie_free) > 0,
           'Missing regime, layout, or branch variation')
    _check(min(upper_left_empty, mostly_ties, test_tie_free, small_train, combined,
               fixed_layout, varied_layout, varied_background, repeated_roles,
               unseen_test_colors) > 0, 'Missing a review-requested variation')
    for i, (nt, ns) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1),
                                   (None, 4), (2, None), (10, None), (6, 2))):
        puzzle = generate_puzzle(1000 + i, nt, ns)
        validate_generated_puzzle(puzzle)
        _check(nt is None or len(puzzle['train']) == nt, 'Train count ignored')
        _check(ns is None or len(puzzle['test']) == ns, 'Test count ignored')
    for kwargs in ({'num_train': 1}, {'num_train': 11}, {'num_test': 0},
                   {'num_test': 5}, {'num_train': 2, 'num_test': 2},
                   {'num_train': 3, 'num_test': 4}, {'num_train': True},
                   {'num_train': 2.0}):
        try:
            generate_puzzle(99, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError('Expected ValueError for illegal count')
