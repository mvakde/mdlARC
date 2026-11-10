"""Corner-and-cue painting puzzles. No hidden color map or direction rule is used."""
import hashlib
import numpy as np

CORNERS = (0, 1, 2, 3)  # top-left, top-right, bottom-left, bottom-right
SOURCE_DIGESTS = frozenset((
    '57dbc3f9a8baa7ca8447578adaed9334d21a0cea7de81cbad46ab31182c0f380',
    'bfb1b17130b9876436a545fd45d76c1a8aab150be7abcd903985749f0bef3745',
    'df4285f2d9a5727d0b5e75ef8a2f06ea576b10ef2a2de210ab1a89ad2e81f477',
    'f5729af3e36b08351f773e75446f811a3f1403015200b77b20be64f2ece255a7',
))


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer or None')
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f'{name} out of range')
    return value


def _resolve_counts(seed, num_train, num_test):
    train = _count(num_train, 'num_train', 2, 10)
    test = _count(num_test, 'num_test', 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        low = max(2, test + 1)
        train = low + (seed // 4) % (11 - low)
    elif train is None:
        low = max(2, test + 1)
        train = low + seed % (11 - low)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError('num_train must exceed num_test')
    return int(train), int(test)


def _signature(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _source_copy(inp, out):
    if inp.shape != (15, 15) or out.shape != (15, 15):
        return False
    digest = hashlib.sha256(inp.astype(np.int8).tobytes() + out.astype(np.int8).tobytes()).hexdigest()
    return digest in SOURCE_DIGESTS


def _side(rng, cap=28):
    band = int(rng.choice(3, p=(0.23, 0.49, 0.28)))
    low, high = ((4, 8), (9, 17), (18, cap))[band]
    return int(rng.integers(low if low <= high else 4, min(high, cap) + 1))


def _shape(rng, regime, mode, train_shapes, fixed):
    if regime == 'source_train':
        if fixed is not None:
            return fixed
        if mode == 2 and not train_shapes:
            return int(rng.integers(17, 29)), int(rng.integers(17, 29))
        return _side(rng, 23 if mode in (3, 5) else 28), _side(rng, 23 if mode in (0, 3, 4) else 28)
    max_h = max(x[0] for x in train_shapes)
    max_w = max(x[1] for x in train_shapes)
    if mode == 0:  # only width grows, as in the source task
        return max_h, int(rng.integers(max_w + 1, min(28, max_w + 5) + 1))
    if mode == 1:  # equal train/test sizes
        return train_shapes[int(rng.integers(len(train_shapes)))]
    if mode == 2:  # a larger training panel, smaller test
        return int(rng.integers(4, max_h)), int(rng.integers(4, max_w))
    if mode == 3:  # both dimensions grow
        return (int(rng.integers(max_h + 1, min(28, max_h + 5) + 1)),
                int(rng.integers(max_w + 1, min(28, max_w + 5) + 1)))
    if mode == 5:  # only height grows
        return int(rng.integers(max_h + 1, min(28, max_h + 5) + 1)), max_w
    if rng.random() < 0.62:  # mixed regime also exercises source-like width growth
        return max_h, int(rng.integers(max_w + 1, min(28, max_w + 5) + 1))
    return _side(rng), _side(rng)


def _margins(rng, side, force_fifteen):
    remaining = 30 - side
    if force_fifteen:
        extra = 15 - side
    else:
        draw = rng.random()
        if draw < 0.16:
            extra = remaining
        elif draw < 0.62:
            extra = int(rng.integers(2, min(8, remaining) + 1))
        else:
            extra = int(rng.integers(2, remaining + 1))
    if rng.random() < 0.43:
        first = 1 if rng.random() < 0.5 else extra - 1
    else:
        first = int(rng.integers(1, extra))
    return first, extra - first


def _train_plan(rng, count, mode):
    order = tuple(int(x) for x in rng.permutation(4))
    if mode == 2:  # stable instruction count, yet full orientation coverage
        fixed = int(rng.integers(2, 5))
        plan = ([order[:2], order[2:]] if fixed == 2 else
                [order[:3], order[1:]] if fixed == 3 else [order, order])
        high = fixed + 1
        plan.extend(tuple(int(x) for x in rng.choice(4, fixed, replace=False))
                    for _ in range(count - 2))
    elif mode == 3:  # four-corner training before simpler tests
        plan = [order, order[:int(rng.integers(1, 4))]]
        high = 5
        plan.extend(tuple(int(x) for x in rng.choice(4, int(rng.integers(1, high)), replace=False))
                    for _ in range(count - 2))
    else:
        cut = int(rng.integers(1, 4))
        plan = [order[:cut], order[cut:]]
        high = 4 if mode == 0 else 5
        plan.extend(tuple(int(x) for x in rng.choice(4, int(rng.integers(1, high)), replace=False))
                    for _ in range(count - 2))
    rng.shuffle(plan)
    return plan


def _test_plan(rng, count, mode, train_plan):
    if mode == 0:
        return [CORNERS for _ in range(count)]
    if mode == 2:
        sizes = [len(train_plan[0])] * count
    elif mode == 3:
        sizes = [int(rng.integers(1, 4)) for _ in range(count)]
    else:
        sizes = [int(rng.integers(1, 5)) for _ in range(count)]
    return [tuple(int(x) for x in rng.choice(4, size, replace=False)) for size in sizes]


def _colors(rng, stable_roles, previous_markers):
    if stable_roles is None:
        bg = int(rng.choice(previous_markers)) if previous_markers and rng.random() < 0.46 else int(rng.integers(10))
        fill = int(rng.choice([c for c in range(10) if c != bg]))
    else:
        bg, fill = stable_roles
    palette = tuple(int(x) for x in rng.permutation([c for c in range(10) if c not in (bg, fill)]))
    return bg, fill, palette


def _category(category, h, w, panel_h, panel_w):
    if category == 0:
        return True
    if category == 1:
        return 2 <= h <= 4 and 2 <= w <= 4
    if category == 2:
        return (h == 1 and w > 1) or (w == 1 and h > 1)
    if category == 3:
        return 3 * w >= 2 * panel_w and h <= max(2, panel_h // 3)
    if category == 4:
        return 3 * h >= 2 * panel_h and w <= max(2, panel_w // 3)
    if category == 5:
        return 2 * h >= panel_h and 2 * w >= panel_w
    return h == panel_h or w == panel_w


def _rectangles(rng, panel_h, panel_w, active, force_strip, allow_strip):
    # Each far row and column is considered independently. Only painted-cell
    # overlap, marker collisions, and covering another active anchor are barred.
    for _ in range(24):
        painted = np.zeros((panel_h, panel_w), dtype=bool)
        cues = set()
        rectangles = {}
        order = tuple(int(x) for x in rng.permutation(active))
        for index, corner in enumerate(order):
            category = 2 if force_strip and index == 0 else int(rng.choice(
                7, p=(0.24, 0.10, 0.11, 0.17, 0.17, 0.10, 0.11)))
            anchor_r = 0 if corner in (0, 1) else panel_h - 1
            anchor_c = 0 if corner in (0, 2) else panel_w - 1
            other_anchors = {(0 if c in (0, 1) else panel_h - 1,
                              0 if c in (0, 2) else panel_w - 1)
                             for c in active if c != corner and c not in rectangles}
            options = []
            for h in range(1, panel_h + 1):
                for w in range(1, panel_w + 1):
                    if h == w == 1:
                        continue
                    if not allow_strip and (h == 1 or w == 1):
                        continue
                    end_r = h - 1 if anchor_r == 0 else panel_h - h
                    end_c = w - 1 if anchor_c == 0 else panel_w - w
                    r0, r1 = min(anchor_r, end_r), max(anchor_r, end_r)
                    c0, c1 = min(anchor_c, end_c), max(anchor_c, end_c)
                    if painted[r0:r1 + 1, c0:c1 + 1].any():
                        continue
                    if any(r0 <= r <= r1 and c0 <= c <= c1 for r, c in other_anchors):
                        continue
                    h_cue = (-1 if anchor_r == 0 else panel_h, end_c)
                    v_cue = (end_r, -1 if anchor_c == 0 else panel_w)
                    if h_cue in cues or v_cue in cues:
                        continue
                    options.append((h, w, r0, r1, c0, c1, h_cue, v_cue))
            if not options:
                break
            preferred = [o for o in options if _category(category, o[0], o[1], panel_h, panel_w)]
            pool = preferred if preferred else options
            h, w, r0, r1, c0, c1, h_cue, v_cue = pool[int(rng.integers(len(pool)))]
            painted[r0:r1 + 1, c0:c1 + 1] = True
            cues.update((h_cue, v_cue))
            rectangles[corner] = (r0, r1, c0, c1, h_cue, v_cue)
        if len(rectangles) == len(active):
            return rectangles
    raise RuntimeError('no nonoverlapping rectangle layout')


def _generate_pair(rng, regime, shape, active, colors, force_strip, allow_strip):
    panel_h, panel_w = shape
    fifteen = panel_h <= 13 and panel_w <= 13 and rng.random() < 0.14
    top, bottom_margin = _margins(rng, panel_h, fifteen)
    left, right_margin = _margins(rng, panel_w, fifteen)
    bg, fill, palette = colors
    inp = np.full((top + panel_h + bottom_margin, left + panel_w + right_margin), bg, dtype=np.int8)
    out = inp.copy()
    inp[top:top + panel_h, left:left + panel_w] = fill
    out[top:top + panel_h, left:left + panel_w] = fill
    rectangles = _rectangles(rng, panel_h, panel_w, active, force_strip, allow_strip)
    markers = tuple(palette[:len(active)])
    for corner, color in zip(active, markers):
        r0, r1, c0, c1, h_cue, v_cue = rectangles[corner]
        ar = 0 if corner in (0, 1) else panel_h - 1
        ac = 0 if corner in (0, 2) else panel_w - 1
        inp[top + ar, left + ac] = color
        inp[top + h_cue[0], left + h_cue[1]] = color
        inp[top + v_cue[0], left + v_cue[1]] = color
        out[top + r0:top + r1 + 1, left + c0:left + c1 + 1] = color
    return inp, out, markers


def _analyze_pair(inp, out):
    assert inp.ndim == 2 and inp.shape == out.shape
    bg = int(inp[0, 0])
    values, counts = np.unique(inp[inp != bg], return_counts=True)
    assert len(values) >= 2
    winner = np.flatnonzero(counts == counts.max())
    assert len(winner) == 1 and counts[winner[0]] > 3
    fill = int(values[winner[0]])
    positions = np.argwhere(inp == fill)
    top, left = (int(v) for v in positions.min(axis=0))
    bottom, right = (int(v) for v in positions.max(axis=0))
    panel_h, panel_w = bottom - top + 1, right - left + 1
    assert 4 <= panel_h <= 28 and 4 <= panel_w <= 28
    assert top >= 1 and left >= 1 and bottom + 1 < inp.shape[0] and right + 1 < inp.shape[1]
    corners = ((top, left), (top, right), (bottom, left), (bottom, right))
    expected_in = np.full_like(inp, bg)
    expected_out = np.full_like(inp, bg)
    expected_in[top:bottom + 1, left:right + 1] = fill
    expected_out[top:bottom + 1, left:right + 1] = fill
    painted = np.zeros(inp.shape, dtype=bool)
    used_cues = set()
    active, marker_colors, rects = [], [], []
    for corner, anchor in enumerate(corners):
        color = int(inp[anchor])
        assert color != bg
        if color == fill:
            continue
        active.append(corner)
        marker_colors.append(color)
        points = [tuple(int(v) for v in p) for p in np.argwhere(inp == color)]
        assert len(points) == 3
        edge_row = top - 1 if corner in (0, 1) else bottom + 1
        edge_col = left - 1 if corner in (0, 2) else right + 1
        h_cues = [p for p in points if p[0] == edge_row and left <= p[1] <= right]
        v_cues = [p for p in points if p[1] == edge_col and top <= p[0] <= bottom]
        assert len(h_cues) == len(v_cues) == 1
        assert set(points) == {anchor, h_cues[0], v_cues[0]}
        assert h_cues[0] not in used_cues and v_cues[0] not in used_cues
        used_cues.update((h_cues[0], v_cues[0]))
        r0, r1 = sorted((anchor[0], v_cues[0][0]))
        c0, c1 = sorted((anchor[1], h_cues[0][1]))
        assert (r1 - r0 + 1, c1 - c0 + 1) != (1, 1)
        assert not painted[r0:r1 + 1, c0:c1 + 1].any()
        painted[r0:r1 + 1, c0:c1 + 1] = True
        expected_in[anchor] = color
        expected_in[h_cues[0]] = color
        expected_in[v_cues[0]] = color
        expected_out[r0:r1 + 1, c0:c1 + 1] = color
        rects.append((r0 - top, r1 - top, c0 - left, c1 - left))
    assert 1 <= len(active) <= 4 and len(set(marker_colors)) == len(marker_colors)
    assert np.array_equal(inp, expected_in) and np.array_equal(out, expected_out)
    assert not np.array_equal(inp, out)
    return {'background': bg, 'fill': fill, 'panel_shape': (panel_h, panel_w),
            'canvas_shape': inp.shape, 'active': frozenset(active),
            'markers': frozenset(marker_colors), 'rectangles': tuple(rects),
            'margins': (top, inp.shape[0] - bottom - 1, left, inp.shape[1] - right - 1)}


def _validate(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    inputs, pairs = set(), set()
    infos = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = pair['input'], pair['output']
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert 0 <= int(grid.min()) and int(grid.max()) <= 9
            sig = (_signature(inp), _signature(out))
            assert sig[0] not in inputs and sig not in pairs
            inputs.add(sig[0])
            pairs.add(sig)
            assert not _source_copy(inp, out)
            infos[split].append(_analyze_pair(inp, out))
    covered = set().union(*(info['active'] for info in infos['train']))
    assert covered == set(CORNERS)
    assert any(any((r1 - r0) != (c1 - c0) for r0, r1, c0, c1 in info['rectangles'])
               for info in infos['train'])
    train_strip = any(any(r0 == r1 or c0 == c1 for r0, r1, c0, c1 in info['rectangles'])
                      for info in infos['train'])
    test_strip = any(any(r0 == r1 or c0 == c1 for r0, r1, c0, c1 in info['rectangles'])
                     for info in infos['test'])
    assert train_strip or not test_strip


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _validate(puzzle)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError('malformed generated puzzle') from exc


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    n_train, n_test = _resolve_counts(int(seed), num_train, num_test)
    count_mode = int(rng.integers(4))
    size_mode = int(rng.integers(6))
    stable_roles = None
    if rng.random() < 0.31:
        stable_roles = tuple(int(x) for x in rng.choice(10, 2, replace=False))
    fixed_shape = None
    if size_mode == 1 and rng.random() < 0.36:
        fixed_shape = (_side(rng), _side(rng))
    strip_puzzle = rng.random() < 0.53
    train_plan = _train_plan(rng, n_train, count_mode)
    test_plan = _test_plan(rng, n_test, count_mode, train_plan)
    puzzle = {'train': [], 'test': []}
    inputs = set()
    previous_markers = ()
    training_shapes = []
    for split, plan in (('train', train_plan), ('test', test_plan)):
        for index, active in enumerate(plan):
            regime = 'source_train' if split == 'train' else 'target_test'
            for _ in range(80):
                child = np.random.default_rng(int(rng.integers(0, (1 << 63) - 1)))
                shape = _shape(child, regime, size_mode, training_shapes, fixed_shape)
                colors = _colors(child, stable_roles, previous_markers)
                force_strip = strip_puzzle and split == 'train' and index == 0
                try:
                    inp, out, markers = _generate_pair(child, regime, shape, active, colors, force_strip, strip_puzzle)
                except RuntimeError:
                    continue
                if _signature(inp) in inputs or _source_copy(inp, out):
                    continue
                inputs.add(_signature(inp))
                puzzle[split].append({'input': inp, 'output': out})
                previous_markers = markers
                if split == 'train':
                    training_shapes.append(shape)
                break
            else:
                raise RuntimeError('could not construct a unique readable pair')
    validate_generated_puzzle(puzzle)
    return puzzle


def _equal_puzzles(first, second):
    return all(np.array_equal(a[key], b[key]) for split in ('train', 'test')
               for a, b in zip(first[split], second[split]) for key in ('input', 'output'))


def self_test() -> None:
    for seed, requested_train, requested_test in (
        (1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
        (2001, None, 1), (2002, None, 4), (2003, 2, None), (2004, 10, None),
        (271828, 7, 3), (314159, None, None),
    ):
        puzzle = generate_puzzle(seed, requested_train, requested_test)
        assert _equal_puzzles(puzzle, generate_puzzle(seed, requested_train, requested_test))
        if requested_train is not None:
            assert len(puzzle['train']) == requested_train
        if requested_test is not None:
            assert len(puzzle['test']) == requested_test
        validate_generated_puzzle(puzzle)
    for index, args in enumerate((
        {'num_train': 1}, {'num_train': 11}, {'num_test': 0}, {'num_test': 5},
        {'num_train': 2, 'num_test': 2}, {'num_train': 3, 'num_test': 4},
        {'num_train': 2.0, 'num_test': 1},
    )):
        try:
            generate_puzzle(3000 + index, **args)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count request accepted')
    train_counts, test_counts, signatures = set(), set(), set()
    panel_shapes, canvas_shapes, corner_sets, role_pairs, rectangle_sizes = set(), set(), set(), set(), set()
    mixed_tests = stable_counts = reverse_counts = source_shifts = 0
    variable_roles = stable_roles = marker_reuse = 0
    strips = trained_strips = shared_columns = shared_rows = edge_endpoints = 0
    equal_size = smaller_test = width_growth = height_growth = both_growth = 0
    stable_size = variable_size = small_panel = large_panel = 0
    tall_panel = wide_panel = canvas_fifteen = canvas_thirty = combined_axes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        signatures.add(tuple((_signature(p['input']), _signature(p['output']))
                             for split in ('train', 'test') for p in puzzle[split]))
        infos = {split: [_analyze_pair(p['input'], p['output']) for p in puzzle[split]]
                 for split in ('train', 'test')}
        train, test = infos['train'], infos['test']
        tc = {len(info['active']) for info in train}
        ec = {len(info['active']) for info in test}
        mixed_tests += len(ec) > 1
        stable_counts += len(tc | ec) == 1
        reverse_counts += 4 in tc and any(count < 4 for count in ec)
        source_shifts += max(tc) < 4 and ec == {4}
        roles = {(info['background'], info['fill']) for info in train + test}
        variable_roles += len(roles) > 1
        stable_roles += len(roles) == 1
        all_shapes = {info['panel_shape'] for info in train + test}
        variable_size += len(all_shapes) > 1
        stable_size += len(all_shapes) == 1
        max_train_h = max(info['panel_shape'][0] for info in train)
        max_train_w = max(info['panel_shape'][1] for info in train)
        equal_size += any(info['panel_shape'] in {item['panel_shape'] for item in train}
                          for info in test)
        smaller_test += any(info['panel_shape'][0] < max_train_h and
                            info['panel_shape'][1] < max_train_w for info in test)
        width_growth += any(info['panel_shape'][0] == max_train_h and
                            info['panel_shape'][1] > max_train_w for info in test)
        height_growth += any(info['panel_shape'][0] > max_train_h and
                             info['panel_shape'][1] == max_train_w for info in test)
        both_growth += any(info['panel_shape'][0] > max_train_h and
                           info['panel_shape'][1] > max_train_w for info in test)
        combined_axes += len(roles) > 1 and len(all_shapes) > 1 and len(tc | ec) > 1
        for first, second in zip(train + test, (train + test)[1:]):
            marker_reuse += bool(first['markers'] & {second['background'], second['fill']})
        trained_strips += any(any(r0 == r1 or c0 == c1 for r0, r1, c0, c1 in info['rectangles'])
                              for info in train)
        for info in train + test:
            panel_shapes.add(info['panel_shape'])
            canvas_shapes.add(info['canvas_shape'])
            small_panel += min(info['panel_shape']) <= 6
            large_panel += max(info['panel_shape']) >= 24
            tall_panel += info['panel_shape'][0] > info['panel_shape'][1]
            wide_panel += info['panel_shape'][1] > info['panel_shape'][0]
            canvas_fifteen += info['canvas_shape'] == (15, 15)
            canvas_thirty += 30 in info['canvas_shape']
            corner_sets.add(info['active'])
            role_pairs.add((info['background'], info['fill']))
            rects = info['rectangles']
            for r0, r1, c0, c1 in rects:
                rectangle_sizes.add((r1 - r0 + 1, c1 - c0 + 1))
                strips += r0 == r1 or c0 == c1
                edge_endpoints += (r1 - r0 + 1 == info['panel_shape'][0] or
                                   c1 - c0 + 1 == info['panel_shape'][1])
            for j, (r0, r1, c0, c1) in enumerate(rects):
                for s0, s1, d0, d1 in rects[j + 1:]:
                    shared_columns += max(c0, d0) <= min(c1, d1) and (r1 < s0 or s1 < r0)
                    shared_rows += max(r0, s0) <= min(r1, s1) and (c1 < d0 or d1 < c0)
    assert len(train_counts) >= 4 and len(test_counts) >= 3 and len(signatures) == 42
    assert len(panel_shapes) >= 35 and len(canvas_shapes) >= 35 and len(corner_sets) >= 10
    assert len(role_pairs) >= 20 and len(rectangle_sizes) >= 35
    assert mixed_tests and stable_counts and reverse_counts and source_shifts
    assert variable_roles and stable_roles and marker_reuse
    assert strips and trained_strips and shared_columns and shared_rows and edge_endpoints
    assert equal_size and smaller_test and width_growth and height_growth and both_growth
    assert stable_size and variable_size and small_panel and large_panel
    assert tall_panel and wide_panel and canvas_fifteen and canvas_thirty and combined_axes
