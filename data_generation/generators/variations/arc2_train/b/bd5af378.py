"""Boundary-L ARC puzzle generator."""
import numpy as np

CORNERS = ('UL', 'UR', 'LL', 'LR')
SOURCE = (("LR", 5, 6, 7, 9, 8), ("LL", 5, 8, 3, 6, 8),
          ("LR", 8, 6, 5, 1, 8), ("UL", 10, 10, 1, 6, 8),
          ("LL", 10, 8, 7, 3, 8))
KINDS = ('square', 'near_tall', 'near_wide', 'tall', 'wide')
MODES = ('retain', 'one_role', 'cross_pair', 'reverse', 'novel')


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer or None')
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return value


def _counts(seed, rng, num_train, num_test):
    train = _count(num_train, 'num_train', 2, 10)
    test = _count(num_test, 'num_test', 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        minimum = max(2, test + 1)
        train = minimum + (seed // 4) % (11 - minimum)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError('num_train must exceed num_test')
    return train, test


def _corner(height, width, orientation):
    assert orientation in CORNERS
    return (0 if orientation[0] == 'U' else height - 1,
            0 if orientation[1] == 'L' else width - 1,
            1 if orientation[0] == 'U' else -1,
            1 if orientation[1] == 'L' else -1)


def _input(height, width, orientation, l_color, field_color):
    grid = np.full((height, width), field_color, dtype=np.int64)
    row, col, _, _ = _corner(height, width, orientation)
    grid[row, :] = l_color
    grid[:, col] = l_color
    return grid


def _output(height, width, orientation, l_color, field_color, neutral):
    grid = np.full((height, width), neutral, dtype=np.int64)
    row, col, dr, dc = _corner(height, width, orientation)
    grid[row, :] = field_color
    grid[:, col] = field_color
    grid[row, col] = neutral
    inset_row, inset_col = row + dr, col + dc
    for c in range(inset_col + dc, width if dc > 0 else -1, dc):
        grid[inset_row, c] = l_color
    for r in range(inset_row + dr, height if dr > 0 else -1, dr):
        grid[r, inset_col] = l_color
    r, c = row + 2 * dr, col + 2 * dc
    while 0 <= r < height and 0 <= c < width:
        grid[r, c] = field_color
        r += dr
        c += dc
    return grid


def _analyze(grid):
    h, w = grid.shape
    assert h >= 3 and w >= 3
    colors = [int(c) for c in np.unique(grid)]
    assert len(colors) == 2
    matches = []
    for orientation in CORNERS:
        for l_color in colors:
            field_color = next(c for c in colors if c != l_color)
            if np.array_equal(grid, _input(h, w, orientation, l_color, field_color)):
                matches.append((orientation, h, w, l_color, field_color))
    assert len(matches) == 1
    return matches[0]


def _sig(grid):
    return grid.shape, tuple(int(cell) for cell in grid.flat)


def _source_signatures():
    return {(_sig(_input(h, w, o, l, f)), _sig(_output(h, w, o, l, f, n)))
            for o, h, w, l, f, n in SOURCE}


def _length(rng):
    band = int(rng.choice(4, p=[0.10, 0.34, 0.40, 0.16]))
    lo, hi = ((3, 4), (5, 8), (9, 20), (21, 30))[band]
    return int(rng.integers(lo, hi + 1))


def _dimensions(rng, kind):
    if kind == 'square':
        side = _length(rng)
        return side, side
    if kind == 'near_tall':
        width = min(29, _length(rng))
        return width + 1, width
    if kind == 'near_wide':
        height = min(29, _length(rng))
        return height, height + 1
    short = min(28, _length(rng))
    long = min(30, short + int(rng.integers(2, 13)))
    return (long, short) if kind == 'tall' else (short, long)


def _kind_matches(h, w, kind):
    return ((kind == 'square' and h == w) or
            (kind == 'near_tall' and h == w + 1) or
            (kind == 'near_wide' and w == h + 1) or
            (kind == 'tall' and h >= w + 2) or
            (kind == 'wide' and w >= h + 2))


def _orientations(rng, train_count, test_count):
    order = [str(x) for x in rng.permutation(CORNERS)]
    breadth = 2 if train_count == 2 else int(rng.integers(2, min(4, train_count) + 1))
    selected = order[:breadth]
    training = selected + [selected[int(rng.integers(0, breadth))]
                           for _ in range(train_count - breadth)]
    familiar = list(dict.fromkeys(training))
    unseen = [o for o in CORNERS if o not in familiar]
    testing = []
    for index in range(test_count):
        pool = unseen if unseen and (index == 1 or (index == 0 and rng.random() < 0.45)) else familiar
        testing.append(pool[int(rng.integers(0, len(pool)))])
    if test_count > 1 and len(set(testing)) == 1:
        other = [o for o in CORNERS if o != testing[0]]
        testing[-1] = other[int(rng.integers(0, len(other)))]
    return training, testing


def _test_role(rng, train_roles, palette, mode):
    anchor = train_roles[int(rng.integers(0, len(train_roles)))]
    if mode == 'retain':
        return anchor
    if mode == 'one_role':
        if rng.random() < 0.5:
            choices = [c for c in palette if c not in anchor]
            return anchor[0], choices[int(rng.integers(0, len(choices)))]
        choices = [c for c in palette if c not in anchor]
        return choices[int(rng.integers(0, len(choices)))], anchor[1]
    if mode == 'cross_pair':
        others = [role for role in train_roles if role != anchor and role[0] != anchor[1]]
        if others:
            return anchor[1], others[int(rng.integers(0, len(others)))][0]
        return train_roles[0][1], train_roles[1][0]
    if mode == 'reverse':
        return anchor[1], anchor[0]
    used = {c for role in train_roles for c in role}
    novel = [c for c in palette if c not in used]
    assert novel
    color = novel[int(rng.integers(0, len(novel)))]
    return (color, anchor[1]) if rng.random() < 0.5 else (anchor[0], color)


def _pair(h, w, orientation, role, neutral):
    l_color, field_color = role
    return {'input': _input(h, w, orientation, l_color, field_color),
            'output': _output(h, w, orientation, l_color, field_color, neutral)}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    if seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    colors = [int(c) for c in rng.permutation(10)]
    neutral, palette = colors[0], colors[1:]
    train_os, test_os = _orientations(rng, train_count, test_count)
    train_kinds = ['wide', 'tall'] + (['square'] if train_count >= 3 else [])
    train_kinds += [KINDS[int(rng.integers(0, 5))] for _ in range(train_count - len(train_kinds))]
    rng.shuffle(train_kinds)
    test_kinds = [KINDS[int(rng.integers(0, 5))] for _ in range(test_count)]
    if test_count > 1 and len(set(test_kinds)) == 1:
        test_kinds[-1] = KINDS[(KINDS.index(test_kinds[-1]) + int(rng.integers(1, 5))) % 5]
    train_roles = [(palette[0], palette[1]), (palette[2], palette[3])]
    for _ in range(train_count - 2):
        l_color = palette[int(rng.integers(0, 6))]
        choices = [c for c in palette[:6] if c != l_color]
        train_roles.append((l_color, choices[int(rng.integers(0, len(choices)))]))
    test_roles = []
    for index in range(test_count):
        role = _test_role(rng, train_roles, palette, MODES[(seed + index) % 5])
        if role in test_roles:
            role = _test_role(rng, train_roles, palette, MODES[(seed + index + 2) % 5])
        if role in test_roles:
            role = next((a, b) for a in palette for b in palette if a != b and (a, b) not in test_roles)
        test_roles.append(role)
    jobs = [('train', o, k, r) for o, k, r in zip(train_os, train_kinds, train_roles)]
    jobs += [('test', o, k, r) for o, k, r in zip(test_os, test_kinds, test_roles)]
    puzzle = {'train': [], 'test': []}
    source, used_shapes = _source_signatures(), set()
    pair_seeds = rng.integers(0, 2**64, size=len(jobs), dtype=np.uint64)
    for job_index, (section, orientation, kind, role) in enumerate(jobs):
        pair_rng = np.random.default_rng(int(pair_seeds[job_index]) +
                                         (job_index + 1) * 2**64)
        chosen = None
        for attempt in range(180):
            h, w = _dimensions(pair_rng, kind)
            if (h, w) in used_shapes or (job_index == 0 and min(h, w) < 5):
                continue
            candidate = _pair(h, w, orientation, role, neutral)
            if (_sig(candidate['input']), _sig(candidate['output'])) not in source:
                chosen = candidate
                break
        if chosen is None:
            for h in range(3, 31):
                for w in range(3, 31):
                    if ((h, w) in used_shapes or not _kind_matches(h, w, kind)
                            or (job_index == 0 and min(h, w) < 5)):
                        continue
                    candidate = _pair(h, w, orientation, role, neutral)
                    if (_sig(candidate['input']), _sig(candidate['output'])) not in source:
                        chosen = candidate
                        break
                if chosen is not None:
                    break
        if chosen is None:
            raise RuntimeError('no distinct pair available')
        puzzle[section].append(chosen)
        used_shapes.add(chosen['input'].shape)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    source, inputs, pairs, shapes, neutrals = _source_signatures(), set(), set(), set(), set()
    records = {'train': [], 'test': []}
    for section in ('train', 'test'):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = pair['input'], pair['output']
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert 0 <= int(grid.min()) and int(grid.max()) <= 9
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert inp.shape not in shapes
            shapes.add(inp.shape)
            pair_sig = (_sig(inp), _sig(out))
            assert pair_sig[0] not in inputs and pair_sig not in pairs and pair_sig not in source
            inputs.add(pair_sig[0]); pairs.add(pair_sig)
            orientation, h, w, l_color, field_color = _analyze(inp)
            output_colors = {int(c) for c in np.unique(out)}
            assert len(output_colors) == 3 and {l_color, field_color} < output_colors
            neutral = next(iter(output_colors - {l_color, field_color}))
            neutrals.add(neutral)
            assert np.array_equal(out, _output(h, w, orientation, l_color, field_color, neutral))
            records[section].append((orientation, h, w, l_color, field_color))
    assert len(neutrals) == 1
    neutral = next(iter(neutrals))
    assert all(neutral not in rec[3:5] for section in records.values() for rec in section)
    assert any(h < w for _, h, w, _, _ in records['train'])
    assert any(h > w for _, h, w, _, _ in records['train'])
    assert len({rec[0] for rec in records['train']}) >= 2
    assert any(min(rec[1], rec[2]) >= 5 for rec in records['train'])
    if len(records['train']) >= 3:
        assert any(h == w for _, h, w, _, _ in records['train'])
    if len(records['test']) > 1:
        assert len({rec[0] for rec in records['test']}) >= 2
        assert len({rec[3:5] for rec in records['test']}) >= 2


def _puzzle_sig(puzzle):
    return tuple(tuple((_sig(pair['input']), _sig(pair['output'])) for pair in puzzle[section])
                 for section in ('train', 'test'))


def self_test() -> None:
    train_counts, test_counts, neutrals = set(), set(), set()
    train_os, test_os, train_aspects, test_aspects = set(), set(), set(), set()
    test_sizes, layouts, normalized, roles = set(), set(), set(), set()
    corner_aspects, color_modes, mode_combinations, source_layouts = set(), set(), set(), 0
    source_shapes = {(o, h, w) for o, h, w, _, _, _ in SOURCE}
    familiar = unseen = multicolor_tests = total = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle['train'])); test_counts.add(len(puzzle['test']))
        train_records = [_analyze(p['input']) for p in puzzle['train']]
        test_records = [_analyze(p['input']) for p in puzzle['test']]
        seen_train_o = {rec[0] for rec in train_records}
        train_roles = {rec[3:5] for rec in train_records}
        train_l = {rec[3] for rec in train_records}
        train_f = {rec[4] for rec in train_records}
        if len({rec[3:5] for rec in test_records}) > 1:
            multicolor_tests += 1
        for section, records in (('train', train_records), ('test', test_records)):
            for rec, pair in zip(records, puzzle[section]):
                o, h, w, l_color, field_color = rec
                extra = {int(c) for c in np.unique(pair['output'])} - {l_color, field_color}
                assert len(extra) == 1
                neutral = next(iter(extra)); neutrals.add(neutral)
                assert np.array_equal(pair['output'], _output(h, w, o, l_color, field_color, neutral))
                aspect = 'square' if h == w else 'tall' if h > w else 'wide'
                corner_aspects.add((section, o, aspect))
                source_layouts += (o, h, w) in source_shapes
                if section == 'train':
                    train_os.add(o); train_aspects.add(aspect)
                else:
                    test_os.add(o); test_aspects.add(aspect); test_sizes.add((h, w))
                    familiar += o in seen_train_o; unseen += o not in seen_train_o
                    roles.add((l_color, field_color))
                    observed_modes = set()
                    if (l_color, field_color) in train_roles:
                        observed_modes.add('retain')
                    if (field_color, l_color) in train_roles:
                        observed_modes.add('reverse')
                    if (l_color in train_f and field_color in train_l and
                            all((l_color != right or field_color != left)
                                for left, right in train_roles)):
                        observed_modes.add('cross_pair')
                    if l_color not in train_l | train_f or field_color not in train_l | train_f:
                        observed_modes.add('novel')
                    if any((l_color == left and field_color != right) or
                           (field_color == right and l_color != left)
                           for left, right in train_roles):
                        observed_modes.add('one_role')
                    color_modes.update(observed_modes)
                    mode_combinations.update((mode, o, aspect) for mode in observed_modes)
                layouts.add((o, h, w))
                symbolic = np.where(pair['input'] == l_color, 1, 0).astype(np.int64)
                normalized.add(_sig(symbolic)); total += 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3 and len(neutrals) >= 7
    assert train_os == test_os == set(CORNERS)
    assert train_aspects == test_aspects == {'wide', 'tall', 'square'}
    assert len(corner_aspects) == 24
    assert color_modes == set(MODES)
    assert len(mode_combinations) >= 45
    assert ('novel', 'UR', 'square') in mode_combinations
    assert familiar > 10 and unseen > 5 and multicolor_tests >= 10
    assert any(min(h, w) == 3 for h, w in test_sizes)
    assert any(max(h, w) >= 21 for h, w in test_sizes)
    assert any(max(h, w) == 30 for h, w in test_sizes)
    assert any(h == w + 1 for h, w in test_sizes)
    assert source_layouts > 0
    assert len(layouts) >= 100 and len(normalized) >= 100 and len(roles) >= 30 and total > 250
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4), (7, 3),
                                           (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(1000 + index, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None: assert len(puzzle['train']) == train
        if test is not None: assert len(puzzle['test']) == test
    assert _puzzle_sig(generate_puzzle(12345, 7, 3)) == _puzzle_sig(generate_puzzle(12345, 7, 3))
    assert len({_puzzle_sig(generate_puzzle(2000 + i, 6, 2)) for i in range(12)}) >= 10
    for index, (train, test) in enumerate(((1, None), (11, None), (None, 0), (None, 5),
                                           (2, 2), (2, 3), (3, 4), (4, 4))):
        try:
            generate_puzzle(3000 + index, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
