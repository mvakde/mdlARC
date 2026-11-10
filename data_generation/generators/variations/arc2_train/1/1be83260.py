"""A panel array whose exceptional panel is a miniature spatial color key."""

import hashlib
import numpy as np

_SOURCE_INPUT_DIGESTS = {
    '49ece5d5e55f03047d669a740abcc8a1b4171b3acbec468f2c8318a4beeace78',
    'f61fe7bb9235abf1ce4a522b7f3f8898c7e23e35e5e7107d21962a01c4d3f88c',
    '104892b8efca5cd31690db03e5ff78569054e9b172ee19f9842a80fd68847b6d',
    '415d9902272eecde20873645dcbd6729a2c8db8172dc089fcd444452212dcd39',
}
_SHAPES = ((2, 2), (3, 2), (2, 3), (3, 3),
           (1, 2), (2, 1), (1, 3), (3, 1))


def _tight_size(n):
    return 2 * n * n + 2 * n - 1


def _checked_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _checked_count(num_train, 'num_train', 2, 10)
    num_test = _checked_count(num_test, 'num_test', 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return num_train, num_test


def _padding(rng, size):
    available = 30 - size
    if available == 0:
        return 0, 0
    if rng.random() < 0.23:
        total = int(rng.integers(max(0, available - 3), available + 1))
    else:
        total = int(rng.integers(0, min(available, 9) + 1))
    before = int(rng.integers(0, total + 1))
    return before, total - before


def _sample_map(rng, rows, cols, carrier, remaining, teaching):
    count = rows * cols
    if teaching:
        holes = int(rng.integers(2, count))
        accents = int(rng.integers(2, min(holes, 5) + 1))
    else:
        holes = int(rng.integers(1, count + 1))
        accents = int(rng.integers(1, min(holes, 5) + 1))
    colors = [int(v) for v in rng.choice(remaining, size=accents, replace=False)]
    values = colors + [int(v) for v in rng.choice(colors, size=holes - accents)]
    rng.shuffle(values)
    positions = rng.choice(count, size=holes, replace=False)
    result = np.full(count, carrier, dtype=np.int64)
    result[positions] = values
    return result.reshape(rows, cols)


def _repaint(rows, cols, carrier, key_map):
    ph, pw = 2 * rows + 1, 2 * cols + 1
    out = np.full((_tight_size(rows), _tight_size(cols)), carrier, dtype=np.int64)
    solid = np.ones((ph, pw), dtype=bool)
    solid[1::2, 1::2] = key_map == carrier
    for r in range(rows):
        for c in range(cols):
            panel = out[r * (ph + 1):r * (ph + 1) + ph,
                        c * (pw + 1):c * (pw + 1) + pw]
            panel[solid] = key_map[r, c]
    return out


def _build_pair(rng, shape, key_position, teaching):
    rows, cols = shape
    colors = [int(v) for v in rng.permutation(10)]
    background, carrier = colors[:2]
    key_map = _sample_map(rng, rows, cols, carrier, colors[2:], teaching)
    ph, pw = 2 * rows + 1, 2 * cols + 1
    tight = np.full((_tight_size(rows), _tight_size(cols)), background, dtype=np.int64)
    ordinary = np.full((ph, pw), carrier, dtype=np.int64)
    ordinary[1::2, 1::2] = np.where(key_map == carrier, carrier, background)
    key = np.full((ph, pw), carrier, dtype=np.int64)
    key[1::2, 1::2] = key_map
    for r in range(rows):
        for c in range(cols):
            y, x = r * (ph + 1), c * (pw + 1)
            tight[y:y + ph, x:x + pw] = key if (r, c) == key_position else ordinary
    top, bottom = _padding(rng, tight.shape[0])
    left, right = _padding(rng, tight.shape[1])
    inp = np.full((top + tight.shape[0] + bottom,
                   left + tight.shape[1] + right), background, dtype=np.int64)
    inp[top:top + tight.shape[0], left:left + tight.shape[1]] = tight
    return {'input': inp, 'output': _repaint(rows, cols, carrier, key_map)}


def _parse_input(inp):
    """Recover the key and stencil from pixels, including edge-touching arrays."""
    solutions = []
    for background in (int(v) for v in np.unique(inp)):
        occupied = np.argwhere(inp != background)
        if not len(occupied):
            continue
        top, left = (int(v) for v in occupied.min(axis=0))
        end_y, end_x = (int(v) + 1 for v in occupied.max(axis=0))
        bbox = inp[top:end_y, left:end_x]
        for rows, cols in _SHAPES:
            if bbox.shape != (_tight_size(rows), _tight_size(cols)):
                continue
            carrier = int(bbox[0, 0])
            if carrier == background:
                continue
            ph, pw = 2 * rows + 1, 2 * cols + 1
            lattice = np.zeros((ph, pw), dtype=bool)
            lattice[1::2, 1::2] = True
            panels = {}
            valid = True
            for r in range(rows):
                for c in range(cols):
                    y, x = r * (ph + 1), c * (pw + 1)
                    panel = bbox[y:y + ph, x:x + pw]
                    if panel.shape != (ph, pw) or not np.all(panel[~lattice] == carrier):
                        valid = False
                        break
                    panels[r, c] = panel
            if not valid:
                continue
            keys = [pos for pos, panel in panels.items()
                    if np.any((panel != background) & (panel != carrier))]
            if len(keys) != 1:
                continue
            key_position = keys[0]
            key_map = panels[key_position][1::2, 1::2].copy()
            if background in key_map or not np.any(key_map != carrier):
                continue
            ordinary = np.where(key_map == carrier, carrier, background)
            if any(not np.array_equal(panel[1::2, 1::2], ordinary)
                   for pos, panel in panels.items() if pos != key_position):
                continue
            expected = np.full(bbox.shape, background, dtype=np.int64)
            for (r, c), panel in panels.items():
                y, x = r * (ph + 1), c * (pw + 1)
                expected[y:y + ph, x:x + pw] = panel
            if not np.array_equal(expected, bbox):
                continue
            solutions.append({'background': background, 'carrier': carrier,
                              'rows': rows, 'cols': cols, 'key_position': key_position,
                              'key_map': key_map,
                              'padding': (top, inp.shape[0] - end_y,
                                          left, inp.shape[1] - end_x)})
    assert len(solutions) == 1, f'input has {len(solutions)} interpretations'
    return solutions[0]


def _pattern(info):
    ids = {info['carrier']: 0}
    values = []
    for item in info['key_map'].flat:
        item = int(item)
        if item not in ids:
            ids[item] = len(ids)
        values.append(ids[item])
    return info['rows'], info['cols'], tuple(values)


def _layout_plan(rng, train_count, test_count):
    mode = int(rng.integers(4))
    if mode == 0:
        train, pool = [(2, 2), (3, 2)], [(2, 3), (3, 3)]
    elif mode == 1:
        train, pool = [(2, 3), (3, 3)], [(2, 2), (3, 2)]
    elif mode == 2:
        train, pool = [(2, 2), (3, 3)], [(2, 2), (3, 3)]
    else:
        train = [(2, 3), (1, 3) if rng.random() < 0.5 else (3, 1)]
        pool = [(3, 2), (2, 1), (1, 2), (3, 3), (1, 3), (3, 1)]
        rng.shuffle(pool)
    extra_pool = (tuple(train) if mode in (0, 1) else _SHAPES)
    train.extend(extra_pool[int(rng.integers(len(extra_pool)))]
                 for _ in range(train_count - 2))
    tests = [pool[i % len(pool)] for i in range(test_count)]
    return train, tests


def _source_copy(inp):
    return hashlib.sha256(inp.astype(np.uint8).tobytes()).hexdigest() in _SOURCE_INPUT_DIGESTS


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    train_shapes, test_shapes = _layout_plan(rng, num_train, num_test)
    puzzle = {'train': [], 'test': []}
    used_inputs, used_patterns = set(), set()
    first_key = None
    for split, shapes in (('train', train_shapes), ('test', test_shapes)):
        for index, shape in enumerate(shapes):
            rows, cols = shape
            positions = [(r, c) for r in range(rows) for c in range(cols)]
            if split == 'train' and index == 1 and first_key in positions:
                positions.remove(first_key)
            for _ in range(160):
                key_position = positions[int(rng.integers(len(positions)))]
                pair = _build_pair(rng, shape, key_position,
                                   teaching=split == 'train' and index == 0)
                inp = pair['input']
                inp_key = (inp.shape, inp.tobytes())
                if inp_key in used_inputs or _source_copy(inp):
                    continue
                info = _parse_input(inp)
                pattern = _pattern(info)
                if pattern in used_patterns:
                    continue
                used_inputs.add(inp_key)
                used_patterns.add(pattern)
                puzzle[split].append(pair)
                if split == 'train' and index == 0:
                    first_key = key_position
                break
            else:
                raise RuntimeError(f'could not construct distinct {shape} pair')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    inputs, pairs, patterns = set(), set(), set()
    analyses = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dim <= 30 for dim in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            inp, out = pair['input'], pair['output']
            input_key = (inp.shape, inp.tobytes())
            pair_key = input_key, (out.shape, out.tobytes())
            assert input_key not in inputs and pair_key not in pairs
            assert not _source_copy(inp)
            inputs.add(input_key)
            pairs.add(pair_key)
            info = _parse_input(inp)
            pattern = _pattern(info)
            assert pattern not in patterns
            patterns.add(pattern)
            expected = _repaint(info['rows'], info['cols'], info['carrier'], info['key_map'])
            assert np.array_equal(out, expected)
            assert not np.array_equal(inp, out)
            analyses[split].append(info)
    train = analyses['train']
    assert train[0]['rows'] >= 2 and train[0]['cols'] >= 2
    assert len(set(int(x) for x in train[0]['key_map'].flat) - {train[0]['carrier']}) >= 2
    assert np.any(train[0]['key_map'] == train[0]['carrier'])
    assert (train[0]['rows'], train[0]['cols']) != (train[1]['rows'], train[1]['cols'])
    assert train[0]['key_position'] != train[1]['key_position']


def _equal_puzzles(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k])
        for x, y in zip(a[s], b[s]) for k in ('input', 'output'))
        for s in ('train', 'test'))


def self_test() -> None:
    for i, (nt, ne) in enumerate(((2, 1), (10, 1), (10, 4),
                                   (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(1000 + i, nt, ne)
        validate_generated_puzzle(puzzle)
        if nt is not None:
            assert len(puzzle['train']) == nt
        if ne is not None:
            assert len(puzzle['test']) == ne
    for nt, ne in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(1200, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    assert _equal_puzzles(generate_puzzle(1300, 6, 3), generate_puzzle(1300, 6, 3))
    assert not _equal_puzzles(generate_puzzle(1300, 6, 3), generate_puzzle(1301, 6, 3))
    train_counts, test_counts = set(), set()
    shapes = {'train': set(), 'test': set()}
    keys, occupancy, accents, paddings, signatures = set(), set(), set(), set(), set()
    split_keys = {'train': set(), 'test': set()}
    single_hole = full_key = repeats = large_margin = edge_touch = max_canvas = 0
    rich_train = fresh_test = truly_unseen_test = role_exchange = combined = 0
    equal_richness = train_richer = test_richer = asymmetric_maps = 0
    changed_backgrounds = changed_carriers = 0
    source_like = reverse = matched = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        local = {'train': [], 'test': []}
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _parse_input(pair['input'])
                assert np.array_equal(_repaint(info['rows'], info['cols'],
                                               info['carrier'], info['key_map']), pair['output'])
                local[split].append(info)
                shapes[split].add((info['rows'], info['cols']))
                keys.add((info['rows'], info['cols'], info['key_position']))
                split_keys[split].add(info['key_position'])
                holes = int(np.count_nonzero(info['key_map'] != info['carrier']))
                accent_count = len(set(int(v) for v in info['key_map'].flat) - {info['carrier']})
                occupancy.add(holes)
                accents.add(accent_count)
                paddings.add(info['padding'])
                signatures.add(_pattern(info))
                single_hole += holes == 1
                full_key += holes == info['rows'] * info['cols']
                repeats += holes > accent_count
                large_margin += max(info['padding']) >= 4
                edge_touch += 0 in info['padding']
                max_canvas += 30 in pair['input'].shape
                if info['rows'] > 1 and info['cols'] > 1:
                    asymmetric_maps += (not np.array_equal(info['key_map'],
                                                           np.flipud(info['key_map'])) and
                                        not np.array_equal(info['key_map'],
                                                           np.fliplr(info['key_map'])))
        changed_backgrounds += len({x['background'] for xs in local.values() for x in xs}) > 1
        changed_carriers += len({x['carrier'] for xs in local.values() for x in xs}) > 1
        tr = {(x['rows'], x['cols']) for x in local['train']}
        te = {(x['rows'], x['cols']) for x in local['test']}
        train_accents = set().union(*(set(int(v) for v in x['key_map'].flat)
                                      - {x['carrier']} for x in local['train']))
        test_accents = set().union(*(set(int(v) for v in x['key_map'].flat)
                                     - {x['carrier']} for x in local['test']))
        rich_train += any(len(set(int(v) for v in x['key_map'].flat)
                              - {x['carrier']}) >= 3 for x in local['train'])
        fresh_test += bool(test_accents - train_accents)
        train_visible = {int(v) for pair in puzzle['train'] for v in pair['input'].flat}
        test_visible = {int(v) for pair in puzzle['test'] for v in pair['input'].flat}
        truly_unseen_test += bool(test_visible - train_visible)
        train_max = max(len(set(int(v) for v in x['key_map'].flat) - {x['carrier']})
                        for x in local['train'])
        test_max = max(len(set(int(v) for v in x['key_map'].flat) - {x['carrier']})
                       for x in local['test'])
        equal_richness += train_max == test_max
        train_richer += train_max > test_max
        test_richer += train_max < test_max
        role_exchange += bool({x['carrier'] for xs in local.values() for x in xs}
                              & (train_accents | test_accents))
        combined += (len(tr | te) >= 3 and
                     len({x['carrier'] for xs in local.values() for x in xs}) >= 3 and
                     any(np.all(x['key_map'] != x['carrier'])
                         for xs in local.values() for x in xs))
        source_like += all(c == 2 for _, c in tr) and all(c == 3 for _, c in te)
        reverse += all(c == 3 for _, c in tr) and all(c == 2 for _, c in te)
        matched += bool(tr & te)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes['train']) == 8 and len(shapes['test']) == 8
    assert len(keys) >= 22 and len(occupancy) >= 7
    assert len(split_keys['train']) == 9 and len(split_keys['test']) == 9
    assert len(accents) == 5 and len(paddings) >= 60
    assert len(signatures) >= 90
    assert changed_backgrounds >= 35 and changed_carriers >= 35
    assert source_like >= 2 and reverse >= 2 and matched >= 5
    assert single_hole >= 10 and full_key >= 10 and repeats >= 20
    assert large_margin >= 20 and edge_touch >= 20 and max_canvas >= 5
    assert rich_train >= 10 and fresh_test >= 10 and role_exchange >= 10
    assert truly_unseen_test >= 5
    assert equal_richness >= 5 and train_richer >= 5 and test_richer >= 5
    assert asymmetric_maps >= 30
    assert combined >= 5
