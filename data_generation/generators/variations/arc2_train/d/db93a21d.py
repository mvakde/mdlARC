import numpy as np


SCALES = (2, 4, 6, 8)
SOURCE_LAYOUTS = {
    (20, 20, ((3, 2, 4, 3), (6, 15, 9, 18), (12, 6, 17, 11))),
    (15, 15, ((2, 8, 5, 11), (6, 3, 7, 4), (9, 13, 10, 14))),
    (20, 20, ((0, 7, 1, 10), (8, 2, 11, 5), (9, 14, 14, 19))),
    (10, 10, ((0, 6, 1, 7), (7, 1, 8, 2))),
    (30, 30, ((2, 16, 5, 19), (7, 9, 10, 12), (16, 21, 21, 26), (22, 4, 23, 5))),
}


def _counts(rng, num_train, num_test):
    for value, lo, hi, name in ((num_train, 2, 10, 'num_train'), (num_test, 1, 4, 'num_test')):
        if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
            raise ValueError(f'{name} outside legal range')
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _key(grid):
    return (grid.shape, grid.tobytes())


def _components(grid, core):
    mask = grid == core
    seen = np.zeros(grid.shape, dtype=bool)
    boxes = []
    height, width = grid.shape
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= yy < height and 0 <= xx < width and mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
            r0, r1 = min(y for y, _ in cells), max(y for y, _ in cells)
            c0, c1 = min(x for _, x in cells), max(x for _, x in cells)
            span, tall = c1 - c0 + 1, r1 - r0 + 1
            assert span in SCALES and tall <= span
            assert tall == span or r0 == 0 or r1 == height - 1
            assert len(cells) == span * tall
            boxes.append((r0, c0, r1, c1))
    return sorted(boxes)


def _separate(boxes):
    for i, (r0, c0, r1, c1) in enumerate(boxes):
        for q0, d0, q1, d1 in boxes[:i]:
            if not (r1 + 1 < q0 or q1 + 1 < r0 or c1 + 1 < d0 or d1 + 1 < c0):
                return False
    return True


def _masks(shape, boxes):
    height, width = shape
    frames, stems = [], []
    for r0, c0, r1, c1 in boxes:
        t = (c1 - c0 + 1) // 2
        frame = np.zeros(shape, dtype=bool)
        frame[max(0, r0 - t):min(height, r1 + t + 1), max(0, c0 - t):min(width, c1 + t + 1)] = True
        stem = np.zeros(shape, dtype=bool)
        stem[r1 + t + 1:, c0:c1 + 1] = True
        frames.append(frame)
        stems.append(stem)
    return frames, stems


def _render(shape, boxes, colors):
    bg, core, frame, stem = colors
    out = np.full(shape, bg, dtype=np.int8)
    frames, stems = _masks(shape, boxes)
    for mask in stems:
        out[mask] = stem
    for mask in frames:
        out[mask] = frame
    for r0, c0, r1, c1 in boxes:
        out[r0:r1 + 1, c0:c1 + 1] = core
    return out


def _features(shape, boxes):
    height, width = shape
    frames, stems = _masks(shape, boxes)
    cores = np.zeros(shape, dtype=bool)
    for r0, c0, r1, c1 in boxes:
        cores[r0:r1 + 1, c0:c1 + 1] = True
    overlap = any(np.any(a & b & ~cores) for i, a in enumerate(frames) for b in frames[:i])
    occlusion = any(np.any(a & b & ~cores) for i, a in enumerate(stems) for j, b in enumerate(frames) if i != j)
    touch = any(not np.any(a & b) and (np.any(a[1:] & b[:-1]) or np.any(a[:-1] & b[1:]) or np.any(a[:, 1:] & b[:, :-1]) or np.any(a[:, :-1] & b[:, 1:])) for i, a in enumerate(frames) for b in frames[:i])
    clipped = [False] * 4
    for r0, c0, r1, c1 in boxes:
        t = (c1 - c0 + 1) // 2
        clipped = [x or y for x, y in zip(clipped, (r0 - t < 0, c0 - t < 0, r1 + t >= height, c1 + t >= width))]
    all_frames = np.logical_or.reduce(frames)
    visible_stems = [bool(np.any(s & ~cores & ~all_frames)) for s in stems]
    return {'overlap': overlap, 'occlusion': occlusion, 'touch': touch,
            'isolated': not overlap and not occlusion and not touch,
            'clipped': any(clipped), 'unclipped': not any(clipped),
            'top_clip': clipped[0], 'side_clip': clipped[1] or clipped[3],
            'bottom_clip': clipped[2],
            'core_clip': any(r1 - r0 + 1 < c1 - c0 + 1 for r0, c0, r1, c1 in boxes),
            'no_visible_stem': not all(visible_stems), 'all_stems_visible': all(visible_stems)}


def _row_start(rng, height, size):
    mode = int(rng.choice(3, p=(0.10, 0.10, 0.80)))
    if mode == 0:
        return -int(rng.integers(1, size // 2 + 1))
    if mode == 1:
        return height - size + int(rng.integers(1, size // 2 + 1))
    return int(rng.integers(0, height - size + 1))


def _layout(rng, shape, sizes, interior=False):
    height, width = shape
    boxes = []
    for size in rng.permutation(sizes):
        size = int(size)
        if size > height or size > width:
            return None
        options = []
        for _ in range(70):
            if interior:
                t = size // 2
                if height < size + 2 * t or width < size + 2 * t:
                    return None
                r0 = int(rng.integers(t, height - size - t + 1))
                c0 = int(rng.integers(t, width - size - t + 1))
            else:
                r0 = _row_start(rng, height, size)
                c0 = int(rng.integers(0, width - size + 1))
            box = (max(0, r0), c0, min(height - 1, r0 + size - 1), c0 + size - 1)
            if _separate(boxes + [box]):
                options.append(box)
        if not options:
            return None
        boxes.append(options[int(rng.integers(len(options)))])
    return sorted(boxes)


def _dimensions(rng, sizes, regime):
    count, largest = len(sizes), max(sizes)
    low = max(10, largest + 4, 10 + 3 * max(0, count - 2))
    if largest == 8:
        low = max(low, 17)
    if regime == 'target_test' and rng.random() < 0.22:
        return (30, 30)
    if regime == 'source_train' and rng.random() < 0.15:
        choices = [x for x in (10, 15, 20, 30) if x >= low]
        side = int(rng.choice(choices))
        return side, side
    return int(rng.integers(low, 31)), int(rng.integers(low, 31))


def _make_pair(rng, sizes, regime, colors, desired):
    bg, core, _, _ = colors
    best = None
    for _ in range(120 if isinstance(desired, tuple) else 55):
        shape = _dimensions(rng, sizes, regime)
        boxes = _layout(rng, shape, sizes, interior=desired == 'unclipped')
        if boxes is None or (shape[0], shape[1], tuple(boxes)) in SOURCE_LAYOUTS:
            continue
        features = _features(shape, boxes)
        grid = np.full(shape, bg, dtype=np.int8)
        for r0, c0, r1, c1 in boxes:
            grid[r0:r1 + 1, c0:c1 + 1] = core
        if np.count_nonzero(grid == bg) <= np.count_nonzero(grid == core):
            continue
        pair = {'input': grid, 'output': _render(shape, boxes, colors)}
        if bg not in pair['output'] or colors[2] not in pair['output']:
            continue
        if desired == 'demonstrate':
            good = features['all_stems_visible'] and not features['core_clip']
        else:
            good = all(features[name] for name in desired) if isinstance(desired, tuple) else features[desired]
        if good:
            return pair
        if best is None:
            best = pair
    if isinstance(desired, tuple):
        shape = (20, 20)
        boxes = [(0, 7, 1, 10), (8, 6, 13, 11), (18, 1, 19, 2)]
        grid = np.full(shape, bg, dtype=np.int8)
        for r0, c0, r1, c1 in boxes:
            grid[r0:r1 + 1, c0:c1 + 1] = core
        return {'input': grid, 'output': _render(shape, boxes, colors)}
    if best is None:
        shape = (30, 30)
        boxes = [(10 * (i // 2), 10 * (i % 2), 10 * (i // 2) + size - 1,
                  10 * (i % 2) + size - 1) for i, size in enumerate(sizes)]
        grid = np.full(shape, bg, dtype=np.int8)
        for r0, c0, r1, c1 in boxes:
            grid[r0:r1 + 1, c0:c1 + 1] = core
        return {'input': grid, 'output': _render(shape, boxes, colors)}
    return best


def _pair_features(pair):
    inp = pair['input']
    vals, counts = np.unique(inp, return_counts=True)
    boxes = _components(inp, int(vals[np.argmin(counts)]))
    return _features(inp.shape, boxes)


def _roles(train):
    input_roles, added = [], set()
    for pair in train:
        inp, out = pair['input'], pair['output']
        vals, counts = np.unique(inp, return_counts=True)
        assert len(vals) == 2 and counts[0] != counts[1]
        bg, core = int(vals[np.argmax(counts)]), int(vals[np.argmin(counts)])
        input_roles.append((bg, core))
        assert {bg, core} <= set(int(x) for x in np.unique(out))
        added |= set(int(x) for x in np.unique(out)) - {bg, core}
    assert len(added) == 2
    a, b = sorted(added)
    possible = []
    for frame, stem in ((a, b), (b, a)):
        if all(np.array_equal(_render(pair['input'].shape, _components(pair['input'], core), (bg, core, frame, stem)), pair['output']) for pair, (bg, core) in zip(train, input_roles)):
            possible.append((frame, stem))
    assert len(possible) == 1
    return possible[0], input_roles


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen = set()
    for pair in puzzle['train'] + puzzle['test']:
        assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
        for grid in pair.values():
            assert isinstance(grid, np.ndarray) and grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer)
            assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
            assert np.all((0 <= grid) & (grid <= 9))
        assert pair['input'].shape == pair['output'].shape
        key = _key(pair['input'])
        assert key not in seen
        seen.add(key)
    (frame, stem), train_roles = _roles(puzzle['train'])
    assert len({background for background, _ in train_roles}) > 1
    assert len({core for _, core in train_roles}) > 1
    train_scales, train_stems, train_frames = set(), False, False
    for split in ('train', 'test'):
        for index, pair in enumerate(puzzle[split]):
            inp, out = pair['input'], pair['output']
            vals, counts = np.unique(inp, return_counts=True)
            assert len(vals) == 2 and counts[0] != counts[1]
            bg, core = int(vals[np.argmax(counts)]), int(vals[np.argmin(counts)])
            assert len({bg, core, frame, stem}) == 4
            if split == 'train':
                assert (bg, core) == train_roles[index]
            boxes = _components(inp, core)
            assert 1 <= len(boxes) <= 5 and _separate(boxes)
            assert (inp.shape[0], inp.shape[1], tuple(boxes)) not in SOURCE_LAYOUTS
            assert np.array_equal(out, _render(inp.shape, boxes, (bg, core, frame, stem)))
            assert not np.array_equal(inp, out)
            if split == 'train':
                train_scales.update(c1 - c0 + 1 for r0, c0, r1, c1 in boxes)
                train_stems |= stem in out
                train_frames |= frame in out
    assert {2, 4, 6} <= train_scales and train_stems and train_frames
    training_features = {name for pair in puzzle['train'] for name, value in _pair_features(pair).items() if value}
    testing_features = {name for pair in puzzle['test'] for name, value in _pair_features(pair).items() if value}
    assert {'core_clip', 'occlusion'} & testing_features <= training_features


def _size_specs(rng, n, regime):
    specs = []
    weights = (0.34, 0.34, 0.28, 0.04) if regime == 'source_train' else (0.25, 0.25, 0.30, 0.20)
    for _ in range(n):
        count = int(rng.choice((1, 2, 3, 4, 5), p=(0.11, 0.28, 0.31, 0.23, 0.07)))
        if rng.random() < 0.22:
            sizes = [int(rng.choice(SCALES, p=weights))] * count
        else:
            sizes = [int(x) for x in rng.choice(SCALES, count, p=weights)]
        specs.append(sizes)
    if regime == 'source_train':
        missing = list({2, 4, 6} - {s for sizes in specs for s in sizes})
        rng.shuffle(missing)
        for size in missing:
            choices = [i for i, sizes in enumerate(specs) if len(sizes) < 5]
            if choices:
                specs[int(rng.choice(choices))].append(size)
            else:
                counts = {s: sum(row.count(s) for row in specs) for s in SCALES}
                slots = [(i, j) for i, row in enumerate(specs) for j, old in enumerate(row)
                         if old == 8 or counts[old] > 1]
                index, slot = slots[int(rng.integers(len(slots)))]
                specs[index][slot] = size
    return specs


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    ntrain, ntest = _counts(rng, num_train, num_test)
    frame, stem = (int(x) for x in rng.choice(10, 2, replace=False))
    available = [x for x in range(10) if x not in (frame, stem)]
    train_specs = _size_specs(rng, ntrain, 'source_train')
    test_specs = _size_specs(rng, ntest, 'target_test')
    puzzle = {'train': [], 'test': []}
    seen = set()
    first_roles = None
    modes = ('isolated', 'touch', 'overlap', 'occlusion', 'unclipped', 'unclipped', 'top_clip', 'side_clip', 'bottom_clip', 'core_clip', 'no_visible_stem')
    for split, specs in (('train', train_specs), ('test', test_specs)):
        for index, sizes in enumerate(specs):
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            roles = tuple(int(x) for x in pair_rng.choice(available, 2, replace=False))
            if split == 'train' and index == 0:
                first_roles = roles
            if split == 'train' and index == 1:
                bg, core = roles
                if bg == first_roles[0]:
                    bg = next(color for color in available if color not in (first_roles[0], core))
                if core == first_roles[1]:
                    core = next(color for color in available if color not in (first_roles[1], bg))
                roles = (bg, core)
            desired = 'demonstrate' if split == 'train' and index == 0 else str(pair_rng.choice(modes))
            for _ in range(16):
                pair = _make_pair(pair_rng, sizes, 'source_train' if split == 'train' else 'target_test', (*roles, frame, stem), desired)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise ValueError('could not create unique pair')
    training_features = {name for pair in puzzle['train'] for name, value in _pair_features(pair).items() if value}
    testing_features = {name for pair in puzzle['test'] for name, value in _pair_features(pair).items() if value}
    needed = tuple(sorted(({'core_clip', 'occlusion'} & testing_features) - training_features))
    if needed:
        index = len(puzzle['train']) - 1
        old = puzzle['train'][index]
        vals, counts = np.unique(old['input'], return_counts=True)
        roles = (int(vals[np.argmax(counts)]), int(vals[np.argmin(counts)]))
        for _ in range(16):
            candidate = _make_pair(pair_rng, [2, 4, 6], 'source_train', (*roles, frame, stem), needed)
            key = _key(candidate['input'])
            if key not in seen or key == _key(old['input']):
                puzzle['train'][index] = candidate
                break
        else:
            raise ValueError('could not demonstrate test interaction')
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    assert len({_key(p['train'][0]['input']) for p in puzzles}) >= 40
    features, dims, scales, counts = set(), set(), set(), set()
    coverage = {}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        profiles = {'train': [], 'test': []}
        for index, pair in enumerate(puzzle['train'] + puzzle['test']):
            split = 'train' if index < len(puzzle['train']) else 'test'
            inp = pair['input']
            vals, n = np.unique(inp, return_counts=True)
            boxes = _components(inp, int(vals[np.argmin(n)]))
            pair_features = _features(inp.shape, boxes)
            features.update(k for k, value in pair_features.items() if value)
            dims.add(inp.shape)
            counts.add(len(boxes))
            sizes = tuple(c1 - c0 + 1 for r0, c0, r1, c1 in boxes)
            scales.update(sizes)
            profiles[split].append((inp.shape, len(boxes), sizes, pair_features,
                                    (int(vals[np.argmax(n)]), int(vals[np.argmin(n)]))))
            for feature in ('isolated', 'touch', 'overlap', 'occlusion', 'unclipped', 'top_clip',
                            'side_clip', 'bottom_clip', 'core_clip', 'no_visible_stem'):
                coverage[feature] = coverage.get(feature, 0) + int(pair_features[feature])
            coverage['no_stem_output'] = coverage.get('no_stem_output', 0) + int(len(np.unique(pair['output'])) == 3)
            coverage['top_core_clip'] = coverage.get('top_core_clip', 0) + int(any(r0 == 0 and r1-r0+1 < c1-c0+1 for r0,c0,r1,c1 in boxes))
            coverage['bottom_core_clip'] = coverage.get('bottom_core_clip', 0) + int(any(r1 == inp.shape[0]-1 and r1-r0+1 < c1-c0+1 for r0,c0,r1,c1 in boxes))
            coverage['combined'] = coverage.get('combined', 0) + int(pair_features['core_clip'] and pair_features['occlusion'])
            coverage['same_scale'] = coverage.get('same_scale', 0) + int(len(set(sizes)) == 1)
            coverage['rectangular_train'] = coverage.get('rectangular_train', 0) + int(split == 'train' and inp.shape[0] != inp.shape[1])
            coverage['large_square_test'] = coverage.get('large_square_test', 0) + int(split == 'test' and inp.shape == (30, 30))
            coverage['small_test'] = coverage.get('small_test', 0) + int(split == 'test' and max(inp.shape) <= 20)
        train, test = profiles['train'], profiles['test']
        coverage['within_train_dimensions'] = coverage.get('within_train_dimensions', 0) + int(len({x[0] for x in train}) > 1)
        coverage['within_train_counts'] = coverage.get('within_train_counts', 0) + int(len({x[1] for x in train}) > 1)
        coverage['within_train_scales'] = coverage.get('within_train_scales', 0) + int(any(len(set(x[2])) == 1 for x in train) and any(len(set(x[2])) > 1 for x in train))
        coverage['within_train_simple_crowded'] = coverage.get('within_train_simple_crowded', 0) + int(any(x[3]['isolated'] for x in train) and any(x[3]['overlap'] for x in train))
        coverage['within_train_colors'] = coverage.get('within_train_colors', 0) + int(len({x[4][0] for x in train}) > 1 and len({x[4][1] for x in train}) > 1)
        coverage['unseen_eight_test'] = coverage.get('unseen_eight_test', 0) + int(any(8 in x[2] for x in test) and not any(8 in x[2] for x in train))
        coverage['smaller_test'] = coverage.get('smaller_test', 0) + int(any(a*b < max(h*w for h,w in (x[0] for x in train)) for a,b in (x[0] for x in test)))
    assert len(dims) >= 80 and counts == {1, 2, 3, 4, 5} and scales == set(SCALES)
    assert set(('isolated', 'touch', 'overlap', 'occlusion', 'unclipped', 'top_clip', 'side_clip', 'bottom_clip', 'core_clip', 'no_visible_stem')) <= features
    minimums = {'isolated': 30, 'touch': 20, 'overlap': 40, 'occlusion': 40,
                'unclipped': 25, 'top_clip': 30, 'side_clip': 30, 'bottom_clip': 30,
                'core_clip': 25, 'no_visible_stem': 30, 'no_stem_output': 5,
                'top_core_clip': 10, 'bottom_core_clip': 10, 'combined': 10,
                'same_scale': 30, 'rectangular_train': 50, 'large_square_test': 3,
                'small_test': 3, 'within_train_dimensions': 30, 'within_train_counts': 25,
                'within_train_scales': 20, 'within_train_simple_crowded': 15,
                'within_train_colors': 42, 'unseen_eight_test': 5, 'smaller_test': 20}
    assert all(coverage.get(name, 0) >= minimum for name, minimum in minimums.items()), coverage
    for i, (a, b) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(8000 + i, a, b)
        assert a is None or len(p['train']) == a
        assert b is None or len(p['test']) == b
        validate_generated_puzzle(p)
    for i, (a, b) in enumerate(((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4))):
        try:
            generate_puzzle(9000 + i, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    x, y = generate_puzzle(7001, 6, 3), generate_puzzle(7001, 6, 3)
    assert all(np.array_equal(a[k], b[k]) for split in ('train', 'test') for a, b in zip(x[split], y[split]) for k in ('input', 'output'))
    assert _key(x['train'][0]['input']) != _key(generate_puzzle(7002, 6, 3)['train'][0]['input'])
