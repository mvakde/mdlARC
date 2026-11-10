import numpy as np


_SOURCE_PAIRS = (
    (
        np.array(
            [
                [4, 4, 4, 4, 4, 7, 7],
                [4, 7, 7, 7, 4, 7, 5],
                [4, 7, 1, 7, 4, 7, 5],
                [4, 7, 7, 7, 4, 7, 7],
                [4, 4, 4, 4, 4, 7, 5],
                [7, 7, 7, 7, 7, 7, 5],
                [5, 5, 5, 5, 5, 7, 7],
            ],
            dtype=np.int8,
        ),
        np.array(
            [
                [7, 7, 7, 7, 7, 7, 4, 4, 4, 4],
                [7, 7, 5, 5, 5, 7, 4, 4, 4, 4],
                [7, 7, 5, 5, 5, 7, 4, 4, 4, 4],
                [1, 7, 5, 5, 5, 7, 4, 4, 4, 4],
            ],
            dtype=np.int8,
        ),
    ),
    (
        np.array(
            [
                [9, 9, 9, 9],
                [7, 7, 9, 7],
                [9, 9, 9, 9],
                [7, 7, 7, 7],
                [7, 6, 7, 6],
                [7, 6, 7, 6],
                [7, 7, 7, 7],
            ],
            dtype=np.int8,
        ),
        np.array(
            [
                [7, 7, 7, 9, 9, 9],
                [6, 6, 7, 9, 9, 9],
                [6, 6, 7, 9, 9, 9],
            ],
            dtype=np.int8,
        ),
    ),
    (
        np.array(
            [
                [7, 7, 7, 0, 0, 0, 7, 7, 7, 4],
                [0, 0, 7, 0, 7, 0, 7, 9, 7, 4],
                [0, 0, 7, 0, 0, 0, 7, 7, 7, 4],
                [7, 7, 7, 7, 7, 7, 7, 0, 7, 4],
                [7, 7, 7, 7, 7, 0, 0, 0, 7, 7],
            ],
            dtype=np.int8,
        ),
        np.array(
            [
                [7, 7, 7, 7, 7, 0, 0, 0, 0],
                [7, 7, 7, 7, 7, 0, 0, 0, 0],
                [7, 7, 4, 4, 7, 0, 0, 0, 0],
                [9, 7, 4, 4, 7, 0, 0, 0, 0],
            ],
            dtype=np.int8,
        ),
    ),
)



def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _construct_output(background, size_to_color):
    sides = sorted(size_to_color)
    height = sides[-1]
    width = sum(sides) + len(sides) - 1
    grid = np.full((height, width), background, dtype=np.int8)
    column = 0
    for side in sides:
        grid[height-side:, column:column+side] = size_to_color[side]
        column += side + 1
    return grid


def _analyze_input(grid):
    counts = np.bincount(grid.ravel(), minlength=10)
    leaders = np.flatnonzero(counts == counts.max())
    assert len(leaders) == 1, "background must be uniquely dominant"
    background = int(leaders[0])
    color_to_side = {}
    for color in np.flatnonzero(counts):
        color = int(color)
        if color == background:
            continue
        count = int(counts[color])
        side = int(np.sqrt(count))
        assert side * side == count, "payload count must be square"
        assert side not in color_to_side.values(), "payload sides must differ"
        color_to_side[color] = side
    assert 2 <= len(color_to_side) <= 5
    assert counts[background] > max(counts[color] for color in color_to_side)
    assert counts[background] * 5 >= grid.size * 2, "background field too weak"
    return background, color_to_side


def _expected_output(grid):
    background, color_to_side = _analyze_input(grid)
    return _construct_output(background, {side: color for color, side in color_to_side.items()})


def _array_key(grid):
    return (grid.shape, grid.tobytes())


def _is_source_pair(input_grid, output_grid):
    return any(np.array_equal(input_grid, a) and np.array_equal(output_grid, b)
               for a, b in _SOURCE_PAIRS)


def _sample_sides(rng, permitted, lower=2, upper=5, avoid=None):
    permitted = list(permitted)
    maximum = min(upper, len(permitted))
    for _ in range(24):
        n = int(rng.integers(lower, maximum + 1))
        sides = tuple(sorted(int(v) for v in rng.choice(permitted, n, replace=False)))
        if sides != avoid and sum(sides) + n - 1 <= 30:
            return sides
    for n in range(lower, maximum + 1):
        for start in range(len(permitted) - n + 1):
            sides = tuple(sorted(permitted[start:start+n]))
            if sides != avoid and sum(sides) + n - 1 <= 30:
                return sides
    raise AssertionError("no size set available")


def _choose_dimensions(rng, sides):
    payload = sum(side * side for side in sides)
    required = max(payload + max(sides)**2 + 1, (payload * 20 + 10) // 11)
    max_area = min(900, max(100, int(required * 2.25)))
    aspect = int(rng.integers(0, 3))
    candidates = []
    for h in range(5, 31):
        for w in range(5, 31):
            area = h * w
            if not required <= area <= max_area:
                continue
            if aspect == 0 and h * 4 < w * 5:
                continue
            if aspect == 1 and max(h, w) * 5 > min(h, w) * 6:
                continue
            if aspect == 2 and w * 4 < h * 5:
                continue
            candidates.append((h, w))
    if not candidates:
        candidates = [(h, w) for h in range(5, 31) for w in range(5, 31)
                      if required <= h*w <= max_area]
    return candidates[int(rng.integers(len(candidates)))]


def _place_payload(rng, shape, background, size_to_color):
    h, w = shape
    grid = np.full((h, w), background, dtype=np.int8)
    reserved = np.zeros((h, w), dtype=bool)
    sides = list(size_to_color)
    modes = ['scatter', 'patch', 'bar', 'bend', 'fragments', 'interleave']
    chosen = list(rng.choice(modes, size=len(sides), replace=True))
    if len(sides) > 1 and chosen[0] == chosen[1]:
        chosen[1] = modes[(modes.index(chosen[0]) + int(rng.integers(1, len(modes)))) % len(modes)]
    eligible = [i for i, side in enumerate(sides) if side >= 4]
    if eligible and rng.random() < 0.48:
        chosen[int(rng.choice(eligible))] = 'frame'
    order = [i for i in range(len(sides)) if chosen[i] == 'frame']
    order += [int(i) for i in rng.permutation([i for i in range(len(sides)) if i not in order])]
    rr, cc = np.indices((h, w))
    for index in order:
        side = sides[index]
        color = size_to_color[side]
        count = side * side
        mode = chosen[index]
        if mode == 'frame':
            placements = [(r, c) for r in range(h-4) for c in range(w-4)
                          if np.all((grid[r:r+5, c:c+5] == background) & ~reserved[r:r+5, c:c+5])]
            if placements:
                r, c = placements[int(rng.integers(len(placements)))]
                block = grid[r:r+5, c:c+5]
                block[0, :] = color
                block[-1, :] = color
                block[:, 0] = color
                block[:, -1] = color
                reserved[r+1:r+4, c+1:c+4] = True
                count -= 16
            else:
                mode = 'patch'
        if count == 0:
            continue
        available = (grid == background) & ~reserved
        assert np.count_nonzero(available) >= count
        r0 = int(rng.integers(h))
        c0 = int(rng.integers(w))
        noise = rng.random((h, w))
        if mode == 'patch' or mode == 'frame':
            score = (rr-r0)**2 + (cc-c0)**2 + noise * 1.5
        elif mode == 'bar':
            if rng.random() < 0.5:
                score = abs(rr-r0) * (w+1) + abs(cc-c0) + noise
            else:
                score = abs(cc-c0) * (h+1) + abs(rr-r0) + noise
        elif mode == 'bend':
            score = np.minimum(abs(rr-r0) + np.maximum(0, c0-cc)*3,
                               abs(cc-c0) + np.maximum(0, r0-rr)*3) + noise * 1.5
        elif mode == 'fragments':
            centers = [(int(rng.integers(h)), int(rng.integers(w))) for _ in range(3)]
            score = np.minimum.reduce([(rr-r)**2 + (cc-c)**2 for r, c in centers]) + noise * 5
        elif mode == 'interleave':
            score = ((rr+cc+int(rng.integers(2))) % 2) * 2 + noise * 3
        else:
            score = noise
        score = np.where(available, score, np.inf)
        selected = np.argpartition(score.ravel(), count-1)[:count]
        grid.ravel()[selected] = color
    return grid


def _new_pair(rng, background, size_to_color):
    dimensions = _choose_dimensions(rng, tuple(size_to_color))
    input_grid = _place_payload(rng, dimensions, background, size_to_color)
    return {'input': input_grid, 'output': _construct_output(background, size_to_color)}


def _assign_colors(rng, sides, background, familiar, novel, regime):
    familiar = [v for v in familiar if v != background]
    novel = [v for v in novel if v != background]
    n = len(sides)
    if regime == 'familiar' or not novel:
        source = familiar
        if len(source) < n:
            source = familiar + novel
        colors = rng.choice(source, n, replace=False)
    elif regime == 'novel' and len(novel) >= n:
        colors = rng.choice(novel, n, replace=False)
    else:
        new_count = min(len(novel), max(1, int(rng.integers(1, n))))
        old_count = n - new_count
        if old_count > len(familiar):
            old_count = len(familiar)
            new_count = n - old_count
        colors = list(rng.choice(familiar, old_count, replace=False)) + list(rng.choice(novel, new_count, replace=False))
        colors = rng.permutation(colors)
    return dict(zip(sides, (int(v) for v in colors)))


def _choose_counts(rng, num_train, num_test):
    train = _checked_count('num_train', num_train, 2, 10)
    test = _checked_count('num_test', num_test, 1, 4)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test+1), 11))
    elif train is None:
        train = int(rng.integers(max(2, test+1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train-1)+1))
    if train <= test:
        raise ValueError('num_train must exceed num_test')
    return train, test


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _choose_counts(rng, num_train, num_test)
    base_background = int(rng.integers(10))
    palette = [int(v) for v in rng.permutation([v for v in range(10) if v != base_background])]
    familiar_pool, novel_pool = palette[:6], palette[6:]
    variable_background = bool(rng.integers(2))
    train = []
    seen = set()
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max, size=train_count+test_count, dtype=np.int64)
    pair_rngs = [np.random.default_rng(int(value)) for value in pair_seeds]
    first_sides = _sample_sides(rng, range(1, 7), lower=3, upper=4)
    first_map = _assign_colors(rng, first_sides, base_background, familiar_pool, [], 'familiar')
    maps = [(base_background, first_map)]
    second_background = int(rng.choice(list(first_map.values()))) if variable_background else base_background
    second_sides = _sample_sides(rng, range(1, 7), lower=2, upper=4, avoid=first_sides)
    second_map = _assign_colors(rng, second_sides, second_background, familiar_pool, [], 'familiar')
    shared = [c for c in first_map.values() if c != second_background]
    shared_color = int(rng.choice(shared))
    old_side = next(s for s, c in first_map.items() if c == shared_color)
    new_side = int(rng.choice([s for s in second_sides if s != old_side]))
    old_color = second_map[new_side]
    if shared_color in second_map.values():
        prior_side = next(s for s, c in second_map.items() if c == shared_color)
        second_map[prior_side] = old_color
    second_map[new_side] = shared_color
    if variable_background and base_background not in second_map.values():
        replace_side = next(s for s, c in second_map.items() if c != shared_color)
        second_map[replace_side] = base_background
    maps.append((second_background, second_map))
    for i in range(2, train_count):
        if i == 2 and train_count >= 4:
            maps.append((base_background, first_map.copy()))
        else:
            background = base_background if not variable_background else int(rng.choice([base_background] + familiar_pool))
            sides = _sample_sides(rng, range(1, 7), lower=2, upper=5)
            maps.append((background, _assign_colors(rng, sides, background, familiar_pool + [base_background], [], 'familiar')))
    for i, (background, size_to_color) in enumerate(maps):
        pair_rng = pair_rngs[i]
        for _ in range(24):
            pair = _new_pair(pair_rng, background, size_to_color)
            key = _array_key(pair['input'])
            if key not in seen and not _is_source_pair(pair['input'], pair['output']):
                break
        else:
            raise AssertionError('unique training construction failed')
        seen.add(key)
        train.append(pair)
    rng.shuffle(train)
    train_sides = sorted({side for pair in train for side in _analyze_input(pair['input'])[1].values()})
    training_colors = {color for pair in train for color in _analyze_input(pair['input'])[1]}
    test = []
    test_outputs = set()
    for j in range(test_count):
        pair_rng = pair_rngs[train_count+j]
        for _ in range(48):
            background = base_background if not variable_background else int(rng.choice(list(training_colors)))
            upper = 3 if j % 3 == 1 else 5
            sides = _sample_sides(rng, train_sides, lower=2, upper=upper)
            regime = ('familiar', 'novel', 'mixed')[int(rng.integers(3))]
            if regime == 'novel' and len(sides) > len(novel_pool):
                regime = 'mixed'
            size_to_color = _assign_colors(rng, sides, background, list(training_colors), novel_pool, regime)
            pair = _new_pair(pair_rng, background, size_to_color)
            key = _array_key(pair['input'])
            out_key = _array_key(pair['output'])
            if key not in seen and out_key not in test_outputs and not _is_source_pair(pair['input'], pair['output']):
                break
        else:
            raise AssertionError('unique test construction failed')
        seen.add(key)
        test_outputs.add(out_key)
        test.append(pair)
    rng.shuffle(test)
    puzzle = {'train': train, 'test': test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    inputs, pairs = set(), set()
    analyses = {'train': [], 'test': []}
    test_outputs = set()
    for section in ('train', 'test'):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for name in ('input', 'output'):
                grid = pair[name]
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = pair['input'], pair['output']
            ikey, okey = _array_key(inp), _array_key(out)
            assert ikey not in inputs and (ikey, okey) not in pairs
            inputs.add(ikey)
            pairs.add((ikey, okey))
            assert not np.array_equal(inp, out)
            assert not _is_source_pair(inp, out)
            analysis = _analyze_input(inp)
            analyses[section].append(analysis)
            assert np.array_equal(out, _expected_output(inp))
            if section == 'test':
                assert okey not in test_outputs, 'test outputs should differ'
                test_outputs.add(okey)
    train_sides = {side for _, mapping in analyses['train'] for side in mapping.values()}
    train_backgrounds = {background for background, _ in analyses['train']}
    train_payload_colors = {color for _, mapping in analyses['train'] for color in mapping}
    for background, mapping in analyses['test']:
        assert set(mapping.values()) <= train_sides, 'test size not demonstrated'
        assert background in train_backgrounds or len(train_backgrounds) >= 2
        if background in train_payload_colors and background not in train_backgrounds:
            assert train_backgrounds & train_payload_colors, 'background role change absent from training'
    train_color_sides = {}
    for _, mapping in analyses['train']:
        for color, side in mapping.items():
            train_color_sides.setdefault(color, set()).add(side)
    assert any(len(sides) >= 2 for sides in train_color_sides.values()), 'no color reassignment demonstrated'


def _puzzles_equal(a, b):
    return all(len(a[section]) == len(b[section]) and
               all(np.array_equal(x[k], y[k]) for x, y in zip(a[section], b[section]) for k in ('input', 'output'))
               for section in ('train', 'test'))


def self_test() -> None:
    def predict_from_counts(train_pairs, test_input):
        def predict(grid):
            counts = {int(color): int(np.count_nonzero(grid == color)) for color in np.unique(grid)}
            background = max(counts, key=counts.get)
            sides = []
            for color, count in counts.items():
                if color != background:
                    side = int(np.sqrt(count))
                    assert side * side == count
                    sides.append((side, color))
            sides.sort()
            assert len({side for side, _ in sides}) == len(sides)
            height = max(side for side, _ in sides)
            width = sum(side for side, _ in sides) + len(sides) - 1
            predicted = np.full((height, width), background, dtype=np.int8)
            column = 0
            for side, color in sides:
                for row in range(height-side, height):
                    for x in range(column, column+side):
                        predicted[row, x] = color
                column += side+1
            return predicted
        for pair in train_pairs:
            assert np.array_equal(predict(pair['input']), pair['output'])
        return predict(test_input)

    assert _puzzles_equal(generate_puzzle(90126, 6, 3), generate_puzzle(90126, 6, 3))
    assert not _puzzles_equal(generate_puzzle(90126), generate_puzzle(90127))
    for i, (train, test) in enumerate(((2,1), (10,1), (10,4), (None,1), (None,4), (2,None), (10,None))):
        puzzle = generate_puzzle(91000+i, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle['train']) == train
        if test is not None:
            assert len(puzzle['test']) == test
    for i, kwargs in enumerate(({'num_train':1}, {'num_train':11}, {'num_test':0}, {'num_test':5},
                                {'num_train':2,'num_test':2}, {'num_train':3,'num_test':4})):
        try:
            generate_puzzle(92000+i, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f'illegal counts accepted: {kwargs}')
    train_counts, test_counts, signatures = set(), set(), set()
    shapes, outputs, side_sets, backgrounds = set(), set(), set(), set()
    aspect = set()
    variable_backgrounds = 0
    no_singleton_tests = 0
    large_side_tests = 0
    frames = 0
    test_palette_cases = set()
    repeated_train_outputs = 0
    combinations = set()
    train_aspects, test_aspects = set(), set()
    payload_counts = {'train': set(), 'test': set()}
    total_pairs = 0
    for seed in range(1000, 1042):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        signatures.add(tuple((section, tuple((_array_key(p['input']), _array_key(p['output'])) for p in puzzle[section])) for section in ('train','test')))
        pair_backgrounds = set()
        train_payload_colors = {color for pair in puzzle['train'] for color in _analyze_input(pair['input'])[1]}
        repeated_train_outputs += len({_array_key(pair['output']) for pair in puzzle['train']}) < len(puzzle['train'])
        for section in ('train', 'test'):
            for pair in puzzle[section]:
                total_pairs += 1
                inp, out = pair['input'], pair['output']
                bg, mapping = _analyze_input(inp)
                assert np.array_equal(out, _construct_output(bg, {s:c for c,s in mapping.items()}))
                pair_backgrounds.add(bg)
                shapes.add(inp.shape)
                outputs.add(out.shape)
                side_sets.add(tuple(sorted(mapping.values())))
                backgrounds.add(bg)
                h, w = inp.shape
                shape_class = 'tall' if h > w*1.2 else 'wide' if w > h*1.2 else 'square'
                aspect.add(shape_class)
                (train_aspects if section == 'train' else test_aspects).add(shape_class)
                payload_counts[section].add(len(mapping))
                frame = False
                for color, side in mapping.items():
                    if side < 4:
                        continue
                    for row in range(h-4):
                        for col in range(w-4):
                            block = inp[row:row+5, col:col+5]
                            if (np.all(block[0, :] == color) and np.all(block[-1, :] == color)
                                    and np.all(block[:, 0] == color) and np.all(block[:, -1] == color)
                                    and np.all(block[1:4, 1:4] == bg)):
                                frame = True
                                break
                        if frame:
                            break
                    if frame:
                        break
                frames += frame
                if section == 'test':
                    no_singleton_tests += (1 not in mapping.values())
                    large_side_tests += (max(mapping.values()) > 4)
                    familiar = sum(color in train_payload_colors for color in mapping)
                    palette_case = 'familiar' if familiar == len(mapping) else 'novel' if familiar == 0 else 'mixed'
                    test_palette_cases.add(palette_case)
                    combinations.add((shape_class, len(mapping), max(mapping.values()), palette_case, frame))
                    assert np.array_equal(predict_from_counts(puzzle['train'], inp), out)
        variable_backgrounds += len(pair_backgrounds) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) == 42
    assert len(shapes) >= 30 and len(outputs) >= 15 and len(side_sets) >= 18
    assert len(backgrounds) >= 8 and aspect == {'tall', 'wide', 'square'}
    assert variable_backgrounds >= 10 and no_singleton_tests >= 10 and large_side_tests >= 10
    assert train_aspects == test_aspects == {'tall', 'wide', 'square'}
    assert payload_counts['train'] == payload_counts['test'] == {2, 3, 4, 5}
    assert frames >= 30 and repeated_train_outputs >= 20
    assert test_palette_cases == {'familiar', 'novel', 'mixed'} and len(combinations) >= 25
    assert total_pairs >= 200
