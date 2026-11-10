import numpy as np


_SOURCE_INPUT_TEXTS = {
    (
        "7777777777/7577777577/7277777277/7277777277/7077777277/"
        "7277777277/7277777077/7277777277/7277777777/7777777777"
    ),
    (
        "77777777/77777777/77777777/75220227/"
        "77777777/77777777/77777777/77777777"
    ),
    (
        "777777777777/777777777777/720257777777/777777777777/"
        "777777777777/777777777777/777777777777/777777777777/"
        "777777777777/752202277777/777777777777/772222202257"
    ),
    (
        "7777777777777777/7777777777777777/7777777777777577/"
        "7777777777777277/7277777777777277/7277522022777077/"
        "7277777777777277/7077777777777277/7277777777777777/"
        "7277277777777777/7277077777777777/7277272777502777/"
        "7277272777777777/7277270777777777/7277272777777777/"
        "7577575777777205"
    ),
}

_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid_text(grid):
    return "/".join(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _array_key(grid):
    return (
        grid.shape,
        tuple(int(value) for value in grid.ravel()),
    )


def _role_tuple(parameters):
    return (
        parameters["background"],
        parameters["rod"],
        parameters["anchor"],
        parameters["hinge"],
    )


def _parse_and_transform(inp, background, rod, anchor, hinge):
    if not isinstance(inp, np.ndarray) or inp.ndim != 2:
        raise ValueError("grid is not a two-dimensional NumPy array")

    height, width = inp.shape

    def inside(row, column):
        return 0 <= row < height and 0 <= column < width

    anchor_positions = [
        tuple(int(value) for value in position)
        for position in np.argwhere(inp == anchor)
    ]
    if not anchor_positions:
        raise ValueError("input contains no anchors")

    records = []
    claimed = set()

    for anchor_row, anchor_column in anchor_positions:
        candidates = []

        for delta_row, delta_column in _DIRECTIONS:
            row = anchor_row + delta_row
            column = anchor_column + delta_column

            stem = []
            while (
                inside(row, column)
                and int(inp[row, column]) == rod
            ):
                stem.append((row, column))
                row += delta_row
                column += delta_column

            if (
                not inside(row, column)
                or int(inp[row, column]) != hinge
            ):
                continue

            hinge_position = (row, column)
            row += delta_row
            column += delta_column

            tail = []
            while (
                inside(row, column)
                and int(inp[row, column]) == rod
            ):
                tail.append((row, column))
                row += delta_row
                column += delta_column

            if not tail:
                continue

            if (
                inside(row, column)
                and int(inp[row, column]) != background
            ):
                continue

            cells = {
                (anchor_row, anchor_column),
                hinge_position,
                *stem,
                *tail,
            }

            candidates.append(
                {
                    "anchor": (anchor_row, anchor_column),
                    "direction": (delta_row, delta_column),
                    "stem": tuple(stem),
                    "hinge": hinge_position,
                    "tail": tuple(tail),
                    "cells": cells,
                    "orientation": (
                        "V" if delta_row != 0 else "H"
                    ),
                }
            )

        if len(candidates) != 1:
            raise ValueError(
                "an anchor does not define exactly one valid rod"
            )

        record = candidates[0]
        if claimed.intersection(record["cells"]):
            raise ValueError("rods overlap")

        claimed.update(record["cells"])
        records.append(record)

    non_background = {
        tuple(int(value) for value in position)
        for position in np.argwhere(inp != background)
    }
    if claimed != non_background:
        raise ValueError(
            "non-background cells are not exactly disjoint rods"
        )

    result = inp.copy()

    for record in records:
        for row, column in record["tail"]:
            result[row, column] = background

    destinations = set()

    for record in records:
        hinge_row, hinge_column = record["hinge"]

        if record["orientation"] == "H":
            relation = 2 * hinge_row - (height - 1)
            if relation == 0:
                raise ValueError(
                    "horizontal hinge lies on the central row"
                )
            turn = (-1, 0) if relation > 0 else (1, 0)
        else:
            relation = 2 * hinge_column - (width - 1)
            if relation == 0:
                raise ValueError(
                    "vertical hinge lies on the central column"
                )
            turn = (0, -1) if relation > 0 else (0, 1)

        new_tail = []
        for distance in range(1, len(record["tail"]) + 1):
            position = (
                hinge_row + turn[0] * distance,
                hinge_column + turn[1] * distance,
            )

            if (
                not inside(*position)
                or int(inp[position]) != background
            ):
                raise ValueError(
                    "a rotated tail is blocked or leaves the grid"
                )

            if position in destinations:
                raise ValueError("rotated tails collide")

            destinations.add(position)
            new_tail.append(position)

        record["turn"] = turn
        record["new_tail"] = tuple(new_tail)

    for row, column in destinations:
        result[row, column] = rod

    return result, records


def _pair_roles(inp, anchor, hinge):
    colors, counts = np.unique(inp, return_counts=True)
    colors = [int(c) for c in colors]
    if len(colors) != 4 or anchor not in colors or hinge not in colors:
        raise ValueError('four distinct roles are required')
    top = int(np.argmax(counts))
    if int(counts[top]) <= max(int(v) for i, v in enumerate(counts) if i != top):
        raise ValueError('background is not uniquely modal')
    background = colors[top]
    if background in (anchor, hinge):
        raise ValueError('marker cannot be background')
    rod = next(c for c in colors if c not in (background, anchor, hinge))
    return background, rod, anchor, hinge


def _valid_conventions(pairs):
    if len(pairs) < 2:
        return []
    shared = set(int(c) for c in np.unique(pairs[0]['input']))
    shared &= set(int(c) for c in np.unique(pairs[1]['input']))
    if len(shared) != 2:
        return []
    a, b = sorted(shared)
    valid = []
    for anchor, hinge in ((a, b), (b, a)):
        analyses = []
        try:
            for pair in pairs:
                roles = _pair_roles(pair['input'], anchor, hinge)
                expected, records = _parse_and_transform(pair['input'], *roles)
                if not np.array_equal(expected, pair['output']):
                    raise ValueError('training output differs from rule')
                analyses.append((roles, records))
        except (ValueError, IndexError):
            continue
        valid.append(((anchor, hinge), analyses))
    return valid


def _length(rng, maximum, allow_zero):
    if maximum < 1:
        return 0 if allow_zero else None
    if allow_zero and rng.random() < 0.24:
        return 0
    bands = ((1, min(3, maximum)), (4, min(8, maximum)), (9, maximum))
    weights = (0.48, 0.36, 0.16)
    choice = int(rng.choice(3, p=weights))
    if bands[choice][0] > bands[choice][1]:
        choice = next(i for i in (1, 0, 2) if bands[i][0] <= bands[i][1])
    low, high = bands[choice]
    return int(rng.integers(low, high + 1))


def _random_candidate(rng, height, width, orientation, turn_sign,
                      minimum_stem=0, minimum_tail=1):
    if orientation == 'H':
        turn_positions = [r for r in range(height)
                          if (1 if 2*r < height-1 else -1 if 2*r > height-1 else 0) == turn_sign]
        if not turn_positions or width < 3:
            return None
        side = int(rng.choice(turn_positions))
        along = int(rng.integers(1, width-1))
        axis = 1 if rng.random() < 0.5 else -1
        max_stem = along-1 if axis == 1 else width-along-2
        max_tail = width-along-1 if axis == 1 else along
        max_turn = height-side-1 if turn_sign == 1 else side
        anchor_dir = (0, axis)
        turn_dir = (turn_sign, 0)
        hinge = (side, along)
    else:
        turn_positions = [c for c in range(width)
                          if (1 if 2*c < width-1 else -1 if 2*c > width-1 else 0) == turn_sign]
        if not turn_positions or height < 3:
            return None
        side = int(rng.choice(turn_positions))
        along = int(rng.integers(1, height-1))
        axis = 1 if rng.random() < 0.5 else -1
        max_stem = along-1 if axis == 1 else height-along-2
        max_tail = height-along-1 if axis == 1 else along
        max_turn = width-side-1 if turn_sign == 1 else side
        anchor_dir = (axis, 0)
        turn_dir = (0, turn_sign)
        hinge = (along, side)
    maximum_tail = min(max_tail, max_turn)
    if maximum_tail < minimum_tail or max_stem < minimum_stem:
        return None
    stem_len = _length(rng, max_stem, True)
    tail_len = _length(rng, maximum_tail, False)
    stem_len = max(stem_len, minimum_stem)
    tail_len = max(tail_len, minimum_tail)
    if max_stem > 0 and rng.random() < 0.12:
        stem_len = max_stem
    if rng.random() < 0.10:
        tail_len = maximum_tail
    anchor = (hinge[0]-anchor_dir[0]*(stem_len+1),
              hinge[1]-anchor_dir[1]*(stem_len+1))
    stem = tuple((anchor[0]+anchor_dir[0]*d, anchor[1]+anchor_dir[1]*d)
                 for d in range(1, stem_len+1))
    tail = tuple((hinge[0]+anchor_dir[0]*d, hinge[1]+anchor_dir[1]*d)
                 for d in range(1, tail_len+1))
    new_tail = tuple((hinge[0]+turn_dir[0]*d, hinge[1]+turn_dir[1]*d)
                     for d in range(1, tail_len+1))
    cells = {anchor, hinge, *stem, *tail}
    return {'anchor': anchor, 'stem': stem, 'hinge': hinge, 'tail': tail,
            'new_tail': new_tail, 'input_cells': cells}


def _place_rods(rng, height, width, specifications):
    for _ in range(16):
        rods = []
        occupied = set()
        near = set()
        destinations = set()
        for index in rng.permutation(len(specifications)):
            specification = specifications[int(index)]
            orientation, turn_sign = specification[:2]
            minimum_stem, minimum_tail = (
                specification[2:] if len(specification) == 4 else (0, 1)
            )
            chosen = None
            for _ in range(240):
                item = _random_candidate(rng, height, width, orientation, turn_sign,
                                         minimum_stem, minimum_tail)
                if item is None:
                    continue
                cells = item['input_cells']
                new = set(item['new_tail'])
                if cells & near or cells & destinations or new & occupied or new & destinations:
                    continue
                chosen = item
                break
            if chosen is None:
                break
            rods.append(chosen)
            occupied.update(chosen['input_cells'])
            destinations.update(chosen['new_tail'])
            for r, c in chosen['input_cells']:
                near.update(((r,c), (r-1,c), (r+1,c), (r,c-1), (r,c+1)))
        if len(rods) == len(specifications):
            return rods
    return None


def _specifications(rng, count, required=(), orientations=None):
    specs = list(required)
    while len(specs) < count:
        orientation = str(rng.choice(orientations or ('H','V')))
        sign = 1 if rng.random() < 0.5 else -1
        specs.append((orientation, sign))
    return specs


def _sample_shape(rng, count, regime, style):
    floor = max(6, 5+count)
    if regime == 'source_train' and style == 'source':
        ceiling = 21
    elif regime == 'target_test' and style == 'source':
        floor = max(floor, 13)
        ceiling = 30
    elif style == 'simple':
        ceiling = 26
    else:
        ceiling = 30
    floor = min(floor, ceiling)
    base = int(rng.integers(floor, ceiling+1))
    if rng.random() < 0.43:
        other = int(np.clip(base + int(rng.integers(-7, 8)), floor, 30))
    else:
        other = base
    if rng.random() < 0.5:
        return base, other
    return other, base


def _generate_pair(rng, regime, markers, colors, specifications, style):
    count = len(specifications)
    height, width = _sample_shape(rng, count, regime, style)
    for step in range(9):
        rods = _place_rods(rng, height, width, specifications)
        if rods is not None:
            background, rod_color = colors
            anchor_color, hinge_color = markers
            inp = np.full((height, width), background, dtype=np.int8)
            for item in rods:
                inp[item['anchor']] = anchor_color
                inp[item['hinge']] = hinge_color
                for cell in (*item['stem'], *item['tail']):
                    inp[cell] = rod_color
            try:
                roles = _pair_roles(inp, *markers)
                out, parsed = _parse_and_transform(inp, *roles)
                if len(parsed) == count and not np.array_equal(inp, out):
                    return inp, out
            except ValueError:
                pass
        if step % 2 == 0:
            height = min(30, height+2)
        else:
            width = min(30, width+2)
    raise ValueError('could not place a valid set of rods')


def _validated_optional_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer or None')
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f'{name} is outside its legal range')
    return value


def _resolve_counts(rng, num_train, num_test):
    train = _validated_optional_count(num_train, 'num_train', 2, 10)
    test = _validated_optional_count(num_test, 'num_test', 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError('num_train must exceed num_test')
    elif train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test+1), 11))
    elif train is None:
        train = int(rng.integers(max(2, test+1), 11))
    else:
        test = int(rng.integers(1, min(4, train-1)+1))
    return train, test


def _plan_specs(rng, train_count, test_count, style):
    if style == 'mixed':
        train = [[('H',1,1,2), ('V',-1)], [('H',-1), ('V',1)]]
    else:
        train = [[('H',1,1,2), ('H',-1)], [('V',1), ('V',-1)]]
    for _ in range(2, train_count):
        count = int(rng.integers(1, 6))
        orientation = None
        if rng.random() < 0.52:
            orientation = (str(rng.choice(('H','V'))),)
        train.append(_specifications(rng, count, orientations=orientation))
    tests = []
    for _ in range(test_count):
        if style == 'source':
            count = int(rng.integers(4, 8))
            required = [('H', int(rng.choice((-1,1)))), ('V', int(rng.choice((-1,1))))]
            orientation = None
        elif style == 'simple':
            count = int(rng.integers(1, 5))
            required = []
            orientation = (str(rng.choice(('H','V'))),) if rng.random() < 0.8 else None
        else:
            count = int(rng.integers(2, 8))
            required = []
            orientation = (str(rng.choice(('H','V'))),) if rng.random() < 0.3 else None
        tests.append(_specifications(rng, count, required, orientation))
    all_plans = train + tests
    if len({len(plan) for plan in all_plans}) == 1:
        replacement_count = 1 if len(tests[0]) != 1 else 3
        tests[0] = _specifications(rng, replacement_count)
    if all(len({spec[0] for spec in plan}) == 2 for plan in train + tests):
        tests[0] = _specifications(
            rng, len(tests[0]), orientations=(str(rng.choice(('H','V'))),)
        )
    return train, tests


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    markers = tuple(int(v) for v in rng.choice(10, size=2, replace=False))
    remaining = [v for v in range(10) if v not in markers]
    remaining = [int(v) for v in rng.permutation(remaining)]
    style = str(rng.choice(('source','mixed','simple')))
    train_specs, test_specs = _plan_specs(rng, train_count, test_count, style)
    puzzle = {'train': [], 'test': []}
    seen = set()
    specs_by_split = (('train', train_specs), ('test', test_specs))
    index = 0
    for split, plans in specs_by_split:
        for specs in plans:
            # The first two demonstrations have disjoint body/background colors.
            # Every pair keeps the same two marker colors.
            offset = 2*(index % 4)
            colors = (remaining[offset], remaining[offset+1])
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63-1)))
            for _ in range(12):
                inp, out = _generate_pair(pair_rng,
                                          'source_train' if split == 'train' else 'target_test',
                                          markers, colors, specs, style)
                key = _array_key(inp)
                if key not in seen and _grid_text(inp) not in _SOURCE_INPUT_TEXTS:
                    break
            else:
                raise ValueError('could not produce a unique pair')
            seen.add(key)
            puzzle[split].append({'input': inp, 'output': out})
            index += 1
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair_deltas(inp, out, roles, records):
    background, body, anchor, hinge = roles
    removed = set()
    added = set()
    for record in records:
        assert inp[record['anchor']] == out[record['anchor']] == anchor
        assert inp[record['hinge']] == out[record['hinge']] == hinge
        assert all(inp[cell] == out[cell] == body for cell in record['stem'])
        assert len(record['tail']) == len(record['new_tail']) >= 1
        pivot = record['hinge']
        if record['orientation'] == 'H':
            relation = 2*pivot[0] - (inp.shape[0]-1)
            direction = (-1 if relation > 0 else 1, 0)
        else:
            relation = 2*pivot[1] - (inp.shape[1]-1)
            direction = (0, -1 if relation > 0 else 1)
        assert relation != 0
        for distance, cell in enumerate(record['new_tail'], 1):
            assert cell == (pivot[0]+direction[0]*distance,
                            pivot[1]+direction[1]*distance)
            added.add(cell)
        removed.update(record['tail'])
    changed = {tuple(int(v) for v in cell) for cell in np.argwhere(inp != out)}
    assert changed == removed | added
    assert not removed & added
    assert all(inp[cell] == body and out[cell] == background for cell in removed)
    assert all(inp[cell] == background and out[cell] == body for cell in added)


def _validate_generated_puzzle_impl(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train','test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4 and len(puzzle['train']) > len(puzzle['test'])
    all_pairs = puzzle['train'] + puzzle['test']
    seen_inputs, seen_pairs = set(), set()
    for pair in all_pairs:
        assert isinstance(pair, dict) and set(pair) == {'input','output'}
        for grid in (pair['input'], pair['output']):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert all(1 <= n <= 30 for n in grid.shape)
            assert np.issubdtype(grid.dtype, np.integer)
            assert np.all((grid >= 0) & (grid <= 9))
            assert len(np.unique(grid)) == 4
        assert pair['input'].shape == pair['output'].shape
        assert not np.array_equal(pair['input'], pair['output'])
        key = _array_key(pair['input'])
        pairkey = (key, _array_key(pair['output']))
        assert key not in seen_inputs and pairkey not in seen_pairs
        seen_inputs.add(key)
        seen_pairs.add(pairkey)
        assert _grid_text(pair['input']) not in _SOURCE_INPUT_TEXTS
    conventions = _valid_conventions(puzzle['train'])
    assert len(conventions) == 1
    markers, train_analyses = conventions[0]
    first_colors = set(int(c) for c in np.unique(puzzle['train'][0]['input']))
    second_colors = set(int(c) for c in np.unique(puzzle['train'][1]['input']))
    assert first_colors & second_colors == set(markers)
    observed = set()
    long_fixed = False
    for roles, records in train_analyses:
        for record in records:
            observed.add((record['orientation'],
                          record['turn'][0] if record['orientation'] == 'H' else record['turn'][1]))
            long_fixed |= bool(record['stem']) and len(record['tail']) >= 2
    assert observed == {('H',-1),('H',1),('V',-1),('V',1)}
    assert long_fixed
    for pair in all_pairs:
        roles = _pair_roles(pair['input'], *markers)
        expected, records = _parse_and_transform(pair['input'], *roles)
        assert np.array_equal(expected, pair['output'])
        _check_pair_deltas(pair['input'], pair['output'], roles, records)
        assert len(records) >= 1
        assert set(int(v) for v in np.unique(pair['output'])) == set(roles)
        assert np.count_nonzero(pair['output'] == roles[0]) > max(
            np.count_nonzero(pair['output'] == color) for color in roles[1:])
        for record in records:
            assert len(record['tail']) >= 1
            assert len(record['tail']) == len(record['new_tail'])
            assert record['anchor'] not in record['new_tail']
    # All four direction cases and marker meanings are visible in training.
    # Body and background are identified separately from each pair's input.


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _validate_generated_puzzle_impl(puzzle)
    except AssertionError:
        raise
    except Exception as error:
        raise AssertionError('invalid generated puzzle') from error


def _puzzles_equal(first, second):
    if set(first) != set(second):
        return False
    for split in ('train','test'):
        if len(first[split]) != len(second[split]):
            return False
        for a, b in zip(first[split], second[split]):
            if not np.array_equal(a['input'], b['input']) or not np.array_equal(a['output'], b['output']):
                return False
    return True


def self_test() -> None:
    automatic = [generate_puzzle(7000+i) for i in range(42)]
    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
    assert _puzzles_equal(generate_puzzle(123456,6,3), generate_puzzle(123456,6,3))
    assert _puzzles_equal(generate_puzzle(98765), generate_puzzle(98765))
    assert not _puzzles_equal(automatic[0], automatic[1])
    for train, test in ((2,1),(10,1),(10,4)):
        puzzle = generate_puzzle(8100+train+test, train, test)
        assert (len(puzzle['train']),len(puzzle['test'])) == (train,test)
    for seed, train, test in ((8201,None,1),(8202,None,4),(8203,2,None),(8204,10,None)):
        puzzle = generate_puzzle(seed, train, test)
        assert (train is None or len(puzzle['train']) == train)
        assert (test is None or len(puzzle['test']) == test)
    for arguments in ({'num_train':1},{'num_train':11},{'num_test':0},{'num_test':5},
                      {'num_train':2,'num_test':2},{'num_train':3,'num_test':4},
                      {'num_train':True,'num_test':1}):
        try:
            generate_puzzle(9000, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    assert len({len(p['train']) for p in automatic}) >= 4
    assert len({len(p['test']) for p in automatic}) >= 3
    shapes, palettes, signatures, inputs = set(), set(), set(), set()
    stems, tails, counts, orientations = set(), set(), set(), set()
    edges = rectangular = mixed_train = single_train = 0
    sparse_large = compact_multi = 0
    pair_color_variation = 0
    review_counts = {
        'zero_multi': 0, 'zero_single': 0, 'long_short': 0,
        'short_long': 0, 'cross_center': 0, 'parallel_one_side': 0,
        'one_orientation_test': 0, 'test_fewer': 0, 'test_not_larger': 0,
        'full_size': 0, 'middle_size': 0, 'upper_size': 0,
        'diagonal_near': 0, 'combined': 0,
    }
    train_shapes, test_shapes = set(), set()
    for puzzle in automatic:
        markers, _ = _valid_conventions(puzzle['train'])[0]
        palettes.add(markers)
        backgrounds = set()
        bodies = set()
        pair_features = []
        for split in ('train','test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                roles = _pair_roles(inp, *markers)
                backgrounds.add(roles[0]); bodies.add(roles[1])
                expected, records = _parse_and_transform(inp, *roles)
                assert np.array_equal(expected, out)
                shapes.add(inp.shape)
                (train_shapes if split == 'train' else test_shapes).add(inp.shape)
                rectangular += inp.shape[0] != inp.shape[1]
                review_counts['full_size'] += 30 in inp.shape
                review_counts['middle_size'] += any(13 <= v <= 15 for v in inp.shape)
                review_counts['upper_size'] += any(25 <= v <= 30 for v in inp.shape)
                inputs.add(_array_key(inp))
                counts.add(len(records))
                orientation_set = {r['orientation'] for r in records}
                orientations.add(tuple(sorted(orientation_set)))
                stems.update(len(r['stem']) for r in records)
                tails.update(len(r['tail']) for r in records)
                pair_features.append((split, inp.shape, len(records), orientation_set))
                if min(inp.shape) >= 23 and len(records) <= 2: sparse_large += 1
                if max(inp.shape) <= 16 and len(records) >= 4: compact_multi += 1
                if split == 'train':
                    single_train += len(records) == 1
                    mixed_train += len(orientation_set) == 2
                else:
                    review_counts['one_orientation_test'] += len(orientation_set) == 1
                edge_cells = np.concatenate((inp[0],inp[-1],inp[:,0],inp[:,-1]))
                edge = bool(np.any(edge_cells != roles[0]))
                edges += edge
                zero = any(len(r['stem']) == 0 for r in records)
                review_counts['zero_multi'] += zero and len(records) > 1
                review_counts['zero_single'] += zero and len(records) == 1
                review_counts['combined'] += zero and edge and inp.shape[0] != inp.shape[1]
                turns = {(r['orientation'], r['turn'][0] if r['orientation'] == 'H'
                          else r['turn'][1]) for r in records}
                review_counts['parallel_one_side'] += len(records) >= 2 and len(orientation_set) == 1 and len(turns) == 1
                for record in records:
                    sl, tl = len(record['stem']), len(record['tail'])
                    review_counts['long_short'] += sl >= 5 and tl <= 2
                    review_counts['short_long'] += sl <= 1 and tl >= 5
                    center = (inp.shape[1]-1)/2 if record['orientation'] == 'H' else (inp.shape[0]-1)/2
                    axis = 1 if record['orientation'] == 'H' else 0
                    review_counts['cross_center'] += ((record['hinge'][axis]-center)
                                                      * (record['tail'][-1][axis]-center) < 0)
                review_counts['diagonal_near'] += any(
                    abs(a[0]-b[0]) == 1 and abs(a[1]-b[1]) == 1
                    for i, first in enumerate(records)
                    for j, second in enumerate(records) if i != j
                    for a in first['tail'] for b in second['new_tail']
                )
                signatures.add((inp.shape,len(records),tuple(sorted(
                    (r['orientation'],r['anchor'],len(r['stem']),len(r['tail'])) for r in records))))
        pair_color_variation += len(backgrounds) >= 2 and len(bodies) >= 2
        assert len({feature[2] for feature in pair_features}) >= 2
        assert len({tuple(sorted(feature[3])) for feature in pair_features}) >= 2
        training = [feature for feature in pair_features if feature[0] == 'train']
        testing = [feature for feature in pair_features if feature[0] == 'test']
        review_counts['test_fewer'] += any(feature[2] < max(row[2] for row in training)
                                           for feature in testing)
        review_counts['test_not_larger'] += any(min(feature[1]) <= max(max(row[1]) for row in training)
                                                for feature in testing)
    assert len(shapes) >= 30 and len(signatures) >= 100
    assert len(inputs) >= 0.98*sum(len(p['train'])+len(p['test']) for p in automatic)
    assert len(palettes) >= 25
    assert {0,1,4,7}.issubset(stems) and {1,4,5}.issubset(tails)
    assert 1 in counts and max(counts) >= 6
    assert ('H','V') in orientations and ('H',) in orientations and ('V',) in orientations
    assert min(edges,rectangular,mixed_train,single_train,sparse_large,compact_multi) > 0
    assert pair_color_variation == len(automatic)
    assert len(train_shapes & test_shapes) >= 5
    assert all(value > 0 for value in review_counts.values())
    source_test_text = next(value for value in _SOURCE_INPUT_TEXTS if len(value.split('/')) == 16)
    source_test = np.array([[int(c) for c in row] for row in source_test_text.split('/')], dtype=np.int8)
    source_expected_text = (
        '7777777777777777/7777777777777777/7777777777777577/'
        '7777777777777277/7777777777777277/7777522077722077/'
        '7777777277777777/7022277277777777/7277777777777777/'
        '7277777777777777/7277027777727777/7277277777507777/'
        '7277277777777777/7277270227777777/7277272777777727/'
        '7577575777777705'
    )
    expected = np.array([[int(c) for c in row] for row in source_expected_text.split('/')], dtype=np.int8)
    actual, source_records = _parse_and_transform(source_test, 7, 2, 5, 0)
    assert np.array_equal(actual, expected)
    assert sum(len(record['stem']) == 0 for record in source_records) == 2
    assert max(len(record['stem']) for record in source_records) == 7
    source_training_outputs = {
        10: ('7777777777/7577777577/7277777277/7277777277/'
             '7022227277/7777777277/7777772077/7777777777/'
             '7777777777/7777777777'),
        8: ('77777777/77777777/77777777/75220777/'
            '77772777/77772777/77777777/77777777'),
        12: ('777777777777/777777777777/770257777777/772777777777/'
             '777777777777/777777777777/777777727777/777727727777/'
             '777727727777/752207727777/777777727777/777777702257'),
    }
    for height, output_text in source_training_outputs.items():
        input_text = next(value for value in _SOURCE_INPUT_TEXTS
                          if len(value.split('/')) == height)
        inp = np.array([[int(c) for c in row] for row in input_text.split('/')], dtype=np.int8)
        expected = np.array([[int(c) for c in row] for row in output_text.split('/')], dtype=np.int8)
        actual, records = _parse_and_transform(inp, 7, 2, 5, 0)
        assert np.array_equal(actual, expected)
        if height == 12:
            assert max(len(record['tail']) for record in records) == 5
