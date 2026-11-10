import numpy as np

_SIDES = ("down", "left", "up", "right")


def _normalize_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    return int(seed) % (1 << 64)


def _count_value(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed_key, num_train, num_test):
    train_count = None if num_train is None else _count_value(num_train, "num_train")
    test_count = None if num_test is None else _count_value(num_test, "num_test")
    if train_count is not None and not 2 <= train_count <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if test_count is not None and not 1 <= test_count <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count
    if train_count is None and test_count is None:
        test_count = 1 + seed_key % 4
        train_count = 2 + (seed_key // 4) % 9
        if train_count <= test_count:
            train_count = test_count + 1 + (seed_key // 36) % (10 - test_count)
        return int(train_count), int(test_count)
    if train_count is None:
        train_count = test_count + 1 + seed_key % (10 - test_count)
        return int(train_count), int(test_count)
    test_count = 1 + seed_key % min(4, train_count - 1)
    return int(train_count), int(test_count)


def _is_horizontal(side):
    return side in ("down", "up")


def _is_reference_positive(side):
    return side in ("down", "right")


def _structure_mask(shape, side, barrier):
    height, width = shape
    mask = np.zeros(shape, dtype=bool)
    if side == "down":
        mask[barrier, :] = True
        mask[barrier:, 0] = True
        mask[barrier:, width - 1] = True
    elif side == "up":
        mask[barrier, :] = True
        mask[:barrier + 1, 0] = True
        mask[:barrier + 1, width - 1] = True
    elif side == "right":
        mask[:, barrier] = True
        mask[0, barrier:] = True
        mask[height - 1, barrier:] = True
    elif side == "left":
        mask[:, barrier] = True
        mask[0, :barrier + 1] = True
        mask[height - 1, :barrier + 1] = True
    else:
        raise AssertionError("unknown enclosure orientation")
    return mask


def _axis_facts(side, axis_len, barrier):
    if _is_reference_positive(side):
        return dict(ref_axis=barrier + 1, near_axis=barrier - 1,
                    far_axis=0, ref_axes=tuple(range(barrier + 1, axis_len)),
                    move_axes=tuple(range(barrier)))
    return dict(ref_axis=barrier - 1, near_axis=barrier + 1,
                far_axis=axis_len - 1, ref_axes=tuple(range(barrier)),
                move_axes=tuple(range(barrier + 1, axis_len)))


def _sample_lanes(rng, cross_len, regime):
    interior = cross_len - 2
    active_count = int(rng.integers(4, min(11, interior) + 1))
    # Choose span and location separately. Endpoints are occupied; missing
    # interior cells are jointly empty lanes, with no special fixed position.
    style = rng.random()
    if style < 0.23 and active_count + 1 <= interior:
        # A dense run with one jointly empty interior lane also appears
        # in the original test, but is available in either split.
        span = active_count + 1
        start = int(rng.integers(1, cross_len - span))
        empty = start + int(rng.integers(1, span - 1))
        return tuple(c for c in range(start, start + span) if c != empty)
    if style < 0.48 and 2 * active_count - 1 <= interior:
        # Separated lanes are sampled independently of the other roles.
        span = int(rng.integers(2 * active_count - 1, interior + 1))
        start = int(rng.integers(1, cross_len - span))
        gaps = [2] * (active_count - 1)
        for _ in range(span - (2 * active_count - 1)):
            gaps[int(rng.integers(len(gaps)))] += 1
        coords = [start]
        for gap in gaps:
            coords.append(coords[-1] + gap)
        return tuple(coords)
    if regime == "target_test" and rng.random() < 0.35:
        span = min(interior, active_count + int(rng.integers(0, 3)))
    else:
        span = int(rng.integers(active_count, interior + 1))
    start = int(rng.integers(1, cross_len - span))
    middle = rng.choice(np.arange(start + 1, start + span - 1),
                        size=active_count - 2, replace=False)
    return tuple(sorted((start, *[int(x) for x in middle], start + span - 1)))


def _generate_pair(rng, regime, side, train_index=0, required_symbol=None,
                   forbidden_symbol=None):
    if regime not in ("source_train", "target_test"):
        raise AssertionError("unknown regime")
    small_probability = 0.32 if regime == "source_train" else 0.55
    axis_len = int(rng.integers(7, 13) if rng.random() < small_probability
                   else rng.integers(13, 31))
    cross_len = int(rng.integers(7, 13) if rng.random() < small_probability
                    else rng.integers(13, 31))
    ref_depth = int(rng.integers(3, axis_len - 3))
    move_depth = axis_len - ref_depth - 1
    barrier = move_depth if _is_reference_positive(side) else ref_depth
    facts = _axis_facts(side, axis_len, barrier)
    horizontal = _is_horizontal(side)
    shape = (axis_len, cross_len) if horizontal else (cross_len, axis_len)
    coords = _sample_lanes(rng, cross_len, regime)

    # Literal digits are local to a pair. The reserved digit is withheld
    # only from training symbols, so the first test has a new identity.
    role_pool = [c for c in range(10) if c != required_symbol]
    structure = int(rng.choice(role_pool))
    role_pool.remove(structure)
    ref_bg = int(rng.choice(role_pool))
    role_pool.remove(ref_bg)
    distinct = rng.random() < (0.42 if regime == "source_train" else 0.62)
    move_bg = int(rng.choice(role_pool)) if distinct else ref_bg
    symbol_pool = [c for c in range(10)
                   if c not in (structure, ref_bg, move_bg, forbidden_symbol)]
    symbol_count = int(rng.integers(2, min(6, len(coords), len(symbol_pool)) + 1))
    if required_symbol is None:
        symbols = [int(x) for x in rng.choice(symbol_pool, size=symbol_count,
                                               replace=False)]
    else:
        symbols = [required_symbol] + [int(x) for x in rng.choice(
            [c for c in symbol_pool if c != required_symbol],
            size=symbol_count - 1, replace=False)]
    rng.shuffle(symbols)

    # Every chosen identity is visible, while equality statuses and repeats
    # vary independently of the palette size and lane positions.
    references = [symbols[i % len(symbols)] for i in range(len(coords))]
    rng.shuffle(references)
    if regime == "source_train" and train_index < 2:
        match_count = int(rng.integers(1, len(coords)))
    else:
        match_count = int(rng.integers(0, len(coords) + 1))
    statuses = np.array([True] * match_count + [False] * (len(coords) - match_count))
    rng.shuffle(statuses)
    movers = [ref if status else int(rng.choice([x for x in symbols if x != ref]))
              for ref, status in zip(references, statuses)]
    if required_symbol is not None and required_symbol not in movers:
        candidates = [i for i, (ref, mover) in enumerate(zip(references, movers))
                      if ref != required_symbol and ref != mover]
        if candidates:
            movers[candidates[0]] = required_symbol
        else:
            index = references.index(required_symbol)
            movers[index] = required_symbol
            # Keep a mismatch demonstration after making this a match.
            if all(ref == mover for ref, mover in zip(references, movers)):
                other = next(i for i in range(len(movers)) if i != index)
                movers[other] = next(x for x in symbols if x != references[other])

    move_axes = list(facts["move_axes"])
    if regime == "source_train" and train_index == 0:
        # A visibly moving match establishes the near destination.
        move_axes.remove(facts["near_axis"])
    elif regime == "source_train" and train_index == 1:
        # A match already at its destination demonstrates the no-op case.
        move_axes = [facts["near_axis"]]
    if regime == "target_test" and rng.random() < 0.4:
        move_axis = facts["near_axis"]
    else:
        move_axis = int(rng.choice(move_axes))
    # Avoid an entirely unchanged example while permitting one-branch
    # sequences in extra training and test pairs.
    if all(ref == mover for ref, mover in zip(references, movers)) and move_axis == facts["near_axis"]:
        move_axis = next(axis for axis in facts["move_axes"] if axis != facts["near_axis"])
    if all(ref != mover for ref, mover in zip(references, movers)) and move_axis == facts["far_axis"]:
        move_axis = facts["near_axis"]

    input_grid = np.full(shape, move_bg, dtype=np.int8)
    view = input_grid if horizontal else input_grid.T
    for axis in facts["ref_axes"]:
        view[axis, :] = ref_bg
    input_grid[_structure_mask(shape, side, barrier)] = structure
    for cross, ref, mover in zip(coords, references, movers):
        view[facts["ref_axis"], cross] = ref
        view[move_axis, cross] = mover

    output_grid = input_grid.copy()
    output_view = output_grid if horizontal else output_grid.T
    for cross, ref, mover in zip(coords, references, movers):
        output_view[move_axis, cross] = move_bg
        destination = facts["near_axis"] if mover == ref else facts["far_axis"]
        output_view[destination, cross] = mover
    return input_grid, output_grid


def _grid_key(grid):
    return (tuple(grid.shape), tuple(int(x) for x in grid.ravel()))


def _pair_key(input_grid, output_grid):
    return (_grid_key(input_grid), _grid_key(output_grid))


def _rows_to_grid(rows):
    return np.asarray([[int(character) for character in row] for row in rows],
                      dtype=np.int8)

_SOURCE_ROW_PAIRS = (
    (
        (
            "9999999999999999",
            "9999999999999999",
            "9999999999999999",
            "9593959293929599",
            "9999999999999999",
            "9999999999999999",
            "9999999999999999",
            "0000000000000000",
            "0595939593959290",
            "0999999999999990",
            "0999999999999990",
            "0999999999999990",
            "0999999999999990",
            "0999999999999990",
            "0999999999999990",
            "0999999999999990",
        ),
        (
            "9993959299929599",
            "9999999999999999",
            "9999999999999999",
            "9999999999999999",
            "9999999999999999",
            "9999999999999999",
            "9599999993999999",
            "0000000000000000",
            "0595939593959290",
            "0999999999999990",
            "0999999999999990",
            "0999999999999990",
            "0999999999999990",
            "0999999999999990",
            "0999999999999990",
            "0999999999999990",
        ),
    ),
    (
        (
            "4447777777777777",
            "7147777787777777",
            "7747777777777777",
            "7147777727777777",
            "7747777777777777",
            "7347777787777777",
            "7747777777777777",
            "7147777717777777",
            "7747777777777777",
            "7847777787777777",
            "7747777777777777",
            "7147777757777777",
            "7747777777777777",
            "7747777777777777",
            "7847777717777777",
            "4447777777777777",
        ),
        (
            "4447777777777777",
            "7147777777777778",
            "7747777777777777",
            "7147777777777772",
            "7747777777777777",
            "7347777777777778",
            "7747777777777777",
            "7141777777777777",
            "7747777777777777",
            "7848777777777777",
            "7747777777777777",
            "7147777777777775",
            "7747777777777777",
            "7747777777777777",
            "7847777777777771",
            "4447777777777777",
        ),
    ),
    (
        (
            "44444333",
            "88854033",
            "88854533",
            "88884333",
            "88824233",
            "88824033",
            "88804533",
            "44444333",
        ),
        (
            "44444333",
            "88854330",
            "88854533",
            "88884333",
            "88824233",
            "88824330",
            "88804335",
            "44444333",
        ),
    ),
)


_SOURCE_PAIRS = tuple((_rows_to_grid(i), _rows_to_grid(o))
                      for i, o in _SOURCE_ROW_PAIRS)


def _is_source_pair(input_grid, output_grid):
    return any(input_grid.shape == old_i.shape and output_grid.shape == old_o.shape
               and np.array_equal(input_grid, old_i)
               and np.array_equal(output_grid, old_o)
               for old_i, old_o in _SOURCE_PAIRS)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed_key = _normalize_seed(seed)
    train_count, test_count = _resolve_counts(seed_key, num_train, num_test)
    rng = np.random.default_rng(seed_key)
    # The only puzzle-wide convention is the equality/near, inequality/far
    # rule. This sampled digit is a coverage device, not a hidden rule.
    fresh_symbol = int(rng.integers(10))
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    # Sides are pair-level. An unseen test side is legitimate because the
    # U shape identifies the reference side in every input.
    sides = [str(rng.choice(_SIDES)) for _ in range(train_count + test_count)]
    specifications = ([('train', 'source_train', i) for i in range(train_count)]
                      + [('test', 'target_test', i) for i in range(test_count)])
    for pair_number, (split, regime, index) in enumerate(specifications):
        for _ in range(50):
            pair_rng = np.random.default_rng(int(rng.integers(0, 1 << 63)))
            input_grid, output_grid = _generate_pair(
                pair_rng, regime, sides[pair_number], train_index=index,
                required_symbol=fresh_symbol if split == 'test' and index == 0 else None,
                forbidden_symbol=fresh_symbol if split == 'train' else None)
            key = _grid_key(input_grid)
            if key not in used_inputs and not _is_source_pair(input_grid, output_grid):
                used_inputs.add(key)
                puzzle[split].append({'input': input_grid, 'output': output_grid})
                break
        else:
            raise AssertionError('could not produce distinct pairs')
    validate_generated_puzzle(puzzle)
    return puzzle


def _unique_mode(values):
    counts = np.bincount(np.asarray(values, dtype=int).ravel(), minlength=10)
    winners = np.flatnonzero(counts == counts.max())
    assert len(winners) == 1
    return int(winners[0])


def _find_structure(grid):
    height, width = grid.shape
    candidates = []
    for side in _SIDES:
        axis_len = height if _is_horizontal(side) else width
        for barrier in range(1, axis_len - 1):
            line = grid[barrier, :] if _is_horizontal(side) else grid[:, barrier]
            if not np.all(line == line[0]):
                continue
            color = int(line[0])
            mask = _structure_mask(grid.shape, side, barrier)
            if np.array_equal(grid == color, mask):
                candidates.append((side, barrier, color, mask))
    assert len(candidates) == 1
    return candidates[0]


def _analyze_pair(input_grid, output_grid):
    assert input_grid.shape == output_grid.shape
    side, barrier, structure, mask = _find_structure(input_grid)
    assert np.array_equal(output_grid == structure, mask)
    horizontal = _is_horizontal(side)
    view = input_grid if horizontal else input_grid.T
    axis_len, cross_len = view.shape
    facts = _axis_facts(side, axis_len, barrier)
    assert len(facts['ref_axes']) >= 3 and len(facts['move_axes']) >= 3
    ref_bg = _unique_mode(view[list(facts['ref_axes']), 1:cross_len - 1])
    move_bg = _unique_mode(view[list(facts['move_axes']), 1:cross_len - 1])
    assert structure not in (ref_bg, move_bg)
    ref_axis, near_axis, far_axis = (facts[x] for x in ('ref_axis', 'near_axis', 'far_axis'))
    for axis in facts['ref_axes']:
        if axis != ref_axis:
            assert np.all(view[axis, 1:cross_len - 1] == ref_bg)
    active_axes = [axis for axis in facts['move_axes']
                   if np.any(view[axis, 1:cross_len - 1] != move_bg)]
    assert len(active_axes) == 1
    move_axis = active_axes[0]
    for axis in facts['move_axes']:
        if axis != move_axis:
            assert np.all(view[axis, 1:cross_len - 1] == move_bg)
    ref_coords = set(int(x) + 1 for x in np.flatnonzero(view[ref_axis, 1:cross_len - 1] != ref_bg))
    move_coords = set(int(x) + 1 for x in np.flatnonzero(view[move_axis, 1:cross_len - 1] != move_bg))
    assert ref_coords == move_coords
    assert 4 <= len(ref_coords) <= 11
    coords = tuple(sorted(ref_coords))
    refs = tuple(int(view[ref_axis, c]) for c in coords)
    movers = tuple(int(view[move_axis, c]) for c in coords)
    symbols = frozenset(refs + movers)
    assert structure not in symbols and ref_bg not in symbols and move_bg not in symbols
    assert len(symbols) >= 2
    expected = input_grid.copy()
    expected_view = expected if horizontal else expected.T
    matches = mismatches = 0
    for cross, ref, mover in zip(coords, refs, movers):
        for destination in (near_axis, far_axis):
            if destination != move_axis:
                assert int(view[destination, cross]) == move_bg
        expected_view[move_axis, cross] = move_bg
        if ref == mover:
            destination = near_axis
            matches += 1
        else:
            destination = far_axis
            mismatches += 1
        expected_view[destination, cross] = mover
    assert np.array_equal(expected, output_grid)
    return dict(side=side, horizontal=horizontal, barrier=barrier,
                structure=structure, ref_bg=ref_bg, move_bg=move_bg,
                ref_axis=ref_axis, near_axis=near_axis, far_axis=far_axis,
                move_axis=move_axis, coords=coords, symbols=symbols,
                matches=matches, mismatches=mismatches, shape=input_grid.shape,
                mover_symbols=frozenset(movers),
                empty_internal=coords[-1] - coords[0] + 1 - len(coords),
                move_distance=abs(move_axis - near_axis))


def _validate_generated_puzzle_impl(puzzle):
    assert type(puzzle) is dict and set(puzzle) == {'train', 'test'}
    assert type(puzzle['train']) is list and type(puzzle['test']) is list
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    analyses = {'train': [], 'test': []}
    input_keys, pair_keys = set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert type(pair) is dict and set(pair) == {'input', 'output'}
            i, o = pair['input'], pair['output']
            for grid in (i, o):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            assert i.shape == o.shape and not np.array_equal(i, o)
            ik, pk = _grid_key(i), _pair_key(i, o)
            assert ik not in input_keys and pk not in pair_keys
            input_keys.add(ik)
            pair_keys.add(pk)
            assert not _is_source_pair(i, o)
            analyses[split].append(_analyze_pair(i, o))
    # The training set establishes both destinations, including a moving
    # match and a match already next to the barrier.
    train = analyses['train']
    assert all(a['matches'] >= 1 and a['mismatches'] >= 1 for a in train[:2])
    assert any(a['matches'] and a['move_distance'] > 0 for a in train)
    assert any(a['matches'] and a['move_distance'] == 0 for a in train)
    # The fresh identity in test confirms that the rule is color-general.
    train_symbols = set().union(*(a['symbols'] for a in train))
    assert any(a['mover_symbols'] - train_symbols for a in analyses['test'])


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _validate_generated_puzzle_impl(puzzle)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError('invalid generated puzzle') from exc


def _puzzle_key(puzzle):
    return tuple((split, tuple(_pair_key(p['input'], p['output'])
                               for p in puzzle[split])) for split in ('train', 'test'))


def self_test() -> None:
    automatic = generate_puzzle(314159)
    validate_generated_puzzle(automatic)
    explicit = generate_puzzle(271828, 7, 3)
    assert _puzzle_key(explicit) == _puzzle_key(generate_puzzle(271828, 7, 3))
    assert _puzzle_key(explicit) != _puzzle_key(generate_puzzle(271829, 7, 3))
    for n, (tr, te) in enumerate(((2, 1), (10, 1), (10, 4))):
        p = generate_puzzle(8000 + n, tr, te)
        assert (len(p['train']), len(p['test'])) == (tr, te)
    for n, (tr, te) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(9000 + n, tr, te)
        assert (tr is None or len(p['train']) == tr)
        assert (te is None or len(p['test']) == te)
    for kwargs in ({'num_train': 1}, {'num_train': 11}, {'num_test': 0},
                   {'num_test': 5}, {'num_train': 2, 'num_test': 2},
                   {'num_train': 3, 'num_test': 4},
                   {'num_train': True, 'num_test': 1}):
        try:
            generate_puzzle(17, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f'illegal counts accepted: {kwargs}')
    train_counts, test_counts, puzzles = set(), set(), set()
    sides, shapes, layouts, structures, symbol_sets = set(), set(), set(), set(), set()
    small_train = large_test = two_tone_train = same_fill_test = 0
    dense_train = sparse_test = distant_test_match = unseen_test_side = 0
    role_changes = symbol_set_changes = 0
    same_orientation = horizontal_test_after_vertical_train = 0
    vertical_test_after_horizontal_train = 0
    structure_to_symbol = background_to_symbol = 0
    train_all_matches = train_all_mismatches = 0
    test_all_matches = test_all_mismatches = 0
    source_like_train = source_like_test = 0
    empty_counts, match_counts, ref_depths = set(), set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        train_counts.add(len(p['train']))
        test_counts.add(len(p['test']))
        puzzles.add(_puzzle_key(p))
        parsed = {split: [_analyze_pair(q['input'], q['output']) for q in p[split]]
                  for split in ('train', 'test')}
        tr, te = parsed['train'], parsed['test']
        all_pairs = tr + te
        train_sides = {a['side'] for a in tr}
        unseen_test_side += any(a['side'] not in train_sides for a in te)
        role_changes += len({a['structure'] for a in all_pairs}) > 1
        symbol_set_changes += len({a['symbols'] for a in all_pairs}) > 1
        same_orientation += any(tr[i]['side'] == tr[i + 1]['side']
                                for i in range(len(tr) - 1))
        horizontal_test_after_vertical_train += (
            all(not a['horizontal'] for a in tr)
            and any(a['horizontal'] for a in te))
        vertical_test_after_horizontal_train += (
            all(a['horizontal'] for a in tr)
            and any(not a['horizontal'] for a in te))
        for color in range(10):
            structure_to_symbol += (
                any(a['structure'] == color for a in all_pairs)
                and any(color in a['symbols'] for a in all_pairs))
            background_to_symbol += (
                any(color in (a['ref_bg'], a['move_bg']) for a in all_pairs)
                and any(color in a['symbols'] for a in all_pairs))
        for split, items in parsed.items():
            for a in items:
                sides.add(a['side'])
                shapes.add(a['shape'])
                structures.add(a['structure'])
                symbol_sets.add(a['symbols'])
                layouts.add((a['side'], a['shape'], a['barrier'], a['move_axis'], a['coords']))
                empty_counts.add(a['empty_internal'])
                match_counts.add(a['matches'])
                ref_depths.add(abs(a['ref_axis'] - a['barrier']) +
                               (a['shape'][0 if a['horizontal'] else 1]
                                - a['barrier'] - 2
                                if a['side'] in ('down', 'right')
                                else a['barrier'] - 1))
                if split == 'train':
                    small_train += max(a['shape']) <= 12
                    two_tone_train += a['ref_bg'] != a['move_bg']
                    dense_train += any(b - c == 1 for c, b in zip(a['coords'], a['coords'][1:]))
                    train_all_matches += a['mismatches'] == 0
                    train_all_mismatches += a['matches'] == 0
                    source_like_train += (
                        min(a['shape']) >= 14 and len(a['coords']) == 7
                        and a['ref_bg'] == a['move_bg']
                        and a['move_distance'] > 0
                        and all(b - c >= 2 for c, b in zip(a['coords'], a['coords'][1:])))
                else:
                    large_test += min(a['shape']) >= 14
                    same_fill_test += a['ref_bg'] == a['move_bg']
                    sparse_test += any(b - c >= 3 for c, b in zip(a['coords'], a['coords'][1:]))
                    distant_test_match += a['matches'] > 0 and a['move_distance'] > 0
                    test_all_matches += a['mismatches'] == 0
                    test_all_mismatches += a['matches'] == 0
                    source_like_test += (
                        max(a['shape']) <= 12
                        and a['ref_bg'] != a['move_bg']
                        and a['move_distance'] == 0
                        and a['empty_internal'] == 1)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzles) == 42 and sides == set(_SIDES)
    assert len(shapes) >= 50 and len(layouts) >= 120
    assert len(structures) == 10 and len(symbol_sets) >= 20
    assert min(small_train, large_test, two_tone_train, same_fill_test,
               dense_train, sparse_test, distant_test_match, unseen_test_side,
               role_changes, symbol_set_changes, same_orientation,
               horizontal_test_after_vertical_train,
               vertical_test_after_horizontal_train, structure_to_symbol,
               background_to_symbol, train_all_matches,
               train_all_mismatches, test_all_matches,
               test_all_mismatches, source_like_train,
               source_like_test) > 0
    assert len(empty_counts) >= 5 and len(match_counts) >= 5
    assert len(ref_depths) >= 5
