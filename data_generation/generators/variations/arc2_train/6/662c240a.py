import numpy as np


_SOURCE_PAIRS = (
    (
        (
            (2, 2, 2),
            (2, 2, 3),
            (2, 3, 3),
            (5, 7, 7),
            (7, 5, 5),
            (7, 5, 5),
            (8, 8, 1),
            (1, 8, 1),
            (1, 8, 1),
        ),
        (
            (8, 8, 1),
            (1, 8, 1),
            (1, 8, 1),
        ),
    ),
    (
        (
            (1, 5, 5),
            (5, 1, 1),
            (5, 1, 1),
            (3, 3, 3),
            (3, 6, 3),
            (3, 6, 6),
            (7, 7, 7),
            (7, 2, 2),
            (7, 2, 2),
        ),
        (
            (3, 3, 3),
            (3, 6, 3),
            (3, 6, 6),
        ),
    ),
    (
        (
            (8, 8, 4),
            (4, 4, 4),
            (4, 4, 8),
            (1, 1, 3),
            (1, 3, 3),
            (3, 3, 1),
            (6, 2, 2),
            (2, 2, 2),
            (2, 2, 6),
        ),
        (
            (8, 8, 4),
            (4, 4, 4),
            (4, 4, 8),
        ),
    ),
    (
        (
            (8, 9, 8),
            (9, 8, 8),
            (8, 8, 8),
            (2, 2, 1),
            (2, 2, 1),
            (1, 1, 2),
            (4, 4, 4),
            (4, 4, 3),
            (3, 3, 3),
        ),
        (
            (4, 4, 4),
            (4, 4, 3),
            (3, 3, 3),
        ),
    ),
    (
        (
            (5, 4, 4),
            (4, 5, 4),
            (4, 5, 4),
            (3, 3, 2),
            (3, 3, 2),
            (2, 2, 3),
            (1, 1, 1),
            (1, 8, 8),
            (1, 8, 8),
        ),
        (
            (5, 4, 4),
            (4, 5, 4),
            (4, 5, 4),
        ),
    ),
)



def _is_integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(rng, seed, num_train, num_test):
    if num_train is not None:
        if not _is_integer(num_train) or not 2 <= int(num_train) <= 10:
            raise ValueError('num_train must be an integer from 2 to 10')
        num_train = int(num_train)
    if num_test is not None:
        if not _is_integer(num_test) or not 1 <= int(num_test) <= 4:
            raise ValueError('num_test must be an integer from 1 to 4')
        num_test = int(num_test)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        low = max(2, num_test + 1)
        num_train = low + (seed // 4) % (11 - low)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return num_train, num_test


def _grid_key(grid):
    return (grid.shape, tuple(int(v) for v in grid.flat))


def _source_pair(inp, out):
    a = tuple(tuple(int(v) for v in row) for row in inp)
    b = tuple(tuple(int(v) for v in row) for row in out)
    return (a, b) in _SOURCE_PAIRS


def _layout(inp):
    height, width = inp.shape
    if height > width and height % width == 0:
        side, count, direction = width, height // width, 'vertical'
        tiles = [inp[i * side:(i + 1) * side, :] for i in range(count)]
    elif width > height and width % height == 0:
        side, count, direction = height, width // height, 'horizontal'
        tiles = [inp[:, i * side:(i + 1) * side] for i in range(count)]
    else:
        raise AssertionError('input does not reveal a strip of square tiles')
    assert 2 <= side <= 5 and 2 <= count <= 5
    return direction, side, count, tiles


def _mismatches(tile):
    return sum(int(tile[i, j] != tile[j, i])
               for i in range(tile.shape[0]) for j in range(i + 1, tile.shape[0]))


def _roles(rng, side, colors, symmetric, mismatch_count, horizontal_case=False):
    if horizontal_case:
        for _ in range(40):
            top = rng.integers(0, colors, size=3, dtype=np.int8)
            middle = rng.integers(0, colors, size=3, dtype=np.int8)
            result = np.stack((top, middle, top)).astype(np.int8)
            if len(np.unique(result)) == colors and _mismatches(result) > 0:
                return result
    for _ in range(80):
        result = np.zeros((side, side), dtype=np.int8)
        for i in range(side):
            for j in range(i, side):
                result[i, j] = int(rng.integers(colors))
                result[j, i] = result[i, j]
        if not symmetric:
            pairs = [(i, j) for i in range(side) for j in range(i + 1, side)]
            chosen = rng.choice(len(pairs), size=mismatch_count, replace=False)
            for index in np.atleast_1d(chosen):
                i, j = pairs[int(index)]
                alternatives = [v for v in range(colors) if v != result[i, j]]
                result[j, i] = alternatives[int(rng.integers(len(alternatives)))]
        if len(np.unique(result)) == colors:
            return result
    # This fallback is deterministic and preserves the requested mismatch count.
    result = np.zeros((side, side), dtype=np.int8)
    independent = [(i, j) for i in range(side) for j in range(i, side)]
    for index, (i, j) in enumerate(independent):
        result[i, j] = index % colors
        result[j, i] = result[i, j]
    if not symmetric:
        pairs = [(i, j) for i in range(side) for j in range(i + 1, side)]
        for i, j in pairs[:mismatch_count]:
            result[j, i] = 1
    return result


def _palettes(rng, count, colors, mode, anchor):
    order = [int(v) for v in rng.permutation(10)]
    if mode == 'all_same':
        palette = tuple(anchor[:colors]) if colors == 2 and rng.random() < .7 else tuple(order[:colors])
        return [palette] * count
    if mode == 'two_same' and count >= 3:
        first = tuple(order[:colors])
        result = [first, first]
        used = {frozenset(first)}
        for index in range(2, count):
            for offset in range(1, 11):
                palette = tuple(order[(index * colors + offset + j) % 10]
                                for j in range(colors))
                identity = frozenset(palette)
                if identity not in used:
                    result.append(palette)
                    used.add(identity)
                    break
        return result
    if mode == 'disjoint' and count * colors <= 10:
        return [tuple(order[i * colors:(i + 1) * colors]) for i in range(count)]
    if mode == 'chain':
        return [tuple(order[((i * (colors - 1) + j) % 10)] for j in range(colors))
                for i in range(count)]
    return [tuple(int(v) for v in rng.choice(10, size=colors, replace=False))
            for _ in range(count)]


def _choose_motif(rng, bank, side, colors, symmetric, mismatches, palette, reuse_rate,
                  horizontal_case=False):
    key = (side, colors, symmetric)
    candidates = [roles for roles, old_palette in bank.get(key, ())
                  if old_palette != palette and (symmetric or _mismatches(roles) == mismatches)]
    if candidates and rng.random() < reuse_rate:
        return candidates[int(rng.integers(len(candidates)))].copy()
    return _roles(rng, side, colors, symmetric, mismatches, horizontal_case)


def _make_pair(rng, geometry, colors, position, mismatch_count, mode, anchor, bank,
               seen_inputs, reuse_rate, horizontal_case=False):
    direction, side, count = geometry
    for _ in range(30):
        palettes = _palettes(rng, count, colors, mode, anchor)
        tiles = []
        pending = []
        for index in range(count):
            symmetric = index != position
            roles = _choose_motif(rng, bank, side, colors, symmetric, mismatch_count,
                                  palettes[index], reuse_rate,
                                  horizontal_case and not symmetric)
            lookup = np.asarray(palettes[index], dtype=np.int8)
            tiles.append(lookup[roles])
            pending.append(((side, colors, symmetric), roles.copy(), palettes[index]))
        inp = np.vstack(tiles) if direction == 'vertical' else np.hstack(tiles)
        out = tiles[position].copy()
        key = _grid_key(inp)
        if key in seen_inputs or _source_pair(inp, out):
            continue
        seen_inputs.add(key)
        for motif_key, roles, palette in pending:
            bank.setdefault(motif_key, []).append((roles, palette))
        return {'input': inp.astype(np.int8), 'output': out.astype(np.int8)}
    raise RuntimeError('unable to construct a distinct pair')


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if not _is_integer(seed) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, seed, num_train, num_test)
    # Color order is the only sampled puzzle-level preference; it never changes
    # the selection rule. The same two colors can play either tile role.
    anchor = tuple(int(v) for v in rng.choice(10, size=2, replace=False))
    train_plan = [('vertical', 3, 3),
                  ('horizontal', 3, int(rng.integers(2, 6)))]
    if train_count >= 3:
        train_plan.append(('vertical', int(rng.choice((2, 4, 5))),
                           int(rng.integers(3, 6))))
    for _ in range(3, train_count):
        train_plan.append((('vertical', 'horizontal')[int(rng.integers(2))],
                           int(rng.integers(2, 6)), int(rng.integers(2, 6))))
    bank = {}
    seen_inputs = set()
    train = []
    test = []
    for index, geometry in enumerate(train_plan):
        _, side, count = geometry
        colors = 3 if index == 3 or (index > 3 and rng.random() < .28) else 2
        position = index if index < 3 else int(rng.integers(count))
        if index == 0:
            mismatch_count = 1
        elif index == 1:
            mismatch_count = int(rng.integers(2, 4))
        else:
            mismatch_count = int(rng.integers(1, side * (side - 1) // 2 + 1))
        mode = ('disjoint', 'all_same', 'chain')[index] if index < 3 else (
            'disjoint', 'all_same', 'chain', 'two_same', 'mixed')[int(rng.integers(5))]
        horizontal_case = index >= 2 and side == 3 and colors == 2 and rng.random() < .27
        train.append(_make_pair(rng, geometry, colors, position, mismatch_count,
                                mode, anchor, bank, seen_inputs, .24,
                                horizontal_case))
    for index in range(test_count):
        # Each test geometry and palette size has been shown in training.
        exemplar = int(rng.integers(train_count))
        geometry = train_plan[exemplar]
        _, side, count = geometry
        colors = len(np.unique(_layout(train[exemplar]['input'])[3][0]))
        if train_count == 2:
            position = int(rng.integers(min(2, count)))
        else:
            position = int(rng.integers(count))
        mismatch_count = int(rng.integers(1, side * (side - 1) // 2 + 1))
        mode = ('disjoint', 'all_same', 'chain', 'two_same', 'mixed')[int(rng.integers(5))]
        horizontal_case = side == 3 and colors == 2 and rng.random() < .27
        test.append(_make_pair(rng, geometry, colors, position, mismatch_count,
                               mode, anchor, bank, seen_inputs, .64,
                               horizontal_case))
    puzzle = {'train': train, 'test': test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert np.issubdtype(grid.dtype, np.integer)
    assert all(1 <= extent <= 30 for extent in grid.shape)
    assert bool(np.all((0 <= grid) & (grid <= 9)))


def _inspect_pair(pair):
    inp, out = pair['input'], pair['output']
    direction, side, count, tiles = _layout(inp)
    assert out.shape == (side, side)
    color_counts = [len(np.unique(tile)) for tile in tiles]
    assert len(set(color_counts)) == 1 and color_counts[0] in (2, 3)
    wrong = [i for i, tile in enumerate(tiles) if not np.array_equal(tile, tile.T)]
    assert len(wrong) == 1
    assert np.array_equal(out, tiles[wrong[0]])
    assert not np.array_equal(inp, out)
    return direction, side, count, wrong[0], color_counts[0], _mismatches(out), tiles


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    train, test = puzzle['train'], puzzle['test']
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen_inputs, seen_pairs = set(), set()
    train_features = []
    test_features = []
    for split, destination in ((train, train_features), (test, test_features)):
        for pair in split:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = pair['input'], pair['output']
            _check_grid(inp)
            _check_grid(out)
            input_key = _grid_key(inp)
            pair_key = (input_key, _grid_key(out))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert not _source_pair(inp, out)
            destination.append(_inspect_pair(pair))
    assert train_features[0][:4] == ('vertical', 3, 3, 0)
    assert train_features[0][4:6] == (2, 1)
    assert train_features[1][0] == 'horizontal' and train_features[1][1] == 3
    assert train_features[1][3] == 1 and train_features[1][4] == 2
    assert train_features[1][5] in (2, 3)
    if len(train) >= 3:
        assert train_features[2][3] == 2
    if len(train) >= 4:
        assert train_features[3][4] == 3
    demonstrated = {(f[0], f[1], f[2], f[4]) for f in train_features}
    for feature in test_features:
        assert (feature[0], feature[1], feature[2], feature[4]) in demonstrated
        if len(train) == 2:
            assert feature[3] in {0, 1}


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p['input']), _grid_key(p['output']))
                       for p in puzzle[split]) for split in ('train', 'test'))


def _expect_error(seed, train, test):
    try:
        generate_puzzle(seed, train, test)
    except ValueError:
        return
    raise AssertionError('expected ValueError')


def _canonical_motif(tile):
    mapping = {}
    result = []
    for value in tile.flat:
        value = int(value)
        if value not in mapping:
            mapping[value] = len(mapping)
        result.append(mapping[value])
    return tuple(result)


def _predict_from_demonstrations(train, test_input):
    for example in train:
        _, _, _, candidates = _layout(example['input'])
        selected = [tile for tile in candidates if _mismatches(tile) > 0]
        assert len(selected) == 1 and np.array_equal(example['output'], selected[0])
    _, _, _, candidates = _layout(test_input)
    selected = [tile for tile in candidates if _mismatches(tile) > 0]
    assert len(selected) == 1
    return selected[0].copy()


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, positions, test_positions, colors, mismatches = set(), set(), set(), set(), set()
    directions, sides, candidates = set(), set(), set()
    modes = set()
    motif_recolors = {'symmetric': 0, 'asymmetric': 0}
    horizontal_asymmetry = 0
    adjacent_one = 0
    exactly_two_shared = 0
    combined_changes = 0
    palette_role_reuse = 0
    within_orientation = 0
    within_size = 0
    within_count = 0
    puzzles = set()
    all_inputs = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        puzzles.add(_puzzle_key(puzzle))
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        train_motifs = set()
        selected_palettes = set()
        symmetric_palettes = set()
        local_directions, local_sides, local_counts = set(), set(), set()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                assert _grid_key(inp) not in all_inputs
                all_inputs.add(_grid_key(inp))
                direction, side, count, position, color_count, mismatch, tiles = _inspect_pair(pair)
                assert not _source_pair(inp, out)
                shapes.add(inp.shape)
                directions.add(direction)
                sides.add(side)
                candidates.add(count)
                local_directions.add(direction)
                local_sides.add(side)
                local_counts.add(count)
                positions.add(position)
                colors.add(color_count)
                mismatches.add(mismatch)
                if split == 'test':
                    test_positions.add(position)
                if side == 3 and np.array_equal(out, out[::-1, :]):
                    horizontal_asymmetry += 1
                palettes = [frozenset(int(v) for v in np.unique(tile)) for tile in tiles]
                multiplicities = sorted((palettes.count(palette) for palette in set(palettes)),
                                        reverse=True)
                exactly_two_shared += (count >= 3 and multiplicities[0] == 2 and
                                       all(value == 1 for value in multiplicities[1:]))
                selected_palettes.add(palettes[position])
                symmetric_palettes.update(palettes[i] for i in range(count) if i != position)
                adjacent_one += any(len(palettes[i] & palettes[i + 1]) == 1
                                    for i in range(count - 1))
                if len(set(palettes)) == 1:
                    modes.add('all_same')
                    mode = 'all_same'
                elif all(palettes[i].isdisjoint(palettes[j]) for i in range(count) for j in range(i + 1, count)):
                    modes.add('disjoint')
                    mode = 'disjoint'
                elif any(palettes[i] & palettes[j] for i in range(count) for j in range(i + 1, count)):
                    modes.add('shared')
                    mode = 'shared'
                else:
                    raise AssertionError('unclassified palette arrangement')
                combined_changes += (direction == 'horizontal' and side != 3 and
                                     count != 3 and color_count == 3 and mode == 'shared')
                for tile in tiles:
                    motif = (side, color_count, _canonical_motif(tile), bool(np.array_equal(tile, tile.T)))
                    palette = frozenset(int(v) for v in np.unique(tile))
                    if split == 'test' and (motif, palette) not in train_motifs and any(
                            old_motif == motif and old_palette != palette
                            for old_motif, old_palette in train_motifs):
                        motif_recolors['symmetric' if np.array_equal(tile, tile.T)
                                       else 'asymmetric'] += 1
                    if split == 'train':
                        train_motifs.add((motif, palette))
        palette_role_reuse += bool(selected_palettes & symmetric_palettes)
        within_orientation += len(local_directions) > 1
        within_size += len(local_sides) > 1
        within_count += len(local_counts) > 1
        for pair in puzzle['test']:
            predicted = _predict_from_demonstrations(puzzle['train'], pair['input'].copy())
            assert np.array_equal(predicted, pair['output'])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzles) == 42 and len(all_inputs) > 250
    assert directions == {'vertical', 'horizontal'}
    assert sides == {2, 3, 4, 5} and candidates == {2, 3, 4, 5}
    assert colors == {2, 3} and {0, 1, 2}.issubset(positions)
    assert test_positions == {0, 1, 2, 3, 4}
    assert {1, 2, 3}.issubset(mismatches)
    assert modes == {'all_same', 'disjoint', 'shared'}
    assert len(shapes) >= 15 and horizontal_asymmetry >= 3
    assert motif_recolors['symmetric'] >= 5 and motif_recolors['asymmetric'] >= 2
    assert adjacent_one >= 20 and exactly_two_shared >= 10 and combined_changes >= 2
    assert palette_role_reuse >= 20
    assert within_orientation == 42 and within_size >= 30 and within_count >= 30
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                        (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(91, train, test))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (4, 4), (2, 4), (True, 1), (2, False)):
        _expect_error(7, train, test)
