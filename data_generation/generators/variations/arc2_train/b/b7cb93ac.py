import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_PROFILES = (
    (6, 5, 1), (7, 3, 2), (6, 3, 3),
    (6, 3, 2, 1), (7, 2, 2, 1), (6, 4, 1, 1),
    (6, 2, 2, 1, 1),
)
_MASK_CACHE = {}


def _normalize(cells):
    cells = tuple((int(r), int(c)) for r, c in cells)
    if not cells:
        return ()
    top = min(r for r, _ in cells)
    left = min(c for _, c in cells)
    return tuple(sorted((r - top, c - left) for r, c in cells))


def _rotate(shape, turns):
    shape = _normalize(shape)
    for _ in range(int(turns) % 4):
        shape = _normalize((c, -r) for r, c in shape)
    return shape


def _shape_dims(shape):
    return (max(r for r, _ in shape) + 1,
            max(c for _, c in shape) + 1)


def _is_connected(cells):
    cells = set(cells)
    if not cells:
        return False
    seen = {next(iter(cells))}
    stack = list(seen)
    while stack:
        r, c = stack.pop()
        for dr, dc in _DIRS:
            neighbor = (r + dr, c + dc)
            if neighbor in cells and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen == cells


def _connected_masks(height, width):
    key = (height, width)
    if key not in _MASK_CACHE:
        groups = {size: [] for size in range(1, height * width + 1)}
        for mask in range(1, 1 << (height * width)):
            cells = [(i // width, i % width)
                     for i in range(height * width) if mask & (1 << i)]
            if _is_connected(cells):
                groups[len(cells)].append(mask)
        _MASK_CACHE[key] = groups
    return _MASK_CACHE[key]


def _mask_cells(mask, width):
    return tuple((i // width, i % width)
                 for i in range(mask.bit_length()) if mask & (1 << i))


def _random_partition(rng, shape, profile):
    height, width = shape
    groups = _connected_masks(height, width)
    full = (1 << (height * width)) - 1

    def fill(remaining, sizes):
        if not sizes:
            return [] if remaining == 0 else None
        first = remaining & -remaining
        for size in rng.permutation(sorted(set(sizes))):
            size = int(size)
            choices = [mask for mask in groups[size]
                       if mask & first and mask & remaining == mask]
            for index in rng.permutation(len(choices)):
                mask = choices[int(index)]
                rest = list(sizes)
                rest.remove(size)
                answer = fill(remaining ^ mask, rest)
                if answer is not None:
                    return [mask, *answer]
        return None

    for index in rng.permutation(len(groups[profile[0]])):
        anchor = groups[profile[0]][int(index)]
        fillers = fill(full ^ anchor, profile[1:])
        if fillers is not None:
            fillers.sort(key=lambda mask: (-mask.bit_count(), mask))
            return [_mask_cells(mask, width) for mask in [anchor, *fillers]]
    raise RuntimeError('profile has no connected rectangular tiling')


def _extract_components(grid, background=None):
    if background is None:
        background = int(grid[0, 0])
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    components = []
    for row in range(height):
        for col in range(width):
            if seen[row, col] or int(grid[row, col]) == background:
                continue
            color = int(grid[row, col])
            seen[row, col] = True
            stack = [(row, col)]
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in _DIRS:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < height and 0 <= cc < width and not seen[rr, cc] and int(grid[rr, cc]) != background:
                        assert int(grid[rr, cc]) == color, 'input tiles must be monochromatic and separated'
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            components.append({'color': color, 'shape': _normalize(cells), 'area': len(cells)})
    return components


def _solve_components(components, target_shape):
    """Enumerate at most two minimum-turn color grids with a fixed anchor."""
    height, width = target_shape
    if not components or sum(c['area'] for c in components) != height * width:
        return None, []
    largest = max(c['area'] for c in components)
    if sum(c['area'] == largest for c in components) != 1:
        return None, []
    anchor_index = next(i for i, c in enumerate(components) if c['area'] == largest)
    records = []
    for index, component in enumerate(components):
        shape = component['shape']
        orientations = {shape: 0} if index == anchor_index else {}
        if index != anchor_index:
            for turns in range(4):
                oriented = _rotate(shape, turns)
                cost = min(turns, 4 - turns)
                orientations[oriented] = min(cost, orientations.get(oriented, 3))
        placements = []
        for oriented, cost in orientations.items():
            sh, sw = _shape_dims(oriented)
            for r0 in range(height - sh + 1):
                for c0 in range(width - sw + 1):
                    cells = tuple(sorted((r0 + r) * width + c0 + c for r, c in oriented))
                    mask = sum(1 << cell for cell in cells)
                    placements.append((cost, mask, cells))
        placements.sort(key=lambda item: (item[0], item[1]))
        records.append({'color': component['color'], 'area': component['area'],
                        'shape': shape, 'placements': placements,
                        'signature': (component['color'], shape, index == anchor_index)})
    ordered = [records[anchor_index]] + sorted(
        (record for i, record in enumerate(records) if i != anchor_index),
        key=lambda record: (len(record['placements']), -record['area'], record['signature']))
    maximum = 2 * (len(ordered) - 1)
    for budget in range(maximum + 1):
        board = [0] * (height * width)
        solutions = set()
        previous = {}
        suffix = [0] * (len(ordered) + 1)
        for i in range(len(ordered) - 1, -1, -1):
            options = ordered[i]['placements']
            suffix[i] = suffix[i + 1] + (min(x[0] for x in options) if options else maximum + 1)

        def search(index, occupied, cost):
            if len(solutions) >= 2 or cost + suffix[index] > budget:
                return
            if index == len(ordered):
                if cost == budget and occupied.bit_count() == height * width:
                    solutions.add(tuple(board))
                return
            record = ordered[index]
            signature = record['signature']
            lower = previous.get(signature, -1)
            for choice, (turn_cost, mask, cells) in enumerate(record['placements']):
                if choice < lower or cost + turn_cost > budget or mask & occupied:
                    continue
                prior = previous.get(signature)
                previous[signature] = choice
                for cell in cells:
                    board[cell] = record['color']
                search(index + 1, occupied | mask, cost + turn_cost)
                for cell in cells:
                    board[cell] = 0
                if prior is None:
                    del previous[signature]
                else:
                    previous[signature] = prior
                if len(solutions) >= 2:
                    return
        search(0, 0, 0)
        if solutions:
            return budget, sorted(solutions)
    return None, []


def _packing_solutions(input_grid, target_shape):
    return _solve_components(_extract_components(input_grid), target_shape)


def _make_output(tiles, colors, shape, background):
    output = np.full(shape, background, dtype=np.int8)
    for tile, color in zip(tiles, colors):
        for r, c in tile:
            assert output[r, c] == background
            output[r, c] = color
    assert np.all(output != background)
    return output


def _scatter_tiles(rng, tiles, colors, background):
    for _ in range(80):
        height = int(rng.integers(9, 25))
        width = int(rng.integers(9, 25))
        occupied = set()
        placed = [None] * len(tiles)
        for index in sorted(range(len(tiles)), key=lambda i: -len(tiles[i])):
            shape = _normalize(tiles[index])
            sh, sw = _shape_dims(shape)
            positions = [(r, c) for r in range(1, height - sh)
                         for c in range(1, width - sw)]
            choice = None
            for j in rng.permutation(len(positions)):
                r0, c0 = positions[int(j)]
                cells = tuple((r0 + r, c0 + c) for r, c in shape)
                if any(cell in occupied or any((cell[0] + dr, cell[1] + dc) in occupied
                                               for dr, dc in _DIRS) for cell in cells):
                    continue
                choice = cells
                break
            if choice is None:
                break
            placed[index] = choice
            occupied.update(choice)
        if all(cells is not None for cells in placed):
            grid = np.full((height, width), background, dtype=np.int8)
            for cells, color in zip(placed, colors):
                for r, c in cells:
                    grid[r, c] = color
            return grid
    raise RuntimeError('could not scatter separated tiles')


def _grid_key(grid):
    return tuple(grid.shape), tuple(int(x) for x in grid.flat)


def _rows_key(rows):
    return (len(rows), len(rows[0])), tuple(int(char) for row in rows for char in row)


_SOURCE_RAW = (
    (("000000000000", "000000000000", "000020000000", "000022200000",
      "000020000000", "000000003330", "080000000030", "000000000330",
      "000000000000", "000000000000"), ("2333", "2223", "2833")),
    (("0000000000000", "0000000000800", "0000000000800", "0000100000000",
      "0111100000000", "0100100000000", "0000000000000", "0000000000000",
      "0000022200000", "0000000000000"), ("2221", "1111", "1881")),
    (("00000000000", "00010000000", "00111100000", "00010000000",
      "00000000000", "00000002200", "00008000000", "00000000200",
      "08000000200", "00000000000"), ("8122", "1111", "8122")),
    (("0000000000", "0000000000", "0088880000", "0080080060",
      "0000000660", "0000000000", "0000300000", "0003300000",
      "0000000000", "0000000000"), ("8888", "8368", "3366")),
)
_SOURCE_PAIR_KEYS = frozenset((_rows_key(a), _rows_key(b)) for a, b in _SOURCE_RAW)


def _count_value(name, value):
    if value is None:
        return None
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be an integer or None')
    return int(value)


def _resolve_counts(seed, rng, num_train, num_test):
    train = _count_value('num_train', num_train)
    test = _count_value('num_test', num_test)
    if train is not None and not 2 <= train <= 10:
        raise ValueError('num_train must be between 2 and 10')
    if test is not None and not 1 <= test <= 4:
        raise ValueError('num_test must be between 1 and 4')
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError('num_train must exceed num_test')
    return train, test


def _sample_colors(rng, tiles, parameters, role):
    palette = parameters['palette']
    pivot, legacy, novel = parameters['pivot'], parameters['legacy'], parameters['novel']
    count = len(tiles)
    if role == 'train_bridge':
        colors = [legacy, pivot]
        colors += [int(rng.choice([c for c in palette if c not in (novel, legacy)]))
                   for _ in range(count - 2)]
    elif role == 'train_repeated':
        anchor = int(rng.choice([c for c in palette if c not in (novel, pivot)]))
        colors = [anchor] + [pivot] * (count - 1)
    elif role == 'test_bridge':
        colors = [pivot, legacy, novel]
        colors += [int(rng.choice([c for c in palette if c != pivot])) for _ in range(count - 3)]
    else:
        available = [c for c in palette if c != novel] if role.startswith('train') else palette
        anchor = int(rng.choice(available))
        colors = [anchor]
        for index in range(1, count):
            earlier = [colors[j] for j in range(1, index)]
            if earlier and rng.random() < 0.32:
                colors.append(int(rng.choice(earlier)))
            else:
                colors.append(int(rng.choice([c for c in available if c != anchor])))
        if count == 5 and rng.random() < 0.7:
            colors[1:] = [colors[1]] * 4
    return colors


def _generate_pair(rng, shape, background, profile, parameters, role,
                   require_turn=False, require_l_turn=False, require_source_geometry=False):
    for _ in range(1800 if require_source_geometry else 700):
        tiles = _random_partition(rng, shape, profile)
        if require_source_geometry and (
            _shape_dims(_normalize(tiles[0])) != (2, 4) or
            any(_shape_dims(_normalize(tile)) != (2, 2) for tile in tiles[1:])
        ):
            continue
        colors = _sample_colors(rng, tiles, parameters, role)
        output = _make_output(tiles, colors, shape, background)
        turns = [0]
        for tile in tiles[1:]:
            turns.append(int(rng.integers(0, 4)) if len(tile) > 1 else 0)
        if require_l_turn:
            ls = [i for i, tile in enumerate(tiles) if i and len(tile) == 3 and _shape_dims(_normalize(tile)) == (2, 2)]
            if not ls:
                continue
            turns[int(rng.choice(ls))] = int(rng.choice((1, 3)))
        elif require_turn:
            candidates = [i for i, tile in enumerate(tiles) if i and _rotate(tile, 1) != _normalize(tile)]
            if not candidates:
                continue
            turns[int(rng.choice(candidates))] = int(rng.choice((1, 3)))
        input_tiles = [_rotate(tile, -turn) for tile, turn in zip(tiles, turns)]
        components = [{'color': color, 'shape': tile, 'area': len(tile)}
                      for tile, color in zip(input_tiles, colors)]
        cost, solutions = _solve_components(components, shape)
        if len(solutions) != 1 or solutions[0] != tuple(int(x) for x in output.flat):
            continue
        if require_turn and cost == 0:
            continue
        if require_l_turn and cost == 0:
            continue
        input_grid = _scatter_tiles(rng, input_tiles, colors, background)
        return {'input': input_grid, 'output': output}
    raise RuntimeError(f'could not make unique pair for {profile}, {shape}, {role}')


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        raise ValueError('seed must be an integer')
    seed = int(seed)
    rng = np.random.default_rng(seed % (1 << 64))
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    background = int(rng.integers(0, 10))
    palette = [int(c) for c in rng.permutation([c for c in range(10) if c != background])]
    parameters = {'palette': palette, 'pivot': palette[0], 'legacy': palette[1], 'novel': palette[2]}
    shape = ((3, 4), (4, 3), (2, 6), (6, 2))[int(rng.choice(4, p=[0.48, 0.25, 0.15, 0.12]))]
    profiles = (list(_PROFILES) if min(shape) >= 3 else
                [(6, 5, 1), (7, 3, 2), (6, 3, 3), (6, 3, 2, 1)])
    train_profiles = [profiles[int(rng.integers(0, len(profiles)))] for _ in range(train_count)]
    bridge_train = int(rng.integers(0, train_count))
    repeated_train = (bridge_train + int(rng.integers(1, train_count))) % train_count
    rotation_train = int(rng.integers(0, train_count))
    train_profiles[bridge_train] = ((6, 5, 1), (7, 3, 2), (6, 3, 3),
                                    (6, 3, 2, 1))[int(rng.integers(0, 4))]
    repeated_options = ((6, 5, 1), (6, 2, 2, 1, 1), (6, 4, 1, 1)) if min(shape) >= 3 else ((6, 5, 1),)
    train_profiles[repeated_train] = repeated_options[int(rng.integers(0, len(repeated_options)))]
    if len(train_profiles[bridge_train]) == len(train_profiles[repeated_train]):
        alternatives = [p for p in ((6, 5, 1), (7, 3, 2), (6, 3, 3), (6, 3, 2, 1))
                        if len(p) != len(train_profiles[repeated_train])]
        train_profiles[bridge_train] = alternatives[int(rng.integers(0, len(alternatives)))]
    test_profiles = [profiles[int(rng.integers(0, len(profiles)))] for _ in range(test_count)]
    bridge_test = int(rng.integers(0, test_count))
    test_profiles[bridge_test] = ((6, 5, 1), (7, 3, 2), (6, 3, 3),
                                  (6, 3, 2, 1))[int(rng.integers(0, 4))]
    if shape == (3, 4) and rng.random() < 0.35:
        test_profiles[bridge_test] = (6, 3, 3)
    puzzle = {'train': [], 'test': []}
    seen = set()
    for section, count, selected_profiles in (('train', train_count, train_profiles),
                                               ('test', test_count, test_profiles)):
        for index in range(count):
            if section == 'train':
                role = 'train_bridge' if index == bridge_train else 'train_repeated' if index == repeated_train else 'train_other'
            else:
                role = 'test_bridge' if index == bridge_test else 'test_other'
            pair_rng = np.random.default_rng(int(rng.integers(0, 1 << 63, dtype=np.uint64)))
            source_geometry = (section == 'test' and index == bridge_test and
                               shape == (3, 4) and selected_profiles[index] == (6, 3, 3))
            for _ in range(12):
                pair = _generate_pair(pair_rng, shape, background, selected_profiles[index], parameters,
                                      role, require_turn=(section == 'train' and index == rotation_train),
                                      require_l_turn=source_geometry,
                                      require_source_geometry=source_geometry)
                key = _grid_key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[section].append(pair)
                    break
            else:
                raise RuntimeError('could not make a distinct input')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    background = None
    output_shape = None
    seen_inputs = set()
    seen_pairs = set()
    train_anchor, train_filler, train_colors = set(), set(), set()
    test_anchor, test_filler, test_colors = set(), set(), set()
    rotation_in_train = False
    repeated_different_sizes = False
    observed_profiles = set()
    observed_counts = set()
    for section in ('train', 'test'):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = pair['input'], pair['output']
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            bg = int(inp[0, 0])
            if background is None:
                background = bg
                output_shape = out.shape
            assert bg == background and out.shape == output_shape
            assert np.all(inp[0, :] == bg) and np.all(inp[-1, :] == bg)
            assert np.all(inp[:, 0] == bg) and np.all(inp[:, -1] == bg)
            assert np.all(out != bg)
            assert inp.shape[0] > out.shape[0] and inp.shape[1] > out.shape[1]
            assert int(np.count_nonzero(inp != bg)) == out.size
            for color in range(10):
                if color != bg:
                    assert int(np.count_nonzero(inp == color)) == int(np.count_nonzero(out == color))
            key = _grid_key(inp)
            pair_key = (key, _grid_key(out))
            assert key not in seen_inputs and pair_key not in seen_pairs
            assert pair_key not in _SOURCE_PAIR_KEYS
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            components = _extract_components(inp, bg)
            assert 3 <= len(components) <= 5
            observed_counts.add(len(components))
            observed_profiles.add(tuple(sorted((c['area'] for c in components), reverse=True)))
            assert sum(c['area'] for c in components) == out.size
            anchor_size = max(c['area'] for c in components)
            assert sum(c['area'] == anchor_size for c in components) == 1
            anchor = next(c for c in components if c['area'] == anchor_size)
            cost, solutions = _solve_components(components, output_shape)
            assert cost is not None and len(solutions) == 1
            assert solutions[0] == tuple(int(x) for x in out.flat)
            anchors = train_anchor if section == 'train' else test_anchor
            fillers = train_filler if section == 'train' else test_filler
            colors = train_colors if section == 'train' else test_colors
            anchors.add(anchor['color'])
            colors.update(c['color'] for c in components)
            fillers.update(c['color'] for c in components if c is not anchor)
            if section == 'train':
                rotation_in_train |= cost > 0
                groups = {}
                for c in components:
                    groups.setdefault(c['color'], set()).add(c['area'])
                repeated_different_sizes |= any(len(sizes) > 1 for sizes in groups.values())
    assert rotation_in_train
    assert repeated_different_sizes
    assert train_filler & test_anchor
    assert train_anchor & test_filler
    assert test_colors - train_colors
    assert len(observed_profiles) >= 2
    assert len(observed_counts) >= 2


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p['input']), _grid_key(p['output']))
                       for p in puzzle[section]) for section in ('train', 'test'))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    assert len({_puzzle_key(p) for p in puzzles}) >= 40
    assert _puzzle_key(generate_puzzle(777, 6, 3)) == _puzzle_key(generate_puzzle(777, 6, 3))
    assert _puzzle_key(generate_puzzle(777, 6, 3)) != _puzzle_key(generate_puzzle(778, 6, 3))
    for seed, nt, ns in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                         (1101, None, 1), (1102, None, 4), (1103, 2, None), (1104, 10, None)):
        puzzle = generate_puzzle(seed, nt, ns)
        if nt is not None:
            assert len(puzzle['train']) == nt
        if ns is not None:
            assert len(puzzle['test']) == ns
        validate_generated_puzzle(puzzle)
    for kw in ({'num_train': 1}, {'num_train': 11}, {'num_test': 0}, {'num_test': 5},
               {'num_train': 2, 'num_test': 2}, {'num_train': 3, 'num_test': 4}):
        try:
            generate_puzzle(1200, **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f'illegal counts accepted: {kw}')
    all_pairs = [pair for p in puzzles for section in ('train', 'test') for pair in p[section]]
    dimensions = {pair['input'].shape for pair in all_pairs}
    output_shapes = {pair['output'].shape for pair in all_pairs}
    profiles = {tuple(sorted((c['area'] for c in _extract_components(pair['input'])), reverse=True))
                for pair in all_pairs}
    anchors = {next(c['shape'] for c in _extract_components(pair['input'])
                    if c['area'] == max(x['area'] for x in _extract_components(pair['input'])))
               for pair in all_pairs}
    backgrounds = {int(pair['input'][0, 0]) for pair in all_pairs}
    assert len(dimensions) >= 25 and len(output_shapes) >= 3
    assert (6, 5, 1) in profiles and (6, 3, 3) in profiles and (6, 2, 2, 1, 1) in profiles
    assert len(profiles) >= 6 and len(anchors) >= 12 and len(backgrounds) >= 6
    assert len({_grid_key(pair['input']) for pair in all_pairs}) >= len(all_pairs) - 1

    anchor_smaller = anchor_offset = filler_touch = l_turn = nonbar_turn = 0
    source_like_tests = zero_cost = high_cost = 0
    reused_profile_puzzles = extended_profile_puzzles = 0
    for puzzle in puzzles:
        train_profiles = {
            tuple(sorted((c['area'] for c in _extract_components(pair['input'])), reverse=True))
            for pair in puzzle['train']
        }
        test_profiles = {
            tuple(sorted((c['area'] for c in _extract_components(pair['input'])), reverse=True))
            for pair in puzzle['test']
        }
        assert len(train_profiles | test_profiles) >= 2
        reused_profile_puzzles += bool(train_profiles & test_profiles)
        extended_profile_puzzles += bool(test_profiles - train_profiles)
        for section in ('train', 'test'):
            for pair in puzzle[section]:
                inp, out = pair['input'], pair['output']
                components = _extract_components(inp)
                anchor = max(components, key=lambda c: c['area'])
                anchor_cells = np.argwhere(out == anchor['color'])
                top, left = anchor_cells.min(axis=0)
                bottom, right = anchor_cells.max(axis=0)
                anchor_smaller += (bottom - top + 1) * (right - left + 1) < out.size
                anchor_offset += bool(top or left)
                filler_colors = {c['color'] for c in components if c is not anchor}
                filler_touch += any(
                    int(out[r, c]) in filler_colors and int(out[r, c + 1]) in filler_colors
                    and out[r, c] != out[r, c + 1]
                    for r in range(out.shape[0]) for c in range(out.shape[1] - 1)
                ) or any(
                    int(out[r, c]) in filler_colors and int(out[r + 1, c]) in filler_colors
                    and out[r, c] != out[r + 1, c]
                    for r in range(out.shape[0] - 1) for c in range(out.shape[1])
                )
                colors = [c['color'] for c in components]
                for component in components:
                    if component is anchor or colors.count(component['color']) != 1:
                        continue
                    oriented = _normalize(np.argwhere(out == component['color']))
                    turned = oriented != component['shape']
                    assert oriented in {_rotate(component['shape'], t) for t in range(4)}
                    if turned and component['area'] > 2 and min(_shape_dims(component['shape'])) > 1:
                        nonbar_turn += 1
                    if turned and component['area'] == 3 and _shape_dims(component['shape']) == (2, 2):
                        l_turn += 1
                cost, _ = _packing_solutions(inp, out.shape)
                zero_cost += cost == 0
                high_cost += cost >= 2
                areas = tuple(sorted((c['area'] for c in components), reverse=True))
                if section == 'test' and areas == (6, 3, 3):
                    source_like_tests += (
                        _shape_dims(anchor['shape']) == (2, 4)
                        and all(_shape_dims(c['shape']) == (2, 2)
                                for c in components if c is not anchor)
                    )
    assert anchor_smaller >= 40 and anchor_offset >= 20 and filler_touch >= 15
    assert nonbar_turn >= 20 and l_turn >= 10 and source_like_tests >= 1
    assert zero_cost >= 20 and high_cost >= 15
    assert reused_profile_puzzles >= 10 and extended_profile_puzzles >= 5
