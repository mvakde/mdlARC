import numpy as np


_D4 = ("r0", "r90", "r180", "r270", "flr", "ftb", "diag", "anti")


def _grid_key(grid):
    return (grid.shape, tuple(int(x) for x in grid.ravel()))


def _parse_grid(text):
    rows = text.split("/")
    return np.array([[int(ch) for ch in row] for row in rows], dtype=np.int8)


_SOURCE_TEXT_PAIRS = (
    (
        "0000000000000/0000200000000/0044410000000/0000400000000/"
        "0033430002000/0000000004440/0000000004000/0000000004000/"
        "0000000000000/0000040000000/0000040000000/0004442000000/"
        "0000000000000",
        "0000000000000/0000200000000/0044410000000/0000400000000/"
        "0033430002000/0000000014440/0000000004000/0000000034330/"
        "0000000000000/0003040000000/0003040000000/0004442000000/"
        "0003010000000",
    ),
    (
        "0000000000000/0010120000000/0444440000400/0303030000400/"
        "0000000000400/0000000000400/0000000000420/0000000000000/"
        "0004444400000/0002000000000/0000000000000/0000000000000/"
        "0000000000000",
        "0000000000000/0010120000000/0444440003400/0303030000410/"
        "0000000003400/0000000000410/0000000003420/0003030300000/"
        "0004444400000/0002101000000/0000000000000/0000000000000/"
        "0000000000000",
    ),
    (
        "0000000000000/0004200000000/0144400000000/0034000000000/"
        "0001000000000/0000000240000/0000000444000/0000000040000/"
        "0000000000000/0000400000000/0004440000000/0002400000000/"
        "0000000000000",
        "0000000000000/0004200000000/0144400000000/0034000000000/"
        "0001000000000/0000000240000/0000000444100/0000000043000/"
        "0000100010000/0000430000000/0004441000000/0002400000000/"
        "0000000000000",
    ),
    (
        "000000000000000/000000000000000/001300000000000/"
        "004420000004400/004043000040400/000441000044200/"
        "000000000000000/000000000000000/000000000000000/"
        "044000002440000/040400004040000/024400004400000/"
        "000000000000000/000000000000000",
        "000000000000000/000000000000000/001300000000000/"
        "004420000004410/004043000040430/000441000044200/"
        "000000000013000/000000000000000/000000000310000/"
        "144000002440000/340400034040000/024400014400000/"
        "003100000000000/000000000000000",
    ),
)

_SOURCE_PAIR_KEYS = {
    (_grid_key(_parse_grid(inp)), _grid_key(_parse_grid(out)))
    for inp, out in _SOURCE_TEXT_PAIRS
}


def _transform_point(row, col, height, width, transform):
    if transform == "r0":
        return row, col
    if transform == "r90":
        return col, height - 1 - row
    if transform == "r180":
        return height - 1 - row, width - 1 - col
    if transform == "r270":
        return width - 1 - col, row
    if transform == "flr":
        return row, width - 1 - col
    if transform == "ftb":
        return height - 1 - row, col
    if transform == "diag":
        return col, row
    if transform == "anti":
        return width - 1 - col, height - 1 - row
    raise ValueError("unknown dihedral transform")


def _transform_map(mapping, height, width, transform):
    return {
        _transform_point(row, col, height, width, transform): color
        for (row, col), color in mapping.items()
    }


def _validate_count(
    name,
    value,
    lower,
    upper,
):
    if value is None:
        return None

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(
            f"{name} must be an integer or None"
        )

    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(
            f"{name} is outside its legal range"
        )
    return value


def _resolve_counts(
    seed,
    num_train,
    num_test,
):
    num_train = _validate_count(
        "num_train",
        num_train,
        2,
        10,
    )
    num_test = _validate_count(
        "num_test",
        num_test,
        1,
        4,
    )

    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        lower = max(2, num_test + 1)
        num_train = (
            lower
            + (seed // 4) % (11 - lower)
        )
    elif num_train is None:
        lower = max(2, num_test + 1)
        num_train = (
            lower
            + seed % (11 - lower)
        )
    elif num_test is None:
        upper = min(4, num_train - 1)
        if upper < 1:
            raise ValueError(
                "no legal num_test exists for num_train"
            )
        num_test = 1 + seed % upper

    if num_train <= num_test:
        raise ValueError(
            "num_train must be greater than num_test"
        )

    return num_train, num_test


def _normal_map(mapping):
    top = min(r for r, _ in mapping)
    left = min(c for _, c in mapping)
    return {(r - top, c - left): int(v) for (r, c), v in mapping.items()}, (top, left)


def _signature(mapping):
    return tuple(sorted((r, c, int(v)) for (r, c), v in mapping.items()))


def _turn(mapping, height, width, operation):
    return _normal_map(_transform_map(mapping, height, width, operation))[0]


def _neighbors(cell):
    r, c = cell
    return ((r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if dr or dc)


def _sample_scaffold(rng, height, width, mode):
    if mode == 'line':
        length = max(height, width)
        cells = {(0, c) for c in range(length)}
    elif mode == 'cross':
        side = max(3, min(height, width))
        mid = side // 2
        cells = {(mid, c) for c in range(side)} | {(r, mid) for r in range(side)}
    elif mode == 'elbow':
        cells = {(0, c) for c in range(width)} | {(r, 0) for r in range(height)}
    elif mode == 'open':
        cells = {(0, c) for c in range(width)} | {(r, 0) for r in range(height)} | {(r, width - 1) for r in range(1, height)}
    elif mode == 'ring':
        cells = {(r, c) for r in range(height) for c in range(width)
                 if r in (0, height - 1) or c in (0, width - 1)}
    else:
        target = max(4, min(height * width - 1,
                            int(round(height * width * (0.35 if mode == 'branch' else 0.7)))))
        cells = {(int(rng.integers(height)), int(rng.integers(width)))}
        while len(cells) < target:
            frontier = sorted({(r + dr, c + dc) for r, c in cells
                               for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                               if 0 <= r + dr < height and 0 <= c + dc < width
                               and (r + dr, c + dc) not in cells})
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])
    return cells


def _sample_motif(rng, colors, force_asymmetric=False):
    scaffold_color, marker_color = colors[1:3]
    detail_palette = colors[3:]
    modes = ('line', 'cross', 'elbow', 'open', 'ring', 'branch', 'dense')
    for _ in range(160):
        mode = modes[int(rng.integers(len(modes)))]
        height = int(rng.integers(3, 7))
        width = int(rng.integers(3, 7))
        if mode == 'line':
            height, width = 1, int(rng.integers(3, 8))
        elif mode == 'cross':
            height = width = int(rng.integers(3, 6))
        elif mode == 'ring':
            height = max(3, height)
            width = max(3, width)
        cells = _sample_scaffold(rng, height, width, mode)
        if len(cells) < 4:
            continue
        cells, _ = _normal_map({point: 1 for point in cells})
        cells = set(cells)
        height = max(r for r, _ in cells) + 1
        width = max(c for _, c in cells) + 1
        marker = (height // 2, width // 2) if mode == 'cross' else sorted(cells)[int(rng.integers(len(cells)))]
        if marker not in cells:
            marker = sorted(cells)[int(rng.integers(len(cells)))]
        retained = {point: (marker_color if point == marker else scaffold_color) for point in cells}
        retained_signatures = {}
        stabilizers = []
        for operation in _D4:
            moved = _turn(retained, height, width, operation)
            sig = _signature(moved)
            retained_signatures[operation] = sig
            if sig == _signature(retained):
                stabilizers.append(operation)
        if force_asymmetric and len(stabilizers) != 1:
            continue
        candidates = {(r, c) for r in range(-1, height + 1)
                      for c in range(-1, width + 1) if (r, c) not in cells
                      and any(n in cells for n in _neighbors((r, c)))}
        # Every selected orbit is invariant under the visible stencil's symmetries.
        orbits = []
        remaining = set(candidates)
        while remaining:
            point = min(remaining)
            orbit = set()
            for operation in stabilizers:
                moved = _transform_map({point: 1}, height, width, operation)
                transformed_retained = _transform_map(retained, height, width, operation)
                _, shift = _normal_map(transformed_retained)
                for rr, cc in moved:
                    orbit.add((rr - shift[0], cc - shift[1]))
            if orbit <= candidates and orbit <= remaining:
                orbits.append(tuple(sorted(orbit)))
            remaining -= orbit
        if not orbits:
            continue
        internal_orbits = [orbit for orbit in orbits if all(
            0 <= r < height and 0 <= c < width for r, c in orbit)]
        if internal_orbits and int(rng.integers(5)) == 0:
            orbits = internal_orbits
        rng.shuffle(orbits)
        target = (int(rng.integers(2, 6)) if int(rng.integers(5)) < 3
                  else int(rng.integers(6, 11)))
        details = {}
        for orbit in orbits:
            if len(details) + len(orbit) > max(target + 1, 4) and len(details) >= 2:
                continue
            color = detail_palette[len(details) % len(detail_palette)] if len(details) < len(detail_palette) else detail_palette[int(rng.integers(len(detail_palette)))]
            for point in orbit:
                details[point] = color
            if len(details) >= target:
                break
        if not details or len(details) > 12:
            continue
        if len(details) == 1 and len(detail_palette) > 1:
            continue
        # Validity is about the completed colored map, not the transform label.
        completions = {}
        for operation in _D4:
            moved_retained = _transform_map(retained, height, width, operation)
            _, shift = _normal_map(moved_retained)
            moved_details = {(r - shift[0], c - shift[1]): v for (r, c), v in
                             _transform_map(details, height, width, operation).items()}
            sig = retained_signatures[operation]
            if sig in completions and completions[sig] != moved_details:
                break
            completions[sig] = moved_details
        else:
            return retained, details, height, width, mode
    raise RuntimeError('could not sample a recoverable motif')


def _pair_colors(rng, previous=None):
    for _ in range(30):
        detail_count = int(rng.choice((1, 2, 3, 4), p=(0.17, 0.61, 0.19, 0.03)))
        colors = tuple(int(x) for x in rng.choice(10, size=3 + detail_count, replace=False))
        if colors != previous:
            return colors
    raise RuntimeError('could not vary palette')


def _make_pair(rng, regime, colors, operations, force_asymmetric=False):
    background = colors[0]
    for _ in range(100):
        retained, details, height, width, mode = _sample_motif(rng, colors, force_asymmetric)
        objects = []
        for index, operation in enumerate(('r0',) + tuple(operations)):
            moved_retained = _transform_map(retained, height, width, operation)
            input_map, shift = _normal_map(moved_retained)
            moved_details = _transform_map(details, height, width, operation)
            output_map = dict(input_map)
            output_map.update({(r - shift[0], c - shift[1]): v for (r, c), v in moved_details.items()})
            if index == 0:
                input_map = dict(output_map)
            objects.append((input_map, output_map))
        count = len(objects)
        max_span = max(max(max(r for r, _ in obj) - min(r for r, _ in obj) + 1,
                           max(c for _, c in obj) - min(c for _, c in obj) + 1)
                       for _, obj in objects)
        low = max(8, min(24, int(np.ceil(np.sqrt(count) * (max_span + 1)))))
        if regime == 'target_test':
            low = min(26, low + int(rng.integers(0, 3)))
        canvas_height = int(rng.integers(low, min(30, low + 9) + 1))
        canvas_width = int(rng.integers(low, min(30, low + 9) + 1))
        if int(rng.integers(12)) == 0:
            if int(rng.integers(2)):
                canvas_height = 30
            else:
                canvas_width = 30
        for _ in range(5):
            placed = []
            occupied = set()
            indices = rng.permutation(count)
            for idx in indices:
                input_map, output_map = objects[int(idx)]
                rows = [r for r, _ in output_map]
                cols = [c for _, c in output_map]
                row_low, row_high = -min(rows), canvas_height - 1 - max(rows)
                col_low, col_high = -min(cols), canvas_width - 1 - max(cols)
                if row_high < row_low or col_high < col_low:
                    break
                for _ in range(300):
                    top = int(rng.integers(row_low, row_high + 1))
                    left = int(rng.integers(col_low, col_high + 1))
                    full = {(top + r, left + c) for r, c in output_map}
                    if full.isdisjoint(occupied) and all((rr, cc) not in occupied for r, c in full
                           for rr, cc in _neighbors((r, c))):
                        placed.append((input_map, output_map, top, left))
                        occupied.update(full)
                        break
                else:
                    break
            if len(placed) == count:
                input_grid = np.full((canvas_height, canvas_width), background, dtype=np.int8)
                output_grid = input_grid.copy()
                for input_map, output_map, top, left in placed:
                    for (r, c), color in input_map.items():
                        input_grid[top + r, left + c] = color
                    for (r, c), color in output_map.items():
                        output_grid[top + r, left + c] = color
                if (np.any(input_grid != output_grid) and
                    np.count_nonzero(input_grid == background) > input_grid.size // 2):
                    return {'input': input_grid, 'output': output_grid}
            canvas_height = min(30, canvas_height + 2)
            canvas_width = min(30, canvas_width + 2)
    raise RuntimeError('could not place objects')


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    # The invariant convention is local prototype transfer; color identities are visible per pair.
    forced_turn = ('r90', 'r270')[int(rng.integers(2))]
    forced_reflection = ('flr', 'ftb', 'diag', 'anti')[int(rng.integers(4))]
    puzzle = {'train': [], 'test': []}
    used = set()
    previous_colors = None
    for section, amount, regime in (('train', num_train, 'source_train'),
                                    ('test', num_test, 'target_test')):
        for pair_index in range(amount):
            for _ in range(80):
                colors = _pair_colors(rng, previous_colors)
                if section == 'train' and pair_index == 0:
                    operations = (forced_turn, forced_reflection)
                else:
                    ncopy = int(rng.integers(1, 4 if section == 'train' else 5))
                    operations = tuple(_D4[int(rng.integers(8))] for _ in range(ncopy))
                pair = _make_pair(rng, regime, colors, operations,
                                  force_asymmetric=(section == 'train' and pair_index == 0))
                key = _grid_key(pair['input'])
                if key not in used and (key, _grid_key(pair['output'])) not in _SOURCE_PAIR_KEYS:
                    break
            else:
                raise RuntimeError('could not sample distinct pair')
            used.add(key)
            previous_colors = colors
            puzzle[section].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _components(grid, background):
    height, width = grid.shape
    seen = set()
    result = []
    for row in range(height):
        for col in range(width):
            if grid[row, col] == background or (row, col) in seen:
                continue
            stack = [(row, col)]
            seen.add((row, col))
            component = set()
            while stack:
                cell = stack.pop()
                component.add(cell)
                for rr, cc in _neighbors(cell):
                    if (0 <= rr < height and 0 <= cc < width and
                        (rr, cc) not in seen and grid[rr, cc] != background):
                        seen.add((rr, cc))
                        stack.append((rr, cc))
            result.append(component)
    return result


def _analyze_pair(input_grid, output_grid, regime):
    assert input_grid.shape == output_grid.shape
    counts = np.bincount(input_grid.ravel().astype(int), minlength=10)
    background = int(np.argmax(counts))
    assert counts[background] > input_grid.size // 2
    components = _components(input_grid, background)
    assert 2 <= len(components) <= 5
    maps = [{point: int(input_grid[point]) for point in comp} for comp in components]
    color_sets = [set(mapping.values()) for mapping in maps]
    shared = set.intersection(*color_sets)
    assert len(shared) == 2
    marker_candidates = [color for color in shared if all(list(mapping.values()).count(color) == 1
                                                          for mapping in maps)]
    assert len(marker_candidates) == 1
    marker = marker_candidates[0]
    scaffold = next(iter(shared - {marker}))
    assert all(list(mapping.values()).count(scaffold) >= 3 for mapping in maps)
    prototypes = [i for i, colors in enumerate(color_sets) if colors > shared]
    assert len(prototypes) == 1
    prototype = maps[prototypes[0]]
    assert all(colors == shared for i, colors in enumerate(color_sets) if i != prototypes[0])
    details_abs = {point: value for point, value in prototype.items() if value not in shared}
    retained_abs = {point: value for point, value in prototype.items() if value in shared}
    retained, proto_origin = _normal_map(retained_abs)
    details = {(r - proto_origin[0], c - proto_origin[1]): v
               for (r, c), v in details_abs.items()}
    height = max(r for r, _ in retained) + 1
    width = max(c for _, c in retained) + 1
    assert len(details) >= 1
    expected = input_grid.copy()
    transformations = []
    ambiguous = 0
    outside = 0
    origins = []
    for i, component in enumerate(maps):
        if i == prototypes[0]:
            continue
        target, origin = _normal_map(component)
        origins.append(origin)
        matches = []
        for operation in _D4:
            moved_retained = _transform_map(retained, height, width, operation)
            aligned, shift = _normal_map(moved_retained)
            if aligned == target:
                moved_details = _transform_map(details, height, width, operation)
                aligned_details = {(r - shift[0], c - shift[1]): v
                                   for (r, c), v in moved_details.items()}
                matches.append((operation, aligned_details))
        assert matches
        assert all(m[1] == matches[0][1] for m in matches)
        transformations.append(tuple(m[0] for m in matches))
        ambiguous += len(matches) > 1
        detail_map = matches[0][1]
        outside += any(not (0 <= r < max(x for x, _ in target) + 1 and
                           0 <= c < max(y for _, y in target) + 1)
                       for r, c in detail_map)
        for (r, c), color in detail_map.items():
            rr, cc = origin[0] + r, origin[1] + c
            assert 0 <= rr < input_grid.shape[0] and 0 <= cc < input_grid.shape[1]
            assert expected[rr, cc] == background
            expected[rr, cc] = color
    assert np.array_equal(expected, output_grid)
    assert len(_components(output_grid, background)) == len(components)
    assert np.any(input_grid != output_grid)
    assert np.all(input_grid[input_grid != output_grid] == background)
    assert set(int(x) for x in np.unique(input_grid)) == {background, scaffold, marker} | set(details.values())
    return {'roles': (background, scaffold, marker, tuple(sorted(set(details.values())))),
            'shape': input_grid.shape, 'count': len(components),
            'transforms': tuple(transformations), 'ambiguous': ambiguous,
            'outside': outside, 'detail_count': len(details),
            'retained_count': len(retained), 'motif_dims': (height, width),
            'structure': (_signature(retained), _signature(details)),
            'layout': tuple(sorted(origins))}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4 and len(puzzle['train']) > len(puzzle['test'])
    inputs, pairs, palettes, train_infos = set(), set(), set(), []
    for section, regime in (('train', 'source_train'), ('test', 'target_test')):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in (pair['input'], pair['output']):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            key = _grid_key(pair['input'])
            pair_key = (key, _grid_key(pair['output']))
            assert key not in inputs and pair_key not in pairs and pair_key not in _SOURCE_PAIR_KEYS
            inputs.add(key)
            pairs.add(pair_key)
            info = _analyze_pair(pair['input'], pair['output'], regime)
            palettes.add(info['roles'])
            if section == 'train':
                train_infos.append(info)
    assert len(palettes) >= 2
    # A non-symmetric training stencil demonstrates both a turn and a reflection.
    assert any(len(info['transforms']) >= 2 and
               any('r90' in group or 'r270' in group for group in info['transforms']) and
               any(any(t in ('flr', 'ftb', 'diag', 'anti') for t in group)
                   for group in info['transforms']) for info in train_infos)


def _puzzle_key(puzzle):
    return tuple((_grid_key(pair['input']), _grid_key(pair['output']))
                 for section in ('train', 'test') for pair in puzzle[section])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    measures = {name: 0 for name in (
        'external', 'ambiguous', 'external_ambiguous', 'sparse',
        'many_details', 'repeated_colors', 'one_color', 'three_colors',
        'small', 'large', 'square', 'central', 'one_copy', 'four_copies',
        'repeated_orientation', 'same_side', 'shared_orientation_puzzles',
        'heldout_orientation_puzzles', 'varying_palettes')}
    for puzzle in puzzles:
        train_ops, test_ops, palettes = set(), set(), set()
        for section in ('train', 'test'):
            for pair in puzzle[section]:
                info = _analyze_pair(pair['input'], pair['output'], section)
                palettes.add(info['roles'])
                operations = {op for group in info['transforms'] for op in group}
                if section == 'train':
                    train_ops.update(operations)
                else:
                    test_ops.update(operations)
                measures['external'] += info['outside'] > 0
                measures['ambiguous'] += info['ambiguous'] > 0
                measures['external_ambiguous'] += info['outside'] > 0 and info['ambiguous'] > 0
                measures['sparse'] += info['retained_count'] / (info['motif_dims'][0] * info['motif_dims'][1]) <= 0.6
                measures['many_details'] += info['detail_count'] > 5
                measures['repeated_colors'] += info['detail_count'] > len(info['roles'][3])
                measures['one_color'] += len(info['roles'][3]) == 1
                measures['three_colors'] += len(info['roles'][3]) >= 3
                measures['small'] += max(info['shape']) <= 14
                measures['large'] += max(info['shape']) >= 25
                measures['square'] += info['shape'][0] == info['shape'][1]
                background = info['roles'][0]
                grid = pair['input']
                measures['central'] += (np.any(grid[grid.shape[0] // 2, :] != background)
                                        or np.any(grid[:, grid.shape[1] // 2] != background))
                measures['one_copy'] += info['count'] == 2
                measures['four_copies'] += info['count'] == 5
                measures['repeated_orientation'] += len(info['transforms']) > len(set(info['transforms']))
                centers = [sum(r for r, _ in comp) / len(comp)
                           for comp in _components(grid, background)]
                measures['same_side'] += (sum(r < grid.shape[0] / 2 for r in centers) >= 2
                                          or sum(r >= grid.shape[0] / 2 for r in centers) >= 2)
        measures['shared_orientation_puzzles'] += test_ops <= train_ops
        measures['heldout_orientation_puzzles'] += bool(test_ops - train_ops)
        measures['varying_palettes'] += len(palettes) >= 2
    assert measures['external'] >= 100 and measures['ambiguous'] >= 10
    assert measures['external_ambiguous'] >= 10 and measures['sparse'] >= 50
    assert measures['many_details'] >= 20 and measures['repeated_colors'] >= 100
    assert measures['one_color'] >= 10 and measures['three_colors'] >= 10
    assert measures['small'] >= 5 and measures['large'] >= 5 and measures['square'] >= 5
    assert measures['central'] >= 100 and measures['same_side'] >= 100
    assert measures['one_copy'] >= 20 and measures['four_copies'] >= 5
    assert measures['repeated_orientation'] >= 20
    assert measures['shared_orientation_puzzles'] >= 5
    assert measures['heldout_orientation_puzzles'] >= 5
    assert measures['varying_palettes'] == 42
    infos = [_analyze_pair(pair['input'], pair['output'], section)
             for puzzle in puzzles for section in ('train', 'test') for pair in puzzle[section]]
    assert len({info['shape'] for info in infos}) >= 25
    assert len({info['count'] for info in infos}) >= 4
    assert len({info['structure'] for info in infos}) >= 100
    assert len({info['layout'] for info in infos}) >= 100
    assert sum(info['outside'] > 0 for info in infos) > 50
    assert sum(info['ambiguous'] > 0 for info in infos) > 5
    assert any(info['detail_count'] > 5 for info in infos)
    assert any(len(info['roles'][3]) == 1 for info in infos)
    assert any(len(info['roles'][3]) >= 3 for info in infos)
    assert any(info['shape'][0] == 30 or info['shape'][1] == 30 for info in infos)
    assert any(info['shape'][0] != info['shape'][1] for info in infos)
    assert any(any('r0' in group for group in info['transforms']) for info in infos)
    assert any(any('diag' in group for group in info['transforms']) for info in infos)
    assert any(any('anti' in group for group in info['transforms']) for info in infos)
    explicit = generate_puzzle(314159, 7, 3)
    assert _puzzle_key(explicit) == _puzzle_key(generate_puzzle(314159, 7, 3))
    for ntrain, ntest in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(700 + ntrain * 10 + ntest, ntrain, ntest)
        assert len(puzzle['train']) == ntrain and len(puzzle['test']) == ntest
    for seed, ntrain, ntest in ((811, None, 1), (812, None, 4),
                                (813, 2, None), (814, 10, None)):
        puzzle = generate_puzzle(seed, ntrain, ntest)
        assert ntrain is None or len(puzzle['train']) == ntrain
        assert ntest is None or len(puzzle['test']) == ntest
    for kwargs in ({'num_train': 1}, {'num_train': 11}, {'num_test': 0},
                   {'num_test': 5}, {'num_train': 2, 'num_test': 2},
                   {'num_train': 3, 'num_test': 4}):
        try:
            generate_puzzle(123, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid counts accepted')
