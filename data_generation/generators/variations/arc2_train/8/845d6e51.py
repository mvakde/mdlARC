import numpy as np


_SHAPES = {
    "domino": ((0, 0), (0, 1)),
    "l_triomino": ((0, 0), (1, 0), (1, 1)),
    "t_tetromino": ((0, 0), (0, 1), (0, 2), (1, 1)),
    "u_pentomino": ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
    "plus_pentomino": ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
}


_SOURCE_INPUT_ROWS = (
    (
        "0204401110500300", "0204000100500330", "0000000000500300",
        "5555555555500000", "0000000000000000", "0300030000033300",
        "0330030000003000", "0000000030000000", "0000000030000000",
        "0330000000003300", "0000000000000300", "0333000300000000",
        "0030003300330000", "0000000000300030", "0003300000000330",
        "0000000000000030",
    ),
    (
        "0200110500000000", "2220100500333000", "0000000500030000",
        "5555555500000000", "0000000000000330", "0000000000000300",
        "0030000330000000", "0330000030000000", "0030000000000000",
        "0000000000000000", "0000000000030000", "0003000000333000",
        "0033000000000000", "0000003000000300", "0000003300003300",
        "0000003000000000",
    ),
    (
        "2200700500", "2007770500", "2200700500", "5555555500",
        "0030000000", "0333030300", "0030033300", "0000000000",
        "0003330000", "0003030000",
    ),
    (
        "02010044000500000", "22011040088500300", "00010044000500300",
        "55555555555500000", "00000000000000033", "00003000000000000",
        "33033300000000000", "30000000330000000", "00000000030003000",
        "00000000330003300", "00000000000003000", "00333000000000000",
        "00303000000000030", "00000000033300030", "00000000003000000",
        "00003300000000300", "33000000000003300", "03000000000000000",
    ),
)


def _normalize(cells):
    cells = [(int(r), int(c)) for r, c in cells]
    assert cells
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _rotations(cells):
    current = _normalize(cells)
    result = []
    for _ in range(4):
        current = _normalize(current)
        if current not in result:
            result.append(current)
        current = tuple((c, -r) for r, c in current)
    return tuple(result)


def _canonical(cells):
    return min(_rotations(cells))


def _components(mask):
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    result = []

    for start_r in range(height):
        for start_c in range(width):
            if not mask[start_r, start_c] or seen[start_r, start_c]:
                continue

            stack = [(start_r, start_c)]
            seen[start_r, start_c] = True
            component = []

            while stack:
                r, c = stack.pop()
                component.append((r, c))

                for dr, dc in (
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                ):
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and mask[nr, nc]
                        and not seen[nr, nc]
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            result.append(tuple(sorted(component)))

    return result


def _find_structure(grid):
    height, width = grid.shape
    counts = np.bincount(grid.ravel(), minlength=10)
    background = int(np.argmax(counts))
    candidates = []

    for separator_row in range(2, height - 1):
        for separator_col in range(2, width - 1):
            separator = int(grid[separator_row, 0])

            if separator == background:
                continue

            if not np.all(
                grid[separator_row, :separator_col + 1] == separator
            ):
                continue

            if not np.all(
                grid[:separator_row + 1, separator_col] == separator
            ):
                continue

            expected_count = separator_row + separator_col + 1
            if int(np.count_nonzero(grid == separator)) != expected_count:
                continue

            candidates.append(
                (
                    background,
                    separator,
                    separator_row,
                    separator_col,
                )
            )

    assert candidates
    longest = max(row + col + 1 for _, _, row, col in candidates)
    candidates = [candidate for candidate in candidates
                  if candidate[2] + candidate[3] + 1 == longest]
    assert len(candidates) == 1
    return candidates[0]


def _analyze_top_left(grid):
    (
        background,
        separator,
        separator_row,
        separator_col,
    ) = _find_structure(grid)

    height, width = grid.shape

    catalog_mask = np.zeros((height, width), dtype=bool)
    catalog_mask[:separator_row, :separator_col] = (
        grid[:separator_row, :separator_col] != background
    )

    prototype_components = _components(catalog_mask)
    assert 2 <= len(prototype_components) <= 4

    prototypes = []
    prototype_colors = set()
    prototype_canonicals = set()

    for coords in prototype_components:
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1

        color = next(iter(colors))
        assert color not in (background, separator)
        assert color not in prototype_colors

        footprint = _normalize(coords)
        canonical = _canonical(coords)

        assert canonical not in prototype_canonicals

        prototype_colors.add(color)
        prototype_canonicals.add(canonical)
        prototypes.append(
            {
                "coords": coords,
                "footprint": footprint,
                "canonical": canonical,
                "color": color,
            }
        )

    outside_catalog = np.ones((height, width), dtype=bool)
    outside_catalog[:separator_row, :separator_col] = False

    exterior_colors = {
        int(value)
        for value in grid[outside_catalog]
        if int(value) not in (background, separator)
    }
    assert len(exterior_colors) == 1

    target_color = next(iter(exterior_colors))
    assert target_color not in prototype_colors
    assert not np.any(
        grid[:separator_row, :separator_col] == target_color
    )

    target_mask = (grid == target_color) & outside_catalog
    target_components = _components(target_mask)
    assert target_components

    by_canonical = {
        prototype["canonical"]: prototype
        for prototype in prototypes
    }

    targets = []
    for coords in target_components:
        footprint = _normalize(coords)
        canonical = _canonical(coords)
        assert canonical in by_canonical

        targets.append(
            {
                "coords": coords,
                "footprint": footprint,
                "canonical": canonical,
            }
        )

    return {
        "background": background,
        "separator": separator,
        "target_color": target_color,
        "separator_row": separator_row,
        "separator_col": separator_col,
        "prototypes": prototypes,
        "prototype_by_canonical": by_canonical,
        "targets": targets,
        "target_mask": target_mask,
    }


def _analyze_input(grid):
    for turns in (0, 2):
        canonical_grid = np.rot90(grid, turns)
        try:
            analysis = _analyze_top_left(canonical_grid)
        except AssertionError:
            continue
        analysis['turns'] = turns
        analysis['target_mask'] = np.rot90(analysis['target_mask'], -turns)
        return analysis
    raise AssertionError('no unique corner catalog')


def _solve_grid(input_grid):
    analysis = _analyze_input(input_grid)
    output = np.rot90(input_grid, analysis['turns']).copy()

    for target in analysis["targets"]:
        prototype = analysis["prototype_by_canonical"][
            target["canonical"]
        ]
        label = prototype["color"]

        for r, c in target["coords"]:
            output[r, c] = label

    return np.rot90(output, -analysis['turns']).copy()


# Shapes are selected by rotation-independent geometry, never by a global name.
def _shape_pool():
    current = {((0, 0),)}
    result = []
    for size in range(2, 7):
        grown = set()
        for shape in current:
            occupied = set(shape)
            for r, c in shape:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    point = (r + dr, c + dc)
                    if point not in occupied:
                        grown.add(_canonical((*shape, point)))
        current = grown
        result.extend(sorted(grown))
    return tuple(result)


_SHAPE_POOL = _shape_pool()
_SMALL_SHAPES = tuple(shape for shape in _SHAPE_POOL
                      if len(shape) <= 4 and
                      max(r for r, _ in shape) < 3 and
                      max(c for _, c in shape) < 3)


def _rand_choice(rng, values):
    return values[int(rng.integers(len(values)))]


def _place_catalog(grid, shapes, labels, row, col, rng):
    # Randomized backtracking packs the colored exemplars with a one-cell moat.
    ordered = sorted(shapes, key=lambda s: -len(s))
    placed = {}
    occupied = set()
    budget = [6000]

    def recurse(index):
        if index == len(ordered):
            return True
        if budget[0] <= 0:
            return False
        shape = ordered[index]
        options = []
        for footprint in _rotations(shape):
            bh = max(r for r, _ in footprint) + 1
            bw = max(c for _, c in footprint) + 1
            for top in range(row - bh + 1):
                for left in range(col - bw + 1):
                    cells = frozenset((top + r, left + c) for r, c in footprint)
                    options.append((cells, footprint))
        for j in rng.permutation(len(options)):
            budget[0] -= 1
            cells, footprint = options[int(j)]
            if cells & occupied or any(
                (r + dr, c + dc) in occupied
                for r, c in cells
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ):
                continue
            occupied.update(cells)
            placed[shape] = (cells, footprint)
            if recurse(index + 1):
                return True
            occupied.difference_update(cells)
            del placed[shape]
        return False

    if not recurse(0):
        return None
    for shape, (cells, _) in placed.items():
        for r, c in cells:
            grid[r, c] = labels[shape]
    return {shape: footprint for shape, (_, footprint) in placed.items()}


def _target_options(grid, footprint, region, row, col, background):
    height, width = grid.shape
    bh = max(r for r, _ in footprint) + 1
    bw = max(c for _, c in footprint) + 1
    options = []
    for top in range(height - bh + 1):
        for left in range(width - bw + 1):
            if region == 'right' and not (top + bh <= row and left > col):
                continue
            if region == 'left' and not (top > row and left + bw <= col):
                continue
            if region == 'lower_right' and not (top > row and left > col):
                continue
            cells = frozenset((top + r, left + c) for r, c in footprint)
            if any(r < row and c < col for r, c in cells):
                continue
            if any(grid[r, c] != background for r, c in cells):
                continue
            if any(
                0 <= r + dr < height and 0 <= c + dc < width
                and (r + dr, c + dc) not in cells
                and grid[r + dr, c + dc] != background
                for r, c in cells
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ):
                continue
            options.append(cells)
    return options


def _place_targets(grid, specifications, row, col, background, target_color, rng):
    option_lists = [
        _target_options(grid, footprint, region, row, col, background)
        for _, footprint, region in specifications
    ]
    if any(not options for options in option_lists):
        return False
    order = sorted(range(len(specifications)), key=lambda i: len(option_lists[i]))
    for _ in range(32):
        occupied = set()
        chosen = []
        for index in order:
            feasible = [cells for cells in option_lists[index] if
                        not cells & occupied and not any(
                            (r + dr, c + dc) in occupied
                            for r, c in cells
                            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                        )]
            if not feasible:
                break
            cells = _rand_choice(rng, feasible)
            chosen.append(cells)
            occupied.update(cells)
        if len(chosen) == len(specifications):
            for cells in chosen:
                for r, c in cells:
                    grid[r, c] = target_color
            return True
    return False


def _pair_dimensions(rng, shapes, regime):
    n = len(shapes)
    small_ok = n == 2 and all(shape in _SMALL_SHAPES for shape in shapes)
    if small_ok and rng.random() < 0.13:
        return 10, 10
    if rng.random() < 0.12:
        return 16, 16
    band = int(rng.choice(3, p=(0.24, 0.43, 0.33) if regime == 'source_train'
                          else (0.18, 0.37, 0.45)))
    if band == 0 and small_ok:
        return int(rng.integers(10, 14)), int(rng.integers(10, 14))
    if band == 0:
        band = 1
    if band == 1:
        return int(rng.integers(14, 19)), int(rng.integers(13, 20))
    return int(rng.integers(19, 27)), int(rng.integers(17, 27))


def _target_count(rng, height, width, n, regime):
    area = height * width
    if area <= 170:
        low, high = 2, 6
    elif area <= 330:
        low, high = 3, 11
    else:
        low, high = 5, 16
    if regime == 'target_test':
        low = min(high, low + 1)
    if rng.random() < 0.13:
        high = min(high, max(low, 4))
    return int(rng.integers(low, high + 1))


def _make_pair(rng, regime, shapes, roles, require_right=False,
               require_rotation=False, require_all_targets=False,
               no_rotation=False, turns=0):
    background, separator, target_color = roles
    shapes = list(shapes)
    for attempt in range(70):
        height, width = _pair_dimensions(rng, shapes, regime)
        min_row = 3 if len(shapes) <= 2 else 4
        max_row = min(8, height - 5)
        min_col = 5 if len(shapes) <= 2 else 7
        max_col = min(15, width - 5)
        if min_row > max_row or min_col > max_col:
            continue
        if min_row == 3 and rng.random() < 0.23:
            row = 3
        else:
            row = int(rng.integers(min_row, max_row + 1))
        col = int(rng.integers(min_col, max_col + 1))
        grid = np.full((height, width), background, dtype=np.int8)
        grid[row, :col + 1] = separator
        grid[:row + 1, col] = separator
        colors = [int(v) for v in rng.permutation(
            [v for v in range(10) if v not in roles])[:len(shapes)]]
        labels = dict(zip(shapes, colors))
        footprints = _place_catalog(grid, shapes, labels, row, col, rng)
        if footprints is None:
            continue
        count = _target_count(rng, height, width, len(shapes), regime)
        # The catalog is allowed to contain unused classes. Uneven repetition
        # makes color counting an unreliable shortcut.
        active_count = (len(shapes) if require_all_targets
                        else int(rng.integers(1, len(shapes) + 1)))
        if count >= 5 and rng.random() < 0.7:
            active_count = max(2, active_count)
        active = [shapes[int(i)] for i in rng.choice(
            len(shapes), size=active_count, replace=False)]
        if count >= active_count and (require_all_targets or rng.random() < 0.65):
            target_shapes = list(active)
            target_shapes += [_rand_choice(rng, active)
                              for _ in range(count - len(active))]
        else:
            weights = rng.dirichlet(np.full(len(active), 0.9))
            target_shapes = [active[int(rng.choice(len(active), p=weights))]
                             for _ in range(count)]
        target_shapes = [target_shapes[int(i)] for i in rng.permutation(len(target_shapes))]
        specs = []
        regions = []
        if require_right or rng.random() < (0.53 if regime == 'source_train' else 0.62):
            regions.append('right')
        if rng.random() < 0.64:
            regions.append('left')
        if rng.random() < 0.60:
            regions.append('lower_right')
        if len(regions) > count:
            regions = regions[:count]
        rng.shuffle(regions)
        changed = False
        for i, shape in enumerate(target_shapes):
            choices = list(_rotations(shape))
            if no_rotation:
                choices = [footprints[shape]]
            if require_rotation and not changed and i == count - 1 and len(choices) > 1:
                choices = [v for v in choices if v != footprints[shape]]
            footprint = _rand_choice(rng, choices)
            changed |= footprint != footprints[shape]
            region = regions[i] if i < len(regions) else 'any'
            specs.append((shape, footprint, region))
        if require_rotation and not changed:
            rotatable = [i for i, shape in enumerate(target_shapes)
                         if len(_rotations(shape)) > 1]
            if not rotatable:
                continue
            i = int(_rand_choice(rng, rotatable))
            choices = [v for v in _rotations(target_shapes[i])
                       if v != footprints[target_shapes[i]]]
            specs[i] = (specs[i][0], _rand_choice(rng, choices), specs[i][2])
        if not _place_targets(grid, specs, row, col, background, target_color, rng):
            continue
        if _is_source_input(grid):
            continue
        grid = np.rot90(grid, -turns).copy()
        return grid, _solve_grid(grid)
    raise RuntimeError('unable to construct catalog and targets')


def _validate_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} out of range')
    return int(value)


def _resolve_counts(rng, num_train, num_test):
    num_train = _validate_count('num_train', num_train, 2, 10)
    num_test = _validate_count('num_test', num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return num_train, num_test


def _is_source_input(grid):
    rows = tuple(''.join(str(int(v)) for v in line) for line in grid)
    return rows in _SOURCE_INPUT_ROWS


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def _pair_features(pair):
    analysis = _analyze_input(pair['input'])
    row, col = analysis['separator_row'], analysis['separator_col']
    targets = analysis['targets']
    return {
        'height': pair['input'].shape[0], 'width': pair['input'].shape[1],
        'classes': len(analysis['prototypes']), 'targets': len(targets),
        'row': row, 'col': col,
        'right': sum(all(r < row and c > col for r, c in t['coords']) for t in targets),
        'left': sum(all(r > row and c < col for r, c in t['coords']) for t in targets),
        'lower_right': sum(all(r > row and c > col for r, c in t['coords']) for t in targets),
        'rotated': sum(t['footprint'] != analysis['prototype_by_canonical'][t['canonical']]['footprint']
                       for t in targets),
        'unused': len(set(p['canonical'] for p in analysis['prototypes']) -
                      set(t['canonical'] for t in targets)),
        'shapes': frozenset(p['canonical'] for p in analysis['prototypes']),
        'colors': frozenset(p['color'] for p in analysis['prototypes']),
        'corner': 'upper_left' if analysis['turns'] == 0 else 'lower_right',
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen = set()
    roles = set()
    train_features = []
    test_features = []
    train_maps = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = pair['input'], pair['output']
            for grid in (a, b):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((grid >= 0) & (grid <= 9))
            assert a.shape == b.shape and not np.array_equal(a, b)
            key = _grid_key(a)
            assert key not in seen and not _is_source_input(a)
            seen.add(key)
            analysis = _analyze_input(a)
            roles.add((analysis['background'], analysis['separator'], analysis['target_color']))
            assert np.array_equal(_solve_grid(a), b)
            assert np.array_equal(a != b, analysis['target_mask'])
            assert 2 <= len(analysis['prototypes']) <= 4
            assert len(analysis['targets']) >= 1
            feature = _pair_features(pair)
            if split == 'train':
                train_features.append(feature)
                train_maps.append({p['canonical']: p['color'] for p in analysis['prototypes']})
            else:
                test_features.append(feature)
    assert len(roles) == 1
    # The first two demonstrations include rotations. All class assignments
    # are read locally, so a test-only exemplar needs no learned color map.
    assert train_features[0]['rotated'] >= 1
    assert train_features[1]['rotated'] >= 1
    for pair in puzzle['train'][:2]:
        analysis = _analyze_input(pair['input'])
        assert {p['canonical'] for p in analysis['prototypes']} == {
            t['canonical'] for t in analysis['targets']}
    assert train_features[0]['corner'] == 'upper_left'
    assert any(f['right'] for f in train_features)
    assert any(train_maps[0][shape] != train_maps[1][shape]
               for shape in train_maps[0].keys() & train_maps[1].keys())
    if any(f['corner'] == 'lower_right' for f in test_features):
        assert any(f['corner'] == 'lower_right' for f in train_features)
    assert all(f['classes'] in (2, 3, 4) for f in test_features)
    assert all(f['shapes'] <= set(_SHAPE_POOL) for f in train_features + test_features)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    roles = tuple(int(v) for v in rng.permutation(10)[:3])
    source_shapes = tuple(_canonical(s) for s in _SHAPES.values())
    anchor_pool = [s for s in source_shapes if s in _SMALL_SHAPES]
    anchors = [anchor_pool[int(i)] for i in rng.choice(
        len(anchor_pool), size=2, replace=False)]
    pool = [s for s in _SHAPE_POOL if s not in anchors]
    vocabulary = list(anchors)
    extra = int(rng.integers(2, 5))
    vocabulary.extend(pool[int(i)] for i in rng.choice(len(pool), size=extra, replace=False))
    train_subsets = [list(anchors), list(anchors)]
    for i in range(2, num_train):
        n = int(rng.integers(2, 5))
        selected = [vocabulary[int(j)] for j in rng.choice(len(vocabulary), size=n, replace=False)]
        train_subsets.append(selected)
    test_subsets = []
    test_turns = []
    for _ in range(num_test):
        n = int(rng.integers(2, 5))
        if rng.random() < 0.22:
            selected = list(anchors)
        else:
            selected = [vocabulary[int(j)] for j in rng.choice(len(vocabulary), size=n, replace=False)]
        if rng.random() < 0.40:
            novel = _rand_choice(rng, [s for s in _SHAPE_POOL if s not in vocabulary])
            selected[int(rng.integers(len(selected)))] = novel
        test_subsets.append(selected)
        test_turns.append(2 if rng.random() < 0.23 else 0)
    train_turns = [0] + [2 if rng.random() < 0.22 else 0
                         for _ in range(num_train - 1)]
    train_no_rotation = [False, False] + [rng.random() < 0.15
                                         for _ in range(num_train - 2)]
    test_no_rotation = [rng.random() < 0.20 for _ in range(num_test)]
    if 2 in test_turns:
        train_turns[1] = 2
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max,
                              size=num_train + num_test, dtype=np.int64)
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, subsets in (('train', train_subsets), ('test', test_subsets)):
        for i, subset in enumerate(subsets):
            global_index = i if split == 'train' else num_train + i
            for retry in range(12):
                pair_rng = np.random.default_rng(np.random.SeedSequence(
                    [int(pair_seeds[global_index]), retry]))
                try:
                    a, b = _make_pair(pair_rng, 'source_train' if split == 'train'
                                      else 'target_test', subset, roles,
                                      require_right=(split == 'train' and i == 0),
                                      require_rotation=(split == 'train' and i < 2),
                                      require_all_targets=(split == 'train' and i < 2),
                                      no_rotation=(train_no_rotation[i] if split == 'train'
                                                   else test_no_rotation[i]),
                                      turns=(train_turns[i] if split == 'train'
                                             else test_turns[i]))
                except RuntimeError:
                    continue
                key = _grid_key(a)
                if split == 'train' and i == 1:
                    first = _analyze_input(puzzle['train'][0]['input'])
                    second = _analyze_input(a)
                    fm = {p['canonical']: p['color'] for p in first['prototypes']}
                    sm = {p['canonical']: p['color'] for p in second['prototypes']}
                    if not any(fm[s] != sm[s] for s in fm.keys() & sm.keys()):
                        continue
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append({'input': a, 'output': b})
                    break
            else:
                raise RuntimeError(f'pair construction failed: {split} {i}')
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s])
        for k in ('input', 'output')) for s in ('train', 'test'))


def self_test() -> None:
    # A prototype can form a shorter L at the grid corner; the divider wins.
    validate_generated_puzzle(generate_puzzle(7684698557379658648, None, 3))
    for args in ((2, 1), (10, 1), (10, 4), (None, 1),
                 (None, 4), (2, None), (10, None), (6, 3)):
        puzzle = generate_puzzle(19480 + (args[0] or 0) * 10 + (args[1] or 0), *args)
        validate_generated_puzzle(puzzle)
        assert _puzzles_equal(puzzle, generate_puzzle(
            19480 + (args[0] or 0) * 10 + (args[1] or 0), *args))
    for args in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4)):
        try:
            generate_puzzle(0, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f'illegal counts accepted: {args}')
    features = []
    train_counts, test_counts, signatures, roles, shape_sets = set(), set(), set(), set(), set()
    right_train, no_right_test, unused, dense, novel_test, simple_test = 0, 0, 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        signatures.add(tuple(_grid_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        tr = [_pair_features(p) for p in puzzle['train']]
        te = [_pair_features(p) for p in puzzle['test']]
        features.extend(tr + te)
        right_train += sum(f['right'] > 0 for f in tr)
        no_right_test += sum(f['right'] == 0 for f in te)
        simple_test += sum(f['rotated'] == 0 for f in te)
        unused += sum(f['unused'] > 0 for f in tr + te)
        dense += sum(f['targets'] >= 8 for f in tr + te)
        train_shapes = set().union(*(f['shapes'] for f in tr))
        novel_test += sum(bool(f['shapes'] - train_shapes) for f in te)
        a = _analyze_input(puzzle['train'][0]['input'])
        roles.add((a['background'], a['separator'], a['target_color']))
        shape_sets.add(tuple(sorted(train_shapes)))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) == 42 and len(roles) >= 20 and len(shape_sets) >= 15
    assert len(set((f['height'], f['width']) for f in features)) >= 40
    assert len(set((f['row'], f['col']) for f in features)) >= 15
    assert {f['classes'] for f in features} == {2, 3, 4}
    assert min(f['targets'] for f in features) <= 3 and max(f['targets'] for f in features) >= 10
    assert right_train >= 35 and no_right_test >= 8
    assert unused >= 10 and dense >= 15 and novel_test >= 5
    assert simple_test >= 5
