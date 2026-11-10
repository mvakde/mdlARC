import numpy as np


_SOURCE_INPUT_ROWS = frozenset({
    ("008040", "800124", "880010"),
    ("204080", "030888", "000080"),
    ("20040080", "02408888", "04200080", "40020080"),
    ("3001", "0220", "0220", "3003", "0880", "8888", "8008", "8888"),
})


def _require(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        low = max(2, test + 1)
        train = low + (seed // 4) % (11 - low)
    elif train is None:
        low = max(2, test + 1)
        train = low + seed % (11 - low)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _fingerprint(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _source_copy(grid):
    return tuple("".join(str(int(x)) for x in row) for row in grid) in _SOURCE_INPUT_ROWS


def _stencil(rng, n):
    # Geometry and density are independent of the train/test split.
    mode = int(rng.integers(0, 9))
    mask = np.zeros((n, n), dtype=bool)
    if mode == 0:  # irregular, with a broad density range
        count = int(rng.integers(2, n * n))
        mask.flat[rng.choice(n * n, count, replace=False)] = True
    elif mode == 1:  # cross
        r, c = int(rng.integers(n)), int(rng.integers(n))
        mask[r, :] = True
        mask[:, c] = True
    elif mode == 2:  # bar, sometimes thick
        width = int(rng.integers(1, max(2, n // 2 + 1)))
        start = int(rng.integers(0, n - width + 1))
        if rng.integers(2):
            mask[start:start + width, :] = True
        else:
            mask[:, start:start + width] = True
    elif mode == 3:  # T or L
        mask[int(rng.integers(n)), :] = True
        c = int(rng.integers(n))
        mask[:int(rng.integers(1, n + 1)), c] = True
    elif mode == 4:  # diagonal strokes
        for r in range(n):
            mask[r, r] = True
            if rng.integers(2) and r + 1 < n:
                mask[r, r + 1] = True
        if rng.integers(2):
            mask = np.fliplr(mask)
    elif mode == 5:  # enclosed gap
        mask[:] = True
        if n >= 3:
            mask[1:-1, 1:-1] = False
        else:
            mask[0, 0] = False
    elif mode == 6:  # near solid, with holes
        mask[:] = True
        holes = int(rng.integers(1, max(2, n * n // 3 + 1)))
        mask.flat[rng.choice(n * n, holes, replace=False)] = False
    elif mode == 7:  # connected walk
        r, c = int(rng.integers(n)), int(rng.integers(n))
        for _ in range(int(rng.integers(n + 1, n * n * 3 + 1))):
            mask[r, c] = True
            dr, dc = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
            r, c = min(n - 1, max(0, r + dr)), min(n - 1, max(0, c + dc))
    else:  # reflection symmetry
        left = rng.random((n, (n + 1) // 2)) < float(rng.uniform(.3, .8))
        mask[:, :(n + 1) // 2] = left
        mask[:, n // 2:] = np.fliplr(left)
    if mask.all() or not mask.any():
        mask[0, 0] = not mask[0, 0]
    return mask


def _layout(rng, n, background, palette, allow_single, allow_full):
    # Occupancy is sampled separately from the stencil. The layout mode
    # organizes both blank cells and color placements.
    mode = int(rng.integers(9))
    fill = float(rng.choice(np.array([.18, .32, .5, .7, .88, 1.])))
    if not allow_full and fill == 1.:
        fill = .88
    palette_count = 1 if allow_single and rng.integers(6) == 0 else int(rng.integers(2, min(5, len(palette)) + 1))
    if fill == 1.:
        palette_count = max(3, palette_count)
    chosen = np.asarray(rng.choice(np.asarray(palette), palette_count, replace=False), dtype=np.int8)
    layout = np.full((n, n), background, dtype=np.int8)
    if mode == 0:  # independent colors
        layout = np.where(rng.random((n, n)) < fill, rng.choice(chosen, (n, n)), background).astype(np.int8)
    elif mode == 1:  # copied rows
        row = np.where(rng.random(n) < fill, rng.choice(chosen, n), background)
        layout[:] = row
        for r in range(n):
            if rng.integers(3) == 0:
                layout[r] = np.where(rng.random(n) < fill, rng.choice(chosen, n), background)
    elif mode == 2:  # copied columns
        col = np.where(rng.random(n) < fill, rng.choice(chosen, n), background)
        layout[:] = col[:, None]
        for c in range(n):
            if rng.integers(3) == 0:
                layout[:, c] = np.where(rng.random(n) < fill, rng.choice(chosen, n), background)
    elif mode == 3:  # mirrored colors
        half = np.where(rng.random((n, (n + 1) // 2)) < fill,
                        rng.choice(chosen, (n, (n + 1) // 2)), background)
        layout[:, :(n + 1) // 2] = half
        layout[:, n // 2:] = np.fliplr(half)
        if rng.integers(2):
            layout = layout.T.copy()
    elif mode == 4:  # alternating stripes
        first = np.where(rng.random(n) < fill, rng.choice(chosen, n), background)
        second = np.where(rng.random(n) < fill, rng.choice(chosen, n), background)
        for i in range(n):
            layout[i] = first if i % 2 == 0 else second
        if rng.integers(2):
            layout = layout.T.copy()
    elif mode == 5:  # colored diagonal or anti-diagonal
        layout = np.where(rng.random((n, n)) < fill * .65, rng.choice(chosen, (n, n)), background).astype(np.int8)
        for i in range(n):
            layout[i, i if rng.integers(2) else n - 1 - i] = chosen[i % len(chosen)]
    elif mode == 6:  # compact group plus scattered cells
        r, c = int(rng.integers(n)), int(rng.integers(n))
        rr, cc = np.indices((n, n))
        prob = np.maximum(.06, fill - .18 * (abs(rr - r) + abs(cc - c)))
        layout = np.where(rng.random((n, n)) < prob, rng.choice(chosen, (n, n)), background).astype(np.int8)
    elif mode == 7:  # alternating color bands
        axis = int(rng.integers(2))
        for i in range(n):
            band = np.where(rng.random(n) < fill, chosen[i % len(chosen)], background)
            if axis:
                layout[:, i] = band
            else:
                layout[i] = band
    else:  # structured half and independent half
        layout = np.where(rng.random((n, n)) < fill, rng.choice(chosen, (n, n)), background).astype(np.int8)
        for r in range(1, n, 2):
            layout[r, :n // 2] = layout[r - 1, :n // 2]
    return layout


def _make_layout(rng, n, background, palette, stable_marker, force_clear):
    for _ in range(80):
        layout = _layout(rng, n, background, palette, stable_marker and not force_clear,
                         stable_marker and not force_clear)
        colors = set(int(x) for x in np.unique(layout)) - {background}
        if len(colors) < (2 if force_clear or not stable_marker else 1):
            continue
        if (force_clear or not stable_marker) and background not in layout:
            continue
        if background not in layout and len(colors) < 3:
            continue
        if len(colors) == 1 and background not in layout:
            continue
        return layout
    # Bounded fallback for small panels and restrictive palettes.
    layout = np.full((n, n), background, dtype=np.int8)
    colors = rng.choice(np.asarray(palette), 2, replace=False)
    layout[0, 0], layout[-1, -1] = colors
    return layout


def _render(layout, stencil, background):
    n = len(stencil)
    out = np.full((n * n, n * n), background, dtype=np.int8)
    for r in range(n):
        for c in range(n):
            if layout[r, c] != background:
                out[r*n:(r+1)*n, c*n:(c+1)*n][stencil] = layout[r, c]
    return out


def _make_pair(rng, n, orientation, stencil_first, stable_marker, fixed_marker, fixed_background, force_clear):
    background = fixed_background if fixed_background is not None else int(rng.integers(10))
    marker = fixed_marker if stable_marker else int(rng.choice([x for x in range(10) if x != background]))
    if marker == background:
        background = (background + 1) % 10
    palette = [x for x in range(10) if x not in (background, marker)]
    stencil = _stencil(rng, n)
    layout = _make_layout(rng, n, background, palette, stable_marker, force_clear)
    marked = np.where(stencil, marker, background).astype(np.int8)
    panels = (marked, layout) if stencil_first else (layout, marked)
    inp = np.concatenate(panels, axis=1 if orientation == 'horizontal' else 0)
    return {'input': inp, 'output': _render(layout, stencil, background)}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    # The only possible puzzle-wide convention is a fixed marker. Its role is
    # shown by clear multicolor examples before binary/full layouts are used.
    stable_marker = bool(rng.integers(2))
    fixed_marker = int(rng.integers(10)) if stable_marker else None
    fixed_background = int(rng.choice([x for x in range(10) if x != fixed_marker])) if rng.integers(2) else None
    constant_size = bool(rng.integers(3) == 0)
    common_n = int(rng.integers(3, 6))
    placement_regime = int(rng.integers(4))
    if placement_regime == 0:  # original horizontal-to-vertical shift
        train_orientations = ['horizontal'] * train_count
        test_orientations = ['vertical'] * test_count
    elif placement_regime == 1:  # mixed training, independent tests
        train_orientations = ['horizontal', 'vertical'] + [str(rng.choice(['horizontal', 'vertical'])) for _ in range(train_count - 2)]
        rng.shuffle(train_orientations)
        test_orientations = [str(rng.choice(['horizontal', 'vertical'])) for _ in range(test_count)]
    else:  # no shift or a shift of order within the same axis
        orientation = 'horizontal' if placement_regime == 2 else 'vertical'
        train_orientations = [orientation] * train_count
        test_orientations = [orientation] * test_count
    train_orders = [True, False] + [bool(rng.integers(2)) for _ in range(train_count - 2)]
    rng.shuffle(train_orders)
    test_orders = [bool(rng.integers(2)) for _ in range(test_count)]
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max, size=train_count + test_count)
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, count, orientations, orders, offset in (
        ('train', train_count, train_orientations, train_orders, 0),
        ('test', test_count, test_orientations, test_orders, train_count),
    ):
        for i in range(count):
            pair_rng = np.random.default_rng(int(pair_seeds[offset + i]))
            n = common_n if constant_size else int(pair_rng.choice([2, 3, 3, 4, 4, 5, 5]))
            # First training pair must identify the local background and marker.
            force_clear = split == 'train' and i == 0
            for _ in range(80):
                pair = _make_pair(pair_rng, n, orientations[i], orders[i], stable_marker,
                                  fixed_marker, fixed_background, force_clear)
                key = _fingerprint(pair['input'])
                if key not in seen and not _source_copy(pair['input']):
                    break
            else:
                raise RuntimeError('could not construct a unique pair')
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _split(grid):
    h, w = grid.shape
    if w == 2 * h:
        return h, 'horizontal', (grid[:, :h], grid[:, h:])
    if h == 2 * w:
        return w, 'vertical', (grid[:w], grid[w:])
    raise AssertionError('input must contain two equal square panels')


def _decode(grid, learned_marker=None):
    n, orientation, panels = _split(grid)
    palettes = [set(int(x) for x in np.unique(p)) for p in panels]
    candidates = []
    for s in (0, 1):
        other = 1 - s
        if len(palettes[s]) != 2:
            continue
        shared = palettes[s] & palettes[other]
        if len(shared) == 1 and len(palettes[other]) >= 3:
            background = next(iter(shared))
            marker = next(iter(palettes[s] - {background}))
            candidates.append((s, background, marker))
        elif learned_marker is not None and learned_marker in palettes[s]:
            background = next(iter(palettes[s] - {learned_marker}))
            if learned_marker not in palettes[other] and len(palettes[other] - {background}) >= 1:
                candidates.append((s, background, learned_marker))
    _require(len(candidates) == 1, 'panel roles or background are ambiguous')
    s, background, marker = candidates[0]
    layout = panels[1 - s]
    _require(marker not in layout, 'marker used in layout')
    _require(0 < int(np.count_nonzero(panels[s] != background)) < n*n, 'stencil must have active and inactive cells')
    _require(np.count_nonzero(layout != background) >= 1, 'layout must contain payload')
    if background not in layout:
        _require(len(palettes[1 - s]) >= 3, 'full layout needs three payload colors')
    return {'n': n, 'orientation': orientation, 'stencil_first': s == 0,
            'background': background, 'marker': marker,
            'stencil': panels[s] != background, 'layout': layout,
            'payload_count': len(palettes[1 - s] - {background})}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle schema')
    _require(isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list), 'splits must be lists')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'invalid pair counts')
    seen = set()
    first = puzzle['train'][0]
    _require(isinstance(first, dict) and set(first) == {'input', 'output'}, 'invalid first pair')
    # Recover the possible learned marker solely from an unambiguous training pair.
    first_info = _decode(first['input'])
    learned_marker = first_info['marker']
    records = {'train': [], 'test': []}
    ambiguous_cases = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair schema')
            for grid in (pair['input'], pair['output']):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, 'grid must be a 2D array')
                _require(np.issubdtype(grid.dtype, np.integer), 'grid dtype must be integer')
                _require(all(1 <= d <= 30 for d in grid.shape), 'grid dimensions out of range')
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9, 'grid colors out of range')
            key = _fingerprint(pair['input'])
            _require(key not in seen and not _source_copy(pair['input']), 'duplicate input or original source copy')
            seen.add(key)
            info = _decode(pair['input'], learned_marker)
            n = info['n']
            _require(pair['output'].shape == (n*n, n*n), 'invalid output shape')
            _require(np.array_equal(pair['output'], _render(info['layout'], info['stencil'], info['background'])), 'block substitution mismatch')
            _require(info['marker'] not in pair['output'], 'marker persists in output')
            _require(not np.array_equal(pair['input'], pair['output']), 'unchanged pair')
            if info['payload_count'] == 1 or info['background'] not in info['layout']:
                ambiguous_cases = True
                _require(info['marker'] == learned_marker, 'learned marker mismatch')
            records[split].append(info)
    if ambiguous_cases:
        _require(all(x['marker'] == learned_marker for x in records['train']), 'marker convention is not stable in training')
    _require(len({x['stencil_first'] for x in records['train']}) == 2, 'training must show both panel orders')
    # Both split regimes obey the same local rule. The source's axis shift is
    # one sampled regime, while other puzzles teach both axes or keep one axis.
    _require(all(2 <= x['n'] <= 5 for split in records.values() for x in split), 'panel size out of range')


def _equal(a, b):
    return all(len(a[k]) == len(b[k]) and all(np.array_equal(x[q], y[q])
               for x, y in zip(a[k], b[k]) for q in ('input', 'output'))
               for k in ('train', 'test'))


def _connected(mask):
    cells = set(zip(*np.where(mask)))
    seen = {next(iter(cells))}
    queue = list(seen)
    for r, c in queue:
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            neighbor = (r + dr, c + dc)
            if neighbor in cells and neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen == cells


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    counts = {(len(p['train']), len(p['test'])) for p in puzzles}
    _require(len({x for x, _ in counts}) >= 4 and len({y for _, y in counts}) >= 3, 'count diversity')
    shapes, stencils, layouts, palettes = set(), set(), set(), set()
    train_axes, test_axes, orders = set(), set(), set()
    sparse_train = dense_train = sparse_test = dense_test = 0
    symmetric = full = binary = repeated_rows = repeated_cols = 0
    mixed_sizes = constant_sizes = changed_background = changed_marker = 0
    connected_count = hole_count = unseen_payload_count = 0
    source_shift = no_shift = mixed_train_axes = reversed_density = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        infos = {'train': [], 'test': []}
        marker = _decode(puzzle['train'][0]['input'])['marker']
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                x = _decode(pair['input'], marker)
                infos[split].append(x)
                shapes.add(pair['input'].shape)
                stencils.add(_fingerprint(x['stencil'].astype(np.int8)))
                layouts.add(_fingerprint(x['layout']))
                palettes.add(tuple(sorted(int(z) for z in np.unique(pair['input']))))
                orders.add((x['orientation'], x['stencil_first'], split))
                density = float(x['stencil'].mean())
                if split == 'train':
                    sparse_train += density < .4
                    dense_train += density > .6
                    train_axes.add(x['orientation'])
                else:
                    sparse_test += density < .4
                    dense_test += density > .6
                    test_axes.add(x['orientation'])
                symmetric += np.array_equal(x['stencil'], np.fliplr(x['stencil']))
                connected_count += _connected(x['stencil'])
                hole_count += (x['n'] >= 3 and
                               np.any(~x['stencil'][1:-1, 1:-1]) and
                               np.all(x['stencil'][0]) and np.all(x['stencil'][-1]) and
                               np.all(x['stencil'][:, 0]) and np.all(x['stencil'][:, -1]))
                full += x['background'] not in x['layout']
                binary += x['payload_count'] == 1
                repeated_rows += len({tuple(row) for row in x['layout']}) < x['n']
                repeated_cols += len({tuple(col) for col in x['layout'].T}) < x['n']
        all_infos = infos['train'] + infos['test']
        sizes = {x['n'] for x in all_infos}
        mixed_sizes += len(sizes) > 1
        constant_sizes += len(sizes) == 1
        changed_background += len({x['background'] for x in all_infos}) > 1
        changed_marker += len({x['marker'] for x in all_infos}) > 1
        train_info, test_info = infos['train'], infos['test']
        train_orientations = {x['orientation'] for x in train_info}
        test_orientations = {x['orientation'] for x in test_info}
        source_shift += train_orientations == {'horizontal'} and test_orientations == {'vertical'}
        no_shift += len(train_orientations | test_orientations) == 1
        mixed_train_axes += len(train_orientations) == 2
        reversed_density += (np.mean([x['stencil'].mean() for x in test_info]) <
                             np.mean([x['stencil'].mean() for x in train_info]))
        train_colors = {int(color) for x in train_info for color in np.unique(x['layout'])
                        if color != x['background']}
        test_colors = {int(color) for x in test_info for color in np.unique(x['layout'])
                       if color != x['background']}
        unseen_payload_count += bool(test_colors - train_colors)
    _require(len(stencils) >= 70 and len(layouts) >= 100 and len(palettes) >= 20, 'structural diversity')
    _require({(n, 2*n) for n in (2, 3, 4, 5)}.issubset(shapes), 'missing horizontal sizes')
    _require({(2*n, n) for n in (2, 3, 4, 5)}.issubset(shapes), 'missing vertical sizes')
    _require(train_axes == test_axes == {'horizontal', 'vertical'}, 'axis diversity')
    _require(len(orders) == 8, 'split/order diversity')
    _require(min(sparse_train, dense_train, sparse_test, dense_test) >= 5, 'stencil density diversity')
    _require(min(symmetric, full, binary, repeated_rows, repeated_cols) >= 2, 'motif/layout diversity')
    _require(min(mixed_sizes, constant_sizes, changed_background, changed_marker) >= 2, 'puzzle diversity')
    _require(min(connected_count, hole_count, unseen_payload_count) >= 2, 'geometry/color diversity')
    _require(min(source_shift, no_shift, mixed_train_axes, reversed_density) >= 2, 'regime diversity')
    _require(_equal(puzzles[7], generate_puzzle(7)), 'automatic determinism')
    _require(_equal(generate_puzzle(777, 6, 2), generate_puzzle(777, 6, 2)), 'explicit determinism')
    for i, (a, b) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(1000 + i, a, b)
        validate_generated_puzzle(p)
        if a is not None:
            _require(len(p['train']) == a, 'explicit train count')
        if b is not None:
            _require(len(p['test']) == b, 'explicit test count')
    for kwargs in ({'num_train': 1}, {'num_train': 11}, {'num_test': 0}, {'num_test': 5},
                   {'num_train': 2, 'num_test': 2}, {'num_train': 3, 'num_test': 4}, {'num_train': True}):
        try:
            generate_puzzle(1200, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f'illegal count accepted: {kwargs}')
