import numpy as np


# Exact source inputs are excluded. Rejecting an identical input is slightly
# stronger than merely rejecting an identical input/output pair.
_SOURCE_INPUT_ROWS = {
    (
        "57575775",
        "57557757",
        "57777557",
        "57775757",
        "57777757",
        "55577777",
    ),
    (
        "77757",
        "75577",
        "55777",
        "77755",
    ),
    (
        "77775577577",
        "55775577755",
        "55777575775",
        "55777777777",
        "55777557775",
        "77777757757",
        "75755757577",
        "77775757777",
        "57777777775",
        "55775757577",
        "75577575757",
        "75557777777",
    ),
    (
        "7557777755557757777",
        "7757777777755577577",
        "7757757575777775577",
        "5775777557777575757",
        "7775757775777777755",
        "7575777777577777755",
        "7557577777777755775",
        "7777775577755777757",
        "7757755755775577777",
        "7777775577577777577",
        "5777777777575777755",
        "7555777757777777757",
        "7577777755777757755",
        "7557775755755575777",
        "7777755757777755575",
        "7777775577777757777",
    ),
}

_DIR8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)
_DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid_signature(grid):
    canonical = grid.astype(np.int8, copy=False)
    return canonical.shape, canonical.tobytes()


def _is_source_input(grid):
    return tuple(''.join(str(int(v)) for v in row) for row in grid) in _SOURCE_INPUT_ROWS


def _diagonal_runs(mask):
    h, w = mask.shape
    qualifying = np.zeros_like(mask, dtype=bool)
    runs = []
    for dc in (-1, 1):
        for r in range(h):
            for c in range(w):
                if not mask[r, c]:
                    continue
                if r and 0 <= c - dc < w and mask[r - 1, c - dc]:
                    continue
                cells = []
                rr, cc = r, c
                while rr < h and 0 <= cc < w and mask[rr, cc]:
                    cells.append((rr, cc))
                    rr += 1
                    cc += dc
                if len(cells) >= 3:
                    runs.append((dc, tuple(cells)))
                    for cell in cells:
                        qualifying[cell] = True
    return qualifying, runs


def _component_labels(mask):
    """Eight-neighbor components, including corner contacts."""
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=np.int16)
    sizes = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] >= 0:
                continue
            component = len(sizes)
            stack = [(r, c)]
            labels[r, c] = component
            size = 0
            while stack:
                rr, cc = stack.pop()
                size += 1
                for dr, dc in _DIR8:
                    nr, nc = rr + dr, cc + dc
                    if (0 <= nr < h and 0 <= nc < w and mask[nr, nc]
                            and labels[nr, nc] < 0):
                        labels[nr, nc] = component
                        stack.append((nr, nc))
            sizes.append(size)
    return labels, sizes


def _analyze_mask(mask):
    qualifying, runs = _diagonal_runs(mask)
    labels, sizes = _component_labels(mask)
    selected_ids = set(int(v) for v in labels[qualifying])
    selected_ids.discard(-1)
    marked = np.isin(labels, list(selected_ids)) & mask
    corner_branch = False
    short_component = False
    short_orientations = set()
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            component = int(labels[r, c])
            if component not in selected_ids and sizes[component] == 2:
                neighbors = [(r + dr, c + dc) for dr, dc in _DIR8
                             if 0 <= r + dr < h and 0 <= c + dc < w
                             and mask[r + dr, c + dc]]
                if len(neighbors) == 1 and abs(neighbors[0][0] - r) == 1 and abs(neighbors[0][1] - c) == 1:
                    short_component = True
                    if neighbors[0][0] > r:
                        short_orientations.add(neighbors[0][1] - c)
            if marked[r, c] and not qualifying[r, c]:
                diagonal_contact = any(
                    0 <= r + dr < h and 0 <= c + dc < w and mask[r + dr, c + dc]
                    for dr in (-1, 1) for dc in (-1, 1)
                )
                edge_contact = any(
                    0 <= r + dr < h and 0 <= c + dc < w and mask[r + dr, c + dc]
                    for dr, dc in _DIR4
                )
                if diagonal_contact and not edge_contact:
                    corner_branch = True
    return {
        'qualifying': qualifying,
        'runs': runs,
        'orientations': {dc for dc, _ in runs},
        'labels': labels,
        'sizes': sizes,
        'marked': marked,
        'marked_components': len(selected_ids),
        'unmarked_components': len(sizes) - len(selected_ids),
        'expansion': int(np.count_nonzero(marked & ~qualifying)),
        'corner_branch': corner_branch,
        'short_component': short_component,
        'short_orientations': short_orientations,
    }


def _expected_output(input_grid, foreground, highlight):
    output = input_grid.copy()
    output[_analyze_mask(input_grid == foreground)['marked']] = highlight
    return output


def _plant_run(mask, rng, direction, length, allowed=None):
    h, w = mask.shape
    if length > min(h, w):
        return False
    starts = [(r, c) for r in range(h - length + 1)
              for c in (range(w - length + 1) if direction == 1 else range(length - 1, w))]
    for ix in rng.permutation(len(starts)):
        r, c = starts[int(ix)]
        cells = [(r + k, c + direction * k) for k in range(length)]
        if allowed is not None and not all(allowed[cell] for cell in cells):
            continue
        if sum(bool(mask[cell]) for cell in cells) > 1:
            continue
        for cell in cells:
            mask[cell] = True
        return True
    return False


def _add_shapes(mask, rng, target_density, allowed, protected=None):
    """Grow bent, branched and thick patches from both old and new seeds."""
    h, w = mask.shape
    target = min(int(round(target_density * mask.size)), int(mask.size * 0.44))
    if protected is None:
        protected = np.zeros_like(mask, dtype=bool)
    for _ in range(mask.size * 4):
        if int(np.count_nonzero(mask)) >= target:
            break
        existing = np.argwhere(mask & allowed)
        if len(existing) and rng.random() < 0.46:
            r, c = (int(v) for v in existing[int(rng.integers(len(existing)))])
        else:
            r, c = int(rng.integers(h)), int(rng.integers(w))
        steps = int(rng.integers(1, 13))
        for _ in range(steps):
            if int(np.count_nonzero(mask)) >= target:
                break
            if allowed[r, c] and not protected[r, c]:
                mask[r, c] = True
            dr, dc = _DIR4[int(rng.integers(4))]
            r = min(h - 1, max(0, r + dr))
            c = min(w - 1, max(0, c + dc))
            if rng.random() < 0.18:
                nr, nc = r + dc, c + dr
                if 0 <= nr < h and 0 <= nc < w and allowed[nr, nc] and not protected[nr, nc]:
                    mask[nr, nc] = True
    # Scatter a few small detached pieces when the walks are inefficient.
    available = np.argwhere(allowed & ~protected & ~mask)
    for ix in rng.permutation(len(available)):
        if int(np.count_nonzero(mask)) >= target:
            break
        r, c = available[int(ix)]
        mask[r, c] = True


def _make_witness(rng, direction):
    """A training example that fixes the threshold and corner propagation."""
    for _ in range(128):
        h = int(rng.integers(14, 23))
        w = int(rng.integers(16, 25))
        left_width = w - 5
        r = int(rng.integers(3, h - 5))
        c = int(rng.integers(3, left_width - 3))
        if direction == -1:
            c = left_width - 1 - c
        run = [(r + k, c + direction * k) for k in range(3)]
        if any(not 0 <= cc < left_width for _, cc in run):
            continue
        mask = np.zeros((h, w), dtype=bool)
        for cell in run:
            mask[cell] = True
        options = []
        for ar, ac in run:
            for dr in (-1, 1):
                for dc in (-1, 1):
                    nr, nc = ar + dr, ac + dc
                    if not (0 <= nr < h and 0 <= nc < left_width) or mask[nr, nc]:
                        continue
                    if any(0 <= nr + er < h and 0 <= nc + ec < w and mask[nr + er, nc + ec]
                           for er, ec in _DIR4):
                        continue
                    trial = mask.copy()
                    trial[nr, nc] = True
                    a = _analyze_mask(trial)
                    if a['corner_branch'] and not a['qualifying'][nr, nc]:
                        options.append((nr, nc))
        if not options:
            continue
        corner = options[int(rng.integers(len(options)))]
        mask[corner] = True
        protected = np.zeros_like(mask)
        for rr in range(max(0, corner[0] - 1), min(h, corner[0] + 2)):
            for cc in range(max(0, corner[1] - 1), min(w, corner[1] + 2)):
                if not mask[rr, cc]:
                    protected[rr, cc] = True
        for rr, cc in (run[0], run[-1]):
            er = rr + (-1 if (rr, cc) == run[0] else 1)
            ec = cc + (-direction if (rr, cc) == run[0] else direction)
            if 0 <= er < h and 0 <= ec < w:
                protected[er, ec] = True
        allowed = np.zeros_like(mask)
        allowed[:, :left_width] = True
        _add_shapes(mask, rng, float(rng.uniform(0.08, 0.13)), allowed, protected)
        nr = int(rng.integers(1, h - 2))
        mask[nr, w - 3] = True
        mask[nr + 1, w - 2] = True
        other_rows = [row for row in range(1, h - 2) if abs(row - nr) >= 4]
        other_row = other_rows[int(rng.integers(len(other_rows)))]
        mask[other_row, w - 2] = True
        mask[other_row + 1, w - 3] = True
        a = _analyze_mask(mask)
        if (a['corner_branch'] and a['short_orientations'] == {-1, 1}
                and a['orientations'] == {direction}
                and all(len(cells) == 3 for _, cells in a['runs'])
                and np.count_nonzero(mask) < mask.size / 2):
            return mask, direction
    mask = np.zeros((16, 18), dtype=bool)
    start_col = 3 if direction == 1 else 8
    for k in range(3):
        mask[4 + k, start_col + direction * k] = True
    mask[4, start_col + 2 * direction] = True
    mask[1, 15], mask[2, 16] = True, True
    mask[10, 16], mask[11, 15] = True, True
    assert _analyze_mask(mask)['short_orientations'] == {-1, 1}
    return mask, direction


def _sample_shape(rng, regime):
    # Both splits cover the same sizes; tests modestly favor larger grids.
    bands = ((5, 10), (11, 18), (19, 28))
    probabilities = (0.24, 0.43, 0.33) if regime == 'source_train' else (0.19, 0.39, 0.42)
    hb = int(rng.choice(3, p=probabilities))
    wb = int(rng.choice(3, p=probabilities))
    return (int(rng.integers(bands[hb][0], bands[hb][1] + 1)),
            int(rng.integers(bands[wb][0], bands[wb][1] + 1)))


def _make_general(rng, regime, required_direction=None, density_mode=None,
                  single_slope=False, avoid_length_three=False):
    for _ in range(64):
        h, w = _sample_shape(rng, regime)
        if single_slope and max(h, w) > 18:
            continue
        mask = np.zeros((h, w), dtype=bool)
        direction = required_direction if required_direction is not None else (1 if rng.random() < 0.5 else -1)
        max_length = min(h, w, 8)
        if avoid_length_three and max_length >= 4:
            length = int(rng.integers(4, max_length + 1))
        else:
            length = int(rng.integers(3, max_length + 1))
        if not _plant_run(mask, rng, direction, length):
            continue
        extra = 0 if single_slope or avoid_length_three else int(rng.integers(0, min(5, max(1, mask.size // 70)) + 1))
        for _ in range(extra):
            dc = 1 if rng.random() < 0.5 else -1
            run_length = int(rng.integers(3, min(h, w, 9) + 1))
            _plant_run(mask, rng, dc, run_length)
        mode = density_mode or ('dense' if rng.random() < 0.72 else 'sparse')
        if single_slope or avoid_length_three:
            density = float(rng.uniform(0.13, 0.20))
        else:
            if mode == 'anchor_dense':
                density = float(rng.uniform(0.34, 0.40))
            else:
                density = float(rng.uniform(0.28, 0.40) if mode == 'dense' else rng.uniform(0.14, 0.23))
        _add_shapes(mask, rng, density, np.ones_like(mask))
        a = _analyze_mask(mask)
        if (np.count_nonzero(mask) <= 0.40 * mask.size and len(a['runs']) >= 1
                and direction in a['orientations'] and a['expansion'] >= 1
                and (not single_slope or a['orientations'] == {direction})
                and (not avoid_length_three or all(len(cells) != 3 for _, cells in a['runs']))):
            return mask
    mask = np.zeros((12, 14), dtype=bool)
    direction = required_direction if required_direction is not None else 1
    length = 4 if avoid_length_three else 3
    start_col = 3 if direction == 1 else 10
    for k in range(length):
        mask[3 + k, start_col + direction * k] = True
    mask[2, start_col] = True
    if density_mode in ('dense', 'anchor_dense') and not (single_slope or avoid_length_three):
        mask[5:12, 0:8] = True
    return mask


def _validate_requested_count(name, value, minimum, maximum):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not minimum <= int(value) <= maximum:
        raise ValueError(f'{name} must be an integer from {minimum} through {maximum}')
    return int(value)


def _resolve_counts(seed, rng, num_train, num_test):
    train = _validate_requested_count('num_train', num_train, 2, 10)
    test = _validate_requested_count('num_test', num_test, 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError('num_train must exceed num_test')
        return train, test
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _infer_roles(pair):
    inp, out = pair['input'], pair['output']
    values, counts = np.unique(inp, return_counts=True)
    assert len(values) == 2 and counts[0] != counts[1]
    fg = int(values[int(np.argmin(counts))])
    bg = int(values[int(np.argmax(counts))])
    highlight = set(int(v) for v in np.unique(out)) - set(int(v) for v in values)
    assert len(highlight) == 1
    return bg, fg, highlight.pop()


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    base = [int(v) for v in rng.choice(10, size=3, replace=False)]
    base_bg, base_fg, highlight = base
    palette = [c for c in range(10) if c != highlight]
    puzzle = {'train': [], 'test': []}
    seen = set()
    witness_direction = 1 if rng.random() < 0.5 else -1
    witness_index = int(rng.integers(min(3, train_count)))
    contrast_index = 1 if witness_index == 0 else 0
    dense_train_index = max(witness_index, contrast_index) + 1
    if dense_train_index >= train_count:
        dense_train_index = None
    for split, count in (('train', train_count), ('test', test_count)):
        for index in range(count):
            for attempt in range(16):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                try:
                    if split == 'train' and index == witness_index:
                        mask, _ = _make_witness(pair_rng, witness_direction)
                    else:
                        required = -witness_direction if split == 'train' and index == contrast_index else None
                        force_dense_test = split == 'test' and index == 0 and dense_train_index is None
                        mode = 'anchor_dense' if (split == 'train' and index == dense_train_index) or force_dense_test else None
                        single = split == 'test' and index == 0 and not force_dense_test and rng.random() < 0.38
                        clean_train = split == 'train' and (index < witness_index or index == contrast_index)
                        mask = _make_general(pair_rng, 'source_train' if split == 'train' else 'target_test',
                                             required, mode, single_slope=single or clean_train,
                                             avoid_length_three=clean_train)
                except ValueError:
                    continue
                if split == 'train' and index == 0:
                    bg, fg = base_bg, base_fg
                elif split == 'train' and index == 1:
                    bg, fg = base_fg, base_bg
                elif pair_rng.random() < 0.38:
                    bg, fg = (base_bg, base_fg) if pair_rng.random() < 0.5 else (base_fg, base_bg)
                else:
                    bg, fg = (int(v) for v in pair_rng.choice(palette, size=2, replace=False))
                inp = np.full(mask.shape, bg, dtype=np.int8)
                inp[mask] = fg
                if _is_source_input(inp) or _grid_signature(inp) in seen:
                    continue
                out = _expected_output(inp, fg, highlight)
                seen.add(_grid_signature(inp))
                puzzle[split].append({'input': inp, 'output': out})
                break
            else:
                raise ValueError('Unable to construct a distinct pair')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen = set()
    highlight = None
    orientations = set()
    length_three = short_pair = corner_branch = remaining = False
    train_roles = []
    train_analyses = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = pair['input'], pair['output']
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            sig = _grid_signature(inp)
            assert sig not in seen and not _is_source_input(inp)
            seen.add(sig)
            bg, fg, pair_highlight = _infer_roles(pair)
            assert bg != fg and pair_highlight not in (bg, fg)
            if highlight is None:
                highlight = pair_highlight
            assert pair_highlight == highlight
            assert np.array_equal(out, _expected_output(inp, fg, highlight))
            a = _analyze_mask(inp == fg)
            assert a['runs'] and a['expansion'] >= 1
            if split == 'train':
                train_roles.append((bg, fg))
                train_analyses.append(a)
                orientations |= a['orientations']
                length_three |= any(len(cells) == 3 for _, cells in a['runs'])
                short_pair |= a['short_component']
                corner_branch |= a['corner_branch']
                remaining |= a['unmarked_components'] > 0
    assert orientations == {-1, 1}
    assert length_three and short_pair and corner_branch and remaining
    assert len(set(train_roles)) >= 2
    # The clean threshold example rules out lengths two and four; the other
    # slope is selected in a different training input, so either-slope
    # behavior is observable even when later dense masks merge components.
    witnesses = [a for a in train_analyses
                 if a['short_orientations'] == {-1, 1} and a['corner_branch']
                 and len(a['orientations']) == 1
                 and a['runs'] and all(len(cells) == 3 for _, cells in a['runs'])]
    assert any(any(b['orientations'] == {-next(iter(a['orientations']))}
                   and all(len(cells) >= 4 for _, cells in b['runs'])
                   for b in train_analyses)
               for a in witnesses)


def _puzzles_equal(a, b):
    return all(np.array_equal(pa[key], pb[key])
               for split in ('train', 'test')
               for pa, pb in zip(a[split], b[split])
               for key in ('input', 'output')) and all(len(a[s]) == len(b[s]) for s in ('train', 'test'))


def self_test() -> None:
    source_changes = []
    for rows in sorted(_SOURCE_INPUT_ROWS, key=lambda item: (len(item), len(item[0]))):
        source_mask = np.array([[cell == '5' for cell in row] for row in rows], dtype=bool)
        source_changes.append(int(np.count_nonzero(_analyze_mask(source_mask)['marked'])))
    assert source_changes == [5, 7, 15, 37]
    a = generate_puzzle(123, 7, 3)
    assert _puzzles_equal(a, generate_puzzle(123, 7, 3))
    for nt, ns in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(240 + (nt or 0) * 10 + (ns or 0), nt, ns))
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (4, 4)):
        try:
            generate_puzzle(3, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid counts accepted')
    train_counts, test_counts, colors, shapes, layouts = set(), set(), set(), set(), set()
    densities, run_counts, size_by_split = [], {'train': [], 'test': []}, {'train': [], 'test': []}
    first_length_three, density_spreads = set(), []
    one_slope_tests = mixed_component_pairs = 0
    component_sizes, balances = set(), set()
    role_changes = corner_examples = short_examples = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        roles = [_infer_roles(p) for p in puzzle['train'] + puzzle['test']]
        role_changes += len(set((bg, fg) for bg, fg, _ in roles)) >= 2
        colors.add(roles[0][2])
        puzzle_densities = []
        first_length_three.add(next(i for i, pair in enumerate(puzzle['train'])
                                    if any(len(cells) == 3 for _, cells in
                                           _analyze_mask(pair['input'] == _infer_roles(pair)[1])['runs'])))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input']
                _, fg, _ = _infer_roles(pair)
                mask = inp == fg
                analysis = _analyze_mask(mask)
                shapes.add(inp.shape)
                layouts.add((inp.shape, mask.tobytes()))
                densities.append(float(mask.mean()))
                puzzle_densities.append(float(mask.mean()))
                run_counts[split].append(len(analysis['runs']))
                size_by_split[split].append(inp.size)
                corner_examples += analysis['corner_branch']
                short_examples += analysis['short_component']
                one_slope_tests += split == 'test' and len(analysis['orientations']) == 1
                mixed_component_pairs += analysis['marked_components'] > 0 and analysis['unmarked_components'] > 0
                component_sizes.update(analysis['sizes'])
                balances.add((analysis['marked_components'], analysis['unmarked_components']))
        density_spreads.append(max(puzzle_densities) - min(puzzle_densities))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(colors) >= 5 and len(shapes) >= 40 and len(layouts) >= 0.95 * sum(len(run_counts[s]) for s in run_counts)
    assert role_changes == 42 and corner_examples >= 42 and short_examples >= 42
    assert first_length_three == {0, 1, 2}
    assert min(density_spreads) > 0.1
    assert sum(d < 0.23 for d in densities) >= 100
    assert sum(d >= 0.28 for d in densities) >= 100
    assert one_slope_tests >= 5 and mixed_component_pairs >= 100
    assert len(component_sizes) >= 20 and len(balances) >= 20
    assert min(run_counts['test']) < max(run_counts['train'])
    assert min(size_by_split['test']) < max(size_by_split['train'])
