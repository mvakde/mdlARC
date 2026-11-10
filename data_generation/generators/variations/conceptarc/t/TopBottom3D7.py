"""Crossing-completion puzzles for ARC task 1900289e."""

from collections import Counter
import numpy as np

_SOURCE_INPUTS = {tuple(s.split('/')) for s in (
    '00000600/11111111/00000600/00000600/11111611/00000600/00000600/11111111/00000600/11111111/00000600/00000600/11111611',
    '00808080/77878777/00808080/00808080/77777777/00808080/77777787/00808080/00808080/00808080/00808080/77778787/00808080',
    '00909090/00909090/33339333/00909090/00909090/00909090/00909090/00909090/00909090/00909090/00909090/33933333/00909090',
    '09009009/09009009/55559559/09009009/09009009/55555555/09009009/09009009/09009009/09009009/09009009/09009009/59559555',
    '06006006/06006006/06006006/44446444/06006006/06006006/06006006/06006006/46444444/06006006/06006006/06006006/06006006',
    '40004040/22222222/22222222/40004040/40004040/40004040/40004040/40004040/40004040/40004040/42222222/40004040/40004040',
)}


def _signature(grid):
    return tuple(''.join(str(int(v)) for v in row) for row in grid)


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f'{name} must be in [{low}, {high}]')
    return value


def _nonadjacent_rows(rng, height, count):
    x = np.sort(rng.choice(height - count + 1, size=count, replace=False))
    return tuple(int(v) for v in x + np.arange(count))


def _rows(rng, height, count, adjacent=False):
    if adjacent:
        start = int(rng.integers(height - 1))
        rest = [r for r in range(height) if r not in (start, start + 1)]
        extra = rng.choice(rest, size=count - 2, replace=False)
        return tuple(sorted((start, start + 1, *(int(v) for v in extra))))
    return tuple(sorted(int(v) for v in rng.choice(height, size=count, replace=False)))


def _short_intervals(width, cols):
    occupied = set(cols)
    return [(a, b) for a in range(width) for b in range(a + 3, width)
            if a not in occupied and b not in occupied
            and 1 <= sum(a <= c <= b for c in cols) < len(cols)
            and 2 <= sum(a <= c <= b and c not in occupied for c in range(width)) < width - len(cols)]


def _columns(rng, width, count, need_short=False, neighbor=False):
    for _ in range(40):
        cols = tuple(sorted(int(v) for v in rng.choice(width, size=count, replace=False)))
        if neighbor and not any(b == a + 1 for a, b in zip(cols, cols[1:])):
            continue
        if need_short and not _short_intervals(width, cols):
            continue
        return cols
    if need_short:
        return (2, width // 2, width - 3)
    return tuple(range(count))


def _layout(rng, kind):
    if kind in ('source_like', 'ordinary'):
        if rng.random() < 0.2:
            height, width = 13, 8
        else:
            height, width = int(rng.integers(8, 19)), int(rng.integers(7, 18))
        if kind == 'ordinary' or rng.random() < 0.55:
            start = int(rng.integers(width - 4))
            cols = (start, start + 2, start + 4)
        else:
            cols = (int(rng.integers(width)),)
        bands = int(rng.integers(2, min(4, (height - 1) // 2) + 1))
        return height, width, cols, _nonadjacent_rows(rng, height, bands)
    if kind == 'coverage':
        height = int(rng.integers(11, 21))
        width = int(rng.integers(10, 24))
        if rng.random() < 0.4:
            width = int(rng.integers(height + 1, 25))
        count = int(rng.integers(2, min(4, (width - 1) // 2) + 1))
        cols = _columns(rng, width, count, need_short=True)
        bands = int(rng.integers(3, min(6, (height - 1) // 2) + 1))
        return height, width, cols, _rows(rng, height, bands, adjacent=True)
    height, width = int(rng.integers(7, 21)), int(rng.integers(7, 25))
    if width <= 12 and rng.random() < 0.14:
        count = width - 3
        start = int(rng.integers(1, width - count))
        cols = tuple(range(start, start + count))
    else:
        count = int(rng.integers(1, min(5, (width - 1) // 2) + 1))
        cols = _columns(rng, width, count, neighbor=count >= 2 and rng.random() < 0.22)
    bands = int(rng.integers(1, min(8, (height - 1) // 2) + 1))
    return height, width, cols, _rows(rng, height, bands, adjacent=bands >= 2 and rng.random() < 0.4)


def _colors(rng, n_strand, n_band, palette=None, avoid_background=None):
    if palette is not None and n_strand == n_band == 1:
        return palette[0], (palette[1],), (palette[2],)
    choices = [c for c in range(10) if c != avoid_background]
    background = int(rng.choice(choices))
    available = [c for c in range(10) if c != background]
    picked = [int(v) for v in rng.choice(available, size=n_strand + n_band, replace=False)]
    return background, tuple(picked[:n_strand]), tuple(picked[n_strand:])


def _make_pair(rng, kind, orientation, palette_bank, first_background):
    height, width, cols, rows = _layout(rng, kind)
    if kind == 'coverage':
        n_strand = len(cols)
        n_band = int(rng.integers(2, min(4, len(rows) - 1) + 1))
    elif kind in ('source_like', 'ordinary'):
        n_strand = n_band = 1
    else:
        n_strand = len(cols) if 2 <= len(cols) <= 5 and rng.random() < 0.32 else 1
        n_band = int(rng.integers(2, min(4, len(rows)) + 1)) if len(rows) >= 2 and rng.random() < 0.32 else 1
        if cols[0] == 0 and cols[-1] == width - 1 and n_strand == 1:
            n_strand = len(cols)
    palette = None
    if n_strand == n_band == 1 and rng.random() < (0.7 if kind in ('source_like', 'ordinary') else 0.45):
        palette = palette_bank[int(rng.integers(len(palette_bank)))]
    background, sc, bc = _colors(rng, n_strand, n_band, palette,
                                  first_background if kind == 'coverage' else None)
    strand_colors = tuple(sc[i] if n_strand > 1 else sc[0] for i in range(len(cols)))
    band_colors = [bc[0]] * len(rows)
    coverage_pair = None
    short_index = None
    if n_band > 1:
        for i in range(len(rows)):
            band_colors[i] = bc[i % n_band]
        if kind == 'coverage':
            j = next(i for i in range(len(rows) - 1) if rows[i + 1] == rows[i] + 1)
            coverage_pair = (j, j + 1)
            short_options = [i for i in range(len(rows)) if i not in coverage_pair]
            short_index = int(rng.choice(short_options))
            band_colors[j] = band_colors[j + 1] = bc[0]
            if len(set(band_colors)) == 1:
                band_colors[-1 if j != len(rows) - 2 else 0] = bc[1]
    shorts = _short_intervals(width, cols)
    intervals = []
    for i, row in enumerate(rows):
        if kind in ('source_like', 'ordinary'):
            interval = (0, width - 1)
        elif kind == 'coverage' and i == short_index:
            interval = shorts[int(rng.integers(len(shorts)))]
        elif shorts and rng.random() < 0.43:
            interval = shorts[int(rng.integers(len(shorts)))]
        else:
            interval = (0, width - 1)
        if i and kind == 'coverage' and coverage_pair == (i - 1, i):
            interval = intervals[-1]
        elif i != short_index and i and row == rows[i - 1] + 1 and band_colors[i] == band_colors[i - 1] and rng.random() < 0.8:
            interval = intervals[-1]
        intervals.append(interval)
    output = np.full((height, width), background, dtype=np.int8)
    for col, color in zip(cols, strand_colors):
        output[:, col] = color
    crossings = []
    for row, (left, right), color in zip(rows, intervals, band_colors):
        output[row, left:right + 1] = color
        for col, strand_color in zip(cols, strand_colors):
            if left <= col <= right:
                output[row, col] = strand_color
                crossings.append((row, col, color))
    input_grid = output.copy()
    total = len(crossings)
    if kind == 'source_like':
        hidden = rng.choice(total, size=int(rng.integers(1, total)), replace=False)
    else:
        mode = int(rng.integers(5))
        if mode == 0:
            hidden = np.arange(total)
        elif mode == 1:
            hidden = rng.choice(total, size=1, replace=False)
        elif mode == 2:
            hidden = rng.choice(total, size=int(rng.integers(max(1, 3 * total // 4), total + 1)), replace=False)
        elif mode == 3 and len(rows) >= 2:
            first = [i for i, x in enumerate(crossings) if x[0] == rows[0]]
            other = [i for i, x in enumerate(crossings) if x[0] != rows[0]]
            hidden = np.asarray(first + ([int(rng.choice(other))] if other else []), dtype=int)
        else:
            hidden = rng.choice(total, size=int(rng.integers(1, total + 1)), replace=False)
    for index in hidden:
        row, col, color = crossings[int(index)]
        input_grid[row, col] = color
    if orientation == 'horizontal':
        input_grid, output = input_grid.T.copy(), output.T.copy()
    return {'input': input_grid, 'output': output}


def _analyze_vertical(input_grid):
    row_values = [tuple(int(v) for v in row) for row in input_grid]
    exposed, frequency = Counter(row_values).most_common(1)[0]
    assert frequency > input_grid.shape[0] - frequency
    colors, amounts = np.unique(np.asarray(exposed), return_counts=True)
    assert len(colors) >= 2 and sum(amounts == max(amounts)) == 1
    background = int(colors[int(np.argmax(amounts))])
    if exposed[0] == exposed[-1] and exposed[0] != background:
        background = exposed[0]
    cols = tuple(c for c, v in enumerate(exposed) if v != background)
    assert 1 <= len(cols) <= input_grid.shape[1] - 2
    strand_colors = {c: exposed[c] for c in cols}
    noncols = [c for c in range(input_grid.shape[1]) if c not in cols]
    band_rows, band_colors = [], {}
    for r, row in enumerate(row_values):
        if row == exposed:
            continue
        values = {row[c] for c in noncols if row[c] != background}
        assert len(values) == 1
        color = values.pop()
        assert color not in strand_colors.values()
        colored = [c for c in range(input_grid.shape[1]) if row[c] == color]
        assert sum(c in noncols for c in colored) >= 2
        left, right = min(colored), max(colored)
        assert right - left >= 2
        assert all(row[c] == color for c in noncols if left <= c <= right)
        assert all(row[c] == background for c in noncols if c < left or c > right)
        assert all(row[c] in (strand_colors[c], color) for c in cols)
        band_rows.append(r)
        band_colors[r] = color
    assert band_rows
    result = input_grid.copy()
    for r in band_rows:
        for c in cols:
            if input_grid[r, c] == band_colors[r]:
                result[r, c] = strand_colors[c]
    assert not np.array_equal(result, input_grid)
    return result, (background, cols, tuple(band_rows), strand_colors, band_colors)


def _solve(input_grid, orientation):
    canonical = input_grid if orientation == 'vertical' else input_grid.T
    result, data = _analyze_vertical(canonical)
    return (result if orientation == 'vertical' else result.T.copy()), data


def _learn_orientation(train):
    valid = []
    for orientation in ('vertical', 'horizontal'):
        try:
            if all(np.array_equal(_solve(p['input'], orientation)[0], p['output']) for p in train):
                valid.append(orientation)
        except (AssertionError, ValueError, IndexError):
            pass
    assert len(valid) == 1
    return valid[0]


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    orientation = 'horizontal' if rng.random() < 0.27 else 'vertical'
    palette_bank = [tuple(int(v) for v in rng.choice(10, size=3, replace=False)) for _ in range(3)]
    puzzle = {'train': [], 'test': []}
    seen, first_background = set(), None
    for section, kinds in (
        ('train', ['source_like', 'coverage'] + ['free'] * (num_train - 2)),
        ('test', [('ordinary' if rng.random() < 0.32 else 'free')] + ['free'] * (num_test - 1)),
    ):
        for kind in kinds:
            for _ in range(60):
                pair = _make_pair(rng, kind, orientation, palette_bank, first_background)
                key = _signature(pair['input'])
                if key not in seen and key not in _SOURCE_INPUTS:
                    seen.add(key)
                    puzzle[section].append(pair)
                    if first_background is None:
                        first_background = _solve(pair['input'], orientation)[1][0]
                    break
            else:
                raise RuntimeError('could not sample a distinct pair')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen_inputs, seen_pairs = set(), set()
    for section in ('train', 'test'):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in (pair['input'], pair['output']):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert pair['input'].shape == pair['output'].shape
            inp, out = _signature(pair['input']), _signature(pair['output'])
            assert inp not in seen_inputs and (inp, out) not in seen_pairs
            assert inp not in _SOURCE_INPUTS
            seen_inputs.add(inp)
            seen_pairs.add((inp, out))
    orientation = _learn_orientation(puzzle['train'])
    simple = puzzle['train'][0]
    _, (_, simple_cols, simple_rows, _, simple_bands) = _solve(simple['input'], orientation)
    simple_input = simple['input'] if orientation == 'vertical' else simple['input'].T
    simple_noncols = [c for c in range(simple_input.shape[1]) if c not in simple_cols]
    assert len(simple_cols) in (1, 3)
    assert len(simple_cols) == 1 or tuple(np.diff(simple_cols)) == (2, 2)
    assert all(b > a + 1 for a, b in zip(simple_rows, simple_rows[1:]))
    assert all(all(simple_input[r, c] == simple_bands[r] for c in simple_noncols) for r in simple_rows)
    assert 0 < np.count_nonzero(simple['input'] != simple['output']) < len(simple_cols) * len(simple_rows)
    complex_pair = puzzle['train'][1]
    _, (_, complex_cols, complex_rows, complex_strands, complex_bands) = _solve(complex_pair['input'], orientation)
    complex_input = complex_pair['input'] if orientation == 'vertical' else complex_pair['input'].T
    complex_noncols = [c for c in range(complex_input.shape[1]) if c not in complex_cols]
    assert len(set(complex_strands.values())) > 1 and len(set(complex_bands.values())) > 1
    assert any(b == a + 1 and complex_bands[a] == complex_bands[b] and
               all((complex_input[a, c] == complex_bands[a]) == (complex_input[b, c] == complex_bands[b]) for c in complex_noncols)
               for a, b in zip(complex_rows, complex_rows[1:]))
    assert any(sum(complex_input[r, c] == complex_bands[r] for c in complex_noncols) < len(complex_noncols)
               for r in complex_rows)
    covered = {'strand': False, 'band': False, 'adjacent': False, 'short': False}
    needed = dict.fromkeys(covered, False)
    train_visible = False
    for section in ('train', 'test'):
        for pair in puzzle[section]:
            predicted, data = _solve(pair['input'], orientation)
            assert np.array_equal(predicted, pair['output'])
            _, cols, rows, strand_colors, band_colors = data
            canonical = pair['input'] if orientation == 'vertical' else pair['input'].T
            canonical_output = pair['output'] if orientation == 'vertical' else pair['output'].T
            changed = list(zip(*np.where(canonical != canonical_output)))
            assert changed and all(r in rows and c in cols and canonical[r, c] == band_colors[r] for r, c in changed)
            noncols = [c for c in range(canonical.shape[1]) if c not in cols]
            features = {
                'strand': len(set(strand_colors.values())) > 1,
                'band': len(set(band_colors.values())) > 1,
                'adjacent': any(b == a + 1 for a, b in zip(rows, rows[1:])),
                'short': any(sum(canonical[r, c] == band_colors[r] for c in noncols) < len(noncols) for r in rows),
            }
            if section == 'train':
                for key, value in features.items():
                    covered[key] |= value
                train_visible |= any(canonical[r, c] == strand_colors[c] for r in rows for c in cols)
            else:
                for key, value in features.items():
                    needed[key] |= value
    assert train_visible
    assert all(not needed[key] or covered[key] for key in needed)


def _puzzle_signature(puzzle):
    return tuple(tuple((_signature(p['input']), _signature(p['output'])) for p in puzzle[s]) for s in ('train', 'test'))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_signature(p) for p in puzzles}) >= 40
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    shapes, counts, orientations, backgrounds, layouts = set(), set(), set(), set(), set()
    features = Counter()
    varied_backgrounds = varied_shapes = varied_counts = palette_reuse = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        orientation = _learn_orientation(puzzle['train'])
        orientations.add(orientation)
        local_backgrounds = set()
        local_shapes, local_strands, local_bands, local_palettes = set(), set(), set(), []
        for section in ('train', 'test'):
            for pair in puzzle[section]:
                inp, out = pair['input'], pair['output']
                prediction, data = _solve(inp.copy(), orientation)
                assert np.array_equal(prediction, out)
                background, cols, rows, strand_colors, band_colors = data
                canonical = inp if orientation == 'vertical' else inp.T
                noncols = [c for c in range(canonical.shape[1]) if c not in cols]
                local_backgrounds.add(background)
                local_shapes.add(inp.shape)
                local_strands.add(len(cols))
                local_bands.add(len(rows))
                backgrounds.add(background)
                shapes.add(inp.shape)
                counts.add((len(cols), len(rows)))
                layouts.add((inp.shape, cols, rows))
                changes = int(np.count_nonzero(inp != out))
                features['one_change'] += changes == 1
                crossing_count = 0
                row_totals, row_hidden, footprints = [], [], []
                for r in rows:
                    band_cells = [c for c in noncols if canonical[r, c] == band_colors[r]]
                    footprints.append(tuple(band_cells))
                    involved = cols if len(band_cells) == len(noncols) else [
                        c for c in cols if min(band_cells) <= c <= max(band_cells)
                    ]
                    crossing_count += len(involved)
                    row_totals.append(len(involved))
                    row_hidden.append(sum(canonical[r, c] == band_colors[r] for c in involved))
                features['all_covered'] += changes == crossing_count
                features['mixed_band'] += any(h == t for h, t in zip(row_hidden, row_totals)) and any(0 < h < t for h, t in zip(row_hidden, row_totals))
                features['visible_band'] += any(h == 0 for h in row_hidden)
                features['heavy_cover'] += changes >= 0.75 * crossing_count
                features['thick_band'] += any(b == a + 1 and band_colors[a] == band_colors[b] and footprints[i] == footprints[i + 1]
                                              for i, (a, b) in enumerate(zip(rows, rows[1:])))
                features['adjacent_band'] += any(b == a + 1 for a, b in zip(rows, rows[1:]))
                features['neighbor_strand'] += any(b == a + 1 for a, b in zip(cols, cols[1:]))
                features['multicolor'] += len(set(strand_colors.values())) > 1 or len(set(band_colors.values())) > 1
                features['short'] += any(sum(canonical[r, c] == band_colors[r] for c in noncols) < len(noncols) for r in rows)
                features['wide'] += canonical.shape[1] > canonical.shape[0]
                features['wide_block'] += len(cols) > canonical.shape[1] / 2
                features[f'{section}_single'] += len(cols) == 1
                features[f'{section}_two'] += len(cols) == 2
                features[f'{section}_uneven'] += len(cols) >= 3 and len(set(np.diff(cols))) > 1
                features[f'{section}_interior_uneven'] += len(cols) >= 3 and len(set(np.diff(cols))) > 1 and cols[0] > 0 and cols[-1] < canonical.shape[1] - 1
                features[f'{section}_wide_gap'] += len(cols) >= 2 and max(np.diff(cols)) >= 4
                features[f'{section}_edge_three'] += len(cols) == 3 and (cols[0] == 0 or cols[-1] == canonical.shape[1] - 1)
                features[f'{section}_adjacent'] += any(b == a + 1 for a, b in zip(rows, rows[1:]))
                features[f'{section}_short'] += any(sum(canonical[r, c] == band_colors[r] for c in noncols) < len(noncols) for r in rows)
                features[f'{section}_multi'] += len(set(strand_colors.values())) > 1 or len(set(band_colors.values())) > 1
                features[f'{section}_one_band'] += len(rows) == 1
                features[f'{section}_many_bands'] += len(rows) >= 6
                if section == 'test':
                    features['test_ordinary'] += (len(cols) == 3 and tuple(np.diff(cols)) == (2, 2)
                                                  and not any(b == a + 1 for a, b in zip(rows, rows[1:]))
                                                  and all(len(v) == len(noncols) for v in footprints)
                                                  and len(set(strand_colors.values())) == len(set(band_colors.values())) == 1)
                if len(set(strand_colors.values())) == len(set(band_colors.values())) == 1:
                    local_palettes.append((background, next(iter(strand_colors.values())), next(iter(band_colors.values()))))
        varied_backgrounds += len(local_backgrounds) > 1
        varied_shapes += len(local_shapes) > 1
        varied_counts += len(local_strands) > 1 and len(local_bands) > 1
        palette_reuse += len(set(local_palettes)) < len(local_palettes)
    assert len(shapes) >= 65 and len(layouts) >= 150 and len(counts) >= 20
    assert orientations == {'vertical', 'horizontal'}
    assert len(backgrounds) >= 8 and min(varied_backgrounds, varied_shapes, varied_counts) >= 30
    assert palette_reuse >= 10
    assert min(features.values()) >= 5
    for train, test in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(1000 + train * 10 + test, train, test)
        assert (len(p['train']), len(p['test'])) == (train, test)
    for train, test in ((None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(2000 + (train or 0) * 10 + (test or 0), train, test))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(7, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal pair counts were accepted')
    assert _puzzle_signature(generate_puzzle(123456)) == _puzzle_signature(generate_puzzle(123456))
    assert _puzzle_signature(generate_puzzle(123456, 7, 3)) == _puzzle_signature(generate_puzzle(123456, 7, 3))
