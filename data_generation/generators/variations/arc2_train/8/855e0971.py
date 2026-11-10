"""Generate ARC band puzzles with sparse seeds completed into contained crosscuts."""

import hashlib
import numpy as np


# Hashes of the five original inputs and outputs, including their dimensions.
# Shapes themselves are deliberately unrestricted.
_SOURCE_GRID_HASHES = frozenset({
    '5e7f41370904ba3759f1484357e17f1ac9d3af4511d66d183dc73d314d17847a',
    '44dd7df9e34aa1c19d8b252194486a0d0515a65f1a82f8b1f3b1f7f1b65ec937',
    'cfe9fce0727f72ababe78b27455ce0ede7c6460f3559ef9c6a325e36b0358f92',
    '9914e55da537b89e34e2b44ebb72feba5fa71714c014d2df1530333e5e4d2a33',
    '82ee1bf04ce9129a1cd102b0af97a80978b4fc94e79fc487b37eb97051f3d64c',
    'e201b8c9c7269960826784c03420a225c3f236732e2a7a233c76d8d48cd325fb',
    'd57309258cc54e7efdf712f72c681f8964597d62dc31a8e536a4a84858d014af',
    'c5af76b044a941833bd9f8352e47d1179d9941fd1bd8cc8d968dc8cd8c59bafb',
    '6bca5fa7d4f07aec30a689d1458cae116fef6d0f72e9ac340962a2a6ea32c412',
    'd6a68c0c047a83d7c5d2bebf9e9a3d30a391b5843a2fad8c048c324c2191cef2',
})


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer or None')
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f'{name} must be between {low} and {high}')
    return value


def _resolve_counts(seed, num_train, num_test):
    train = _count('num_train', num_train, 2, 10)
    test = _count('num_test', num_test, 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError('num_train must exceed num_test')
    elif train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + ((seed // 4 + 3 * seed) % (10 - test))
    elif train is None:
        train = test + 1 + ((seed // 5 + 5 * seed) % (10 - test))
    else:
        test = 1 + ((seed // 3 + seed) % min(4, train - 1))
    return train, test


def _grid_hash(grid):
    h, w = grid.shape
    data = f'{h}x{w}:' + ''.join(str(int(v)) for v in grid.flat)
    return hashlib.sha256(data.encode()).hexdigest()


def _signature(grid):
    return (grid.shape, grid.tobytes())


def _thicknesses(rng, count, regime):
    mode = int(rng.integers(0, 3))
    if mode == 0:  # Exactly equal bands, including source-like thicknesses.
        size = int(rng.integers(2, min(8, 30 // count) + 1))
        return (size,) * count
    cap = 26 if regime == 'source_train' else 30
    total = int(rng.integers(2 * count, cap + 1))
    extra = total - 2 * count
    if mode == 1:
        allocation = rng.multinomial(extra, np.ones(count) / count)
    else:  # A conspicuously thin/thick mixture.
        allocation = np.zeros(count, dtype=int)
        if extra:
            thick = int(rng.integers(0, count))
            allocation[thick] = max(1, int(round(extra * float(rng.uniform(.55, .9)))))
            allocation[thick] = min(extra, allocation[thick])
            rest = extra - allocation[thick]
            if rest:
                allocation += rng.multinomial(rest, np.ones(count) / count)
    return tuple(int(v + 2) for v in allocation)


def _fills(rng, count, marker):
    available = [v for v in range(10) if v != marker]
    colors = []
    for index in range(count):
        repeated = [v for v in colors[:-1] if v != colors[-1]] if colors else []
        fresh = [v for v in available if v not in colors]
        if repeated and rng.random() < .42:
            color = int(rng.choice(repeated))
        else:
            pool = fresh or [v for v in available if not colors or v != colors[-1]]
            color = int(rng.choice(pool))
        colors.append(color)
    return tuple(colors)


def _marker_counts(rng, count, length, mode, area):
    maximum = 4 if length >= 18 else (3 if length >= 12 else 2)
    weights = np.array([.19, .48, .25, .07, .01][:maximum + 1], dtype=float)
    weights /= weights.sum()
    counts = [int(rng.choice(maximum + 1, p=weights)) for _ in range(count)]
    if mode == 'ones':
        counts = [1] * count
    if mode == 'multi' and max(counts) < 2:
        counts[int(rng.integers(count))] = 2
    if mode == 'unmarked' and 0 not in counts:
        counts[int(rng.integers(count))] = 0
    if not any(counts):
        counts[int(rng.integers(count))] = 1
    # Keep the interrupting color visibly sparse even for tiny/thin canvases.
    limit = max(2, int(area * .12))
    while sum(counts) > limit:
        candidates = [i for i, n in enumerate(counts) if n > 1]
        if not candidates:
            candidates = [i for i, n in enumerate(counts) if n == 1 and sum(counts) > 1]
        counts[max(candidates, key=lambda i: (counts[i], i))] -= 1
    return tuple(counts)


def _positions(rng, length, count):
    if count == 0:
        return ()
    # Most markers are interior; some lie on the first or last grid line.
    for _ in range(120):
        endpoint = int(rng.integers(0, 2)) if rng.random() < .08 else None
        pool = np.arange(length) if endpoint is not None else np.arange(1, length - 1)
        if endpoint is not None:
            endpoint = (length - 1) * endpoint
            pool = pool[pool != endpoint]
            others = rng.choice(pool, count - 1, replace=False)
            values = sorted([endpoint] + [int(v) for v in others])
        else:
            values = sorted(int(v) for v in rng.choice(pool, count, replace=False))
        if all(b - a > 1 for a, b in zip(values, values[1:])):
            return tuple(values)
    # Bounded fallback still allows endpoint markers.
    return tuple(range(0, 2 * count, 2))


def _make_pair(rng, regime, orientation, count, marker, marker_mode):
    thicknesses = _thicknesses(rng, count, regime)
    length = int(rng.integers(8, 31))
    total = sum(thicknesses)
    shape = (total, length) if orientation == 'horizontal' else (length, total)
    fills = _fills(rng, count, marker)
    base = np.empty(shape, dtype=np.int8)
    bounds = []
    start = 0
    for thickness, color in zip(thicknesses, fills):
        end = start + thickness
        if orientation == 'horizontal':
            base[start:end, :] = color
        else:
            base[:, start:end] = color
        bounds.append((start, end, color))
        start = end
    counts = _marker_counts(rng, count, length, marker_mode, base.size)
    for _ in range(150):
        coords = []
        for (start, end, _), number in zip(bounds, counts):
            for position in _positions(rng, length, number):
                offset = int(rng.integers(start, end))
                coords.append((offset, position) if orientation == 'horizontal'
                              else (position, offset))
        if all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1
               for i, a in enumerate(coords) for b in coords[i + 1:]):
            break
    else:
        raise RuntimeError('could not isolate seed markers')
    inp = base.copy()
    for row, col in coords:
        inp[row, col] = marker
    out = _complete(inp, marker, orientation, tuple(bounds))
    return {'input': inp, 'output': out}


def _complete(inp, marker, orientation, bands):
    out = inp.copy()
    for start, end, _ in bands:
        if orientation == 'horizontal':
            positions = np.flatnonzero(np.any(inp[start:end, :] == marker, axis=0))
            out[start:end, positions] = marker
        else:
            positions = np.flatnonzero(np.any(inp[:, start:end] == marker, axis=1))
            out[positions, start:end] = marker
    return out


def _parse_for_marker(inp, marker):
    candidates = []
    for orientation in ('horizontal', 'vertical'):
        lines = inp if orientation == 'horizontal' else inp.T
        profile = []
        for line in lines:
            fill = np.unique(line[line != marker])
            if len(fill) != 1:
                break
            profile.append(int(fill[0]))
        if len(profile) != len(lines):
            continue
        bands = []
        start = 0
        for index in range(1, len(profile) + 1):
            if index == len(profile) or profile[index] != profile[start]:
                bands.append((start, index, profile[start]))
                start = index
        if 2 <= len(bands) <= 6 and all(end - start >= 2 for start, end, _ in bands):
            candidates.append((orientation, tuple(bands)))
    return candidates


def _inspect(inp, out=None):
    # Infer the interrupting color from the input alone. Every row/column must
    # retain a visible fill, so a filled band cannot masquerade as markers.
    possibilities = []
    for marker in (int(v) for v in np.unique(inp)):
        marker_cells = np.argwhere(inp == marker)
        if len(marker_cells) > max(2, int(inp.size * .12)):
            continue
        if any(max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1
               for i, a in enumerate(marker_cells) for b in marker_cells[i + 1:]):
            continue
        for orientation, bands in _parse_for_marker(inp, marker):
            counts = []
            positions_by_band = []
            offsets_by_band = []
            for start, end, _ in bands:
                coords = np.argwhere(inp[start:end, :] == marker) if orientation == 'horizontal' else np.argwhere(inp[:, start:end] == marker)
                positions = tuple(int(v) for v in (coords[:, 1] if orientation == 'horizontal' else coords[:, 0]))
                offsets = tuple(int(v) for v in (coords[:, 0] if orientation == 'horizontal' else coords[:, 1]))
                if len(set(positions)) != len(positions):
                    break
                counts.append(len(positions))
                positions_by_band.append(positions)
                offsets_by_band.append(offsets)
            else:
                if any(counts) and max(counts) <= 4:
                    possibilities.append(dict(marker=marker, orientation=orientation,
                                              bands=bands, counts=tuple(counts),
                                              positions=tuple(positions_by_band),
                                              offsets=tuple(offsets_by_band),
                                              shape=inp.shape))
    _require(len(possibilities) == 1, 'input must reveal one sparse marker and band partition')
    info = possibilities[0]
    if out is not None:
        expected = _complete(inp, info['marker'], info['orientation'], info['bands'])
        _require(np.array_equal(out, expected), 'output must complete precisely the contained crosscuts')
        _require(np.any(inp != out), 'a pair must change at least one cell')
    return info


def _validate_impl(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle schema')
    _require(isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list), 'sections must be lists')
    train_count, test_count = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             'invalid pair counts')
    seen = set()
    infos = {'train': [], 'test': []}
    for section in ('train', 'test'):
        for pair in puzzle[section]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair schema')
            inp, out = pair['input'], pair['output']
            for grid in (inp, out):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, 'grid must be a 2-D array')
                _require(np.issubdtype(grid.dtype, np.integer), 'grid dtype must be integer')
                _require(all(1 <= v <= 30 for v in grid.shape), 'invalid grid size')
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9, 'invalid ARC color')
                _require(_grid_hash(grid) not in _SOURCE_GRID_HASHES, 'exact source grid copied')
            _require(inp.shape == out.shape, 'shape must remain unchanged')
            signature = _signature(inp)
            _require(signature not in seen, 'duplicate input')
            seen.add(signature)
            infos[section].append(_inspect(inp, out))
    train = infos['train']
    test = infos['test']
    train_markers = {v['marker'] for v in train}
    if len(train_markers) == 1:
        _require(all(v['marker'] in train_markers for v in test), 'untrained marker color')
    else:
        _require(train_count >= 3 and len(train_markers) >= 3,
                 'varying marker role needs several distinct demonstrations')
    _require(any(max(v['counts']) >= 2 for v in train), 'train must demonstrate independent multiple cuts')
    _require(any(0 in v['counts'] for v in train), 'train must demonstrate an untouched band')
    train_orientations = {v['orientation'] for v in train}
    _require(all(v['orientation'] in train_orientations for v in test),
             'test orientation must be demonstrated')
    _require(all(not (0 in v['counts']) or any(0 in t['counts'] for t in train) for v in test),
             'untouched test band lacks a training example')
    # The test rule is identical; variation lies in dimensions, counts,
    # arrangement, endpoints, and colors rather than a fixed size split.
    return infos


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _validate_impl(puzzle)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(f'invalid generated puzzle: {exc}') from exc


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    orientation_mode = int(rng.choice(3, p=[.24, .18, .58]))
    if orientation_mode == 0:
        train_orientations = ['horizontal'] * train_count
    elif orientation_mode == 1:
        train_orientations = ['vertical'] * train_count
    else:
        train_orientations = ['horizontal', 'vertical'] + [
            ('horizontal' if rng.random() < .5 else 'vertical')
            for _ in range(train_count - 2)]
        rng.shuffle(train_orientations)
    choices = sorted(set(train_orientations))
    test_orientations = [str(rng.choice(choices)) for _ in range(test_count)]
    if len(choices) == 2 and test_count >= 2 and rng.random() < .48:
        test_orientations[:2] = ['horizontal', 'vertical']
        rng.shuffle(test_orientations)
    train_counts = [int(rng.integers(2, 4)), int(rng.integers(3, 5))]
    train_counts += [int(rng.integers(2, 5)) for _ in range(train_count - 2)]
    train_modes = ['multi', 'unmarked'] + ['ones' if rng.random() < .18 else 'general'
                                           for _ in range(train_count - 2)]
    order = rng.permutation(train_count)
    train_specs = [('source_train', train_orientations[int(i)], train_counts[int(i)], train_modes[int(i)])
                   for i in order]
    test_specs = [('target_test', test_orientations[i],
                   int(rng.choice([2, 3, 4, 5, 6], p=[.18, .19, .30, .22, .11])),
                   'ones' if rng.random() < .19 else 'general')
                  for i in range(test_count)]
    varying_markers = train_count >= 3 and rng.random() < .34
    color_order = [int(v) for v in rng.permutation(10)]
    marker_colors = [color_order[i % 10] if varying_markers else color_order[0]
                     for i in range(train_count + test_count)]
    seeds = rng.integers(0, np.iinfo(np.int64).max, size=len(marker_colors), dtype=np.int64)
    puzzle = {'train': [], 'test': []}
    used = set()
    for index, ((regime, orientation, band_count, mode), marker, pair_seed) in enumerate(
            zip(train_specs + test_specs, marker_colors, seeds)):
        pair_rng = np.random.default_rng(int(pair_seed))
        for _ in range(80):
            try:
                pair = _make_pair(pair_rng, regime, orientation, band_count, marker, mode)
                _inspect(pair['input'], pair['output'])
            except (AssertionError, RuntimeError):
                continue
            signature = _signature(pair['input'])
            if signature in used or any(_grid_hash(grid) in _SOURCE_GRID_HASHES
                                        for grid in pair.values()):
                continue
            used.add(signature)
            puzzle['train' if index < train_count else 'test'].append(pair)
            break
        else:
            raise RuntimeError('could not build a unique, interpretable pair')
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(a, b):
    return all(len(a[section]) == len(b[section]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[section], b[section])
        for k in ('input', 'output')) for section in ('train', 'test'))


def self_test() -> None:
    _require(_puzzles_equal(generate_puzzle(314159), generate_puzzle(314159)), 'automatic determinism')
    _require(_puzzles_equal(generate_puzzle(271828, 7, 3), generate_puzzle(271828, 7, 3)), 'explicit determinism')
    for train, test in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(9000 + train + test, train, test)
        _require((len(p['train']), len(p['test'])) == (train, test), 'explicit counts')
    for train, test in ((None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(8100 + (train or 0) + (test or 0), train, test)
        _require((train is None or len(p['train']) == train) and
                 (test is None or len(p['test']) == test), 'partial counts')
    for kwargs in ({'num_train': 1}, {'num_train': 11}, {'num_test': 0}, {'num_test': 5},
                   {'num_train': 2, 'num_test': 2}, {'num_train': 2, 'num_test': 4}):
        try:
            generate_puzzle(21, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f'illegal counts accepted: {kwargs}')
    observed = {key: set() for key in ('train_counts', 'test_counts', 'test_orientations',
                'test_band_counts', 'shapes', 'layouts', 'palettes', 'markers')}
    flags = {key: False for key in ('equal', 'contrast', 'repeat_fill', 'endpoint',
             'unmarked_test', 'multi3', 'varying_marker', 'same_offset',
             'source_shape', 'test_not_taller', 'mixed_test')}
    within = {key: 0 for key in ('shape', 'band_count', 'marker_counts', 'palette')}
    train_four = same_orientation = mixed_orientation = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        infos = _validate_impl(puzzle)
        all_info = infos['train'] + infos['test']
        train_four += any(len(v['bands']) == 4 for v in infos['train'])
        same_orientation += len({v['orientation'] for v in all_info}) == 1
        mixed_orientation += len({v['orientation'] for v in all_info}) == 2
        within['shape'] += len({v['shape'] for v in all_info}) > 1
        within['band_count'] += len({len(v['bands']) for v in all_info}) > 1
        within['marker_counts'] += len({v['counts'] for v in all_info}) > 1
        within['palette'] += len({tuple(b[2] for b in v['bands']) for v in all_info}) > 1
        observed['train_counts'].add(len(puzzle['train']))
        observed['test_counts'].add(len(puzzle['test']))
        observed['markers'].update(v['marker'] for v in infos['train'] + infos['test'])
        flags['varying_marker'] |= len({v['marker'] for v in infos['train']}) > 1
        flags['mixed_test'] |= len({v['orientation'] for v in infos['test']}) > 1
        flags['test_not_taller'] |= min(v['shape'][0] for v in infos['test']) <= max(v['shape'][0] for v in infos['train'])
        for section in ('train', 'test'):
            for info in infos[section]:
                bands = info['bands']
                thickness = tuple(end - start for start, end, _ in bands)
                fills = tuple(color for _, _, color in bands)
                observed['shapes'].add(info['shape'])
                observed['layouts'].add((info['orientation'], info['shape'], thickness, info['counts']))
                observed['palettes'].add((info['marker'], fills))
                flags['equal'] |= len(set(thickness)) == 1
                flags['contrast'] |= max(thickness) - min(thickness) >= 4
                flags['repeat_fill'] |= len(set(fills)) < len(fills)
                flags['multi3'] |= max(info['counts']) >= 3
                flags['same_offset'] |= any(len(offsets) > 1 and len(set(offsets)) < len(offsets)
                                            for offsets in info['offsets'])
                length = info['shape'][1 if info['orientation'] == 'horizontal' else 0]
                flags['endpoint'] |= any(0 in positions or length - 1 in positions for positions in info['positions'])
                flags['source_shape'] |= info['shape'] in {(13, 15), (14, 13), (14, 15), (15, 19), (17, 15)}
                if section == 'test':
                    observed['test_orientations'].add(info['orientation'])
                    observed['test_band_counts'].add(len(bands))
                    flags['unmarked_test'] |= 0 in info['counts']
    _require(len(observed['train_counts']) >= 4 and len(observed['test_counts']) >= 3, 'count diversity')
    _require(observed['test_orientations'] == {'horizontal', 'vertical'}, 'test orientation diversity')
    _require({2, 3, 4, 5, 6}.issubset(observed['test_band_counts']), 'test band count diversity')
    _require(len(observed['shapes']) >= 55 and len(observed['layouts']) >= 150 and
             len(observed['palettes']) >= 100 and len(observed['markers']) >= 7, 'structural diversity')
    _require(all(flags.values()), f'missing review coverage: {[k for k, v in flags.items() if not v]}')
    _require(train_four >= 10 and same_orientation >= 5 and mixed_orientation >= 5,
             'train band count or orientation regimes collapsed')
    _require(all(value >= 38 for value in within.values()), 'within-puzzle variation collapsed')
    _require(not _puzzles_equal(generate_puzzle(0), generate_puzzle(1)), 'cross-seed difference')
