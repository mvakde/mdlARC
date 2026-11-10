"""Unary keys select a descending length rank among equal-capped bars.

Scene conventions are visible in each input. Puzzle-level parameters choose the
teaching regime, not a hidden color dictionary or an obligatory geometry.
"""
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def _resolve_counts(seed, rng, num_train, num_test):
    train = _checked_count("num_train", num_train, 2, 10)
    test = _checked_count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        minimum = max(2, test + 1)
        train = minimum + (seed // 4) % (11 - minimum)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _grid_signature(grid):
    return (grid.shape, tuple(int(v) for v in grid.flat))


def _puzzle_signature(puzzle):
    return tuple(tuple((_grid_signature(p['input']), _grid_signature(p['output']))
                       for p in puzzle[s]) for s in ('train', 'test'))


def _validate_grid(grid, name):
    _require(isinstance(grid, np.ndarray), f"{name} must be a NumPy array")
    _require(grid.ndim == 2, f"{name} must be two dimensional")
    _require(all(1 <= d <= 30 for d in grid.shape), f"{name} has invalid dimensions")
    _require(np.issubdtype(grid.dtype, np.integer), f"{name} must have integer dtype")
    _require(bool(np.all((0 <= grid) & (grid <= 9))), f"{name} has invalid colors")


def _segments(indices):
    if len(indices) == 0:
        return []
    return np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)


def _parse_canonical(grid):
    """Decode vertical bars with their shared baseline below the unary key.

    Column segmentation, rather than connected components, keeps adjacent bars
    distinct. A key and a bar may occupy the same column with a blank gap.
    No corner, reserved strip, particular color, or canvas size is assumed.
    """
    h, w = grid.shape
    frequencies = np.bincount(grid.ravel().astype(int), minlength=10)
    bg = int(np.argmax(frequencies))
    _require(np.sum(frequencies == frequencies[bg]) == 1, "background must be obvious")
    keys, bars = [], []
    for c in range(w):
        for run in _segments(np.flatnonzero(grid[:, c] != bg)):
            a, b = int(run[0]), int(run[-1])
            values = grid[a:b + 1, c]
            cap = int(values[0])
            if np.all(values == cap):
                keys.append({'control': cap, 'rank': b - a + 1, 'top': a, 'column': c})
            else:
                _require(b - a >= 2, "bar needs two caps and an interior")
                _require(int(values[-1]) == cap, "unequal bar caps")
                payload = int(values[1])
                _require(payload != cap and np.all(values[1:-1] == payload),
                         "bar interior must be uniform and distinct")
                bars.append({'control': cap, 'payload': payload, 'top': a,
                             'bottom': b, 'column': c, 'length': b - a + 1})
    _require(1 <= len(keys) <= 8 and bars, "missing compact key or rack")
    controls = tuple(k['control'] for k in keys)
    _require(len(set(controls)) == len(controls), "one unary run per cap identity")
    _require(len({k['top'] for k in keys}) == 1, "key runs must align")
    _require(len({k['column'] for k in keys}) == len(keys), "key runs must be separate")
    _require(keys[-1]['column'] - keys[0]['column'] <= 3 * len(keys), "key is scattered")
    _require(len({b['bottom'] for b in bars}) == 1, "bars must share a baseline")
    _require(len({b['column'] for b in bars}) == len(bars), "one rack bar per column")
    baseline = bars[0]['bottom']
    _require(max(k['top'] + k['rank'] for k in keys) <= baseline, "key must precede baseline")
    _require(all(b['control'] in controls for b in bars), "bar cap lacks a key")
    _require(all(b['payload'] not in controls for b in bars), "pair-local color roles overlap")
    for key in keys:
        for bar in bars:
            if abs(key['column'] - bar['column']) == 1:
                _require(key['top'] + key['rank'] <= bar['top'] or key['top'] > bar['bottom'],
                         "key must remain visibly separate from the rack")
    by_control = {c: [b for b in bars if b['control'] == c] for c in controls}
    ranks = {k['control']: k['rank'] for k in keys}
    for c, family in by_control.items():
        _require(len(family) >= 2, "a ranking family needs competitors")
        _require(len({b['length'] for b in family}) == len(family), "tied lengths")
        _require(1 <= ranks[c] <= len(family), "requested rank does not exist")
    return {'height': h, 'width': w, 'background': bg, 'controls': controls,
            'key_ranks': ranks, 'keys': keys, 'bars': bars, 'baseline': baseline,
            'bars_by_control': by_control}


def _decode_input(grid):
    candidates = []
    for rotation in range(4):
        canonical = np.rot90(grid, -rotation)
        try:
            metadata = _parse_canonical(canonical)
        except AssertionError:
            continue
        candidates.append((rotation, canonical, metadata))
    _require(len(candidates) == 1, "input must have one unambiguous bar axis and key")
    return candidates[0]


def _expected_canonical(grid, metadata):
    expected = grid.copy()
    selected = []
    for cap, family in metadata['bars_by_control'].items():
        ordered = sorted(family, key=lambda b: b['length'], reverse=True)
        chosen = ordered[metadata['key_ranks'][cap] - 1]
        expected[chosen['top'] + 1:chosen['bottom'], chosen['column']] = cap
        selected.append(chosen)
    return expected, selected


def solve_input(grid):
    """Answer using only visible pixels; no generation record is consulted."""
    rotation, canonical, metadata = _decode_input(grid)
    expected, _ = _expected_canonical(canonical, metadata)
    return np.ascontiguousarray(np.rot90(expected, rotation))


def _analyze_pair(pair):
    inp, out = pair['input'], pair['output']
    _require(inp.shape == out.shape, "dimensions must be preserved")
    rotation, canonical, metadata = _decode_input(inp)
    expected, selected = _expected_canonical(canonical, metadata)
    _require(np.array_equal(np.rot90(out, -rotation), expected), "incorrect ordinal fill")
    _require(np.any(inp != out), "each example must demonstrate the rule")
    return {'orientation': rotation, 'input': canonical, 'metadata': metadata,
            'selected': selected}


def _check_source_train(analyses):
    """Teaching coverage is distinct from the per-scene transformation checks."""
    ranks = {r for a in analyses for r in a['metadata']['key_ranks'].values()}
    _require({1, 2} <= ranks, "training must demonstrate descending ordinal progression")
    _require(any(a['metadata']['key_ranks'][c] == 2 and len(family) >= 3
                 for a in analyses for c, family in a['metadata']['bars_by_control'].items()),
             "training needs a non-extreme ordinal selection")
    cap_ranks = {}
    for a in analyses:
        for cap, rank in a['metadata']['key_ranks'].items():
            cap_ranks.setdefault(cap, set()).add(rank)
    _require(any({1, 2} <= ranks for ranks in cap_ranks.values()),
             "training must contrast the same cap at different ranks")
    _require(not _training_shortcuts(analyses), "training leaves a spatial or fixed-rank shortcut")
    return ranks


def _training_shortcuts(analyses):
    surviving = []
    for rule in ('left', 'right', 'second_shortest'):
        consistent = True
        for a in analyses:
            m = a['metadata']
            for chosen in a['selected']:
                cap = chosen['control']
                family = m['bars_by_control'][cap]
                if rule == 'second_shortest':
                    alternative = sorted(family, key=lambda b: b['length'])[1]
                else:
                    ordered = sorted(family, key=lambda b: b['column'], reverse=rule == 'right')
                    alternative = ordered[m['key_ranks'][cap] - 1]
                consistent &= alternative['column'] == chosen['column']
        if consistent:
            surviving.append(rule)
    return surviving


def _check_target_test(analysis, training_ranks):
    # The source itself extrapolates 1,2 -> 3. Extend at most one step, only
    # when every preceding ordinal is demonstrated somewhere in training.
    for rank in analysis['metadata']['key_ranks'].values():
        _require(rank in training_ranks or set(range(1, rank)) <= training_ranks,
                 "unfamiliar rank lacks a demonstrated progression")


def predict_from_training(train_pairs, test_input):
    """Check the demonstrations, then predict without looking at a test label."""
    analyses = [_analyze_pair(p) for p in train_pairs]
    training_ranks = {r for a in analyses for r in a['metadata']['key_ranks'].values()}
    rotation, canonical, metadata = _decode_input(test_input)
    _check_target_test({'metadata': metadata}, training_ranks)
    expected, _ = _expected_canonical(canonical, metadata)
    return np.ascontiguousarray(np.rot90(expected, rotation))


def _is_source_copy(pair):
    # Exact content comparison against the four evidence fixtures below.
    for inputs, outputs in _SOURCE_FIXTURES:
        if pair['input'].shape != (len(inputs), len(inputs[0])):
            continue
        if (all(''.join(str(int(v)) for v in row) == text for row, text in zip(pair['input'], inputs))
                and all(''.join(str(int(v)) for v in row) == text for row, text in zip(pair['output'], outputs))):
            return True
    return False


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _require(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
             and len(puzzle['train']) > len(puzzle['test']), "invalid pair counts")
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            for role in pair:
                _validate_grid(pair[role], role)
            signature = _grid_signature(pair['input'])
            _require(signature not in seen, "duplicate input (and hence pair)")
            seen.add(signature)
            _require(not _is_source_copy(pair), "exact source copy")
            analyses[split].append(_analyze_pair(pair))
    ranks = _check_source_train(analyses['train'])
    for a in analyses['test']:
        _check_target_test(a, ranks)
    # Literal colors, payload sharing, orientation, margins and family counts
    # are input-visible. Consistency means the same transformation, not the
    # same scene parameters. There are no additional split geometry bans.


def sample_puzzle_parameters(rng):
    level = int(rng.choice(3, p=[0.22, 0.43, 0.35]))
    shift = str(rng.choice(['recombine', 'families', 'mixing', 'orientation', 'rank', 'source'],
                           p=[0.35, 0.12, 0.14, 0.12, 0.15, 0.12]))
    palette = [int(c) for c in rng.permutation(10)]
    return {'level': level, 'shift': shift, 'max_families': (2, 4, 5)[level],
            'max_rank': (2, 3, 4)[level], 'palette': palette,
            'orientation': int(rng.integers(4)),
            'mixed_orientation': level > 0 and bool(rng.integers(2)),
            'mixed_background': level > 0 and bool(rng.integers(2)),
            'rename': level > 0 and bool(rng.integers(2))}


def _scene_specs(rng, parameters, train_count, test_count):
    p = parameters
    specs = {'train': [], 'test': []}
    # Contrast positions vary. Even a two-training-pair request has a clear
    # 1/2 anchor and, when needed, a separate rank-three demonstration.
    contrast_indices = [int(i) for i in rng.choice(train_count, 2, replace=False)]
    for split, count in (('train', train_count), ('test', test_count)):
        for index in range(count):
            target = split == 'test'
            shift = p['shift']
            max_f = p['max_families']
            if not target and shift in ('source', 'families'):
                max_f = min(2, max_f)
            f = int(rng.integers(1, max_f + 1))
            if target and shift in ('source', 'families'):
                f = int(rng.integers(min(3, p['max_families']), p['max_families'] + 1))
            rank_limit = p['max_rank']
            if not target and shift in ('source', 'rank'):
                rank_limit = max(2, rank_limit - 1)
            # All-longest and repeated ranks coexist with independent values.
            ranks = [int(rng.integers(1, rank_limit + 1)) for _ in range(f)]
            if rng.random() < 0.18:
                ranks = [int(rng.integers(1, rank_limit + 1))] * f
            anchor = None
            if not target and index in contrast_indices:
                anchor = contrast_indices.index(index) + 1
                ranks[0] = anchor
                if p['max_rank'] == 4 and anchor == 1:
                    if len(ranks) == 1:
                        ranks.append(3)
                    else:
                        ranks[1] = 3
                    f = len(ranks)
            mixing = str(rng.choice(['grouped', 'clusters', 'shuffle']))
            if p['level'] == 0:
                mixing = str(rng.choice(['grouped', 'clusters'], p=[0.8, 0.2]))
            if shift in ('source', 'mixing'):
                mixing = 'shuffle' if target else 'grouped'
            rotation = p['orientation']
            if p['mixed_orientation'] and shift != 'orientation':
                rotation = int(rng.integers(4))
            if shift == 'orientation' and target:
                rotation = (rotation + int(rng.choice([1, 3]))) % 4
            specs[split].append({'families': f, 'ranks': ranks, 'anchor': anchor,
                                 'mixing': mixing, 'rotation': rotation})
    return specs


def _family_sequence(rng, counts, mode):
    order = [int(f) for f in rng.permutation(len(counts))]
    if mode == 'grouped':
        return [f for f in order for _ in range(counts[f])]
    if mode == 'shuffle':
        sequence = [f for f in order for _ in range(counts[f])]
        rng.shuffle(sequence)
        return sequence
    # Uneven chunks, sampled independently; no balanced rounds.
    chunks = []
    for f in order:
        remaining = counts[f]
        while remaining:
            chunk = int(rng.integers(1, remaining + 1))
            chunks.append([f] * chunk)
            remaining -= chunk
    rng.shuffle(chunks)
    return [f for chunk in chunks for f in chunk]


def _allocate_gaps(rng, slots, budget, maximum):
    gaps = [0] * slots
    if slots == 0 or budget <= 0:
        return gaps
    desired = int(rng.integers(0, min(budget, slots * maximum) + 1))
    for _ in range(desired):
        available = [i for i, value in enumerate(gaps) if value < maximum]
        gaps[int(rng.choice(available))] += 1
    return gaps


def generate_pair(rng, regime, parameters, spec):
    if regime not in ('source_train', 'target_test'):
        raise ValueError('unknown regime')
    f, ranks = spec['families'], spec['ranks']
    p = parameters
    palette = p['palette']
    # Repeated anchor identity survives optional recoloring of other families.
    anchor_cap = palette[1]
    bg_choices = [c for c in range(10) if c != anchor_cap]
    bg = int(rng.choice(bg_choices)) if p['mixed_background'] else palette[0]
    candidates = [c for c in palette[1:] + palette[:1] if c != bg and c != anchor_cap]
    if p['rename']:
        rng.shuffle(candidates)
    controls = ([anchor_cap] + candidates[:f - 1] if spec['anchor'] is not None
                else ([anchor_cap] + candidates)[:f])
    if spec['anchor'] is None and p['rename']:
        choices = [c for c in range(10) if c != bg]
        controls = [int(c) for c in rng.choice(choices, f, replace=False)]
    payload_palette = [c for c in range(10) if c != bg and c not in controls]
    rng.shuffle(payload_palette)
    sharing = 'common' if p['level'] == 0 else str(rng.choice(['common', 'family', 'bar']))
    classes = 1 if sharing == 'common' else int(rng.integers(1, min(f, len(payload_palette)) + 1))
    if sharing == 'bar':
        classes = int(rng.integers(2, min(4, len(payload_palette)) + 1))
    payload_palette = payload_palette[:classes]
    assignments = [i % classes for i in range(f)]
    rng.shuffle(assignments)

    minimum = [max(2, r) for r in ranks]
    if spec['anchor'] == 2:
        minimum[0] = max(3, minimum[0])
    if spec.get('avoid_spatial', False):
        minimum[0] = max(3, minimum[0])
    counts = [int(rng.integers(n, 8)) for n in minimum]
    while sum(counts) > 24:
        choices = [i for i in range(f) if counts[i] > minimum[i]]
        counts[int(rng.choice(choices))] -= 1
    key_order = [int(i) for i in rng.permutation(f)]
    key_gaps = [int(rng.integers(3)) for _ in range(f - 1)]
    if p['level'] == 0:
        key_gaps = [0] * (f - 1)
    key_columns, column = [], 0
    for i in range(f):
        key_columns.append(column)
        if i < f - 1:
            column += 1 + key_gaps[i]
    key_width = column + 1
    side = bool(rng.integers(2))
    if p['level'] == 0:
        side = True
    side_gap = int(rng.integers(1, 4))
    if key_width + side_gap + sum(counts) > 30:
        side = False
    separation = int(rng.integers(1, 3))
    length_limit = 28 if side else 30 - max(ranks) - separation
    max_count = max(counts)
    upper = int(rng.integers(max_count + 2, length_limit + 1))
    lengths = []
    for n in counts:
        style = int(rng.integers(3))
        if style == 0:
            start = int(rng.integers(3, min(10, upper - n + 1) + 1))
            values = list(range(start, start + n))
        elif style == 1:
            values = [int(v) for v in np.rint(np.linspace(3, upper, n))]
        else:
            values = [int(v) for v in rng.choice(np.arange(3, upper + 1), n, replace=False)]
        rng.shuffle(values)
        lengths.append(values)
    # A bounded density fallback leaves ample visible background even with
    # 24 bars and a single common payload. It does not resample a seed.
    if sum(map(sum, lengths)) + sum(ranks) > 430:
        lengths = [[3 + i for i in rng.permutation(n)] for n in counts]
    if spec.get('avoid_spatial', False):
        # This optional teaching repair rules out both spatial directions and
        # a fixed second-shortest choice, without changing the ordinal rule.
        longest = int(np.argmax(lengths[0]))
        middle = int(rng.integers(1, counts[0] - 1))
        lengths[0][longest], lengths[0][middle] = lengths[0][middle], lengths[0][longest]
    max_length = max(max(v) for v in lengths)
    sequence = _family_sequence(rng, counts, spec['mixing'])
    n_bars = len(sequence)
    reserved = key_width + side_gap if side else 0
    gaps = _allocate_gaps(rng, n_bars - 1, 30 - reserved - n_bars, 3)
    bar_columns, column = [], reserved
    for i in range(n_bars):
        bar_columns.append(column)
        if i < n_bars - 1:
            column += 1 + gaps[i]
    width = max(column + 1, key_width)
    width += int(rng.integers(0, min(3, 30 - width) + 1))
    key_x = 0 if side else int(rng.integers(0, width - key_width + 1))
    key_y = int(rng.integers(0, min(4, max_length - max(ranks) - 1) + 1)) if side else 0
    if p['level'] == 0:
        key_y = 0
    cursors = [0] * f
    records = []
    for x, family in zip(bar_columns, sequence):
        length = lengths[family][cursors[family]]
        cursors[family] += 1
        payload = int(rng.choice(payload_palette)) if sharing == 'bar' else payload_palette[assignments[family]]
        records.append({'family': family, 'column': x, 'length': length, 'payload': payload})
    baseline = max_length - 1
    # Keep a blank gap beneath each key run, including its immediate neighbors.
    # This prevents lateral contact with the key while long bars elsewhere may
    # still start above it. No full-height strip or global top band is reserved.
    length_at = {b['column']: b['length'] for b in records}
    for x, family in zip(key_columns, key_order):
        for neighboring_column in range(key_x + x - 1, key_x + x + 2):
            if neighboring_column in length_at:
                baseline = max(baseline, key_y + ranks[family] + separation
                               + length_at[neighboring_column] - 1)
    height = max(baseline + 1, key_y + max(ranks) + 1)
    _require(height <= 30, 'construction exceeded axis budget')
    grid = np.full((height, width), bg, dtype=np.int8)
    for x, family in zip(key_columns, key_order):
        grid[key_y:key_y + ranks[family], key_x + x] = controls[family]
    for b in records:
        x, cap = b['column'], controls[b['family']]
        top = baseline - b['length'] + 1
        grid[top, x] = grid[baseline, x] = cap
        grid[top + 1:baseline, x] = b['payload']
    # Translation and detached baselines are independent of the split.
    vertical = int(rng.integers(0, min(6, 30 - height) + 1))
    horizontal = int(rng.integers(0, min(6, 30 - width) + 1))
    top = int(rng.integers(vertical + 1))
    left = int(rng.integers(horizontal + 1))
    if p['level'] == 0:
        top = left = 0
    grid = np.pad(grid, ((top, vertical - top), (left, horizontal - left)), constant_values=bg)
    frequencies = np.bincount(grid.ravel().astype(int), minlength=10)
    while frequencies[bg] <= max(frequencies[c] for c in range(10) if c != bg):
        # At most 60 padding steps; <=430 occupied cells guarantees room.
        axis = 0 if grid.shape[0] < 30 and (grid.shape[0] <= grid.shape[1] or grid.shape[1] == 30) else 1
        padding = ((0, 1), (0, 0)) if axis == 0 else ((0, 0), (0, 1))
        grid = np.pad(grid, padding, constant_values=bg)
        frequencies = np.bincount(grid.ravel().astype(int), minlength=10)
    if bool(rng.integers(2)):
        grid = np.fliplr(grid)
    grid = np.ascontiguousarray(np.rot90(grid, spec['rotation']))
    # Ground truth comes from a fresh decode, not records or requested ranks.
    return {'input': grid, 'output': solve_input(grid)}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    parameters = sample_puzzle_parameters(rng)
    specs = _scene_specs(rng, parameters, train_count, test_count)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for spec in specs[split]:
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            pair = generate_pair(pair_rng, 'source_train' if split == 'train' else 'target_test', parameters, spec)
            pair = _unique_variant(pair, seen, parameters['palette'][1])
            seen.add(_grid_signature(pair['input']))
            puzzle[split].append(pair)
    if _training_shortcuts([_analyze_pair(pair) for pair in puzzle['train']]):
        index = next(i for i, spec in enumerate(specs['train']) if spec['anchor'] == 1)
        repair_spec = dict(specs['train'][index], avoid_spatial=True)
        seen.remove(_grid_signature(puzzle['train'][index]['input']))
        pair = generate_pair(rng, 'source_train', parameters, repair_spec)
        puzzle['train'][index] = _unique_variant(pair, seen, parameters['palette'][1])
    validate_generated_puzzle(puzzle)
    return puzzle


def _unique_variant(pair, seen, anchor):
    if _grid_signature(pair['input']) not in seen and not _is_source_copy(pair):
        return pair
    # Bounded fallback with 36 distinct rotations/color renamings for at most
    # 13 previous inputs. Preserve the repeated cap anchor and all geometry.
    other_colors = [c for c in range(10) if c != anchor]
    for rotation in range(4):
        for offset in range(9):
            mapping = np.arange(10, dtype=np.int8)
            for i, color in enumerate(other_colors):
                mapping[color] = other_colors[(i + offset) % 9]
            candidate = {k: np.ascontiguousarray(mapping[np.rot90(v, rotation)]) for k, v in pair.items()}
            if _grid_signature(candidate['input']) not in seen and not _is_source_copy(candidate):
                return candidate
    raise AssertionError('unique fallback exhausted')


def measure_pair(pair):
    """Review features reconstructed from returned pixels, not sampled settings."""
    a = _analyze_pair(pair)
    m = a['metadata']
    controls = m['controls']
    sequence = [controls.index(b['control']) for b in m['bars']]
    runs = [v for i, v in enumerate(sequence) if i == 0 or v != sequence[i - 1]]
    grouped = len(runs) == len(controls)
    split_families = sum(runs.count(i) > 1 for i in range(len(controls)))
    payloads = {b['payload'] for b in m['bars']}
    family_payloads = [{b['payload'] for b in m['bars_by_control'][c]} for c in controls]
    populations = [len(m['bars_by_control'][c]) for c in controls]
    ranks = [m['key_ranks'][c] for c in controls]
    lengths = [b['length'] for b in m['bars']]
    key_columns = [k['column'] for k in m['keys']]
    columns = [b['column'] for b in m['bars']]
    ordered_lengths = [sorted(b['length'] for b in m['bars_by_control'][c]) for c in controls]
    return {'shape': list(pair['input'].shape), 'canonical_shape': [m['height'], m['width']],
            'orientation': a['orientation'], 'background': m['background'],
            'controls': list(controls),
            'payloads_by_cap': {str(c): sorted(family_payloads[i]) for i, c in enumerate(controls)},
            'families': len(controls), 'ranks': ranks, 'populations': populations,
            'count_rank': [[n, r] for n, r in zip(populations, ranks)],
            'last_ranked': sum(n == r for n, r in zip(populations, ranks)),
            'non_extreme': sum(1 < r < n for n, r in zip(populations, ranks)),
            'payload_classes': len(payloads),
            'partial_sharing': len(controls) == 4 and len(payloads) in (2, 3),
            'within_family_payloads': any(len(s) > 1 for s in family_payloads),
            'shared_across_families': any(family_payloads[i] & family_payloads[j]
                                         for i in range(len(controls)) for j in range(i)),
            'sequence': sequence, 'grouped': grouped, 'split_families': split_families,
            'unequal_population': max(populations) - min(populations),
            'adjacent': any(b - a == 1 for a, b in zip(columns, columns[1:])),
            'gaps': [b - a - 1 for a, b in zip(columns, columns[1:])],
            'key_gaps': [b - a - 1 for a, b in zip(key_columns, key_columns[1:])],
            'key_translation': [m['keys'][0]['top'], min(key_columns)],
            'corner_key': m['keys'][0]['top'] == 0 and (min(key_columns) == 0 or max(key_columns) == m['width'] - 1),
            'key_over_rack': bool(set(key_columns) & set(columns)),
            'baseline_margin': m['height'] - 1 - m['baseline'],
            'min_bar_top': min(b['top'] for b in m['bars']), 'max_length': max(lengths),
            'length_gaps': [b - a for ls in ordered_lengths for a, b in zip(ls, ls[1:])],
            'layout': [[sequence[i], b['column'], b['top'], b['length']] for i, b in enumerate(m['bars'])]}


def _assert_diversity(puzzles):
    measured = [[(s, measure_pair(p)) for s in ('train', 'test') for p in puzzle[s]] for puzzle in puzzles]
    rows = [m for puzzle in measured for _, m in puzzle]
    _require(len({_puzzle_signature(p) for p in puzzles}) >= 40, 'cross-seed puzzle diversity')
    _require(len({len(p['train']) for p in puzzles}) >= 4, 'train count diversity')
    _require(len({len(p['test']) for p in puzzles}) >= 3, 'test count diversity')
    _require({1, 2, 3, 4, 5} <= {m['families'] for m in rows}, 'family count coverage')
    combinations = {tuple(v) for m in rows for v in m['count_rank']}
    _require({(n, 3) for n in range(3, 8)} <= combinations, 'rank three population coverage')
    _require({(2, 1), (2, 2), (6, 4)} <= combinations, 'small families and higher ordinals')
    _require(sum(m['last_ranked'] for m in rows) >= 20 and sum(m['non_extreme'] for m in rows) >= 50,
             'extreme and intermediate rank diversity')
    for field in ('partial_sharing', 'within_family_payloads', 'key_over_rack', 'adjacent'):
        _require(sum(bool(m[field]) for m in rows) >= 5, f'{field} coverage')
    _require({1, 2, 3, 4} <= {m['payload_classes'] for m in rows}, 'payload sharing spectrum')
    _require(any(m['unequal_population'] >= 4 for m in rows), 'uneven populations')
    _require(any(m['key_gaps'] and max(m['key_gaps']) >= 2 for m in rows), 'spaced key')
    _require(any(m['baseline_margin'] >= 2 for m in rows), 'detached baseline')
    _require(any(m['min_bar_top'] <= 1 for m in rows), 'early long bar')
    _require(any(m['max_length'] > 20 for m in rows), 'long bars')
    _require(any(10 in m['shape'] for m in rows) and any(max(m['shape']) == 30 for m in rows), 'size endpoints')
    _require(len({tuple(m['shape']) for m in rows}) >= 50, 'dimension diversity')
    _require(any(1 in m['length_gaps'] for m in rows) and any(max(m['length_gaps']) >= 8 for m in rows),
             'close and widely separated lengths')
    for split in ('train', 'test'):
        selected = [m for puzzle in measured for s, m in puzzle if s == split]
        _require(any(m['split_families'] >= 2 for m in selected) and any(m['grouped'] for m in selected),
                 'mixing must not be dictated by split')
        _require(len({m['families'] for m in selected}) >= 4, 'family variation by split')
        _require(any(m['families'] >= 2 and len(set(m['ranks'])) == 1 for m in selected), 'shared ordinals')
    for field in ('background', 'orientation', 'families', 'payload_classes'):
        varying = sum(len({m[field] for _, m in puzzle}) > 1 for puzzle in measured)
        _require(varying >= 5, f'within-puzzle {field} variation')
    _require(any(len({m['orientation'] for _, m in puzzle}) == 1 for puzzle in measured), 'common orientation')
    _require(sum(all(m['families'] <= 2 and m['payload_classes'] == 1 and m['corner_key']
                     for _, m in puzzle) and len({m['orientation'] for _, m in puzzle}) == 1
                 for puzzle in measured) >= 2, 'simple puzzles with compact corner keys')
    payload_changes = role_changes = test_family_variation = orientation_holdouts = 0
    contrast_positions = set()
    for puzzle in measured:
        train = [m for s, m in puzzle if s == 'train']
        test = [m for s, m in puzzle if s == 'test']
        test_family_variation += len({m['families'] for m in test}) > 1
        orientation_holdouts += bool({m['orientation'] % 2 for m in test}
                                     - {m['orientation'] % 2 for m in train})
        histories = {}
        cap_colors, payload_colors = set(), set()
        for _, m in puzzle:
            cap_colors.update(m['controls'])
            for cap, payloads in m['payloads_by_cap'].items():
                histories.setdefault(cap, set()).add(tuple(payloads))
                payload_colors.update(payloads)
        payload_changes += any(len(values) > 1 for values in histories.values())
        role_changes += bool(cap_colors & payload_colors)
        for i, first in enumerate(train):
            first_ranks = dict(zip(first['controls'], first['ranks']))
            for j, second in enumerate(train):
                if i >= j:
                    continue
                second_ranks = dict(zip(second['controls'], second['ranks']))
                if any({first_ranks[c], second_ranks[c]} == {1, 2}
                       for c in first_ranks.keys() & second_ranks.keys()):
                    contrast_positions.add((i, j))
    _require(payload_changes >= 20 and role_changes >= 5, 'pair-level payload and role changes')
    _require(test_family_variation >= 5, 'within-test family diversity')
    _require(orientation_holdouts >= 1, 'held-out orientations')
    _require(len(contrast_positions) >= 5, 'contrast positions must vary')
    _require(sum(any(m['partial_sharing'] and m['key_over_rack'] and m['split_families'] >= 1
                     for _, m in puzzle) for puzzle in measured) >= 2, 'combined variations')
    _require(any(min(m['canonical_shape'][1] for s, m in puzzle if s == 'test') <=
                 max(m['canonical_shape'][1] for s, m in puzzle if s == 'train') for puzzle in measured),
             'test width must not always exceed training')


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for pair in puzzle['test']:
            prediction = predict_from_training(
                [{k: v.copy() for k, v in p.items()} for p in puzzle['train']], pair['input'].copy())
            _require(np.array_equal(prediction, pair['output']), 'prediction check')
    _assert_diversity(puzzles)
    for seed, train, test in [(17, None, None), (1, 2, 1), (7100, 2, 1), (7101, 10, 1), (7102, 10, 4),
                               (7200, None, 1), (7201, None, 4), (7202, 2, None), (7203, 10, None)]:
        a, b = generate_puzzle(seed, train, test), generate_puzzle(seed, train, test)
        _require(_puzzle_signature(a) == _puzzle_signature(b), 'determinism')
        _require(train is None or len(a['train']) == train, 'explicit training count')
        _require(test is None or len(a['test']) == test, 'explicit test count')
        validate_generated_puzzle(a)
    for train, test in [(1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (2.5, 1)]:
        try:
            generate_puzzle(7300, train, test)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    for input_rows, output_rows in _SOURCE_FIXTURES:
        inp = np.array([[int(c) for c in row] for row in input_rows], dtype=np.int8)
        out = np.array([[int(c) for c in row] for row in output_rows], dtype=np.int8)
        _require(np.array_equal(solve_input(inp), out), 'original source rule')
        for rotation in range(4):
            for reflected in (False, True):
                a = np.rot90(np.fliplr(inp) if reflected else inp, rotation)
                b = np.rot90(np.fliplr(out) if reflected else out, rotation)
                for palette in (np.arange(10), np.array([8, 2, 7, 0, 5, 1, 9, 3, 6, 4])):
                    x = np.pad(palette[a], ((1, 2), (2, 1)), constant_values=int(palette[0]))
                    y = np.pad(palette[b], ((1, 2), (2, 1)), constant_values=int(palette[0]))
                    _require(np.array_equal(solve_input(x), y), 'visible geometry/color invariance')
    # A label mutation is rejected even when the surrounding puzzle is valid.
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    pair = bad['test'][0]
    row, col = np.argwhere(pair['input'] != pair['output'])[0]
    pair['output'][row, col] = pair['input'][row, col]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        return
    raise AssertionError('incorrect output accepted')


# Exact source fixtures: validation evidence only, never a sampling template.
_SOURCE_FIXTURES = (
    (
        (
            '200000000000000',
            '200020000000000',
            '000050000000000',
            '000050020000000',
            '000050050000000',
            '000050050000000',
            '002050050000000',
            '005050050000000',
            '005050050000000',
            '002020020000000',
        ),
        (
            '200000000000000',
            '200020000000000',
            '000050000000000',
            '000050020000000',
            '000050020000000',
            '000050020000000',
            '002050020000000',
            '005050020000000',
            '005050020000000',
            '002020020000000',
        ),
    ),
    (
        (
            '2000000000',
            '0000000000',
            '0000000000',
            '0000002000',
            '0000003000',
            '0000003000',
            '0020003000',
            '0030003000',
            '0030003000',
            '0020002000',
        ),
        (
            '2000000000',
            '0000000000',
            '0000000000',
            '0000002000',
            '0000002000',
            '0000002000',
            '0020002000',
            '0030002000',
            '0030002000',
            '0020002000',
        ),
    ),
    (
        (
            '103000000000000',
            '003000000000000',
            '000000000000000',
            '000000000000000',
            '001000000030000',
            '007000000050000',
            '007001000050030',
            '007007010050050',
            '007007070050050',
            '001001010030030',
        ),
        (
            '103000000000000',
            '003000000000000',
            '000000000000000',
            '000000000000000',
            '001000000030000',
            '001000000050000',
            '001001000050030',
            '001007010050030',
            '001007070050030',
            '001001010030030',
        ),
    ),
    (
        (
            '12340000000000000000',
            '10300000000000001000',
            '00300000000000005000',
            '00000000300000005000',
            '02000000500000005010',
            '05040000500000005050',
            '05050000503000005051',
            '05050030505002045055',
            '05050050505005055055',
            '02040030303002041011',
        ),
        (
            '12340000000000000000',
            '10300000000000001000',
            '00300000000000005000',
            '00000000300000005000',
            '02000000500000005010',
            '02040000500000005010',
            '02040000503000005011',
            '02040030505002045015',
            '02040030505005055015',
            '02040030303002041011',
        ),
    ),
)
