"""Solved ARC puzzles built from seed-generated diagonal traces."""

import numpy as np


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _rows(*rows):
    return np.array([[int(c) for c in row] for row in rows], dtype=np.int8)


_SOURCE_PAIRS = (
    (_rows('20', '00'), _rows('2200', '2200', '0010', '0001')),
    (_rows('00050', '05000', '00000', '05000', '00000'),
     _rows('1000005500', '0100005500', '0055000010', '0055000001',
           '1000100000', '0100010000', '0055001000', '0055000100',
           '0000100010', '0000010001')),
    (_rows('000003', '000000', '030000', '000000', '000000', '000000'),
     _rows('000000000033', '000000000033', '100000000000',
           '010000000000', '003300000000', '003300000000',
           '000010000000', '000001000000', '000000100000',
           '000000010000', '000000001000', '000000000100')),
    (_rows('040', '000', '400'),
     _rows('004400', '004400', '000010', '000001', '440000', '440000')),
)


def _key(grid):
    return (grid.shape, tuple(int(value) for value in grid.flat))


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output']))
                       for p in puzzle[split]) for split in ('train', 'test'))


def _source_copy(inp, out):
    return any(np.array_equal(inp, a) and np.array_equal(out, b)
               for a, b in _SOURCE_PAIRS)


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer or None')
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f'{name} must be between {low} and {high}')
    return value


def _counts(seed, num_train, num_test):
    tr = _count('num_train', num_train, 2, 10)
    te = _count('num_test', num_test, 1, 4)
    if tr is not None and te is not None:
        if tr <= te:
            raise ValueError('num_train must exceed num_test')
        return tr, te
    if tr is None and te is None:
        te = 1 + seed % 4
        tr = te + 1 + (seed // 4) % (10 - te)
    elif tr is None:
        tr = te + 1 + seed % (10 - te)
    else:
        te = 1 + seed % min(4, tr - 1)
    return tr, te


def _trace_cells(inp, bg):
    offsets = {int(r) - int(c) for r, c in np.argwhere(inp != bg)}
    return [(r, c) for r in range(inp.shape[0]) for c in range(inp.shape[1])
            if inp[r, c] == bg and r - c in offsets]


def _render(inp, bg, trace):
    n = inp.shape[0]
    out = np.full((2 * n, 2 * n), bg, dtype=np.int8)
    for r, c in _trace_cells(inp, bg):
        out[2 * r, 2 * c] = trace
        out[2 * r + 1, 2 * c + 1] = trace
    for r, c in np.argwhere(inp != bg):
        out[2 * r:2 * r + 2, 2 * c:2 * c + 2] = inp[r, c]
    return out


def _diagonal(n, offset):
    return [(r, r - offset) for r in range(n) if 0 <= r - offset < n]


def _positions(rng, n, mode):
    if mode == 'anchor':
        r = int(rng.integers(1, n - 1))
        c = int(rng.integers(1, n - 1))
        return [(r, c)]
    if mode == 'isolated':
        return [((0, n - 1), (n - 1, 0))[int(rng.integers(2))]]
    if mode == 'saturated':
        return [(0, 0), (1, 1)]
    if mode == 'one_trace':
        corner = ((0, 2), (2, 0))[int(rng.integers(2))]
        short = ((0, 1), (1, 2), (1, 0), (2, 1))[int(rng.integers(4))]
        return [corner, short]
    limit = min(5, (n * n - 1) // 2)
    count = int(rng.integers(2, limit + 1)) if limit >= 2 and mode != 'free_single' else 1
    if mode == 'same':
        count = min(count, (n - 1) // 2)
        offsets = [d for d in range(1 - n, n) if len(_diagonal(n, d)) >= 2 * count + 1]
        if offsets:
            diagonal = _diagonal(n, int(rng.choice(offsets)))
            if count >= 2:
                # Each gap, including both end gaps, contains a background cell.
                extra = len(diagonal) - (2 * count + 1)
                gaps = 1 + rng.multinomial(extra, np.full(count + 1, 1 / (count + 1)))
                index = int(gaps[0])
                chosen = []
                for i in range(count):
                    chosen.append(diagonal[index])
                    index += 1 + int(gaps[i + 1])
                return chosen
    if mode == 'different':
        offsets = rng.choice(np.arange(1 - n, n), size=min(count, 2 * n - 1), replace=False)
        return [_diagonal(n, int(d))[int(rng.integers(len(_diagonal(n, int(d)))))]
                for d in offsets]
    indices = rng.choice(n * n, size=count, replace=False)
    return [(int(i) // n, int(i) % n) for i in indices]


def _make_pair(rng, n, mode, bg, trace, colors, multicolor=False):
    positions = _positions(rng, n, mode)
    inp = np.full((n, n), bg, dtype=np.int8)
    if multicolor and len(positions) < 2:
        positions = _positions(rng, n, 'free')
    chosen = rng.choice(colors, size=len(positions), replace=True)
    if multicolor:
        chosen[0], chosen[1] = colors[0], colors[1]
    for (r, c), color in zip(positions, chosen):
        inp[r, c] = int(color)
    return {'input': inp, 'output': _render(inp, bg, trace)}


def _background(inp, learned_backgrounds=()):
    colors, counts = np.unique(inp, return_counts=True)
    biggest = counts.max()
    leaders = [int(c) for c, count in zip(colors, counts) if count == biggest]
    if len(leaders) == 1:
        return leaders[0]
    choices = set(leaders) & set(learned_backgrounds)
    _check(len(choices) == 1, 'tied background is not learned from other training pairs')
    return next(iter(choices))


def _pair_features(pair, bg):
    inp = pair['input']
    positions = [(int(r), int(c)) for r, c in np.argwhere(inp != bg)]
    offsets = {r - c for r, c in positions}
    colors = {int(inp[r, c]) for r, c in positions}
    return {'size': inp.shape[0], 'seeds': len(positions), 'offsets': len(offsets),
            'trace': len(_trace_cells(inp, bg)), 'seed_colors': colors,
            'same_diagonal': len(positions) > 1 and len(offsets) == 1,
            'background': bg}


def _same_diagonal_gaps(inp, bg):
    positions = [(int(r), int(c)) for r, c in np.argwhere(inp != bg)]
    offsets = {r - c for r, c in positions}
    if len(positions) < 2 or len(offsets) != 1:
        return False
    diagonal = _diagonal(inp.shape[0], next(iter(offsets)))
    indices = sorted(diagonal.index(position) for position in positions)
    return (indices[0] > 0 and indices[-1] < len(diagonal) - 1
            and any(b - a > 1 for a, b in zip(indices, indices[1:])))


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = [int(c) for c in rng.permutation(10)]
    backgrounds, trace, seed_colors = palette[:3], palette[3], palette[4:]
    training_colors = seed_colors[:3]
    novel_colors = seed_colors[3:]

    # Required demonstrations are shuffled, so their structural role is not tied to order.
    anchor_size = int(rng.integers(4, 13))
    multi_size = int(rng.integers(4, 13))
    if multi_size == anchor_size:
        multi_size = 4 + (multi_size - 3) % 9
    plans = [('anchor', anchor_size, False),
             ('different' if rng.integers(2) else 'free', multi_size, True)]
    size_choices = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    modes = ('free_single', 'free', 'same', 'different', 'isolated')
    for _ in range(train_count - 2):
        n = int(rng.choice(size_choices))
        mode = modes[int(rng.integers(len(modes)))]
        if n == 2 and rng.integers(4) == 0:
            mode = 'saturated'
        elif n == 2 and mode == 'same':
            mode = 'free_single'
        plans.append((mode, n, False))
    rng.shuffle(plans)

    puzzle = {'train': [], 'test': []}
    used = set()
    # The two required demonstrations use different backgrounds in every puzzle.
    if rng.integers(2):
        anchor_background, multi_background = backgrounds[:2]
    else:
        multi_background, anchor_background = backgrounds[:2]
    train_backgrounds = [int(rng.choice(backgrounds)) for _ in plans]
    for i, (mode, _, multi) in enumerate(plans):
        if mode == 'anchor':
            train_backgrounds[i] = anchor_background
        elif multi:
            train_backgrounds[i] = multi_background
    for index, (mode, n, multi) in enumerate(plans):
        bg = train_backgrounds[index]
        for _ in range(128):
            # A saturated 2x2 tie is only allowed with a background visible in another train pair.
            if mode == 'saturated':
                bg = anchor_background
            pair = _make_pair(rng, n, mode, bg, trace, training_colors, multi)
            key = _key(pair['input'])
            if key not in used and not _source_copy(pair['input'], pair['output']):
                break
        else:
            raise RuntimeError('could not make a unique training pair')
        puzzle['train'].append(pair)
        used.add(key)

    # Place a trace-bearing demonstration before any legitimate no-trace example.
    puzzle['train'].sort(key=lambda pair: not bool(np.any(pair['output'] == trace)))
    demonstrated_colors = tuple(color for color in training_colors
                                if any(color in p['input'] for p in puzzle['train']))
    train_sizes = [p['input'].shape[0] for p in puzzle['train']]
    for _ in range(test_count):
        for _ in range(128):
            relationship = int(rng.integers(4))
            if relationship == 0:
                n = int(rng.choice(train_sizes))
            elif relationship == 1:
                n = int(rng.integers(min(train_sizes), max(train_sizes) + 1))
            elif relationship == 2 and max(train_sizes) < 15:
                n = int(rng.integers(max(train_sizes) + 1, 16))
            else:
                n = int(rng.integers(2, 16))
            mode = ('free_single', 'free', 'same', 'different', 'isolated', 'one_trace')[
                int(rng.integers(6))]
            if mode == 'one_trace':
                n = 3
            if n < 4 and mode == 'same':
                mode = 'free'
            bg = int(rng.choice(backgrounds))
            reuse = bool(rng.integers(2))
            colors = demonstrated_colors if reuse else novel_colors
            multi = n >= 4 and bool(rng.integers(4) == 0)
            pair = _make_pair(rng, n, mode, bg, trace, colors, multi)
            key = _key(pair['input'])
            if key not in used and not _source_copy(pair['input'], pair['output']):
                break
        else:
            raise RuntimeError('could not make a unique test pair')
        puzzle['test'].append(pair)
        used.add(key)
    validate_generated_puzzle(puzzle)
    return puzzle


def _assert_pair_rule(inp, out, bg, trace):
    n = inp.shape[0]
    _check(inp.shape == (n, n) and 2 <= n <= 15, 'inputs must be square, size 2..15')
    _check(out.shape == (2 * n, 2 * n), 'output must double input dimensions')
    positions = np.argwhere(inp != bg)
    _check(len(positions) >= 1 and len(positions) < n * n, 'sparse seeds required')
    offsets = {int(r) - int(c) for r, c in positions}
    for r in range(n):
        for c in range(n):
            tile = out[2*r:2*r+2, 2*c:2*c+2]
            value = int(inp[r, c])
            if value != bg:
                expected = np.full((2, 2), value, dtype=np.int8)
            elif r - c in offsets:
                expected = np.array(((trace, bg), (bg, trace)), dtype=np.int8)
            else:
                expected = np.full((2, 2), bg, dtype=np.int8)
            _check(np.array_equal(tile, expected), 'incorrect diagonal tile or seed priority')


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle schema')
    _check(isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list), 'splits must be lists')
    tr, te = len(puzzle['train']), len(puzzle['test'])
    _check(2 <= tr <= 10 and 1 <= te <= 4 and tr > te, 'invalid pair counts')
    seen = set()
    pairs = puzzle['train'] + puzzle['test']
    for pair in pairs:
        _check(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair schema')
        for grid in pair.values():
            _check(isinstance(grid, np.ndarray) and grid.ndim == 2, 'grid must be a 2D ndarray')
            _check(np.issubdtype(grid.dtype, np.integer), 'grid must use integer dtype')
            _check(all(1 <= d <= 30 for d in grid.shape), 'grid dimensions outside ARC bounds')
            _check(0 <= int(grid.min()) <= int(grid.max()) <= 9, 'invalid ARC color')
        key = _key(pair['input'])
        _check(key not in seen, 'duplicate input')
        seen.add(key)
        _check(not _source_copy(pair['input'], pair['output']), 'exact source pair copied')
        _check(not np.array_equal(pair['input'], pair['output']), 'input equals output')

    # Every training pair except an explicit 2x2 saturation has an identifiable majority background.
    known_bgs = []
    for pair in puzzle['train']:
        inp = pair['input']
        colors, counts = np.unique(inp, return_counts=True)
        if int(np.sum(counts == counts.max())) == 1:
            known_bgs.append(int(colors[int(np.argmax(counts))]))
    _check(len(set(known_bgs)) >= 2, 'training must demonstrate local backgrounds')
    roles = [_background(p['input'], known_bgs) for p in pairs]
    introduced = set()
    for pair, bg in zip(puzzle['train'], roles[:tr]):
        new = set(map(int, np.unique(pair['output']))) - set(map(int, np.unique(pair['input'])))
        _check(len(new) <= 1, 'training introduces multiple trace colors')
        introduced |= new
    _check(len(introduced) == 1, 'training must reveal exactly one trace color')
    trace = next(iter(introduced))
    _check(trace not in roles, 'trace conflicts with background')
    _check(trace in puzzle['train'][0]['output'],
           'a visible trace must precede any no-trace training example')
    train_seed_colors = set()
    visible_trace = False
    bidirectional = False
    multicolor = False
    train_sizes = set()
    for i, (pair, bg) in enumerate(zip(pairs, roles)):
        inp, out = pair['input'], pair['output']
        _check(trace not in inp, 'trace must not appear in input')
        _assert_pair_rule(inp, out, bg, trace)
        features = _pair_features(pair, bg)
        if i < tr:
            train_sizes.add(features['size'])
            train_seed_colors |= features['seed_colors']
            visible_trace |= features['trace'] > 0
            multicolor |= len(features['seed_colors']) >= 2
            for r, c in np.argwhere(inp != bg):
                r, c = int(r), int(c)
                before = any(inp[r-k, c-k] == bg for k in range(1, min(r, c)+1))
                after = any(inp[r+k, c+k] == bg for k in range(1, min(inp.shape[0]-1-r, inp.shape[1]-1-c)+1))
                bidirectional |= before and after
        else:
            _check(features['size'] >= 2, 'invalid target size')
            if len(features['seed_colors']) >= 2:
                _check(multicolor, 'multicolor test lacks a training demonstration')
    _check(visible_trace and bidirectional and multicolor,
           'training must show trace color, both directions, and multiple seed colors')
    _check(all(2 <= n <= 15 for n in train_sizes), 'invalid source_train size')


def _expect_value_error(**kwargs):
    try:
        generate_puzzle(991, **kwargs)
    except ValueError:
        return
    raise AssertionError(f'expected ValueError: {kwargs}')


def self_test() -> None:
    a = generate_puzzle(314159)
    _check(_puzzle_key(a) == _puzzle_key(generate_puzzle(314159)), 'nondeterministic puzzle')
    b = generate_puzzle(271828, 7, 3)
    _check(_puzzle_key(b) == _puzzle_key(generate_puzzle(271828, 7, 3)), 'nondeterministic explicit counts')
    _check((len(b['train']), len(b['test'])) == (7, 3), 'explicit counts ignored')
    train_counts, test_counts, puzzles, roles, layouts, sizes = set(), set(), set(), set(), set(), set()
    train_sizes, test_sizes, test_traces, test_seeds = set(), set(), set(), set()
    one_trace = no_trace = many_trace = same = multi_offsets = multicolor = 0
    reused = novel = local_bgs = large_test = familiar_test = 0
    intermediate_test = same_gap = train_saturated = train_small_single = 0
    train_no_trace = new_background = combined_hard = 0
    train_locations, test_locations = set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        puzzles.add(_puzzle_key(puzzle))
        learned = [_background(p['input']) for p in puzzle['train'] if p['input'].shape != (2, 2)]
        bgs = [_background(p['input'], learned) for p in puzzle['train'] + puzzle['test']]
        new = set()
        for p in puzzle['train']:
            new |= set(map(int, np.unique(p['output']))) - set(map(int, np.unique(p['input'])))
        trace = next(iter(new))
        roles.add((tuple(sorted(set(bgs))), trace))
        local_bgs += len(set(bgs)) >= 2
        train_palette = set()
        training_sizes = {p['input'].shape[0] for p in puzzle['train']}
        for p, bg in zip(puzzle['train'], bgs):
            f = _pair_features(p, bg)
            train_sizes.add(f['size'])
            train_palette |= f['seed_colors']
            train_saturated += f['size'] == 2 and f['seeds'] == 2 and f['trace'] == 0
            train_small_single += f['size'] == 2 and f['seeds'] == 1
            train_no_trace += f['trace'] == 0
            for r, c in np.argwhere(p['input'] != bg):
                train_locations.add('corner' if r in (0, f['size']-1) and c in (0, f['size']-1)
                                    else 'edge' if r in (0, f['size']-1) or c in (0, f['size']-1)
                                    else 'interior')
        for p, bg in zip(puzzle['test'], bgs[len(puzzle['train']):]):
            f = _pair_features(p, bg)
            test_sizes.add(f['size'])
            test_traces.add(f['trace'])
            test_seeds.add(f['seeds'])
            one_trace += f['trace'] == 1
            no_trace += f['trace'] == 0
            many_trace += f['trace'] >= 5
            same += f['same_diagonal']
            multi_offsets += f['offsets'] >= 2
            multicolor += len(f['seed_colors']) >= 2
            reused += bool(f['seed_colors'] & train_palette)
            novel += bool(f['seed_colors'] - train_palette)
            large_test += f['size'] > max(training_sizes)
            familiar_test += f['size'] in training_sizes
            intermediate_test += min(training_sizes) < f['size'] < max(training_sizes) and f['size'] not in training_sizes
            new_background += bg not in bgs[:len(puzzle['train'])]
            same_gap += _same_diagonal_gaps(p['input'], bg)
            combined_hard += (f['size'] > max(training_sizes) and f['seeds'] >= 2
                              and f['trace'] >= 5)
            for r, c in np.argwhere(p['input'] != bg):
                test_locations.add('corner' if r in (0, f['size']-1) and c in (0, f['size']-1)
                                   else 'edge' if r in (0, f['size']-1) or c in (0, f['size']-1)
                                   else 'interior')
        for p, bg in zip(puzzle['train'] + puzzle['test'], bgs):
            f = _pair_features(p, bg)
            sizes.add(f['size'])
            layouts.add((f['size'], tuple(int(v != bg) for v in p['input'].flat)))
    _check(len(train_counts) >= 4 and len(test_counts) >= 3, 'automatic counts lack diversity')
    _check(len(puzzles) >= 40 and len(roles) >= 15 and len(layouts) >= 100, 'puzzle diversity too narrow')
    _check({2, 3, 4, 15} <= sizes and {2, 3, 4} <= train_sizes, 'size range not covered')
    _check(len(test_sizes) >= 10 and {1, 2, 3} <= test_seeds, 'test size or seed-count range too narrow')
    _check(one_trace and no_trace and many_trace and same and multi_offsets and multicolor,
           'missing a reviewed test geometry')
    _check(reused and novel and local_bgs and large_test and familiar_test,
           'missing color/background/size regimes')
    _check(intermediate_test and same_gap and train_saturated and train_small_single
           and train_no_trace and new_background and combined_hard,
           'missing short, saturated, same-diagonal, or combined regime')
    _check(train_locations == {'corner', 'edge', 'interior'} and
           test_locations == {'corner', 'edge', 'interior'},
           'seed locations lack boundary/interior variation')
    for seed, tr, te in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4)):
        p = generate_puzzle(seed, tr, te)
        _check((len(p['train']), len(p['test'])) == (tr, te), 'boundary counts ignored')
    for seed, tr, te in ((1101, None, 1), (1102, None, 4),
                         (1103, 2, None), (1104, 10, None)):
        p = generate_puzzle(seed, tr, te)
        _check((tr is None or len(p['train']) == tr) and
               (te is None or len(p['test']) == te), 'partial count ignored')
    for kwargs in ({'num_train': 1}, {'num_train': 11}, {'num_test': 0},
                   {'num_test': 5}, {'num_train': 2, 'num_test': 2},
                   {'num_train': 3, 'num_test': 4}, {'num_train': True},
                   {'num_test': 1.5}):
        _expect_value_error(**kwargs)
