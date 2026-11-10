"""Solved generators for the nested, self-describing ARC block puzzle."""

import numpy as np


_SIDE_TO_ORDER = {n * n + n - 1: n for n in range(2, 6)}
_SOURCE_INPUT_TEXTS = frozenset({
    '32022/33022/00000/32032/33033',
    '11011/18018/00050/51585/55088',
    '33403340444/33403340444/43304330444/00000000000/33403340444/33403345444/43305330444/00005500000/44455540334/45553340334/44504330433',
    '44204420222/24202420222/44404445225/00000000005/22204425222/22202525522/22504550252/00500000000/45554425442/55502520242/45404440444',
    '1112011120111202522/1221012210122105522/2111021110211105222/1211012110121105222/0000000000000555000/1112022220222555112/1221555220225501221/2115552220255202551/1255022225555555255/0050000055050000055/2255011555151201152/5555012215555101551/2222021115515505511/2222012115155555211/0000000005550550000/1112022225111555112/1221022250525555221/2111022520551152111/1211022220155551211',
})


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _text(grid):
    return '/'.join(''.join(str(int(x)) for x in row) for row in grid)


def _signature(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _counts(seed, rng, num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10),
                                    ('num_test', num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= value <= high):
            raise ValueError(f'{name} is outside its legal range')
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _order(grid):
    _require(grid.shape[0] == grid.shape[1]
             and grid.shape[0] in _SIDE_TO_ORDER, 'invalid hierarchical size')
    return _SIDE_TO_ORDER[grid.shape[0]]


def _divider_mask(n):
    side = n * n + n - 1
    mask = np.zeros((side, side), dtype=bool)
    for k in range(1, n):
        d = k * (n + 1) - 1
        mask[d, :] = True
        mask[:, d] = True
    return mask


def _block_slice(n, i, j):
    r, c = i * (n + 1), j * (n + 1)
    return np.s_[r:r+n, c:c+n]


def _build(template, divider, carrier, accent):
    n = template.shape[0]
    side = n * n + n - 1
    inp = np.full((side, side), divider, dtype=np.int8)
    out = inp.copy()
    for i in range(n):
        for j in range(n):
            sl = _block_slice(n, i, j)
            if template[i, j] == carrier:
                inp[sl] = template
                out[sl] = carrier
            else:
                inp[sl] = accent
                out[sl] = template
    return inp, out


def _template(rng, n, carrier, accent, used):
    cells = n * n
    for _ in range(120):
        style = int(rng.integers(0, 6))
        if style in (0, 1, 5):
            if style == 0:
                count = int(rng.integers(1, min(3, cells - 1) + 1))
            elif style == 1:
                count = cells - int(rng.integers(1, min(3, cells - 1) + 1))
            else:
                count = int(rng.integers(1, cells))
            bits = np.zeros(cells, dtype=bool)
            bits[rng.choice(cells, size=count, replace=False)] = True
            bits = bits.reshape(n, n)
        elif style == 2:
            bits = np.zeros((n, n), dtype=bool)
            lines = rng.choice(n, size=int(rng.integers(1, n)), replace=False)
            if rng.integers(2):
                bits[lines, :] = True
            else:
                bits[:, lines] = True
            if n >= 3 and rng.integers(3) == 0:
                bits[int(rng.integers(n)), int(rng.integers(n))] ^= True
        elif style == 3:
            bits = rng.integers(0, 2, size=(n, (n + 1) // 2)).astype(bool)
            bits = np.concatenate((bits, np.fliplr(bits[:, :n // 2])), axis=1)
            if rng.integers(2):
                bits = bits.T
        else:
            bits = np.zeros((n, n), dtype=bool)
            r, c = int(rng.integers(n)), int(rng.integers(n))
            goal = int(rng.integers(2, cells))
            for _ in range(goal * 5):
                bits[r, c] = True
                r = max(0, min(n - 1, r + int(rng.integers(-1, 2))))
                c = max(0, min(n - 1, c + int(rng.integers(-1, 2))))
                if np.count_nonzero(bits) >= goal:
                    break
        if not 0 < np.count_nonzero(bits) < cells:
            continue
        key = (n, bits.tobytes())
        if key not in used:
            used.add(key)
            return np.where(bits, carrier, accent).astype(np.int8)
    # Exhaust only the small vocabulary; larger orders have ample random space.
    if n > 3:
        raise ValueError('no unused template after bounded sampling')
    for mask in rng.permutation(np.arange(1, (1 << cells) - 1)):
        bits = np.array([(int(mask) >> k) & 1 for k in range(cells)], dtype=bool).reshape(n, n)
        key = (n, bits.tobytes())
        if key not in used:
            used.add(key)
            return np.where(bits, carrier, accent).astype(np.int8)
    raise ValueError('no unused template')


def _colors(rng, divider, corruption, previous):
    available = [c for c in range(10) if c not in (divider, corruption)]
    if previous and rng.random() < 0.48:
        a, b = previous[int(rng.integers(len(previous)))]
        if rng.random() < 0.5:
            a, b = b, a
    else:
        a, b = (int(x) for x in rng.choice(available, size=2, replace=False))
    previous.append((a, b))
    return a, b


def _patch(rng, n, desired):
    side = n * n + n - 1
    divider = int(rng.integers(1, n)) * (n + 1) - 1
    start = (max(0, min(side - 1, divider + int(rng.integers(-2, 3)))),
             max(0, min(side - 1, divider + int(rng.integers(-2, 3)))))
    chosen = {start}
    frontier = {start}
    while len(chosen) < desired and frontier:
        options = set()
        for r, c in frontier:
            for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= rr < side and 0 <= cc < side and (rr, cc) not in chosen:
                    options.add((rr, cc))
        if not options:
            break
        point = sorted(options)[int(rng.integers(len(options)))]
        chosen.add(point)
        if len(frontier) > 3 and rng.random() < 0.35:
            frontier = {point}
        else:
            frontier.add(point)
    return chosen


def _mask(rng, clean_input, template, carrier, mode):
    n = template.shape[0]
    side = clean_input.shape[0]
    mask = np.zeros_like(clean_input, dtype=bool)
    dividers = _divider_mask(n)
    if mode == 'clean':
        return mask
    if mode in ('patch', 'hybrid'):
        fraction = rng.uniform(0.10, 0.32 if n >= 4 else 0.23)
        for r, c in _patch(rng, n, max(3, int(side * side * fraction))):
            mask[r, c] = True
    if mode == 'scatter_all':
        for i in range(n):
            for j in range(n):
                sl = _block_slice(n, i, j)
                count = int(rng.integers(1, max(2, n * n // 2 + 1)))
                idx = rng.choice(n * n, size=count, replace=False)
                block = mask[sl]
                block.flat[idx] = True
    if mode in ('light', 'hybrid', 'occlude_carriers', 'whole_block'):
        fraction = rng.uniform(0.015, 0.07 if mode == 'light' else 0.12)
        idx = rng.choice(side * side, size=max(1, int(side * side * fraction)), replace=False)
        mask.flat[idx] = True
    if mode in ('occlude_carriers', 'whole_block'):
        positions = np.argwhere(template == carrier)
        i, j = (int(x) for x in positions[int(rng.integers(len(positions)))])
        sl = _block_slice(n, i, j)
        if mode == 'whole_block':
            mask[sl] = True
        else:
            block = mask[sl]
            block[template == carrier] = True
    if mode == 'scatter_all':
        count = int(rng.integers(1, max(2, dividers.sum() // 3)))
        choices = np.flatnonzero(dividers)
        mask.flat[rng.choice(choices, size=count, replace=False)] = True
    elif mode not in ('patch', 'hybrid'):
        # A corrupted training pair always reveals divider repair as well.
        choices = np.flatnonzero(dividers)
        mask.flat[int(rng.choice(choices))] = True
    if mode in ('patch', 'hybrid') and not np.any(mask & dividers):
        choices = np.flatnonzero(dividers)
        mask.flat[int(rng.choice(choices))] = True
    if not np.any(mask & ~dividers):
        choices = np.flatnonzero(~dividers)
        mask.flat[int(rng.choice(choices))] = True
    return mask


def _solutions(input_grid, divider, corruption, limit=2):
    """Enumerate globally consistent role/template reconstructions, stopping at limit."""
    n = _order(input_grid)
    dividers = _divider_mask(n)
    if not np.all((input_grid[dividers] == divider) | (input_grid[dividers] == corruption)):
        return []
    colors = sorted(int(x) for x in np.unique(input_grid[~dividers])
                    if x not in (divider, corruption))
    if len(colors) != 2 or not np.all(np.isin(input_grid[~dividers],
                                               colors + [corruption])):
        return []
    blocks = []
    for i in range(n):
        for j in range(n):
            block = input_grid[_block_slice(n, i, j)]
            blocks.append([(k, int(v)) for k, v in enumerate(block.flat) if v != corruption])
    found = []
    for carrier, accent in ((colors[0], colors[1]), (colors[1], colors[0])):
        observations = [[(k, int(v == carrier)) for k, v in block]
                        for block in blocks]
        def propagate(state):
            state = state.copy()
            changed = True
            while changed:
                changed = False
                for i, obs in enumerate(observations):
                    if any(v == 1 for _, v in obs):
                        if state[i] == 0:
                            return None
                        if state[i] == -1:
                            state[i] = 1
                            changed = True
                    if state[i] == 1:
                        for j, v in obs:
                            if state[j] != -1 and state[j] != v:
                                return None
                            if state[j] == -1:
                                state[j] = v
                                changed = True
            return state
        def visit(state):
            if len(found) >= limit:
                return
            state = propagate(state)
            if state is None or np.all(state == 0) or np.all(state == 1):
                return
            unknown = np.flatnonzero(state == -1)
            if len(unknown):
                # Highly observed variables propagate the most information.
                index = max(unknown, key=lambda x: sum(k == x for obs in observations for k, _ in obs))
                for value in (0, 1):
                    child = state.copy()
                    child[index] = value
                    visit(child)
                return
            template = np.where(state.reshape(n, n) == 1, carrier, accent).astype(np.int8)
            expected, output = _build(template, divider, carrier, accent)
            if np.all((input_grid == expected) | (input_grid == corruption)):
                found.append((carrier, accent, template, output))
        visit(np.full(n * n, -1, dtype=np.int8))
        if len(found) >= limit:
            break
    return found


def _make_pair(rng, n, divider, carrier, accent, corruption, mode, used_templates, used_inputs):
    for _ in range(100):
        template = _template(rng, n, carrier, accent, used_templates)
        clean_input, output = _build(template, divider, carrier, accent)
        modes = [mode] if mode == 'clean' else [mode, mode, mode, 'light', 'light']
        for trial_mode in modes:
            for _ in range(5 if trial_mode != 'clean' else 1):
                mask = _mask(rng, clean_input, template, carrier, trial_mode)
                inp = clean_input.copy()
                inp[mask] = corruption
                signature = _signature(inp)
                if (signature in used_inputs or _text(inp) in _SOURCE_INPUT_TEXTS
                        or np.array_equal(inp, output)):
                    continue
                if len(_solutions(inp, divider, corruption)) != 1:
                    continue
                used_inputs.add(signature)
                return {'input': inp, 'output': output}
    raise ValueError('could not build an unambiguous pair')


def _convention(train):
    dividers = set()
    input_colors, output_colors = set(), set()
    for pair in train:
        out = pair['output']
        n = _order(out)
        mask = _divider_mask(n)
        values = np.unique(out[mask])
        _require(len(values) == 1, 'output dividers are inconsistent')
        d = int(values[0])
        _require(not np.any(out[~mask] == d), 'divider appears in payload')
        dividers.add(d)
        input_colors.update(int(x) for x in np.unique(pair['input']))
        output_colors.update(int(x) for x in np.unique(out))
    _require(len(dividers) == 1, 'divider must be fixed within the puzzle')
    corruption = input_colors - output_colors
    _require(len(corruption) == 1, 'training must expose one corruption color')
    d, c = next(iter(dividers)), next(iter(corruption))
    _require(d != c, 'divider and corruption must differ')
    return d, c


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    divider, corruption = (int(x) for x in rng.choice(10, size=2, replace=False))
    # Two orders and one damaged example establish both scales and the repair rule.
    basic = [(2, 'clean'), (3, 'light')]
    if rng.integers(2):
        basic = [(2, 'light'), (3, 'clean')]
    rng.shuffle(basic)
    train_specs = basic[:]
    for _ in range(train_count - 2):
        n = int(rng.choice([2, 3, 4], p=[0.31, 0.45, 0.24]))
        mode = str(rng.choice(['clean', 'light', 'patch', 'scatter_all', 'hybrid'],
                              p=[0.25, 0.29, 0.20, 0.11, 0.15]))
        train_specs.append((n, mode))
    test_specs = []
    for index in range(test_count):
        choices = [2, 3, 4] if train_count < 4 else [2, 3, 4, 5]
        weights = [0.12, 0.19, 0.69] if train_count < 4 else [0.09, 0.15, 0.53, 0.23]
        n = int(rng.choice(choices, p=weights))
        mode = str(rng.choice(['light', 'scatter_all', 'patch', 'hybrid',
                               'occlude_carriers', 'whole_block'],
                              p=[0.11, 0.22, 0.22, 0.20, 0.15, 0.10]))
        test_specs.append((n, mode))
    puzzle = {'train': [], 'test': []}
    previous, used_templates, used_inputs = [], set(), set()
    for section, specs in (('train', train_specs), ('test', test_specs)):
        for n, mode in specs:
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            carrier, accent = _colors(pair_rng, divider, corruption, previous)
            pair = _make_pair(pair_rng, n, divider, carrier, accent, corruption,
                              mode, used_templates, used_inputs)
            puzzle[section].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'},
             'puzzle must have exactly train and test')
    train, test = puzzle['train'], puzzle['test']
    _require(isinstance(train, list) and isinstance(test, list), 'sections must be lists')
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             'invalid pair counts')
    all_pairs = train + test
    seen = set()
    for pair in all_pairs:
        _require(isinstance(pair, dict) and set(pair) == {'input', 'output'},
                 'pair must contain exactly input and output')
        for grid in pair.values():
            _require(isinstance(grid, np.ndarray) and grid.ndim == 2
                     and np.issubdtype(grid.dtype, np.integer), 'grid must be a 2D integer array')
            _require(all(1 <= x <= 30 for x in grid.shape)
                     and np.all((0 <= grid) & (grid <= 9)), 'invalid grid size or colors')
        inp, out = pair['input'], pair['output']
        _require(inp.shape == out.shape and not np.array_equal(inp, out),
                 'shape changed or pair unchanged')
        _order(inp)
        sig = _signature(inp)
        _require(sig not in seen, 'duplicate input or pair')
        seen.add(sig)
        _require(_text(inp) not in _SOURCE_INPUT_TEXTS, 'source input copied')
    divider, corruption = _convention(train)
    _require(_convention(all_pairs) == (divider, corruption),
             'test changes puzzle conventions')
    train_orders = set()
    train_corruption = False
    train_divider_damage = False
    train_payload_damage = False
    for section, pairs in (('train', train), ('test', test)):
        for pair in pairs:
            inp, out = pair['input'], pair['output']
            n = _order(inp)
            if section == 'train':
                train_orders.add(n)
                train_corruption |= bool(np.any(inp == corruption))
                divider_cells = _divider_mask(n)
                train_divider_damage |= bool(np.any(inp[divider_cells] == corruption))
                train_payload_damage |= bool(np.any(inp[~divider_cells] == corruption))
            else:
                _require(np.any(inp == corruption), 'test regime lacks damaged cases')
            sol = _solutions(inp, divider, corruption)
            _require(len(sol) == 1, 'input has multiple or no global reconstructions')
            carrier, accent, template, predicted = sol[0]
            _require(np.array_equal(out, predicted), 'incorrect two-scale swap')
            _require(carrier != accent and 0 < np.count_nonzero(template == carrier) < n*n,
                     'template is uniform')
            mask = _divider_mask(n)
            _require(np.any(inp[mask] == divider), 'divider invisible')
            _require(corruption not in np.unique(out), 'corruption survives output')
    _require({2, 3}.issubset(train_orders), 'training must demonstrate orders 2 and 3')
    _require(train_corruption and train_divider_damage and train_payload_damage,
             'training must show payload and divider repair')


def _same(first, second):
    return all(len(first[s]) == len(second[s]) and
               all(np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s])
                   for k in ('input', 'output')) for s in ('train', 'test'))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    _require(len({len(p['train']) for p in puzzles}) >= 4, 'few train counts')
    _require(len({len(p['test']) for p in puzzles}) >= 3, 'few test counts')
    _require(len({_text(p['test'][0]['input']) for p in puzzles}) == 42,
             'repeated test grids')
    _require({2, 3, 4, 5}.issubset({ _order(pair['input']) for p in puzzles
                                     for pair in p['test']}), 'test orders too narrow')
    metrics = {name: 0 for name in (
        'damaged_order2_train', 'clean_order3_train', 'clean_order4_train',
        'same_order_test', 'mixed_test_orders', 'order5_test', 'reused_colors',
        'reversed_colors', 'clean_blocks', 'all_blocks_damaged',
        'occluded_carrier', 'whole_block', 'large_patch', 'many_dividers',
        'sparse', 'dense', 'symmetric', 'striped', 'clustered', 'irregular')}
    layouts = set()
    combinations = set()
    for puzzle in puzzles:
        divider, corruption = _convention(puzzle['train'])
        train_orders = [_order(pair['input']) for pair in puzzle['train']]
        test_orders = [_order(pair['input']) for pair in puzzle['test']]
        metrics['same_order_test'] += bool(set(train_orders) & set(test_orders))
        metrics['mixed_test_orders'] += len(set(test_orders)) > 1
        metrics['order5_test'] += 5 in test_orders
        prior = []
        for section in ('train', 'test'):
            for pair in puzzle[section]:
                inp = pair['input']
                n = _order(inp)
                carrier, accent, template, predicted = _solutions(inp, divider, corruption)[0]
                _require(np.array_equal(predicted, pair['output']), 'self-test prediction failed')
                damaged = inp == corruption
                count = int(damaged.sum())
                if section == 'train':
                    metrics['damaged_order2_train'] += n == 2 and count > 0
                    metrics['clean_order3_train'] += n == 3 and count == 0
                    metrics['clean_order4_train'] += n == 4 and count == 0
                palette = tuple(sorted((carrier, accent)))
                metrics['reused_colors'] += any(tuple(sorted(x)) == palette for x in prior)
                metrics['reversed_colors'] += (accent, carrier) in prior
                prior.append((carrier, accent))
                bits = template == carrier
                layouts.add((n, bits.tobytes()))
                metrics['sparse'] += int(bits.sum()) <= 2
                metrics['dense'] += int((~bits).sum()) <= 2
                symmetric = any(np.array_equal(bits, other) for other in
                                (np.fliplr(bits), np.flipud(bits), bits.T))
                striped = np.all(bits == bits[:, :1]) or np.all(bits == bits[:1, :])
                metrics['symmetric'] += symmetric
                metrics['striped'] += striped
                members = set(map(tuple, np.argwhere(bits)))
                components = 0
                while members:
                    components += 1
                    stack = [members.pop()]
                    while stack:
                        r, c = stack.pop()
                        for neighbor in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                            if neighbor in members:
                                members.remove(neighbor)
                                stack.append(neighbor)
                middle_density = 2 < int(bits.sum()) < n*n-2
                metrics['clustered'] += components == 1 and middle_density and not striped
                metrics['irregular'] += middle_density and not symmetric and not striped
                blocks = [damaged[_block_slice(n, i, j)] for i in range(n) for j in range(n)]
                metrics['clean_blocks'] += any(not np.any(b) for b in blocks)
                metrics['all_blocks_damaged'] += all(np.any(b) for b in blocks)
                metrics['whole_block'] += any(np.all(b) for b in blocks)
                occluded = any(bits[i, j] and np.all(damaged[_block_slice(n, i, j)][bits])
                               for i in range(n) for j in range(n))
                metrics['occluded_carrier'] += occluded
                div_count = int(np.count_nonzero(damaged & _divider_mask(n)))
                metrics['many_dividers'] += div_count >= 20
                remaining = set(map(tuple, np.argwhere(damaged)))
                largest = 0
                while remaining:
                    stack = [remaining.pop()]
                    size = 0
                    while stack:
                        r, c = stack.pop()
                        size += 1
                        for neighbor in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                            if neighbor in remaining:
                                remaining.remove(neighbor)
                                stack.append(neighbor)
                    largest = max(largest, size)
                metrics['large_patch'] += largest >= 25
                combinations.add((section, n, count == 0, occluded,
                                  any(tuple(sorted(x)) == palette for x in prior[:-1])))
    for name, minimum in (
        ('damaged_order2_train', 5), ('clean_order3_train', 5),
        ('clean_order4_train', 2), ('same_order_test', 5),
        ('mixed_test_orders', 5), ('order5_test', 5),
        ('reused_colors', 20), ('reversed_colors', 10),
        ('clean_blocks', 20), ('all_blocks_damaged', 5),
        ('occluded_carrier', 5), ('whole_block', 2),
        ('large_patch', 10), ('many_dividers', 5),
        ('sparse', 20), ('dense', 20), ('symmetric', 20), ('striped', 10),
        ('clustered', 5), ('irregular', 5)):
        _require(metrics[name] >= minimum, f'insufficient {name} coverage')
    _require(len(layouts) >= 120 and len(combinations) >= 20,
             'too few structural layouts or combined variation modes')
    for nt, ns in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                   (2, None), (10, None), (7, 3)):
        a = generate_puzzle(314159, nt, ns)
        b = generate_puzzle(314159, nt, ns)
        _require(_same(a, b), 'nondeterministic puzzle')
        validate_generated_puzzle(a)
        if nt is not None:
            _require(len(a['train']) == nt, 'explicit train count ignored')
        if ns is not None:
            _require(len(a['test']) == ns, 'explicit test count ignored')
    for nt, ns in ((1, 1), (11, 1), (2, 0), (6, 5), (3, 3), (2, 4)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid count accepted')
