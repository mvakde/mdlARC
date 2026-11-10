"""Two-color mask-complement ARC puzzle generator."""
import numpy as np

_SOURCE_ROWS = (
    ((5,5,5),(0,0,5),(0,0,0)), ((0,8,0),(0,8,0),(8,0,0)),
    ((8,0,8),(0,8,0),(0,8,0)), ((0,0,3),(0,3,0),(3,0,0)),
    ((5,0,0),(5,5,0),(5,0,0)), ((8,0,0),(0,8,0),(0,0,0)),
    ((0,5,0),(5,5,0),(0,0,5)), ((3,0,0),(3,3,3),(0,0,3)),
)
_RECTANGLES = ((2,3),(3,2),(3,4),(4,3),(4,5),(5,4),(3,6),(6,3),(5,7),(7,5))
_SQUARES = ((4,4),(5,5),(6,6),(7,7))


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _key(grid):
    return (tuple(grid.shape), tuple(int(v) for v in grid.flat))


_SOURCE_KEYS = frozenset(_key(np.asarray(rows, dtype=np.int8)) for rows in _SOURCE_ROWS)


def _seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    return int(seed)


def _count(name, value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    return int(value)


def _counts(seed, num_train, num_test):
    train, test = _count('num_train', num_train), _count('num_test', num_test)
    if train is not None and not 2 <= train <= 10:
        raise ValueError('num_train must be 2..10')
    if test is not None and not 1 <= test <= 4:
        raise ValueError('num_test must be 1..4')
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError('num_train must exceed num_test')
    return train, test


def _colors(rng, seed, train_count):
    mapping_count = 1 if seed % 11 == 0 else int(rng.integers(1, min(3, train_count) + 1))
    background_count = 1 if train_count == 2 else int(rng.choice((1,2,3), p=(0.61,0.32,0.07)))
    background_count = min(background_count, 10 - 2 * mapping_count)
    colors = [int(v) for v in rng.permutation(10)]
    if rng.random() < 0.55:
        colors.remove(0)
        colors.insert(0, 0)
    # Exercise an azure-only puzzle, absent from the reviewed palette bank.
    if mapping_count == 1 and seed % 11 == 0:
        colors.remove(8)
        colors.insert(background_count, 8)
    backgrounds = tuple(colors[:background_count])
    sources = tuple(colors[background_count:background_count + mapping_count])
    partners = colors[background_count + mapping_count:background_count + 2 * mapping_count]
    return backgrounds, dict(zip(sources, partners)), sources


def _shape(rng, allow_three=True):
    draw = float(rng.random())
    if allow_three and draw < 0.42:
        return (3,3)
    options = _RECTANGLES if draw < 0.77 else _SQUARES
    return options[int(rng.integers(len(options)))]


def _density(rng, area):
    draw = float(rng.random())
    if draw < 0.105:
        return 1
    if draw < 0.21:
        return area - 1
    if draw < 0.48:
        low, high = 2, max(2, area // 3)
    elif draw < 0.77:
        low, high = max(2, area // 3 + 1), min(area - 2, 2 * area // 3)
    else:
        low, high = min(area - 2, 2 * area // 3 + 1), area - 2
    return int(rng.integers(low, max(low, high) + 1))


def _mask(rng, shape, density):
    flat = np.zeros(shape[0] * shape[1], dtype=bool)
    flat[rng.choice(flat.size, size=density, replace=False)] = True
    return flat.reshape(shape)


def _mask_key(mask):
    return (tuple(mask.shape), tuple(bool(v) for v in mask.flat))


def _make_pair(rng, shape, density, background, source, partner, used,
               forbidden_masks=None, fixed_mask=None):
    for attempt in range(512):
        d = density if attempt < 256 or fixed_mask is not None else int(rng.integers(1, shape[0] * shape[1]))
        mask = fixed_mask if fixed_mask is not None else _mask(rng, shape, d)
        inp = np.where(mask, source, background).astype(np.int8)
        key = _key(inp)
        if key in used or key in _SOURCE_KEYS or (forbidden_masks is not None and _mask_key(mask) in forbidden_masks):
            if fixed_mask is not None:
                raise RuntimeError('controlled recolor duplicated an input')
            continue
        used.add(key)
        out = np.where(mask, background, partner).astype(np.int8)
        return {'input': inp, 'output': out}
    raise RuntimeError('could not construct a distinct valid mask')


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _seed(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    backgrounds, mapping, sources = _colors(rng, seed, train_count)

    train_sources = list(sources)
    train_sources.extend(sources[int(rng.integers(len(sources)))] for _ in range(train_count - len(sources)))
    rng.shuffle(train_sources)
    train_backgrounds = list(backgrounds)
    train_backgrounds.extend(backgrounds[int(rng.integers(len(backgrounds)))] for _ in range(train_count - len(backgrounds)))
    rng.shuffle(train_backgrounds)
    if test_count > 1 and rng.random() < 0.55:
        cycle = [int(v) for v in rng.permutation(sources)]
        test_sources = [cycle[i % len(cycle)] for i in range(test_count)]
    else:
        test_sources = [sources[int(rng.integers(len(sources)))] for _ in range(test_count)]
    test_backgrounds = [backgrounds[int(rng.integers(len(backgrounds)))] for _ in range(test_count)]

    if rng.random() < 0.24:
        train_shapes, test_shapes = [(3,3)] * train_count, [(3,3)] * test_count
    else:
        train_shapes = [(3,3), _shape(rng, allow_three=False)]
        train_shapes.extend(_shape(rng) for _ in range(train_count - 2))
        test_shapes = [_shape(rng) for _ in range(test_count)]
    train_densities = [_density(rng, h * w) for h, w in train_shapes]
    train_densities[0] = int(rng.integers(1, max(2, train_shapes[0][0] * train_shapes[0][1] // 3) + 1))
    area_second = train_shapes[1][0] * train_shapes[1][1]
    train_densities[1] = int(rng.integers(area_second // 2 + 1, area_second))
    test_densities = [_density(rng, h * w) for h, w in test_shapes]

    recolor = None
    if len(sources) > 1 and train_count >= 3 and rng.random() < 0.43:
        options = [(i,j) for i in (0,1) for j in range(2, train_count)
                   if train_sources[i] != train_sources[j]
                   and (train_backgrounds[j] == train_backgrounds[i]
                        or train_backgrounds.count(train_backgrounds[j]) > 1)]
        if options:
            recolor = options[int(rng.integers(len(options)))]
            i, j = recolor
            train_shapes[j], train_densities[j], train_backgrounds[j] = (
                train_shapes[i], train_densities[i], train_backgrounds[i]
            )

    used = set()
    training = []
    for i in range(train_count):
        fixed = None
        if recolor is not None and i == recolor[1]:
            fixed = training[recolor[0]]['input'] == train_sources[recolor[0]]
        training.append(_make_pair(rng, train_shapes[i], train_densities[i],
                                  train_backgrounds[i], train_sources[i],
                                  mapping[train_sources[i]], used, fixed_mask=fixed))
    training_masks = {_mask_key(p['input'] == source) for p, source in zip(training, train_sources)}
    testing = [_make_pair(rng, test_shapes[i], test_densities[i], test_backgrounds[i],
                          test_sources[i], mapping[test_sources[i]], used,
                          forbidden_masks=training_masks) for i in range(test_count)]
    puzzle = {'train': training, 'test': testing}
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid, label):
    _require(isinstance(grid, np.ndarray), f'{label} must be an array')
    _require(grid.ndim == 2, f'{label} must be 2D')
    _require(all(1 <= d <= 30 for d in grid.shape), f'{label} dimensions are invalid')
    _require(np.issubdtype(grid.dtype, np.integer), f'{label} dtype is not integer')
    _require(np.all((grid >= 0) & (grid <= 9)), f'{label} has invalid color')


def _roles(pair, label):
    inp, out = pair['input'], pair['output']
    _require(inp.shape == out.shape, f'{label} changed dimensions')
    inp_colors = set(int(v) for v in np.unique(inp))
    out_colors = set(int(v) for v in np.unique(out))
    _require(len(inp_colors) == len(out_colors) == 2, f'{label} is not two-color')
    common = inp_colors & out_colors
    _require(len(common) == 1, f'{label} lacks a unique shared background')
    background = next(iter(common))
    source = next(iter(inp_colors - common))
    partner = next(iter(out_colors - common))
    _require(len({background, source, partner}) == 3, f'{label} color roles overlap')
    mask = inp == source
    _require(np.array_equal(out, np.where(mask, background, partner)), f'{label} is not an exact complement')
    _require(_key(inp) not in _SOURCE_KEYS, f'{label} copied a source input')
    return background, source, partner, _mask_key(mask)


def _training_convention(training):
    backgrounds, mapping = set(), {}
    for i, pair in enumerate(training):
        bg, source, partner, _ = _roles(pair, f'train[{i}]')
        backgrounds.add(bg)
        if source in mapping:
            _require(mapping[source] == partner, 'training mapping is inconsistent')
        mapping[source] = partner
    _require(len(set(mapping.values())) == len(mapping), 'output partners are not distinct')
    _require(not (backgrounds & set(mapping)), 'source and background roles overlap')
    _require(not (backgrounds & set(mapping.values())), 'partner and background roles overlap')
    _require(not (set(mapping) & set(mapping.values())), 'partner and source roles overlap')
    return backgrounds, mapping


def _predict(training, test_input):
    backgrounds, mapping = _training_convention(training)
    colors = set(int(v) for v in np.unique(test_input))
    source_colors = colors & set(mapping)
    bg_colors = colors & backgrounds
    _require(len(colors) == 2 and len(source_colors) == len(bg_colors) == 1,
             'test input has unseen or ambiguous color roles')
    source, bg = next(iter(source_colors)), next(iter(bg_colors))
    return np.where(test_input == source, bg, mapping[source]).astype(np.int8)


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train','test'}, 'wrong puzzle keys')
    _require(isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list), 'splits must be lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'illegal pair counts')
    inputs, pairs, train_masks, roles = set(), set(), set(), []
    for split in ('train','test'):
        for i, pair in enumerate(puzzle[split]):
            label = f'{split}[{i}]'
            _require(isinstance(pair, dict) and set(pair) == {'input','output'}, f'{label} has wrong keys')
            _validate_grid(pair['input'], label + '.input')
            _validate_grid(pair['output'], label + '.output')
            ikey = _key(pair['input'])
            pkey = (ikey, _key(pair['output']))
            _require(ikey not in inputs and pkey not in pairs, f'{label} duplicates an input or pair')
            inputs.add(ikey)
            pairs.add(pkey)
            current = _roles(pair, label)
            roles.append(current)
            if split == 'train':
                train_masks.add(current[3])
            else:
                _require(current[3] not in train_masks, f'{label} recolors a training mask')
    backgrounds, mapping = _training_convention(puzzle['train'])
    _require(len({r[3] for r in roles[:nt]}) >= 2, 'training lacks spatial variation')
    first_density = int(np.count_nonzero(puzzle['train'][0]['input'] == roles[0][1]))
    second_density = int(np.count_nonzero(puzzle['train'][1]['input'] == roles[1][1]))
    _require(3 * first_density <= puzzle['train'][0]['input'].size,
             'first training mask must show sparse occupancy')
    _require(2 * second_density > puzzle['train'][1]['input'].size,
             'second training mask must show dense occupancy')
    for i, pair in enumerate(puzzle['test']):
        bg, source, partner, _ = roles[nt+i]
        _require(bg in backgrounds and source in mapping and mapping[source] == partner,
                 f'test[{i}] has an untrained convention')
        _require(np.array_equal(_predict(puzzle['train'], pair['input'].copy()), pair['output']),
                 f'test[{i}] is not recoverable from training')


def _signature(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s])
                 for s in ('train','test'))


def _expect_value_error(**kwargs):
    try:
        generate_puzzle(71, **kwargs)
    except ValueError:
        return
    raise AssertionError(f'expected ValueError for {kwargs}')


def self_test() -> None:
    for seed, nt, nq in ((701,2,1),(702,10,1),(703,10,4),(711,None,1),
                         (712,None,4),(713,2,None),(714,10,None),(715,7,3)):
        puzzle = generate_puzzle(seed, nt, nq)
        validate_generated_puzzle(puzzle)
        if nt is not None:
            _require(len(puzzle['train']) == nt, 'requested train count was lost')
        if nq is not None:
            _require(len(puzzle['test']) == nq, 'requested test count was lost')
    for kwargs in ({'num_train':1},{'num_train':11},{'num_test':0},{'num_test':5},
                   {'num_train':2,'num_test':2},{'num_train':3,'num_test':4},
                   {'num_train':True},{'num_test':2.0}):
        _expect_value_error(**kwargs)
    _require(_signature(generate_puzzle(314159,7,3)) == _signature(generate_puzzle(314159,7,3)),
             'generation is nondeterministic')
    counts_train, counts_test, signatures = set(), set(), set()
    shapes, sources, partners, backgrounds, density3, structures = set(), set(), set(), set(), set(), set()
    variable_bg = variable_size = recolors = same_density = sparse_test = dense_test = only8 = test8 = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        counts_train.add(len(puzzle['train']))
        counts_test.add(len(puzzle['test']))
        signatures.add(_signature(puzzle))
        train_roles = [_roles(p,'self-test train') for p in puzzle['train']]
        test_roles = [_roles(p,'self-test test') for p in puzzle['test']]
        variable_bg += len({r[0] for r in train_roles}) > 1
        variable_size += len({p['input'].shape for p in puzzle['train'] + puzzle['test']}) > 1
        only8 += {r[1] for r in train_roles} == {8}
        test8 += any(r[1] == 8 for r in test_roles)
        train_density = {int(np.count_nonzero(p['input'] == r[1]))
                         for p,r in zip(puzzle['train'], train_roles)}
        same_density += any(int(np.count_nonzero(p['input'] == r[1])) in train_density
                            for p,r in zip(puzzle['test'], test_roles))
        for p,r in zip(puzzle['test'], test_roles):
            d = int(np.count_nonzero(p['input'] == r[1]))
            sparse_test += d / p['input'].size <= 1/3
            dense_test += d / p['input'].size >= 2/3
            _require(np.array_equal(_predict(puzzle['train'], p['input'].copy()), p['output']),
                     'training-only prediction failed')
        for i, first in enumerate(train_roles):
            for other in train_roles[i+1:]:
                recolors += first[3] == other[3] and first[1] != other[1]
        for split in ('train','test'):
            for p in puzzle[split]:
                bg, source, partner, mask = _roles(p,'self-test pair')
                shapes.add(tuple(p['input'].shape))
                sources.add(source)
                partners.add(partner)
                backgrounds.add(bg)
                structures.add(mask)
                if p['input'].shape == (3,3):
                    density3.add(int(np.count_nonzero(p['input'] == source)))
    _require(len(counts_train) >= 4 and len(counts_test) >= 3, 'counts lack diversity')
    _require(len(signatures) >= 40 and len(structures) >= 100, 'puzzles or masks lack diversity')
    _require(len(shapes) >= 8 and (3,3) in shapes and any(h == w and h > 3 for h,w in shapes)
             and any(h != w for h,w in shapes), 'sizes lack diversity')
    _require({1,8}.issubset(density3), '3x3 extremes were not sampled')
    _require(len(sources) >= 7 and len(partners) >= 7 and len(backgrounds) >= 4,
             'color conventions lack diversity')
    _require(variable_bg >= 3 and variable_size >= 15 and recolors >= 1,
             'within-puzzle background, size, or mask recolor variation is missing')
    _require(same_density >= 5 and sparse_test >= 5 and dense_test >= 5,
             'test density regimes lack diversity')
    _require(only8 >= 1 and test8 >= 1, 'color 8 is not exercised as a complete mapping')
