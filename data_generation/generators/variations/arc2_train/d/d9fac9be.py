import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid_fingerprint(grid):
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.reshape(-1)),
    )


def _literal_pair_fingerprint(rows, output_color):
    input_fp = (
        (len(rows), len(rows[0])),
        tuple(int(ch) for row in rows for ch in row),
    )
    output_fp = ((1, 1), (int(output_color),))
    return input_fp, output_fp


_SOURCE_PAIR_FINGERPRINTS = {
    _literal_pair_fingerprint(
        (
            "120002000",
            "002000000",
            "201202011",
            "010020002",
            "000000100",
            "000000000",
            "022200000",
            "121200020",
            "022200002",
            "001000000",
            "000200000",
        ),
        1,
    ),
    _literal_pair_fingerprint(
        (
            "808000008",
            "000080000",
            "008003330",
            "800303830",
            "000003330",
            "008000000",
            "300800080",
        ),
        8,
    ),
    _literal_pair_fingerprint(
        (
            "080000000038",
            "300000080300",
            "033800000008",
            "000380000000",
            "300000000080",
            "000380000000",
            "030000000000",
            "000333008030",
            "003383000000",
            "000333000000",
            "003000000000",
        ),
        8,
    ),
    _literal_pair_fingerprint(
        (
            "200002002",
            "044400000",
            "042400200",
            "044400020",
            "200002000",
        ),
        2,
    ),
    _literal_pair_fingerprint(
        (
            "000000000100",
            "100000400100",
            "000000000000",
            "000014004000",
            "040100000000",
            "000001044001",
            "100000000000",
            "000100004000",
            "000011100004",
            "400014110000",
            "000011100004",
            "004400010000",
        ),
        4,
    ),
}


def _background(grid):
    counts = np.bincount(grid.reshape(-1), minlength=10)
    return int(np.argmax(counts))


def _find_enclosed_cells(grid):
    bg = _background(grid)
    h, w = grid.shape
    found = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            center = int(grid[r, c])
            ring = int(grid[r - 1, c - 1])
            if center == bg or ring == bg or center == ring:
                continue
            if all(int(grid[r + dr, c + dc]) == ring
                   for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                   if dr or dc):
                found.append((r, c, center, ring))
    return found


def _cross_decoys(grid):
    bg = _background(grid)
    h, w = grid.shape
    full = {(r, c) for r, c, _, _ in _find_enclosed_cells(grid)}
    result = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            center = int(grid[r, c])
            ring = int(grid[r - 1, c])
            if (r, c) in full or center == bg or ring == bg or center == ring:
                continue
            if (int(grid[r + 1, c]) == ring
                    and int(grid[r, c - 1]) == ring
                    and int(grid[r, c + 1]) == ring):
                result.append((r, c))
    return result


def _check_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer or None')
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f'{name} must be between {low} and {high}')
    return value


def _resolve_counts(rng, num_train, num_test):
    nt = _check_count('num_train', num_train, 2, 10)
    nx = _check_count('num_test', num_test, 1, 4)
    if nt is None and nx is None:
        # The compact four-example format is common, alongside longer lessons.
        if rng.random() < .38:
            nt, nx = 4, 1
        else:
            nx = int(rng.integers(1, 5))
            nt = int(rng.integers(nx + 1, 11))
    elif nt is None:
        nt = int(rng.integers(max(2, nx + 1), 11))
    elif nx is None:
        nx = int(rng.integers(1, min(4, nt - 1) + 1))
    if nt <= nx:
        raise ValueError('num_train must exceed num_test')
    return nt, nx


def _puzzle_plan(rng, n_train, n_test):
    main_bg = 0 if rng.random() < .48 else int(rng.integers(1, 10))
    backgrounds = [main_bg]
    if n_train >= 3 and rng.random() < .28:
        backgrounds.append(int(rng.choice([c for c in range(10) if c != main_bg])))
    available = [c for c in range(10) if c not in backgrounds]
    size = int(rng.integers(3, min(5, 2 * n_train) + 1))
    if 0 in available and rng.random() < .60:
        pool = [0] + [int(x) for x in rng.choice([c for c in available if c],
                                                  size=size - 1, replace=False)]
    else:
        pool = [int(x) for x in rng.choice(available, size=size, replace=False)]
    rng.shuffle(pool)
    held = tuple(int(x) for x in rng.choice(pool, size=2, replace=False))
    candidates = [(a, b) for a in pool for b in pool
                  if a != b and set((a, b)) != set(held)]
    a, b = held
    rest = [color for color in pool if color not in held]
    if len(rest) == 1:
        train_roles = [(a, rest[0]), (rest[0], b)]
    else:
        train_roles = [(a, rest[0]), (rest[1], b)]
        train_roles.extend((color, rest[0]) for color in rest[2:])
    while len(train_roles) < n_train:
        unused = [role for role in candidates if role not in train_roles]
        choices = unused if unused else candidates
        train_roles.append(choices[int(rng.integers(len(choices)))])
    rng.shuffle(train_roles)
    _require({color for role in train_roles for color in role} == set(pool),
             'role schedule missed a color')
    test_roles = [(held if rng.random() < .5 else held[::-1])]
    for _ in range(1, n_test):
        choice = rng.random()
        if choice < .35:
            role = train_roles[int(rng.integers(len(train_roles)))]
        elif choice < .70:
            role = train_roles[int(rng.integers(len(train_roles)))][::-1]
        else:
            role = candidates[int(rng.integers(len(candidates)))]
        test_roles.append(role)
    train_bgs = [backgrounds[i % len(backgrounds)] for i in range(n_train)]
    rng.shuffle(train_bgs)
    test_bgs = [backgrounds[int(rng.integers(len(backgrounds)))]
                for _ in range(n_test)]
    teach_decoy = rng.random() < .64
    decoy_train_index = int(rng.integers(n_train)) if teach_decoy else -1
    return train_roles, test_roles, train_bgs, test_bgs, decoy_train_index


def _sample_shape(rng, split, train_shapes, decoy, favor_center):
    if split == 'test' and rng.random() < .23:
        base_h, base_w = train_shapes[int(rng.integers(len(train_shapes)))]
        h = min(25, base_h + int(rng.integers(0, 4)))
        w = min(25, base_w + int(rng.integers(0, 4)))
    else:
        h = int(rng.integers(5, 14)) if rng.random() < .83 else int(rng.integers(14, 25))
        w = int(rng.integers(5, 14)) if rng.random() < .83 else int(rng.integers(14, 25))
    if decoy:
        h, w = max(h, 8), max(w, 8)
    if favor_center and not decoy:
        while h * w < 35:
            if h <= w:
                h += 1
            else:
                w += 1
    return h, w


def _place_motifs(rng, h, w, bg, center_color, ring_color, decoy_kind):
    grid = np.full((h, w), bg, dtype=np.int64)
    positions = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    if rng.random() < .38:
        edge_positions = [(r, c) for r, c in positions
                          if r in (1, h - 2) or c in (1, w - 2)]
        r, c = edge_positions[int(rng.integers(len(edge_positions)))]
    else:
        r, c = positions[int(rng.integers(len(positions)))]
    grid[r - 1:r + 2, c - 1:c + 2] = ring_color
    grid[r, c] = center_color
    locked = {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
    if decoy_kind is not None:
        choices = [(rr, cc) for rr, cc in positions
                   if abs(rr - r) >= 3 or abs(cc - c) >= 3]
        if not choices:
            return None
        rr, cc = choices[int(rng.integers(len(choices)))]
        grid[rr - 1:rr + 2, cc - 1:cc + 2] = ring_color
        if decoy_kind in ('diagonal', 'orthogonal'):
            grid[rr, cc] = center_color
            if decoy_kind == 'diagonal':
                dr, dc = ((-1, -1), (-1, 1), (1, -1), (1, 1))[int(rng.integers(4))]
            else:
                dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[int(rng.integers(4))]
            grid[rr + dr, cc + dc] = bg
        elif decoy_kind == 'empty':
            grid[rr, cc] = bg
        locked.update((rr + dr, cc + dc)
                      for dr in (-1, 0, 1) for dc in (-1, 0, 1))
    return grid, locked, (r, c)


def _distractor_order(rng, free, style):
    if style == 'speckle':
        rng.shuffle(free)
        return free
    anchors = [free[int(rng.integers(len(free)))] for _ in range(int(rng.integers(1, 4)))]
    jitter = rng.random(len(free))
    if style == 'clump':
        score = [min(abs(r - ar) + abs(c - ac) for ar, ac in anchors) + 1.5 * noise
                 for (r, c), noise in zip(free, jitter)]
    elif style == 'bars':
        score = []
        for (r, c), noise in zip(free, jitter):
            d = min((abs(r - ar) + max(0, abs(c - ac) - 2))
                    if i % 2 else (abs(c - ac) + max(0, abs(r - ar) - 2))
                    for i, (ar, ac) in enumerate(anchors))
            score.append(d + noise)
    else:  # a short diagonal texture, never a compulsory full-canvas band
        ar, ac = anchors[0]
        slope = -1 if rng.random() < .5 else 1
        score = [abs((r - ar) - slope * (c - ac))
                 + .2 * (abs(r - ar) + abs(c - ac)) + noise
                 for (r, c), noise in zip(free, jitter)]
    return [free[i] for i in np.argsort(score)]


def _generate_pair(rng, split, shape, bg, roles, decoy_kind,
                   allow_untaught_cross, favor_center):
    h, w = shape
    center_color, ring_color = roles
    for layout_attempt in range(18):
        placed = _place_motifs(rng, h, w, bg, center_color, ring_color, decoy_kind)
        if placed is None:
            continue
        base, locked, motif = placed
        if [x[:2] for x in _find_enclosed_cells(base)] != [motif]:
            continue
        free = [(r, c) for r in range(h) for c in range(w) if (r, c) not in locked]
        style = ('speckle', 'clump', 'bars', 'diagonal')[int(rng.choice(4, p=[.53, .25, .17, .05]))]
        regime = int(rng.choice(4, p=[.16, .35, .31, .18] if split == 'train'
                                else [.10, .29, .36, .25]))
        if regime == 0:
            target = int(rng.integers(0, 5))
        else:
            lo, hi = ((.02, .10), (.10, .21), (.21, .33))[regime - 1]
            target = int(rng.uniform(lo, hi) * h * w)
        # Keep the background unequivocally dominant, including on small grids.
        max_extra = max(0, (h * w - 1) // 2 - int(np.count_nonzero(base != bg)))
        target = min(target, len(free), max_extra)
        if layout_attempt >= 10:
            target //= 2 ** (layout_attempt - 9)
        base_center = int(np.count_nonzero(base == center_color))
        base_ring = int(np.count_nonzero(base == ring_color))
        needed = base_ring - base_center + 1
        if favor_center and max_extra >= needed:
            target = max(target, needed)
            minimum_center = (target + base_ring - base_center) // 2 + 1
            center_extra = int(rng.integers(minimum_center, target + 1))
        else:
            center_share = float(rng.uniform(.12, .48 if not favor_center else .70))
            center_extra = int(round(target * center_share))
        ring_extra = target - center_extra
        if (not favor_center and regime == 0 and decoy_kind is None
                and rng.random() < .50):
            center_extra += ring_extra
            ring_extra = 0
        colors = [center_color] * center_extra + [ring_color] * ring_extra
        rng.shuffle(colors)
        grid = base.copy()
        for (r, c), color in zip(_distractor_order(rng, free, style), colors):
            grid[r, c] = color
        matches = _find_enclosed_cells(grid)
        if len(matches) != 1 or matches[0][:2] != motif:
            continue
        if _background(grid) != bg:
            continue
        if not allow_untaught_cross and _cross_decoys(grid):
            continue
        return grid, np.array([[center_color]], dtype=np.int64)
    raise RuntimeError('could not construct a unique ring')


def _pair_info(pair):
    grid, output = pair['input'], pair['output']
    matches = _find_enclosed_cells(grid)
    _require(len(matches) == 1, 'input must have one complete contrasting ring')
    r, c, center, ring = matches[0]
    bg = _background(grid)
    _require(int(output[0, 0]) == center, 'output differs from enclosed center')
    _require(set(int(x) for x in np.unique(grid)) == {bg, center, ring},
             'pair must have background and two foreground colors')
    counts = np.bincount(grid.reshape(-1), minlength=10)
    _require(counts[bg] > sum(counts) // 2, 'background is not dominant')
    return {'shape': grid.shape, 'position': (r, c), 'background': bg,
            'center': center, 'ring': ring, 'center_count': int(counts[center]),
            'ring_count': int(counts[ring]), 'cross_decoys': len(_cross_decoys(grid))}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'},
             'puzzle requires train and test')
    train, test = puzzle['train'], puzzle['test']
    _require(isinstance(train, list) and isinstance(test, list), 'splits must be lists')
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4
             and len(train) > len(test), 'illegal pair counts')
    seen_inputs, seen_pairs = set(), set()
    infos = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'},
                     'pair requires input and output')
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         'grid must be a 2D NumPy array')
                h, w = grid.shape
                _require(1 <= h <= 30 and 1 <= w <= 30, 'grid size out of bounds')
                _require(np.issubdtype(grid.dtype, np.integer)
                         and int(grid.min()) >= 0 and int(grid.max()) <= 9,
                         'grid cells must be ARC integers')
            _require(pair['input'].shape[0] >= 5 and pair['input'].shape[1] >= 5,
                     'input too small for this family')
            _require(pair['output'].shape == (1, 1), 'output must be 1x1')
            input_fp = _grid_fingerprint(pair['input'])
            pair_fp = (input_fp, _grid_fingerprint(pair['output']))
            _require(input_fp not in seen_inputs and pair_fp not in seen_pairs,
                     'duplicate input or pair')
            _require(pair_fp not in _SOURCE_PAIR_FINGERPRINTS, 'source pair copied')
            seen_inputs.add(input_fp)
            seen_pairs.add(pair_fp)
            infos[split].append(_pair_info(pair))
    train_colors = {color for info in infos['train']
                    for color in (info['center'], info['ring'])}
    test_colors = {color for info in infos['test']
                   for color in (info['center'], info['ring'])}
    _require(test_colors <= train_colors, 'test uses undemonstrated foreground color')
    train_bgs = {info['background'] for info in infos['train']}
    _require({info['background'] for info in infos['test']} <= train_bgs,
             'test uses undemonstrated background')
    _require(len({info['center'] for info in infos['train']}) >= 2,
             'training must show multiple answer colors')
    if any(info['cross_decoys'] for info in infos['test']):
        _require(any(info['cross_decoys'] for info in infos['train']),
                 'test four-neighbor decoy was not taught')


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    n_train, n_test = _resolve_counts(rng, num_train, num_test)
    train_roles, test_roles, train_bgs, test_bgs, decoy_train_index = (
        _puzzle_plan(rng, n_train, n_test))
    puzzle = {'train': [], 'test': []}
    shapes = []
    seen = set()
    taught = decoy_train_index >= 0
    center_favored_train = 1 if decoy_train_index != 1 else 0
    for split, roles, bgs in (('train', train_roles, train_bgs),
                              ('test', test_roles, test_bgs)):
        for index, (role, bg) in enumerate(zip(roles, bgs)):
            if split == 'train' and index == center_favored_train:
                favor_center = True
            elif split == 'train' and index == 1 - center_favored_train:
                favor_center = False
            else:
                favor_center = bool(rng.random() < .45)
            if split == 'train' and index == decoy_train_index:
                decoy = 'diagonal'
            elif taught and not (split == 'train' and index == center_favored_train) and rng.random() < (.22 if split == 'train' else .48):
                decoy = ('diagonal', 'orthogonal', 'solid', 'empty')[
                    int(rng.choice(4, p=[.50, .20, .18, .12]))]
            else:
                decoy = None
            for _ in range(32):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                shape = _sample_shape(pair_rng, split, shapes, decoy is not None,
                                      favor_center)
                grid, output = _generate_pair(pair_rng, split, shape, bg, role,
                                              decoy, taught, favor_center)
                fp = _grid_fingerprint(grid)
                if fp in seen or (fp, _grid_fingerprint(output)) in _SOURCE_PAIR_FINGERPRINTS:
                    continue
                pair = {'input': grid, 'output': output}
                seen.add(fp)
                puzzle[split].append(pair)
                if split == 'train':
                    shapes.append(shape)
                break
            else:
                raise RuntimeError('failed to generate distinct pair')
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(a, b):
    return all(len(a[split]) == len(b[split]) and all(
        np.array_equal(x['input'], y['input']) and np.array_equal(x['output'], y['output'])
        for x, y in zip(a[split], b[split])) for split in ('train', 'test'))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    _require(_puzzles_equal(puzzles[17], generate_puzzle(17)), 'not deterministic')
    _require(len({len(p['train']) for p in puzzles}) >= 4, 'train counts collapsed')
    _require(len({len(p['test']) for p in puzzles}) >= 3, 'test counts collapsed')
    _require(sum(len(p['train']) == 4 and len(p['test']) == 1 for p in puzzles) >= 5,
             'compact puzzles absent')
    signatures = {tuple((_grid_fingerprint(pair['input']),
                         _grid_fingerprint(pair['output']))
                        for split in ('train', 'test') for pair in p[split])
                  for p in puzzles}
    _require(len(signatures) >= 40, 'puzzles repeat across seeds')
    infos = [(_pair_info(pair), split, p)
             for p in puzzles for split in ('train', 'test') for pair in p[split]]
    _require(len({i['shape'] for i, _, _ in infos}) >= 45, 'shapes collapsed')
    _require(len({(i['shape'], i['position']) for i, _, _ in infos}) >= 90,
             'placements collapsed')
    _require(len({i['background'] for i, _, _ in infos}) >= 5,
             'background colors collapsed')
    _require(any(i['background'] != 0 and 0 in (i['center'], i['ring'])
                 for i, _, _ in infos), 'zero foreground absent')
    _require(any(i['ring_count'] == 8 for i, _, _ in infos),
             'clean rings absent')
    _require({int(np.sign(i['center_count'] - i['ring_count']))
              for i, _, _ in infos} >= {-1, 1}, 'frequency roles collapsed')
    _require(any(i['cross_decoys'] for i, split, _ in infos if split == 'train'),
             'training lacks four-neighbor decoys')
    _require(any(i['cross_decoys'] for i, split, _ in infos if split == 'test'),
             'tests lack four-neighbor decoys')
    _require(any(i['position'][0] in (1, i['shape'][0] - 2)
                 or i['position'][1] in (1, i['shape'][1] - 2)
                 for i, _, _ in infos), 'edge-touching rings absent')
    _require(any(any(t['shape'][0] <= max(i['shape'][0] for i in train_infos)
                     and t['shape'][1] <= max(i['shape'][1] for i in train_infos)
                     for t in test_infos)
                 for p in puzzles
                 for train_infos, test_infos in [
                     ([_pair_info(x) for x in p['train']],
                      [_pair_info(x) for x in p['test']])]),
             'smaller tests absent')
    _require(all({-1, 1} <= {
        int(np.sign(info['center_count'] - info['ring_count']))
        for info in (_pair_info(pair) for pair in p['train'])}
        for p in puzzles), 'training permits a frequency shortcut')
    _require(len({len({color for pair in p['train']
                       for info in [_pair_info(pair)]
                       for color in (info['center'], info['ring'])})
                  for p in puzzles}) >= 3, 'palette pool sizes collapsed')
    _require(sum(len({_pair_info(pair)['background'] for pair in p['train']}) > 1
                 for p in puzzles) >= 5, 'within-puzzle backgrounds do not vary')
    _require(sum(info['background'] != 0 and 0 in (info['center'], info['ring'])
                 for info, _, _ in infos) >= 5, 'zero foreground is too rare')
    _require(sum(info['ring_count'] == 8 for info, _, _ in infos) >= 5,
             'clean rings are too rare')
    _require(sum(max(info['shape']) >= 18
                 for info, _, _ in infos) >= 20, 'large grids are too rare')
    _require(sum(info['shape'][0] >= 2 * info['shape'][1]
                 for info, _, _ in infos) >= 5
             and sum(info['shape'][1] >= 2 * info['shape'][0]
                     for info, _, _ in infos) >= 5,
             'strong aspect-ratio variation is missing')
    novel, repeated, reversed_roles, answers_unseen = 0, 0, 0, 0
    for p in puzzles:
        training = [_pair_info(pair) for pair in p['train']]
        roles = {(info['center'], info['ring']) for info in training}
        answers = {info['center'] for info in training}
        for pair in p['test']:
            info = _pair_info(pair)
            role = (info['center'], info['ring'])
            novel += role not in roles and role[::-1] not in roles
            repeated += role in roles
            reversed_roles += role[::-1] in roles and role not in roles
            answers_unseen += info['center'] not in answers
    _require(novel >= 10 and repeated >= 5 and reversed_roles >= 1,
             'test color-role relationships collapsed')
    _require(answers_unseen >= 1, 'tests never require a new answer color')
    for seed, nt, nx in ((1001, 7, 3), (2001, 2, 1),
                         (2002, 10, 1), (2003, 10, 4)):
        a, b = generate_puzzle(seed, nt, nx), generate_puzzle(seed, nt, nx)
        _require(_puzzles_equal(a, b) and len(a['train']) == nt
                 and len(a['test']) == nx, 'explicit counts or determinism failed')
    for seed, nt, nx in ((3001, None, 1), (3002, None, 4),
                         (3003, 2, None), (3004, 10, None)):
        p = generate_puzzle(seed, nt, nx)
        _require((nt is None or len(p['train']) == nt)
                 and (nx is None or len(p['test']) == nx), 'partial counts failed')
    for kwargs in ({'num_train': 1}, {'num_train': 11}, {'num_test': 0},
                   {'num_test': 5}, {'num_train': 2, 'num_test': 2},
                   {'num_train': 3, 'num_test': 4},
                   {'num_train': True, 'num_test': 1}):
        try:
            generate_puzzle(0, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid count accepted')
