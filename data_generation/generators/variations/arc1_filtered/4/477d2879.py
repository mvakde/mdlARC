"""Generate seeded recolorings of connected regions in orthogonal drawings."""
import itertools
import numpy as np

DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
SOURCE_INPUTS = {
    "0000000000000/0000000000300/0000111100000/0000100100000/0011100110000/0020008010000/0010111010000/0010101010000/0010101011000/0110101001000/0100101011000/0111101110000/0000000000000",
    "0000010000000/0203110018100/0001000010100/1111000010110/0010000010010/0010000110110/0011110100100/0000010106110/0701110100010/0001000100110/0011000111100/0010000000000/0010040000000",
    "0090100000000/1100100111110/0161100100010/0000000100110/0000011100100/0800030000110/0000010080010/0000011000010/1110001011110/0010001110000/0012100000000/0000110000000/7000010000000",
    "0503000000000/0001001111100/0011001000100/1110001100110/0070000100010/0000000108010/0000000100010/0000011100060/0000010000010/1111111111110/0000000010000/0004000110000/0000000100000",
}


def _key(grid):
    return '/'.join(''.join(str(int(v)) for v in row) for row in grid)


def _components(base):
    h, w = base.shape
    seen = np.zeros((h, w), dtype=bool)
    result = []
    for sy in range(h):
        for sx in range(w):
            if seen[sy, sx]:
                continue
            role = int(base[sy, sx])
            cells = []
            queue = [(sy, sx)]
            seen[sy, sx] = True
            for y, x in queue:
                cells.append((y, x))
                for dy, dx in DIRECTIONS:
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and not seen[yy, xx] and int(base[yy, xx]) == role:
                        seen[yy, xx] = True
                        queue.append((yy, xx))
            result.append((role, cells))
    return result


def _neighbor_roles(base, y, x):
    h, w = base.shape
    return [int(base[yy, xx]) for dy, dx in DIRECTIONS
            if 0 <= (yy := y + dy) < h and 0 <= (xx := x + dx) < w]


def _valid_restorations(inp, limit=2):
    """Enumerate role restorations satisfying one seed per 4-connected region.

    The search uses only visible input cells; colors do not encode the role.
    """
    positions = [tuple(map(int, p)) for p in np.argwhere(inp >= 2)]
    if not positions or len(positions) > 9:
        return []
    choices = []
    for y, x in positions:
        nearby = _neighbor_roles(inp, y, x)
        options = tuple(v for v in (0, 1) if v in nearby)
        if not options:
            return []
        choices.append(options)
    result = []
    for roles in itertools.product(*choices):
        base = inp.copy()
        for (y, x), role in zip(positions, roles):
            base[y, x] = role
        comps = _components(base)
        if len(comps) != len(positions):
            continue
        out = np.empty_like(inp)
        for _, cells in comps:
            seeds = [int(inp[y, x]) for y, x in cells if int(inp[y, x]) >= 2]
            if len(seeds) != 1:
                break
            for y, x in cells:
                out[y, x] = seeds[0]
        else:
            result.append((base, out, comps, positions, roles))
            if len(result) >= limit:
                break
    return result


def _far_from_existing(grid, mask):
    if np.any(mask & (grid == 1)):
        return False
    for dy, dx in DIRECTIONS:
        ys, xs = np.nonzero(mask)
        yy, xx = ys + dy, xs + dx
        inside = (yy >= 0) & (yy < grid.shape[0]) & (xx >= 0) & (xx < grid.shape[1])
        if np.any(grid[yy[inside], xx[inside]] == 1):
            return False
    return True


def _span(grid, rng, partial=False):
    """A bent separator or a long open divider with an optional branch."""
    h, w = grid.shape
    horizontal = bool(rng.integers(0, 2))
    view = grid if horizontal else grid.T
    a, b = view.shape
    if partial:
        y = int(rng.integers(3, a - 3))
        end = b - int(rng.integers(2, 4))
        view[y, :end + 1] = 1
        if rng.random() < 0.7:
            x = int(rng.integers(3, end - 1))
            length = int(rng.integers(2, min(5, y) + 1))
            view[y - length:y, x] = 1
        return
    y1 = int(rng.integers(3, a - 5))
    jump = int(rng.choice((-3, -2, 2, 3)))
    y2 = max(2, min(a - 3, y1 + jump))
    x = int(rng.integers(3, b - 3))
    view[y1, :x + 1] = 1
    view[min(y1, y2):max(y1, y2) + 1, x] = 1
    view[y2, x:] = 1


def _loop_mask(shape, rng, divided=False):
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    if h < 9 or w < 9:
        return mask
    height = int(rng.integers(5 if not divided else 7, min(9, h - 2) + 1))
    width = int(rng.integers(5, min(9, w - 2) + 1))
    y = int(rng.integers(1, h - height))
    x = int(rng.integers(1, w - width))
    b, r = y + height - 1, x + width - 1
    mask[y:b + 1, x] = mask[y:b + 1, r] = True
    mask[y, x:r + 1] = mask[b, x:r + 1] = True
    if divided:
        split = int(rng.integers(y + 3, b - 2))
        mask[split, x:r + 1] = True
    if bool(rng.integers(0, 2)):
        side = int(rng.integers(0, 4))
        length = int(rng.integers(1, 4))
        if side == 0 and y - length >= 0:
            mask[y - length:y, x + width // 2] = True
        elif side == 1 and b + length < h:
            mask[b + 1:b + length + 1, x + width // 2] = True
        elif side == 2 and x - length >= 0:
            mask[y + height // 2, x - length:x] = True
        elif side == 3 and r + length < w:
            mask[y + height // 2, r + 1:r + length + 1] = True
    return mask


def _open_mask(shape, rng):
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    height = int(rng.integers(3, min(8, h - 2) + 1))
    width = int(rng.integers(3, min(8, w - 2) + 1))
    y = int(rng.integers(1, h - height))
    x = int(rng.integers(1, w - width))
    b, r = y + height - 1, x + width - 1
    kind = int(rng.integers(0, 4))
    if kind == 0:  # L
        mask[y, x:r + 1] = True
        mask[y:b + 1, r] = True
    elif kind == 1:  # U
        mask[y:b + 1, x] = mask[y:b + 1, r] = True
        mask[b, x:r + 1] = True
    elif kind == 2:  # two bends
        mid = x + width // 2
        mask[y, x:mid + 1] = True
        mask[y:b + 1, mid] = True
        mask[b, mid:r + 1] = True
    else:  # T branch
        mid = x + width // 2
        mask[y, x:r + 1] = True
        mask[y:b + 1, mid] = True
    if bool(rng.integers(0, 2)):
        mask = np.flipud(mask)
    if bool(rng.integers(0, 2)):
        mask = np.fliplr(mask)
    return mask


def _build_base(rng, regime, index):
    for _ in range(100):
        if regime == 'source_train' and index == 0:
            h = int(rng.integers(11, 20))
            w = int(rng.integers(11, 20))
            span = False
            loops = 1
            opens = int(rng.integers(0, 2))
            divided = False
        else:
            h = int(rng.integers(13, 24))
            w = int(rng.integers(13, 24))
            if regime == 'target_test':
                span = bool(rng.integers(0, 4) != 0)
                loops = int(rng.integers(1, 3))
                opens = int(rng.integers(0, 3))
            elif index == 1:
                span = bool(rng.integers(0, 2))
                loops = int(rng.integers(1, 3))
                opens = int(rng.integers(1, 3))
            else:
                span = bool(rng.integers(0, 2))
                loops = int(rng.integers(0, 3))
                opens = int(rng.integers(0, 3))
            divided = bool(rng.integers(0, 4) == 0)
        base = np.zeros((h, w), dtype=np.int8)
        if span:
            _span(base, rng, partial=bool(rng.random() < 0.35))
        success = True
        for i in range(loops):
            placed = False
            for _ in range(50):
                mask = _loop_mask(base.shape, rng, divided and i == 0)
                if np.any(mask) and _far_from_existing(base, mask):
                    base[mask] = 1
                    placed = True
                    break
            if not placed:
                success = False
                break
        if not success:
            continue
        for _ in range(opens):
            placed = False
            for _ in range(50):
                mask = _open_mask(base.shape, rng)
                if np.any(mask) and _far_from_existing(base, mask):
                    base[mask] = 1
                    placed = True
                    break
            if not placed:
                success = False
                break
        if not success:
            continue
        comps = _components(base)
        count = len(comps)
        fg_count = sum(role == 1 for role, _ in comps)
        pockets = sum(role == 0 and not any(y in (0, h - 1) or x in (0, w - 1) for y, x in cells)
                      for role, cells in comps)
        if 3 <= count <= 8 and pockets >= 1 and (index != 1 or regime != 'source_train' or fg_count >= 2):
            return base, comps
    raise RuntimeError('Could not construct an orthogonal drawing')


def _seed_class(base, y, x):
    neighbors = []
    h, w = base.shape
    for dy, dx in DIRECTIONS:
        yy, xx = y + dy, x + dx
        neighbors.append(0 <= yy < h and 0 <= xx < w and base[yy, xx] == 1)
    degree = sum(neighbors)
    if degree == 1:
        return 'endpoint'
    if degree == 2:
        return 'straight' if (neighbors[0] and neighbors[2]) or (neighbors[1] and neighbors[3]) else 'bend'
    return 'branch'


def _choose_seeds(base, comps, rng):
    chosen = []
    h, w = base.shape
    for role, cells in comps:
        if role == 1:
            groups = {name: [] for name in ('endpoint', 'bend', 'straight', 'branch')}
            for y, x in cells:
                groups[_seed_class(base, y, x)].append((y, x))
            names = [name for name in groups if groups[name]]
            weights = np.array([{'endpoint': 3, 'bend': 4, 'straight': 4, 'branch': 1}[name] for name in names], dtype=float)
            name = names[int(rng.choice(len(names), p=weights / weights.sum()))]
            candidates = groups[name]
        else:
            candidates = cells
            # A mixture of near-boundary, edge, and deep field placements.
            preference = int(rng.integers(0, 3))
            if preference == 0:
                near = [(y, x) for y, x in cells if 1 in _neighbor_roles(base, y, x)]
                if near:
                    candidates = near
            elif preference == 1:
                edge = [(y, x) for y, x in cells if y in (0, h - 1) or x in (0, w - 1)]
                if edge:
                    candidates = edge
        y, x = candidates[int(rng.integers(0, len(candidates)))]
        chosen.append((y, x))
    return chosen


def _color_components(comps, rng, heldout, regime, force_background_color=None):
    pool = [c for c in range(2, 10) if regime == 'target_test' or c != heldout]
    colors = [None] * len(comps)
    locked = None
    if force_background_color is not None:
        indices = [i for i, (role, _) in enumerate(comps) if role == 0]
        chosen = indices[int(rng.integers(0, len(indices)))]
        colors[chosen] = force_background_color
        locked = chosen
        pool.remove(force_background_color)
    shuffled = [int(c) for c in rng.permutation(pool)]
    for i, (role, _) in enumerate(comps):
        if colors[i] is not None:
            continue
        if shuffled:
            colors[i] = shuffled.pop()
        else:
            compatible = [colors[j] for j in range(len(comps)) if colors[j] is not None and comps[j][0] == role]
            colors[i] = int(compatible[int(rng.integers(0, len(compatible)))] if compatible else pool[0])
    if regime == 'target_test' and heldout not in colors and rng.random() < 0.65:
        colors[int(rng.integers(0, len(colors)))] = heldout
    if len(comps) >= 4 and rng.random() < 0.4:
        roles = [role for role, _ in comps]
        candidates = [(i, j) for i in range(len(comps)) for j in range(i + 1, len(comps)) if roles[i] == roles[j] and j != locked]
        if candidates:
            i, j = candidates[int(rng.integers(0, len(candidates)))]
            colors[j] = colors[i]
    return colors


def _generate_pair(rng, regime, index, heldout, force_background_color=None):
    for _ in range(100):
        base, comps = _build_base(rng, regime, index)
        for _ in range(35):
            positions = _choose_seeds(base, comps, rng)
            if len(set(positions)) != len(positions):
                continue
            colors = _color_components(comps, rng, heldout, regime, force_background_color)
            inp = base.copy()
            out = np.empty_like(base)
            for (role, cells), (y, x), color in zip(comps, positions, colors):
                inp[y, x] = color
                for cy, cx in cells:
                    out[cy, cx] = color
            restorations = _valid_restorations(inp)
            if len(restorations) == 1 and np.array_equal(restorations[0][0], base) and np.array_equal(restorations[0][1], out):
                return {'input': inp, 'output': out}
    raise RuntimeError('Could not place uniquely recoverable seeds')


def _resolve_counts(seed, num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi):
            raise ValueError(f'{name} is invalid')
    if num_train is None and num_test is None:
        nt = 1 + seed % 4
        nr = nt + 1 + (seed // 4) % (10 - nt)
    elif num_train is None:
        nt = int(num_test)
        nr = nt + 1 + seed % (10 - nt)
    elif num_test is None:
        nr = int(num_train)
        nt = 1 + seed % min(4, nr - 1)
    else:
        nr, nt = int(num_train), int(num_test)
    if nr <= nt:
        raise ValueError('num_train must exceed num_test')
    return nr, nt


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    rng = np.random.default_rng(seed)
    ntrain, ntest = _resolve_counts(seed, num_train, num_test)
    heldout = int(rng.integers(2, 10))
    puzzle = {'train': [], 'test': []}
    seen = set()
    for section, regime, count in (('train', 'source_train', ntrain), ('test', 'target_test', ntest)):
        for index in range(count):
            force_background_color = None
            if section == 'train' and index == 1:
                first_input = puzzle['train'][0]['input']
                _, _, _, seed_positions, roles = _valid_restorations(first_input)[0]
                force_background_color = next(int(first_input[y, x]) for (y, x), role in zip(seed_positions, roles) if role == 1)
            for _ in range(20):
                child = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                pair = _generate_pair(child, regime, index, heldout, force_background_color)
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[section].append(pair)
                    break
            else:
                raise RuntimeError('Could not generate a distinct pair')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    def check(value, message):
        if not value:
            raise AssertionError(message)
    check(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Puzzle schema')
    check(isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list), 'Sections must be lists')
    nr, nt = len(puzzle['train']), len(puzzle['test'])
    check(2 <= nr <= 10 and 1 <= nt <= 4 and nr > nt, 'Pair counts')
    seen = set()
    training_roles = set()
    training_fg_counts = set()
    training_pockets = False
    first_foreground_color = None
    second_background_colors = set()
    for section in ('train', 'test'):
        for index, pair in enumerate(puzzle[section]):
            check(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Pair schema')
            inp, out = pair['input'], pair['output']
            for grid in (inp, out):
                check(isinstance(grid, np.ndarray) and grid.ndim == 2, 'Grid type')
                check(np.issubdtype(grid.dtype, np.integer) and 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30, 'Grid dimensions or dtype')
                check(grid.size > 0 and 0 <= int(grid.min()) <= int(grid.max()) <= 9, 'Grid colors')
            check(inp.shape == out.shape and not np.array_equal(inp, out), 'Output shape or degenerate pair')
            check(_key(inp) not in SOURCE_INPUTS, 'Source input copied')
            key = (inp.shape, inp.tobytes())
            check(key not in seen, 'Duplicate input')
            seen.add(key)
            check(0 in inp and 1 in inp, 'Drawing roles must remain visible')
            found = _valid_restorations(inp)
            check(len(found) == 1, 'Seed roles are ambiguous or invalid')
            base, expected, comps, positions, roles = found[0]
            check(np.array_equal(out, expected), 'Incorrect component recoloring')
            check(3 <= len(comps) <= 8, 'Region count outside supported range')
            fg_count = sum(role == 1 for role, _ in comps)
            check(fg_count >= 1, 'No foreground network')
            pockets = sum(role == 0 and not any(y in (0, inp.shape[0] - 1) or x in (0, inp.shape[1] - 1) for y, x in cells)
                          for role, cells in comps)
            check(pockets >= 1, 'No enclosed region')
            check(not any(np.all(base[y, :] == 1) for y in range(base.shape[0])) and
                  not any(np.all(base[:, x] == 1) for x in range(base.shape[1])), 'Full straight divider')
            if section == 'train':
                training_roles.update(roles)
                training_fg_counts.add(fg_count)
                training_pockets |= pockets > 0
                if index == 0:
                    first_foreground_color = next(int(inp[y, x]) for (y, x), role in zip(positions, roles) if role == 1)
                if index == 1:
                    check(fg_count >= 2, 'Second training example needs separate networks')
                    second_background_colors = {int(inp[y, x]) for (y, x), role in zip(positions, roles) if role == 0}
            else:
                check(training_pockets, 'Training did not demonstrate pocket filling')
    check(training_roles == {0, 1}, 'Training did not expose both roles')
    check(any(n >= 2 for n in training_fg_counts), 'Training did not demonstrate separate foreground networks')
    check(first_foreground_color in second_background_colors, 'Training did not demonstrate seed-color role reassignment')


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ('input', 'output')) for s in ('train', 'test'))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    sizes, layouts, region_counts, fg_counts = set(), set(), set(), set()
    endpoint = bend = repeats = new_test_colors = partial_dividers = 0
    within_region = within_shape = within_palette = 0
    multi_fg_multi_pocket = border_endpoint = 0
    fingerprints = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        pair_features = []
        fingerprints.add(tuple((p['input'].shape, p['input'].tobytes()) for s in ('train', 'test') for p in puzzle[s]))
        train_colors = {int(v) for p in puzzle['train'] for v in p['input'].flat if v >= 2}
        for section in ('train', 'test'):
            for p in puzzle[section]:
                inp = p['input']
                base, expected, comps, positions, roles = _valid_restorations(inp)[0]
                assert np.array_equal(expected, p['output'])
                sizes.add(inp.shape)
                layouts.add((base.shape, base.tobytes()))
                region_counts.add(len(comps))
                fg_counts.add(sum(role == 1 for role, _ in comps))
                pair_features.append((len(comps), inp.shape, tuple(sorted({int(inp[y, x]) for y, x in positions}))))
                repeats += len({int(inp[y, x]) for y, x in positions}) < len(positions)
                fg_number = sum(role == 1 for role, _ in comps)
                pocket_number = sum(role == 0 and not any(y in (0, inp.shape[0] - 1) or x in (0, inp.shape[1] - 1) for y, x in cells) for role, cells in comps)
                multi_fg_multi_pocket += fg_number > 1 and pocket_number > 1
                for (y, x), role in zip(positions, roles):
                    if role == 1:
                        endpoint += _seed_class(base, y, x) == 'endpoint'
                        bend += _seed_class(base, y, x) == 'bend'
                        border_endpoint += _seed_class(base, y, x) == 'endpoint' and (y in (0, inp.shape[0] - 1) or x in (0, inp.shape[1] - 1))
                partial_dividers += any(sum(base[y, :] == 1) >= base.shape[1] // 2 for y in range(base.shape[0]))
                if section == 'test':
                    new_test_colors += any(int(inp[y, x]) not in train_colors for y, x in positions)
        within_region += len({f[0] for f in pair_features}) > 1
        within_shape += len({f[1] for f in pair_features}) > 1
        within_palette += len({f[2] for f in pair_features}) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(fingerprints) == 42 and len(sizes) >= 30 and len(layouts) >= 100
    assert len(region_counts) >= 4 and len(fg_counts) >= 3
    assert endpoint > 10 and bend > 10 and repeats > 5 and new_test_colors > 5 and partial_dividers > 10, (endpoint, bend, repeats, new_test_colors, partial_dividers)
    assert within_region >= 35 and within_shape == 42 and within_palette == 42
    assert multi_fg_multi_pocket > 20 and border_endpoint > 5
    for nr, nt in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(71, nr, nt)
        validate_generated_puzzle(p)
        assert nr is None or len(p['train']) == nr
        assert nt is None or len(p['test']) == nt
    for nr, nt in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(79, nr, nt)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal count accepted')
    for args in ((19, None, None), (19, 7, 3), (19, 2, 1)):
        assert _puzzles_equal(generate_puzzle(*args), generate_puzzle(*args))
