"""Synthetic completion of a square pattern with all eight dihedral symmetries."""

import numpy as np


_SOURCE_INPUTS = (
    '2122655665562212166156522565166126165552255561622166622222266612655658577585655656528858833333655552555883333355622278888333332662227888833333265552555883333355565288588588256565565857758565562166622222266612261655522555616216615652256516612122655665562212',
    '8993333322778998989333332717998999833333727228998923333317229298772333337872227771727277772727172727872882787272227177822877172222717782287717222727872882787272717272777727271777222787787222778929227117229298998227277272289998997172271799898998772222778998',
    '2252999333332522254495233333445254549253333345452444595333334442999596933333599995296699996692599255997997995529595299966999259559529996699925959255997997995529952966999966925999959699996959992444595225954442545492555529454525449529925944522252999559992522',
    '5525255555525255522555222255522522585225522585225585525555255855255546693333355255226699333332555225696933333225525599993333352552559999333335255225696996965225552266999966225525554669966455525585525555255855225852255225852252255522225552255525255555525255',
)


def _key(n: int, row: int, col: int) -> tuple[int, int]:
    a = min(row, n - 1 - row)
    b = min(col, n - 1 - col)
    return (a, b) if a <= b else (b, a)


def _orbit_counts(n: int) -> dict[tuple[int, int], int]:
    counts = {}
    for row in range(n):
        for col in range(n):
            key = _key(n, row, col)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _placement(n: int, p: int, top: int, left: int, counts: dict[tuple[int, int], int]) -> tuple[bool, int]:
    hidden = {}
    for row in range(top, top + p):
        for col in range(left, left + p):
            key = _key(n, row, col)
            hidden[key] = hidden.get(key, 0) + 1
    exposure = min(counts[key] - count for key, count in hidden.items())
    return exposure > 0, exposure


def _position_pool(n: int, p: int) -> list[tuple[int, int, int]]:
    counts = _orbit_counts(n)
    return [
        (top, left, exposure)
        for top in range(n - p + 1)
        for left in range(n - p + 1)
        for valid, exposure in [_placement(n, p, top, left, counts)]
        if valid
    ]


def _axis(n: int, p: int, start: int) -> bool:
    if n % 2:
        return start <= n // 2 < start + p
    return start <= n // 2 - 1 and n // 2 < start + p


def _choose_position(
    rng: np.random.Generator, n: int, p: int, split: str,
    regime: int, index: int, used: set[tuple[int, int, int, int]],
) -> tuple[int, int]:
    pool = [(t, l, e) for t, l, e in _position_pool(n, p)
            if (n, p, t, l) not in used]
    if not pool:
        raise RuntimeError('no new recoverable position')

    def matches(item: tuple[int, int, int]) -> bool:
        t, l, e = item
        if regime == 0:  # Familiar top/right training, central test.
            if split == 'train' and index == 0:
                return t == 0
            if split == 'train' and index == 1:
                return l >= n // 2
            if split == 'test':
                return (_axis(n, p, t) or _axis(n, p, l)) and l >= n // 2
        elif regime == 1:  # More occlusion in training, move the test away.
            if split == 'train' and index == 0:
                return (_axis(n, p, t) or _axis(n, p, l)) and e <= 2
            if split == 'train' and index == 1:
                return l <= n // 3 or t >= n - p - 1
            if split == 'test':
                return l <= n // 3 or t >= n - p - 1
        elif regime == 2:
            if split == 'train' and index == 0:
                return t == n - p or l == 0
            if split == 'train' and index == 1:
                return 0 < t < n - p and 0 < l < n - p
            if split == 'test':
                return 0 < t < n - p and 0 < l < n - p
        else:
            if split == 'test':
                return e <= 3
        return True

    preferred = [item for item in pool if matches(item)]
    if preferred:
        pool = preferred
    choice = pool[int(rng.integers(len(pool)))]
    return choice[0], choice[1]


def _find_mask(grid: np.ndarray) -> tuple[int, int, int, int]:
    n = grid.shape[0]
    candidates = []
    for color in np.unique(grid):
        coordinates = np.argwhere(grid == color)
        area = len(coordinates)
        p = int(np.sqrt(area))
        if p < 3 or p > 7 or p * p != area or p >= n:
            continue
        top, left = coordinates.min(axis=0)
        bottom, right = coordinates.max(axis=0)
        if bottom - top + 1 == p and right - left + 1 == p:
            candidates.append((int(color), int(top), int(left), p))
    assert len(candidates) == 1, 'mask must be an identifiable solid square'
    return candidates[0]


def _recover(grid: np.ndarray) -> tuple[int, int, int, int, np.ndarray, np.ndarray, int]:
    n = grid.shape[0]
    color, top, left, p = _find_mask(grid)
    observed = {}
    visible_count = {}
    for row in range(n):
        for col in range(n):
            if top <= row < top + p and left <= col < left + p:
                continue
            key = _key(n, row, col)
            value = int(grid[row, col])
            if key in observed:
                assert observed[key] == value, 'visible cells break symmetry'
            else:
                observed[key] = value
            visible_count[key] = visible_count.get(key, 0) + 1
    assert len(observed) == len(_orbit_counts(n)), 'a full orbit is hidden'
    restored = np.empty_like(grid)
    for row in range(n):
        for col in range(n):
            restored[row, col] = observed[_key(n, row, col)]
    patch = restored[top:top + p, left:left + p].copy()
    assert not np.any(restored == color), 'mask color exists in the pattern'
    exposure = min(visible_count[_key(n, r, c)]
                   for r in range(top, top + p) for c in range(left, left + p))
    return color, top, left, p, patch, restored, exposure


def _pattern(rng: np.random.Generator, n: int, palette: tuple[int, ...]) -> np.ndarray:
    half = (n + 1) // 2
    keys = [(a, b) for a in range(half) for b in range(a, half)]
    style = int(rng.integers(6))
    values = {}
    if style == 1:  # Coarse blocks.
        block = int(rng.integers(2, 5))
        table = rng.choice(palette, size=(half // block + 1, half // block + 1))
    elif style == 2:  # Sparse accents on a broad plain region.
        base = int(rng.choice(palette))
        accents = rng.choice(len(keys), size=max(2, len(keys) // int(rng.integers(5, 10))), replace=False)
        accent_keys = {keys[int(i)] for i in accents}
    elif style == 3:  # Concentric bands.
        band_width = int(rng.integers(1, 3))
    elif style == 4:  # Repeated motif.
        period = int(rng.integers(2, 5))
        motif = rng.choice(palette, size=(period, period))
    for a, b in keys:
        if style == 0:  # Irregular orbit arrangement.
            value = int(rng.choice(palette))
        elif style == 1:
            value = int(table[a // block, b // block])
        elif style == 2:
            value = int(rng.choice([x for x in palette if x != base])) if (a, b) in accent_keys else base
        elif style == 3:  # Concentric bands, occasionally broken.
            value = int(palette[(a // band_width) % len(palette)])
        elif style == 4:
            value = int(motif[a % period, b % period])
        else:  # Structured diagonal and radial sectors.
            value = int(palette[((b - a) // 2 + a // 3) % len(palette)])
        values[(a, b)] = value
    if style in (1, 3, 4, 5):
        edits = int(rng.integers(1, max(3, len(keys) // 8)))
        for index in rng.choice(len(keys), size=edits, replace=False):
            key = keys[int(index)]
            values[key] = int(rng.choice([x for x in palette if x != values[key]]))
    grid = np.empty((n, n), dtype=np.int8)
    for row in range(n):
        for col in range(n):
            grid[row, col] = values[_key(n, row, col)]
    return grid


def _new_pair(
    rng: np.random.Generator, n: int, p: int, position: tuple[int, int],
    mask_color: int, palette: tuple[int, ...],
) -> dict:
    top, left = position
    for _ in range(100):
        underlying = _pattern(rng, n, palette)
        if len(np.unique(underlying)) < 2:
            continue
        patch = underlying[top:top + p, left:left + p].copy()
        if len(np.unique(patch)) < 2:
            continue
        masked = underlying.copy()
        masked[top:top + p, left:left + p] = mask_color
        if n == 16 and ''.join(str(int(v)) for v in masked.flat) in _SOURCE_INPUTS:
            continue
        try:
            found = _recover(masked)
        except AssertionError:
            continue
        if found[:4] == (mask_color, top, left, p) and np.array_equal(found[4], patch):
            return {'input': masked, 'output': patch}
    raise RuntimeError('could not construct an unambiguous pair')


def _check_count(value: int | None, name: str, low: int, high: int) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= value <= high):
        raise ValueError(f'{name} must be an integer from {low} to {high}')


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    _check_count(num_train, 'num_train', 2, 10)
    _check_count(num_test, 'num_test', 1, 4)
    if num_train is not None and num_test is not None and num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if num_test is None:
        num_test = int(rng.integers(1, min(4, (num_train or 10) - 1) + 1))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))

    regime = int(rng.integers(4))
    fixed_size = bool(rng.integers(2))
    fixed_mask = bool(rng.integers(2))
    shared_n = int(rng.integers(12, 25))
    shared_p = int(rng.integers(3, min(7, shared_n // 2) + 1))
    shared_mask = int(rng.integers(10))
    puzzle = {'train': [], 'test': []}
    used_inputs = set()
    used_placements = set()
    train_palettes = []
    train_masks = []
    train_patterns = set()
    first_size = None

    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            for _ in range(120):
                if fixed_size:
                    n, p = shared_n, shared_p
                else:
                    n = int(rng.integers(12, 25))
                    p = int(rng.integers(3, min(7, n // 2) + 1))
                    if split == 'train' and index == 1 and (n, p) == first_size:
                        continue
                if split == 'train' and index == 0:
                    first_size = (n, p)
                position = _choose_position(rng, n, p, split, regime, index, used_placements)
                if split == 'test' and rng.random() < 0.5:
                    palette = train_palettes[int(rng.integers(len(train_palettes)))]
                    if fixed_mask and shared_mask in palette:
                        continue
                else:
                    palette = None
                if fixed_mask:
                    mask_color = shared_mask
                elif split == 'train' and index == 1:
                    mask_color = int(rng.choice([c for c in range(10) if c != train_masks[0] and (palette is None or c not in palette)]))
                else:
                    mask_color = int(rng.choice([c for c in range(10) if palette is None or c not in palette]))
                if palette is None:
                    choices = [c for c in range(10) if c != mask_color]
                    size = int(rng.integers(2, 8))
                    palette = tuple(sorted(int(c) for c in rng.choice(choices, size=size, replace=False)))
                pair = _new_pair(rng, n, p, position, mask_color, palette)
                signature = (n, pair['input'].tobytes())
                if signature in used_inputs:
                    continue
                recovered = _recover(pair['input'])
                pattern = recovered[5]
                pattern_signature = (n, pattern.tobytes())
                if split == 'train' and pattern_signature in train_patterns:
                    continue
                used_inputs.add(signature)
                used_placements.add((n, p, *position))
                puzzle[split].append(pair)
                if split == 'train':
                    train_masks.append(mask_color)
                    train_palettes.append(tuple(int(v) for v in np.unique(pattern)))
                    train_patterns.add(pattern_signature)
                break
            else:
                raise RuntimeError('could not create a distinct puzzle pair')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    train, test = puzzle['train'], puzzle['test']
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen_inputs = set()
    seen_pairs = set()
    train_masks = []
    train_positions = set()
    train_patterns = set()
    for split, pairs in (('train', train), ('test', test)):
        for pair in pairs:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = pair['input'], pair['output']
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= dimension <= 30 for dimension in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            n = inp.shape[0]
            assert inp.shape == (n, n) and 12 <= n <= 24
            color, top, left, p, recovered, restored, exposure = _recover(inp)
            assert 3 <= p <= 7 and p <= n // 2
            assert out.shape == (p, p) and np.array_equal(out, recovered)
            assert len(np.unique(restored)) >= 2 and len(np.unique(out)) >= 2
            assert exposure >= 1
            signature = (inp.shape, inp.tobytes())
            pair_signature = (signature, out.shape, out.tobytes())
            assert signature not in seen_inputs and pair_signature not in seen_pairs
            seen_inputs.add(signature)
            seen_pairs.add(pair_signature)
            if n == 16:
                assert ''.join(str(int(v)) for v in inp.flat) not in _SOURCE_INPUTS
            if split == 'train':
                train_masks.append(color)
                train_positions.add((n, p, top, left))
                train_patterns.add((n, restored.tobytes()))
            else:
                assert (n, p, top, left) not in train_positions
                if len(set(train_masks)) == 1:
                    assert color == train_masks[0]
    assert len(train_positions) >= 2 and len(train_patterns) >= 2


def _same(a: dict, b: dict) -> bool:
    return all(len(a[split]) == len(b[split]) and all(
        np.array_equal(x[role], y[role])
        for x, y in zip(a[split], b[split]) for role in ('input', 'output'))
        for split in ('train', 'test'))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    canvas_sizes, patch_sizes, positions = set(), set(), set()
    masks, palette_sizes = set(), set()
    fixed_size_count = varied_size_count = 0
    fixed_mask_count = varied_mask_count = 0
    palette_reuse = palette_shift = 0
    high_occlusion_train = 0
    bottom = left = corner = interior = axis_train = 0
    dominance = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        if seed < 5:
            assert _same(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        sizes = set()
        puzzle_masks = set()
        train_palettes = set()
        test_palettes = []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                n = inp.shape[0]
                color, top, col, p, recovered, restored, exposure = _recover(inp)
                assert np.array_equal(out, recovered)
                for row in range(top, top + p):
                    for column in range(col, col + p):
                        mates = [int(inp[r, c]) for r in range(n) for c in range(n)
                                 if _key(n, r, c) == _key(n, row, column)
                                 and not (top <= r < top + p and col <= c < col + p)]
                        assert mates and set(mates) == {int(out[row - top, column - col])}
                sizes.add((n, p))
                canvas_sizes.add(n)
                patch_sizes.add(p)
                positions.add((n, p, top, col))
                masks.add(color)
                puzzle_masks.add(color)
                palette = tuple(int(v) for v in np.unique(restored))
                palette_sizes.add(len(palette))
                counts = np.unique(restored, return_counts=True)[1]
                dominance.append(float(max(counts) / (n * n)))
                if split == 'train':
                    train_palettes.add(palette)
                    high_occlusion_train += exposure <= 2
                    bottom += top == n - p
                    left += col == 0
                    corner += top in (0, n - p) and col in (0, n - p)
                    interior += 0 < top < n - p and 0 < col < n - p
                    axis_train += _axis(n, p, top) or _axis(n, p, col)
                else:
                    test_palettes.append(palette)
        fixed_size_count += len(sizes) == 1
        varied_size_count += len(sizes) > 1
        fixed_mask_count += len(puzzle_masks) == 1
        varied_mask_count += len(puzzle_masks) > 1
        palette_reuse += sum(p in train_palettes for p in test_palettes)
        palette_shift += sum(p not in train_palettes for p in test_palettes)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(canvas_sizes) >= 9 and len(patch_sizes) == 5
    assert len(positions) >= 130 and len(masks) >= 8
    assert {2, 3} <= palette_sizes and max(palette_sizes) >= 6
    assert fixed_size_count >= 5 and varied_size_count >= 5
    assert fixed_mask_count >= 5 and varied_mask_count >= 5
    assert palette_reuse >= 5 and palette_shift >= 5
    assert high_occlusion_train >= 1
    assert min(bottom, left, corner, interior, axis_train) >= 1
    assert min(dominance) < 0.3 and max(dominance) > 0.55
    for a, b in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        seed = 919 + (a or 0) * 10 + (b or 0)
        puzzle = generate_puzzle(seed, a, b)
        validate_generated_puzzle(puzzle)
        assert (a is None or len(puzzle['train']) == a) and (b is None or len(puzzle['test']) == b)
        assert _same(puzzle, generate_puzzle(seed, a, b))
    for a, b in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(777, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal pair counts were accepted')
