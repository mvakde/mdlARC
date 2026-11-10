import numpy as np


_SOURCE_PAIRS = (
    (
        (
            "8727872787",
            "8727877787",
            "8727872787",
            "8777877787",
            "8777877787",
            "8888888888",
        ),
        (
            "8777877787",
            "8727877787",
            "8727872787",
            "8727872787",
            "8777877787",
            "8888888888",
        ),
    ),
    (
        (
            "766776667",
            "776777677",
            "776777777",
            "277727772",
        ),
        (
            "766777777",
            "776776667",
            "776777677",
            "277727772",
        ),
    ),
    (
        (
            "7444777771",
            "7474744471",
            "7444744471",
            "7777774771",
            "7777774771",
            "7777777711",
            "7777777711",
            "7777711111",
            "1111111111",
        ),
        (
            "7777777771",
            "7777777771",
            "7777744471",
            "7777744471",
            "7444774771",
            "7474774711",
            "7444777711",
            "7777711111",
            "1111111111",
        ),
    ),
    (
        (
            "7777977",
            "7777777",
            "5557775",
            "5557775",
            "5555555",
            "5555555",
        ),
        (
            "7777777",
            "7777777",
            "5557975",
            "5557775",
            "5555555",
            "5555555",
        ),
    ),
    (
        (
            "77777777777999997777",
            "77999799977979797777",
            "77979797977999997777",
            "77979797977979797777",
            "77979997977999997777",
            "77977777777777777777",
            "77977777777777777777",
            "77777777777777777777",
            "77777777777171717171",
            "11111111117111111111",
        ),
        (
            "77777777777777777777",
            "77777777777777777777",
            "77999799977999997777",
            "77979797977979797777",
            "77979797977999997777",
            "77979997977979797777",
            "77977777777999997777",
            "77977777777777777777",
            "77777777777171717171",
            "11111111117111111111",
        ),
    ),
)


def _checked_count(
    value: int | None,
    name: str,
    low: int,
    high: int,
) -> int | None:
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")

    return value


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _checked_count(num_train, "num_train", 2, 10)
    num_test = _checked_count(num_test, "num_test", 1, 4)
    code = int(seed) & 0xFFFFFFFF

    if num_train is None and num_test is None:
        num_test = 1 + code % 4
        low = max(2, num_test + 1)
        num_train = low + ((code // 4 + 3 * code) % (11 - low))
    elif num_train is None:
        low = max(2, num_test + 1)
        num_train = low + ((5 * code + 1) % (11 - low))
    elif num_test is None:
        high = min(4, num_train - 1)
        num_test = 1 + ((7 * code + 2) % high)

    if not (
        2 <= num_train <= 10
        and 1 <= num_test <= 4
        and num_train > num_test
    ):
        raise ValueError(
            "pair counts must be legal and num_train must exceed num_test"
        )

    return num_train, num_test


def _sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    return {
        "background": int(rng.integers(0, 10)),
    }


def _sample_pair_colors(rng: np.random.Generator, background: int,
                        used: set[int] | None = None) -> tuple[int, int]:
    choices = [c for c in range(10) if c != background and (used is None or c not in used)]
    colors = rng.choice(choices, size=2, replace=False)
    return int(colors[0]), int(colors[1])


def _components(mask: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            seen[row, col] = True
            stack = [(row, col)]
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            result.append(tuple(cells))
    return result


def _bbox(cells: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    return (min(r for r, _ in cells), max(r for r, _ in cells),
            min(c for _, c in cells), max(c for _, c in cells))


def _forbidden_cells(scaffold: np.ndarray) -> np.ndarray:
    h, w = scaffold.shape
    forbidden = scaffold.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            sr0, sr1 = max(0, -dr), min(h, h - dr)
            sc0, sc1 = max(0, -dc), min(w, w - dc)
            forbidden[sr0 + dr:sr1 + dr, sc0 + dc:sc1 + dc] |= scaffold[sr0:sr1, sc0:sc1]
    forbidden[-1, :] = True
    return forbidden


def _settle(components: list[tuple[tuple[int, int], ...]],
            forbidden: np.ndarray) -> tuple[list[set[tuple[int, int]]], list[int]]:
    h = forbidden.shape[0]
    for i, first in enumerate(components):
        for second in components[i + 1:]:
            if max(r for r, _ in first) == max(r for r, _ in second):
                assert {c for _, c in first}.isdisjoint(c for _, c in second)
    ordered = sorted(enumerate(components),
                     key=lambda item: (-max(r for r, _ in item[1]),
                                       min(c for _, c in item[1])))
    occupied: set[tuple[int, int]] = set()
    placed: list[set[tuple[int, int]]] = [set() for _ in components]
    falls = [0 for _ in components]
    for index, component in ordered:
        cells = set(component)
        assert all(not forbidden[r, c] for r, c in cells)
        assert not (cells & occupied)
        fall = 0
        while all(r + fall + 1 < h and not forbidden[r + fall + 1, c]
                  and (r + fall + 1, c) not in occupied for r, c in cells):
            fall += 1
        shifted = {(r + fall, c) for r, c in cells}
        placed[index] = shifted
        falls[index] = fall
        occupied.update(shifted)
    return placed, falls


def _solve_grid(input_grid: np.ndarray, background: int,
                scaffold_color: int, payload_color: int) -> np.ndarray:
    scaffold = input_grid == scaffold_color
    components = _components(input_grid == payload_color)
    placed, _ = _settle(components, _forbidden_cells(scaffold))
    output = np.full(input_grid.shape, background, dtype=np.int8)
    output[scaffold] = scaffold_color
    for cells in placed:
        for r, c in cells:
            output[r, c] = payload_color
    return output


def _dimensions(rng: np.random.Generator, regime: str) -> tuple[int, int]:
    compact = rng.random() < (0.72 if regime == 'source_train' else 0.42)
    if compact:
        if regime == 'source_train':
            return int(rng.integers(6, 14)), int(rng.integers(9, 18))
        return int(rng.integers(8, 14)), int(rng.integers(9, 21))
    if regime == 'source_train':
        return int(rng.integers(13, 20)), int(rng.integers(14, 27))
    return int(rng.integers(13, 23)), int(rng.integers(15, 29))


def _scaffold(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mode = int(rng.integers(0, 4))
    if mode == 0:  # compartment walls; their heights vary independently
        mask[h - int(rng.integers(1, 3)):, :] = True
        possible = list(range(2, w - 2))
        rng.shuffle(possible)
        walls = []
        for col in possible:
            if all(abs(col - old) >= 4 for old in walls):
                walls.append(col)
            if len(walls) >= min(3, max(1, w // 6)):
                break
        for col in walls:
            top = 0 if rng.random() < 0.5 else int(rng.integers(1, max(2, h // 2)))
            thickness = 1 if rng.random() < 0.8 else 2
            mask[top:, col:min(w, col + thickness)] = True
        for col in (0, w - 1):
            if rng.random() < 0.5:
                top = 0 if rng.random() < 0.5 else int(rng.integers(1, h // 2 + 1))
                mask[top:, col] = True
    elif mode == 1:  # disconnected posts, including possible thick posts
        columns = {0, w - 1}
        target = int(rng.integers(3, min(7, w // 2 + 1)))
        while len(columns) < target:
            columns.add(int(rng.integers(1, w - 1)))
        for col in columns:
            post_h = int(rng.integers(1, min(4, h - 3)))
            thickness = 1 if rng.random() < 0.75 else 2
            mask[h - post_h:, col:min(w, col + thickness)] = True
    elif mode == 2:  # stepped solid foundation with local rises and notches
        base = int(rng.integers(1, 4))
        mask[h - base:, :] = True
        col = 0
        while col < w:
            run = int(rng.integers(2, 6))
            rise = int(rng.integers(0, min(5, h - base - 2)))
            if rise:
                mask[h - base - rise:h - base, col:min(w, col + run)] = True
            col += run
        if w >= 10 and rng.random() < 0.7:
            start = int(rng.integers(2, w - 3))
            notch_w = int(rng.integers(1, min(4, w - start)))
            top = max(2, h - base - int(rng.integers(0, 3)))
            mask[top:h - 1, start:start + notch_w] = False
    else:  # separated slabs, variable gap, heights, thickness, and teeth
        gap = int(rng.integers(1, min(4, w - 6)))
        split = int(rng.integers(3, w - gap - 2))
        spans = [(0, split), (split + gap, w)]
        for left, right in spans:
            base = int(rng.integers(1, min(6, h - 3)))
            mask[h - base:, left:right] = True
            if rng.random() < 0.8:
                spacing = int(rng.integers(2, 5))
                offset = int(rng.integers(0, spacing))
                for col in range(left + offset, right, spacing):
                    if h - base - 1 >= 2:
                        mask[h - base - 1, col] = True
            if right - left >= 5 and rng.random() < 0.35:
                notch = int(rng.integers(left + 1, right - 1))
                mask[h - base:h - 1, notch] = False
    return mask


def _shape(rng: np.random.Generator, complex_bias: bool) -> tuple[tuple[int, int], ...]:
    kinds = ('singleton', 'branch', 'frame', 'cap', 'multi', 'comb', 'poly', 'solid')
    weights = (2, 3, 3, 3, 2, 2, 3, 2) if not complex_bias else (0, 1, 4, 3, 4, 2, 1, 2)
    kind = str(rng.choice(kinds, p=np.array(weights) / sum(weights)))
    w = int(rng.integers(3, 9))
    h = int(rng.integers(3, 7))
    cells: set[tuple[int, int]] = set()
    if kind == 'singleton':
        cells.add((0, 0))
    elif kind == 'branch':
        length = int(rng.integers(3, 8))
        vertical = rng.random() < 0.35
        cells.update((r, 1) if vertical else (1, r) for r in range(length))
        for _ in range(int(rng.integers(1, 4))):
            along = int(rng.integers(0, length))
            limb = int(rng.integers(1, 4))
            side = -1 if rng.random() < 0.5 else 1
            for step in range(1, limb + 1):
                cells.add((along, 1 + side * step) if vertical else (1 + side * step, along))
    elif kind == 'frame':
        cells.update((r, c) for r in range(h) for c in range(w)
                     if r in (0, h - 1) or c in (0, w - 1))
        if rng.random() < 0.45 and w >= 5:
            divider = int(rng.integers(2, w - 2))
            cells.update((r, divider) for r in range(h))
    elif kind == 'cap':
        cap_h = int(rng.integers(1, 3))
        cells.update((r, c) for r in range(cap_h) for c in range(w))
        stem_col = int(rng.integers(0, w))
        cells.update((r, stem_col) for r in range(cap_h, cap_h + int(rng.integers(1, 4))))
        if rng.random() < 0.4:
            cells.update((cap_h - 1, c) for c in range(stem_col, min(w, stem_col + 2)))
    elif kind == 'multi':
        w = max(5, w)
        h = max(4, h)
        columns = {0, w - 1}
        columns.update(int(c) for c in rng.choice(np.arange(2, w - 2),
                      size=min(2, w - 4), replace=False))
        cells.update((r, c) for r in range(h) for c in range(w)
                     if c in columns or r in (0, h - 1))
        if h >= 5 and rng.random() < 0.6:
            row = int(rng.integers(2, h - 1))
            start = int(rng.integers(0, w - 2))
            cells.update((row, c) for c in range(start, w))
    elif kind == 'comb':
        cells.update((0, c) for c in range(w))
        prongs = {0, w - 1}
        prongs.update(int(c) for c in rng.choice(np.arange(1, w - 1),
                      size=int(rng.integers(1, min(4, w - 1))), replace=False))
        for col in prongs:
            length = int(rng.integers(2, h + 1))
            cells.update((r, col) for r in range(1, length))
        if rng.random() < 0.45:
            cells.update((1, c) for c in range(w))
    elif kind == 'poly':
        cells.add((0, 0))
        for _ in range(int(rng.integers(4, 13))):
            boundary = [(r + dr, c + dc) for r, c in cells
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                        if 0 <= r + dr < h and 0 <= c + dc < w and
                        (r + dr, c + dc) not in cells]
            if boundary:
                cells.add(boundary[int(rng.integers(0, len(boundary)))])
    else:
        cells.update((r, c) for r in range(h) for c in range(w))
        corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
        rng.shuffle(corners)
        for corner in corners[:int(rng.integers(1, 4))]:
            cells.remove(corner)
    if rng.random() < 0.5:
        max_col = max(c for _, c in cells)
        cells = {(r, max_col - c) for r, c in cells}
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _placements(shape: tuple[tuple[int, int], ...], forbidden: np.ndarray,
                existing: set[tuple[int, int]]) -> list[tuple[int, int, int]]:
    h, w = forbidden.shape
    sh = max(r for r, _ in shape) + 1
    sw = max(c for _, c in shape) + 1
    options = []
    for top in range(h - sh + 1):
        for left in range(w - sw + 1):
            cells = {(top + r, left + c) for r, c in shape}
            if any(forbidden[r, c] or (r, c) in existing or
                   (r + 1, c) in existing or (r - 1, c) in existing or
                   (r, c + 1) in existing or (r, c - 1) in existing
                   for r, c in cells):
                continue
            fall = 0
            while all(r + fall + 1 < h and not forbidden[r + fall + 1, c]
                      for r, c in cells):
                fall += 1
            options.append((top, left, fall))
    return options


def _compose(scaffold: np.ndarray, components: list[tuple[tuple[int, int], ...]],
             colors: tuple[int, int, int]) -> tuple[dict, list[int]]:
    background, scaffold_color, payload_color = colors
    input_grid = np.full(scaffold.shape, background, dtype=np.int8)
    input_grid[scaffold] = scaffold_color
    for component in components:
        for r, c in component:
            assert input_grid[r, c] == background
            input_grid[r, c] = payload_color
    placed, falls = _settle(components, _forbidden_cells(scaffold))
    output = np.full(scaffold.shape, background, dtype=np.int8)
    output[scaffold] = scaffold_color
    for cells in placed:
        for r, c in cells:
            output[r, c] = payload_color
    return {'input': input_grid, 'output': output}, falls


def _generate_stack(rng: np.random.Generator, colors: tuple[int, int, int]) -> dict:
    for _ in range(80):
        h, w = _dimensions(rng, 'source_train')
        h = max(9, h)
        scaffold = _scaffold(rng, h, w)
        forbidden = _forbidden_cells(scaffold)
        candidates = []
        for col in range(w):
            free = [r for r in range(h) if not forbidden[r, col]]
            if len(free) < 5:
                continue
            bottom = max(free)
            for count in (3, 2):
                rows = [bottom - 2 * k for k in range(count)]
                if min(rows) >= 0 and all(not forbidden[r, col] for r in rows):
                    candidates.append((col, rows))
        if not candidates:
            continue
        col, rows = candidates[int(rng.integers(0, len(candidates)))]
        components = [((r, col),) for r in rows]
        pair, falls = _compose(scaffold, components, colors)
        if any(falls) and len(_components(pair['output'] == colors[2])) < len(components):
            return pair
    raise AssertionError('stack construction failed')


def _generate_regular(rng: np.random.Generator, colors: tuple[int, int, int],
                      regime: str, mixed: bool) -> dict:
    for _ in range(100):
        h, w = _dimensions(rng, regime)
        scaffold = _scaffold(rng, h, w)
        forbidden = _forbidden_cells(scaffold)
        n = 2 if mixed else int(rng.integers(1 if regime == 'source_train' else 2, 5))
        components = []
        existing: set[tuple[int, int]] = set()
        for index in range(n):
            desired = 'stationary' if mixed and index == 0 else 'moving' if mixed or index == 0 else str(rng.choice(('stationary', 'short', 'long', 'any'), p=(.17, .35, .3, .18)))
            options = []
            shape = None
            for _ in range(12):
                complex_bias = (mixed and index == 0) or (regime == 'target_test' and rng.random() < 0.65) or (rng.random() < 0.35)
                candidate_shape = _shape(rng, complex_bias)
                candidate_options = _placements(candidate_shape, forbidden, existing)
                if desired == 'stationary':
                    candidate_options = [o for o in candidate_options if o[2] == 0]
                elif desired == 'moving':
                    candidate_options = [o for o in candidate_options if o[2] > 0]
                elif desired == 'short':
                    candidate_options = [o for o in candidate_options if 1 <= o[2] <= 2]
                elif desired == 'long':
                    candidate_options = [o for o in candidate_options if o[2] >= 3]
                if candidate_options:
                    shape, options = candidate_shape, candidate_options
                    break
            if not options:
                break
            top, left, _ = options[int(rng.integers(0, len(options)))]
            absolute = tuple((top + r, left + c) for r, c in shape)
            components.append(absolute)
            existing.update(absolute)
        if len(components) != n:
            continue
        pair, falls = _compose(scaffold, components, colors)
        if not any(falls):
            continue
        if mixed and not (0 in falls and any(f > 0 for f in falls)):
            continue
        if len(_components(pair['input'] == colors[2])) != n:
            continue
        return pair
    raise AssertionError('regular construction failed')


def _input_key(grid: np.ndarray) -> tuple:
    return grid.shape, grid.tobytes()


def _pair_key(pair: dict) -> tuple:
    return _input_key(pair['input']), _input_key(pair['output'])


def _rows(grid: np.ndarray) -> tuple[str, ...]:
    return tuple(''.join(str(int(v)) for v in row) for row in grid)


def _is_source_pair(pair: dict) -> bool:
    return any((_rows(pair['input']), _rows(pair['output'])) == source
               for source in _SOURCE_PAIRS)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    params = _sample_puzzle_parameters(rng)
    background = params['background']
    colors = []
    first = _sample_pair_colors(rng, background)
    second = _sample_pair_colors(rng, background, set(first))
    colors.extend((first, second))
    for _ in range(num_train + num_test - 2):
        colors.append(_sample_pair_colors(rng, background))
    puzzle = {'train': [], 'test': []}
    seen = set()
    train_modes = ['stack', 'mixed'] + ['regular'] * (num_train - 2)
    rng.shuffle(train_modes)
    for regime, count in (('source_train', num_train), ('target_test', num_test)):
        for index in range(count):
            color_index = index if regime == 'source_train' else num_train + index
            pair_colors = (background, *colors[color_index])
            for _ in range(25):
                pair_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                pair_rng = np.random.default_rng(pair_seed)
                try:
                    if regime == 'source_train' and train_modes[index] == 'stack':
                        pair = _generate_stack(pair_rng, pair_colors)
                    else:
                        pair = _generate_regular(pair_rng, pair_colors, regime,
                                                 regime == 'source_train' and train_modes[index] == 'mixed')
                except AssertionError:
                    continue
                key = _input_key(pair['input'])
                if key in seen or _is_source_pair(pair):
                    continue
                seen.add(key)
                puzzle['train' if regime == 'source_train' else 'test'].append(pair)
                break
            else:
                raise ValueError('could not generate a unique valid pair')
    validate_generated_puzzle(puzzle)
    return puzzle


def _infer_background_from_training(train: list[dict]) -> int:
    intersection = set(int(v) for v in np.unique(train[0]['input']))
    for pair in train[1:]:
        intersection &= set(int(v) for v in np.unique(pair['input']))
    assert len(intersection) == 1
    return next(iter(intersection))


def _roles_from_input(input_grid: np.ndarray, background: int) -> tuple[int, int]:
    palette = set(int(v) for v in np.unique(input_grid))
    assert len(palette) == 3 and background in palette
    others = palette - {background}
    scaffold = [c for c in others if np.any(input_grid[-1] == c)]
    assert len(scaffold) == 1
    return scaffold[0], next(iter(others - set(scaffold)))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    input_keys = set()
    pair_keys = set()
    for pair in puzzle['train'] + puzzle['test']:
        assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
        for grid in (pair['input'], pair['output']):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
            assert np.issubdtype(grid.dtype, np.integer)
            assert np.all((0 <= grid) & (grid <= 9))
        assert pair['input'].shape == pair['output'].shape
        assert _input_key(pair['input']) not in input_keys
        assert _pair_key(pair) not in pair_keys
        assert not _is_source_pair(pair)
        input_keys.add(_input_key(pair['input']))
        pair_keys.add(_pair_key(pair))
    background = _infer_background_from_training(puzzle['train'])
    stacking = False
    mixed_motion = False
    for regime in ('train', 'test'):
        for pair in puzzle[regime]:
            input_grid, output_grid = pair['input'], pair['output']
            assert np.array_equal(np.unique(input_grid), np.unique(output_grid))
            scaffold_color, payload_color = _roles_from_input(input_grid, background)
            scaffold = input_grid == scaffold_color
            payload = input_grid == payload_color
            assert scaffold.any() and payload.any()
            assert np.array_equal(scaffold, output_grid == scaffold_color)
            assert np.count_nonzero(payload) == np.count_nonzero(output_grid == payload_color)
            forbidden = _forbidden_cells(scaffold)
            assert not np.any(forbidden & payload)
            assert not np.any(forbidden & (output_grid == payload_color))
            components = _components(payload)
            placed, falls = _settle(components, forbidden)
            expected = np.full(input_grid.shape, background, dtype=np.int8)
            expected[scaffold] = scaffold_color
            for cells in placed:
                for r, c in cells:
                    expected[r, c] = payload_color
            assert np.array_equal(expected, output_grid)
            assert any(falls)
            if regime == 'train':
                stacking |= len(_components(output_grid == payload_color)) < len(components)
                mixed_motion |= len(falls) >= 2 and 0 in falls and any(f > 0 for f in falls)
            else:
                assert len(components) >= 2
    assert stacking and mixed_motion


def _puzzles_equal(a: dict, b: dict) -> bool:
    return (len(a['train']) == len(b['train']) and len(a['test']) == len(b['test'])
            and all(np.array_equal(x[k], y[k]) for regime in ('train', 'test')
                    for x, y in zip(a[regime], b[regime]) for k in ('input', 'output')))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    assert all(not _puzzles_equal(puzzles[0], p) for p in puzzles[1:])
    assert _puzzles_equal(generate_puzzle(12345, 7, 3), generate_puzzle(12345, 7, 3))
    scaffold_signatures = set()
    shape_signatures = set()
    sizes = set()
    backgrounds = set()
    motion_classes = set()
    component_counts = set()
    scaffold_traits = set()
    payload_traits = set()
    pendant_sides = set()
    pendant_lengths = set()
    test_component_counts_by_index: dict[int, set[int]] = {}
    compact_tests = 0
    large_training = 0
    combined_scenes = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        bg = _infer_background_from_training(puzzle['train'])
        backgrounds.add(bg)
        for regime in ('train', 'test'):
            for pair_index, pair in enumerate(puzzle[regime]):
                grid = pair['input']
                scaffold_color, payload_color = _roles_from_input(grid, bg)
                scaffold = grid == scaffold_color
                components = _components(grid == payload_color)
                _, falls = _settle(components, _forbidden_cells(scaffold))
                scaffold_signatures.add((grid.shape, scaffold.tobytes()))
                sizes.add((regime, grid.shape))
                component_counts.add((regime, len(components)))
                motion_classes.update('stationary' if f == 0 else 'short' if f <= 2 else 'long' for f in falls)
                if regime == 'test':
                    test_component_counts_by_index.setdefault(pair_index, set()).add(len(components))
                    compact_tests += int(grid.shape[0] <= 12 and grid.shape[1] <= 20)
                else:
                    large_training += int(grid.shape[0] >= 15 and grid.shape[1] >= 18)
                bottom = scaffold[-1]
                base_runs = sum(bool(bottom[c] and (c == 0 or not bottom[c - 1]))
                                for c in range(grid.shape[1]))
                tops = [int(np.flatnonzero(scaffold[:, c])[0])
                        for c in range(grid.shape[1]) if scaffold[:, c].any()]
                if len(_components(scaffold)) > 1:
                    scaffold_traits.add('disconnected')
                if base_runs > 1:
                    scaffold_traits.add('separated_bases')
                if len(set(tops)) > 1:
                    scaffold_traits.add('stepped')
                if any(scaffold[:, c].all() for c in range(1, grid.shape[1] - 1)):
                    scaffold_traits.add('full_height_wall')
                if any(tops[i] < tops[i - 1] and tops[i] < tops[i + 1]
                       for i in range(1, len(tops) - 1)):
                    scaffold_traits.add('tooth')
                combined_scenes += int(base_runs > 1 and len(components) >= 2
                                       and max(falls) >= 3)
                for component in components:
                    r0, _, c0, _ = _bbox(component)
                    normalized = tuple(sorted((r - r0, c - c0) for r, c in component))
                    shape_signatures.add(normalized)
                    cells = set(normalized)
                    sh = max(r for r, _ in cells) + 1
                    sw = max(c for _, c in cells) + 1
                    if len(cells) == 1:
                        payload_traits.add('singleton')
                    if any(sum((r + dr, c + dc) in cells for dr, dc in
                               ((1, 0), (-1, 0), (0, 1), (0, -1))) >= 3
                           for r, c in cells):
                        payload_traits.add('branched')
                    empty = {(r, c) for r in range(sh) for c in range(sw)} - cells
                    while empty:
                        start = empty.pop()
                        region = {start}
                        frontier = [start]
                        while frontier:
                            r, c = frontier.pop()
                            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                                neighbour = (r + dr, c + dc)
                                if neighbour in empty:
                                    empty.remove(neighbour)
                                    region.add(neighbour)
                                    frontier.append(neighbour)
                        if all(0 < r < sh - 1 and 0 < c < sw - 1 for r, c in region):
                            payload_traits.add('hole')
                    stem = 0
                    tip = None
                    for r in range(sh - 1, -1, -1):
                        row_cols = [c for rr, c in cells if rr == r]
                        if len(row_cols) != 1:
                            break
                        stem += 1
                        tip = row_cols[0]
                    if 0 < stem < sh and sw >= 3:
                        pendant_lengths.add(stem)
                        pendant_sides.add('left' if tip == 0 else
                                         'right' if tip == sw - 1 else 'middle')
    assert len(backgrounds) >= 6
    assert len(scaffold_signatures) >= 120
    assert len(shape_signatures) >= 50
    assert len(sizes) >= 40
    assert len(component_counts) >= 5
    assert motion_classes == {'stationary', 'short', 'long'}
    assert scaffold_traits == {'disconnected', 'separated_bases', 'stepped',
                               'full_height_wall', 'tooth'}
    assert payload_traits == {'singleton', 'branched', 'hole'}
    assert pendant_sides == {'left', 'right', 'middle'}
    assert len(pendant_lengths) >= 3
    assert compact_tests >= 5 and large_training >= 5
    assert combined_scenes >= 5
    assert test_component_counts_by_index[0] == {2, 3, 4}
    for seed, nt, ns in ((101, 2, 1), (102, 10, 1), (103, 10, 4),
                         (104, None, 1), (105, None, 4), (106, 2, None),
                         (107, 10, None)):
        puzzle = generate_puzzle(seed, nt, ns)
        validate_generated_puzzle(puzzle)
        if nt is not None:
            assert len(puzzle['train']) == nt
        if ns is not None:
            assert len(puzzle['test']) == ns
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(999, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts did not raise ValueError')
