"""Synthetic puzzles for rightward traces and obstacle-triggered bands."""

import hashlib
import numpy as np


# Source inputs are kept only for exact-copy rejection and source regressions.
# Compact row strings make the self-test independent of repository files.
_SOURCE_INPUTS = (
    '00080800080800000000/80000008000000000808/08080000800800800000/00008008000008000000/00000800080000008080/00000000808000800000/10000800000800000000/08000000080000000080/00080000800080080000/00000000080000000000/80000800000800008008/10800000800000000080/00000000000080808000/00000000800800000080/00000080000000008000/10808000008000080000/00080000000000000080/00000008000008000000/00080000080080800080/00800008000000008000',
    '0050000000/0000050500/0000000050/1050000000/0005000050/0000000000/0050005000/0000000000/1000500000/0500000050',
    '000000/005000/000000/100500/000000/000000',
    '000000/000000/108000/080000/000080/000000',
    '5000005000000/0505000000000/0000000000050/0000500005000/1050000050005/0000050005000/0505000000000/0000000500000/0000050000000/5005005050050/0000500005000/1000000000050/0000505005000',
    '0008000/0000000/0000000/1080800/0000000/0080000/0000000',
    '00800000008000000000800000/80008080000000000000080000/00800000008000000008000000/00000000000080000000000000/00000000000000000880000000/10000808008008000000080000/00008000000000000000000000/00000008000000000000000008/08000080000080000080000000/00000000000000000000000080/10000000000808000000080000/00000800080000000000008000/08000000800000000000080008/00080000000800000008000000/80000080000080800000000800/08000000000800080008000000/00000000000000000000000080/10000800008080080080000000/80000000000000000000800000/00008000800000000000000008/08000000000800000000080000/00000008000000800000800800/80000000800800080800000000/00008000000008000008000000/00000800000000000000000000/00080000800000008080000080',
)
_SOURCE_OUTPUT_DIGESTS = (
    '2e1f2eca14c7ac80d6a409fc4b3674b7dff6539b6d99987d5e44d71f12ddc892',
    'cdfad0ca6e2452b380aff7811d03e985908eddc3b967db45091e8abeb4ca231c',
    'af7f1b42952e91aa2c81b586d55b100cd504b3c745957138ca5316ca1a704441',
    'a72fb502bf0488066ed5dc58d8c970570574546188f31527df61358fbc541016',
    '1c53ddf13058a26ba829627c40cddde64a0b3b063ef4d3e54c56c51c3f8103df',
    '321c835808555cc01adc5e0e6d649b97e8689d1c98fbc5fa4937d795bc9b8a38',
    '17b63b8c4ce089ef67c4dc6f35a52c6157b64aaf1a67b8991b6ba689bf4e1b78',
)


def _source_grids() -> tuple[np.ndarray, ...]:
    return tuple(
        np.array([[int(ch) for ch in row] for row in text.split('/')], dtype=np.int8)
        for text in _SOURCE_INPUTS
    )


def _blocked_contact(grid: np.ndarray, row: int, col: int, marker: int) -> bool:
    """A diagonal fixed-pixel pair on one side stops that row's paint."""
    height, width = grid.shape
    if row == 0 or row == height - 1:
        return False
    return (
        (col > 0 and grid[row - 1, col - 1] == marker
         and grid[row + 1, col - 1] == marker)
        or (col + 1 < width and grid[row - 1, col + 1] == marker
            and grid[row + 1, col + 1] == marker)
    )


def _transform(grid: np.ndarray, background: int, seed_color: int, marker: int) -> np.ndarray:
    """Grow traces and three-wide neighboring bands, retaining fixed pixels."""
    height, width = grid.shape
    output = grid.copy()
    spans: list[tuple[int, int, int, bool]] = []
    known = set()

    def add_span(row: int, left: int, right: int, seed_trace: bool = False) -> bool:
        item = (row, left, right, seed_trace)
        if item in known:
            return False
        known.add(item)
        spans.append(item)
        return True

    for row in np.flatnonzero(grid[:, 0] == seed_color):
        add_span(int(row), 0, width - 1, True)

    changed = True
    while changed:
        changed = False
        for row, left, initial_right, seed_trace in tuple(spans):
            col = max(0, left)
            right = min(width - 1, initial_right)
            ink_to_left = seed_trace or (col > 0 and output[row, col - 1] == seed_color)
            blocked = False
            while col <= right:
                if grid[row, col] == marker:
                    if not ink_to_left:
                        break
                    if row > 0:
                        changed |= add_span(row - 1, col - 1, col + 1)
                    if row + 1 < height:
                        changed |= add_span(row + 1, col - 1, col + 1)
                    if _blocked_contact(grid, row, col, marker):
                        if not seed_trace:
                            break
                        blocked = True
                    else:
                        blocked = False
                        right = max(right, min(width - 1, col + 1))
                else:
                    if not blocked and output[row, col] == background:
                        output[row, col] = seed_color
                        changed = True
                    ink_to_left = True
                col += 1
    return output


def _resolve_counts(rng: np.random.Generator, num_train: int | None,
                    num_test: int | None) -> tuple[int, int]:
    if num_train is not None and (type(num_train) is not int or not 2 <= num_train <= 10):
        raise ValueError('num_train must be an integer from 2 to 10')
    if num_test is not None and (type(num_test) is not int or not 1 <= num_test <= 4):
        raise ValueError('num_test must be an integer from 1 to 4')
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return num_train, num_test


def _palette(rng: np.random.Generator, train_count: int) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    seed_color = int(rng.integers(1, 10))
    remaining = [color for color in range(1, 10) if color != seed_color]
    rng.shuffle(remaining)
    backgrounds = (0, remaining.pop()) if rng.random() < 0.68 else (0,)
    marker_count = 2 if rng.random() < 0.72 else 1
    markers = tuple(remaining[:marker_count])
    return seed_color, backgrounds, markers


def _shape(rng: np.random.Generator, regime: str, lesson: str) -> tuple[int, int]:
    if lesson == 'clear':
        low, high = 6, 13
    elif lesson == 'rich':
        low, high = 14, 23
    else:
        category = int(rng.choice(3, p=(0.24, 0.46, 0.30) if regime == 'source_train'
                                  else (0.20, 0.38, 0.42)))
        low, high = ((6, 11), (10, 21), (19, 30))[category]
    h = int(rng.integers(low, high + 1))
    aspect = int(rng.integers(0, 3))
    if aspect == 0:
        w = h
    elif aspect == 1:
        w = int(rng.integers(max(6, h - 9), h + 1))
    else:
        w = int(rng.integers(h, min(30, h + 9) + 1))
    if lesson == 'rich':
        w = max(14, w)
    return h, w


def _seed_rows(rng: np.random.Generator, height: int, count: int,
               lesson: str) -> list[int]:
    if lesson == 'rich':
        # The two teaching motifs need room on both sides, but their spacing varies.
        first = int(rng.integers(2, max(3, height // 3 + 1)))
        second = int(rng.integers(max(first + 4, 2 * height // 3), height - 2))
        return sorted((first, second))
    return sorted(int(x) for x in rng.choice(height, size=count, replace=False))


def _generate_pair(rng: np.random.Generator, regime: str, lesson: str,
                   background: int, seed_color: int, marker: int) -> dict:
    height, width = _shape(rng, regime, lesson)
    if lesson == 'clear':
        seed_count = 1
    elif lesson == 'rich':
        seed_count = 2
    else:
        seed_count = int(rng.integers(1, min(3, height) + 1))
    rows = _seed_rows(rng, height, seed_count, lesson)
    grid = np.full((height, width), background, dtype=np.int8)
    for row in rows:
        grid[row, 0] = seed_color

    protected = set()
    anchors = []

    def put_marker(row: int, col: int, protect: int = 1) -> bool:
        if not (0 <= row < height and 1 <= col < width):
            return False
        if grid[row, col] == seed_color:
            return False
        grid[row, col] = marker
        for rr in range(max(0, row - protect), min(height, row + protect + 1)):
            for cc in range(max(0, col - protect), min(width, col + protect + 1)):
                protected.add((rr, cc))
        return True

    if lesson == 'clear':
        col = int(rng.integers(1, width))
        put_marker(rows[0], col, 2)
        anchors.append((rows[0], col))
    elif lesson == 'rich':
        chain_row = rows[0]
        block_row = rows[1]
        chain_col = int(rng.integers(2, max(3, width // 2)))
        block_col = int(rng.integers(max(chain_col + 4, width // 2), width - 1))
        put_marker(chain_row, chain_col, 1)
        anchors.append((chain_row, chain_col))
        direction = 1 if chain_row + 2 < height and (rng.random() < 0.5 or chain_row < 2) else -1
        if chain_row + 2 * direction < 0 or chain_row + 2 * direction >= height:
            direction *= -1
        put_marker(chain_row + direction, chain_col + 1, 1)
        if rng.random() < 0.6 and chain_col + 2 < width:
            put_marker(chain_row + 2 * direction, chain_col + 2, 1)
        put_marker(block_row, block_col, 1)
        anchors.append((block_row, block_col))
        if 0 < block_row < height - 1:
            put_marker(block_row - 1, block_col + 1, 1)
            put_marker(block_row + 1, block_col + 1, 1)
        # A separate partial block displays one surviving neighboring band.
        if chain_col > 1 and 0 <= chain_row - direction < height:
            put_marker(chain_row - direction, chain_col - 1, 0)
    else:
        for row in rows:
            if len(rows) > 1 and rng.random() < 0.36:
                continue
            anchor_count = int(rng.integers(1, min(4, width)))
            choices = np.arange(1, width)
            rng.shuffle(choices)
            for col in choices[:anchor_count]:
                if put_marker(row, int(col), 1):
                    anchors.append((row, int(col)))
        if not anchors:
            row = int(rng.choice(rows))
            col = int(rng.integers(1, width))
            put_marker(row, col, 1)
            anchors.append((row, col))
        if len(rows) > 1 and np.min(np.diff(rows)) <= 2 and rng.random() < 0.55:
            # Nearby traces can feed the same neighboring band.
            col = int(rng.integers(1, width))
            for row in rows:
                put_marker(row, col, 1)
        # Chains may rise, descend, change direction, or branch.
        if rng.random() < 0.64:
            row, col = anchors[int(rng.integers(len(anchors)))]
            direction = int(rng.choice((-1, 1)))
            for _ in range(int(rng.integers(1, 5))):
                directions = [step for step in (-1, 1) if 0 <= row + step < height]
                if not directions or col + 1 >= width:
                    break
                if direction not in directions or rng.random() < 0.25:
                    direction = int(rng.choice(directions))
                step = direction
                row += step
                col += int(rng.integers(0, 2))
                if col >= width:
                    break
                put_marker(row, col, 1)
                if rng.random() < 0.27 and 0 <= row - step < height and col + 1 < width:
                    put_marker(row - step, col + 1, 1)
        if rng.random() < 0.28:
            row, col = anchors[int(rng.integers(len(anchors)))]
            if 2 <= row < height - 2 and col + 2 < width:
                for side in (-1, 1):
                    put_marker(row + side, col + 1, 1)
                    put_marker(row + 2 * side, col + 2, 1)
        if rng.random() < 0.33:
            row, col = anchors[int(rng.integers(len(anchors)))]
            side = int(rng.choice((-1, 1)))
            if 0 <= row + side < height:
                put_marker(row + side, max(1, col - 1), 0)

    density_band = int(rng.choice(3, p=(0.40, 0.35, 0.25)))
    lo, hi = ((0.008, 0.035), (0.035, 0.09), (0.09, 0.17))[density_band]
    if lesson == 'clear':
        lo, hi = 0.005, 0.025
    density = float(rng.uniform(lo, hi))
    target = max(len(anchors), int(round(density * height * width)))
    candidates = [(row, col) for row in range(height) for col in range(width)
                  if row not in rows and (row, col) not in protected
                  and grid[row, col] == background]
    rng.shuffle(candidates)
    needed = max(0, target - int(np.count_nonzero(grid == marker)))
    for row, col in candidates[:needed]:
        grid[row, col] = marker
    # Background remains the most frequent role, so a learner can identify it
    # in an unseen pair even when its literal color differs between pairs.
    assert np.count_nonzero(grid == background) > np.count_nonzero(grid == marker)
    output = _transform(grid, background, seed_color, marker)
    return {'input': grid, 'output': output}


def _inferred_roles(train: list[dict], grid: np.ndarray) -> tuple[int, int, int]:
    growing = set()
    for pair in train:
        before = np.bincount(pair['input'].ravel(), minlength=10)
        after = np.bincount(pair['output'].ravel(), minlength=10)
        increased = set(np.flatnonzero(after > before).tolist())
        assert len(increased) == 1
        growing.update(increased)
    assert len(growing) == 1
    seed_color = growing.pop()
    colors = set(np.unique(grid).tolist())
    assert seed_color in colors and len(colors) == 3
    other = colors - {seed_color}
    background = max(other, key=lambda color: int(np.count_nonzero(grid == color)))
    marker = (other - {background}).pop()
    assert np.count_nonzero(grid == background) > np.count_nonzero(grid == marker)
    return background, seed_color, marker


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    seed_color, backgrounds, markers = _palette(rng, train_count)
    source_signatures = {(a.shape, a.tobytes()) for a in _source_grids()}
    seen = set(source_signatures)
    puzzle = {'train': [], 'test': []}
    for regime, count in (('source_train', train_count), ('target_test', test_count)):
        key = 'train' if regime == 'source_train' else 'test'
        for index in range(count):
            lesson = ('clear', 'rich')[index] if regime == 'source_train' and index < 2 else 'free'
            background = backgrounds[index % len(backgrounds)] if regime == 'source_train' else int(rng.choice(backgrounds))
            marker = markers[index % len(markers)] if regime == 'source_train' else int(rng.choice(markers))
            for _ in range(12):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                pair = _generate_pair(pair_rng, regime, lesson, background, seed_color, marker)
                grid = pair['input']
                signature = (grid.shape, grid.tobytes())
                if signature not in seen and not np.array_equal(grid, pair['output']):
                    seen.add(signature)
                    puzzle[key].append(pair)
                    break
            else:
                raise RuntimeError('could not generate a distinct, changing pair')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _has_clear_band(pair: dict, background: int, seed_color: int, marker: int) -> bool:
    grid, output = pair['input'], pair['output']
    height, width = grid.shape
    for row in np.flatnonzero(grid[:, 0] == seed_color):
        for col in np.flatnonzero(grid[row, 1:] == marker) + 1:
            left, right = max(0, col - 1), min(width, col + 2)
            for neighbor in (row - 1, row + 1):
                if 0 <= neighbor < height and np.all(grid[neighbor, left:right] == background):
                    if np.all(output[neighbor, left:right] == seed_color):
                        return True
    return False


def _has_blocked_contact(pair: dict, background: int, seed_color: int,
                         marker: int) -> bool:
    grid, output = pair['input'], pair['output']
    for row in np.flatnonzero(grid[:, 0] == seed_color):
        for col in np.flatnonzero(grid[row, 1:] == marker) + 1:
            if col + 1 < grid.shape[1] and _blocked_contact(grid, int(row), int(col), marker):
                if output[row, col + 1] == background:
                    return True
    return False


def _has_far_chain(pair: dict, background: int, seed_color: int) -> bool:
    grid, output = pair['input'], pair['output']
    seeds = np.flatnonzero(grid[:, 0] == seed_color)
    painted_rows = np.flatnonzero(np.any((grid == background) & (output == seed_color), axis=1))
    return any(np.min(np.abs(seeds - row)) >= 2 for row in painted_rows)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    train, test = puzzle['train'], puzzle['test']
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    source_signatures = {(a.shape, a.tobytes()) for a in _source_grids()}
    train_backgrounds, train_markers, train_seed_colors = set(), set(), set()
    for regime, pairs in (('source_train', train), ('target_test', test)):
        for pair in pairs:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            grid, output = pair['input'], pair['output']
            for array in (grid, output):
                assert isinstance(array, np.ndarray) and array.ndim == 2
                assert 1 <= array.shape[0] <= 30 and 1 <= array.shape[1] <= 30
                assert np.issubdtype(array.dtype, np.integer)
                assert np.all((0 <= array) & (array <= 9))
            assert grid.shape == output.shape
            signature = (grid.shape, grid.tobytes())
            assert signature not in seen and signature not in source_signatures
            seen.add(signature)
            background, seed_color, marker = _inferred_roles(train, grid)
            assert len(set(np.unique(grid).tolist())) == 3
            assert np.any(grid[:, 0] == seed_color)
            assert np.all(grid[:, 1:] != seed_color)
            assert np.array_equal(output[grid == marker], grid[grid == marker])
            assert np.array_equal(output[grid == seed_color], grid[grid == seed_color])
            assert np.all((grid != background) | ((output == background) | (output == seed_color)))
            assert np.array_equal(output, _transform(grid, background, seed_color, marker))
            assert np.count_nonzero((grid == background) & (output == seed_color)) > 0
            if regime == 'source_train':
                train_backgrounds.add(background)
                train_markers.add(marker)
                train_seed_colors.add(seed_color)
            else:
                assert background in train_backgrounds and marker in train_markers
    assert len(train_seed_colors) == 1
    assert any(_has_clear_band(pair, *_inferred_roles(train, pair['input'])) for pair in train)
    assert any(_has_blocked_contact(pair, *_inferred_roles(train, pair['input'])) for pair in train)
    assert any(_has_far_chain(pair, *_inferred_roles(train, pair['input'])[:2]) for pair in train)
    # Test predictions use only demonstrations and the test input.
    for pair in test:
        background, seed_color, marker = _inferred_roles(train, pair['input'].copy())
        prediction = _transform(pair['input'].copy(), background, seed_color, marker)
        assert np.array_equal(prediction, pair['output'])


def _equal_puzzles(first: dict, second: dict) -> bool:
    return all(len(first[key]) == len(second[key]) and all(
        np.array_equal(a['input'], b['input']) and np.array_equal(a['output'], b['output'])
        for a, b in zip(first[key], second[key])) for key in ('train', 'test'))


def self_test() -> None:
    for grid, digest in zip(_source_grids(), _SOURCE_OUTPUT_DIGESTS):
        marker = (set(np.unique(grid).tolist()) - {0, 1}).pop()
        actual = _transform(grid, 0, 1, marker)
        assert hashlib.sha256(actual.tobytes()).hexdigest() == digest

    train_counts, test_counts = set(), set()
    shapes, layouts, colors = set(), set(), set()
    seed_counts = set()
    square = small = near_edge_seed = near_edge_marker = 0
    no_anchor_seed_rows = close_seed_rows = 0
    sparse_tests = smaller_tests = single_test_seeds = 0
    blocked_contacts = partial_bands = far_chains = long_chains = 0
    rises = descents = branches = joins = untouched = clipped_bands = 0
    clean_single_bands = untouched_clusters = 0
    varied_backgrounds = varied_markers = single_seed_first = 0
    sparse_single_tests = square_rising_tests = close_chain_tests = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _equal_puzzles(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        train_max_area = max(p['input'].size for p in puzzle['train'])
        train_max_marks = max(int(np.count_nonzero(p['input'] == _inferred_roles(puzzle['train'], p['input'])[2]))
                              for p in puzzle['train'])
        train_roles = [_inferred_roles(puzzle['train'], p['input']) for p in puzzle['train']]
        varied_backgrounds += len({role[0] for role in train_roles}) > 1
        varied_markers += len({role[2] for role in train_roles}) > 1
        single_seed_first += np.count_nonzero(puzzle['train'][0]['input'][:, 0] == train_roles[0][1]) == 1
        for key in ('train', 'test'):
            for pair in puzzle[key]:
                grid, output = pair['input'], pair['output']
                bg, ink, marker = _inferred_roles(puzzle['train'], grid)
                shapes.add(grid.shape)
                layouts.add((grid.shape, grid.tobytes()))
                colors.add((bg, ink, marker))
                rows = np.flatnonzero(grid[:, 0] == ink)
                seed_counts.add(len(rows))
                square += grid.shape[0] == grid.shape[1]
                small += max(grid.shape) <= 8
                near_edge_seed += any(r <= 1 or r >= grid.shape[0] - 2 for r in rows)
                near_edge_marker += bool(np.any(grid[:, 1:3] == marker) or np.any(grid[:, -2:] == marker))
                no_anchor_seed_rows += any(not np.any(grid[r, 1:] == marker) for r in rows)
                close_seed_rows += len(rows) > 1 and bool(np.any(np.diff(rows) <= 2))
                new = (grid == bg) & (output == ink)
                new_rows = np.flatnonzero(np.any(new, axis=1))
                max_distance = max(min(abs(int(r) - int(s)) for s in rows) for r in new_rows)
                clean_single_bands += (len(rows) == 1 and
                                       np.count_nonzero(grid[rows[0], 1:] == marker) == 1 and
                                       max_distance == 1)
                far_chains += max_distance >= 2
                long_chains += max_distance >= 3
                rise = min(new_rows) < min(rows) - 1
                descent = max(new_rows) > max(rows) + 1
                rises += rise
                descents += descent
                branches += len(rows) == 1 and rise and descent
                close = len(rows) > 1 and bool(np.any(np.diff(rows) <= 2))
                joins += close and any(np.count_nonzero(grid[rows, c] == marker) >= 2
                                       for c in range(1, grid.shape[1]))
                clipped_bands += any((r in (0, grid.shape[0] - 1) and
                                      np.any(grid[r, 1:] == marker)) for r in rows)
                untouched += any(not np.any(new[max(0, r - 1):min(grid.shape[0], r + 2),
                                                  max(0, c - 1):min(grid.shape[1], c + 2)])
                                 for r, c in np.argwhere(grid == marker))
                untouched_clusters += any(
                    grid[r + dr, c + dc] == marker
                    and not np.any(new[max(0, r - 1):min(grid.shape[0], r + 2),
                                       max(0, c - 1):min(grid.shape[1], c + 2)])
                    and not np.any(new[max(0, r + dr - 1):min(grid.shape[0], r + dr + 2),
                                       max(0, c + dc - 1):min(grid.shape[1], c + dc + 2)])
                    for r, c in np.argwhere(grid == marker)
                    for dr, dc in ((0, 1), (1, -1), (1, 0), (1, 1))
                    if 0 <= r + dr < grid.shape[0] and 0 <= c + dc < grid.shape[1])
                for r in rows:
                    for c in np.flatnonzero(grid[r, 1:] == marker) + 1:
                        blocked_contacts += (_blocked_contact(grid, int(r), int(c), marker)
                                             and c + 1 < grid.shape[1] and output[r, c + 1] == bg)
                        for near in (r - 1, r + 1):
                            if 0 <= near < grid.shape[0] and c > 0:
                                partial_bands += (grid[near, c - 1] == marker and
                                                  np.any(output[near, c:min(grid.shape[1], c + 2)] == bg))
                assert np.all(output[grid == marker] == marker)
                assert np.all(output[grid == ink] == ink)
                if key == 'test':
                    single_test_seeds += len(rows) == 1
                    smaller_tests += grid.size <= train_max_area
                    sparse_tests += np.count_nonzero(grid == marker) <= train_max_marks
                    sparse_single_tests += len(rows) == 1 and np.count_nonzero(grid == marker) / grid.size < 0.04
                    square_rising_tests += grid.shape[0] == grid.shape[1] and rise
                    close_chain_tests += close and max_distance >= 2
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 70 and len(layouts) >= 100 and len(colors) >= 20
    assert seed_counts == {1, 2, 3}
    assert min(square, small, near_edge_seed, near_edge_marker,
               no_anchor_seed_rows, close_seed_rows) > 0
    assert min(sparse_tests, smaller_tests, single_test_seeds) > 0
    assert min(blocked_contacts, partial_bands, far_chains, long_chains,
               rises, descents, branches, joins, untouched, clipped_bands,
               clean_single_bands, untouched_clusters,
               varied_backgrounds, varied_markers, single_seed_first,
               sparse_single_tests, square_rising_tests, close_chain_tests) > 0
    for tr, te in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                   (2, None), (10, None)):
        puzzle = generate_puzzle(98765, tr, te)
        assert _equal_puzzles(puzzle, generate_puzzle(98765, tr, te))
        if tr is not None:
            assert len(puzzle['train']) == tr
        if te is not None:
            assert len(puzzle['test']) == te
    for tr, te in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (4, 4), (2.0, 1)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError((tr, te))
