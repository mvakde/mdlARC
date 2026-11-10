"""Compositional generators for ordered, occluding rectangular outlines."""

import hashlib
import numpy as np


_SOURCE_HASHES = {
    "9bd1aae7ced804d16c2dfe2455597da66bb4ca9fbd76b967c46f89148cb4e03c",
    "16c10f2afcf1423d2e1a319cfde5b5ee58485f92315167dc095811111d16479e",
    "6ff5a01107c450fd20410fa41cf0644e40a6f6388c9bb50cef7c9a8a7d9d4016",
    "2a2ca4c2005b4abab2fa2765397aa49a38da96c69b95ed1287c5e44dba1d5249",
    "3a4d6b6c1e58382cfb57e0edb61302ccc6cd16d97275b3280c72fc2f68eb2a27",
}


def _choose_counts(rng, num_train, num_test):
    for value, name, low, high in ((num_train, 'num_train', 2, 10),
                                   (num_test, 'num_test', 1, 4)):
        if value is not None and (not isinstance(value, (int, np.integer))
                                  or isinstance(value, (bool, np.bool_))
                                  or not low <= value <= high):
            raise ValueError(f'{name} must be an integer from {low} to {high}')
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _perimeter(rect):
    t, l, b, r = rect
    return {(t, c) for c in range(l, r + 1)} | {(b, c) for c in range(l, r + 1)} | \
           {(row, l) for row in range(t + 1, b)} | {(row, r) for row in range(t + 1, b)}


def _render(shape, background, rectangles, colors):
    grid = np.full(shape, background, dtype=np.int8)
    for rect, color in zip(rectangles, colors):
        for row, col in _perimeter(rect):
            grid[row, col] = color
    return grid


def _background(grid):
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _analyze_input(grid):
    """Reconstruct from visible color extents and sort visible occlusion edges."""
    background = _background(grid)
    colors = [int(c) for c in np.unique(grid) if int(c) != background]
    assert 2 <= len(colors) <= 8
    rectangles = {}
    perimeters = {}
    for color in colors:
        spots = np.argwhere(grid == color)
        t, l = (int(x) for x in spots.min(axis=0))
        b, r = (int(x) for x in spots.max(axis=0))
        assert b - t >= 1 and r - l >= 1
        rect = (t, l, b, r)
        perimeter = _perimeter(rect)
        assert all((int(y), int(x)) in perimeter for y, x in spots)
        rectangles[color] = rect
        perimeters[color] = perimeter
    owners = {}
    for color, perimeter in perimeters.items():
        for cell in perimeter:
            owners.setdefault(cell, []).append(color)
    edges = {color: set() for color in colors}
    direct_pairs = set()
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            visible = int(grid[row, col])
            occupants = owners.get((row, col), ())
            if not occupants:
                assert visible == background
            else:
                assert visible in occupants
                for hidden in occupants:
                    if hidden != visible:
                        edges[hidden].add(visible)
                for i, a in enumerate(occupants):
                    for b in occupants[i + 1:]:
                        direct_pairs.add(tuple(sorted((a, b))))
    indegree = {color: 0 for color in colors}
    for targets in edges.values():
        for color in targets:
            indegree[color] += 1
    remaining = set(colors)
    order = []
    while remaining:
        available = [c for c in remaining if indegree[c] == 0]
        assert len(available) == 1
        back = available[0]
        remaining.remove(back)
        order.append(back)
        for front in edges[back]:
            indegree[front] -= 1
    assert np.array_equal(_render(grid.shape, background,
                                  [rectangles[c] for c in order], order), grid)
    return {'order': tuple(order), 'rectangles': rectangles,
            'direct_pairs': len(direct_pairs), 'edges': edges,
            'background': background}


def _rand_rect(rng, height, width, small=False, edge=False):
    if small:
        if rng.random() < .65:
            rh, rw = (2, 3) if bool(rng.integers(2)) else (3, 2)
        else:
            rh = int(rng.integers(2, min(5, height) + 1))
            rw = int(rng.integers(2, min(5, width) + 1))
    else:
        rh = int(rng.integers(3, min(height, max(6, int(height * .75))) + 1))
        rw = int(rng.integers(3, min(width, max(6, int(width * .75))) + 1))
    top = int(rng.integers(0, height - rh + 1))
    left = int(rng.integers(0, width - rw + 1))
    if edge:
        side = int(rng.integers(4))
        if side == 0:
            top = 0
        elif side == 1:
            left = 0
        elif side == 2:
            top = height - rh
        else:
            left = width - rw
    return (top, left, top + rh - 1, left + rw - 1)


def _chain_layout(rng, height, width, count):
    """Build an intersection path with distant rectangles disjoint."""
    horizontal = bool(rng.integers(2))
    length = width if horizontal else height
    breadth = height if horizontal else width
    # Consecutive outlines share one or two columns/rows; non-neighbors do not.
    step = max(2, (length - 5) // max(1, count - 1))
    span = min(step + int(rng.integers(1, max(2, step + 1))), 2 * step - 1)
    span = max(3, span)
    if (count - 1) * step + span > length:
        step = max(2, (length - 4) // count)
        span = min(2 * step - 1, length - (count - 1) * step)
    start = int(rng.integers(0, length - ((count - 1) * step + span) + 1))
    upper = int(rng.integers(0, max(1, breadth // 3)))
    lower = int(rng.integers(max(upper + 3, 2 * breadth // 3), breadth))
    rectangles = []
    for i in range(count):
        # Common long-axis boundaries permit aligned sides, as in the source.
        a = start + i * step
        z = a + span - 1
        top = max(0, min(upper + int(rng.integers(-1, 2)), lower - 2))
        bottom = min(breadth - 1, max(top + 2, lower + int(rng.integers(-1, 2))))
        rect = (top, a, bottom, z) if horizontal else (a, top, z, bottom)
        rectangles.append(rect)
    if bool(rng.integers(2)):
        rectangles.reverse()
    return rectangles


def _free_layout(rng, height, width, count, small, edge, density):
    rectangles = [_rand_rect(rng, height, width, edge=edge)]
    perimeters = [_perimeter(rectangles[0])]
    for index in range(1, count):
        chosen = None
        for _ in range(180):
            candidate = _rand_rect(rng, height, width,
                                   small=small and index == count - 1)
            if candidate in rectangles:
                continue
            perimeter = _perimeter(candidate)
            contacts = sum(bool(perimeter & old) for old in perimeters)
            if not perimeter & perimeters[-1]:
                continue
            if density == 'sparse' and contacts > 2:
                continue
            if density == 'dense' and index >= 2 and contacts < 2:
                continue
            chosen = (candidate, perimeter)
            break
        if chosen is None:
            return None
        rectangles.append(chosen[0])
        perimeters.append(chosen[1])
    return rectangles


def _make_pair(rng, count, mode, shared_background=None, require_nonmonotonic=False,
               forbidden_color=None, required_color=None):
    for attempt in range(160):
        if count >= 6:
            height = int(rng.integers(18, 31))
            width = int(rng.integers(19, 31))
        elif count <= 3:
            height = int(rng.integers(10, 30))
            width = int(rng.integers(11, 31))
        else:
            height = int(rng.integers(13, 31))
            width = int(rng.integers(14, 31))
        if mode == 'chain':
            rectangles = _chain_layout(rng, height, width, count)
        else:
            rectangles = _free_layout(rng, height, width, count,
                                      small=mode == 'small',
                                      edge=mode == 'edge',
                                      density='dense' if mode == 'dense' else 'mixed')
        if rectangles is None:
            continue
        background = (int(shared_background) if shared_background is not None
                      else int(rng.integers(10)))
        if required_color == background:
            continue
        available = [c for c in range(10) if c != background and c != forbidden_color]
        if required_color is None:
            colors = [int(c) for c in rng.choice(available, count, replace=False)]
        else:
            available.remove(required_color)
            colors = [int(required_color)] + [int(c) for c in rng.choice(available, count - 1, replace=False)]
            rng.shuffle(colors)
        if require_nonmonotonic and (colors == sorted(colors) or
                                     colors == sorted(colors, reverse=True)):
            continue
        grid = _render((height, width), background, rectangles, colors)
        try:
            analysis = _analyze_input(grid)
        except AssertionError:
            continue
        if (analysis['background'] != background or
                analysis['order'] != tuple(colors)):
            continue
        if any(analysis['rectangles'][color] != rect
               for rect, color in zip(rectangles, colors)):
            continue
        if mode == 'chain' and analysis['direct_pairs'] != count - 1:
            continue
        if mode == 'dense' and analysis['direct_pairs'] < max(1, count - 1):
            continue
        if mode == 'small' and min(min(b - t + 1, r - l + 1)
                                   for t, l, b, r in analysis['rectangles'].values()) > 4:
            continue
        if mode == 'edge' and not any(t == 0 or l == 0 or b == height - 1 or r == width - 1
                                      for t, l, b, r in analysis['rectangles'].values()):
            continue
        output = np.asarray(colors, dtype=np.int8).reshape(-1, 1)
        return {'input': grid, 'output': output}
    raise RuntimeError(f'could not make {count}-outline {mode} layout')


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _source_hash(grid):
    return hashlib.sha256(''.join(str(int(c)) for c in grid.flat).encode()).hexdigest()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    num_train, num_test = _choose_counts(rng, num_train, num_test)
    # Only the transformation is a puzzle-level convention. Some puzzles repeat a
    # background for ease of reading; neither color nor size encodes depth.
    shared_background = int(rng.integers(10)) if rng.random() < .28 else None
    small_train_regime = rng.random() < .32
    if small_train_regime:
        train_counts = [3, 4] + [int(rng.integers(2, 6)) for _ in range(num_train - 2)]
    else:
        train_counts = [int(rng.integers(3, 5)), int(rng.integers(5, 7))]
        train_counts += [int(rng.integers(2, 8)) for _ in range(num_train - 2)]
    rng.shuffle(train_counts)
    train_modes = ['chain', 'dense'] + [str(rng.choice(['chain', 'dense', 'mixed', 'small', 'edge']))
                                         for _ in range(num_train - 2)]
    rng.shuffle(train_modes)
    if not any(mode == 'chain' and count >= 3
               for mode, count in zip(train_modes, train_counts)):
        chain_index = train_modes.index('chain')
        other_index = next(i for i, count in enumerate(train_counts) if count >= 3)
        train_modes[chain_index], train_modes[other_index] = (
            train_modes[other_index], train_modes[chain_index])
    test_counts = [int(rng.integers(2, 8)) for _ in range(num_test)]
    if small_train_regime:
        test_counts[0] = int(rng.integers(max(train_counts) + 1, 8))
    test_modes = [str(rng.choice(['chain', 'dense', 'mixed', 'small', 'edge']))
                  for _ in range(num_test)]
    # Source-like 4/5 examples remain common, while test counts cross the
    # training range in both directions across seeds.
    puzzle = {'train': [], 'test': []}
    seen = set()
    holdout = (int(rng.choice([c for c in range(10) if c != shared_background]))
               if rng.random() < .55 else None)
    needs_nonmonotonic = True
    for split, counts, modes in (('train', train_counts, train_modes),
                                 ('test', test_counts, test_modes)):
        for index, (count, mode) in enumerate(zip(counts, modes)):
            for _ in range(8):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                pair = _make_pair(pair_rng, count, mode, shared_background,
                                  require_nonmonotonic=(split == 'train' and
                                                        needs_nonmonotonic and count >= 3),
                                  forbidden_color=holdout if split == 'train' else None,
                                  required_color=holdout if split == 'test' and index == 0 else None)
                key = _grid_key(pair['input'])
                if key not in seen and _source_hash(pair['input']) not in _SOURCE_HASHES:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == 'train' and count >= 3:
                        needs_nonmonotonic = False
                    break
            else:
                raise RuntimeError('could not make a unique pair')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen = set()
    all_pairs = puzzle['train'] + puzzle['test']
    train_counts = set()
    nonnumeric = False
    train_chain = False
    for split, pairs in puzzle.items():
        for pair in pairs:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = pair['input'], pair['output']
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            assert out.shape[1] == 1
            assert _source_hash(inp) not in _SOURCE_HASHES
            key = _grid_key(inp)
            assert key not in seen
            seen.add(key)
            analysis = _analyze_input(inp)
            order = analysis['order']
            assert out.shape[0] == len(order)
            assert tuple(int(c) for c in out[:, 0]) == order
            assert set(np.unique(inp)) == set(order) | {analysis['background']}
            if split == 'train':
                train_counts.add(len(order))
                nonnumeric |= list(order) != sorted(order) and list(order) != sorted(order, reverse=True)
                train_chain |= len(order) >= 3 and analysis['direct_pairs'] == len(order) - 1
    assert len(train_counts) >= 2
    assert train_chain
    # With two small examples, a nonnumeric sequence can be impossible to show.
    assert nonnumeric or max(train_counts) <= 3


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ('input', 'output'))
        for s in ('train', 'test'))


def self_test() -> None:
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                        (2, None), (10, None), (7, 3)):
        p = generate_puzzle(9137 + (train or 0) * 10 + (test or 0), train, test)
        validate_generated_puzzle(p)
        assert train is None or len(p['train']) == train
        assert test is None or len(p['test']) == test
    for kw in ({'num_train': 1}, {'num_train': 11}, {'num_test': 0},
               {'num_test': 5}, {'num_train': 2, 'num_test': 2},
               {'num_train': 3, 'num_test': 4}):
        try:
            generate_puzzle(17, **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f'illegal counts accepted: {kw}')
    assert _puzzles_equal(generate_puzzle(812, 7, 3), generate_puzzle(812, 7, 3))
    assert not _puzzles_equal(generate_puzzle(812, 7, 3), generate_puzzle(813, 7, 3))
    train_amounts, test_amounts = set(), set()
    shapes, counts, backgrounds, layouts = set(), set(), set(), set()
    sparse = small = edge = six = 0
    exact_small = hidden_corner = aligned = nested = dense = 0
    compact = spacious = combined = unseen_test = 0
    changed_background = shared_background = more_test = fewer_test = equal_test = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        train_amounts.add(len(p['train']))
        test_amounts.add(len(p['test']))
        train_lengths = [len(pair['output']) for pair in p['train']]
        test_lengths = [len(pair['output']) for pair in p['test']]
        train_colors = {int(c) for pair in p['train'] for c in pair['output'][:, 0]}
        more_test += any(n > max(train_lengths) for n in test_lengths)
        fewer_test += any(n < max(train_lengths) for n in test_lengths)
        equal_test += any(n in train_lengths for n in test_lengths)
        puzzle_backgrounds = set()
        for split, pair in [(s, q) for s in ('train', 'test') for q in p[s]]:
            g = pair['input']
            a = _analyze_input(g)
            shapes.add(g.shape)
            counts.add(len(a['order']))
            backgrounds.add(a['background'])
            puzzle_backgrounds.add(a['background'])
            layouts.add((g.shape, tuple(sorted(a['rectangles'].values()))))
            sparse += a['direct_pairs'] == len(a['order']) - 1
            dense += a['direct_pairs'] >= len(a['order']) and len(a['order']) >= 3
            rects = list(a['rectangles'].values())
            is_small = any(min(b - t + 1, r - l + 1) <= 4 for t, l, b, r in rects)
            is_exact_small = any(sorted((b - t + 1, r - l + 1)) == [2, 3]
                                 for t, l, b, r in rects)
            is_edge = any(t == 0 or l == 0 or b == g.shape[0] - 1 or r == g.shape[1] - 1
                          for t, l, b, r in rects)
            is_hidden_corner = any(g[y, x] != c for c, (t, l, b, r) in a['rectangles'].items()
                                   for y, x in ((t, l), (t, r), (b, l), (b, r)))
            small += is_small
            exact_small += is_exact_small
            edge += is_edge
            hidden_corner += is_hidden_corner
            combined += is_exact_small and is_edge and is_hidden_corner
            aligned += any(x[0] in (y[0], y[2]) or x[2] in (y[0], y[2]) or
                           x[1] in (y[1], y[3]) or x[3] in (y[1], y[3])
                           for i, x in enumerate(rects) for y in rects[i + 1:])
            nested += any(x[0] <= y[0] and x[1] <= y[1] and x[2] >= y[2] and x[3] >= y[3]
                          for i, x in enumerate(rects) for j, y in enumerate(rects) if i != j)
            compact += min(g.shape) <= 15
            spacious += max(g.shape) >= 28
            six += len(a['order']) >= 6
            if split == 'test':
                unseen_test += bool(set(a['order']) - train_colors)
        changed_background += len(puzzle_backgrounds) > 1
        shared_background += len(puzzle_backgrounds) == 1
    assert len(train_amounts) >= 4 and len(test_amounts) >= 3
    assert len(shapes) >= 25 and len(layouts) >= 100 and len(backgrounds) >= 8
    assert {2, 3, 4, 5, 6, 7} <= counts
    assert min(sparse, small, edge, six) >= 10
    assert min(exact_small, hidden_corner, aligned, nested, dense) >= 10
    assert min(compact, spacious, unseen_test) >= 10
    assert combined >= 5
    assert changed_background >= 10 and shared_background >= 3
    assert min(more_test, fewer_test, equal_test) >= 5
